"""Stage executor: runs ONE Stage to completion using NavDP for low-level
motion and the explicit-CoT executor prompt for searching/approaching/
arrived/wrong_object judgement.

ASYNC by design: Qwen (grounding + status check) and NavDP run at their own
independent paces on separate threads, not sequentially in one loop.
Previously goal_point_fn (a VLM grounding call) ran INLINE on every single
NavDP tick -- NavDP's whole control rate was bottlenecked by a ~1-3s VLM
generate call every iteration, and the periodic status check blocked the
tick loop the same way. Now a background _qwen_worker thread loops
grounding -> check -> grounding -> check... back to back as fast as the
model allows, publishing its latest result into a lock-protected
_SharedPerception; the main tick loop (NavDP) runs at its own tick_hz,
reading whatever the worker most recently produced without ever waiting on
it. This mirrors LemNAV's object_leg.py design (NavDP drives every tick
unconditionally; Qwen is a slower, independent channel that only
retargets/declares arrival) but as genuinely concurrent threads instead of
one blocking loop.

Every tick is logged as a discrete (qwen_action, navdp_action) pair via
TraceLogger -- see trace_log.py -- same per-tick JSONL shape LemNAV's
qwen_navdp_trace.py already used, so validate_against_logs.py-style replay
works on Space2nav's own live runs too.

Perception is injected via two small interfaces so this stays usable
without a specific hardware stack:

  MotionInterface  -- drive_straight/turn_in_place (blind, for move/turn
                      stages) + set_velocity/get_rgbd (for the live loop).
  goal_point_fn    -- (rgb, depth, Stage) -> (x, y, z) meters in robot
                      frame, the goal for the object/instruction stage's
                      target phrase. Defaults to QwenGoalPointFn -- Qwen
                      itself detects/grounds the phrase (qwen_grounding,
                      no Grounding-DINO) and the box is projected through
                      depth into a point for NavDP's point-goal encoder
                      (the vendored navdp-cross-modal.ckpt checkpoint has
                      no trained bare-pixel-goal encoder). Pass an explicit
                      goal_point_fn (or None) to override. When it returns
                      None (nothing detected / no depth), the executor
                      falls back to NavDP's no-goal/explore mode and relies
                      on the CoT executor prompt's "arrived" judgement.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Optional, Protocol, Tuple, Union

import numpy as np

from explicit_cot.prompts import STAGE_EXECUTOR_PROMPT
from explicit_cot.vlm import ExplicitCoTVLM
from nav_dp import NavDPStandalone, trajectory_to_vw

from .qwen_goal import QwenGoalPointFn
from .trace_log import TraceLogger
from .types import Stage, StageFeedback

GoalPointFn = Callable[[np.ndarray, "Optional[np.ndarray]", Stage], "Optional[Tuple[float, float, float]]"]
_DEFAULT = object()  # sentinel distinguishing "use QwenGoalPointFn" from an explicit None override


class MotionInterface(Protocol):
    def drive_straight(self, meters: float) -> dict: ...
    def turn_in_place(self, degrees: float) -> dict: ...
    def set_velocity(self, linear: float, angular: float) -> None: ...
    def get_rgbd(self) -> "Tuple[np.ndarray, Optional[np.ndarray]]": ...


class _SharedPerception:
    """Qwen worker thread writes here; NavDP tick loop reads -- the only
    thing coupling the two threads together, lock-protected."""

    def __init__(self, target: str):
        self.lock = threading.Lock()
        self.goal_point: Optional[Tuple[float, float, float]] = None
        self.status: str = "searching"
        self.target: str = target
        self.action_hint: str = "forward"
        self.reason: str = ""
        self.last_qwen_action: Optional[dict] = None  # for the trace log


class StageExecutor:
    def __init__(self, navdp: NavDPStandalone, vlm: ExplicitCoTVLM, motion: MotionInterface,
                 goal_point_fn: "Union[GoalPointFn, None, object]" = _DEFAULT,
                 tick_hz: float = 2.5, timeout_s: float = 45.0, max_new_tokens: int = 300,
                 stop_event: Optional[threading.Event] = None,
                 trace_logger: Optional[TraceLogger] = None):
        self.navdp = navdp
        self.vlm = vlm
        self.motion = motion
        self.goal_point_fn = QwenGoalPointFn(vlm) if goal_point_fn is _DEFAULT else goal_point_fn
        self.tick_dt = 1.0 / tick_hz
        self.timeout_s = timeout_s
        self.max_new_tokens = max_new_tokens
        # External interrupt (e.g. the GUI's "Kill run" / server's /control
        # stop) -- checked every tick on both threads. Owned by the caller
        # (one executor per stage; the caller decides whether to clear it
        # between stages or let it stay set to abort the whole plan).
        self.stop_event = stop_event or threading.Event()
        self.trace_logger = trace_logger
        self._frames: "deque[np.ndarray]" = deque(maxlen=self.navdp.memory_size)
        self._frames_lock = threading.Lock()  # now touched by both the tick loop and the worker thread

    # -- blind primitives --------------------------------------------- #
    def _run_move_or_turn(self, stage: Stage) -> StageFeedback:
        t0 = time.monotonic()
        if stage.kind == "move":
            info = self.motion.drive_straight(stage.meters or 0.0)
        else:
            info = self.motion.turn_in_place(stage.degrees or 0.0)
        return StageFeedback(stage=stage, outcome=info.get("outcome", "ok"),
                              reason=info.get("reason", ""),
                              elapsed_s=time.monotonic() - t0, extra=info)

    # -- live object/instruction loop ---------------------------------- #
    def _preprocess(self, rgb: np.ndarray, depth: Optional[np.ndarray]):
        import torch
        size = self.navdp.image_size
        rgb_f = _resize_hwc(rgb, size).astype(np.float32) / 255.0
        with self._frames_lock:
            self._frames.append(rgb_f)
            while len(self._frames) < self.navdp.memory_size:
                self._frames.append(rgb_f)
            images = torch.from_numpy(np.stack(list(self._frames), axis=0)).unsqueeze(0)
        if depth is None:
            depth_f = np.zeros((size, size, 1), dtype=np.float32)
        else:
            depth_2d = depth.astype(np.float32)
            if depth_2d.ndim == 3:
                depth_2d = depth_2d[..., 0]
            depth_f = _resize_hwc(depth_2d, size)[..., None].copy()  # PIL resize can hand back a read-only array
            depth_f[(depth_f < 0.1) | (depth_f > 5.0) | ~np.isfinite(depth_f)] = 0.0
        # Checkpoint convention: images carry memory_size frames, depths
        # carry only the CURRENT frame (see backbone.RGBDBackbone.forward --
        # image_token has memory_size*256 tokens, depth_token has 1*256,
        # concatenated to match former_pe's (memory_size+1)*256 size).
        depths = torch.from_numpy(depth_f).unsqueeze(0).unsqueeze(1)
        return images, depths

    def _recent_history_frames(self, n: int = 2) -> list:
        with self._frames_lock:
            frames = list(self._frames)[:-1][-n:]
        return [(f * 255.0).clip(0, 255).astype(np.uint8) for f in frames]

    def _step_navdp(self, rgb: np.ndarray, depth: Optional[np.ndarray],
                     goal_point: Optional[Tuple[float, float, float]]):
        images, depths = self._preprocess(rgb, depth)
        images = images.to(self.navdp.input_dtype)
        depths = depths.to(self.navdp.input_dtype)
        if goal_point is not None:
            best, _worst, _critic = self.navdp.predict_pointgoal(
                list(goal_point), images, depths, sample_num=16, topk=1)
            mode = "pointgoal"
        else:
            best, _worst, _critic = self.navdp.predict_nogoal(
                images, depths, sample_num=16, topk=1)
            mode = "nogoal"
        trajectory = best[0].detach().cpu().numpy()
        linear, angular = trajectory_to_vw(trajectory)
        return linear, angular, mode

    # -- Qwen worker thread: grounding <-> check, back to back, own pace - #
    def _qwen_worker(self, stage: Stage, shared: _SharedPerception,
                      worker_stop: threading.Event) -> None:
        """worker_stop is THIS STAGE's own lifecycle signal (set once this
        stage ends, win/lose/timeout -- a fresh Event every run() call).
        self.stop_event is the externally-owned abort signal (e.g. the
        server's /control stop) and is never set/cleared by this method --
        only read, so clearing it here could never silently swallow an
        external abort request meant to also halt the stages after this
        one (see run_stagewise_loop's stage-boundary check)."""
        while not worker_stop.is_set() and not self.stop_event.is_set():
            rgb, depth = self.motion.get_rgbd()
            if rgb is None:
                time.sleep(0.05)
                continue

            goal_point = self.goal_point_fn(rgb, depth, stage) if self.goal_point_fn else None
            det = getattr(self.goal_point_fn, "last_detection", None)
            qwen_ground_action = {
                "kind": "grounding", "target": stage.phrase or "",
                "goal_point": list(goal_point) if goal_point is not None else None,
                "detector_score": float(det.score) if det is not None else None,
            }
            with shared.lock:
                shared.goal_point = goal_point
                shared.last_qwen_action = qwen_ground_action
            if self.trace_logger:
                self.trace_logger.log_tick(stage.describe(), stage.kind, qwen_ground_action,
                                            {"kind": "grounding_only", "note": "paired NavDP tick logged separately"})
            if worker_stop.is_set() or self.stop_event.is_set():
                return

            with shared.lock:
                target = shared.target
            history_frames = self._recent_history_frames()
            verdict = self._check(rgb, stage.describe(), target, history_frames=history_frames)
            qwen_check_action = {"kind": "check", **verdict}
            with shared.lock:
                shared.status = verdict.get("status", shared.status)
                shared.target = verdict.get("target", shared.target)
                shared.action_hint = verdict.get("action", shared.action_hint)
                shared.reason = verdict.get("reason", "")
                shared.last_qwen_action = qwen_check_action
            if self.trace_logger:
                self.trace_logger.log_tick(stage.describe(), stage.kind, qwen_check_action,
                                            {"kind": "check_only", "note": "paired NavDP tick logged separately"})

    def run(self, stage: Stage, get_frame: Callable[[], np.ndarray]) -> StageFeedback:
        if stage.kind in ("move", "turn"):
            return self._run_move_or_turn(stage)
        if stage.kind not in ("object", "instruction"):
            return StageFeedback(stage=stage, outcome="skipped",
                                  reason=f"executor has no handler for kind={stage.kind!r}")

        shared = _SharedPerception(target=stage.phrase or "")
        worker_stop = threading.Event()  # this stage's own lifecycle, separate from self.stop_event
        worker = threading.Thread(target=self._qwen_worker, args=(stage, shared, worker_stop), daemon=True)
        worker.start()

        t0 = time.monotonic()
        ticks = 0
        try:
            while True:
                if self.stop_event.is_set():
                    self.motion.set_velocity(0.0, 0.0)
                    return StageFeedback(stage=stage, outcome="aborted", ticks=ticks,
                                          elapsed_s=time.monotonic() - t0,
                                          last_executor_status=shared.status)
                elapsed = time.monotonic() - t0
                if elapsed > self.timeout_s:
                    self.motion.set_velocity(0.0, 0.0)
                    return StageFeedback(stage=stage, outcome="timeout", ticks=ticks,
                                          elapsed_s=elapsed, last_executor_status=shared.status)

                rgb, depth = self.motion.get_rgbd()
                with shared.lock:
                    goal_point = shared.goal_point
                    status = shared.status
                    action_hint = shared.action_hint
                    qwen_action_for_trace = shared.last_qwen_action

                linear, angular, mode = self._step_navdp(rgb, depth, goal_point)

                if status == "arrived":
                    self.motion.set_velocity(0.0, 0.0)
                    return StageFeedback(stage=stage, outcome="reached", ticks=ticks,
                                          elapsed_s=time.monotonic() - t0, last_executor_status="arrived",
                                          reason=shared.reason)
                if status == "searching" and goal_point is None:
                    # No detection yet -- fall back to the worker's own weak
                    # steering hint instead of NavDP's no-goal explore drift.
                    angular = {"left": 0.3, "right": -0.3}.get(action_hint, angular)
                    linear = 0.0 if action_hint in ("left", "right", "stop") else linear

                self.motion.set_velocity(linear, angular)
                if self.trace_logger:
                    self.trace_logger.log_tick(
                        stage.describe(), stage.kind, qwen_action_for_trace,
                        {"kind": "navdp_step", "linear": linear, "angular": angular,
                         "goal_point_used": list(goal_point) if goal_point is not None else None,
                         "mode": mode, "status_used": status})
                ticks += 1
                time.sleep(self.tick_dt)
        finally:
            worker_stop.set()
            worker.join(timeout=2.0)

    def _check(self, rgb: np.ndarray, clause: str, target: str,
               history_frames: "Optional[list]" = None) -> dict:
        images = [rgb]
        history_note = ""
        if history_frames:
            start = 2
            end = 1 + len(history_frames)
            span = f"[{start}]" if start == end else f"[{start}-{end}]"
            history_note = (f" {span} the robot's own recent frame(s), OLDEST first, lower "
                             f"resolution than [1] -- use these ONLY to judge motion/progress "
                             f"since a moment ago (getting closer? passed it? still searching the "
                             f"same view?), not for fine detail.")
            images.extend(history_frames)
        prompt = STAGE_EXECUTOR_PROMPT.format(clause=clause, target=target, history_note=history_note)
        _reasoning, value = self.vlm.ask(
            prompt, images=images, max_new_tokens=self.max_new_tokens, tag="stage_executor")
        if not isinstance(value, dict) or "status" not in value:
            return {"target": target, "status": "searching", "action": "forward",
                    "reason": "unparsed executor reply"}
        value.setdefault("target", target)
        value["status"] = str(value.get("status", "searching")).strip().lower()
        value["action"] = str(value.get("action", "forward")).strip().lower()
        if value["action"] not in ("forward", "left", "right", "stop"):
            value["action"] = "forward"
        return value


def _resize_hwc(arr: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image
    mode = "F" if arr.dtype == np.float32 and arr.ndim == 2 else None
    img = Image.fromarray(arr) if mode is None else Image.fromarray(arr, mode=mode)
    img = img.resize((size, size))
    return np.asarray(img)
