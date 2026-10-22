"""Stagewise planner: Phase 1 (no upfront full plan -- one stage at a time)
+ Phase 2 (formalized StageFeedback) + Phase 3 (frame-grounded re-verification
at every stage boundary, replacing text-only continue/retry/abort) from the
SpaceVLN-without-spatial-memory integration plan.

Where LemNAV had two separate calls -- decompose() (one-shot, text-only,
commits the WHOLE plan before any execution) and workflow_step() (text-only,
sees only an outcome log, never the camera) -- this collapses both into one
call made at every stage boundary, WITH the current camera frame, so a wrong
assumption from three stages ago can be caught and corrected instead of
driving a stale plan to timeout/lost.
"""
from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np

from explicit_cot.prompts import STAGE_PLANNER_PROMPT
from explicit_cot.vlm import ExplicitCoTVLM

from .types import Stage, StageFeedback

MAX_STAGES_DEFAULT = 40


class StagewisePlanner:
    def __init__(self, vlm: ExplicitCoTVLM, max_new_tokens: int = 768):
        self.vlm = vlm
        self.max_new_tokens = max_new_tokens

    @staticmethod
    def _format_history(history: List[StageFeedback]) -> str:
        if not history:
            return "  (none -- this is the first stage)"
        return "\n".join(f"  {i + 1}. {fb.describe()}" for i, fb in enumerate(history))

    def propose_next_stage(self, instruction: str, history: List[StageFeedback],
                            frame: np.ndarray) -> Stage:
        """One planning call: instruction + every stage run so far + the
        CURRENT camera frame -> exactly one next Stage (possibly "done" or
        "abort"). Called again after every StageExecutor.run() -- this is
        the closed loop; there is no separate upfront "whole plan" object
        anywhere in this module."""
        prompt = STAGE_PLANNER_PROMPT.format(
            instruction=instruction,
            stage_history=self._format_history(history),
        )
        _reasoning, value = self.vlm.ask(
            prompt, images=[frame], max_new_tokens=self.max_new_tokens, tag="stage_planner")

        if isinstance(value, list):
            value = value[0] if value else None
        if not isinstance(value, dict) or "kind" not in value:
            # Unparsed/malformed reply -- fail closed into "abort" rather than
            # guessing a stage, so a bad parse can't silently loop forever.
            return Stage(kind="abort", reason=f"unparsed planner reply: {value!r}")
        try:
            return Stage.from_goal_value(value)
        except ValueError as e:
            return Stage(kind="abort", reason=str(e))


def run_stagewise_loop(planner: StagewisePlanner, executor, instruction: str,
                        get_frame, max_stages: int = MAX_STAGES_DEFAULT,
                        stop_event: Optional[threading.Event] = None) -> List[StageFeedback]:
    """Drive the closed loop end to end: propose a stage with a fresh frame,
    execute it, feed the resulting StageFeedback back into the next
    proposal. `executor` is a StageExecutor (or anything with a matching
    .run(stage, get_frame) -> StageFeedback). `get_frame` returns the
    current RGB frame as a (H, W, 3) uint8 array on demand.

    stop_event (e.g. the GUI's "Kill run" / the server's /control stop) is
    checked at every stage BOUNDARY -- if set, no further stage is proposed
    or executed. Mid-stage interruption is the executor's own stop_event
    (StageExecutor.stop_event), which this can be the SAME object as."""
    history: List[StageFeedback] = []
    for _ in range(max_stages):
        if stop_event is not None and stop_event.is_set():
            print("[stagewise] stop requested -- halting before the next stage")
            break
        stage = planner.propose_next_stage(instruction, history, get_frame())
        if stage.is_terminal:
            print(f"[stagewise] {stage.describe()}")
            break
        print(f"[stagewise] next stage: {stage.describe()}")
        feedback = executor.run(stage, get_frame)
        history.append(feedback)
        print(f"[stagewise] stage feedback: {feedback.describe()}")
        if feedback.outcome == "aborted":
            print("[stagewise] stage was aborted -- halting the plan")
            break
    else:
        print(f"[stagewise] hit max_stages={max_stages} without reaching done/abort")
    return history
