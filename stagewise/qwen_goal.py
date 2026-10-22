"""Default goal_point_fn for StageExecutor: Qwen itself does the detection
and grounding (qwen_grounding.detect_with_vlm), not Grounding-DINO. Resolves
a stage's target phrase to a pixel box each call, then projects that box
through metric depth + pinhole intrinsics into the (x,y,z) point
NavDP.predict_pointgoal expects (qwen_grounding.goal_point_from_detection).

Shares the SAME loaded Qwen3-VL weights as the stagewise planner/executor
(one ExplicitCoTVLM instance passed in) -- no second model load.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from explicit_cot.vlm import ExplicitCoTVLM
from qwen_grounding import detect_best_with_vlm, goal_point_from_detection, intrinsics_from_fov

from .types import Stage


class QwenGoalPointFn:
    """Callable matching stagewise.executor.GoalPointFn: (rgb, depth, Stage)
    -> Optional[(x, y, z)] meters in robot frame, or None if the target
    isn't visible (or depth isn't available) this tick -- the executor
    falls back to NavDP's no-goal/explore mode in that case, same as
    before."""

    def __init__(self, vlm: ExplicitCoTVLM, horizontal_fov_deg: float = 90.0,
                 box_threshold: float = 0.35, stop_distance: float = 0.5,
                 max_new_tokens: int = 256):
        self.vlm = vlm
        self.horizontal_fov_deg = horizontal_fov_deg
        self.box_threshold = box_threshold
        self.stop_distance = stop_distance
        self.max_new_tokens = max_new_tokens
        self.last_detection = None  # exposed for logging/debugging

    def __call__(self, rgb: np.ndarray, depth: Optional[np.ndarray],
                 stage: Stage) -> Optional[Tuple[float, float, float]]:
        if stage.kind not in ("object", "instruction") or not stage.phrase or depth is None:
            self.last_detection = None
            return None

        det = detect_best_with_vlm(
            self.vlm, rgb, stage.phrase,
            box_threshold=self.box_threshold, max_new_tokens=self.max_new_tokens)
        self.last_detection = det
        if det is None:
            return None

        h, w = rgb.shape[:2]
        fx, fy, cx, cy = intrinsics_from_fov(w, h, self.horizontal_fov_deg)
        depth_2d = depth[..., 0] if depth.ndim == 3 else depth
        goal = goal_point_from_detection(det.box, depth_2d, fx, fy, cx, cy,
                                          stop_distance=self.stop_distance)
        return tuple(goal.tolist()) if goal is not None else None
