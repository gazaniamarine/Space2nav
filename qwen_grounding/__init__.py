"""Qwen-as-detector: open-vocabulary grounding via the same Qwen3-VL model
already loaded for stagewise planning/execution, instead of Grounding-DINO.
"""
from .detector import Detection, detect_best_with_vlm, detect_with_vlm
from .goal_utils import goal_point_from_detection, intrinsics_from_fov

__all__ = [
    "Detection", "detect_with_vlm", "detect_best_with_vlm",
    "goal_point_from_detection", "intrinsics_from_fov",
]
