"""Bbox + metric depth + pinhole intrinsics -> 3D goal point.

Vendored unchanged (pure numpy/cv2 math, no Qwen- or NavDP-specific
dependency) from
/home/gpu/Desktop/Nihil/LemNAV/nav_pipeline/goal_utils.py.
"""
from typing import Optional, Tuple

import numpy as np


def bbox_median_depth(depth_m: np.ndarray, box: np.ndarray, shrink: float = 0.25) -> Optional[float]:
    """Median of valid depth inside a centrally shrunk bbox (meters)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    x0s, x1s = int(x0 + w * shrink), int(x1 - w * shrink)
    y0s, y1s = int(y0 + h * shrink), int(y1 - h * shrink)
    x0s, y0s = max(x0s, 0), max(y0s, 0)
    x1s = min(max(x1s, x0s + 1), depth_m.shape[1])
    y1s = min(max(y1s, y0s + 1), depth_m.shape[0])
    patch = depth_m[y0s:y1s, x0s:x1s]
    valid = patch[np.isfinite(patch) & (patch > 0.1)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def pixel_depth_to_point(
    u: float, v: float, d: float, fx: float, fy: float, cx: float, cy: float,
    stop_distance: float = 0.0,
) -> np.ndarray:
    """Pixel (u,v) at depth d (m) -> [x fwd, y left, z up] robot-frame goal.

    Camera optical frame (z fwd, x right, y down) is converted to the robot
    frame NavDP expects (x fwd, y left, z up). stop_distance shortens the
    goal along the ray so the rover stops in front of the object.
    """
    x_cam = (u - cx) / fx * d  # right
    y_cam = (v - cy) / fy * d  # down
    goal = np.array([d, -x_cam, -y_cam], dtype=np.float32)
    dist = np.linalg.norm(goal)
    if stop_distance > 0.0 and dist > stop_distance:
        goal *= (dist - stop_distance) / dist
    return goal


def goal_point_from_detection(
    box: np.ndarray, depth_m: np.ndarray, fx: float, fy: float, cx: float, cy: float,
    stop_distance: float = 0.0,
) -> Optional[np.ndarray]:
    """bbox + metric depth + pinhole intrinsics -> [x fwd, y left, z up] goal."""
    d = bbox_median_depth(depth_m, box)
    if d is None:
        return None
    u, v = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    return pixel_depth_to_point(u, v, d, fx, fy, cx, cy, stop_distance)


def intrinsics_from_fov(width: int, height: int, horizontal_fov_deg: float) -> Tuple[float, float, float, float]:
    """Approximate pinhole intrinsics from image size + horizontal FOV."""
    fx = width / (2.0 * np.tan(np.radians(horizontal_fov_deg) / 2.0))
    fy = fx
    return fx, fy, width / 2.0, height / 2.0
