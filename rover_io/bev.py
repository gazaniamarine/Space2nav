"""Live bird's-eye-view occupancy grid -- geometric only (free/occupied
from depth + odometry each tick), no semantic/landmark memory, no
persistence across runs. Deliberately the LemNAV occupancy.py idea (a
traversability-only BEV grid for the GUI's live viewer) without any of its
cross-subtask persistence or the fact3r semantic layer -- that's the
"memory part" explicitly left for later.

Fresh implementation (not vendored): reuses
qwen_grounding.goal_utils.pixel_depth_to_point's pixel+depth -> robot-frame
point convention, just applied over a depth image instead of one bbox
center, then rotated/translated into a world-ish frame by the current
odometry pose so the grid is robot-centric at reset time, not re-centered
every tick.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

UNKNOWN, FREE, OCCUPIED = 0, 1, 2


class BEVOccupancy:
    def __init__(self, size_m: float = 8.0, resolution_m: float = 0.05,
                 max_range_m: float = 4.0, stride: int = 8):
        self.size_m = size_m
        self.resolution_m = resolution_m
        self.max_range_m = max_range_m
        self.stride = stride  # subsample depth pixels for speed
        self.cells = int(size_m / resolution_m)
        self.grid = np.full((self.cells, self.cells), UNKNOWN, dtype=np.uint8)
        self.origin_xy = (0.0, 0.0)  # world-ish (x,y) at the grid's center cell

    def reset(self) -> None:
        self.grid.fill(UNKNOWN)

    def _world_to_cell(self, x: float, y: float) -> Optional[tuple]:
        gx = int((x - self.origin_xy[0]) / self.resolution_m + self.cells / 2)
        gy = int((y - self.origin_xy[1]) / self.resolution_m + self.cells / 2)
        if 0 <= gx < self.cells and 0 <= gy < self.cells:
            return gx, gy
        return None

    def integrate(self, depth_m: np.ndarray, robot_x: float, robot_y: float, robot_theta_rad: float,
                  fx: float, fy: float, cx: float, cy: float) -> None:
        """depth_m: HxW (or HxWx1) meters, current frame only."""
        d = depth_m[..., 0] if depth_m.ndim == 3 else depth_m
        h, w = d.shape
        ys, xs = np.mgrid[0:h:self.stride, 0:w:self.stride]
        depths = d[ys, xs]
        valid = np.isfinite(depths) & (depths > 0.1) & (depths < self.max_range_m)
        if not np.any(valid):
            return
        u, v, depths = xs[valid].astype(np.float32), ys[valid].astype(np.float32), depths[valid]

        # Pixel + depth -> camera-frame [x fwd, y left] (see goal_utils.pixel_depth_to_point).
        x_cam_fwd = depths
        y_cam_left = -(u - cx) / fx * depths

        cos_t, sin_t = math.cos(robot_theta_rad), math.sin(robot_theta_rad)
        world_x = robot_x + x_cam_fwd * cos_t - y_cam_left * sin_t
        world_y = robot_y + x_cam_fwd * sin_t + y_cam_left * cos_t

        robot_cell = self._world_to_cell(robot_x, robot_y)
        for wx, wy in zip(world_x, world_y):
            cell = self._world_to_cell(float(wx), float(wy))
            if cell is None:
                continue
            if robot_cell is not None:
                self._mark_ray_free(robot_cell, cell)
            self.grid[cell[1], cell[0]] = OCCUPIED

    def _mark_ray_free(self, start: tuple, end: tuple, samples: int = 16) -> None:
        (x0, y0), (x1, y1) = start, end
        for i in range(samples):
            t = i / max(samples - 1, 1)
            gx, gy = int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t))
            if 0 <= gx < self.cells and 0 <= gy < self.cells and self.grid[gy, gx] != OCCUPIED:
                self.grid[gy, gx] = FREE

    def to_image(self, robot_x: float = 0.0, robot_y: float = 0.0,
                 robot_theta_rad: float = 0.0) -> np.ndarray:
        """RGB render: unknown=gray, free=white, occupied=dark red, robot=blue dot+heading tick."""
        img = np.full((self.cells, self.cells, 3), 128, dtype=np.uint8)
        img[self.grid == FREE] = (235, 235, 235)
        img[self.grid == OCCUPIED] = (160, 30, 30)
        cell = self._world_to_cell(robot_x, robot_y)
        if cell is not None:
            gx, gy = cell
            img[max(gy - 2, 0):gy + 3, max(gx - 2, 0):gx + 3] = (30, 60, 220)
            hx = int(gx + 6 * math.cos(robot_theta_rad))
            hy = int(gy + 6 * math.sin(robot_theta_rad))
            if 0 <= hx < self.cells and 0 <= hy < self.cells:
                img[hy, hx] = (30, 60, 220)
        return np.flipud(img)  # y-up for display
