"""Fresh, self-contained differential-drive dead-reckoning odometry for
Space2nav's rover adapter -- deliberately simpler than LemNAV's
OdometryLogger (nav_pipeline/odometry_logger.py), which fuses wheel
encoders with a BNO085 gyro and a visual-yaw fallback. This integrates
wheel RPM only: x/y/theta drift with distance and turning, same as any
pure dead-reckoning estimate, but it is enough to close the loop on
drive_straight/turn_in_place over the short distances/angles a single
stage actually commands (bounded by each call's own timeout) -- it does
not try to track position over an entire multi-stage mission.

Kinematics constants below MUST match the rover firmware
(esp32/rover_6wd_complete.ino WHEEL_RADIUS_M / TRACK_WIDTH_M) -- same
values LemNAV's odometry_logger.py uses, since this is the same physical
rover.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

WHEEL_RADIUS_M = 0.056
TRACK_WIDTH_M = 0.345


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    theta_rad: float = 0.0  # heading, CCW from the pose at construction time


class SimpleOdometry:
    """Feed it (left_rpm, right_rpm) samples as they arrive (e.g. from the
    rover's rpm Zenoh topic); reset() at the start of each blind
    move/turn so distance/heading traveled is measured from THAT stage's
    start, not accumulated error from everything before it."""

    def __init__(self):
        self.pose = Pose2D()
        self._last_t: "float | None" = None

    def reset(self) -> None:
        self.pose = Pose2D()
        self._last_t = None

    def update(self, left_rpm: float, right_rpm: float, t: "float | None" = None) -> Pose2D:
        t = time.monotonic() if t is None else t
        if self._last_t is None:
            self._last_t = t
            return self.pose
        dt = t - self._last_t
        self._last_t = t
        if dt <= 0.0 or dt > 1.0:  # clock jump / stale callback -- skip, don't integrate garbage
            return self.pose

        v_left = left_rpm * 2.0 * math.pi * WHEEL_RADIUS_M / 60.0
        v_right = right_rpm * 2.0 * math.pi * WHEEL_RADIUS_M / 60.0
        v = (v_left + v_right) / 2.0
        w = (v_right - v_left) / TRACK_WIDTH_M

        # Midpoint heading for this step's displacement (small-dt arc
        # approximation) -- more accurate than using the pre-step heading
        # alone when both v and w are nonzero.
        theta_mid = self.pose.theta_rad + 0.5 * w * dt
        self.pose.x += v * dt * math.cos(theta_mid)
        self.pose.y += v * dt * math.sin(theta_mid)
        self.pose.theta_rad += w * dt
        return self.pose

    @property
    def distance_traveled_m(self) -> float:
        return math.hypot(self.pose.x, self.pose.y)

    @property
    def heading_deg(self) -> float:
        return math.degrees(self.pose.theta_rad)
