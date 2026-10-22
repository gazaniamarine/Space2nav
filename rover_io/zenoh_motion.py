"""RoverMotionAdapter: Space2nav's stagewise.executor.MotionInterface
implementation for the real rover, speaking the SAME Zenoh/CDR contract as
LemNAV's nav_pipeline/zenoh_node.py (so the Pi/rover-side firmware and
bringup need no changes) -- camera/depth/camera_info/rpm subscriptions,
cmd_vel publish.

What's vendored vs. fresh, deliberately:
  - cdr.py:       wire-format byte parsing, vendored unchanged (has to
                  match the firmware's actual bytes, not something to
                  redesign).
  - odometry.py:  fresh, simpler dead-reckoning (wheel RPM only, no gyro/
                  visual-yaw fusion) -- see its own docstring for why.
  - this file:    fresh drive_straight/turn_in_place closed-loop control
                  and the MotionInterface glue, NOT ported from LemNAV's
                  runtime/mixins/motion.py.

NOT yet connected to a real rover in this session -- this is code-only
plumbing per explicit instruction; connecting to actual hardware and
sending real cmd_vel needs separate, explicit confirmation (rover powered
on, reachable, clear space to move) before use.
"""
from __future__ import annotations

import math
import time
from threading import Lock, Thread
from typing import Optional, Tuple

import numpy as np

from .cdr import (
    parse_camera_info,
    parse_compressed_image,
    parse_float32_multiarray,
    parse_image,
    serialize_twist,
)
from .odometry import SimpleOdometry

CAMERA_KEYS = ["image_raw", "rt/image_raw", "rover_camera", "rt/rover_camera"]
CAMERA_COMPRESSED_KEYS = ["image_raw/compressed", "rt/image_raw/compressed"]
DEPTH_KEYS = ["depth_raw", "rt/depth_raw", "rover_camera_depth", "rt/rover_camera_depth"]
CAMERA_INFO_KEYS = ["image_raw/camera_info", "rt/image_raw/camera_info"]
RPM_KEYS = ["rover/rpm", "rt/rover/rpm"]

# Rover firmware zeroes velocity if no cmd_vel arrives within ~500ms (see
# LemNAV's zenoh_node.py HEARTBEAT_PERIOD_S comment) -- re-send the last
# commanded velocity well under that deadline regardless of caller cadence,
# so a slow StageExecutor tick (e.g. during a VLM call) can't accidentally
# stall the rover mid-command.
HEARTBEAT_PERIOD_S = 0.15


class RoverMotionAdapter:
    """MotionInterface for the real rover over Zenoh. Call close() when done
    to release the session cleanly."""

    def __init__(self, pi_ip: Optional[str] = None, max_linear: float = 0.5,
                 max_angular: float = 0.6, drive_linear: float = 0.3,
                 turn_angular: float = 0.5, move_tol_m: float = 0.05,
                 turn_tol_deg: float = 3.0):
        import zenoh

        self.max_linear = max_linear
        self.max_angular = max_angular
        self.drive_linear = drive_linear
        self.turn_angular = turn_angular
        self.move_tol_m = move_tol_m
        self.turn_tol_deg = turn_tol_deg

        self._lock = Lock()
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_intrinsics: Optional[Tuple[float, float, float, float]] = None
        self.odom = SimpleOdometry()
        self._last_cmd = (0.0, 0.0)
        self._running = True

        config = zenoh.Config()
        if pi_ip:
            config.insert_json5("connect/endpoints", f'["tcp/{pi_ip}:7447"]')
        self.session = zenoh.open(config)
        self.pub_cmd = self.session.declare_publisher("cmd_vel")
        self.subs = (
            [self.session.declare_subscriber(k, self._on_image) for k in CAMERA_KEYS]
            + [self.session.declare_subscriber(k, self._on_image_compressed) for k in CAMERA_COMPRESSED_KEYS]
            + [self.session.declare_subscriber(k, self._on_depth) for k in DEPTH_KEYS]
            + [self.session.declare_subscriber(k, self._on_camera_info) for k in CAMERA_INFO_KEYS]
            + [self.session.declare_subscriber(k, self._on_rpm) for k in RPM_KEYS]
        )
        self._heartbeat_thread = Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    # ---------------- Zenoh callbacks ---------------- #
    def _on_image(self, sample):
        img = parse_image(bytes(sample.payload))
        if img is not None and img.ndim == 3:
            with self._lock:
                self._latest_rgb = img

    def _on_image_compressed(self, sample):
        img = parse_compressed_image(bytes(sample.payload))
        if img is not None:
            with self._lock:
                self._latest_rgb = img

    def _on_depth(self, sample):
        img = parse_image(bytes(sample.payload))
        if img is not None and img.ndim == 2:
            with self._lock:
                self._latest_depth = img[..., None]

    def _on_camera_info(self, sample):
        intr = parse_camera_info(bytes(sample.payload))
        if intr is not None:
            with self._lock:
                self._latest_intrinsics = intr

    def _on_rpm(self, sample):
        data = parse_float32_multiarray(bytes(sample.payload))
        if len(data) >= 2:
            self.odom.update(data[0], data[1])

    # ---------------- heartbeat ---------------- #
    def _heartbeat_loop(self):
        while self._running:
            self.pub_cmd.put(serialize_twist(*self._last_cmd))
            time.sleep(HEARTBEAT_PERIOD_S)

    # ---------------- MotionInterface ---------------- #
    def get_rgbd(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            return self._latest_rgb, self._latest_depth

    @property
    def intrinsics(self) -> Optional[Tuple[float, float, float, float]]:
        with self._lock:
            return self._latest_intrinsics

    def set_velocity(self, linear: float, angular: float) -> None:
        linear = float(np.clip(linear, -self.max_linear, self.max_linear))
        angular = float(np.clip(angular, -self.max_angular, self.max_angular))
        self._last_cmd = (linear, angular)
        self.pub_cmd.put(serialize_twist(linear, angular))

    def drive_straight(self, meters: float, timeout_s: Optional[float] = None) -> dict:
        target = abs(meters)
        sign = 1.0 if meters >= 0 else -1.0
        timeout_s = timeout_s or max(5.0, target / max(self.drive_linear, 0.05) * 2.5)
        self.odom.reset()
        t0 = time.monotonic()
        try:
            while True:
                if self.odom.distance_traveled_m >= target - self.move_tol_m:
                    return {"outcome": "ok", "traveled_m": self.odom.distance_traveled_m}
                if time.monotonic() - t0 > timeout_s:
                    return {"outcome": "timeout", "traveled_m": self.odom.distance_traveled_m}
                self.set_velocity(sign * self.drive_linear, 0.0)
                time.sleep(0.05)
        finally:
            self.set_velocity(0.0, 0.0)

    def turn_in_place(self, degrees: float, timeout_s: Optional[float] = None) -> dict:
        target_rad = abs(math.radians(degrees))
        sign = 1.0 if degrees >= 0 else -1.0
        timeout_s = timeout_s or max(5.0, target_rad / max(self.turn_angular, 0.05) * 2.5)
        self.odom.reset()
        t0 = time.monotonic()
        try:
            while True:
                if abs(self.odom.pose.theta_rad) >= target_rad - math.radians(self.turn_tol_deg):
                    return {"outcome": "ok", "turned_deg": self.odom.heading_deg}
                if time.monotonic() - t0 > timeout_s:
                    return {"outcome": "timeout", "turned_deg": self.odom.heading_deg}
                self.set_velocity(0.0, sign * self.turn_angular)
                time.sleep(0.05)
        finally:
            self.set_velocity(0.0, 0.0)

    def close(self) -> None:
        self._running = False
        self.set_velocity(0.0, 0.0)
        self._heartbeat_thread.join(timeout=1.0)
        for s in self.subs:
            s.undeclare()
        self.pub_cmd.undeclare()
        self.session.close()
