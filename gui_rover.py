#!/usr/bin/env python3
"""Self-contained viewer + launcher for the real rover: ONE process, ONE
command -- no separate server to start first.

  1. PASSIVE VIEWER (always on): opens its OWN Zenoh session and subscribes
     to the rover's camera/depth/RPM feed directly (rover_io.cdr parsers) --
     works whether or not a run is in progress, zero control surface.
  2. LAUNCHER (opt-in, only once you type an instruction and click Launch):
     runs the stagewise loop in a background thread. The FIRST Launch loads
     Qwen3-VL + NavDP (shown in the log, ~15-20s) and caches them on this
     GUI object; every Launch after that reuses the already-loaded models --
     no reload per instruction. A separate RoverMotionAdapter (not the
     passive viewer's own Zenoh session) is what StageExecutor actually
     drives with, lazily constructed alongside the models.
  3. EMERGENCY STOP: publishes a single zero-velocity Twist directly on the
     passive viewer's own Zenoh session, independent of whatever the
     launcher thread is doing -- the one safety path that never depends on
     anything else in this file working correctly.

Run (this is the only command you need):
    python gui_rover.py --pi-ip 10.72.241.246 --yes-drive
    python gui_rover.py --simulate              # no hardware at all
"""
from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import scrolledtext
from typing import Optional

import numpy as np
import zenoh
from PIL import Image, ImageTk

from explicit_cot.vlm import DEFAULT_MODEL_ID, ExplicitCoTVLM
from nav_dp import NavDPStandalone
from qwen_grounding.goal_utils import intrinsics_from_fov
from rover_io.bev import BEVOccupancy
from rover_io.cdr import (
    parse_camera_info,
    parse_compressed_image,
    parse_float32_multiarray,
    parse_image,
    serialize_twist,
)
from rover_io.odometry import SimpleOdometry
from stagewise import StageExecutor, StagewisePlanner, TraceLogger
from stagewise.planner import run_stagewise_loop

CAMERA_KEYS = ["image_raw", "rt/image_raw", "rover_camera", "rt/rover_camera"]
CAMERA_COMPRESSED_KEYS = ["image_raw/compressed", "rt/image_raw/compressed"]
DEPTH_KEYS = ["depth_raw", "rt/depth_raw", "rover_camera_depth", "rt/rover_camera_depth"]
CAMERA_INFO_KEYS = ["image_raw/camera_info", "rt/image_raw/camera_info"]
RPM_KEYS = ["rover/rpm", "rt/rover/rpm"]
BEV_REFRESH_MS = 300


class _SimulatedMotionLike:
    """No-hardware MotionInterface for --simulate, local copy so this file
    has no import-time dependency on run_stagewise_demo.py."""

    def __init__(self, image_size: int = 224):
        self._frame = (np.random.rand(image_size, image_size, 3) * 255).astype("uint8")
        self._depth = np.full((image_size, image_size, 1), 2.0, dtype="float32")

    def drive_straight(self, meters: float) -> dict:
        time.sleep(0.2)
        return {"outcome": "ok"}

    def turn_in_place(self, degrees: float) -> dict:
        time.sleep(0.2)
        return {"outcome": "ok"}

    def set_velocity(self, linear: float, angular: float) -> None:
        pass

    def get_rgbd(self):
        return self._frame, self._depth

    def close(self) -> None:
        pass


class _TeeToQueue:
    """Every print() in this process (vlm.py, planner.py, executor.py, this
    file) goes to the real terminal AND into the GUI's log queue -- so the
    Run Log panel shows live CoT reasoning without this file needing to
    know about any of those callers individually."""

    def __init__(self, real, log_queue: "queue.Queue[str]"):
        self._real = real
        self._queue = log_queue

    def write(self, s: str) -> int:
        self._real.write(s)
        if s.strip():
            self._queue.put(s if s.endswith("\n") else s + "\n")
        return len(s)

    def flush(self) -> None:
        self._real.flush()


class RoverGUI:
    def __init__(self, root: tk.Tk, pi_ip: Optional[str], args: argparse.Namespace,
                 default_instruction: str = ""):
        self.root = root
        self.pi_ip = pi_ip
        self.args = args
        self._log_queue: "queue.Queue[str]" = queue.Queue()
        sys.stdout = _TeeToQueue(sys.stdout, self._log_queue)

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        # Passive viewer's OWN odometry + BEV grid -- geometric only, reset
        # each GUI session, no semantic/landmark memory (left for later).
        self.odom = SimpleOdometry()
        self.bev = BEVOccupancy()

        # Lazily constructed on the FIRST Launch, then cached for the life
        # of this GUI process -- this is the actual fix for "model loads
        # every instruction": there is no separate server process to start;
        # these just live on self for as long as the window is open.
        self.vlm: Optional[ExplicitCoTVLM] = None
        self.navdp: Optional[NavDPStandalone] = None
        self.motion = None  # RoverMotionAdapter or _SimulatedMotionLike
        self.planner: Optional[StagewisePlanner] = None
        self._busy = False
        self._current_stop_event: Optional[threading.Event] = None

        # Passive viewer's own Zenoh session (camera/depth/rpm display) +
        # the independent EMERGENCY STOP -- separate from self.motion,
        # which StageExecutor actually drives with once it exists.
        cfg = zenoh.Config()
        if pi_ip:
            cfg.insert_json5("connect/endpoints", f'["tcp/{pi_ip}:7447"]')
        self.session = zenoh.open(cfg)
        self.pub_cmd = self.session.declare_publisher("cmd_vel")
        self.subs = (
            [self.session.declare_subscriber(k, self._on_image) for k in CAMERA_KEYS]
            + [self.session.declare_subscriber(k, self._on_image_compressed) for k in CAMERA_COMPRESSED_KEYS]
            + [self.session.declare_subscriber(k, self._on_depth) for k in DEPTH_KEYS]
            + [self.session.declare_subscriber(k, self._on_camera_info) for k in CAMERA_INFO_KEYS]
            + [self.session.declare_subscriber(k, self._on_rpm) for k in RPM_KEYS]
        )
        self.intrinsics = None

        self._build_ui(default_instruction)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_log_queue)
        self.root.after(150, self._refresh_frame)
        self.root.after(BEV_REFRESH_MS, self._refresh_bev)

    # ---------------- Zenoh callbacks (passive viewer) ---------------- #
    def _on_image(self, sample):
        img = parse_image(bytes(sample.payload))
        if img is not None and img.ndim == 3:
            self._latest_frame = img

    def _on_image_compressed(self, sample):
        img = parse_compressed_image(bytes(sample.payload))
        if img is not None:
            self._latest_frame = img

    def _on_depth(self, sample):
        img = parse_image(bytes(sample.payload))
        if img is not None and img.ndim == 2:
            self._latest_depth = img

    def _on_camera_info(self, sample):
        intr = parse_camera_info(bytes(sample.payload))
        if intr is not None:
            self.intrinsics = intr

    def _on_rpm(self, sample):
        data = parse_float32_multiarray(bytes(sample.payload))
        if len(data) >= 2:
            self.odom.update(data[0], data[1])

    # ---------------- UI ---------------- #
    def _build_ui(self, default_instruction: str):
        self.root.title(f"Space2nav rover viewer -- {self.pi_ip or ('simulate' if self.args.simulate else 'multicast')}")
        self.root.geometry("1500x850")

        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        tk.Label(top, text="Instruction:").pack(side=tk.LEFT)
        self.instruction_var = tk.StringVar(value=default_instruction)
        tk.Entry(top, textvariable=self.instruction_var, width=60).pack(side=tk.LEFT, padx=4)
        self.launch_btn = tk.Button(top, text="Launch", command=self._on_launch)
        self.launch_btn.pack(side=tk.LEFT, padx=4)
        self.kill_btn = tk.Button(top, text="Stop run", command=self._on_kill)
        self.kill_btn.pack(side=tk.LEFT, padx=4)
        stop_btn = tk.Button(top, text="EMERGENCY STOP (zero cmd_vel)", bg="#c0392b", fg="white",
                              font=("TkDefaultFont", 10, "bold"), command=self._on_emergency_stop)
        stop_btn.pack(side=tk.RIGHT, padx=4)

        body = tk.Frame(self.root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=4)

        cam_frame = tk.Frame(body)
        cam_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(cam_frame, text="Live feed (Qwen grounds the target only during an active "
                                  "stage -- no continuous detection overlay)", anchor="w").pack(
            side=tk.TOP, fill=tk.X)
        self.canvas = tk.Label(cam_frame, text="(no camera frame yet)", width=50, height=20,
                                bg="black", fg="white")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.pose_var = tk.StringVar(value="pose  x=+0.00  y=+0.00  theta=+0deg")
        tk.Label(cam_frame, textvariable=self.pose_var, anchor="w",
                 font=("Courier", 10)).pack(side=tk.TOP, fill=tk.X)

        bev_frame = tk.Frame(body)
        bev_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        bev_header = tk.Frame(bev_frame)
        bev_header.pack(side=tk.TOP, fill=tk.X)
        tk.Label(bev_header, text="BEV occupancy (local, geometric only -- no memory)",
                 anchor="w").pack(side=tk.LEFT)
        tk.Button(bev_header, text="Clear map", command=self._on_clear_bev).pack(side=tk.RIGHT)
        self.bev_canvas = tk.Label(bev_frame, text="(no depth yet)", width=30, height=18,
                                    bg="gray20", fg="white")
        self.bev_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        log_frame = tk.Frame(body)
        log_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        tk.Label(log_frame, text="Run Log -- planner / executor / CoT reasoning (this process's stdout)",
                 anchor="w").pack(side=tk.TOP, fill=tk.X)
        self.log = scrolledtext.ScrolledText(log_frame, width=80, height=30, state=tk.DISABLED,
                                              font=("Courier", 10), wrap=tk.WORD)
        self.log.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="idle (models not loaded yet -- first Launch will load them)")
        tk.Label(self.root, textvariable=self.status_var, anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

    def _refresh_frame(self):
        if self._latest_frame is not None:
            img = Image.fromarray(self._latest_frame).resize((480, 360))
            self._tk_img = ImageTk.PhotoImage(img)  # keep a reference -- Tk drops GC'd images
            self.canvas.configure(image=self._tk_img, text="")
        pose = self.odom.pose
        self.pose_var.set(f"pose  x={pose.x:+.2f}  y={pose.y:+.2f}  "
                           f"theta={math.degrees(pose.theta_rad):+.0f}deg")
        self.root.after(150, self._refresh_frame)

    def _refresh_bev(self):
        if self._latest_depth is not None:
            h, w = self._latest_depth.shape[:2]
            fx, fy, cx, cy = self.intrinsics or intrinsics_from_fov(w, h, 90.0)
            pose = self.odom.pose
            self.bev.integrate(self._latest_depth, pose.x, pose.y, pose.theta_rad, fx, fy, cx, cy)
            bev_img = self.bev.to_image(pose.x, pose.y, pose.theta_rad)
            img = Image.fromarray(bev_img).resize((300, 300), Image.NEAREST)
            self._tk_bev = ImageTk.PhotoImage(img)  # keep a reference -- Tk drops GC'd images
            self.bev_canvas.configure(image=self._tk_bev, text="")
        self.root.after(BEV_REFRESH_MS, self._refresh_bev)

    def _on_clear_bev(self):
        self.bev.reset()
        self.odom.reset()
        self._append_log("[gui] BEV grid cleared and odometry re-zeroed.\n")

    # ---------------- launcher: models load once, cached on self ------ #
    def _ensure_models_loaded(self):
        """Called on the background thread, not the Tk thread -- the only
        thing that makes the FIRST Launch slow and every Launch after it
        fast. Idempotent: returns immediately if already loaded."""
        if self.vlm is not None:
            return
        print(f"[gui] loading {self.args.model_id} + NavDP for the first time "
              f"-- this only happens once for this GUI session...")
        self.vlm = ExplicitCoTVLM(model_id=self.args.model_id, device=self.args.device,
                                   load_in_4bit=self.args.load_in_4bit)
        self.vlm._ensure_loaded()  # eager -- don't leave this Launch's first VLM call to pay for it
        navdp_kwargs = {"device": self.args.device}
        self.navdp = (NavDPStandalone.load(weights_path=self.args.navdp_weights, **navdp_kwargs)
                      if self.args.navdp_weights else NavDPStandalone.load(**navdp_kwargs))
        self.planner = StagewisePlanner(self.vlm)

        if self.args.simulate:
            self.motion = _SimulatedMotionLike()
        else:
            from rover_io import RoverMotionAdapter
            self.motion = RoverMotionAdapter(pi_ip=self.pi_ip, max_linear=self.args.max_linear,
                                              max_angular=self.args.max_angular)
            print("[gui] waiting for first camera frame for the drive adapter...")
            t0 = time.monotonic()
            while self.motion.get_rgbd()[0] is None:
                if time.monotonic() - t0 > 15.0:
                    raise RuntimeError("no camera frame received within 15s for the drive adapter")
                time.sleep(0.2)
            print("[gui] drive adapter ready. LIVE ROVER -- real cmd_vel will be sent.")
        print("[gui] models loaded and staying resident for the rest of this session.")

    def _on_launch(self):
        instruction = self.instruction_var.get().strip()
        if not instruction:
            self._append_log("[gui] type an instruction before launching.\n")
            return
        if self._busy:
            self._append_log("[gui] a run is already in progress -- Stop run first.\n")
            return
        self._busy = True
        self.launch_btn.config(state=tk.DISABLED)
        self.status_var.set(f"running: {instruction!r}")
        threading.Thread(target=self._run_instruction, args=(instruction,), daemon=True).start()

    def _run_instruction(self, instruction: str):
        try:
            self._ensure_models_loaded()
            stop_event = threading.Event()
            self._current_stop_event = stop_event
            trace = TraceLogger(label=instruction.replace(" ", "_")[:30])
            executor = StageExecutor(self.navdp, self.vlm, self.motion, timeout_s=self.args.timeout_s,
                                      stop_event=stop_event, trace_logger=trace)
            print(f"[gui] starting: {instruction!r}  (trace: {trace.path})")
            history = run_stagewise_loop(self.planner, executor, instruction,
                                          lambda: self.motion.get_rgbd()[0],
                                          max_stages=self.args.max_stages, stop_event=stop_event)
            trace.close()
            print(f"[gui] finished {instruction!r} with {len(history)} stage(s)")
        except Exception as e:
            print(f"[gui] run failed: {e!r}")
        finally:
            self._current_stop_event = None
            self._busy = False
            self._log_queue.put("__RUN_DONE__")

    def _on_kill(self):
        if self._current_stop_event is not None:
            self._current_stop_event.set()
            self._append_log("[gui] stop requested for the in-progress run.\n")
        else:
            self._append_log("[gui] nothing running -- Stop run has no effect right now.\n")
        # Independent of whether the stop_event lands in time -- see
        # _on_emergency_stop, the real safety backstop.
        self._on_emergency_stop(log_prefix="[gui] stop also sent a direct cmd_vel: ")

    def _on_emergency_stop(self, log_prefix: str = "[gui] "):
        self.pub_cmd.put(serialize_twist(0.0, 0.0))
        if self.motion is not None:
            self.motion.set_velocity(0.0, 0.0)
        self._append_log(f"{log_prefix}EMERGENCY STOP -- zero-velocity cmd_vel sent directly.\n")

    def _poll_log_queue(self):
        try:
            while True:
                line = self._log_queue.get_nowait()
                if line == "__RUN_DONE__":
                    self.launch_btn.config(state=tk.NORMAL)
                    self.status_var.set("idle (models stay loaded for the next instruction)")
                    continue
                self._append_log(line)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    def _append_log(self, text: str):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text)
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _on_close(self):
        self._on_emergency_stop(log_prefix="[gui] closing -- ")
        if self._current_stop_event is not None:
            self._current_stop_event.set()
        if self.motion is not None and hasattr(self.motion, "close"):
            self.motion.close()
        for s in self.subs:
            s.undeclare()
        self.pub_cmd.undeclare()
        self.session.close()
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pi-ip", type=str, default=None, help="rover's Zenoh endpoint IP")
    ap.add_argument("--simulate", action="store_true", help="no hardware -- SimulatedMotion stub")
    ap.add_argument("--yes-drive", action="store_true",
                     help="required together with --pi-ip before this GUI will send real cmd_vel")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--navdp-weights", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--max-stages", type=int, default=40)
    ap.add_argument("--timeout-s", type=float, default=45.0)
    ap.add_argument("--max-linear", type=float, default=0.5)
    ap.add_argument("--max-angular", type=float, default=0.6)
    ap.add_argument("--default-instruction", type=str, default="")
    args = ap.parse_args()

    if not args.simulate and not args.pi_ip:
        raise SystemExit("pass --simulate or --pi-ip <rover IP> --yes-drive")
    if args.pi_ip and not args.yes_drive:
        raise SystemExit("--pi-ip given without --yes-drive -- refusing to allow real cmd_vel")

    root = tk.Tk()
    RoverGUI(root, args.pi_ip, args, default_instruction=args.default_instruction)
    root.mainloop()


if __name__ == "__main__":
    main()
