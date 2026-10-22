#!/usr/bin/env python3
"""Persistent server: loads Qwen3-VL + NavDP ONCE and keeps them loaded,
accepting new instructions over Zenoh for as long as it runs -- the fix for
"the model is loading the weights after every instruction". Previously
each GUI "Launch" spawned a fresh run_stagewise_demo.py subprocess, which
reloaded both models (~15-20s) every single time.

Start this ONCE:
    python run_stagewise_server.py --pi-ip 10.72.241.246

Then gui_rover.py (or anything) sends instructions by publishing plain text
to the "space2nav/instruction" Zenoh key, and "stop" to "space2nav/control"
to abort the current run. All stdout (CoT reasoning, stage/feedback lines,
same as before) is tee'd to both this process's own terminal AND republished
on "space2nav/status" for any number of GUI viewers to subscribe to.

Only one instruction runs at a time; a new one arriving while busy is
rejected (not queued) -- send a "stop" first if you want to interrupt and
replace what's running.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time

import zenoh

from explicit_cot.vlm import DEFAULT_MODEL_ID, ExplicitCoTVLM
from nav_dp import NavDPStandalone
from stagewise import StageExecutor, StagewisePlanner, TraceLogger
from stagewise.planner import run_stagewise_loop

INSTRUCTION_KEY = "space2nav/instruction"
CONTROL_KEY = "space2nav/control"
STATUS_KEY = "space2nav/status"


class _TeeStdout:
    """Every print() in this process (vlm.py, planner.py, executor.py, this
    file) goes to the real terminal AND gets republished on STATUS_KEY, so
    any number of GUI viewers can watch live without this file needing to
    know about any of those callers individually."""

    def __init__(self, real, pub):
        self._real = real
        self._pub = pub

    def write(self, s: str) -> int:
        self._real.write(s)
        if s.strip():
            try:
                self._pub.put(s)
            except Exception:
                pass
        return len(s)

    def flush(self) -> None:
        self._real.flush()


class _SimulatedMotionLike:
    """Local copy of run_stagewise_demo.py's SimulatedMotion so this file
    doesn't need to import a script named for one-shot CLI use."""

    def __init__(self, image_size: int = 224):
        import numpy as np
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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--navdp-weights", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--max-stages", type=int, default=40)
    p.add_argument("--timeout-s", type=float, default=45.0)
    p.add_argument("--pi-ip", type=str, default=None,
                    help="rover's Zenoh endpoint IP; omit for --simulate")
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--yes-drive", action="store_true",
                    help="required together with --pi-ip, same confirmation as run_stagewise_demo.py")
    p.add_argument("--max-linear", type=float, default=0.5)
    p.add_argument("--max-angular", type=float, default=0.6)
    return p


def main():
    args = build_arg_parser().parse_args()
    if not args.simulate and not args.pi_ip:
        raise SystemExit("pass --simulate or --pi-ip <rover IP> --yes-drive")
    if args.pi_ip and not args.yes_drive:
        raise SystemExit("--pi-ip given without --yes-drive -- refusing to send real cmd_vel")

    cfg = zenoh.Config()
    if args.pi_ip:
        cfg.insert_json5("connect/endpoints", f'["tcp/{args.pi_ip}:7447"]')
    session = zenoh.open(cfg)
    pub_status = session.declare_publisher(STATUS_KEY)
    sys.stdout = _TeeStdout(sys.stdout, pub_status)

    print(f"[server] loading {args.model_id} + NavDP once -- this is the ONLY time this happens...")
    vlm = ExplicitCoTVLM(model_id=args.model_id, device=args.device, load_in_4bit=args.load_in_4bit)
    # ExplicitCoTVLM lazy-loads on its first .ask() call by design (so a
    # script that never calls it never pays the cost) -- force it NOW
    # instead of leaving the first real instruction to eat the ~15-20s
    # load latency inline. This is the actual fix for "the model is
    # loading weights after every instruction": without it, weights only
    # loaded once, but still on the first call, not at startup.
    vlm._ensure_loaded()
    navdp_kwargs = {"device": args.device}
    navdp = (NavDPStandalone.load(weights_path=args.navdp_weights, **navdp_kwargs)
             if args.navdp_weights else NavDPStandalone.load(**navdp_kwargs))
    print("[server] models loaded and staying resident. Waiting for instructions...")

    if args.pi_ip:
        from rover_io import RoverMotionAdapter
        motion = RoverMotionAdapter(pi_ip=args.pi_ip, max_linear=args.max_linear,
                                     max_angular=args.max_angular)
        print("[server] waiting for first camera frame over Zenoh...")
        t0 = time.time()
        while motion.get_rgbd()[0] is None:
            if time.time() - t0 > 15.0:
                raise SystemExit("no camera frame received within 15s -- check the rover/--pi-ip")
            time.sleep(0.2)
        print("[server] first frame received. LIVE ROVER -- real cmd_vel will be sent.")
    else:
        motion = _SimulatedMotionLike()

    planner = StagewisePlanner(vlm)
    busy_lock = threading.Lock()
    current_stop_event: "threading.Event | None" = None

    def run_one(instruction: str):
        nonlocal current_stop_event
        if not busy_lock.acquire(blocking=False):
            print(f"[server] busy -- ignoring {instruction!r} (send 'stop' to /control first)")
            return
        try:
            stop_event = threading.Event()
            current_stop_event = stop_event
            trace = TraceLogger(label=instruction.replace(" ", "_")[:30])
            executor = StageExecutor(navdp, vlm, motion, timeout_s=args.timeout_s,
                                      stop_event=stop_event, trace_logger=trace)
            print(f"[server] starting: {instruction!r}  (trace: {trace.path})")
            history = run_stagewise_loop(planner, executor, instruction,
                                          lambda: motion.get_rgbd()[0],
                                          max_stages=args.max_stages, stop_event=stop_event)
            trace.close()
            print(f"[server] finished {instruction!r} with {len(history)} stage(s)")
        finally:
            current_stop_event = None
            busy_lock.release()

    def on_instruction(sample):
        instruction = bytes(sample.payload).decode("utf-8", errors="replace").strip()
        if instruction:
            threading.Thread(target=run_one, args=(instruction,), daemon=True).start()

    def on_control(sample):
        cmd = bytes(sample.payload).decode("utf-8", errors="replace").strip().lower()
        if cmd == "stop":
            if current_stop_event is not None:
                current_stop_event.set()
                print("[server] STOP requested via /control")
            motion.set_velocity(0.0, 0.0)

    session.declare_subscriber(INSTRUCTION_KEY, on_instruction)
    session.declare_subscriber(CONTROL_KEY, on_control)
    print(f"[server] ready. Subscribed to '{INSTRUCTION_KEY}' / '{CONTROL_KEY}', "
          f"publishing status on '{STATUS_KEY}'.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[server] shutting down")
        motion.set_velocity(0.0, 0.0)
        if hasattr(motion, "close"):
            motion.close()
        session.close()


if __name__ == "__main__":
    main()
