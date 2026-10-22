"""Entry point wiring: explicit-CoT planner <-> explicit-CoT+NavDP executor,
closed over stage boundaries (Phase 1-3 of the SpaceVLN-without-memory plan),
with Qwen itself doing detection/grounding (StageExecutor's default
QwenGoalPointFn) and NavDP doing low-level motion.

Two backends for the MotionInterface:
  --simulate            SimulatedMotion stub (random frame, no real robot) --
                         exercises the stage-boundary loop end to end with
                         no hardware at all.
  --pi-ip <rover IP>     RoverMotionAdapter (rover_io/) -- the REAL rover
                         over Zenoh. Sends real cmd_vel. Requires --yes-drive
                         too, as an explicit second confirmation beyond just
                         supplying an IP, since this moves a physical robot.

    python run_stagewise_demo.py "go to the red chair then turn left" --simulate
    python run_stagewise_demo.py "go to the red chair" --pi-ip 10.86.180.125 --yes-drive
"""
from __future__ import annotations

import argparse
import time
from typing import Optional, Tuple

import numpy as np

from explicit_cot.vlm import DEFAULT_MODEL_ID, ExplicitCoTVLM
from nav_dp import NavDPStandalone
from stagewise import StageExecutor, StagewisePlanner
from stagewise.planner import run_stagewise_loop


class SimulatedMotion:
    """Stand-in MotionInterface for exercising the stagewise loop without a
    robot: returns a fixed blank-ish frame and no-ops on drive commands.
    Replace with a real MotionInterface (odometry + camera) to run this on
    hardware."""

    def __init__(self, image_size: int = 224):
        self._frame = (np.random.rand(image_size, image_size, 3) * 255).astype("uint8")
        # No real depth sensor here -- a flat plausible-range plane just so
        # QwenGoalPointFn's projection path has something to divide by
        # instead of always short-circuiting on depth=None.
        self._depth = np.full((image_size, image_size, 1), 2.0, dtype=np.float32)

    def drive_straight(self, meters: float) -> dict:
        print(f"[sim-motion] drive_straight({meters:+.2f}m)")
        time.sleep(0.2)
        return {"outcome": "ok"}

    def turn_in_place(self, degrees: float) -> dict:
        print(f"[sim-motion] turn_in_place({degrees:+.1f}deg)")
        time.sleep(0.2)
        return {"outcome": "ok"}

    def set_velocity(self, linear: float, angular: float) -> None:
        pass  # would publish to a real base controller

    def get_rgbd(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self._frame, self._depth


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("instruction", help="free-text navigation instruction")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--navdp-weights", default=None,
                    help="path to navdp-cross-modal.ckpt; defaults to nav_dp's built-in path")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--max-stages", type=int, default=40)
    p.add_argument("--timeout-s", type=float, default=45.0)
    p.add_argument("--simulate", action="store_true",
                    help="use SimulatedMotion instead of real hardware")
    p.add_argument("--pi-ip", type=str, default=None,
                    help="rover's Zenoh endpoint IP -- use RoverMotionAdapter (real hardware). "
                         "Requires --yes-drive.")
    p.add_argument("--yes-drive", action="store_true",
                    help="explicit confirmation that sending real cmd_vel to the rover at "
                         "--pi-ip is intended. Required together with --pi-ip; a typo'd or "
                         "forgotten flag refuses to drive rather than guessing.")
    p.add_argument("--max-linear", type=float, default=0.5)
    p.add_argument("--max-angular", type=float, default=0.6)
    return p


def main():
    args = build_arg_parser().parse_args()
    if not args.simulate and not args.pi_ip:
        raise SystemExit("pass --simulate (no hardware) or --pi-ip <rover IP> --yes-drive (real rover)")
    if args.pi_ip and not args.yes_drive:
        raise SystemExit("--pi-ip given without --yes-drive -- refusing to send real cmd_vel "
                          "without that explicit confirmation")
    if args.simulate and args.pi_ip:
        raise SystemExit("--simulate and --pi-ip are mutually exclusive")

    vlm = ExplicitCoTVLM(model_id=args.model_id, device=args.device,
                          load_in_4bit=args.load_in_4bit)
    navdp_kwargs = {"device": args.device}
    if args.navdp_weights:
        navdp = NavDPStandalone.load(weights_path=args.navdp_weights, **navdp_kwargs)
    else:
        navdp = NavDPStandalone.load(**navdp_kwargs)

    if args.pi_ip:
        from rover_io import RoverMotionAdapter
        print(f"[run] LIVE ROVER at {args.pi_ip} -- real cmd_vel will be sent.")
        motion = RoverMotionAdapter(pi_ip=args.pi_ip, max_linear=args.max_linear,
                                     max_angular=args.max_angular)
        print("[run] waiting for first camera frame over Zenoh...")
        t0 = time.time()
        while motion.get_rgbd()[0] is None:
            if time.time() - t0 > 15.0:
                motion.close()
                raise SystemExit("no camera frame received within 15s -- check the rover's "
                                  "camera publisher and --pi-ip before retrying")
            time.sleep(0.2)
        print("[run] first frame received.")
    else:
        motion = SimulatedMotion()

    planner = StagewisePlanner(vlm)
    executor = StageExecutor(navdp, vlm, motion, timeout_s=args.timeout_s)

    try:
        history = run_stagewise_loop(
            planner, executor, args.instruction, lambda: motion.get_rgbd()[0],
            max_stages=args.max_stages,
        )
        print(f"[stagewise] finished with {len(history)} stage(s) executed")
    finally:
        if args.pi_ip:
            motion.set_velocity(0.0, 0.0)
            motion.close()


if __name__ == "__main__":
    main()
