"""Phase 6: validate the stagewise planner against real logged episodes from
LemNAV's runs (logs/qwen_navdp_trace/<leg_label>_<ts>.jsonl + tick_NNNNN.jpg
frames), instead of synthetic test images.

This does NOT re-run NavDP or claim to reproduce what the robot would
physically have done -- there's no live hardware here. It replays each
logged leg's OUTCOME (what actually happened, tick by tick, including how
it ended) as a StageFeedback and asks the NEW stagewise planner, given the
leg's LAST real camera frame, what it would decide next. The point is
checking whether frame-grounded stage-boundary re-verification (Phase 3)
would have caught failure modes the old per-tick-only system didn't notice
until a hardcoded timeout/stall limit fired.

Usage:
    python validate_against_logs.py <path-to-LemNAV>/logs/qwen_navdp_trace/0_object_20261021_162842.jsonl \\
        --model-id Qwen/Qwen3-VL-2B-Instruct
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from explicit_cot.vlm import DEFAULT_MODEL_ID, ExplicitCoTVLM
from stagewise.planner import StagewisePlanner
from stagewise.types import Stage, StageFeedback


def _load_episode(jsonl_path: Path):
    ticks = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    return ticks


def _instruction_text(ticks: list) -> str:
    for t in ticks:
        qa = t.get("qwen_action") or {}
        text = qa.get("instruction_text") or qa.get("target_phrase")
        if text:
            return text
    return "(no instruction text found in log)"


def _tick_found(t: dict) -> bool:
    qa, na = t.get("qwen_action") or {}, t.get("navdp_action") or {}
    status = qa.get("supervisor_status") or na.get("state")
    return bool(status and status not in ("searching", "SEARCH"))


def _tick_moved(t: dict) -> bool:
    qa, na = t.get("qwen_action") or {}, t.get("navdp_action") or {}
    return bool((na.get("linear") or 0) > 0.01 or na.get("goal_point") or qa.get("pixel_goal_uv"))


def _summarize_outcome(ticks: list, tail_n: int = 5) -> StageFeedback:
    """Reconstruct an approximate StageFeedback from what the OLD system's
    per-tick log actually shows -- there is no explicit 'outcome' field in
    this log format (that lived in object_leg.py's in-memory state, not
    written per-tick). Judges by the TAIL of the episode (what it ended up
    doing), not an OR across every tick -- a transient false-positive
    detection mid-episode that was lost again by the end should still read
    as "never actually resolved", not "progressed"."""
    kind = ticks[0].get("leg_kind", "object")
    phrase = _instruction_text(ticks)
    stage = Stage(kind=kind, phrase=phrase)

    tail = ticks[-tail_n:]
    tail_found = any(_tick_found(t) for t in tail)
    tail_moved = any(_tick_moved(t) for t in tail)
    ever_found = any(_tick_found(t) for t in ticks)
    last_qa = ticks[-1].get("qwen_action") or {}
    last_na = ticks[-1].get("navdp_action") or {}
    last_status = last_qa.get("supervisor_status") or last_na.get("state")

    n = len(ticks)
    if tail_found:
        outcome, reason = "progressed", f"actively locating/moving toward target in the final {tail_n} of {n} ticks"
    elif ever_found:
        outcome, reason = ("lost_after_find",
                            f"found/approached the target earlier but reverted to search by the "
                            f"end of {n} ticks -- NOT resolved")
    elif tail_moved:
        outcome, reason = "no_progress", f"attempted motion but never located target across {n} ticks"
    else:
        outcome, reason = "stalled", f"stuck in search/no-motion state for all {n} ticks, zero progress"
    return StageFeedback(stage=stage, outcome=outcome, reason=reason, ticks=n,
                          last_executor_status=last_status)


def validate_episode(planner: StagewisePlanner, jsonl_path: Path):
    ticks = _load_episode(jsonl_path)
    feedback = _summarize_outcome(ticks)
    last_frame_path = jsonl_path.parent / Path(ticks[-1]["frame_path"]).name
    if not last_frame_path.exists():
        # frame_path in the log is relative to the LemNAV repo root, not
        # this script's cwd -- try the episode subdirectory next to the
        # jsonl instead (same layout, just resolved locally).
        last_frame_path = jsonl_path.with_suffix("") / Path(ticks[-1]["frame_path"]).name
    frame = np.asarray(Image.open(last_frame_path).convert("RGB"))

    print(f"\n=== {jsonl_path.name} ===")
    print(f"  old system:    {feedback.describe()}  (leg_kind={feedback.stage.kind})")

    stage = planner.propose_next_stage(
        instruction=feedback.stage.phrase, history=[feedback], frame=frame)
    print(f"  new planner:   {stage.describe()}")
    return feedback, stage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl_paths", nargs="+", type=Path)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    vlm = ExplicitCoTVLM(model_id=args.model_id, device=args.device)
    planner = StagewisePlanner(vlm)
    for jp in args.jsonl_paths:
        validate_episode(planner, jp)


if __name__ == "__main__":
    main()
