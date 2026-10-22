"""Discrete per-tick action trace, mirroring LemNAV's
qwen_navdp_trace.py: one JSON line per NavDP tick, pairing whatever Qwen
action was most recently available (grounding result or status check) with
the NavDP action actually taken that tick -- so it's auditable afterward
exactly which goal Qwen handed NavDP and what NavDP did with it, same as
validate_against_logs.py already consumes for LemNAV's own logs.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional


class TraceLogger:
    def __init__(self, log_dir: str = "logs/qwen_navdp_trace", label: str = "run"):
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_label = "".join(c if c.isalnum() else "_" for c in label)[:40]
        self.path = Path(log_dir) / f"{safe_label}_{ts}.jsonl"
        self._f = open(self.path, "a")
        self._tick = 0

    def log_tick(self, stage_label: str, stage_kind: str,
                 qwen_action: Optional[dict], navdp_action: dict) -> None:
        self._tick += 1
        rec = {
            "t": time.time(),
            "tick": self._tick,
            "stage_label": stage_label,
            "stage_kind": stage_kind,
            "qwen_action": qwen_action,
            "navdp_action": navdp_action,
        }
        self._f.write(json.dumps(rec) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()
