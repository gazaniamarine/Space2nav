"""Data types shared between the stagewise planner and executor."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

STAGE_KINDS = ("move", "turn", "object", "instruction", "done", "abort")
TERMINAL_KINDS = ("done", "abort")

# Defense in depth against a real, logged failure mode: LemNAV's own
# decompose() emitted {"kind":"object","phrase":"go straight"} -- a bare
# direction/motion word, not a real noun to search for -- and LemNAV's own
# regex safety net (_fix_directional_object_subtasks) only ever covered
# left/right/around, not straight/forward/ahead/back, so that one slipped
# through there too. STAGE_PLANNER_PROMPT already tells the model never to
# do this, but this is the code-level backstop for when it does anyway.
_BARE_TURN_WORDS = {"left": 90.0, "right": -90.0, "around": 180.0}
_BARE_NO_SAFE_DEFAULT = {"straight", "forward", "ahead", "back", "backward"}
_BARE_DIRECTION_RE = re.compile(
    r"^(?:go\s+|move\s+|turn\s+|head\s+)?(left|right|around|straight|forward|ahead|back|backward)$",
    re.IGNORECASE)


@dataclass
class Stage:
    """One planner-emitted stage -- the SpaceVLN sigma_k, minus the anchor
    chain/landmark-memory binding (no persistent spatial memory yet)."""

    kind: str
    meters: Optional[float] = None
    degrees: Optional[float] = None
    phrase: Optional[str] = None
    reason: str = ""

    def __post_init__(self):
        if self.kind not in STAGE_KINDS:
            raise ValueError(f"unknown stage kind: {self.kind!r}")

    @property
    def is_terminal(self) -> bool:
        return self.kind in TERMINAL_KINDS

    def describe(self) -> str:
        if self.kind == "move":
            return f"move {(self.meters or 0.0):+.2f}m"
        if self.kind == "turn":
            return f"turn {(self.degrees or 0.0):+.1f}deg"
        if self.kind in ("object", "instruction"):
            return f"{self.kind}: {self.phrase!r}"
        if self.kind == "abort":
            return f"abort ({self.reason})"
        return self.kind

    @classmethod
    def from_goal_value(cls, value: dict) -> "Stage":
        kind = str(value.get("kind", "")).strip().lower()
        phrase = str(value["phrase"]).strip() if value.get("phrase") else None

        if kind in ("object", "instruction") and phrase:
            m = _BARE_DIRECTION_RE.match(phrase.strip().lower())
            if m:
                word = m.group(1).lower()
                if word in _BARE_TURN_WORDS:
                    return cls(kind="turn", degrees=_BARE_TURN_WORDS[word],
                                reason=f"planner emitted {kind}:{phrase!r}, a bare direction "
                                       f"word not a real object -- corrected to turn")
                # "straight"/"forward"/"back"/"ahead" have no safe default
                # distance to guess -- fail closed to abort instead of
                # inventing one, so the next planning call gets a fresh
                # frame + this explicit reason rather than silently
                # searching for an object literally named "go straight".
                raise ValueError(
                    f"{kind} phrase {phrase!r} is a bare direction/motion word, not a "
                    f"real object or landmark -- cannot search for it")

        return cls(
            kind=kind,
            meters=_as_float(value.get("meters")),
            degrees=_as_float(value.get("degrees")),
            phrase=phrase,
            reason=str(value.get("reason", "")),
        )


def _as_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@dataclass
class StageFeedback:
    """F_k: what actually happened while a stage ran. Fed back into the
    planner's NEXT propose_next_stage() call alongside a fresh frame --
    this is the only thing that closes the loop; nothing here is persisted
    into any cross-stage spatial memory (out of scope for this phase)."""

    stage: Stage
    outcome: str            # e.g. "reached", "timeout", "lost", "ok" (move/turn)
    reason: str = ""
    ticks: int = 0
    elapsed_s: float = 0.0
    last_executor_status: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def describe(self) -> str:
        bits = [self.stage.describe(), f"-> {self.outcome}"]
        if self.reason:
            bits.append(f"({self.reason})")
        return " ".join(bits)
