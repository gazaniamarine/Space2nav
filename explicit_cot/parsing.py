"""Parsing helpers for explicit-CoT-tagged VLM replies:
<reasoning>...</reasoning><goal>...</goal>.

Kept separate from any brace-scanning JSON extractor: reasoning prose can
itself contain '{' / '[' characters, so a reply is split on the explicit
tags first and only the <goal> contents are handed to a JSON parser.

Vendored (unchanged logic) from
/home/gpu/Desktop/Nihil/LemNAV/cot_parsing.py.
"""
import json
import re
from typing import Optional, Tuple, Union

_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.S)
_GOAL_RE = re.compile(r"<goal>(.*?)</goal>", re.S)


def split_reasoning(reply: str) -> Tuple[str, str]:
    """Split a CoT reply into (reasoning_text, goal_text).

    If the <goal> tag is missing or malformed, goal_text falls back to the
    full reply so a caller can still attempt a best-effort parse; reasoning
    is "" in that case to make the format failure visible to the caller.
    """
    r_match = _REASONING_RE.search(reply)
    g_match = _GOAL_RE.search(reply)
    reasoning = r_match.group(1).strip() if r_match else ""
    goal_text = g_match.group(1).strip() if g_match else reply
    return reasoning, goal_text


def log_reasoning(tag: str, reasoning: str) -> None:
    if reasoning:
        print(f"[cot-reason:{tag}] {reasoning}")
    else:
        print(f"[cot-reason:{tag}] (no <reasoning> block found in reply)")


def _iter_balanced(text: str, open_ch: str, close_ch: str):
    """Yield every top-level balanced substring starting at each occurrence
    of open_ch, left to right (not just the first)."""
    i = text.find(open_ch)
    while i != -1:
        depth = 0
        for j in range(i, len(text)):
            if text[j] == open_ch:
                depth += 1
            elif text[j] == close_ch:
                depth -= 1
                if depth == 0:
                    yield text[i:j + 1]
                    break
        i = text.find(open_ch, i + 1)


def _is_plausible_goal(value) -> bool:
    """Reject bare scalar arrays like "[1]" -- e.g. a stray "[1]"
    image-reference marker (our own prompts say "Image: [1] ...") that a
    reply without a <goal> tag can leave lying around before any real
    JSON. A real goal payload is always an object, or an array of
    objects (e.g. a subtask list) -- never an array of bare scalars."""
    if isinstance(value, dict):
        return True
    if isinstance(value, list):
        return not value or all(isinstance(v, dict) for v in value)
    return False


def extract_json_value(text: str) -> Optional[Union[dict, list]]:
    """Return the first PLAUSIBLE top-level JSON array or object in text
    (see _is_plausible_goal) -- scans past earlier bracket/brace matches
    that parse but aren't a real goal payload, rather than stopping at the
    first match.

    Prefers a leading '[' over a leading '{' (so a reply that is a
    top-level JSON array isn't truncated to its first element), otherwise
    falls back to a balanced-brace object scan.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()

    bracket_pos = stripped.find("[")
    brace_pos = stripped.find("{")
    if bracket_pos != -1 and (brace_pos == -1 or bracket_pos <= brace_pos):
        for arr in _iter_balanced(stripped, "[", "]"):
            try:
                value = json.loads(arr)
            except Exception:
                continue
            if _is_plausible_goal(value):
                return value

    for obj in _iter_balanced(stripped, "{", "}"):
        try:
            value = json.loads(obj)
        except Exception:
            continue
        if _is_plausible_goal(value):
            return value
    return None
