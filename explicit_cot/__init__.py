"""Explicit chain-of-thought VLM layer: free-form <reasoning>...</reasoning>
prose followed by a <goal>...</goal> tagged JSON payload. Deliberately NOT
SpaceVLN's Spatial-CoT (no fixed 5-field/4-field schema, no spatial memory
grounding here) -- just the explicit-reasoning-then-structured-output
pattern already validated in the LemNAV project's cot_prompts.py /
cot_parsing.py, cleaned up and generalized for the stagewise planner.
"""

from .parsing import extract_json_value, split_reasoning
from .vlm import ExplicitCoTVLM

__all__ = ["ExplicitCoTVLM", "split_reasoning", "extract_json_value"]
