"""Stagewise closed-loop navigation: the SpaceVLN "verifiable space-landmark
stage" idea, WITHOUT SpaceVLN's Spatial Cognitive Memory or Spatial-CoT.

The planner never commits to a full upfront plan (LemNAV's old decompose())
and never re-verifies from text-only outcome history alone (LemNAV's old
workflow_step()) -- instead ONE call, `StagewisePlanner.propose_next_stage`,
is made at every stage boundary with the CURRENT camera frame in view, and
it emits exactly one next stage or signals done/abort. The executor
(`StageExecutor`) then runs that single stage to completion using NavDP for
low-level motion and the explicit-CoT executor prompt for
searching/approaching/arrived/wrong_object judgement each tick -- unchanged
in spirit from LemNAV's object_leg.py loop, just without any of its
DINO/SAM2/fact3r/occupancy machinery (out of scope for this phase).

Detection/grounding for the executor's goal_point_fn is Qwen itself
(qwen_goal.QwenGoalPointFn, via qwen_grounding/), not Grounding-DINO --
reuses the same loaded Qwen3-VL weights as the planner/executor reasoning.
"""

from .executor import StageExecutor
from .planner import StagewisePlanner
from .qwen_goal import QwenGoalPointFn
from .trace_log import TraceLogger
from .types import Stage, StageFeedback

__all__ = ["Stage", "StageFeedback", "StagewisePlanner", "StageExecutor", "QwenGoalPointFn", "TraceLogger"]
