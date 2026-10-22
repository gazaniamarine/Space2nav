"""Vendored NavDP standalone policy (diffusion trajectory policy over RGB-D +
point/pixel/no goal conditioning). No language grounding, no persistent
memory, no VLM dependency -- this is the low-level motion executor that the
stagewise planner/explicit-CoT layer (see ../stagewise, ../explicit_cot)
drives with a point or pixel goal each tick.

Source: /home/gpu/Desktop/Nihil/LemNAV/nav_pipeline/navdp_{crossmodal,backbone}.py
and nav_pipeline/depth_anything/, adapted to keep the same
predict_pointgoal/predict_nogoal/trajectory_to_vw API as the DAT-backbone
variant this replaced. Switched variants because navdp-cross-modal.ckpt is
the only NavDP checkpoint that actually exists on disk (see nav_dp/model.py
docstring) -- the DAT variant's navdp_extracted.pth was never produced here.
"""

from .model import NavDPStandalone, trajectory_to_vw

__all__ = ["NavDPStandalone", "trajectory_to_vw"]
