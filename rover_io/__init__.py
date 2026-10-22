"""Real-rover hardware I/O for Space2nav, speaking the same Zenoh/CDR
contract as LemNAV's nav_pipeline/zenoh_node.py. See zenoh_motion.py's
docstring for what's vendored (wire-format parsing) vs. written fresh
(odometry, motion control) and why. bev.py is a live, geometric-only BEV
occupancy grid -- no semantic/landmark memory (left for later).
"""
from .bev import BEVOccupancy
from .odometry import Pose2D, SimpleOdometry
from .zenoh_motion import RoverMotionAdapter

__all__ = ["RoverMotionAdapter", "SimpleOdometry", "Pose2D", "BEVOccupancy"]
