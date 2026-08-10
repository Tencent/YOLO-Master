"""Multi-Task Vision Modules.

TaskRouter: routes feature tokens to task-specific expert branches,
inspired by MoT token routing and MOT two-stage data association.
MultiTaskHead: unified head supporting detection, segmentation, pose,
classification, and depth estimation simultaneously.
"""

from .router import TaskRouter
from .head import MultiTaskHead

__all__ = ("TaskRouter", "MultiTaskHead")
