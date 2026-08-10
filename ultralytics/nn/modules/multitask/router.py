"""Task-aware Token Router.

Inspired by MoT's expert routing (content-based token-to-expert assignment)
and ByteTracker's two-stage confidence-split association:

- Stage 1 (High-confidence): tokens with clear task affinity → dedicated expert
- Stage 2 (Ambiguous): shared tokens → soft blending across multiple experts
- Task Association: cross-task feature matching (like track-detection IoU matching)

Architecture:
    Input features [B, C, H, W]
        │
    ┌───▼───────────────────────────────────┐
    │  Spatial Token Router                 │
    │  ┌─────────────────────────────────┐  │
    │  │ Per-token → task affinity score │  │  ← MoT-style router
    │  │   [B, num_tasks, H, W]          │  │
    │  └─────────────────────────────────┘  │
    │  ┌─────────────────────────────────┐  │
    │  │ High-conf → TaskExpert          │  │  ← ByteTracker high-conf
    │  │ Low-conf → SharedExpert         │  │  ← ByteTracker low-conf
    │  └─────────────────────────────────┘  │
    └───────────────────────────────────────┘
        │
    ┌───▼───────────────────────────────────┐
    │  Cross-Task Feature Association       │
    │  (IoU-inspired task-feature matching)  │
    └───────────────────────────────────────┘
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.utils import get_safe_groups as _safe_groups
from ultralytics.nn.modules._numeric import FP32RouterMixin


# Task types
TASK_DETECT = 0
TASK_SEGMENT = 1
TASK_POSE = 2
TASK_CLASSIFY = 3
TASK_DEPTH = 4
TASK_NORMAL = 5
TASK_SEMANTIC = 6

TASK_NAMES = {
    TASK_DETECT: "detect",
    TASK_SEGMENT: "segment",
    TASK_POSE: "pose",
    TASK_CLASSIFY: "classify",
    TASK_DEPTH: "depth",
    TASK_NORMAL: "normal",
    TASK_SEMANTIC: "semantic",
}


class TaskRouter(FP32RouterMixin, nn.Module):
    """Task-aware token router: assigns each spatial token to task experts.

    Design philosophy (MOT-inspired two-stage routing):
    1. Compute per-token task affinity scores (like MoT router)
    2. Split tokens by confidence:
       - High-confidence tokens → dedicated task expert (like ByteTracker stage 1)
       - Ambiguous tokens → shared cross-task blending (like ByteTracker stage 2)
    3. Cross-task feature association for complementary information sharing

    Args:
        dim: Input channel dimension.
        num_tasks: Number of task slots (default 7: detect/segment/pose/classify/depth/normal/semantic).
        top_k: Active task experts per token (default 2: primary + secondary).
        temperature: Router softmax temperature.
        balance_loss_coeff: Load-balancing loss weight.
        shared_expert_ratio: Channels allocated to shared expert vs task-specific.
        use_spatial: Token-level vs image-level routing.
    """

    def __init__(
        self,
        dim: int,
        num_tasks: int = 7,
        top_k: int = 2,
        temperature: float = 1.0,
        balance_loss_coeff: float = 0.01,
        shared_expert_ratio: float = 0.25,
        use_spatial: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_tasks = num_tasks
        self._top_k = min(top_k, num_tasks)
        self.balance_loss_coeff = balance_loss_coeff
        self.shared_expert_ratio = shared_expert_ratio

        # Channel split: shared vs task-specific
        self.shared_channels = int(dim * shared_expert_ratio)
        self.task_channels = dim - self.shared_channels

        # Spatial token → task affinity router (MoT-style)
        hidden = max(dim // 8, num_tasks * 4)
        if use_spatial:
            self.affinity_router = nn.Sequential(
                nn.Conv2d(dim, hidden, 1, bias=False),
                nn.GroupNorm(_safe_groups(hidden, 4), hidden),
                nn.SiLU(inplace=False),
                nn.Conv2d(hidden, num_tasks, 1, bias=True),
            )
        else:
            self.affinity_router = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(dim, hidden, bias=False),
                nn.SiLU(inplace=False),
                nn.Linear(hidden, num_tasks, bias=True),
            )
        self.use_spatial = use_spatial

        # Temperature as buffer for annealing
        self.register_buffer("temperature", torch.tensor(max(temperature, 0.1)), persistent=True)
        nn.init.zeros_(self.affinity_router[-1].weight)
        nn.init.zeros_(self.affinity_router[-1].bias)

        # Cross-task feature projector (MOT association inspired)
        self.cross_task_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=_safe_groups(dim, 8), bias=False),
            nn.GroupNorm(_safe_groups(dim, 8), dim),
            nn.SiLU(inplace=False),
            nn.Conv2d(dim, dim, 1, bias=False),
        )

        self.last_affinity: Optional[torch.Tensor] = None
        self.last_assignment: Optional[torch.Tensor] = None
        self.last_routing_stats: dict = {}

    @property
    def top_k(self) -> int:
        return self._top_k

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Route spatial tokens to task experts.

        Args:
            x: [B, C, H, W] feature map

        Returns:
            task_features: [B, num_tasks, C, H, W] task-specific features
            shared_features: [B, shared_C, H, W] shared cross-task features
            routing_stats: diagnostic dict
        """
        B, C, H, W = x.shape

        # Stage 1: Compute task affinity (MoT-style token routing)
        affinity_logits = self.affinity_router(x.float())  # [B, T, H, W]
        router_probs = F.softmax(affinity_logits / self.temperature.float(), dim=1)  # [B, T, H, W]

        # Top-K hard selection + soft blending
        if self._top_k < self.num_tasks:
            topk_vals, topk_idx = router_probs.topk(self._top_k, dim=1)
            topk_weights = topk_vals / topk_vals.sum(dim=1, keepdim=True).clamp_min(1e-8)
            sparse_affinity = torch.zeros_like(router_probs)
            sparse_affinity.scatter_(1, topk_idx, topk_weights)
            assignment = torch.zeros_like(router_probs).scatter_(1, topk_idx, 1.0 / self._top_k)
            if self.training:
                affinity = sparse_affinity * 0.98 + router_probs * 0.02  # exploration
            else:
                affinity = sparse_affinity
        else:
            topk_idx = torch.arange(self.num_tasks, device=x.device).view(1, -1, 1, 1).expand(B, -1, H, W)
            assignment = torch.full_like(router_probs, 1.0 / self.num_tasks)
            affinity = router_probs

        # Keep soft probabilities graph-connected until the criterion consumes the auxiliary loss.
        self.last_affinity = router_probs
        self.last_assignment = assignment.detach()

        # Stage 2: Generate task-specific features via weighted fusion
        # Each token is softly routed to its top-K task experts
        task_features = x.unsqueeze(1) * affinity.unsqueeze(2)  # [B, T, C, H, W]

        # Stage 3: Cross-task feature association (MOT association inspired)
        # Compute task-feature similarity and share complementary information
        shared_features = self.cross_task_proj(x)  # [B, C, H, W]
        if self.shared_channels > 0:
            shared_features = shared_features[:, : self.shared_channels]  # [B, shared_C, H, W]

        # Routing diagnostics
        with torch.no_grad():
            mean_usage = affinity.float().mean(dim=(0, 2, 3))
            self.last_routing_stats = {
                "task_usage": mean_usage.cpu(),
                "top_k": self._top_k,
                "max_task": int(mean_usage.argmax()),
                "entropy": float(-(affinity.float() * (affinity.float() + 1e-8).log()).sum(dim=1).mean()),
            }

        return task_features, shared_features, self.last_routing_stats

    def compute_balance_loss(self) -> torch.Tensor:
        """Return the Switch/GShard auxiliary loss for balanced task-slot usage."""
        if self.last_affinity is None or self.last_assignment is None:
            return torch.zeros((), device=self.temperature.device)
        importance = self.last_affinity.float().mean(dim=(0, 2, 3))
        usage = self.last_assignment.float().mean(dim=(0, 2, 3))
        return self.num_tasks * torch.sum(importance * usage)

    @staticmethod
    def available_tasks() -> dict[int, str]:
        return TASK_NAMES.copy()

    @staticmethod
    def task_mask_from_names(task_names: list[str]) -> torch.Tensor:
        """Create a boolean mask for active tasks by name."""
        name_to_id = {v: k for k, v in TASK_NAMES.items()}
        ids = [name_to_id[n] for n in task_names if n in name_to_id]
        mask = torch.zeros(len(TASK_NAMES), dtype=torch.bool)
        mask[ids] = True
        return mask
