"""MoLoRA routers: CNN-Native image-level routing for YOLO-Master.

Unlike NLP per-token routers, YOLO-Master MoLoRA uses whole-image routing
(linear by default), with optional spatial and hybrid variants.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRouter(nn.Module):
    """Global Average Pool -> Linear router.  One decision per image, O(C)."""

    def __init__(
        self,
        in_channels: int,
        num_experts: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_experts = num_experts
        hidden = hidden_dim or max(in_channels // 4, 1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_experts),
        )
        # Initialize router logits to small values for near-uniform start
        nn.init.normal_(self.fc[-1].weight, std=0.01)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] for Conv2d, or [B, C] for Linear
        Returns:
            logits: [B, num_experts]
        """
        if x.dim() == 2:
            return self.fc(x)  # [B, E]
        pooled = x.mean(dim=[2, 3])  # [B, C]
        return self.fc(pooled)  # [B, E]


class SpatialRouter(nn.Module):
    """1x1 Conv -> Spatial AvgPool router.  Fine-grained spatial awareness, O(C·H·W)."""

    def __init__(
        self,
        in_channels: int,
        num_experts: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_experts = num_experts
        hidden = hidden_dim or max(in_channels // 4, 1)
        self.conv = nn.Conv2d(in_channels, hidden, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(hidden, num_experts, kernel_size=1, bias=True)
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] for Conv2d, or [B, C] for Linear
        Returns:
            logits: [B, num_experts]
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        feat = self.act(self.conv(x))  # [B, hidden, H, W]
        logits = self.proj(feat)  # [B, E, H, W]
        logits = logits.mean(dim=[2, 3])  # [B, E]
        return logits


class HybridRouter(nn.Module):
    """Global + Local fusion with learnable gate α."""

    def __init__(
        self,
        in_channels: int,
        num_experts: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.linear_router = LinearRouter(in_channels, num_experts, hidden_dim)
        self.spatial_router = SpatialRouter(in_channels, num_experts, hidden_dim)
        # P2 fix: init alpha to 0.0 so sigmoid(0.0)=0.5 gives a truly uniform
        # blend at start. The previous 0.5 init produced sigmoid(0.5)≈0.622,
        # biasing the linear router before any learning happens.
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            logits: [B, num_experts]
        """
        logits_linear = self.linear_router(x)  # [B, E]
        logits_spatial = self.spatial_router(x)  # [B, E]
        alpha = torch.sigmoid(self.alpha)
        return alpha * logits_linear + (1 - alpha) * logits_spatial


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_router(
    router_type: str,
    in_channels: int,
    num_experts: int,
    hidden_dim: Optional[int] = None,
) -> nn.Module:
    """Factory for CNN-Native MoLoRA routers.

    Args:
        router_type: "linear" | "spatial" | "hybrid"
        in_channels: input feature channels
        num_experts: number of experts
        hidden_dim: hidden dim for router MLP; auto if None
    """
    if router_type == "linear":
        return LinearRouter(in_channels, num_experts, hidden_dim)
    elif router_type == "spatial":
        return SpatialRouter(in_channels, num_experts, hidden_dim)
    elif router_type == "hybrid":
        return HybridRouter(in_channels, num_experts, hidden_dim)
    else:
        raise ValueError(f"Unknown router_type: {router_type}")
