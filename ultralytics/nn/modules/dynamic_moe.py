"""Legacy dynamic MoE blocks retained for YAML and checkpoint compatibility."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv


class DynamicExpert(nn.Module):
    """Feature expert used by the historical dynamic MoE block."""

    def __init__(self, dim: int, expert_type: str = "spatial"):
        super().__init__()
        self.expert_type = expert_type
        if expert_type == "spatial":
            self.net = nn.Sequential(Conv(dim, dim, 7, 1, 3, g=dim), Conv(dim, dim, 1))
        elif expert_type == "channel":
            self.net = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Conv(dim, dim // 4, 1),
                Conv(dim // 4, dim, 1, act=False),
                nn.Sigmoid(),
            )
        elif expert_type == "detail":
            self.net = nn.Sequential(Conv(dim, dim, 3, 1, 1), Conv(dim, dim, 3, 1, 1))
        else:
            self.net = nn.Sequential(DWConv(dim, dim, 5, 1), Conv(dim, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the selected feature transform."""
        return x * self.net(x) if self.expert_type == "channel" else self.net(x)


class MoEGate(nn.Module):
    """Top-k router used by the historical dynamic MoE block."""

    def __init__(self, dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(dim, num_experts))
        self.balance_loss_weight = 0.01
        self.last_balance_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized top-k routing weights and expert indices."""
        probabilities = F.softmax(self.gate(x), dim=1)
        weights, indices = torch.topk(probabilities, self.top_k, dim=1)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        if self.training:
            from .moe.loss import differentiable_balance_loss

            usage = F.one_hot(indices.reshape(-1), self.num_experts).float().sum(0)
            self.last_balance_loss = differentiable_balance_loss(
                probabilities, usage, self.num_experts, reduce_ddp=True
            )
        else:
            self.last_balance_loss = None
        return weights, indices


class DyMoEBlock(nn.Module):
    """Dynamic routed residual block preserved under its historical class name."""

    def __init__(self, dim: int, num_experts: int = 4, top_k: int = 2, mlp_ratio: float = 2.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        expert_types = ("spatial", "channel", "detail", "texture")
        self.experts = nn.ModuleList(
            DynamicExpert(dim, expert_types[index % len(expert_types)]) for index in range(num_experts)
        )
        self.gate = MoEGate(dim, num_experts, top_k)
        self.mlp = nn.Sequential(Conv(dim, int(dim * mlp_ratio), 1), Conv(int(dim * mlp_ratio), dim, 1, act=False))
        self.gamma1 = nn.Parameter(1e-4 * torch.ones(dim))
        self.gamma2 = nn.Parameter(1e-4 * torch.ones(dim))
        self.last_aux_loss: torch.Tensor | None = None

    @property
    def aux_loss(self) -> torch.Tensor:
        """Return the canonical routed auxiliary loss for this block."""
        from .moe.modules import _registry_get, _zero_aux_loss_like

        loss = _registry_get(self)
        return loss if isinstance(loss, torch.Tensor) else _zero_aux_loss_like(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route inputs through experts and apply the two residual branches."""
        from .moe.modules import _registry_set
        from .moe.utils import BatchedExpertComputation

        weights, indices = self.gate(x)
        routed = BatchedExpertComputation.compute_sparse_experts_batched(
            x, self.experts, weights, indices, self.top_k, self.num_experts
        )
        if self.training and self.gate.last_balance_loss is not None:
            aux = self.gate.balance_loss_weight * self.gate.last_balance_loss
            _registry_set(self, aux)
            self.last_aux_loss = aux
        else:
            self.last_aux_loss = None
        x = x + self.gamma1.view(1, -1, 1, 1) * routed
        return x + self.gamma2.view(1, -1, 1, 1) * self.mlp(x)

    def export_capabilities(self) -> dict:
        """Declare the trace-time dense fallback used by the shared dispatcher."""
        from .routing_protocol import export_capabilities

        return export_capabilities(self)


class DyC2f(nn.Module):
    """C2f-style wrapper around one or more legacy dynamic MoE blocks."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        num_experts: int = 4,
        top_k: int = 2,
        e: float = 0.5,
    ):
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv((1 + n) * hidden, c2, 1)
        self.m = nn.ModuleList(DyMoEBlock(hidden, num_experts, top_k) for _ in range(n))
        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dynamic MoE blocks and the historical residual scaling."""
        features = [self.cv1(x)]
        features.extend(block(features[-1]) for block in self.m)
        output = self.cv2(torch.cat(features, 1))
        return x + self.gamma.view(1, -1, 1, 1) * output

    def export_capabilities(self) -> dict:
        """Declare routed export support for nested dynamic blocks."""
        from .routing_protocol import export_capabilities

        return export_capabilities(self)


# Historical checkpoints serialized these classes from block.py.
for _class in (DynamicExpert, MoEGate, DyMoEBlock, DyC2f):
    _class.__module__ = "ultralytics.nn.modules.block"


__all__ = ("DynamicExpert", "MoEGate", "DyMoEBlock", "DyC2f")
