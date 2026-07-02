# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Dynamic scheduling helpers for Mixture-of-Experts training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch


def usage_gini(usage: Iterable[float] | torch.Tensor) -> float:
    """Return the Gini coefficient for a non-negative expert-usage vector."""
    if isinstance(usage, torch.Tensor):
        values = usage.detach().float().reshape(-1).cpu()
    else:
        values = torch.tensor([float(v) for v in usage], dtype=torch.float32)

    if values.numel() == 0:
        return 0.0
    values = values.clamp_min(0)
    total = float(values.sum())
    if total <= 0:
        return 0.0

    diff_sum = torch.abs(values[:, None] - values[None, :]).sum()
    return float(diff_sum / (2 * values.numel() * total))


def mean_usage_gini_from_model(model: torch.nn.Module) -> float:
    """Average Gini over MoE modules that expose a latest routing snapshot."""
    ginis: list[float] = []
    for module in model.modules():
        snapshot = getattr(module, "last_routing_snapshot", None)
        if not snapshot:
            continue
        usage = snapshot.get("expert_usage")
        if usage is None:
            continue
        ginis.append(usage_gini(usage))
    return float(sum(ginis) / len(ginis)) if ginis else 0.0


@dataclass
class GiniBalanceScheduler:
    """Adjust MoE balance-loss coefficient from expert-usage imbalance."""

    base: float = 1.0
    target: float = 0.25
    alpha: float = 1.0
    beta: float = 0.8
    min_coeff: float = 0.5
    max_coeff: float = 2.0
    ema: float | None = None

    def update(self, gini: float) -> float:
        """Update EMA and return the clipped coefficient."""
        gini = float(max(gini, 0.0))
        self.ema = gini if self.ema is None else self.beta * self.ema + (1.0 - self.beta) * gini
        coeff = self.base * math.exp(self.alpha * (self.ema - self.target))
        return float(min(max(coeff, self.min_coeff), self.max_coeff))


def apply_balance_loss_coeff(model: torch.nn.Module, coeff: float) -> int:
    """Apply balance-loss coefficient to all compatible MoE modules."""
    updated = 0
    for module in model.modules():
        if hasattr(module, "balance_loss_coeff"):
            module.balance_loss_coeff = float(coeff)
            updated += 1
        moe_loss_fn = getattr(module, "moe_loss_fn", None)
        if moe_loss_fn is not None and hasattr(moe_loss_fn, "balance_loss_coeff"):
            moe_loss_fn.balance_loss_coeff = float(coeff)
    return updated

