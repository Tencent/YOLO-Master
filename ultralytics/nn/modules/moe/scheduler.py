"""Dynamic hyperparameter scheduling utilities for MoE training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass
class MoEDynamicSchedulerConfig:
    """Configuration for Gini-driven MoE auxiliary-loss scheduling."""

    enabled: bool = True
    target_gini: float = 0.25
    gain: float = 1.5
    min_balance_coeff: float = 0.02
    max_balance_coeff: float = 2.0
    ema_momentum: float = 0.9


@dataclass
class MoEDynamicScheduleState:
    """Serializable state emitted at each scheduler step."""

    gini: float
    ema_gini: float
    balance_loss_coeff: float
    base_balance_loss_coeff: float
    target_gini: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_gini(expert_usage: torch.Tensor) -> float:
    """Return the Gini coefficient of a non-negative expert-usage vector."""
    usage = expert_usage.detach().float().reshape(-1).clamp_min(0.0)
    if usage.numel() == 0:
        return 0.0
    total = usage.sum()
    if float(total) <= 0.0:
        return 0.0

    sorted_usage = torch.sort(usage / total).values
    n = sorted_usage.numel()
    index = torch.arange(1, n + 1, device=sorted_usage.device, dtype=sorted_usage.dtype)
    gini = (2 * torch.sum(index * sorted_usage) / n) - ((n + 1) / n)
    return float(gini.clamp(0.0, 1.0).cpu())


class MoEDynamicScheduler:
    """Gini-driven scheduler for MoE balance-loss coefficients.

    Formula:
        coeff_t = clamp(base_coeff * (1 + gain * (ema_gini_t - target_gini)),
                        min_balance_coeff, max_balance_coeff)

    A high Gini means expert routing is imbalanced, so the balance coefficient
    increases. A low Gini means routing is already healthy, so the coefficient
    relaxes and lets experts specialize.
    """

    def __init__(self, config: MoEDynamicSchedulerConfig | None = None):
        self.config = config or MoEDynamicSchedulerConfig()
        self.ema_gini: float | None = None
        self.last_state: MoEDynamicScheduleState | None = None

    def step(self, expert_usage: torch.Tensor, base_balance_coeff: float) -> MoEDynamicScheduleState:
        gini = compute_gini(expert_usage)
        if self.ema_gini is None:
            self.ema_gini = gini
        else:
            m = min(max(float(self.config.ema_momentum), 0.0), 0.999)
            self.ema_gini = m * self.ema_gini + (1.0 - m) * gini

        if not self.config.enabled:
            coeff = float(base_balance_coeff)
        else:
            multiplier = 1.0 + float(self.config.gain) * (self.ema_gini - float(self.config.target_gini))
            coeff = float(base_balance_coeff) * max(multiplier, 0.0)
            coeff = min(max(coeff, float(self.config.min_balance_coeff)), float(self.config.max_balance_coeff))

        self.last_state = MoEDynamicScheduleState(
            gini=gini,
            ema_gini=float(self.ema_gini),
            balance_loss_coeff=coeff,
            base_balance_loss_coeff=float(base_balance_coeff),
            target_gini=float(self.config.target_gini),
        )
        return self.last_state

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "ema_gini": self.ema_gini,
            "last_state": self.last_state.to_dict() if self.last_state else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        config = state.get("config")
        if isinstance(config, dict):
            self.config = MoEDynamicSchedulerConfig(**config)
        ema = state.get("ema_gini")
        self.ema_gini = float(ema) if ema is not None else None
        last = state.get("last_state")
        self.last_state = MoEDynamicScheduleState(**last) if isinstance(last, dict) else None
