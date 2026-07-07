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


@dataclass
class MapSaturationSchedulerConfig:
    """Configuration for validation-mAP saturation driven coefficient annealing.

    When validation mAP improvement stalls over a rolling window, the balance
    coefficient is decayed by `decay_factor`. This relaxes routing pressure
    at late training stages so experts can specialize without interference.

    Formula (applied once per epoch):
        If max(mAP[-window:]) - max(mAP[-2*window:-window]) < saturation_threshold:
            saturation_scale *= decay_factor   (floored at min_scale)

    The effective balance coefficient is then:
        effective_coeff = base_coeff * saturation_scale
    """

    enabled: bool = True
    window_size: int = 5
    saturation_threshold: float = 0.001
    decay_factor: float = 0.8
    min_scale: float = 0.1


@dataclass
class MapSaturationScheduleState:
    """Serializable state emitted after each epoch update."""

    val_map: float
    saturation_scale: float
    plateau_detected: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MapSaturationScheduler:
    """Validation-mAP saturation driven annealing of MoE balance-loss coefficients.

    Unlike `MoEDynamicScheduler` which reacts to routing imbalance per training
    step, this scheduler operates at epoch granularity: it monitors whether
    validation mAP is still improving, and relaxes the balance pressure when
    learning plateaus. Call `update(val_map)` once per epoch, then `apply(base_coeff)`
    at each training step to obtain the effective coefficient.
    """

    def __init__(self, config: MapSaturationSchedulerConfig | None = None):
        self.config = config or MapSaturationSchedulerConfig()
        self.map_history: list[float] = []
        self.saturation_scale: float = 1.0
        self.last_state: MapSaturationScheduleState | None = None

    def update(self, val_map: float) -> MapSaturationScheduleState:
        """Record a new validation mAP and update the saturation scale."""
        self.map_history.append(float(val_map))
        plateau = (
            self.config.enabled
            and len(self.map_history) >= 2 * (w := self.config.window_size)
            and max(self.map_history[-w:]) - max(self.map_history[-2 * w : -w])
            < self.config.saturation_threshold
        )
        if plateau:
            self.saturation_scale = max(
                self.saturation_scale * self.config.decay_factor,
                self.config.min_scale,
            )
        self.last_state = MapSaturationScheduleState(
            val_map=float(val_map),
            saturation_scale=self.saturation_scale,
            plateau_detected=plateau,
        )
        return self.last_state

    def apply(self, base_coeff: float) -> float:
        """Return the balance coefficient scaled by the current saturation decay."""
        return float(base_coeff) * (self.saturation_scale if self.config.enabled else 1.0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "map_history": list(self.map_history),
            "saturation_scale": self.saturation_scale,
            "last_state": self.last_state.to_dict() if self.last_state else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        match state.get("config"):
            case dict() as cfg:
                self.config = MapSaturationSchedulerConfig(**cfg)
        self.map_history = list(state.get("map_history", []))
        self.saturation_scale = float(s) if (s := state.get("saturation_scale")) is not None else 1.0
        match state.get("last_state"):
            case dict() as last:
                self.last_state = MapSaturationScheduleState(**last)
            case _:
                self.last_state = None
