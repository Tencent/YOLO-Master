"""Additive YAML registry for YOLO-Master routed modules."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ultralytics.nn.modules.moa import C2fMoA
from ultralytics.nn.modules.block import DyC2f, DyMoEBlock
from ultralytics.nn.modules.moe import (
    A2C2fMoE,
    AdaptiveGateMoE,
    DetailAwareLowRankHybridAdaptiveGateMoE,
    DiversifiedExpertMoE,
    ES_MOE,
    FusedAdaptiveGateMoE,
    GatedFusionMoE,
    HybridAdaptiveGateMoE,
    HybridAdaptiveGateMoEv2,
    LowRankHybridAdaptiveGateMoE,
    ModularRouterExpertMoE,
    MultiHeadRouterMoE,
    OptimalHybridGateMoE,
    RefinedLowRankHybridAdaptiveGateMoE,
    UltimateOptimizedMoE,
    UltraOptimizedMoE,
    VisualEnhancedAdaptiveGateMoE,
)
from ultralytics.nn.modules.moe.config import annotate_mixture_yaml_config
from ultralytics.nn.modules.mot import C2fMoT
from ultralytics.utils.ops import make_divisible


MIXTURE_MODULES = {
    "A2C2fMoE": A2C2fMoE,
    "AdaptiveGateMoE": AdaptiveGateMoE,
    "C2fMoA": C2fMoA,
    "C2fMoT": C2fMoT,
    "DetailAwareLowRankHybridAdaptiveGateMoE": DetailAwareLowRankHybridAdaptiveGateMoE,
    "DiversifiedExpertMoE": DiversifiedExpertMoE,
    "DyC2f": DyC2f,
    "DyMoEBlock": DyMoEBlock,
    "ES_MOE": ES_MOE,
    "FusedAdaptiveGateMoE": FusedAdaptiveGateMoE,
    "GatedFusionMoE": GatedFusionMoE,
    "HybridAdaptiveGateMoE": HybridAdaptiveGateMoE,
    "HybridAdaptiveGateMoEv2": HybridAdaptiveGateMoEv2,
    "LowRankHybridAdaptiveGateMoE": LowRankHybridAdaptiveGateMoE,
    "ModularRouterExpertMoE": ModularRouterExpertMoE,
    "MultiHeadRouterMoE": MultiHeadRouterMoE,
    "OptimalHybridGateMoE": OptimalHybridGateMoE,
    "RefinedLowRankHybridAdaptiveGateMoE": RefinedLowRankHybridAdaptiveGateMoE,
    "UltimateOptimizedMoE": UltimateOptimizedMoE,
    "UltraOptimizedMoE": UltraOptimizedMoE,
    "VisualEnhancedAdaptiveGateMoE": VisualEnhancedAdaptiveGateMoE,
}
MIXTURE_BASE_MODULES = frozenset(MIXTURE_MODULES.values())
MIXTURE_REPEAT_MODULES = frozenset({A2C2fMoE, C2fMoA, C2fMoT, DyC2f})


def get_mixture_module(name: str):
    """Resolve a project YAML name without shadowing official modules."""
    try:
        return MIXTURE_MODULES[name]
    except KeyError as exc:
        raise KeyError(f"unknown model module {name!r}") from exc


def adapt_mixture_args(
    module,
    from_index: int,
    channels: list[int],
    args: list[Any],
    repeats: int,
    *,
    nc: int,
    width: float,
    max_channels: float,
) -> tuple[list[Any], int, int, bool]:
    """Apply upstream width/depth/channel rules to one registered module."""
    if not isinstance(from_index, int):
        raise TypeError(f"{module.__name__} expects one input index, got {from_index!r}")
    if not args:
        raise ValueError(f"{module.__name__} requires an output-channel argument")
    c1, c2 = channels[from_index], args[0]
    if c2 != nc:
        c2 = make_divisible(min(c2, max_channels) * width, 8)
    if module is DyMoEBlock:
        if c1 != c2:
            raise ValueError(f"DyMoEBlock preserves channels, but YAML requested {c1} -> {c2}")
        return [c1, *args[1:]], c1, repeats, False
    adapted = [c1, c2, *args[1:]]
    if module in MIXTURE_REPEAT_MODULES:
        adapted.insert(2, repeats)
        repeats = 1
    return adapted, c2, repeats, module is A2C2fMoE


def finalize_mixture_module(instance, module, yaml_args: list[Any], model_config: dict[str, Any]) -> None:
    """Attach YAML provenance and legacy per-model MoE overrides."""
    if module not in MIXTURE_BASE_MODULES:
        return
    annotate_mixture_yaml_config(instance, module.__name__, deepcopy(yaml_args))
    moe_config = model_config.get("moe_config", {})
    if not moe_config:
        return
    for child in instance.modules():
        if hasattr(child, "balance_loss_coeff") and "balance_loss_coeff" in moe_config:
            child.balance_loss_coeff = moe_config["balance_loss_coeff"]
        if hasattr(child, "router_z_loss_coeff") and "router_z_loss_coeff" in moe_config:
            child.router_z_loss_coeff = moe_config["router_z_loss_coeff"]
        routing = getattr(child, "routing", None)
        if routing is not None and hasattr(routing, "noise_std") and "noise_std" in moe_config:
            routing.noise_std = moe_config["noise_std"]
        if routing is not None and hasattr(routing, "temperature") and "temperature" in moe_config:
            routing.temperature = moe_config["temperature"]
        if hasattr(child, "weight_threshold") and "weight_threshold" in moe_config:
            child.weight_threshold = moe_config["weight_threshold"]
        loss = getattr(child, "moe_loss_fn", None)
        if loss is not None:
            if "balance_loss_coeff" in moe_config:
                loss.balance_loss_coeff = moe_config["balance_loss_coeff"]
            if "router_z_loss_coeff" in moe_config:
                loss.z_loss_coeff = moe_config["router_z_loss_coeff"]


__all__ = [
    "MIXTURE_MODULES",
    "MIXTURE_BASE_MODULES",
    "MIXTURE_REPEAT_MODULES",
    "adapt_mixture_args",
    "finalize_mixture_module",
    "get_mixture_module",
]
