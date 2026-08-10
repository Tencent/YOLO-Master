"""Additive YAML registry for YOLO-Master routed modules."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from typing import Any

from ultralytics.mixture_metadata import MIXTURE_MODULE_KINDS
from ultralytics.nn.modules.block import DyC2f, DyMoEBlock
from ultralytics.nn.modules.moa import C2fMoA
from ultralytics.nn.modules.moe import (
    A2C2fMoE,
    AdaptiveGateMoE,
    ContextRefinedLowRankHybridAdaptiveGateMoE,
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
    SharedExpertMoE,  # Issue #54: Cross-Scale Expert Pool Sharing
    UltimateOptimizedMoE,
    UltraOptimizedMoE,
    VisualEnhancedAdaptiveGateMoE,
)
from ultralytics.nn.modules.moe.config import annotate_mixture_yaml_config
from ultralytics.nn.modules.mot import C2fMoT
from ultralytics.nn.modules.latent_mixture import LatentMixture
from ultralytics.utils.ops import make_divisible


MIXTURE_MODULES = {
    "A2C2fMoE": A2C2fMoE,
    "AdaptiveGateMoE": AdaptiveGateMoE,
    "ContextRefinedLowRankHybridAdaptiveGateMoE": ContextRefinedLowRankHybridAdaptiveGateMoE,
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
    "SharedExpertMoE": SharedExpertMoE,  # Issue #54: Cross-Scale Expert Pool Sharing
    "UltimateOptimizedMoE": UltimateOptimizedMoE,
    "UltraOptimizedMoE": UltraOptimizedMoE,
    "VisualEnhancedAdaptiveGateMoE": VisualEnhancedAdaptiveGateMoE,
    "LatentMixture": LatentMixture,
}
if set(MIXTURE_MODULES) != set(MIXTURE_MODULE_KINDS):
    missing_metadata = sorted(set(MIXTURE_MODULES) - set(MIXTURE_MODULE_KINDS))
    missing_runtime = sorted(set(MIXTURE_MODULE_KINDS) - set(MIXTURE_MODULES))
    raise RuntimeError(
        f"mixture module metadata mismatch: missing_metadata={missing_metadata}, missing_runtime={missing_runtime}"
    )
MIXTURE_BASE_MODULES = frozenset(MIXTURE_MODULES.values())
MIXTURE_REPEAT_MODULES = frozenset({A2C2fMoE, C2fMoA, C2fMoT, DyC2f})
MIXTURE_MULTI_INPUT_MODULES = frozenset({LatentMixture})


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
    if module in MIXTURE_MULTI_INPUT_MODULES:
        if module.__name__ == "MultiScaleLatentMixture":
            raise ValueError(
                "MultiScaleLatentMixture returns list[Tensor] and cannot be registered in the list-to-Tensor parser path"
            )
        if not isinstance(from_index, (list, tuple)) or not from_index:
            raise TypeError(f"{module.__name__} expects a non-empty input index list, got {from_index!r}")
        if len(from_index) < 1 or any(not isinstance(index, int) for index in from_index):
            raise TypeError(f"{module.__name__} input indices must be integers, got {from_index!r}")
        if not args:
            raise ValueError(f"{module.__name__} requires an output-channel argument")
        c1 = [channels[index] for index in from_index]
        c2 = args[0]
        named_config = isinstance(c2, Mapping)
        if named_config:
            config = dict(c2)
            c2 = config.pop("out_channels", config.pop("c2", None))
            if c2 is None:
                raise ValueError(f"{module.__name__} named config requires 'out_channels'")
            named_order = (
                "num_experts",
                "expert_ratio",
                "router_hidden_dim",
                "temperature",
                "balance_loss_coeff",
                "router_z_loss_coeff",
                "residual_init",
                "noise_std",
                "router_init_std",
                "inference_top_k",
                "value_fusion_mode",
                "value_fusion_weights",
                "require_inference_calibration",
            )
            unknown = sorted(set(config).difference(named_order))
            if unknown:
                raise ValueError(f"{module.__name__} named config has unknown keys: {unknown}")
            defaults = (4, 0.25, None, 1.0, 1e-2, 1e-3, 0.0, 0.0, 0.0, None, "router_only", None, False)
            args = [config.get(key, default) for key, default in zip(named_order, defaults)]
        if c2 != nc:
            c2 = make_divisible(min(c2, max_channels) * width, 8)
        # Named configs have already removed their output-channel field;
        # legacy positional configs still carry it at args[0].
        adapted_tail = args if named_config else args[1:]
        return [c1, c2, *adapted_tail], c2, 1, False
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
    "MIXTURE_MULTI_INPUT_MODULES",
    "adapt_mixture_args",
    "finalize_mixture_module",
    "get_mixture_module",
]
