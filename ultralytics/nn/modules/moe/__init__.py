# 🐧Please note that this file has been modified by Tencent on 2026/01/16. All Tencent Modifications are Copyright (C) 2026 Tencent.# 🐧Please note that this file has been modified by Tencent on 2026/01/09. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""
Mixture-of-Experts (MoE) modules, routing layers, and compatibility shims.

This module provides several MoE variants and routers optimized for inference efficiency,
plus backward-compatibility aliases so legacy checkpoints can be loaded without changes.
"""

from .modules import (
    UltraOptimizedMoE,
    AdaptiveCapacityMoE,
    ES_MOE,
    OptimizedMOE,
    OptimizedMOEImproved,
    MOE,
    EfficientSpatialRouterMoE,
    ModularRouterExpertMoE,
    HyperSplitMoE,
    HyperFusedMoE,
    HyperUltimateMoE,
    UltimateOptimizedMoE,
    AdaptiveGateMoE,
    DualStreamGateRouter,
    DualStreamGateRouterV2,
    FusedAdaptiveGateMoE,
    HybridAdaptiveGateMoE,
    HybridAdaptiveGateMoEv2,
    OptimalHybridGateMoE,
    MultiHeadRouterMoE,
    DiversifiedExpertMoE,
    GatedFusionMoE,
    LowRankHybridAdaptiveGateMoE,
    RefinedLowRankHybridAdaptiveGateMoE,
    VisualDetailGate,
    PyramidContextMixer,
    DetailAwareLowRankHybridAdaptiveGateMoE,
    ContextRefinedLowRankHybridAdaptiveGateMoE,
    VisualEnhancedAdaptiveGateMoE,
    A2C2fMoE,
    ABlockMoE,
)

from .experts import (
    OptimizedSimpleExpert,
    FusedGhostExpert,
    SimpleExpert,
    GhostExpert,
    InvertedResidualExpert,
    SharedInvertedExpertGroup,
    EfficientExpertGroup,
    DepthwiseSeparableConv
)

from .routers import (
    UltraEfficientRouter,
    BaseRouter,
    EfficientSpatialRouter,
    AdaptiveRoutingLayer,
    LocalRoutingLayer,
    AdvancedRoutingLayer,
    DynamicRoutingLayer
)

from .utils import (
    FlopsUtils,
    get_safe_groups,
    BatchedExpertComputation,
    index_add_aligned_,
    cast_like,
    is_core_moe_block,
    model_has_core_moe,
    iter_core_moe_expert_params,
)

from .analysis import ExpertUsageTracker, diagnose_model, RoutingCollapseDetector
from .diagnostics import MoELayerDiagnostic, collect_moe_diagnostics, diagnostics_to_dict, format_moe_diagnostics
from .history import MoEDiagnosticsRecorder, export_moe_history_plots
from .pruning import prune_moe_model
from .scheduler import (
    MoEDynamicScheduler,
    MoEDynamicSchedulerConfig,
    MoEDynamicScheduleState,
    MapSaturationScheduler,
    MapSaturationSchedulerConfig,
    MapSaturationScheduleState,
    compute_gini,
)
from .config import (
    MIXTURE_DEFAULTS,
    CLI_FIELDS,
    ResolvedMixtureConfig,
    annotate_mixture_yaml_config,
    resolve_mixture_config,
    apply_mixture_config,
)


# ── API Stability Tiers ──────────────────────────────────────────────
# STABLE: production-ready, well-tested, backward-compatible API.
STABLE_MOE_CLASSES = frozenset({
    "UltraOptimizedMoE",
    "ES_MOE",
    "MOE",
    "AdaptiveGateMoE",
    "OptimalHybridGateMoE",
    "UltimateOptimizedMoE",
})

# EXPERIMENTAL: functional but not yet benchmarked at scale; API may change.
EXPERIMENTAL_MOE_CLASSES = frozenset({
    "AdaptiveCapacityMoE",
    "OptimizedMOE",
    "OptimizedMOEImproved",
    "EfficientSpatialRouterMoE",
    "ModularRouterExpertMoE",
    "HyperSplitMoE",
    "HyperFusedMoE",
    "HyperUltimateMoE",
    "FusedAdaptiveGateMoE",
    "HybridAdaptiveGateMoE",
    "HybridAdaptiveGateMoEv2",
    "MultiHeadRouterMoE",
    "DiversifiedExpertMoE",
    "GatedFusionMoE",
    "LowRankHybridAdaptiveGateMoE",
    "RefinedLowRankHybridAdaptiveGateMoE",
    "DetailAwareLowRankHybridAdaptiveGateMoE",
    "ContextRefinedLowRankHybridAdaptiveGateMoE",
    "VisualEnhancedAdaptiveGateMoE",
})

# LEGACY: kept for checkpoint/YAML compatibility while their public contract is
# migrated to the canonical routing protocol.
LEGACY_MOE_CLASSES = frozenset({
    "A2C2fMoE",
    "ABlockMoE",
})

_MOE_TIER_SETS = (STABLE_MOE_CLASSES, EXPERIMENTAL_MOE_CLASSES, LEGACY_MOE_CLASSES)
_MOE_TIER_OVERLAP = set().union(*(
    _MOE_TIER_SETS[i] & _MOE_TIER_SETS[j]
    for i in range(len(_MOE_TIER_SETS))
    for j in range(i + 1, len(_MOE_TIER_SETS))
))
if _MOE_TIER_OVERLAP:
    raise RuntimeError(f"MoE API tier overlap detected: {_MOE_TIER_OVERLAP}")

def is_stable_moe(class_name: str) -> bool:
    """Check if a MoE class is in the stable (production-ready) tier."""
    return class_name in STABLE_MOE_CLASSES

def is_experimental_moe(class_name: str) -> bool:
    """Check if a MoE class is experimental (API may change)."""
    return class_name in EXPERIMENTAL_MOE_CLASSES


def is_legacy_moe(class_name: str) -> bool:
    """Check if a MoE class is retained for compatibility only."""
    return class_name in LEGACY_MOE_CLASSES


__all__ = [
    "UltraOptimizedMoE",
    "AdaptiveCapacityMoE",
    "ES_MOE",
    "OptimizedMOE",
    "OptimizedMOEImproved",
    "MOE",
    "EfficientSpatialRouterMoE",
    "ModularRouterExpertMoE",
    "HyperSplitMoE",
    "HyperFusedMoE",
    "HyperUltimateMoE",
    "UltimateOptimizedMoE",
    "AdaptiveGateMoE",
    "DualStreamGateRouter",
    "DualStreamGateRouterV2",
    "FusedAdaptiveGateMoE",
    "HybridAdaptiveGateMoE",
    "HybridAdaptiveGateMoEv2",
    "OptimalHybridGateMoE",
    "MultiHeadRouterMoE",
    "DiversifiedExpertMoE",
    "GatedFusionMoE",
    "LowRankHybridAdaptiveGateMoE",
    "RefinedLowRankHybridAdaptiveGateMoE",
    "VisualDetailGate",
    "PyramidContextMixer",
    "DetailAwareLowRankHybridAdaptiveGateMoE",
    "ContextRefinedLowRankHybridAdaptiveGateMoE",
    "VisualEnhancedAdaptiveGateMoE",
    "A2C2fMoE",
    "ABlockMoE",
    "OptimizedSimpleExpert",
    "FusedGhostExpert",
    "SimpleExpert",
    "GhostExpert",
    "InvertedResidualExpert",
    "SharedInvertedExpertGroup",
    "EfficientExpertGroup",
    "DepthwiseSeparableConv",
    "UltraEfficientRouter",
    "BaseRouter",
    "EfficientSpatialRouter",
    "AdaptiveRoutingLayer",
    "LocalRoutingLayer",
    "AdvancedRoutingLayer",
    "DynamicRoutingLayer",
    "FlopsUtils",
    "get_safe_groups",
    "index_add_aligned_",
    "cast_like",
    "BatchedExpertComputation",
    "is_core_moe_block",
    "model_has_core_moe",
    "iter_core_moe_expert_params",
    "ExpertUsageTracker",
    "RoutingCollapseDetector",
    "diagnose_model",
    "MoELayerDiagnostic",
    "collect_moe_diagnostics",
    "diagnostics_to_dict",
    "format_moe_diagnostics",
    "MoEDiagnosticsRecorder",
    "export_moe_history_plots",
    "prune_moe_model",
    "MoEDynamicScheduler",
    "MoEDynamicSchedulerConfig",
    "MoEDynamicScheduleState",
    "MapSaturationScheduler",
    "MapSaturationSchedulerConfig",
    "MapSaturationScheduleState",
    "compute_gini",
    "MIXTURE_DEFAULTS",
    "CLI_FIELDS",
    "ResolvedMixtureConfig",
    "annotate_mixture_yaml_config",
    "resolve_mixture_config",
    "apply_mixture_config",
    "STABLE_MOE_CLASSES",
    "EXPERIMENTAL_MOE_CLASSES",
    "LEGACY_MOE_CLASSES",
    "is_stable_moe",
    "is_experimental_moe",
    "is_legacy_moe",
]
