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
    FusedAdaptiveGateMoE,
    HybridAdaptiveGateMoE,
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
    BatchedExpertComputation
)

from .analysis import ExpertUsageTracker, diagnose_model, RoutingCollapseDetector
from .diagnostics import MoELayerDiagnostic, collect_moe_diagnostics, diagnostics_to_dict, format_moe_diagnostics
from .history import MoEDiagnosticsRecorder, export_moe_history_plots
from .pruning import prune_moe_model

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
    "FusedAdaptiveGateMoE",
    "HybridAdaptiveGateMoE",
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
    "BatchedExpertComputation",
    "ExpertUsageTracker",
    "RoutingCollapseDetector",
    "diagnose_model",
    "MoELayerDiagnostic",
    "collect_moe_diagnostics",
    "diagnostics_to_dict",
    "format_moe_diagnostics",
    "MoEDiagnosticsRecorder",
    "export_moe_history_plots",
    "prune_moe_model"
]
