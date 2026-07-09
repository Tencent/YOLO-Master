# MoE modules.py - Facade re-exporting from split submodules
# This file was split into: _common.py, base.py, advanced.py, hybrid.py, integration.py
# It now serves as a backward-compatible facade.

from ._common import (
    autocast,
    MOE_LOSS_REGISTRY,
    MOE_SNAPSHOT_INTERVAL,
    _MOE_LOSS_REGISTRY_LOCK,
    _registry_set,
    _registry_get,
    _should_record_snapshot,
    _zero_aux_loss_like,
    _detached_zero_like,
    _get_moe_aux_loss,
    _flatten_moe_topk,
    _compute_usage_from_topk,
    _record_moe_snapshot,
    _robust_deepcopy,
)

from .base import (
    UltraOptimizedMoE,
    AdaptiveCapacityMoE,
    ES_MOE,
    OptimizedMOE,
    OptimizedMOEImproved,
    ABlockMoE,
    A2C2fMoE,
)

from .advanced import (
    DualStreamGateRouter,
    DualStreamGateRouterV2,
    AdaptiveGateMoE,
    HyperSplitMoE,
    HyperFusedMoE,
    ZeroCostRouter,
    FusedExpertGroup,
    LowRankFusedExpertGroup,
)

from .hybrid import (
    VisualDetailGate,
    PyramidContextMixer,
    _run_visual_hybrid_moe_forward,
    FusedAdaptiveGateMoE,
    HybridAdaptiveGateMoE,
    HybridAdaptiveGateMoEv2,
    LowRankHybridAdaptiveGateMoE,
    RefinedLowRankHybridAdaptiveGateMoE,
    DetailAwareLowRankHybridAdaptiveGateMoE,
    ContextRefinedLowRankHybridAdaptiveGateMoE,
    VisualEnhancedAdaptiveGateMoE,
    AdaptiveBalanceController,
    OptimalHybridGateMoE,
)

from .integration import (
    MultiHeadRouterV3,
    DiversifiedExpertGroup,
    CrossPathGate,
    MultiHeadRouterMoE,
    DiversifiedExpertMoE,
    GatedFusionMoE,
    UltraLightRouter,
    MatMulFusedExperts,
    HyperUltimateMoE,
    UltimateOptimizedMoE,
    MOE,
    EfficientSpatialRouterMoE,
    ModularRouterExpertMoE,
)
