"""Lightweight metadata shared by mixture configuration and runtime registries."""

from types import MappingProxyType


MIXTURE_KINDS = ("moe", "moa", "mot", "latent")

MIXTURE_MODULE_KINDS = MappingProxyType(
    {
        "A2C2fMoE": "moe",
        "AdaptiveGateMoE": "moe",
        "ContextRefinedLowRankHybridAdaptiveGateMoE": "moe",
        "C2fMoA": "moa",
        "C2fMoT": "mot",
        "DetailAwareLowRankHybridAdaptiveGateMoE": "moe",
        "DiversifiedExpertMoE": "moe",
        "DyC2f": "moe",
        "DyMoEBlock": "moe",
        "ES_MOE": "moe",
        "FusedAdaptiveGateMoE": "moe",
        "GatedFusionMoE": "moe",
        "HybridAdaptiveGateMoE": "moe",
        "HybridAdaptiveGateMoEv2": "moe",
        "LatentMixture": "latent",
        "LowRankHybridAdaptiveGateMoE": "moe",
        "ModularRouterExpertMoE": "moe",
        "MultiHeadRouterMoE": "moe",
        "OptimalHybridGateMoE": "moe",
        "RefinedLowRankHybridAdaptiveGateMoE": "moe",
        "SharedExpertMoE": "moe",  # Issue #54: Cross-Scale Expert Pool Sharing
        "UltimateOptimizedMoE": "moe",
        "UltraOptimizedMoE": "moe",
        "VisualEnhancedAdaptiveGateMoE": "moe",
    }
)

__all__ = ("MIXTURE_KINDS", "MIXTURE_MODULE_KINDS")
