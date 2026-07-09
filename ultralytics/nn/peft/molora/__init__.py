"""MoLoRA (Mixture-of-LoRA) public API.

Usage:
    from ultralytics.nn.peft.molora import (
        MoLoRAConfig, get_peft_molora_model, mark_only_molora_as_trainable
    )
"""
from .config import MoLoRAConfig, MoLoRAConfigBuilder, get_molora_preset
from .router import build_router, LinearRouter, SpatialRouter, HybridRouter
from .layer import MoLoRAExpert, MoLoRALayer
from .loss import MoLoRALoss, compute_expert_usage
from .model import get_peft_molora_model, MoLoRAModel
from .utils import (
    mark_only_molora_as_trainable,
    count_parameters,
    allocate_domain_experts,
)
from .moe_aware import (
    MoLoRAMoEAwareConfig,
    PerExpertRankAllocator,
    RouterCalibration,
    MoLoRAMoEAwareLayer,
    build_moe_aware_layer,
)

__all__ = [
    "MoLoRAConfig",
    "MoLoRAConfigBuilder",
    "get_molora_preset",
    "build_router",
    "LinearRouter",
    "SpatialRouter",
    "HybridRouter",
    "MoLoRAExpert",
    "MoLoRALayer",
    "MoLoRALoss",
    "compute_expert_usage",
    "get_peft_molora_model",
    "MoLoRAModel",
    "mark_only_molora_as_trainable",
    "count_parameters",
    "allocate_domain_experts",
    # MoE-aware extensions
    "MoLoRAMoEAwareConfig",
    "PerExpertRankAllocator",
    "RouterCalibration",
    "MoLoRAMoEAwareLayer",
    "build_moe_aware_layer",
]
