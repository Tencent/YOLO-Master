"""MoLoRA configuration and target detection builder.

MoLoRAConfig extends the standard LoRAConfig with Mixture-of-LoRA specific
parameters: num_experts, top_k, router_type, and balance-loss coefficients.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from ultralytics.utils import LOGGER
from ultralytics.utils.lora.config import LoRAConfig, LoRAConfigBuilder


@dataclass
class MoLoRAConfig(LoRAConfig):
    """Configuration dataclass for Mixture-of-LoRA (MoLoRA) training strategies.

    Inherits all fields from LoRAConfig and adds MoE-style sparse expert
    selection on top of low-rank adapters.
    """

    # MoLoRA core parameters
    num_experts: int = 4
    top_k: int = 2
    router_type: str = "linear"  # "linear", "spatial", "hybrid"
    router_hidden_dim: Optional[int] = None  # auto = C // 4

    # Auxiliary loss coefficients
    balance_loss_coef: float = 0.01
    z_loss_coef: float = 0.001
    diversity_loss_coef: float = 0.0

    # Routing behaviour
    capacity_factor: float = 1.0  # dynamic capacity limit (E=1 means no limit)
    expert_dropout: float = 0.0  # training-time probability of disabling an expert
    top_k_warmup: Optional[int] = None  # gradually increase K from 1 to top_k over N steps
    warmup_steps: int = 0  # number of steps for warmup

    # Continual learning / domain isolation
    domain_experts: Optional[Dict[str, List[int]]] = None
    freeze_experts: Optional[List[int]] = None  # reserved

    # Registry integration
    share_moe_registry: bool = True  # write balance loss to MOE_LOSS_REGISTRY

    # Initialization strategy for each expert
    expert_init: str = "default"  # "default" | "orthogonal" | "gaussian"

    def __post_init__(self):
        """Validate MoLoRA-specific invariants on top of LoRA validation."""
        super().__post_init__()

        if self.num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {self.num_experts}")
        if self.top_k < 1 or self.top_k > self.num_experts:
            raise ValueError(
                f"top_k must be in [1, num_experts={self.num_experts}], got {self.top_k}"
            )
        if self.router_type not in ("linear", "spatial", "hybrid"):
            raise ValueError(
                f"router_type must be 'linear', 'spatial', or 'hybrid', got {self.router_type}"
            )
        if self.balance_loss_coef < 0:
            raise ValueError(f"balance_loss_coef must be >= 0, got {self.balance_loss_coef}")
        if self.z_loss_coef < 0:
            raise ValueError(f"z_loss_coef must be >= 0, got {self.z_loss_coef}")
        if self.diversity_loss_coef < 0:
            raise ValueError(f"diversity_loss_coef must be >= 0, got {self.diversity_loss_coef}")
        if self.expert_init not in ("default", "orthogonal", "gaussian"):
            raise ValueError(
                f"expert_init must be 'default', 'orthogonal', or 'gaussian', got {self.expert_init}"
            )

    @classmethod
    def from_lora_config(cls, lora_config: LoRAConfig, **molora_overrides) -> "MoLoRAConfig":
        """Promote an existing LoRAConfig to MoLoRAConfig, preserving all settings."""
        base = {k: v for k, v in lora_config.__dict__.items() if k in cls.__dataclass_fields__}
        base.update(molora_overrides)
        return cls(**base)

    @classmethod
    def from_args(cls, args=None, **kwargs) -> "MoLoRAConfig":
        """Construct from Ultralytics args or kwargs, mapping 'molora_' prefixed keys."""
        if args is None and not kwargs:
            return cls()

        # First pull standard LoRA fields via parent logic
        lora_fields = set(LoRAConfig.__dataclass_fields__)
        lora_kwargs = {k: v for k, v in kwargs.items() if k in lora_fields}
        lora_args = {k: getattr(args, k, None) for k in lora_fields if args is not None}
        lora_args = {k: v for k, v in lora_args.items() if v is not None}
        lora_kwargs.update(lora_args)

        base = LoRAConfig.from_args(args=args, **lora_kwargs)

        # MoLoRA-specific arg mapping
        molora_mapping = {
            "r": "molora_r",
            "alpha": "molora_alpha",
            "num_experts": "molora_num_experts",
            "top_k": "molora_top_k",
            "router_type": "molora_router_type",
            "router_hidden_dim": "molora_router_hidden_dim",
            "balance_loss_coef": "molora_balance_loss",
            "z_loss_coef": "molora_router_z_loss",
            "diversity_loss_coef": "molora_diversity_loss",
            "capacity_factor": "molora_capacity_factor",
            "expert_dropout": "molora_expert_dropout",
            "top_k_warmup": "molora_top_k_warmup",
            "warmup_steps": "molora_warmup_steps",
            "domain_experts": "molora_domain_experts",
            "freeze_experts": "molora_freeze_experts",
            "share_moe_registry": "molora_share_moe_registry",
            "expert_init": "molora_expert_init",
            "use_rslora": "molora_use_rslora",
        }

        molora_kwargs = {}
        for field_name, arg_name in molora_mapping.items():
            val = kwargs.get(arg_name)
            if val is None and args is not None:
                val = getattr(args, arg_name, None)
            if val is not None:
                molora_kwargs[field_name] = val

        return cls.from_lora_config(base, **molora_kwargs)


class MoLoRAConfigBuilder(LoRAConfigBuilder):
    """Extends LoRAConfigBuilder with MoLoRA-specific target detection.

    Reuses the parent's auto_detect_targets logic entirely; MoLoRA does not
    need to change which layers are adapted, only *how* they are adapted.
    """

    @staticmethod
    def create_molora_config(
        model: nn.Module,
        r: int = 8,
        alpha: Optional[int] = None,
        num_experts: int = 4,
        top_k: int = 2,
        router_type: str = "linear",
        balance_loss_coef: float = 0.01,
        z_loss_coef: float = 0.001,
        use_rslora: bool = True,
        expert_init: str = "default",
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Build a plain-dict config suitable for the in-repo MoLoRA implementation.

        Unlike the parent `create_config` which returns a PEFT Config object,
        this returns a dict that `get_peft_molora_model` consumes directly.
        """
        # Reuse parent detection to find target layers
        detect_kwargs = {
            "include_moe": kwargs.get("include_moe", True),
            "include_attention": kwargs.get("include_attention", False),
            "only_backbone": kwargs.get("only_backbone", False),
            "exclude_modules": kwargs.get("exclude_modules"),
            "last_n": kwargs.get("last_n"),
            "from_layer": kwargs.get("from_layer"),
            "to_layer": kwargs.get("to_layer"),
            "allow_depthwise": kwargs.get("allow_depthwise", False),
            "kernels": kwargs.get("kernels"),
            "skip_stem": kwargs.get("skip_stem", False),
            "min_channels": kwargs.get("min_channels", 0),
            "only_3x3": kwargs.get("only_3x3", False),
        }
        targets = LoRAConfigBuilder.auto_detect_targets(model, r=r, **detect_kwargs)
        if not targets:
            LOGGER.warning("[MoLoRA] No valid target modules found. MoLoRA skipped.")
            return None

        if alpha is None:
            alpha = 2 * r

        return {
            "r": r,
            "alpha": alpha,
            "target_modules": targets,
            "num_experts": num_experts,
            "top_k": top_k,
            "router_type": router_type,
            "balance_loss_coef": balance_loss_coef,
            "z_loss_coef": z_loss_coef,
            "use_rslora": use_rslora,
            "expert_init": expert_init,
            "dropout": kwargs.get("dropout", 0.05),
            "include_head": kwargs.get("include_head", False),
            "freeze_bn": kwargs.get("freeze_bn", False),
        }


# ---------------------------------------------------------------------------
# Preset factory
# ---------------------------------------------------------------------------

def get_molora_preset(name: str) -> Dict[str, Any]:
    """Return a named preset configuration dict.

    Presets:
      - preset_small:   2 experts, top_k=1, r=4, alpha=8   (mobile/minimal)
      - preset_standard: 4 experts, top_k=2, r=8, alpha=16  (default)
      - preset_large:    8 experts, top_k=2, r=16, alpha=32 (high capacity)
      - preset_continual: 8 experts, top_k=2, r=8, alpha=16  (continual learning)
    """
    presets = {
        "preset_small": {
            "num_experts": 2,
            "top_k": 1,
            "r": 4,
            "alpha": 8,
            "router_type": "linear",
        },
        "preset_standard": {
            "num_experts": 4,
            "top_k": 2,
            "r": 8,
            "alpha": 16,
            "router_type": "linear",
        },
        "preset_large": {
            "num_experts": 8,
            "top_k": 2,
            "r": 16,
            "alpha": 32,
            "router_type": "hybrid",
        },
        "preset_continual": {
            "num_experts": 8,
            "top_k": 2,
            "r": 8,
            "alpha": 16,
            "router_type": "linear",
            "domain_experts": None,
        },
    }
    if name not in presets:
        raise ValueError(f"Unknown preset '{name}'. Choose from {list(presets.keys())}.")
    return presets[name].copy()
