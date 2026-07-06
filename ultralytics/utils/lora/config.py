# 🐧Please note that this file has been modified by Tencent on 2026/02/13. All Tencent Modifications are Copyright (C) 2026 Tencent.
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from ultralytics.utils import LOGGER
from .api import (
    AdaLoraConfig,
    BOFTConfig,
    HRAConfig,
    IA3Config,
    LoraConfig as PeftLoraConfig,
    LoHaConfig,
    LoKrConfig,
    OFTConfig,
    PEFT_AVAILABLE,
    _effective_peft_variant,
    _fast_parse_int_list,
    _fast_parse_str_list,
    _is_rtdetr_like_model,
    _normalize_lora_init,
    _supports_peft_kwarg,
    resolve_adalora_total_step,
)
from .fallback import (
    _build_peft_exact_target_regex,
    _filter_target_modules,
    _validate_peft_init_compatibility,
)

@dataclass
class LoRAConfig:
    """
    Configuration dataclass for LoRA training strategies.
    """
    # Core Parameters
    r: int = 0  # LoRA Rank. 0 means disabled.
    alpha: int = 32 # Scaling factor.
    dropout: float = 0.05
    bias: str = "none"  # Options: "none", "all", "lora_only"
    backend: str = "auto"  # Execution backend: "auto", "peft", "fallback"
    variant: str = "lora"  # Adapter variant: "lora", "loha", "dora"
    include_head: bool = False  # Include detection head layers in target selection
    freeze_bn: bool = False  # Freeze BatchNorm layers during LoRA training
    
    # Strategy Control
    lr_mult: float = 1.0
    include_moe: bool = True
    include_attention: bool = False
    only_backbone: bool = False
    exclude_modules: Optional[List[str]] = None
    target_modules: Optional[List[str]] = None

    # Layer Filtering
    last_n: Optional[int] = None
    from_layer: Optional[int] = None
    to_layer: Optional[int] = None

    # Convolution Specifics
    allow_depthwise: bool = False
    kernels: Optional[List[int]] = None

    # Capacity allocation knobs
    skip_stem: bool = False  # Skip backbone stem (first 3 top-level layers)
    min_channels: int = 0    # Skip narrow layers (min(in, out) below this threshold)

    # Advanced Options
    gradient_checkpointing: bool = False
    auto_r_ratio: float = 0.0 # Automatically calculate R based on parameter ratio
    use_dora: bool = False # Enable DoRA (Weight-Decomposed Low-Rank Adaptation)
    allow_rtdetr_dora: bool = False # Allow experimental RT-DETR + DoRA instead of auto-degrading to LoRA
    use_rslora: bool = True # Enable Rank-Stabilized LoRA scaling (alpha / sqrt(r))
    init_lora_weights: Union[str, bool] = True # LoRA init mode: True/False for std init, or "gaussian"/"pissa"/"olora"
    peft_type: str = "lora" # Options: "lora", "loha", "lokr", "adalora", "ia3", "oft", "boft", "hra"
    quantization: str = "none" # Options: "none", "4bit", "8bit" (Requires bitsandbytes)
    only_3x3: bool = False # Skip 1x1 convs during auto target selection

    # Training strategy parameters (synced with default.yaml)
    layer_decay: float = 0.0 # Layer-wise LR decay rate (0=disabled)
    alpha_warmup: int = 0 # Alpha cosine warmup epochs (0=disabled)
    ortho_weight: float = 0.0 # Orthogonal regularization weight (0=disabled)
    ortho_frequency: int = 10 # Compute orthogonal loss every N batches
    dropout_end: float = 0.15 # Final dropout for dynamic schedule
    dropout_start_ratio: float = 0.3 # When to start increasing dropout (fraction of total epochs)

    # HRA specific (only used when peft_type=hra)
    hra_apply_gs: bool = False  # HRA: apply Gram-Schmidt orthogonalization
    oft_block_size: int = 0          # OFT: block size (>0 overrides r)
    oft_coft: bool = False           # OFT: use constrained (Cayley-Neumann) rotations
    oft_eps: float = 6e-5            # OFT: numerical eps
    oft_block_share: bool = False    # OFT: share rotation across blocks
    boft_block_size: int = 2         # BOFT: butterfly block size (must divide kernel dim; 2 for YOLO 3x3 Conv)
    boft_block_num: int = 0          # BOFT: number of butterfly blocks (0 = auto)
    boft_n_butterfly_factor: int = 2 # BOFT: butterfly factor (paper default)

    target_r: int = 8 # AdaLoRA target rank
    init_r: int = 12 # AdaLoRA initial rank
    tinit: int = 0 # AdaLoRA warmup steps before pruning
    tfinal: int = 0 # AdaLoRA final fine-tuning steps
    delta_t: int = 1 # AdaLoRA allocation interval
    beta1: float = 0.85 # AdaLoRA EMA beta1
    beta2: float = 0.85 # AdaLoRA EMA beta2
    orth_reg_weight: float = 0.5 # AdaLoRA orthogonal regularization weight
    total_step: Optional[int] = None # AdaLoRA total training steps, required by PEFT

    # Few-Shot Options
    few_shot_mode: bool = False # Enable few-shot LoRA with enhanced regularization
    few_shot_teacher: Optional[str] = None # Path to teacher model for knowledge distillation
    few_shot_dropconnect: float = 0.1 # DropConnect rate (better than dropout for few-shot)
    few_shot_distill_weight: float = 0.5 # Weight for distillation loss
    few_shot_adaptive_rank: bool = True # Auto-adjust rank based on data scarcity
    # Enhancements
    few_shot_dropconnect_schedule: str = "cosine"  # DropConnect schedule: constant/linear/cosine/exponential
    few_shot_dropconnect_max: float = 0.3  # Initial max DropConnect rate
    few_shot_dropconnect_min: float = 0.0  # Final min DropConnect rate
    few_shot_gradient_importance_weighted: bool = False  # Use gradient-importance weighted DropConnect
    few_shot_hierarchical_distill: bool = False  # Enable multi-layer hierarchical distillation
    few_shot_distill_layers: Optional[List[int]] = None  # Layer indices for intermediate distillation
    few_shot_variational_rank: bool = False  # Enable variational rank selection
    few_shot_rank_budget: float = 0.5  # Budget ratio for rank retention
    few_shot_adaptive_temperature: bool = False  # Enable task-adaptive distillation temperature
    few_shot_curriculum_sampling: bool = False  # Enable curriculum learning sampler
    # v3 Enhancements
    few_shot_distill_schedule: str = "cosine"  # Distillation weight schedule: constant/linear/cosine
    few_shot_distill_weight_max: float = 1.0  # Initial max distillation weight
    few_shot_distill_weight_min: float = 0.1  # Final min distillation weight
    few_shot_use_ema_teacher: bool = False  # Use EMA teacher for progressive self-distillation
    few_shot_ema_decay: float = 0.999  # EMA teacher decay rate
    few_shot_response_distill: bool = False  # Enable detection head response distillation
    few_shot_response_distill_weight: float = 0.3  # Weight for response distillation loss
    few_shot_layerwise_rank: bool = False  # Enable per-layer adaptive rank
    few_shot_hook_cache: bool = True  # Cache hierarchical distillation hooks across batches

    # Planner integration
    planner_enabled: bool = False  # Enable architecture-conditioned PEFT Planner

    def __post_init__(self):
        """Performs parameter validation and type standardization."""
        # Standardize list inputs
        if isinstance(self.kernels, str): self.kernels = _fast_parse_int_list(self.kernels)
        if isinstance(self.exclude_modules, str): self.exclude_modules = _fast_parse_str_list(self.exclude_modules)
        if isinstance(self.target_modules, str): self.target_modules = _fast_parse_str_list(self.target_modules)

        # Logical validation
        if self.auto_r_ratio > 0:
            if self.r < 0: self.r = 0 # Will be handled by auto logic
        elif self.r < 0:
            raise ValueError("lora_r must be >= 0")

        self.init_lora_weights = _normalize_lora_init(self.init_lora_weights)

        # Few-shot config validation
        if self.few_shot_mode:
            if self.few_shot_dropconnect_max < self.few_shot_dropconnect_min:
                raise ValueError(
                    f"lora_few_shot_dropconnect_max ({self.few_shot_dropconnect_max}) "
                    f"must be >= lora_few_shot_dropconnect_min ({self.few_shot_dropconnect_min})"
                )
            if not (0.0 <= self.few_shot_rank_budget <= 1.0):
                raise ValueError(
                    f"lora_few_shot_rank_budget ({self.few_shot_rank_budget}) must be in [0, 1]"
                )
            if self.few_shot_distill_weight_max < self.few_shot_distill_weight_min:
                raise ValueError(
                    f"lora_few_shot_distill_weight_max ({self.few_shot_distill_weight_max}) "
                    f"must be >= lora_few_shot_distill_weight_min ({self.few_shot_distill_weight_min})"
                )
            if not (0.0 < self.few_shot_ema_decay <= 1.0):
                raise ValueError(
                    f"lora_few_shot_ema_decay ({self.few_shot_ema_decay}) must be in (0, 1]"
                )
            if self.few_shot_distill_layers:
                for idx in self.few_shot_distill_layers:
                    if not isinstance(idx, int) or idx < 0:
                        raise ValueError(
                            f"lora_few_shot_distill_layers must contain non-negative ints, got {idx}"
                        )
            if self.few_shot_use_ema_teacher and not self.few_shot_teacher:
                LOGGER.warning(
                    "[LoRA] lora_few_shot_use_ema_teacher=True but no teacher model specified. "
                    "EMA teacher will use student initialization."
                )

    @classmethod
    def from_args(cls, args=None, **kwargs):
        """
        Constructs configuration from Ultralytics args or kwargs.
        Supports automatic mapping of 'lora_' prefixed arguments.
        """
        if args is None and not kwargs:
            return cls()

        # Mapping: LoRAConfig field -> Ultralytics args attribute
        mapping = {
            "r": "lora_r", 
            "alpha": "lora_alpha", 
            "dropout": "lora_dropout",
            "bias": "lora_bias", 
            "backend": "lora_backend",
            "variant": "lora_variant",
            "include_head": "lora_include_head",
            "freeze_bn": "lora_freeze_bn",
            "lr_mult": "lora_lr_mult",
            "include_moe": "lora_include_moe",
            "include_attention": "lora_include_attention",
            "only_backbone": "lora_only_backbone", 
            "exclude_modules": "lora_exclude_modules",
            "last_n": "lora_last_n", 
            "from_layer": "lora_from_layer", 
            "to_layer": "lora_to_layer",
            "allow_depthwise": "lora_allow_depthwise", 
            "kernels": "lora_kernels",
            "skip_stem": "lora_skip_stem",
            "min_channels": "lora_min_channels",
            "target_modules": "lora_target_modules", 
            "gradient_checkpointing": "lora_gradient_checkpointing",
            "auto_r_ratio": "lora_auto_r_ratio",
            "use_dora": "lora_use_dora",
            "allow_rtdetr_dora": "lora_allow_rtdetr_dora",
            "use_rslora": "lora_use_rslora",
            "init_lora_weights": "lora_init_lora_weights",
            "peft_type": "lora_type",
            "quantization": "lora_quantization",
            "only_3x3": "lora_only_3x3",
            "layer_decay": "lora_layer_decay",
            "alpha_warmup": "lora_alpha_warmup",
            "ortho_weight": "lora_ortho_weight",
            "ortho_frequency": "lora_ortho_frequency",
            "dropout_end": "lora_dropout_end",
            "dropout_start_ratio": "lora_dropout_start_ratio",
            "oft_block_size": "lora_oft_block_size",
            "oft_coft": "lora_oft_coft",
            "oft_eps": "lora_oft_eps",
            "oft_block_share": "lora_oft_block_share",
            "boft_block_size": "lora_boft_block_size",
            "boft_block_num": "lora_boft_block_num",
            "boft_n_butterfly_factor": "lora_boft_n_butterfly_factor",
            # HRA
            "hra_apply_gs": "lora_hra_apply_gs",
            "target_r": "lora_target_r",
            "init_r": "lora_init_r",
            "tinit": "lora_tinit",
            "tfinal": "lora_tfinal",
            "delta_t": "lora_delta_t",
            "beta1": "lora_beta1",
            "beta2": "lora_beta2",
            "orth_reg_weight": "lora_orth_reg_weight",
            "total_step": "lora_total_step",
            "few_shot_mode": "lora_few_shot_mode",
            "few_shot_teacher": "lora_few_shot_teacher",
            "few_shot_dropconnect": "lora_few_shot_dropconnect",
            "few_shot_distill_weight": "lora_few_shot_distill_weight",
            "few_shot_adaptive_rank": "lora_few_shot_adaptive_rank",
            # Enhancement mappings
            "few_shot_dropconnect_schedule": "lora_few_shot_dropconnect_schedule",
            "few_shot_dropconnect_max": "lora_few_shot_dropconnect_max",
            "few_shot_dropconnect_min": "lora_few_shot_dropconnect_min",
            "few_shot_gradient_importance_weighted": "lora_few_shot_gradient_importance_weighted",
            "few_shot_hierarchical_distill": "lora_few_shot_hierarchical_distill",
            "few_shot_distill_layers": "lora_few_shot_distill_layers",
            "few_shot_variational_rank": "lora_few_shot_variational_rank",
            "few_shot_rank_budget": "lora_few_shot_rank_budget",
            "few_shot_adaptive_temperature": "lora_few_shot_adaptive_temperature",
            "few_shot_curriculum_sampling": "lora_few_shot_curriculum_sampling",
            # v3 mappings
            "few_shot_distill_schedule": "lora_few_shot_distill_schedule",
            "few_shot_distill_weight_max": "lora_few_shot_distill_weight_max",
            "few_shot_distill_weight_min": "lora_few_shot_distill_weight_min",
            "few_shot_use_ema_teacher": "lora_few_shot_use_ema_teacher",
            "few_shot_ema_decay": "lora_few_shot_ema_decay",
            "few_shot_response_distill": "lora_few_shot_response_distill",
            "few_shot_response_distill_weight": "lora_few_shot_response_distill_weight",
            "few_shot_layerwise_rank": "lora_few_shot_layerwise_rank",
            "few_shot_hook_cache": "lora_few_shot_hook_cache",
            # Planner integration
            "planner_enabled": "lora_planner_enabled",
        }

        dataclass_fields = set(cls.__dataclass_fields__)
        final_args = {key: value for key, value in kwargs.items() if key in dataclass_fields}

        for field, arg_name in mapping.items():
            if field not in final_args and arg_name in kwargs:
                val = kwargs.get(arg_name)
                if val is not None:
                    final_args[field] = val
        
        # Extract arguments from the args object
        if args is not None:
            for field, arg_name in mapping.items():
                if field not in final_args and hasattr(args, arg_name):
                    val = getattr(args, arg_name, None)
                    if val is not None:
                        final_args[field] = val
        
        return cls(**final_args)


# ============================================================================
# 3. Smart Builder
# ============================================================================

class LoRAConfigBuilder:
    """
    Analyzes model structure to generate optimal LoRA configurations.
    """

    # Pre-compiled regex for performance
    _PAT_BACKBONE_EXCLUDE = re.compile(r"(head|detect|box|cls|pred|fpn|pan|seg|pose|enc_score_head|enc_bbox_head|dec_score_head|dec_bbox_head)", re.IGNORECASE)
    _PAT_MOE = re.compile(r"(expert|moe)", re.IGNORECASE)
    _PAT_ATTN = re.compile(r"attn", re.IGNORECASE)
    # YOLO12 Area-Attention pattern: matches Conv2d-based qkv/proj/pe submodules.
    # Excluded from LoRA targets by default to avoid breaking softmax numerical stability.
    _PAT_AREA_ATTN = re.compile(r"\.attn\.(qkv|proj|pe)(\.|$)", re.IGNORECASE)
    # YOLO12 ABlock-internal MLP pattern: ABlock has no LayerNorm; LoRA on the
    # post-attention residual MLP path also causes gradient explosion (→ NaN
    # mid-training). Match the *.m.<n>.<k>.mlp.<*>.conv path that lives inside
    # A2C2f -> ABlock and is therefore on the same residual stream as AAttn.
    _PAT_AREA_ATTN_MLP = re.compile(
        r"\.m\.\d+\.\d+\.mlp\.\d+(\.|$)", re.IGNORECASE
    )
    # RT-DETR MSDeformAttn geometry-sensitive Linear layers.
    # sampling_offsets carries grid-initialized bias encoding the deformable
    # sampling grid; LoRA perturbation breaks sampling geometry consistency
    # and causes bbox regression to drift.
    # attention_weights feeds a softmax whose weights are zero-initialized;
    # even small LoRA deltas saturate the softmax. Both are excluded by
    # default; opt-in requires r<=4 and long alpha_warmup.
    _PAT_MSDEFORM_RISKY = re.compile(
        r"(sampling_offsets|attention_weights)(\.|$)", re.IGNORECASE
    )
    _PAT_INDEX = re.compile(r"^(\d+)\.") # Matches "0" in "0.conv"
    _PAT_INDEX_ANY = re.compile(r"(?:^|\.)(\d+)\.")  # Matches first numeric segment anywhere (e.g. "model.5.m.0.cv1" -> 5)

    @staticmethod
    def _get_layer_index(name: str) -> int:
        """Extract the top-level layer index from a (possibly nested) module name.

        Accepts patterns like:
          - "0.conv" -> 0 (flat sequential)
          - "model.5.m.0.cv1" -> 5 (nested YOLO naming)
          - "backbone.12.bn" -> 12
        Returns -1 when no numeric segment is found.
        """
        # Fast path: flat sequential
        match = LoRAConfigBuilder._PAT_INDEX.search(name)
        if match:
            return int(match.group(1))
        # Fallback: look for first numeric segment anywhere (after a dot or at start)
        match = LoRAConfigBuilder._PAT_INDEX_ANY.search(name)
        return int(match.group(1)) if match else -1

    @staticmethod
    def auto_detect_targets(
        model: nn.Module,
        r: int,
        include_moe: bool = True,
        include_attention: bool = False,
        only_backbone: bool = False,
        exclude_modules: Optional[List[str]] = None,
        layer_from: Optional[int] = None,
        layer_to: Optional[int] = None,
        last_n: Optional[int] = None,
        allow_depthwise: bool = False,
        kernels: Optional[List[int]] = None,
        skip_stem: bool = False,
        min_channels: int = 0,
        planner_enabled: bool = False,
        **kwargs,
    ) -> List[str]:
        """Intelligently detect target layers for LoRA injection.

        Extra knobs for better capacity allocation:
          skip_stem:     if True, exclude the first 3 top-level layers
                         (typical backbone stem). Stem rarely benefits from
                         LoRA in transfer learning.
          min_channels:  if >0, skip layers whose min(in, out) < min_channels.
                         Useful to avoid full-rank reparameterization on
                         narrow layers when using a large base rank.
        """
        targets: Set[str] = set()
        exclude_set = set(exclude_modules) if exclude_modules else set()
        allowed_kernels = set(kernels) if kernels else None

        # Determine layer range
        total_layers = len(model) if hasattr(model, '__len__') else 1000
        start_idx = 0
        end_idx = total_layers

        if last_n is not None:
            start_idx = max(0, total_layers - last_n)
        if layer_from is not None:
            start_idx = max(start_idx, layer_from)
        if layer_to is not None:
            end_idx = min(total_layers, layer_to)
        
        apply_idx_filter = (last_n is not None) or (layer_from is not None) or (layer_to is not None)
        
        if apply_idx_filter:
            LOGGER.debug(f"[LoRA] Layer filter active: {start_idx} - {end_idx}")

        # Planner delegation: architecture-conditioned target detection.
        # When enabled, the PEFTPlanner provides a hint set that pre-filters the
        # scan so architecture-specific rules (e.g., RT-DETR refusal, YOLO12 safe
        # attention) are centralized in the planner rather than duplicated here.
        planner_hint_set: Optional[Set[str]] = None
        if planner_enabled:
            try:
                from .planner import PEFTPlanner
                planner = PEFTPlanner()
                # Reconstruct a minimal LoRAConfig from kwargs for the planner
                config_fields = set(LoRAConfig.__dataclass_fields__)
                planner_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
                planner_config = LoRAConfig(**planner_kwargs) if planner_kwargs else LoRAConfig()
                decision = planner.plan(model, planner_config)
                if hasattr(decision, "target_modules_hint") and decision.target_modules_hint:
                    planner_hint_set = set(decision.target_modules_hint)
                    LOGGER.info(
                        f"[Planner] Architecture-conditioned hint: {len(planner_hint_set)} targets "
                        f"(status={getattr(decision, 'status', 'UNKNOWN')})"
                    )
            except ImportError:
                LOGGER.warning(
                    "[Planner] PEFTPlanner module not found; falling back to standard auto-detection."
                )
            except Exception as exc:
                LOGGER.warning(
                    f"[Planner] PEFTPlanner failed ({type(exc).__name__}: {exc}); "
                    "falling back to standard auto-detection."
                )

        # Iterate through all sub-modules
        for name, module in model.named_modules():
            if not name: continue 
            
            # 0. Explicit Exclusion
            if name in exclude_set:
                continue

            # 0.5 Planner hint filter
            if planner_hint_set is not None and name not in planner_hint_set:
                continue

            # 1. Index Filtering (Valid only if module name starts with a digit)
            if apply_idx_filter:
                idx = LoRAConfigBuilder._get_layer_index(name)
                if idx != -1:
                    if not (start_idx <= idx < end_idx):
                        continue

            # 1b. Skip stem (first three top-level backbone layers).
            # Stem is low-level, rarely benefits from LoRA in transfer learning,
            # and wastes adapter capacity.
            if skip_stem:
                idx = LoRAConfigBuilder._get_layer_index(name)
                if 0 <= idx <= 2:
                    continue

            # 2. Type Filtering (Must be Conv2d or Linear)
            is_conv = isinstance(module, nn.Conv2d)
            is_linear = isinstance(module, nn.Linear)
            if not (is_conv or is_linear):
                continue

            # 2b. Min-channel filter: avoid attaching LoRA to very narrow layers
            # where the requested rank would exceed capacity.
            if min_channels > 0 and is_conv:
                if min(module.in_channels, module.out_channels) < min_channels:
                    continue
            if min_channels > 0 and is_linear:
                if min(module.in_features, module.out_features) < min_channels:
                    continue

            # 3. Backbone Filtering
            if only_backbone and LoRAConfigBuilder._PAT_BACKBONE_EXCLUDE.search(name):
                continue

            # 4. Convolution Specific Checks
            if is_conv:
                # Grouped Conv / Depthwise Checks
                if module.groups > 1:
                    # FIX: Properly handle grouped convolutions.
                    # PEFT requires: LoRA rank must be a multiple of groups for Conv2d.
                    # 
                    # Key distinction:
                    # - Depthwise: groups == in_channels == out_channels (extremely sparse, usually skip)
                    # - Standard grouped conv: groups < in_channels (e.g., C3k2 uses groups=4, 8)
                    #   These should be INCLUDED if r % groups == 0.
                    
                    is_depthwise = (module.in_channels == module.out_channels == module.groups)
                    
                    # Check rank divisibility first
                    if r > 0 and (r % module.groups != 0):
                        # Skip to avoid PEFT ValueError
                        LOGGER.debug(f"[LoRA] Skipping {name}: groups={module.groups}, rank={r} (rank % groups != 0)")
                        continue
                    
                    # Handle depthwise specifically
                    if is_depthwise:
                        # Only include depthwise if explicitly allowed
                        if not allow_depthwise:
                            LOGGER.debug(f"[LoRA] Skipping depthwise {name}: {module.in_channels} channels")
                            continue
                        # Even if allowed, warn as depthwise LoRA is often ineffective
                        LOGGER.info(f"[LoRA] Including depthwise layer {name} (allow_depthwise=True)")
                    # else: standard grouped conv (groups < in_channels) -> ALLOW through
                
                # Pointwise Conv (1x1) Check - Highly Recommended for LoRA
                # Standard Conv (3x3) Check - Supported
                # Kernel Size Check
                if allowed_kernels:
                    k_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                    if k_size not in allowed_kernels:
                        continue
                if kwargs.get("only_3x3", False):
                    k_size = module.kernel_size
                    if k_size == 1 or k_size == (1, 1):
                        continue
            
            # 5. Semantic Name Checks
            lname = name.lower()

            # RT-DETR / YOLO specific exclusions for prediction heads
            # We must prevent LoRA from messing with final prediction layers (score/bbox heads)
            # because they are initialized with specific biases for Focal Loss.
            if LoRAConfigBuilder._PAT_BACKBONE_EXCLUDE.search(lname):
                # If we are strictly checking for head layers, we might want to skip them even if only_backbone=False
                # However, usually we want to LoRA the 'Detect' module's internal convs but NOT the final 1x1 convs.
                # For RT-DETR, the heads are explicit Linear layers.
                if "score_head" in lname or "bbox_head" in lname:
                     continue

            # Detect Head Special Handling
            # YOLO Detect head uses DFL (Distribution Focal Loss) which has a Conv2d layer that should NOT be trained or LoRA-ed usually.
            # DFL conv weight is fixed (non-trainable) in standard YOLO.
            if "dfl" in lname:
                 continue

            # MoE Check
            if not include_moe and LoRAConfigBuilder._PAT_MOE.search(lname):
                continue

            # Attention Check: also handle Conv2d-based attention.
            # YOLO12 AAttn uses Conv2d for qkv/proj/pe; the original logic only
            # filtered nn.Linear, leaking these layers into LoRA targets.
            if not include_attention:
                if is_linear and LoRAConfigBuilder._PAT_ATTN.search(lname):
                    continue
                # Conv2d form: match .attn.{qkv,proj,pe}
                if is_conv and LoRAConfigBuilder._PAT_AREA_ATTN.search(lname):
                    LOGGER.debug(f"[LoRA] Skip Area-Attention conv {name} (include_attention=False)")
                    continue
                # ABlock-internal MLP convs share the AAttn residual stream and
                # have no LayerNorm; LoRA injection here triggers gradient
                # explosion → NaN around epoch ~9–14 in YOLO12 training.
                if is_conv and LoRAConfigBuilder._PAT_AREA_ATTN_MLP.search(lname):
                    LOGGER.debug(
                        f"[LoRA] Skip ABlock-MLP conv {name} (include_attention=False)"
                    )
                    continue

            # RT-DETR MSDeformAttn geometry-sensitive layers.
            # Excluded unconditionally (even when include_attention=True) because
            # the instability source is not the attention softmax but the
            # sampling-grid initialization and zero-init softmax weights.
            # Users who really want to adapt these need to opt-in via explicit
            # target_modules and tune r<=4 + long alpha_warmup.
            if is_linear and LoRAConfigBuilder._PAT_MSDEFORM_RISKY.search(lname):
                LOGGER.debug(
                    f"[LoRA] Skip MSDeformAttn geometry-sensitive layer {name}"
                )
                continue

            targets.add(name)

        return sorted(list(targets))

    @staticmethod
    def calculate_auto_rank(model: nn.Module, targets: List[str], ratio: float) -> int:
        """
        Heuristically calculates the Rank based on the target parameter ratio.
        
        Approximation: LoRA_Params ≈ Num_Targets * Rank * (In_Ch + Out_Ch)
        """
        if not targets or ratio <= 0:
            return 16 

        total_params = sum(p.numel() for p in model.parameters())
        target_param_budget = total_params * ratio

        # Sample layers to calculate average channel dimensions (avoids iterating all)
        in_out_sums = []
        sample_size = min(len(targets), 50)
        step = max(1, len(targets) // sample_size)
        sampled_targets = targets[::step]
        
        modules_dict = dict(model.named_modules())
        
        for name in sampled_targets:
            m = modules_dict.get(name)
            if m:
                if isinstance(m, nn.Conv2d):
                    in_out_sums.append(m.in_channels + m.out_channels)
                elif isinstance(m, nn.Linear):
                    in_out_sums.append(m.in_features + m.out_features)

        if not in_out_sums:
            return 16

        avg_dim = sum(in_out_sums) / len(in_out_sums)
        
        # R = Target_Params / (Num_Targets * Avg_Dim)
        raw_r = target_param_budget / (len(targets) * avg_dim)
        
        # Clamp to range [4, 128] and round to nearest multiple of 4
        estimated_r = int(raw_r)
        estimated_r = max(4, min(128, estimated_r))
        estimated_r = (estimated_r // 4) * 4 or 4

        LOGGER.info(f"[LoRA] Auto-calculated Rank: {estimated_r} (Target ratio: {ratio:.1%})")
        return estimated_r

    @staticmethod
    def create_config(
        model: nn.Module,
        r: int = 16,
        alpha: Optional[int] = None,
        auto_r_ratio: float = 0.0,
        peft_type: str = "lora",
        **kwargs
    ) -> Union['LoraConfig', 'LoHaConfig', 'LoKrConfig',
               'IA3Config', 'OFTConfig', 'BOFTConfig', 'HRAConfig', None]:
        """Factory method: Generates a PEFT Config object."""
        
        targets = kwargs.get('target_modules')

        # 1. Auto-detection & Validation
        # Even if targets are provided explicitly (e.g. ['conv']), we MUST run auto_detect_targets
        # to filter out incompatible layers (e.g. grouped convs where r % groups != 0).
        # We pass the explicit targets as a filter to auto_detect_targets.
        
        # If targets is NOT None, we use it to restrict the search space of auto_detect_targets.
        # But `auto_detect_targets` doesn't inherently support a "whitelist" input, 
        # it scans the whole model.
        # So we modify the logic: Always run auto_detect, but if explicit targets are provided,
        # we check if the auto-detected target matches the explicit list (partial match).
        
        # Actually, simpler approach:
        # Pass the explicit targets (if any) as a "whitelist" to auto_detect_targets?
        # No, auto_detect_targets is designed to scan.
        
        # Better: Let's just always run auto_detect_targets.
        # If kwargs['target_modules'] was set, we need to handle it carefully.
        # If the user said "conv", they imply "all valid convs".
        # So we should clear 'target_modules' from kwargs before calling auto_detect,
        # but use the user's input as a guide.
        
        user_targets = kwargs.get('target_modules')
        
        # If user provided targets, we temporarily remove it to let auto_detect scan freely,
        # but we need to ensure auto_detect respects the USER's intent (e.g. only 'conv').
        # However, auto_detect has its own logic.
        
        # CORRECT APPROACH:
        # Run auto_detect_targets with all constraints.
        # If user_targets is provided (e.g. ['conv']), we treat it as an additional filter on the result.
        # Wait, if user provided ['conv'], auto_detect might return ['model.0.conv', ...].
        # We want the intersection of "valid layers" and "user request".
        
        # So:
        # 1. Run auto_detect to find ALL structurally valid layers (skipping bad grouped convs).
        # 2. If user provided targets, filter the valid list to only include those matching user's string.
        
        # To do this, we must ensure auto_detect doesn't get 'target_modules' in kwargs, 
        # otherwise it might be confused if it expects it to be None for auto-mode.
        
        detect_kwargs = kwargs.copy()
        if 'target_modules' in detect_kwargs:
            del detect_kwargs['target_modules']
            
        valid_targets = LoRAConfigBuilder.auto_detect_targets(model, r=r, **detect_kwargs)
        
        if user_targets:
            targets = _filter_target_modules(valid_targets, user_targets)
        else:
            targets = valid_targets

        if peft_type.lower() == "adalora":
            modules_dict = dict(model.named_modules())
            # Pre-check: AdaLoRA (as of PEFT 0.18) only supports nn.Linear.
            # For YOLO-family models, Conv2d dominates; using AdaLoRA effectively
            # disables LoRA on the whole backbone. Emit a loud warning instead of
            # silently degrading.
            conv_count = sum(1 for n in targets if isinstance(modules_dict.get(n), nn.Conv2d))
            linear_count = sum(1 for n in targets if isinstance(modules_dict.get(n), nn.Linear))
            total = conv_count + linear_count
            if total > 0 and conv_count / total > 0.5:
                LOGGER.warning(
                    f"[LoRA] ⚠️ AdaLoRA was requested but {conv_count}/{total} "
                    f"({100 * conv_count / total:.0f}%) target layers are Conv2d. "
                    f"AdaLoRA currently supports nn.Linear only; Conv layers will be "
                    f"silently skipped. Consider switching to `lora_type=lora` or "
                    f"`lora_use_dora=True` for Conv-heavy architectures like YOLO."
                )
            filtered_targets = [name for name in targets if isinstance(modules_dict.get(name), nn.Linear)]
            if targets and not filtered_targets:
                # P1 FIX: silently dropping all targets means AdaLoRA was
                # effectively disabled with no clear signal. Raise instead so
                # the user can pick a compatible variant explicitly.
                raise ValueError(
                    "AdaLoRA was requested but no nn.Linear targets remain after filtering. "
                    "AdaLoRA only supports Linear layers in PEFT 0.18; for Conv-dominant "
                    "architectures (YOLOv8/v11/v12, YOLOE) switch to lora_type=lora "
                    "(optionally with lora_use_dora=True). Detected target count: "
                    f"{len(targets)} (all non-Linear)."
                )
            targets = filtered_targets

        if not targets:
            return None

        # 2. Auto-Rank calculation
        if auto_r_ratio > 0 and r <= 0:
            r = LoRAConfigBuilder.calculate_auto_rank(model, targets, auto_r_ratio)

        # Default Alpha
        if alpha is None:
            alpha = 2 * r

        normalized_init = _validate_peft_init_compatibility(
            model,
            targets,
            peft_type=peft_type,
            init_lora_weights=kwargs.get("init_lora_weights", True),
        )

        target_modules_val = targets
        if user_targets and peft_type.lower() != "adalora":
            target_modules_val = _build_peft_exact_target_regex(targets)
            
        # 4. Common arguments
        common_kwargs = {
            "r": r,
            "target_modules": target_modules_val,
            "exclude_modules": kwargs.get('exclude_modules'), # FIX: Pass exclude_modules to LoraConfig!
            "task_type": None, # YOLO custom models usually do not require task_type
        }
        
        # 5. Dispatch based on PEFT type
        peft_type = peft_type.lower()
        
        if peft_type == "loha":
            # LoHa specific
            return LoHaConfig(
                alpha=alpha,
                module_dropout=kwargs.get('dropout', 0.0),
                **common_kwargs
            )
            
        elif peft_type == "lokr":
            # LoKr specific
            return LoKrConfig(
                alpha=alpha,
                module_dropout=kwargs.get('dropout', 0.0),
                **common_kwargs
            )

        elif peft_type == "adalora":
            total_step = resolve_adalora_total_step("adalora", kwargs.get("total_step"), 0)
            if total_step is None or total_step <= 0:
                raise ValueError("AdaLoRA requires `total_step > 0`. Pass lora_total_step or let trainer auto-populate it.")
            adalora_kwargs = {
                "lora_alpha": alpha,
                "lora_dropout": kwargs.get('dropout', 0.05),
                "bias": kwargs.get('bias', "none"),
                "use_dora": kwargs.get('use_dora', False),
                **common_kwargs,
            }
            if _supports_peft_kwarg(AdaLoraConfig, "use_rslora"):
                adalora_kwargs["use_rslora"] = kwargs.get('use_rslora', True)
            if _supports_peft_kwarg(AdaLoraConfig, "init_lora_weights"):
                adalora_kwargs["init_lora_weights"] = _normalize_lora_init(kwargs.get('init_lora_weights', True))

            adalora_kwargs["target_r"] = kwargs.get("target_r", r)
            adalora_kwargs["init_r"] = kwargs.get("init_r", max(r, kwargs.get("target_r", r)))
            adalora_kwargs["tinit"] = kwargs.get("tinit", 0)
            adalora_kwargs["tfinal"] = kwargs.get("tfinal", 0)
            adalora_kwargs["deltaT"] = kwargs.get("delta_t", kwargs.get("deltaT", 1))
            adalora_kwargs["beta1"] = kwargs.get("beta1", 0.85)
            adalora_kwargs["beta2"] = kwargs.get("beta2", 0.85)
            adalora_kwargs["orth_reg_weight"] = kwargs.get("orth_reg_weight", 0.5)
            adalora_kwargs["total_step"] = total_step

            return AdaLoraConfig(**adalora_kwargs)

        elif peft_type == "ia3":
            # IA3: only (IA)^3 scaling vectors — no rank, very few params.
            # Works on nn.Linear and nn.Conv2d (PEFT 0.18+).
            # For YOLO we treat every target as a feedforward module since
            # the backbone has no explicit FFN / attn split.
            return IA3Config(
                target_modules=common_kwargs["target_modules"],
                exclude_modules=common_kwargs.get("exclude_modules"),
                feedforward_modules=common_kwargs["target_modules"],
                init_ia3_weights=bool(kwargs.get("init_lora_weights", True)),
                task_type=common_kwargs.get("task_type"),
                modules_to_save=kwargs.get("modules_to_save"),
            )

        elif peft_type == "oft":
            # OFT: Orthogonal Fine-Tuning (block-diagonal Cayley rotations).
            # PEFT 0.18 requires exactly ONE of {r, oft_block_size}; its default
            # oft_block_size=32 collides with our common r. To keep the API
            # consistent, we ignore r entirely in OFT mode and drive capacity
            # through oft_block_size (user-provided or paper default 32).
            oft_block_size = int(kwargs.get("oft_block_size", 0) or 0) or 32
            return OFTConfig(
                target_modules=common_kwargs["target_modules"],
                exclude_modules=common_kwargs.get("exclude_modules"),
                oft_block_size=oft_block_size,
                module_dropout=kwargs.get("dropout", 0.0),
                bias=kwargs.get("bias", "none"),
                coft=bool(kwargs.get("oft_coft", False)),
                eps=float(kwargs.get("oft_eps", 6e-5)),
                block_share=bool(kwargs.get("oft_block_share", False)),
                task_type=common_kwargs.get("task_type"),
                modules_to_save=kwargs.get("modules_to_save"),
            )

        elif peft_type == "boft":
            # BOFT: Butterfly OFT (block-butterfly Cayley rotations).
            # PEFT BOFT requires both Conv in_features AND its effective kernel
            # dim (in_c * kH * kW) to be divisible by boft_block_size.
            # Narrow or 3x3 layers often break this, so we pre-filter targets
            # and auto-downgrade block_size when too many layers are dropped.
            #
            # IMPORTANT: common_kwargs["target_modules"] may be a regex string
            # (after _build_peft_exact_target_regex). We must use the original
            # `targets` list for divisibility checking, then rebuild the regex
            # after filtering.
            boft_block_size = int(kwargs.get("boft_block_size", 2))
            boft_target_list = targets  # always a list — before regex conversion
            modules_dict_boft = dict(model.named_modules())

            def _boft_layer_kdims(name: str, _md=modules_dict_boft):
                """Return (in_dim, kdim) for BOFT divisibility check; None if not applicable."""
                mod = _md.get(name)
                if mod is None:
                    return None
                if isinstance(mod, nn.Conv2d):
                    kdim = mod.in_channels * mod.kernel_size[0] * mod.kernel_size[1]
                    return (mod.in_channels, kdim)
                if isinstance(mod, nn.Linear):
                    return (mod.in_features, None)
                return None

            def _boft_ok(name: str, bs: int) -> bool:
                dims = _boft_layer_kdims(name)
                if dims is None:
                    return True  # unknown — let PEFT decide
                in_dim, kdim = dims
                if kdim is not None:  # Conv2d
                    return in_dim % bs == 0 and kdim % bs == 0
                return in_dim % bs == 0  # Linear

            def _find_compatible_block_size(target_list, preferred_bs):
                """Auto-downgrade block_size if too many targets are incompatible.

                Strategy:
                  - If >=50% targets work with preferred_bs, keep it (drop the rest).
                  - Otherwise try smaller candidates: preferred_bs//2, 3, 2, 1.
                    (3 is included because YOLO first-conv kdim=27 is divisible by 3
                    but not by 2 or 4.)
                """
                if not target_list:
                    return preferred_bs
                ok_count = sum(1 for t in target_list if _boft_ok(t, preferred_bs))
                total = len(target_list)
                if ok_count >= total * 0.5:
                    return preferred_bs  # majority compatible, just filter outliers
                # Try smaller block sizes in descending order.
                # Include 3 because YOLO 3x3 Conv with in_c=3 → kdim=27 → only 1/3/9/27 work.
                for candidate in sorted(
                    {preferred_bs // 2, 3, 2, 1} - {preferred_bs, 0},
                    reverse=True,
                ):
                    c_ok = sum(1 for t in target_list if _boft_ok(t, candidate))
                    if c_ok >= total * 0.5:
                        LOGGER.warning(
                            f"[LoRA] BOFT auto-downgraded boft_block_size "
                            f"{preferred_bs} → {candidate} (only {ok_count}/{total} "
                            f"layers compatible with {preferred_bs})."
                        )
                        return candidate
                return 1  # ultimate fallback — always works

            boft_block_size = _find_compatible_block_size(boft_target_list, boft_block_size)

            # Filter incompatible targets using the original list
            filtered = [t for t in boft_target_list if _boft_ok(t, boft_block_size)]
            dropped = len(boft_target_list) - len(filtered)
            if dropped:
                LOGGER.warning(
                    f"[LoRA] BOFT dropped {dropped} targets whose channels/kernel "
                    f"are not divisible by boft_block_size={boft_block_size}."
                )
            if not filtered:
                raise ValueError(
                    f"BOFT: no target layer is compatible with "
                    f"boft_block_size={boft_block_size}. Try boft_block_size=1."
                )
            # Rebuild regex from the filtered list (same format other PEFT types expect)
            target_modules_final = _build_peft_exact_target_regex(filtered) or filtered
            return BOFTConfig(
                target_modules=target_modules_final,
                exclude_modules=common_kwargs.get("exclude_modules"),
                boft_block_size=boft_block_size,
                boft_block_num=int(kwargs.get("boft_block_num", 0)),
                boft_n_butterfly_factor=int(kwargs.get("boft_n_butterfly_factor", 2)),
                boft_dropout=float(kwargs.get("dropout", 0.0)),
                bias=kwargs.get("bias", "none"),
                task_type=common_kwargs.get("task_type"),
                modules_to_save=kwargs.get("modules_to_save"),
            )

        elif peft_type == "hra":
            # HRA: High Rank Adaptation with Gram-Schmidt orthogonalization.
            # Supports both Conv2d and Linear. apply_GS enables Gram-Schmidt
            # for better numerical stability at higher ranks.
            return HRAConfig(
                r=r,
                apply_GS=bool(kwargs.get("hra_apply_gs", False)),
                target_modules=common_kwargs["target_modules"],
                exclude_modules=common_kwargs.get("exclude_modules"),
                init_weights=bool(kwargs.get("init_lora_weights", True)),
                bias=kwargs.get("bias", "none"),
                modules_to_save=kwargs.get("modules_to_save"),
            )

        else: # Default to LoRA (and DoRA)
            lora_kwargs = {
                "lora_alpha": alpha,
                "lora_dropout": kwargs.get('dropout', 0.05),
                "bias": kwargs.get('bias', "none"),
                "use_dora": kwargs.get('use_dora', False),
                **common_kwargs,
            }
            if _supports_peft_kwarg(PeftLoraConfig, "use_rslora"):
                lora_kwargs["use_rslora"] = kwargs.get('use_rslora', True)
            elif kwargs.get('use_rslora', True):
                LOGGER.warning("[LoRA] Installed PEFT does not support use_rslora; falling back to standard scaling.")

            if _supports_peft_kwarg(PeftLoraConfig, "init_lora_weights"):
                # FIX: Final guard against non-bool/non-str values that PEFT rejects
                if not isinstance(normalized_init, (bool, str)):
                    LOGGER.warning(f"[LoRA] init_lora_weights normalized to unexpected type {type(normalized_init).__name__}; falling back to True.")
                    normalized_init = True
                lora_kwargs["init_lora_weights"] = normalized_init
            else:
                requested_init = _normalize_lora_init(kwargs.get('init_lora_weights', True))
                # Only warn if user explicitly requested a non-default init mode
                if isinstance(requested_init, str) and requested_init not in {"default", "true", "false", "gaussian", "pissa", "olora"}:
                    LOGGER.warning(f"[LoRA] Installed PEFT does not support init_lora_weights='{requested_init}'; using PEFT defaults.")

            return PeftLoraConfig(**lora_kwargs)


# ============================================================================
# 4. Main Entry Point
# ============================================================================

__all__ = ["LoRAConfig", "LoRAConfigBuilder"]
