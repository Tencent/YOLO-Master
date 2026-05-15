# 🐧Please note that this file has been modified by Tencent on 2026/02/13. All Tencent Modifications are Copyright (C) 2026 Tencent.
import torch
import torch.nn as nn
import gc
import inspect
import json
import math
import types
from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any, Set, Tuple, TYPE_CHECKING
from pathlib import Path

import re

from ultralytics.utils import LOGGER
from ultralytics.nn.tasks import (
    DetectionModel, SegmentationModel, PoseModel, ClassificationModel, 
    OBBModel, RTDETRDetectionModel, WorldModel
)

# Attempt to import PEFT with graceful degradation
try:
    from peft import (
        LoraConfig, LoHaConfig, LoKrConfig, AdaLoraConfig,
        IA3Config, OFTConfig, BOFTConfig, HRAConfig,
        get_peft_model, PeftModel
    )
    PEFT_AVAILABLE = True
except ImportError:
    LoraConfig = LoHaConfig = LoKrConfig = AdaLoraConfig = None
    IA3Config = OFTConfig = BOFTConfig = HRAConfig = None
    get_peft_model = PeftModel = None
    PEFT_AVAILABLE = False
    
    # Define a dummy class to pass type checks when PEFT is missing
    class PeftModel:
        """Dummy class to prevent import errors when peft is not installed."""
        pass

# ============================================================================
# 0. Global Constants & Utilities
# ============================================================================

_REGEX_INT = re.compile(r"-?\d+")
_REGEX_SPLIT = re.compile(r"[,;]\s*")  # Supports comma or semicolon delimiters

# PEFT adapter parameter name prefixes for all supported variants.
# Used to identify adapter parameters in named_parameters() for stats and optimizer grouping.
_PEFT_ADAPTER_PREFIXES = ("lora_", "hada_", "lokr_", "oft_", "boft_", "ia3_", "hra_")


def _is_adapter_param(name: str) -> bool:
    """Check if a parameter name belongs to a PEFT adapter (any supported variant)."""
    return any(p in name for p in _PEFT_ADAPTER_PREFIXES)


def _effective_peft_variant(config: Any) -> str:
    """Return the adapter variant that is actually dispatched to PEFT."""
    peft_type = str(getattr(config, "peft_type", getattr(config, "variant", "lora"))).lower()
    if peft_type == "lora" and bool(getattr(config, "use_dora", False)):
        return "dora"
    return peft_type


@dataclass
class ParamStats:
    """Immutable parameter statistics for a model."""

    total: int = 0
    trainable: int = 0
    frozen: int = 0
    adapter: int = 0

    @property
    def trainable_pct(self) -> float:
        return 100 * self.trainable / self.total if self.total > 0 else 0.0

    @property
    def adapter_pct(self) -> float:
        return 100 * self.adapter / self.total if self.total > 0 else 0.0

    @property
    def base_total(self) -> int:
        return self.total - self.adapter


def _compute_param_stats(model: nn.Module) -> ParamStats:
    """Count total/trainable/frozen/adapter parameters in a single pass."""
    stats = ParamStats()
    for name, param in model.named_parameters():
        n = param.numel()
        stats.total += n
        if param.requires_grad:
            stats.trainable += n
        else:
            stats.frozen += n
        if _is_adapter_param(name):
            stats.adapter += n
    return stats


def _unfreeze_detection_head(model: nn.Module) -> int:
    """Unfreeze only real detection-head parameters for adapter fine-tuning.
    
    RT-DETR uses RTDETRDecoder with parameter names like decoder.layers, dec_score_head,
    dec_bbox_head, enc_score_head, enc_bbox_head, input_proj, query_pos_head, 
    denoising_class_embed, enc_output — none of which match the YOLO keywords.
    If the head stays frozen during LoRA training, mAP will be zero because the model
    cannot learn class/box predictions for the new dataset.
    
    Returns count of unfrozen params.
    """
    try:
        from ultralytics.nn.modules.head import Detect, RTDETRDecoder
        head_types = (Detect, RTDETRDecoder)
    except Exception:
        head_types = ()

    head_prefixes = []
    if head_types:
        for module_name, module in model.named_modules():
            if isinstance(module, head_types):
                head_prefixes.append(module_name)

    if not head_prefixes:
        LOGGER.debug("[LoRA] Detection head unfreeze skipped: no known head module found.")
        return 0

    head_unfrozen = 0
    for name, param in model.named_parameters():
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in head_prefixes):
            if not param.requires_grad:
                param.requires_grad = True
                head_unfrozen += param.numel()
    if head_unfrozen > 0:
        LOGGER.info(
            f"[LoRA] Unfrozen {head_unfrozen:,} detection head parameters "
            f"due to class-mismatch re-initialization."
        )
    return head_unfrozen


def _is_rtdetr_like_model(model: nn.Module) -> bool:
    """Return True for RT-DETR models, including wrapped/proxy variants."""
    if isinstance(model, RTDETRDetectionModel):
        return True
    for module in model.modules():
        cls_name = module.__class__.__name__
        if cls_name == "RTDETRDecoder" or "RTDETR" in cls_name:
            return True
    return False


def _fast_parse_int_list(value: Any) -> Optional[List[int]]:
    """
    High-performance integer list parser.
    
    Args:
        value: Input string, number, or list/tuple.
        
    Returns:
        Optional[List[int]]: Parsed list of integers, or None if invalid.
    """
    if value is None: 
        return None
    if isinstance(value, (list, tuple)): 
        return [int(x) for x in value]
    if isinstance(value, (int, float)): 
        return [int(value)]
    if isinstance(value, str):
        # Parse only if the string contains digits
        if _REGEX_INT.search(value):
            return [int(x) for x in _REGEX_INT.findall(value)]
    return None

def _fast_parse_str_list(value: Any) -> Optional[List[str]]:
    """
    High-performance string list parser with automatic deduplication and trimming.
    
    Args:
        value: Input string or list/tuple.
        
    Returns:
        Optional[List[str]]: Cleaned list of strings.
    """
    if value is None: 
        return None
    if isinstance(value, str):
        # Remove brackets and split
        value = value.strip('[]()')
        return list(set(x.strip() for x in _REGEX_SPLIT.split(value) if x.strip()))
    if isinstance(value, (list, tuple)):
        return list(set(str(x).strip() for x in value if str(x).strip()))
    return None


def _normalize_lora_init(value: Any) -> Union[str, bool]:
    """Normalize LoRA init mode names before passing them to PEFT.

    Returns:
        bool True/False for standard initialization, or str for special modes.
        PEFT 0.18.1 Conv2d only supports True/False/"gaussian", so we must
        preserve bool values and avoid converting them to strings.
    """
    # CRITICAL: Preserve bool values - PEFT Conv2d expects True/False, not "true"/"false"
    if isinstance(value, bool):
        return value
    if value is None:
        return True  # Default to standard init instead of "pissa" for compatibility
    if isinstance(value, str):
        normalized = value.strip().lower()
        # Convert string representations of bool
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        aliases = {
            "pi-ssa": "pissa",
            "o-lora": "olora",
        }
        return aliases.get(normalized, normalized or True)
    # FIX: YAML loaders may produce non-str/non-bool types (e.g. numpy bool).
    # Convert anything truthy/falsy to a native Python bool so PEFT never
    # receives an unexpected type.
    try:
        return bool(value)
    except Exception:
        return True


def _supports_peft_kwarg(config_cls: Any, kwarg: str) -> bool:
    """Check whether the installed PEFT config supports a given keyword argument."""
    if config_cls is None:
        return False
    try:
        return kwarg in inspect.signature(config_cls.__init__).parameters
    except (TypeError, ValueError):
        return False


def resolve_adalora_total_step(peft_type: str, total_step: Optional[int], iterations: int) -> Optional[int]:
    """Resolve AdaLoRA total_step, defaulting to trainer iterations when absent."""
    if str(peft_type).lower() != "adalora":
        return total_step
    if total_step is not None and total_step > 0:
        return total_step
    return iterations if iterations > 0 else None


def select_lora_backend(
    config: "LoRAConfig",
    peft_available: bool,
    supports_peft: bool,
    supports_fallback: bool,
) -> Dict[str, str]:
    """Resolve the effective backend for a LoRA request."""
    requested = str(getattr(config, "backend", "auto")).lower()
    if requested == "peft":
        if not (peft_available and supports_peft):
            raise ValueError("Requested lora_backend=peft but PEFT cannot satisfy this request.")
        return {"requested_backend": "peft", "effective_backend": "peft"}
    if requested == "fallback":
        if not supports_fallback:
            raise ValueError("Requested lora_backend=fallback but fallback backend cannot satisfy this request.")
        return {"requested_backend": "fallback", "effective_backend": "fallback"}
    if peft_available and supports_peft:
        return {"requested_backend": "auto", "effective_backend": "peft"}
    if not peft_available:
        fallback_hint = " Set lora_backend=fallback explicitly if you intentionally want the in-repo fallback backend." if supports_fallback else ""
        raise ValueError(
            "Auto LoRA backend requires PEFT. Install it with `pip install peft` instead of silently defaulting to fallback."
            f"{fallback_hint}"
        )
    if supports_fallback:
        raise ValueError(
            "Auto LoRA backend prefers PEFT and will not silently default to fallback for this request. "
            "Set lora_backend=fallback explicitly if you intentionally want the in-repo fallback backend."
        )
    raise ValueError("No LoRA backend can satisfy this request.")


def resolve_effective_lora_request(**kwargs) -> Dict[str, Any]:
    """Normalize runtime LoRA metadata into a serializable dictionary."""
    return dict(kwargs)



from .config import LoRAConfig, LoRAConfigBuilder
from .fallback import (
    FewShotLoRAConv,
    LoRADetectionModel,
    ManualLoRAConv,
    PeftProxy,
    _clear_lora_runtime_state,
    _collect_fallback_adapter_state,
    _load_fallback_adapter_state,
    _merge_fallback_modules,
    _merge_manual_lora_conv,
    _unfreeze_detection_head,
    _wrap_top_level_lora_model,
    apply_manual_lora,
    supports_fallback_request,
    supports_peft_request,
)

def _get_lora_runtime_value(
    args: Any,
    config: LoRAConfig,
    arg_name: str,
    config_name: Optional[str],
    kwargs: Dict[str, Any],
    default: Any = None,
) -> Any:
    """Resolve a runtime LoRA value from trainer args, config, kwargs, then default."""
    value = None
    if args is not None and not isinstance(args, LoRAConfig) and hasattr(args, arg_name):
        value = getattr(args, arg_name, None)
    if value is None and config_name:
        value = getattr(config, config_name, None)
    if value is None:
        value = kwargs.get(arg_name, default)
    return default if value is None else value


def _set_lora_runtime_value(
    args: Any,
    config: LoRAConfig,
    arg_name: str,
    config_name: Optional[str],
    kwargs: Dict[str, Any],
    value: Any,
) -> None:
    """Keep trainer args, LoRAConfig, and kwargs in sync after a safety override."""
    if config_name:
        try:
            setattr(config, config_name, value)
        except Exception:
            pass
    if args is not None and not isinstance(args, LoRAConfig):
        try:
            setattr(args, arg_name, value)
        except Exception:
            pass
    kwargs[arg_name] = value


def _apply_rtdetr_lora_safety(
    model: nn.Module,
    args: Any,
    config: LoRAConfig,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply conservative RT-DETR adapter stability defaults before training setup."""
    if not _is_rtdetr_like_model(model):
        return {}

    changes: Dict[str, Any] = {}
    LOGGER.warning(
        "[LoRA] RT-DETR adapter fine-tuning detected. Applying stability guards: "
        "keep AMP as configured, force lora_alpha_warmup>=3, cap lora_lr_mult<=1.0, "
        "enable safe attention projections, and keep MSDeformAttn geometry layers "
        "excluded from auto targets."
    )

    cur_warmup = _get_lora_runtime_value(
        args, config, "lora_alpha_warmup", "alpha_warmup", kwargs, default=0
    ) or 0
    if cur_warmup < 3:
        _set_lora_runtime_value(args, config, "lora_alpha_warmup", "alpha_warmup", kwargs, 3)
        changes["lora_alpha_warmup"] = {"from": cur_warmup, "to": 3}
        LOGGER.info(f"[LoRA] Force alpha_warmup = 3 for RT-DETR safety (was {cur_warmup}).")

    cur_lr_mult = _get_lora_runtime_value(
        args, config, "lora_lr_mult", "lr_mult", kwargs, default=2.0
    )
    if cur_lr_mult and cur_lr_mult > 1.0:
        _set_lora_runtime_value(args, config, "lora_lr_mult", "lr_mult", kwargs, 1.0)
        changes["lora_lr_mult"] = {"from": cur_lr_mult, "to": 1.0}
        LOGGER.info(f"[LoRA] Cap lora_lr_mult = 1.0 for RT-DETR safety (was {cur_lr_mult}).")

    if not bool(getattr(config, "include_attention", False)):
        _set_lora_runtime_value(args, config, "lora_include_attention", "include_attention", kwargs, True)
        changes["lora_include_attention"] = {"from": False, "to": True}
        LOGGER.info(
            "[LoRA] Enable safe attention projections for RT-DETR "
            "(self_attn.out_proj, cross_attn.value_proj/output_proj remain allowed; "
            "sampling_offsets/attention_weights stay excluded)."
        )

    if bool(getattr(config, "use_dora", False)):
        if bool(getattr(config, "allow_rtdetr_dora", False)):
            LOGGER.warning(
                "[LoRA] RT-DETR + DoRA is experimental and has shown early NaN collapse in local probes. "
                "Proceeding because lora_allow_rtdetr_dora=True."
            )
        else:
            _set_lora_runtime_value(args, config, "lora_use_dora", "use_dora", kwargs, False)
            changes["lora_use_dora"] = {"from": True, "to": False}
            LOGGER.warning(
                "[LoRA] RT-DETR + DoRA is unstable in local probes; auto-degrading to plain LoRA. "
                "Set lora_allow_rtdetr_dora=True to force the experimental path."
            )

    return changes


def apply_lora(
    model: "DetectionModel",
    args=None,
    **kwargs
) -> "DetectionModel":
    """
    Applies the LoRA strategy to an Ultralytics DetectionModel.

    Args:
        model (DetectionModel): The original model instance.
        args: Command line arguments object (optional).
        **kwargs: Configuration override dictionary.

    Returns:
        DetectionModel: The modified model instance with LoRA enabled 
                        (class swapped to LoRADetectionModel).
    """
    # 0. Prevent Re-application
    if getattr(model, "lora_enabled", False):
        LOGGER.warning("[LoRA] Model already has LoRA enabled. Skipping re-application.")
        return model

    # 1. Initialize Configuration
    if isinstance(args, LoRAConfig):
        config = args
    else:
        config = LoRAConfig.from_args(args, **kwargs)

    # Few-shot mode: auto-adjust hyperparameters for small datasets
    if config.few_shot_mode:
        LOGGER.info("[LoRA] 🎯 Few-shot mode enabled — applying adaptive configuration")
        if config.few_shot_adaptive_rank:
            # Increase rank for better expressiveness on limited data
            config.r = max(config.r, 32)
            config.alpha = max(config.alpha, 64)
            LOGGER.info(f"[LoRA]   Adaptive rank: r={config.r}, alpha={config.alpha}")
        # Reduce regularization to preserve signal
        config.dropout = min(config.dropout, 0.02)
        # Enable stronger LR multiplier for faster adaptation
        config.lr_mult = max(config.lr_mult, 3.0)
        LOGGER.info(f"[LoRA]   Dropout={config.dropout}, LR mult={config.lr_mult}")

    # Check if LoRA should be enabled.
    # BOFT/OFT/HRA/IA3 use block_size or other params instead of rank r;
    # they are valid even when r=0. OFT always falls back to block_size=32
    # when config value is 0, so peft_type="oft" alone is sufficient.
    _rankless_peft = str(config.peft_type).lower() in {"boft", "oft", "ia3", "hra"}
    if config.r <= 0 and config.auto_r_ratio <= 0 and not _rankless_peft:
        LOGGER.info("[LoRA] Disabled (r=0).")
        return model

    variant = _effective_peft_variant(config)
    if variant == "loha" and str(config.backend).lower() == "fallback":
        raise ValueError("Fallback variants other than LoRA remain experimental.")

    backend_decision = select_lora_backend(
        config,
        peft_available=PEFT_AVAILABLE,
        supports_peft=supports_peft_request(config),
        supports_fallback=supports_fallback_request(config),
    )
    if backend_decision["effective_backend"] == "fallback":
        return apply_manual_lora(model, config, include_head=config.include_head)

    # 2. Check Dependencies for the PEFT path
    if not PEFT_AVAILABLE:
        LOGGER.error("[LoRA] PEFT library not found. Please install via `pip install peft`.")
        return model

    # Check bitsandbytes for quantization
    if kwargs.get('lora_quantization') in ['4bit', '8bit']:
        try:
            import bitsandbytes as bnb
            LOGGER.info(f"[LoRA] bitsandbytes available for {kwargs.get('lora_quantization')} quantization.")
        except ImportError:
            LOGGER.error("[LoRA] bitsandbytes not found. Install via `pip install bitsandbytes`. Quantization disabled.")
            kwargs['lora_quantization'] = 'none'

    # 2.5 Auto-Disable MoE/Attention if not present in the model architecture
    # This prevents confusing logs claiming MoE is included when the model (e.g. YOLO11) has none.
    has_moe = False
    has_attn = False
    has_area_attn = False  # YOLO12 Area-Attention detection
    for name, _ in model.named_modules():
        if LoRAConfigBuilder._PAT_MOE.search(name):
            has_moe = True
        if LoRAConfigBuilder._PAT_ATTN.search(name):
            has_attn = True
        if LoRAConfigBuilder._PAT_AREA_ATTN.search(name):
            has_area_attn = True
        if has_moe and has_attn and has_area_attn:
            break
    
    if config.include_moe and not has_moe:
        config.include_moe = False
    
    if config.include_attention and not has_attn:
        config.include_attention = False

    rtdetr_safety_changes = _apply_rtdetr_lora_safety(model, args, config, kwargs)
    variant = _effective_peft_variant(config)

    # 2.6 YOLO12 Area-Attention safety guard.
    # AAttn uses Conv2d-based softmax attention; LoRA injection here easily causes
    # numerical collapse (symptom: loss drops to 0 and mAP/P/R become 0 mid-training).
    # Default behavior: drop attn.{qkv,proj,pe} *and* the ABlock-internal MLP conv
    # path (which sits on the same residual stream and has no LayerNorm), plus
    # force alpha warmup when enabled.
    #
    # CRITICAL FIX (P0): Trainer reads `self.args.lora_lr_mult` and
    # `self.args.lora_alpha_warmup` directly when building the optimizer and
    # scheduling alpha warmup. Writing the cap to `kwargs` or `config` alone
    # has *no effect* on the actual training run. We therefore mutate `args`
    # in place (when provided) and also keep `config`/`kwargs` consistent
    # for downstream consumers.
    if has_area_attn:
        LOGGER.warning(
            "[LoRA] YOLO12/A2C2f Area-Attention detected. "
            "Applying safety guards: (1) exclude attn.{qkv,proj,pe} and "
            "ABlock-internal mlp Conv2d from LoRA targets, "
            "(2) force alpha_warmup>=3 epochs if unset, (3) cap lora_lr_mult<=1.0."
        )
        # Resolve current values from args (preferred), then config, then kwargs.
        cur_warmup = _get_lora_runtime_value(
            args, config, "lora_alpha_warmup", "alpha_warmup", kwargs, default=0
        ) or 0
        if cur_warmup < 3:
            _set_lora_runtime_value(args, config, "lora_alpha_warmup", "alpha_warmup", kwargs, 3)
            LOGGER.info(f"[LoRA] Force alpha_warmup = 3 for YOLO12 safety (was {cur_warmup}).")
        # Lower LR multiplier (attention LoRA layers are very LR-sensitive).
        cur_lr_mult = _get_lora_runtime_value(
            args, config, "lora_lr_mult", "lr_mult", kwargs, default=2.0
        )
        if cur_lr_mult and cur_lr_mult > 1.0:
            _set_lora_runtime_value(args, config, "lora_lr_mult", "lr_mult", kwargs, 1.0)
            LOGGER.info(f"[LoRA] Cap lora_lr_mult = 1.0 for YOLO12 safety (was {cur_lr_mult}).")

    # 3. Logging
    LOGGER.info("-" * 60)
    LOGGER.info(f"🚀 Initializing LoRA Strategy")
    for k, v in config.__dict__.items():
        if k not in ['target_modules', 'exclude_modules'] and v is not None:
            LOGGER.info(f"  - {k:<22}: {v}")
    
    # 4. Prepare Builder Parameters
    # CRITICAL FIX: If target_modules is explicitly provided (e.g. ['conv']), we MUST still run it through
    # auto_detect_targets to filter out incompatible layers (like grouped convs).
    # Otherwise, PEFT will try to apply LoRA to ALL layers matching 'conv', causing crashes.
    
    # If target_modules is provided, we treat it as a broad filter for auto_detect
    # forcing auto_detect to only consider layers containing these strings/types
    
    # However, auto_detect_targets logic is: if target_modules is None, it scans everything.
    # If we pass target_modules to it, it doesn't currently use it as a base filter.
    # So we should modify how we call it.
    
    # Actually, let's look at create_config. It calls auto_detect_targets ONLY IF target_modules is None.
    # We need to change this behavior. We want auto_detect_targets to ALWAYS run validation/filtering,
    # even if the user provided a list.
    
    builder_params = {
        "r": config.r,
        "alpha": config.alpha,
        "dropout": config.dropout,
        "bias": config.bias,
        "include_moe": config.include_moe,
        "include_attention": config.include_attention,
        "only_backbone": config.only_backbone,
        "exclude_modules": config.exclude_modules,
        "last_n": config.last_n,
        "from_layer": config.from_layer,
        "to_layer": config.to_layer,
        "allow_depthwise": config.allow_depthwise,
        "kernels": config.kernels,
        "skip_stem": getattr(config, "skip_stem", False),
        "min_channels": getattr(config, "min_channels", 0),
        "target_modules": config.target_modules, # This might be ['conv']
        "gradient_checkpointing": config.gradient_checkpointing,
        "auto_r_ratio": config.auto_r_ratio,
        "use_dora": config.use_dora,
        "use_rslora": config.use_rslora,
        "init_lora_weights": config.init_lora_weights,
        "peft_type": config.peft_type,
        "only_3x3": config.only_3x3,
        "oft_block_size": getattr(config, "oft_block_size", 0),
        "oft_coft": getattr(config, "oft_coft", False),
        "oft_eps": getattr(config, "oft_eps", 6e-5),
        "oft_block_share": getattr(config, "oft_block_share", False),
        "boft_block_size": getattr(config, "boft_block_size", 4),
        "boft_block_num": getattr(config, "boft_block_num", 0),
        "boft_n_butterfly_factor": getattr(config, "boft_n_butterfly_factor", 2),
        "hra_apply_gs": getattr(config, "hra_apply_gs", False),
        "target_r": config.target_r,
        "init_r": config.init_r,
        "tinit": config.tinit,
        "tfinal": config.tfinal,
        "delta_t": config.delta_t,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "orth_reg_weight": config.orth_reg_weight,
        "total_step": config.total_step,
    }

    # Identify incompatible layers to explicitly exclude
    # This acts as a safety net against regex failures or PEFT behavior quirks
    incompatible_layers = []
    # Note: We scan model.model which is the nn.Sequential
    for name, module in model.model.named_modules():
         if isinstance(module, nn.Conv2d) and module.groups > 1:
              if config.r > 0 and config.r % module.groups != 0:
                   incompatible_layers.append(name)
    
    if incompatible_layers:
         current_exclude = builder_params.get("exclude_modules") or []
         if isinstance(current_exclude, str):
              current_exclude = [current_exclude] # Should be handled by parser but just in case
         
         # Add variations to ensure PEFT catches it regardless of prefixing
         variations = []
         for name in incompatible_layers:
             variations.append(name)
             variations.append(f"model.{name}")
             variations.append(f"model.model.{name}")
         
         # Avoid duplicates
         final_exclude = list(set(current_exclude + variations))
         builder_params["exclude_modules"] = final_exclude
         LOGGER.info(f"[LoRA] 🛡️ Automatically excluded {len(incompatible_layers)} incompatible grouped conv layers (r={config.r}).")
         # LOGGER.info(f"DEBUG: Excluded layers sample: {final_exclude[:5]}")

    # 5. Application Process
    try:
        # Handle Quantization (QLoRA)
        if config.quantization in ['4bit', '8bit']:
            try:
                from transformers import BitsAndBytesConfig
                LOGGER.warning("[LoRA] QLoRA (4-bit/8-bit) for YOLO Conv2d layers is experimental and depends on bitsandbytes support.")
                pass 
            except ImportError:
                LOGGER.warning("[LoRA] transformers not found. BitsAndBytesConfig skipped.")

        # Create config using model.model (nn.Sequential)
        
        # 5.1. Target Module Intersection Logic
        # We need to refine 'target_modules' in builder_params.
        # If the user provided explicit targets (e.g. ['conv']), we must still run auto-detect
        # to filter out incompatible layers (grouped convs).
        
        user_targets = builder_params.get("target_modules")
        
        # Temporarily remove targets to let auto-detect scan everything for validity
        detect_params = builder_params.copy()
        if "target_modules" in detect_params:
            del detect_params["target_modules"]
            
        # Run auto-detect to get ALL structurally valid layers
        valid_targets = LoRAConfigBuilder.auto_detect_targets(model.model, **detect_params)
        
        final_targets = []
        if user_targets:
            final_targets = _filter_target_modules(valid_targets, user_targets)
            if not final_targets:
                LOGGER.warning(f"[LoRA] ⚠️ User requested targets {user_targets}, but they were all filtered out (e.g. incompatible grouped convs).")
        else:
            # No user preference, use all valid layers
            final_targets = valid_targets
            
        if final_targets:
            builder_params["target_modules"] = final_targets
        else:
            builder_params["target_modules"] = None
        
        # DEBUG: Print final targets passed to PEFT
        LOGGER.info(f"[LoRA] Final Targets Passed to PEFT (List Length: {len(final_targets) if final_targets else 0})")
        
        # Remove debug logs about regex
        
        peft_config = LoRAConfigBuilder.create_config(model.model, **builder_params)
        
        if peft_config is None:
            LOGGER.warning("[LoRA] ⚠️ No valid target modules found based on filters. LoRA skipped.")
            return model

        # Get the wrapped model
        # Note: get_peft_model wraps model.model inside a PeftModel
        peft_model_wrapper = get_peft_model(model.model, peft_config)

        # [CORE MAGIC] Swap PeftModel class with PeftProxy
        # This makes the wrapper behave exactly like nn.Sequential (supports indexing, slicing, etc.)
        peft_model_wrapper.__class__ = PeftProxy
        
        # Replace the internal structure of the original model
        model.model = peft_model_wrapper

        # [CORE MAGIC] Swap the top-level DetectionModel class to a LoRA-aware wrapper.
        _wrap_top_level_lora_model(model, config)
        model.lora_backend = "peft"
        model.lora_variant = variant
        model.lora_include_head = config.include_head
        model.lora_freeze_bn = bool(getattr(config, "freeze_bn", False))
        model.lora_target_modules = sorted(final_targets)
        model.lora_runtime_metadata = resolve_effective_lora_request(
            requested_backend=config.backend,
            effective_backend="peft",
            requested_variant=config.variant,
            effective_variant=variant,
            peft_type=config.peft_type,
            requested_init_lora_weights=config.init_lora_weights,
            effective_init_lora_weights=config.init_lora_weights,
            include_head=config.include_head,
            freeze_bn=bool(getattr(config, "freeze_bn", False)),
            target_modules=model.lora_target_modules,
            safety_profile="rtdetr_lora" if rtdetr_safety_changes else None,
            safety_overrides=rtdetr_safety_changes or None,
        )
        
        LOGGER.info(f"[LoRA] ✅ Successfully applied to {len(final_targets)} modules.")
        if final_targets:
             LOGGER.info(f"[LoRA] Targets sample: {list(final_targets)[:10]}")

    except Exception as e:
        LOGGER.error(f"[LoRA] ❌ Failed to apply PEFT wrapper: {e}")
        # Clear VRAM to prevent OOM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # P0 FIX: Auto-degrade to manual fallback when PEFT setup fails and the
        # request is in principle representable in the in-repo fallback (plain
        # LoRA, r > 0). This avoids hard-killing training runs over recoverable
        # PEFT-side incompatibilities (e.g. unsupported init mode for a single
        # Conv2d target). Users can still force PEFT-only by setting
        # `lora_backend=peft` (which makes the auto fallback path raise above).
        is_auto_backend = str(getattr(config, "backend", "auto")).lower() == "auto"
        can_fallback = supports_fallback_request(config)
        if is_auto_backend and can_fallback:
            LOGGER.warning(
                "[LoRA] PEFT path failed; auto-degrading to in-repo fallback "
                "manual LoRA backend (set lora_backend=peft to disable this fallback)."
            )
            try:
                return apply_manual_lora(model, config, include_head=config.include_head)
            except Exception as fb_err:
                LOGGER.error(f"[LoRA] Fallback path also failed: {fb_err}")
                raise e
        raise e

    # Unfreeze detection head (may be frozen by PEFT or random init)
    _unfreeze_detection_head(model)

    # P0 FIX: Honor `freeze_bn` on the PEFT path as well. Previously the field
    # was only consumed by `apply_manual_lora` so passing `lora_freeze_bn=True`
    # with the PEFT backend silently had no effect.
    if bool(getattr(config, "freeze_bn", False)):
        _freeze_batchnorm_layers(getattr(model, "model", model))
        LOGGER.info("[LoRA] BatchNorm layers frozen (freeze_bn=True).")

    # 6. Gradient Checkpointing (VRAM Optimization) - Actually activate
    if config.gradient_checkpointing:
        from torch.utils.checkpoint import checkpoint
        
        # Enable the flag on the model for tasks.py to consume
        if hasattr(model, "model"):
            model.model.use_gradient_checkpointing = True
            if hasattr(model.model, "model"):
                model.model.model.use_gradient_checkpointing = True
                # Patch C3k2 / Conv layers to use checkpointing if they support it
                _activate_gradient_checkpointing(model.model.model)
        
        # Set directly on the top-level model (LoRADetectionModel)
        model.use_gradient_checkpointing = True
        LOGGER.info("[LoRA] ✅ Gradient checkpointing activated (reduces VRAM by ~30-50%).")

    # 6.5 MPS Compatibility Check & Warning
    device_type = None
    try:
        for p in model.parameters():
            if p.device.type != 'cpu':
                device_type = p.device.type
                break
    except Exception:
        pass
    
    if device_type == 'mps':
        LOGGER.info("[LoRA] ⚡ MPS backend detected. LoRA inference will use Metal acceleration.")
        LOGGER.info("[LoRA]   Tip: Use lora_r=4~16 on MPS to avoid OOM. Larger ranks increase memory linearly.")

    # 7. Print Statistics
    _print_param_stats(model, peft_type=str(config.peft_type))

    # 8. Performance warning for slow PEFT variants
    _warn_slow_peft_variant(str(config.peft_type))

    return model


def _warn_slow_peft_variant(peft_type: str):
    """Warn about PEFT variants with known performance issues."""
    peft_type = peft_type.lower()
    if peft_type == "hra":
        LOGGER.warning(
            "[LoRA] ⚠️  HRA uses Gram-Schmidt orthogonalization in Python loops during forward. "
            "Training speed may be 3-10x slower than LoRA. Consider LoRA/LoHa for faster training."
        )
    elif peft_type == "oft":
        LOGGER.warning(
            "[LoRA] ⚠️  OFT uses dense orthogonal rotations (high activation memory). "
            "If OOM occurs, reduce batch size or use LoRA/LoHa/LoKr instead."
        )
    elif peft_type == "boft":
        LOGGER.warning(
            "[LoRA] ⚠️  BOFT relies on butterfly orthogonal factors and a CUDA "
            "kernel JIT-compiled at first forward (requires g++/cc1plus). "
            "First-iteration latency can be high; if cc1plus is missing the "
            "kernel falls back to butterfly_factor=1 with reduced expressivity."
        )
    elif peft_type == "adalora":
        LOGGER.warning(
            "[LoRA] ⚠️  AdaLoRA needs `total_step` set to the total number of "
            "training iterations for the rank-budget schedule to work correctly. "
            "We auto-resolve it from trainer iterations, but verify the number "
            "in the log if mAP plateaus early."
        )


def _activate_gradient_checkpointing(module: nn.Module):
    """Recursively enable gradient checkpointing for supported modules."""
    from torch.utils.checkpoint import checkpoint_sequential
    
    for name, child in module.named_children():
        # For C3k2-like blocks, we can wrap their forward with checkpoint
        child_name = type(child).__name__.lower()
        
        if any(kw in child_name for kw in ('c3k', 'c2f', 'bottleneck', 'conv', 'block')):
            if not getattr(child, 'use_gradient_checkpointing', False):
                child.use_gradient_checkpointing = True
        
        # Recurse into children
        if len(list(child.children())) > 0:
            _activate_gradient_checkpointing(child)


# ============================================================================
# 5. Utilities
# ============================================================================

def _get_mps_memory() -> tuple:
    """Get precise MPS memory info using system calls."""
    if not hasattr(torch, 'mps') or not torch.backends.mps.is_available():
        return None, None
    
    try:
        import subprocess
        result = subprocess.run(
            ['vm_stat'], capture_output=True, text=True, timeout=5
        )
        
        page_size = 4096  # macOS page size
        
        # Parse "Pages active"
        for line in result.stdout.split('\n'):
            if 'Pages active:' in line:
                parts = line.strip().split(':')
                if len(parts) >= 2:
                    val = int(parts[1].replace('.', '').strip())
                    return val * page_size, None
    except Exception:
        pass
    
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.used, vm.total
    except Exception:
        pass
    
    return None, None


def _print_param_stats(model: nn.Module, peft_type: str = ""):
    """Prints detailed parameter statistics."""
    s = _compute_param_stats(model)

    LOGGER.info(
        f"[LoRA] 📊 Stats: "
        f"Trainable: {s.trainable:,} ({s.trainable_pct:.3f}%) | "
        f"Frozen Base: {s.frozen:,} | "
        f"Adapter Params: {s.adapter:,} ({s.adapter_pct:.3f}%) | "
        f"Base Total: {s.base_total:,}"
    )

    if s.trainable == s.total:
        LOGGER.warning(
            "[LoRA] ⚠️  ALL parameters are trainable. Check if LoRA adapters were applied correctly."
        )

    # Memory monitoring - GPU/CUDA
    if torch.cuda.is_available():
        try:
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            LOGGER.info(f"[LoRA] 💾 CUDA Memory: Allocated={mem_allocated:.2f}GB, Reserved={mem_reserved:.2f}GB, Total={total_mem:.1f}GB")
        except Exception:
            pass
    # Memory monitoring - MPS (macOS)
    elif torch.backends.mps.is_available():
        used, total = _get_mps_memory()
        if used is not None:
            used_gb = used / 1024**3
            total_gb = total / 1024**3 if total else None
            total_str = f"/ {total_gb:.1f}" if total_gb else ""
            LOGGER.info(f"[LoRA] 💾 MPS Memory: ~{used_gb:.2f}{total_str} GB")
        else:
            LOGGER.info("[LoRA] 💾 Using MPS backend")


def get_lora_param_groups(
    model: nn.Module,
    weight_decay: float = 0.0,
    lora_weight_decay: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Split trainable parameters into LoRA and non-LoRA groups with independent weight decay.

    This is useful for external training loops that want to keep LoRA adapters on zero
    weight decay while preserving the caller's decay for the rest of the trainable model.
    """
    lora_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_adapter_param(name):
            lora_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if lora_params:
        param_groups.append({"params": lora_params, "weight_decay": lora_weight_decay})
    if other_params:
        param_groups.append({"params": other_params, "weight_decay": weight_decay})
    return param_groups



from .io import load_lora_adapters, merge_lora_weights, save_lora_adapters
from .training import LoraTrainingStrategy, get_lora_training_stats, suggest_lora_config_for_dataset

__all__ = [
    "PEFT_AVAILABLE",
    "PeftModel",
    "PeftProxy",
    "LoRAConfig",
    "LoRAConfigBuilder",
    "LoRADetectionModel",
    "FewShotLoRAConv",
    "ManualLoRAConv",
    "apply_lora",
    "get_lora_param_groups",
    "resolve_adalora_total_step",
    "resolve_effective_lora_request",
    "select_lora_backend",
    "save_lora_adapters",
    "load_lora_adapters",
    "merge_lora_weights",
    "LoraTrainingStrategy",
    "get_lora_training_stats",
    "suggest_lora_config_for_dataset",
    "supports_peft_request",
    "supports_fallback_request",
    "_apply_rtdetr_lora_safety",
    "_get_mps_memory",
    "_is_adapter_param",
    "_merge_manual_lora_conv",
    "_unfreeze_detection_head",
]
