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


def _attach_planner_decision(
    model: nn.Module,
    config: "LoRAConfig",
    decision: "PlacementDecision",
    *,
    full_sft: bool = False,
) -> nn.Module:
    """Persist planner decisions on both adapted and full-SFT fallback paths."""
    payload = decision.to_dict()
    model.lora_planner_decision = payload
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module):
        inner.lora_planner_decision = payload

    metadata = dict(getattr(model, "lora_runtime_metadata", {}) or {})
    if full_sft:
        metadata = resolve_effective_lora_request(
            requested_backend=config.backend,
            effective_backend="full_sft",
            requested_variant=config.variant,
            effective_variant="full_sft",
            peft_type=config.peft_type,
            requested_init_lora_weights=config.init_lora_weights,
            effective_init_lora_weights=None,
            include_head=config.include_head,
            freeze_bn=bool(getattr(config, "freeze_bn", False)),
            target_modules=[],
            target_audit={},
        )
    metadata["planner_decision"] = payload
    model.lora_runtime_metadata = metadata
    return model


def build_lora_target_audit(
    valid_targets: Optional[List[str]] = None,
    selected_targets: Optional[List[str]] = None,
    requested_targets: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None,
    incompatible_layers: Optional[List[str]] = None,
    peft_type: str = "lora",
    rank: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a compact, serializable summary of target selection decisions."""
    valid = sorted(set(valid_targets or []))
    selected = sorted(set(selected_targets or []))
    requested = [str(x) for x in (requested_targets or [])]
    excluded = [str(x) for x in (exclude_modules or [])]
    incompatible = sorted(set(incompatible_layers or []))
    filtered_by_user = sorted(set(valid) - set(selected)) if requested else []

    return {
        "peft_type": str(peft_type),
        "rank": rank,
        "valid_count": len(valid),
        "selected_count": len(selected),
        "requested_targets": requested,
        "exclude_count": len(excluded),
        "incompatible_grouped_conv_count": len(incompatible),
        "filtered_by_user_count": len(filtered_by_user),
        "selected_sample": selected[:20],
        "incompatible_grouped_conv_sample": incompatible[:20],
        "filtered_by_user_sample": filtered_by_user[:20],
    }


def _log_lora_target_audit(audit: Dict[str, Any]) -> None:
    """Emit the most important target-selection audit counts."""
    if not audit:
        return
    LOGGER.info(
        "[LoRA] Target audit: "
        f"selected={audit.get('selected_count', 0)}/valid={audit.get('valid_count', 0)}, "
        f"user_filtered={audit.get('filtered_by_user_count', 0)}, "
        f"grouped_conv_excluded={audit.get('incompatible_grouped_conv_count', 0)}"
    )


def _validate_lora_runtime_model(
    model: nn.Module,
    expected_targets: Optional[List[str]] = None,
    context: str = "LoRA runtime",
) -> bool:
    """Validate that an adapted model still satisfies Ultralytics runtime assumptions."""
    errors = []
    backend = getattr(model, "lora_backend", None)
    inner = getattr(model, "model", None)

    if inner is None:
        errors.append("missing top-level `.model` attribute")

    if backend == "peft" and inner is not None:
        try:
            model_len = len(inner)
            if model_len <= 0:
                errors.append("PEFT proxy reports an empty model")
        except Exception as exc:
            errors.append(f"PEFT proxy does not support len(): {exc}")

        try:
            _ = inner[0]
        except Exception as exc:
            errors.append(f"PEFT proxy does not support index access: {exc}")

        try:
            _ = list(inner.children())
        except Exception as exc:
            errors.append(f"PEFT proxy does not expose children(): {exc}")

    stats = _compute_param_stats(model)
    if stats.adapter <= 0:
        errors.append("no adapter parameters were found after wrapping")

    if expected_targets is not None and len(expected_targets) == 0:
        errors.append("target detection produced zero selected modules")

    if errors:
        detail = "; ".join(errors)
        raise RuntimeError(f"{context} sanity check failed: {detail}")
    return True


def load_lora_compatible_state_dict(
    model: nn.Module,
    source_state: Dict[str, torch.Tensor],
    context: str = "LoRA checkpoint",
    adapter_only: bool = False,
) -> Dict[str, int]:
    """Load matching checkpoint tensors while making adapter mismatches explicit.

    Full-model checkpoints and LoRA checkpoints use different key spaces. During
    resume we load every tensor whose name and shape match the adapted model, but
    adapter topology mismatches are treated as hard errors because silently
    reinitializing a partially matching adapter is almost always a bad training
    resume.

    Args:
        adapter_only: If True, only load adapter parameters (lora_, hada_, etc.).
            Base model weights are never touched. This is critical for resume
            from EMA checkpoints because EMA may contain stale or corrupted
            base model weights that must not overwrite the freshly loaded
            pre-trained backbone.
    """
    target_state = model.state_dict()
    target_keys = set(target_state)
    source_keys = set(source_state)
    target_adapter_keys = {k for k in target_keys if _is_adapter_param(k)}
    source_adapter_keys = {k for k in source_keys if _is_adapter_param(k)}

    compatible = {}
    shape_mismatch = []
    for key, value in source_state.items():
        # adapter_only: skip any non-adapter parameter so we never overwrite
        # base model weights with potentially stale EMA averages.
        if adapter_only and not _is_adapter_param(key):
            continue
        target_value = target_state.get(key)
        if target_value is None:
            continue
        if hasattr(value, "shape") and hasattr(target_value, "shape") and tuple(value.shape) != tuple(target_value.shape):
            shape_mismatch.append(key)
            continue
        compatible[key] = value

    adapter_shape_mismatch = sorted(k for k in shape_mismatch if _is_adapter_param(k))
    missing_adapter = sorted(target_adapter_keys - source_adapter_keys)
    unexpected_adapter = sorted(source_adapter_keys - target_adapter_keys)

    if source_adapter_keys and (adapter_shape_mismatch or missing_adapter or unexpected_adapter):
        def _sample(items: List[str]) -> str:
            return ", ".join(items[:5]) + (" ..." if len(items) > 5 else "")

        parts = []
        if adapter_shape_mismatch:
            parts.append(f"shape mismatch: {_sample(adapter_shape_mismatch)}")
        if missing_adapter:
            parts.append(f"missing in checkpoint: {_sample(missing_adapter)}")
        if unexpected_adapter:
            parts.append(f"unexpected in checkpoint: {_sample(unexpected_adapter)}")
        raise RuntimeError(
            f"{context} is incompatible with the current LoRA adapter topology "
            f"({'; '.join(parts)}). Resume with the same lora_type, lora_r, "
            "lora_target_modules, and safety filters, or start a fresh run."
        )

    if target_adapter_keys and not source_adapter_keys:
        LOGGER.warning(
            f"[LoRA] {context} has no adapter tensors. Base weights will be restored where keys match; "
            "current adapters remain freshly initialized."
        )

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    adapter_loaded = sum(1 for key in compatible if _is_adapter_param(key))
    LOGGER.info(
        f"[LoRA] Restored {len(compatible)}/{len(target_state)} compatible tensors from {context} "
        f"(adapter {adapter_loaded}/{len(target_adapter_keys)})."
    )
    if shape_mismatch:
        LOGGER.debug(f"[LoRA] Skipped {len(shape_mismatch)} non-matching tensors while loading {context}.")
    return {
        "loaded": len(compatible),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "adapter_loaded": adapter_loaded,
        "adapter_total": len(target_adapter_keys),
    }



from .config import LoRAConfig, LoRAConfigBuilder
from .fallback import (
    FewShotLoRAConv,
    LoRADetectionModel,
    ManualLoRAConv,
    PeftProxy,
    _build_peft_exact_target_regex,
    _clear_lora_runtime_state,
    _collect_fallback_adapter_state,
    _filter_target_modules,
    _freeze_batchnorm_layers,
    _load_fallback_adapter_state,
    _merge_fallback_modules,
    _merge_manual_lora_conv,
    _validate_peft_init_compatibility,
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

    if str(getattr(config, "quantization", "none")).lower() in {"4bit", "8bit"}:
        raise RuntimeError(
            "QLoRA (4bit/8bit) is not supported for an already-built YOLO model. "
            "Load a bitsandbytes/Transformers-backed model before applying PEFT, "
            "or set lora_quantization=none."
        )

    # Inject the calibration data loader for gradient-sensitivity probing.
    # Stashed on the config via a private attribute so from_args (which only
    # reads known dataclass fields) does not swallow it.
    _sens_loader = kwargs.get("sensitivity_data_loader")
    if _sens_loader is not None:
        config._sensitivity_data_loader = _sens_loader

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

    # ------------------------------------------------------------------
    # PEFT Planner — architecture-conditioned placement decision (opt-in)
    # ------------------------------------------------------------------
    planner_decision = None
    if getattr(config, "planner_enabled", False) or getattr(config, "lora_planner_enabled", False):
        from .planner import PEFTPlanner, RefusalError, is_planner_enabled

        if is_planner_enabled(config):
            planner = PEFTPlanner()
            try:
                decision = planner.plan(model.model if hasattr(model, "model") else model, config)
            except RefusalError as exc:
                from .planner import PlacementDecision

                decision = PlacementDecision(
                    status="REFUSE",
                    refusal_reason=str(exc),
                    safety_overrides={"planner_refused": True},
                )

            planner_decision = decision

            if decision.status == "REFUSE":
                predicted = "unknown" if decision.predicted_delta is None else f"{decision.predicted_delta:.3f}"
                LOGGER.warning(
                    f"[Planner] REFUSE — {decision.refusal_reason} "
                    f"(predicted ΔmAP={predicted}). "
                    f"Falling back to full-model fine-tuning."
                )
                return _attach_planner_decision(model, config, decision, full_sft=True)

            if decision.status == "ADAPT":
                LOGGER.info("[Planner] ADAPT — applying recommended overrides.")
                if decision.recommended_variant:
                    config.peft_type = decision.recommended_variant
                    LOGGER.info(f"[Planner]   variant → {decision.recommended_variant}")
                if decision.recommended_rank is not None:
                    config.r = decision.recommended_rank
                    LOGGER.info(f"[Planner]   rank → {decision.recommended_rank}")
                for k, v in decision.safety_overrides.items():
                    if hasattr(config, k):
                        old = getattr(config, k)
                        setattr(config, k, v)
                        LOGGER.info(f"[Planner]   {k}: {old} → {v}")
                    else:
                        LOGGER.debug(f"[Planner]   skipping unknown override key '{k}'")

            if decision.status == "ACCEPT":
                predicted = "unknown" if decision.predicted_delta is None else f"{decision.predicted_delta:.3f}"
                LOGGER.info(f"[Planner] ACCEPT (predicted ΔmAP={predicted}).")
            planner_targets = list(decision.target_modules_hint or [])
            if not planner_targets:
                LOGGER.warning(
                    "[Planner] No safe target modules were selected; falling back to full-model fine-tuning."
                )
                return _attach_planner_decision(model, config, decision, full_sft=True)
            config.target_modules = planner_targets

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
        model = apply_manual_lora(model, config, include_head=config.include_head)
        if planner_decision is not None:
            _attach_planner_decision(model, config, planner_decision)
        return model

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
    # CRITICAL FIX: Trainer reads `self.args.lora_lr_mult` and
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
        "include_head": config.include_head,
        "only_backbone": config.only_backbone,
        "exclude_modules": config.exclude_modules,
        "last_n": config.last_n,
        "from_layer": config.from_layer,
        "to_layer": config.to_layer,
        "allow_depthwise": config.allow_depthwise,
        "kernels": config.kernels,
        "skip_stem": getattr(config, "skip_stem", True),  # Default True: skip un-normalized stem layers (prevents FP16 NaN)
        "min_channels": getattr(config, "min_channels", 0),
        "target_modules": config.target_modules, # This might be ['conv']
        "planner_enabled": False,
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
        # Architecture-aware sensitivity selection (opt-in). The data loader is
        # injected via apply_lora(..., sensitivity_data_loader=loader) and stashed
        # on the config object so we don't widen create_config's signature.
        "sensitivity_select": getattr(config, "sensitivity_select", False),
        "sensitivity_data_loader": getattr(config, "_sensitivity_data_loader", None),
        "sensitivity_num_batches": getattr(config, "sensitivity_num_batches", 4),
        "sensitivity_top_ratio": getattr(config, "sensitivity_top_ratio", 0.5),
        "sensitivity_beta": getattr(config, "sensitivity_beta", 1.0),
        "sensitivity_max_layers": getattr(config, "sensitivity_max_layers", None),
        "sensitivity_keep_risky": getattr(config, "sensitivity_keep_risky", False),
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
            # Quantization is a model-loading concern. This API receives an
            # already-built native YOLO graph, so importing a config alone
            # would leave Conv2d weights in FP32 while claiming QLoRA.
            raise RuntimeError(
                "QLoRA (4bit/8bit) is not supported for an already-built YOLO model. "
                "Load a bitsandbytes/Transformers-backed model before applying PEFT, "
                "or set lora_quantization=none."
            )

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
        # Sensitivity keys are consumed by create_config, not auto_detect_targets.
        for _sk in (
            "sensitivity_select", "sensitivity_data_loader", "sensitivity_num_batches",
            "sensitivity_top_ratio", "sensitivity_beta", "sensitivity_max_layers",
            "sensitivity_keep_risky",
        ):
            detect_params.pop(_sk, None)

        # Run auto-detect to get ALL structurally valid layers
        valid_targets = LoRAConfigBuilder.auto_detect_targets(model.model, **detect_params)
        if getattr(config, "sensitivity_select", False) and valid_targets:
            from .sensitivity import GradientSensitivitySelector
            report = GradientSensitivitySelector(
                model=model.model,
                data_loader=getattr(config, "_sensitivity_data_loader", None),
                num_batches=getattr(config, "sensitivity_num_batches", 4),
                top_ratio=getattr(config, "sensitivity_top_ratio", 0.5),
                max_layers=getattr(config, "sensitivity_max_layers", None),
            ).select_targets(valid_targets)
            valid_targets = report.selected_targets
            LOGGER.info(f"[LoRA] Gradient sensitivity selected {len(valid_targets)}/{len(report.layers)} targets.")
        
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

        target_audit = build_lora_target_audit(
            valid_targets=valid_targets,
            selected_targets=final_targets,
            requested_targets=user_targets,
            exclude_modules=builder_params.get("exclude_modules") or [],
            incompatible_layers=incompatible_layers,
            peft_type=config.peft_type,
            rank=config.r,
        )
        _log_lora_target_audit(target_audit)
        
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
        model.lora_target_audit = target_audit
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
            target_audit=target_audit,
            safety_profile="rtdetr_lora" if rtdetr_safety_changes else None,
            safety_overrides=rtdetr_safety_changes or None,
            planner_decision=planner_decision.to_dict() if planner_decision else None,
        )

        _validate_lora_runtime_model(model, expected_targets=final_targets, context="PEFT apply_lora")
        
        LOGGER.info(f"[LoRA] ✅ Successfully applied to {len(final_targets)} modules.")
        if final_targets:
             LOGGER.info(f"[LoRA] Targets sample: {list(final_targets)[:10]}")

    except Exception as e:
        LOGGER.error(f"[LoRA] ❌ Failed to apply PEFT wrapper: {e}")
        # Clear VRAM to prevent OOM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # FIX: Auto-degrade to manual fallback when PEFT setup fails and the
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
                model = apply_manual_lora(model, config, include_head=config.include_head)
                if planner_decision is not None:
                    _attach_planner_decision(model, config, planner_decision)
                return model
            except Exception as fb_err:
                LOGGER.error(f"[LoRA] Fallback path also failed: {fb_err}")
                raise e
        raise e

    # Unfreeze detection head (may be frozen by PEFT or random init)
    _unfreeze_detection_head(model)

    # FIX: Honor `freeze_bn` on the PEFT path as well. Previously the field
    # was only consumed by `apply_manual_lora` so passing `lora_freeze_bn=True`
    # with the PEFT backend silently had no effect.
    if bool(getattr(config, "freeze_bn", False)):
        _freeze_batchnorm_layers(getattr(model, "model", model))
        LOGGER.info("[LoRA] BatchNorm layers frozen (freeze_bn=True).")

    # 6. Gradient Checkpointing (VRAM Optimization) - Actually activate
    if config.gradient_checkpointing:
        from ultralytics.nn.modules.moe.utils import model_has_core_moe

        if model_has_core_moe(model):
            # MoE DDP training requires find_unused_parameters=True; combining
            # that with gradient checkpointing triggers
            # "parameter ... marked as ready twice" for unused LoRA adapters.
            LOGGER.warning(
                "[LoRA] Skipping gradient checkpointing on MoE models "
                "(incompatible with DDP find_unused_parameters=True). "
                "Set lora_gradient_checkpointing=False to silence this warning."
            )
        else:
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
from .planner import (
    ArchitectureFingerprint,
    PEFTPlanner,
    PEFTVariantProfile,
    PlacementDecision,
    RefusalError,
    is_planner_enabled,
)
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
    "build_lora_target_audit",
    "load_lora_compatible_state_dict",
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
    "_validate_lora_runtime_model",
    "_merge_manual_lora_conv",
    "_unfreeze_detection_head",
    "ArchitectureFingerprint",
    "PEFTPlanner",
    "PEFTVariantProfile",
    "PlacementDecision",
    "RefusalError",
    "is_planner_enabled",
]
