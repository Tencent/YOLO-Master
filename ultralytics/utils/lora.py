# 🐧Please note that this file has been modified by Tencent on 2026/02/13. All Tencent Modifications are Copyright (C) 2026 Tencent.
import torch
import torch.nn as nn
import gc
import inspect
import json
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
        get_peft_model, PeftModel
    )
    PEFT_AVAILABLE = True
except ImportError:
    LoraConfig = LoHaConfig = LoKrConfig = AdaLoraConfig = get_peft_model = PeftModel = None
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


class ManualLoRAConv(nn.Module):
    """Minimal manual LoRA wrapper for Conv2d fallback paths.

    Supports both dense Conv2d (groups=1) and grouped convolutions
    (groups>1, including depthwise where groups == in_channels == out_channels).
    For grouped convs we allocate one (A, B) pair per group, so the rank `r`
    MUST be divisible by `groups`.
    """

    def __init__(self, conv: nn.Conv2d, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.conv = conv
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(r, 1)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else None

        groups = conv.groups
        if groups > 1 and (r % groups != 0):
            raise ValueError(
                f"ManualLoRAConv: rank r={r} must be a multiple of groups={groups} "
                f"(layer has {conv.in_channels} in / {conv.out_channels} out channels)."
            )
        # Per-group rank. For dense conv (groups=1), r_per_group == r.
        self.groups = groups
        self.r_per_group = r // max(groups, 1)

        # Input patch dimension per group: (in_channels/groups) * k_h * k_w
        in_per_group = (conv.in_channels // groups) * conv.kernel_size[0] * conv.kernel_size[1]
        out_per_group = conv.out_channels // groups
        factory_kwargs = {"device": conv.weight.device, "dtype": conv.weight.dtype}
        # Shape: (groups, in_per_group, r_per_group) and (groups, out_per_group, r_per_group)
        self.lora_A = nn.Parameter(torch.zeros(groups, in_per_group, self.r_per_group, **factory_kwargs))
        self.lora_B = nn.Parameter(torch.zeros(groups, out_per_group, self.r_per_group, **factory_kwargs))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.01)
        nn.init.zeros_(self.lora_B)  # Standard LoRA init: B=0 so initial adapter output is 0

        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.conv(x)

        k_h, k_w = self.conv.kernel_size
        # Unfold yields (B, C_in * k_h * k_w, L) where L = out_h * out_w
        x_unfold = nn.functional.unfold(
            x, (k_h, k_w), padding=self.conv.padding, stride=self.conv.stride, dilation=self.conv.dilation
        )
        if self.lora_dropout is not None:
            x_unfold = self.lora_dropout(x_unfold)

        B_size, _, L = x_unfold.shape
        out_h, out_w = out.shape[2], out.shape[3]

        # Older fallback checkpoints serialized dense adapters without `groups`
        # metadata and with 2D LoRA matrices. Keep them loadable for validation.
        groups = getattr(self, "groups", getattr(self.conv, "groups", 1))
        if groups == 1 and self.lora_A.dim() == 2 and self.lora_B.dim() == 2:
            x_unfold = x_unfold.transpose(1, 2)
            lora = x_unfold @ self.lora_A
            lora = lora @ self.lora_B.t()
            lora = lora * self.scaling
            lora = lora.transpose(1, 2).reshape(B_size, self.conv.out_channels, out_h, out_w)
            return out + lora

        if groups == 1:
            # Dense conv: single (A, B) pair.
            # x_unfold: (B, in_per_group, L) -> transpose to (B, L, in_per_group)
            x_unfold = x_unfold.transpose(1, 2)
            lora = x_unfold @ self.lora_A[0]            # (B, L, r)
            lora = lora @ self.lora_B[0].t()            # (B, L, out_per_group)
            lora = lora * self.scaling
            lora = lora.transpose(1, 2).reshape(B_size, self.conv.out_channels, out_h, out_w)
            return out + lora

        # Grouped conv: split x_unfold per group and apply (A_g, B_g) pair.
        in_per_group = x_unfold.shape[1] // groups
        # Reshape to (B, groups, in_per_group, L) -> (groups, B, L, in_per_group)
        x_grouped = x_unfold.view(B_size, groups, in_per_group, L).permute(1, 0, 3, 2)
        # Batched matmul: (groups, B*L, in_per_group) @ (groups, in_per_group, r_per_group)
        lora = torch.bmm(
            x_grouped.reshape(groups, B_size * L, in_per_group),
            self.lora_A,
        )  # (groups, B*L, r_per_group)
        lora = torch.bmm(lora, self.lora_B.transpose(1, 2))  # (groups, B*L, out_per_group)
        lora = lora * self.scaling
        # Re-assemble: (groups, B, L, out_per_group) -> (B, out_channels, L) -> (B, C_out, H, W)
        out_per_group = self.lora_B.shape[1]
        lora = lora.view(groups, B_size, L, out_per_group).permute(1, 0, 3, 2)
        lora = lora.reshape(B_size, groups * out_per_group, L)
        lora = lora.view(B_size, self.conv.out_channels, out_h, out_w)
        return out + lora


def supports_peft_request(config: "LoRAConfig") -> bool:
    """Return whether the PEFT backend can satisfy the requested variant in principle."""
    variant = str(getattr(config, "variant", getattr(config, "peft_type", "lora"))).lower()
    if variant == "dora":
        return bool(PEFT_AVAILABLE and getattr(config, "use_dora", False))
    return bool(PEFT_AVAILABLE and variant in {"lora", "adalora", "loha", "lokr"})


def supports_fallback_request(config: "LoRAConfig") -> bool:
    """Return whether the in-repo fallback backend can satisfy the requested variant."""
    return getattr(config, "r", 0) > 0 and str(getattr(config, "variant", "lora")).lower() == "lora"


def _is_head_like_module(module_name: str) -> bool:
    """Heuristic to identify detection head-like modules for fallback targeting."""
    lowered = module_name.lower()
    return any(token in lowered for token in ("head", "detect", "dfl"))


def _freeze_batchnorm_layers(module: nn.Module) -> None:
    """Freeze BatchNorm layers for LoRA fine-tuning when requested."""
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()
            for param in child.parameters():
                param.requires_grad = False


def _matches_target_modules(module_name: str, target_modules: Optional[List[str]]) -> bool:
    """Return whether a module name matches the user's explicit target module request."""
    if not target_modules:
        return True
    normalized_module = str(module_name).strip().strip(".")
    while normalized_module.startswith("model."):
        normalized_module = normalized_module[len("model."):]
    for requested in target_modules:
        normalized_requested = str(requested).strip().strip(".")
        while normalized_requested.startswith("model."):
            normalized_requested = normalized_requested[len("model."):]
        if not normalized_requested:
            continue
        if normalized_module == normalized_requested:
            return True
        # Numeric-prefix paths like `0.conv` are treated as exact paths to avoid
        # matching nested modules such as `23.cv2.0.0.conv`.
        first_segment = normalized_requested.split(".", 1)[0]
        if first_segment.isdigit():
            continue
        if normalized_module.endswith(f".{normalized_requested}"):
            return True
    return False


def _filter_target_modules(candidate_modules: List[str], requested_targets: Optional[List[str]]) -> List[str]:
    """Filter detected module names using the same boundary-safe explicit target matching rules."""
    if not requested_targets:
        return list(candidate_modules)
    return [name for name in candidate_modules if _matches_target_modules(name, requested_targets)]


def _build_peft_exact_target_regex(target_modules: List[str]) -> Optional[str]:
    """Build an exact-match regex for PEFT to avoid suffix collisions on full module paths."""
    normalized_targets = []
    for target in target_modules:
        normalized = str(target).strip().strip(".")
        while normalized.startswith("model."):
            normalized = normalized[len("model."):]
        if normalized:
            normalized_targets.append(normalized)
    if not normalized_targets:
        return None
    pattern = "|".join(re.escape(name) for name in sorted(set(normalized_targets)))
    return rf"^(?:model\.)?(?:{pattern})$"


def _validate_peft_init_compatibility(
    model: nn.Module,
    target_modules: List[str],
    peft_type: str,
    init_lora_weights: Union[str, bool],
) -> Union[str, bool]:
    """Fail fast on PEFT init modes that the current target module types cannot support."""
    normalized_init = _normalize_lora_init(init_lora_weights)
    if str(peft_type).lower() != "lora":
        return normalized_init

    modules_dict = dict(model.named_modules())
    conv_targets = [name for name in target_modules if isinstance(modules_dict.get(name), nn.Conv2d)]
    if conv_targets and isinstance(normalized_init, str) and normalized_init not in {"gaussian"}:
        sample = ", ".join(conv_targets[:3])
        raise ValueError(
            f"PEFT Conv2d targets do not support init_lora_weights='{normalized_init}' in the current runtime. "
            f"requested_init_lora_weights={normalized_init} effective_init_lora_weights=unsupported. "
            f"Conv2d sample targets: {sample}. Use 'gaussian' or standard boolean init instead."
        )
    return normalized_init


def _replace_conv_with_manual_lora(module: nn.Module, config: "LoRAConfig", prefix: str = "", include_head: bool = False) -> int:
    """Recursively replace eligible Conv2d children with manual LoRA wrappers.

    Grouped convolutions are now supported when `r % groups == 0`. Depthwise
    convs (where groups == in_channels == out_channels) are still gated by
    `config.allow_depthwise` to match the PEFT backend behavior.
    """
    replaced = 0
    r = getattr(config, "r", 0) or 0
    allow_depthwise = bool(getattr(config, "allow_depthwise", False))
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv2d):
            groups = child.groups
            # Grouped conv compatibility: rank must be divisible by groups.
            if groups > 1:
                if r > 0 and (r % groups != 0):
                    # Skip silently: rank-groups mismatch
                    replaced += _replace_conv_with_manual_lora(child, config, full_name, include_head)
                    continue
                is_depthwise = (child.in_channels == child.out_channels == groups)
                if is_depthwise and not allow_depthwise:
                    replaced += _replace_conv_with_manual_lora(child, config, full_name, include_head)
                    continue
            if not include_head and _is_head_like_module(full_name):
                continue
            if getattr(config, "only_3x3", False):
                kernel = child.kernel_size
                if kernel == 1 or kernel == (1, 1):
                    continue
            if not _matches_target_modules(full_name, getattr(config, "target_modules", None)):
                continue
            setattr(module, name, ManualLoRAConv(child, r=config.r, alpha=config.alpha, dropout=config.dropout))
            replaced += 1
            continue
        replaced += _replace_conv_with_manual_lora(child, config, prefix=full_name, include_head=include_head)
    return replaced


def apply_manual_lora(model: nn.Module, config: "LoRAConfig", include_head: bool = False) -> nn.Module:
    """Apply manual LoRA wrappers to the model for fallback execution."""
    target_root = getattr(model, "model", model)
    if getattr(config, "freeze_bn", False):
        _freeze_batchnorm_layers(target_root)
    replaced = _replace_conv_with_manual_lora(target_root, config, include_head=include_head)
    if replaced == 0:
        raise ValueError("Fallback LoRA did not find any eligible Conv2d targets.")

    model = _wrap_top_level_lora_model(model, config)
    model.lora_enabled = True
    model.lora_backend = "fallback"
    model.lora_variant = "lora"
    model.lora_include_head = include_head
    model.lora_freeze_bn = bool(getattr(config, "freeze_bn", False))
    model.lora_target_modules = sorted(_collect_fallback_adapter_state(model)["modules"])
    model.lora_runtime_metadata = resolve_effective_lora_request(
        requested_backend=config.backend,
        effective_backend="fallback",
        requested_variant=config.variant,
        effective_variant="lora",
        requested_init_lora_weights=config.init_lora_weights,
        effective_init_lora_weights=config.init_lora_weights,
        include_head=include_head,
        freeze_bn=bool(getattr(config, "freeze_bn", False)),
        target_modules=model.lora_target_modules,
    )

    # Unfreeze detection head (may be frozen by trainer or random init)
    head_unfrozen = 0
    for name, param in model.named_parameters():
        if any(k in name for k in ("detect", "cv2", "cv3", "dfl")):
            if not param.requires_grad:
                param.requires_grad = True
                head_unfrozen += param.numel()
    if head_unfrozen > 0:
        LOGGER.info(
            f"[LoRA] 🔓 Unfrozen {head_unfrozen:,} detection head parameters "
            f"(detect/cv2/cv3/dfl) due to class-mismatch re-initialization."
        )

    return model


def _get_module_by_name(root: nn.Module, module_name: str) -> nn.Module:
    """Resolve a dotted child module path relative to the provided root module."""
    current = root
    if not module_name:
        return current
    for part in module_name.split("."):
        if part in current._modules:
            current = current._modules[part]
        else:
            current = getattr(current, part)
    return current


def _set_module_by_name(root: nn.Module, module_name: str, module: nn.Module) -> None:
    """Replace a dotted child module path relative to the provided root module."""
    if "." in module_name:
        parent_name, child_name = module_name.rsplit(".", 1)
        parent = _get_module_by_name(root, parent_name)
    else:
        parent = root
        child_name = module_name
    parent._modules[child_name] = module


def _collect_fallback_adapter_state(model: nn.Module) -> Dict[str, Any]:
    """Collect serializable fallback LoRA adapter state from ManualLoRAConv modules."""
    target_root = getattr(model, "model", model)
    modules = {}
    state = {}
    for name, module in target_root.named_modules():
        if not isinstance(module, ManualLoRAConv):
            continue
        modules[name] = {
            "r": int(module.r),
            "alpha": int(module.alpha),
            "dropout": float(module.lora_dropout.p if module.lora_dropout is not None else 0.0),
        }
        state[name] = {
            "lora_A": module.lora_A.detach().cpu(),
            "lora_B": module.lora_B.detach().cpu(),
        }
    return {"modules": modules, "state": state}


def _load_fallback_adapter_state(model: nn.Module, path: Path, payload: Dict[str, Any]) -> nn.Module:
    """Load fallback LoRA adapter state into a fresh model instance."""
    weight_file = payload.get("weight_file", "fallback_adapter.pt")
    weights_path = path / weight_file
    if not weights_path.exists():
        raise FileNotFoundError(f"Fallback adapter weights not found: {weights_path}")

    saved = torch.load(weights_path, map_location="cpu")
    module_configs = saved.get("modules", {})
    module_state = saved.get("state", {})
    target_root = getattr(model, "model", model)

    for module_name, config in module_configs.items():
        original = _get_module_by_name(target_root, module_name)
        if isinstance(original, ManualLoRAConv):
            wrapped = original
        else:
            if not isinstance(original, nn.Conv2d):
                raise TypeError(f"Fallback adapter target is not Conv2d: {module_name}")
            wrapped = ManualLoRAConv(
                original,
                r=int(config.get("r", 0)),
                alpha=int(config.get("alpha", 0)),
                dropout=float(config.get("dropout", 0.0)),
            )
            _set_module_by_name(target_root, module_name, wrapped)

        params = module_state.get(module_name, {})
        wrapped.lora_A.data.copy_(params["lora_A"])
        wrapped.lora_B.data.copy_(params["lora_B"])

    model = _wrap_top_level_lora_model(model, None)
    model.lora_enabled = True
    model.lora_backend = "fallback"
    model.lora_variant = payload.get("variant", "lora")
    model.lora_runtime_metadata = payload.get("runtime_metadata", {})
    return model


def _merge_manual_lora_conv(module: ManualLoRAConv) -> nn.Conv2d:
    """Materialize a ManualLoRAConv adapter into a plain Conv2d with merged weights.

    Handles both dense (groups=1) and grouped (groups>1) convolutions. The stored
    shapes are:
        lora_A: (groups, in_per_group * kH * kW, r_per_group)
        lora_B: (groups, out_per_group, r_per_group)
    """
    conv = module.conv
    groups = getattr(module, "groups", 1)
    # Per-group delta: (out_per_group, in_per_group * kH * kW)
    delta_per_group = torch.bmm(module.lora_B, module.lora_A.transpose(1, 2))
    # Reshape into Conv2d weight layout: (out_channels, in_channels/groups, kH, kW)
    out_per_group = conv.out_channels // max(groups, 1)
    in_per_group = conv.in_channels // max(groups, 1)
    weight_delta = delta_per_group.reshape(
        conv.out_channels, in_per_group, *conv.kernel_size
    )
    merged_weight = conv.weight.detach().clone()
    merged_weight.add_(
        weight_delta.to(device=merged_weight.device, dtype=merged_weight.dtype) * module.scaling
    )
    conv.weight.data.copy_(merged_weight)
    return conv


def _merge_fallback_modules(module: nn.Module) -> int:
    """Recursively merge ManualLoRAConv children back into Conv2d modules."""
    merged = 0
    for name, child in list(module.named_children()):
        if isinstance(child, ManualLoRAConv):
            setattr(module, name, _merge_manual_lora_conv(child))
            merged += 1
            continue
        merged += _merge_fallback_modules(child)
    return merged


# ============================================================================
# 1. Enhanced Proxy Class
# ============================================================================

class PeftProxy(PeftModel):
    """
    Advanced PEFT Proxy Wrapper.

    This class bridges the gap between PEFT's arbitrary model structure and 
    Ultralytics' strict expectation of `nn.Sequential` behavior.

    Key Optimizations:
    1. **Sequential Emulation**: intercepts `__getitem__`, `__iter__`, and `__len__` to 
       ensure the model behaves like a list of layers (crucial for YOLO).
    2. **Performance Passthrough**: Explicitly implements `forward` to bypass `__getattr__` overhead.
    3. **State Management**: Correctly handles `state_dict` calls.
    """

    def _get_base(self) -> nn.Module:
        """Helper to retrieve the underlying base model, handling nested PEFT wrappers."""
        model = self.base_model
        # Traverse down if multiple wrappers exist (common in some PEFT versions)
        while hasattr(model, 'model') and not isinstance(model, nn.Sequential):
            model = model.model
        return model

    def forward(self, x, *args, **kwargs):
        """Explicitly pass forward calls to avoid `__getattr__` performance penalty."""
        return self.base_model(x, *args, **kwargs)

    def __getitem__(self, idx: Union[int, slice]):
        """
        Supports index and slice access. 
        This is critical for YOLO's architecture analysis (e.g., `model[i]`).
        """
        base = self._get_base()
        try:
            return base[idx]
        except (TypeError, IndexError, KeyError):
            # Fallback strategy for non-standard containers
            if isinstance(idx, int):
                for i, child in enumerate(base.children()):
                    if i == idx:
                        return child
            raise IndexError(f"Index {idx} out of range for model structure.")

    def __len__(self) -> int:
        return len(self._get_base())

    def __iter__(self):
        return iter(self._get_base())

    def children(self):
        """Ensures iteration over the base model's children, not the adapter's."""
        return self._get_base().children()

    def named_children(self):
        return self._get_base().named_children()

    def __getattr__(self, name: str):
        """
        Dynamic attribute forwarding.
        Note: Frequently accessed attributes should be explicitly defined for performance.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._get_base(), name)

    def state_dict(self, *args, **kwargs):
        """
        Delegates to the parent to decide whether to return full weights or just adapters.
        """
        return super().state_dict(*args, **kwargs)

    def fuse(self, verbose: bool = True):
        """
        Intercepts fusion operations to prevent structural damage to LoRA during training/validation.
        """
        if verbose:
            LOGGER.info("[LoRA] ⚠️  Fusion blocked to preserve LoRA structure during training/val.")
        return self


class LoRADetectionModel:
    """
    Mixin class for LoRA-enabled models.
    
    Primary Functions:
    1. Flags the model as LoRA-enabled.
    2. Disables the default Ultralytics `fuse()` logic, preventing premature weight merging.
    """
    def fuse(self, verbose: bool = True):
        if verbose:
            LOGGER.info("[LoRA] Fusion disabled for LoRADetectionModel.")
        return self

# Wrapper classes for pickling support
class LoRADetectionModelWrapper(LoRADetectionModel, DetectionModel): pass
class LoRASegmentationModelWrapper(LoRADetectionModel, SegmentationModel): pass
class LoRAPoseModelWrapper(LoRADetectionModel, PoseModel): pass
class LoRAClassificationModelWrapper(LoRADetectionModel, ClassificationModel): pass
class LoRAOBBModelWrapper(LoRADetectionModel, OBBModel): pass
class LoRARTDETRDetectionModelWrapper(LoRADetectionModel, RTDETRDetectionModel): pass
class LoRAWorldModelWrapper(LoRADetectionModel, WorldModel): pass


def _wrap_top_level_lora_model(model: "DetectionModel", config: Any = None) -> "DetectionModel":
    """Swap the top-level model class to its LoRA-enabled wrapper and attach flags."""
    original_cls = model.__class__
    if not hasattr(model, "lora_original_class"):
        model.lora_original_class = original_cls

    wrappers = {
        DetectionModel: LoRADetectionModelWrapper,
        SegmentationModel: LoRASegmentationModelWrapper,
        PoseModel: LoRAPoseModelWrapper,
        ClassificationModel: LoRAClassificationModelWrapper,
        OBBModel: LoRAOBBModelWrapper,
        RTDETRDetectionModel: LoRARTDETRDetectionModelWrapper,
        WorldModel: LoRAWorldModelWrapper,
    }

    if original_cls in wrappers:
        model.__class__ = wrappers[original_cls]
    else:
        class LoRAWrapped(LoRADetectionModel, original_cls):
            pass

        LoRAWrapped.__name__ = f"LoRA_{original_cls.__name__}"
        model.__class__ = LoRAWrapped

    model.lora_enabled = True
    model.lora_config = config
    return model


# ============================================================================
# 2. Configuration Class
# ============================================================================

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

    # Advanced Options
    gradient_checkpointing: bool = False
    auto_r_ratio: float = 0.0 # Automatically calculate R based on parameter ratio
    use_dora: bool = False # Enable DoRA (Weight-Decomposed Low-Rank Adaptation)
    use_rslora: bool = True # Enable Rank-Stabilized LoRA scaling (alpha / sqrt(r))
    init_lora_weights: Union[str, bool] = True # LoRA init mode: True/False for std init, or "gaussian"/"pissa"/"olora"
    peft_type: str = "lora" # Options: "lora", "loha", "lokr"
    quantization: str = "none" # Options: "none", "4bit", "8bit" (Requires bitsandbytes)
    only_3x3: bool = False # Skip 1x1 convs during auto target selection
    target_r: int = 8 # AdaLoRA target rank
    init_r: int = 12 # AdaLoRA initial rank
    tinit: int = 0 # AdaLoRA warmup steps before pruning
    tfinal: int = 0 # AdaLoRA final fine-tuning steps
    delta_t: int = 1 # AdaLoRA allocation interval
    beta1: float = 0.85 # AdaLoRA EMA beta1
    beta2: float = 0.85 # AdaLoRA EMA beta2
    orth_reg_weight: float = 0.5 # AdaLoRA orthogonal regularization weight
    total_step: Optional[int] = None # AdaLoRA total training steps, required by PEFT

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
            "target_modules": "lora_target_modules", 
            "gradient_checkpointing": "lora_gradient_checkpointing",
            "auto_r_ratio": "lora_auto_r_ratio",
            "use_dora": "lora_use_dora",
            "use_rslora": "lora_use_rslora",
            "init_lora_weights": "lora_init_lora_weights",
            "peft_type": "lora_type",
            "quantization": "lora_quantization",
            "only_3x3": "lora_only_3x3",
            "target_r": "lora_target_r",
            "init_r": "lora_init_r",
            "tinit": "lora_tinit",
            "tfinal": "lora_tfinal",
            "delta_t": "lora_delta_t",
            "beta1": "lora_beta1",
            "beta2": "lora_beta2",
            "orth_reg_weight": "lora_orth_reg_weight",
            "total_step": "lora_total_step",
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
        **kwargs,
    ) -> List[str]:
        """
        Intelligently detects target layers for LoRA injection.
        """
        targets: Set[str] = set()
        # LOGGER.info(f"DEBUG: auto_detect running with r={r}")
        
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

        # Iterate through all sub-modules
        for name, module in model.named_modules():
            if not name: continue 
            
            # 0. Explicit Exclusion
            if name in exclude_set:
                continue

            # 1. Index Filtering (Valid only if module name starts with a digit)
            if apply_idx_filter:
                idx = LoRAConfigBuilder._get_layer_index(name)
                if idx != -1:
                    if not (start_idx <= idx < end_idx):
                        continue

            # 2. Type Filtering (Must be Conv2d or Linear)
            is_conv = isinstance(module, nn.Conv2d)
            is_linear = isinstance(module, nn.Linear)
            if not (is_conv or is_linear):
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

            # Attention Check
            if not include_attention and is_linear and LoRAConfigBuilder._PAT_ATTN.search(lname):
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
    ) -> Union['LoraConfig', 'LoHaConfig', 'LoKrConfig', None]:
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
                LOGGER.warning("[LoRA] AdaLoRA currently supports nn.Linear targets only; all non-linear targets were filtered out.")
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
            
        else: # Default to LoRA (and DoRA)
            lora_kwargs = {
                "lora_alpha": alpha,
                "lora_dropout": kwargs.get('dropout', 0.05),
                "bias": kwargs.get('bias', "none"),
                "use_dora": kwargs.get('use_dora', False),
                **common_kwargs,
            }
            if _supports_peft_kwarg(LoraConfig, "use_rslora"):
                lora_kwargs["use_rslora"] = kwargs.get('use_rslora', True)
            elif kwargs.get('use_rslora', True):
                LOGGER.warning("[LoRA] Installed PEFT does not support use_rslora; falling back to standard scaling.")

            if _supports_peft_kwarg(LoraConfig, "init_lora_weights"):
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

            return LoraConfig(**lora_kwargs)


# ============================================================================
# 4. Main Entry Point
# ============================================================================

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
    config = LoRAConfig.from_args(args, **kwargs)

    # Check if LoRA should be enabled
    if config.r <= 0 and config.auto_r_ratio <= 0:
        LOGGER.info("[LoRA] Disabled (r=0).")
        return model

    variant = str(getattr(config, "variant", config.peft_type)).lower()
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
    for name, _ in model.named_modules():
        if LoRAConfigBuilder._PAT_MOE.search(name):
            has_moe = True
        if LoRAConfigBuilder._PAT_ATTN.search(name):
            has_attn = True
        if has_moe and has_attn:
            break
    
    if config.include_moe and not has_moe:
        config.include_moe = False
    
    if config.include_attention and not has_attn:
        config.include_attention = False

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
        "target_modules": config.target_modules, # This might be ['conv']
        "gradient_checkpointing": config.gradient_checkpointing,
        "auto_r_ratio": config.auto_r_ratio,
        "use_dora": config.use_dora,
        "use_rslora": config.use_rslora,
        "init_lora_weights": config.init_lora_weights,
        "peft_type": config.peft_type,
        "only_3x3": config.only_3x3,
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
        model.lora_variant = config.variant
        model.lora_include_head = config.include_head
        model.lora_freeze_bn = bool(getattr(config, "freeze_bn", False))
        model.lora_target_modules = sorted(final_targets)
        model.lora_runtime_metadata = resolve_effective_lora_request(
            requested_backend=config.backend,
            effective_backend="peft",
            requested_variant=config.variant,
            effective_variant=config.variant,
            requested_init_lora_weights=config.init_lora_weights,
            effective_init_lora_weights=config.init_lora_weights,
            include_head=config.include_head,
            freeze_bn=bool(getattr(config, "freeze_bn", False)),
            target_modules=model.lora_target_modules,
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
        raise e

    # Unfreeze detection head (may be frozen by PEFT or random init)
    head_unfrozen = 0
    for name, param in model.named_parameters():
        if any(k in name for k in ("detect", "cv2", "cv3", "dfl")):
            if not param.requires_grad:
                param.requires_grad = True
                head_unfrozen += param.numel()

    if head_unfrozen > 0:
        LOGGER.info(
            f"[LoRA] 🔓 Unfrozen {head_unfrozen:,} detection head parameters "
            f"(detect/cv2/cv3/dfl) due to class-mismatch re-initialization."
        )

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
    _print_param_stats(model)

    return model


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


def _print_param_stats(model: nn.Module):
    """Prints detailed parameter statistics."""
    trainable_params = 0
    all_params = 0
    lora_params = 0
    frozen_base = 0

    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_base += param.numel()
        if "lora_" in name:
            lora_params += param.numel()

    pct = 100 * trainable_params / all_params if all_params > 0 else 0
    lora_pct = 100 * lora_params / all_params if all_params > 0 else 0
    base_total = all_params - lora_params
    
    LOGGER.info(f"[LoRA] 📊 Stats: "
                f"Trainable: {trainable_params:,} ({pct:.3f}%) | "
                f"Frozen Base: {frozen_base:,} | "
                f"LoRA Params: {lora_params:,} ({lora_pct:.3f}%) | "
                f"Base Total: {base_total:,}")

    if trainable_params == all_params:
        LOGGER.warning("[LoRA] ⚠️  ALL parameters are trainable. Check if LoRA adapters were applied correctly.")
    
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
        if "lora_" in name:
            lora_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if lora_params:
        param_groups.append({"params": lora_params, "weight_decay": lora_weight_decay})
    if other_params:
        param_groups.append({"params": other_params, "weight_decay": weight_decay})
    return param_groups


def save_lora_adapters(model: "DetectionModel", path: Union[str, Path]) -> bool:
    """
    Saves only the LoRA Adapter weights.
    
    Args:
        model: LoRADetectionModel instance.
        path: Directory path for saving.
    """
    # Unwrap DDP
    if hasattr(model, 'module'):
        model = model.module

    if not getattr(model, 'lora_enabled', False):
        LOGGER.debug("[LoRA] Save skipped: LoRA not enabled.")
        return False

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    backend = getattr(model, "lora_backend", "peft")
    variant = getattr(model, "lora_variant", "lora")
    
    try:
        if backend == "fallback":
            fallback_state = _collect_fallback_adapter_state(model)
            weight_file = "fallback_adapter.pt"
            torch.save(fallback_state, path / weight_file)
            payload = {
                "backend": backend,
                "variant": variant,
                "weight_file": weight_file,
                "freeze_bn": bool(getattr(model, "lora_freeze_bn", False)),
                "include_head": bool(getattr(model, "lora_include_head", False)),
                "target_modules": list(getattr(model, "lora_target_modules", sorted(fallback_state["modules"]))),
                "runtime_metadata": getattr(model, "lora_runtime_metadata", {}),
            }
            (path / "adapter_config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            LOGGER.info(f"[LoRA] 💾 Fallback adapter metadata saved to {path}")
            return True

        # model.model is PeftProxy (PeftModel)
        # save_pretrained automatically saves only the adapter weights
        model.model.save_pretrained(str(path))
        runtime_payload = {
            "backend": backend,
            "variant": variant,
            "freeze_bn": bool(getattr(model, "lora_freeze_bn", False)),
            "include_head": bool(getattr(model, "lora_include_head", False)),
            "target_modules": list(getattr(model, "lora_target_modules", [])),
            "runtime_metadata": getattr(model, "lora_runtime_metadata", {}),
        }
        (path / "runtime_metadata.json").write_text(json.dumps(runtime_payload, indent=2, ensure_ascii=False))
        LOGGER.info(f"[LoRA] 💾 Adapters saved to {path}")
        return True
    except Exception as e:
        LOGGER.error(f"[LoRA] Failed to save adapters: {e}")
        return False


def load_lora_adapters(model: "DetectionModel", path: Union[str, Path], merge: bool = False, force_replace: bool = False) -> bool:
    """
    Loads LoRA adapter weights onto an existing Ultralytics model.

    Args:
        model: Base Ultralytics model instance.
        path: Directory containing PEFT adapter files.
        merge: Whether to merge loaded adapters into the base model immediately.
        force_replace: If True, replace existing LoRA adapters with new ones (default False).
    """
    path = Path(path)
    if not path.exists():
        LOGGER.error(f"[LoRA] Adapter path not found: {path}")
        return False

    config_path = path / "adapter_config.json"
    payload = {}
    if config_path.exists():
        try:
            payload = json.loads(config_path.read_text())
        except Exception:
            payload = {}

    if hasattr(model, "module"):
        model = model.module

    if getattr(model, "lora_enabled", False):
        if force_replace:
            LOGGER.info("[LoRA] Force-replacing existing LoRA adapters with new ones.")
            if hasattr(getattr(model, "model", None), "merge_and_unload"):
                merge_lora_weights(model)
            else:
                if hasattr(model, "lora_enabled"):
                    delattr(model, "lora_enabled")
        else:
            LOGGER.warning("[LoRA] Model already has LoRA enabled. Skipping. Use force_replace=True to override.")
            return True

    if payload.get("backend") == "fallback":
        model = _load_fallback_adapter_state(model, path, payload)
        LOGGER.info(f"[LoRA] 📥 Fallback adapter metadata loaded from {path}")
        if merge:
            return merge_lora_weights(model)
        return True

    if not PEFT_AVAILABLE:
        LOGGER.error("[LoRA] PEFT library not found. Please install via `pip install peft`.")
        return False

    try:
        peft_model_wrapper = PeftModel.from_pretrained(model.model, str(path), is_trainable=False)
        peft_model_wrapper.__class__ = PeftProxy
        model.model = peft_model_wrapper
        _wrap_top_level_lora_model(model, getattr(peft_model_wrapper, "peft_config", None))
        runtime_path = path / "runtime_metadata.json"
        runtime_payload = {}
        if runtime_path.exists():
            try:
                runtime_payload = json.loads(runtime_path.read_text())
            except Exception:
                runtime_payload = {}
        model.lora_backend = runtime_payload.get("backend", "peft")
        model.lora_variant = runtime_payload.get("variant", "lora")
        model.lora_include_head = runtime_payload.get("include_head", False)
        model.lora_freeze_bn = runtime_payload.get("freeze_bn", False)
        model.lora_target_modules = runtime_payload.get("target_modules", [])
        model.lora_runtime_metadata = runtime_payload.get("runtime_metadata", {})

        LOGGER.info(f"[LoRA] 📥 Adapters loaded from {path}")
        if merge:
            return merge_lora_weights(model)
        return True
    except Exception as e:
        LOGGER.error(f"[LoRA] Failed to load adapters: {e}")
        return False


def _find_original_model_class(model: "DetectionModel"):
    """Find the original model class before LoRA wrapping by inspecting MRO."""
    from ultralytics.nn.tasks import (
        DetectionModel, SegmentationModel, PoseModel,
        ClassificationModel, OBBModel, RTDETRDetectionModel, WorldModel
    )
    
    # Known original classes
    ORIGINAL_CLASSES = {
        DetectionModel, SegmentationModel, PoseModel,
        ClassificationModel, OBBModel, RTDETRDetectionModel, WorldModel
    }
    
    # Check all bases in MRO order
    for cls in model.__class__.__mro__:
        if cls in ORIGINAL_CLASSES:
            return cls
    
    # Fallback to DetectionModel if we can't determine the original class
    return DetectionModel


def merge_lora_weights(model: "DetectionModel") -> bool:
    """
    Merges LoRA weights back into the base model and unloads adapters.
    Useful for inference acceleration or model export.
    """
    if getattr(model, "lora_backend", None) == "fallback":
        try:
            target_root = getattr(model, "model", model)
            merged_count = _merge_fallback_modules(target_root)
            if merged_count == 0:
                LOGGER.error("[LoRA] Cannot merge fallback adapters: no ManualLoRAConv modules found.")
                return False

            original_cls = getattr(model, "lora_original_class", None)
            if original_cls is not None:
                model.__class__ = original_cls

            for attr in (
                "lora_enabled",
                "lora_config",
                "lora_backend",
                "lora_variant",
                "lora_include_head",
                "lora_runtime_metadata",
                "lora_original_class",
            ):
                if hasattr(model, attr):
                    try:
                        delattr(model, attr)
                    except AttributeError:
                        pass

            LOGGER.info(f"[LoRA] ✅ Fallback merge completed. Merged {merged_count} manual LoRA modules.")
            return True
        except Exception as e:
            LOGGER.error(f"[LoRA] Fallback merge failed: {e}")
            return False

    # Check if wrapped in PeftProxy
    if not hasattr(model, 'model') or not hasattr(getattr(model, 'model', None), 'merge_and_unload'):
        LOGGER.error("[LoRA] Cannot merge: Model does not appear to have LoRA adapters attached.")
        return False

    try:
        LOGGER.info("[LoRA] 🔄 Merging adapters into base model...")
        
        # merge_and_unload returns the clean base model (nn.Sequential)
        merged_base = model.model.merge_and_unload()
        
        # Restore structure
        model.model = merged_base
        
        # Restore original class using robust MRO inspection
        original_cls = _find_original_model_class(model)
        model.__class__ = original_cls
        
        # Clear flags
        for attr in ('lora_enabled', 'lora_config', 'use_gradient_checkpointing'):
            if hasattr(model, attr):
                try:
                    delattr(model, attr)
                except AttributeError:
                    pass
            
        LOGGER.info(f"[LoRA] ✅ Merge completed. Model restored to {original_cls.__name__} architecture.")
        return True
    except Exception as e:
        LOGGER.error(f"[LoRA] Merge failed: {e}")
        return False


# ============================================================================
# 6. Advanced Training Strategies
# ============================================================================

class LoraTrainingStrategy:
    """
    Advanced training strategies for LoRA fine-tuning.

    Provides 4 complementary strategies:
    1. Layer-wise Decay: Reduce LR for deeper layers (stabilizes early training)
    2. Alpha Warmup: Gradually increase lora_alpha (prevents initial instability)
    3. Orthogonal Regularization: Penalize rank collapse in A/B matrices
    4. Dynamic Dropout Scheduling: Increase dropout as training progresses
    """

    def __init__(self, model, config=None, epochs=100):
        self.model = model
        self.config = config or getattr(model, 'lora_config', None)
        self.epochs = epochs
        self._original_alphas = {}  # Store original alpha values per layer
        self._strategy_active = False

    # ── Strategy 1: Layer-wise LR decay ──
    @staticmethod
    def get_layer_decay_factors(model, total_layers=None, decay_rate=0.85) -> Dict[str, float]:
        """
        Compute per-layer LR multipliers with exponential decay by depth.

        Args:
            model: LoRA-enabled model
            total_layers: Total number of YOLO backbone+head blocks (auto-detected if None).
                For YOLO, this is the count of top-level Sequential children (typically ~23).
                NOT the count of all nn.Module descendants.
            decay_rate: Multiplicative factor per layer depth (0.8~0.95 typical)

        Returns:
            Dict mapping parameter name -> lr_multiplier
        """
        if total_layers is None:
            # Auto-detect from YOLO structure. YOLO wraps blocks as a nn.Sequential
            # under `.model` (or `.model.model` if wrapped by PeftProxy).
            # We count the top-level numbered blocks (0..N), not all descendants.
            candidate_roots = []
            for root_attr in ("model", "base_model"):
                cur = getattr(model, root_attr, None)
                # Descend through nested wrappers
                for _ in range(4):
                    if cur is None:
                        break
                    if hasattr(cur, "__len__"):
                        try:
                            n = len(cur)
                            if n > 1:
                                candidate_roots.append(n)
                                break
                        except TypeError:
                            pass
                    cur = getattr(cur, "model", None) or getattr(cur, "base_model", None)

            if candidate_roots:
                total_layers = max(candidate_roots)
            else:
                # Fallback: extract max top-level index from LoRA parameter names
                max_idx = 0
                for name, _ in model.named_parameters():
                    if "lora_" not in name:
                        continue
                    for p in name.split("."):
                        if p.isdigit():
                            max_idx = max(max_idx, int(p))
                            break
                total_layers = max(max_idx + 1, 10)  # Minimum 10 layers

        factors = {}
        for name, param in model.named_parameters():
            if "lora_" not in name:
                continue
            # Extract layer index from name (e.g., "model.23.cv3.0.conv.lora_A.weight")
            parts = name.split(".")
            layer_idx = 0
            for p in parts:
                if p.isdigit():
                    layer_idx = int(p)
                    break

            # Normalize to [0, 1]
            normalized_depth = min(layer_idx / max(total_layers, 1), 1.0)
            # Exponential decay: shallow layers get higher LR
            factor = decay_rate ** normalized_depth
            factors[name] = factor

        return factors

    def apply_layer_decay_to_optimizer(self, optimizer, decay_rate=0.85) -> int:
        """
        Apply layer-wise LR decay to existing optimizer param groups.
        
        This function REPLACES the single LoRA param group with multiple
        param groups, each with a different LR based on layer depth.
        
        PyTorch optimizer requires one param_group per unique LR, so we group
        parameters by their layer index and create one param_group per layer.

        Returns:
            Number of parameters whose LR was adjusted
        """
        # Guardrail: very small decay_rate (e.g. 0.01) is almost always a config mistake.
        # Typical range is 0.8 ~ 0.95; anything below 0.5 collapses all layers to ~0 lr
        # and defeats the purpose of layer-wise decay.
        if decay_rate <= 0.0 or decay_rate > 1.0:
            LOGGER.warning(
                f"[LoRA-Strategy] ⚠️ Invalid lora_layer_decay={decay_rate}. "
                f"Must be in (0, 1]. Skipping layer decay."
            )
            return 0
        if decay_rate < 0.5:
            LOGGER.warning(
                f"[LoRA-Strategy] ⚠️ lora_layer_decay={decay_rate} is very aggressive. "
                f"Recommended range is 0.8~0.95. Deep layers will receive near-zero LR, "
                f"which typically causes adapter under-training (mAP collapse)."
            )

        factors = self.get_layer_decay_factors(self.model, decay_rate=decay_rate)
        if not factors:
            return 0

        # Find the LoRA param group index and its base_lr.
        # Build a name lookup once to avoid O(N*M) scan; then collect ALL LoRA
        # params (the earlier implementation had an off-by-one break that only
        # picked up the first LoRA parameter per group, collapsing everything
        # into a single bucket).
        name_by_id = {id(p): n for n, p in self.model.named_parameters()}

        lora_pg_idx = None
        base_lr = None
        lora_params_in_pg = []

        for idx, pg in enumerate(optimizer.param_groups):
            pg_has_lora = False
            for p in pg.get("params", []):
                name = name_by_id.get(id(p))
                if name is not None and "lora_" in name:
                    pg_has_lora = True
                    lora_params_in_pg.append((name, p, idx))
            if pg_has_lora and base_lr is None:
                base_lr = pg.get('lr', None)
                lora_pg_idx = idx

        if base_lr is None or lora_pg_idx is None:
            LOGGER.warning("[LoRA-Strategy] No LoRA param group found for layer decay.")
            return 0

        # Group LoRA params by layer index for efficient param_group creation
        from collections import defaultdict
        layer_groups = defaultdict(list)
        
        for name, param, _ in lora_params_in_pg:
            factor = factors.get(name, 1.0)
            # Round factor to reduce number of param groups.
            # Use 1 decimal precision: this reduces group count from ~18 to ~3-5
            # while still preserving meaningful stratification across depths.
            # Previous 3-decimal precision created too many groups (18+), slowing optimizer.
            rounded_factor = round(factor, 1)
            layer_groups[rounded_factor].append(param)
        
        # Remove the original LoRA param group (remove from end to keep indices stable)
        # We need to rebuild param_groups since PyTorch doesn't support deletion
        original_groups = optimizer.param_groups.copy()
        
        # Create new param_groups list
        new_param_groups = []
        for idx, pg in enumerate(original_groups):
            if idx == lora_pg_idx:
                # Replace with multiple layer-specific groups
                for factor, params in sorted(layer_groups.items(), reverse=True):
                    new_lr = base_lr * factor
                    # Start with a copy of the original param_group
                    new_pg = {k: v for k, v in pg.items() if k != "params"}
                    new_pg["params"] = params
                    new_pg["lr"] = new_lr
                    new_pg["initial_lr"] = new_lr  # for warmup scheduler
                    new_param_groups.append(new_pg)
            else:
                new_param_groups.append(pg)
        
        # Replace optimizer's param_groups
        optimizer.param_groups = new_param_groups
        
        # Also rebuild state if necessary (state is keyed by parameter object, so it remains valid)
        # But we need to update the optimizer's internal _param_group map if it exists
        if hasattr(optimizer, '_param_groups'):
            optimizer._param_groups = optimizer.param_groups
        
        avg_factor = sum(factors.values()) / len(factors)
        min_factor = min(factors.values())
        max_factor = max(factors.values())

        # Sanity check: a single LR group means depth stratification failed entirely.
        # This typically happens when the layer index detector returns the same index
        # for every LoRA param (e.g. when all names come from a sub-module without a
        # leading digit), or when decay_rate is so extreme that all factors round to
        # the same bucket.
        if len(layer_groups) == 1:
            LOGGER.warning(
                f"[LoRA-Strategy] ⚠️ Layer decay produced only 1 LR group "
                f"(decay_rate={decay_rate}, factor_range=[{min_factor:.4f}, {max_factor:.4f}]). "
                f"Stratification is effectively disabled. Check module naming or raise decay_rate."
            )

        LOGGER.info(
            f"[LoRA-Strategy] 📐 Layer-wise LR decay applied (rate={decay_rate}): "
            f"{len(layer_groups)} LR groups, "
            f"avg_factor={avg_factor:.3f}, range=[{min_factor:.3f}, {max_factor:.3f}]"
        )
        self._layer_decay_factors = factors
        return len(factors)

    # ── Strategy 2: Alpha Warmup ──
    def prepare_alpha_warmup(self):
        """
        Store original alpha scales and set initial scale to 0.

        PEFT LoRA scaling = alpha / r. This function stores the target alpha value
        for each LoRA layer and temporarily sets effective alpha to 0.

        Handles multiple PEFT internal structures:
          - PEFT >= 0.13: LoraLayer with lora_alpha property (may be property or stored in peft_config dict)
          - PEFT < 0.13: Direct 'scaling' attribute
          - PEFT >= 0.18: lora_alpha and scaling are dicts keyed by adapter name (e.g. {'default': 8})
        """
        self._original_alphas.clear()
        found = False

        # Determine config-level defaults
        cfg_alpha = 32  # default
        cfg_r = 8       # default
        if self.config is not None:
            cfg_alpha = getattr(self.config, 'alpha', 32) or getattr(self.config, 'lora_alpha', 32) or 32
            cfg_r = getattr(self.config, 'r', 8) or getattr(self.config, 'lora_r', 8) or 8

        for module in self.model.modules():
            lora_a = getattr(module, 'lora_A', None)
            # Only process actual LoRA layers
            if lora_a is None:
                continue
            # PEFT >= 0.18 uses nn.ModuleDict for lora_A (e.g. {'default': Conv2d}).
            # Older PEFT stores lora_A as a single Parameter or Module with .weight.
            is_lora_layer = False
            if isinstance(lora_a, nn.ModuleDict):
                # Check that at least one adapter entry has a weight attribute
                is_lora_layer = any(hasattr(child, 'weight') for child in lora_a.values())
            elif hasattr(lora_a, 'weight'):
                is_lora_layer = True
            if not is_lora_layer:
                continue

            # Strategy: detect how to control scaling for this PEFT version
            la_attr = getattr(module, 'lora_alpha', None)
            lr_attr = getattr(module, 'r', None)
            sc_attr = getattr(module, 'scaling', None)

            # ── Path A: PEFT >= 0.18 dict-style lora_alpha / scaling ──
            if isinstance(la_attr, dict) and isinstance(sc_attr, dict):
                # Both are dicts keyed by adapter name (e.g. 'default')
                # scaling = alpha / r, so we control scaling dict directly
                adapter_name = list(la_attr.keys())[0] if la_attr else 'default'
                orig_alpha = float(la_attr.get(adapter_name, cfg_alpha))
                orig_scaling = float(sc_attr.get(adapter_name, orig_alpha / max(cfg_r, 1)))
                self._original_alphas[id(module)] = {
                    '_type': 'scaling_dict',
                    'orig_alpha': orig_alpha,
                    'orig_scaling': orig_scaling,
                    'adapter_name': adapter_name,
                    'r': float(lr_attr) if isinstance(lr_attr, (int, float)) else float(cfg_r),
                }
                # Set scaling to 0 to disable LoRA contribution at start
                sc_attr[adapter_name] = 0.0
                found = True
                continue

            # ── Path B: Both lora_alpha and r are directly writable numbers (older PEFT) ──
            if (isinstance(la_attr, (int, float)) and isinstance(lr_attr, (int, float))
                    and lr_attr > 0):
                orig_alpha = float(la_attr)
                self._original_alphas[id(module)] = {
                    '_type': 'direct',
                    'orig_alpha': orig_alpha,
                    'r': float(lr_attr),
                }
                # Set alpha to 0 (scaling becomes 0)
                module.lora_alpha = 0.0
                found = True
                continue

            # ── Path C: lora_alpha might be a property in newer PEFT, but we can try to set it ──
            if la_attr is not None:
                try:
                    _orig_alpha = float(la_attr)
                    _r = float(lr_attr) if isinstance(lr_attr, (int, float)) else float(cfg_r)
                    self._original_alphas[id(module)] = {
                        '_type': 'property',
                        'orig_alpha': _orig_alpha,
                        'r': _r,
                    }
                    # Attempt to set; we'll verify in step
                    module.lora_alpha = 0.0
                    found = True
                    continue
                except (TypeError, ValueError, AttributeError):
                    pass

            # ── Path D: Has numeric 'scaling' attribute (older PEFT or custom) ──
            if isinstance(sc_attr, (int, float)) and sc_attr > 0:
                self._original_alphas[id(module)] = {
                    '_type': 'scaling',
                    'orig_scaling': float(sc_attr),
                }
                module.scaling = 0.0
                found = True
                continue

            # ── Path E: Fallback - try to use peft_config dict if available ──
            peft_config = getattr(module, 'peft_config', None)
            if peft_config is not None:
                try:
                    if isinstance(peft_config, dict) and 'lora_alpha' in peft_config:
                        _orig_alpha = float(peft_config['lora_alpha'])
                        _r = float(peft_config.get('r', cfg_r))
                        self._original_alphas[id(module)] = {
                            '_type': 'config_dict',
                            'orig_alpha': _orig_alpha,
                            'r': _r,
                            'module_ref': module,  # store ref to update dict
                        }
                        peft_config['lora_alpha'] = 0.0
                        found = True
                        continue
                except (TypeError, ValueError):
                    pass

        if found:
            self._strategy_active = True
            # Diagnostic: report the distribution of _type paths so users can quickly
            # verify that the PEFT-version-specific fallback is working correctly.
            from collections import Counter
            type_dist = Counter(v.get('_type', 'unknown') for v in self._original_alphas.values())
            type_summary = ", ".join(f"{t}={c}" for t, c in type_dist.most_common())
            LOGGER.info(
                f"[LoRA-Strategy] 🔥 Alpha warmup prepared ({len(self._original_alphas)} layers) "
                f"| path distribution: {type_summary}"
            )
        else:
            LOGGER.warning(
                "[LoRA-Strategy] ⚠️ No modifiable alpha attributes found for warmup. "
                "This usually indicates a PEFT version mismatch — alpha warmup will be silently disabled "
                "but training will continue normally. Please report PEFT version to maintainers."
            )
        return found

    def step_alpha_warmup(self, epoch, warmup_epochs=5):
        """
        Update alpha scaling based on current epoch (cosine ramp-up).

        Returns current scale factor in [0, 1].
        """
        if not self._original_alphas:
            return 1.0

        progress = min(epoch / max(warmup_epochs, 1), 1.0)
        # Cosine ease-in: starts at 0, ends at 1
        current_scale = 0.5 * (1 - math.cos(math.pi * progress))

        updated = 0
        for module in self.model.modules():
            mid = id(module)
            if mid not in self._original_alphas:
                continue

            orig = self._original_alphas[mid]
            _type = orig['_type']

            try:
                # ── Path A: scaling dict (PEFT >= 0.18) ──
                if _type == 'scaling_dict':
                    sc_attr = getattr(module, 'scaling', None)
                    if isinstance(sc_attr, dict):
                        adapter_name = orig.get('adapter_name', 'default')
                        orig_scaling = orig['orig_scaling']
                        sc_attr[adapter_name] = orig_scaling * current_scale
                        updated += 1
                    continue

                if _type == 'direct':
                    target_alpha = orig['orig_alpha'] * current_scale
                    if hasattr(module, 'lora_alpha'):
                        module.lora_alpha = float(target_alpha)
                        updated += 1

                elif _type == 'property':
                    target_alpha = orig['orig_alpha'] * current_scale
                    if hasattr(module, 'lora_alpha'):
                        module.lora_alpha = float(target_alpha)
                        # Verify the write actually stuck
                        actual = getattr(module, 'lora_alpha', None)
                        if actual is not None and abs(float(actual) - target_alpha) < 0.01:
                            updated += 1
                        else:
                            # Property is read-only, try scaling attribute as fallback
                            if hasattr(module, 'scaling'):
                                orig_scaling = orig['orig_alpha'] / orig['r']
                                module.scaling = orig_scaling * current_scale
                                updated += 1
                                # Update type for future steps
                                orig['_type'] = 'scaling_fallback'
                                orig['orig_scaling'] = orig_scaling

                elif _type == 'scaling' or _type == 'scaling_fallback':
                    orig_scaling = orig.get('orig_scaling', orig.get('orig_alpha', 1.0) / orig.get('r', 1.0))
                    if hasattr(module, 'scaling'):
                        module.scaling = orig_scaling * current_scale
                        updated += 1

                elif _type == 'config_dict':
                    target_alpha = orig['orig_alpha'] * current_scale
                    peft_config = getattr(module, 'peft_config', None)
                    if isinstance(peft_config, dict):
                        peft_config['lora_alpha'] = float(target_alpha)
                        updated += 1

            except Exception as e:
                LOGGER.debug(f"[LoRA-Strategy] Alpha warmup step failed for module {mid}: {e}")
                continue

        return current_scale

    def finalize_alpha_warmup(self):
        """Restore all alphas to their original values."""
        restored = 0
        for module in self.model.modules():
            mid = id(module)
            if mid not in self._original_alphas:
                continue
            orig = self._original_alphas[mid]
            _type = orig['_type']

            try:
                # ── Path A: scaling dict (PEFT >= 0.18) ──
                if _type == 'scaling_dict':
                    sc_attr = getattr(module, 'scaling', None)
                    if isinstance(sc_attr, dict):
                        adapter_name = orig.get('adapter_name', 'default')
                        sc_attr[adapter_name] = float(orig['orig_scaling'])
                        restored += 1
                    continue

                if _type in ('direct', 'property'):
                    if hasattr(module, 'lora_alpha'):
                        module.lora_alpha = float(orig['orig_alpha'])
                        restored += 1

                elif _type in ('scaling', 'scaling_fallback'):
                    if hasattr(module, 'scaling'):
                        module.scaling = float(orig.get('orig_scaling', orig.get('orig_alpha', 1.0) / orig.get('r', 1.0)))
                        restored += 1

                elif _type == 'config_dict':
                    peft_config = getattr(module, 'peft_config', None)
                    if isinstance(peft_config, dict):
                        peft_config['lora_alpha'] = float(orig['orig_alpha'])
                        restored += 1

            except Exception as e:
                LOGGER.debug(f"[LoRA-Strategy] Alpha warmup finalize failed for module {mid}: {e}")
                continue

        LOGGER.info(f"[LoRA-Strategy] Alpha warmup finalized — {restored}/{len(self._original_alphas)} alphas restored.")
        self._strategy_active = False

    # ── Strategy 3: Orthogonal Regularization Loss ──
    @staticmethod
    def compute_orthogonal_loss(model, weight=1e-4) -> torch.Tensor:
        """
        Compute regularization loss encouraging LoRA A/B matrices to stay orthogonal.

        Prevents rank collapse where A·B degenerates into a low-effective-rank product.
        
        Loss = λ × (Σ||A^T A - I||_F + Σ||B^T B - I||_F) / N_pairs
        
        OPTIMIZED: Uses cached module list and avoids redundant device/dtype conversions.
        
        Args:
            model: LoRA-enabled model
            weight: Scaling factor for the loss

        Returns:
            Scalar tensor (orthogonal regularization loss)
        """
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
            
        ortho_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        pair_count = 0

        # OPTIMIZATION: Use a static helper to avoid redefining function on each call
        # and cache the model's modules to avoid generator overhead
        def _iter_weights(attr):
            """Yield weight tensors from either a direct LoRA layer or a ModuleDict (PEFT >=0.18)."""
            if attr is None:
                return
            if isinstance(attr, nn.ModuleDict):
                for child in attr.values():
                    if hasattr(child, 'weight') and child.weight.numel() > 0:
                        yield child.weight
            elif hasattr(attr, 'weight') and attr.weight.numel() > 0:
                yield attr.weight

        # OPTIMIZATION: Iterate through modules once, processing both A and B weights
        # Avoids calling model.named_modules() twice
        for name, module in model.named_modules():
            # Process lora_A weights
            lora_a = getattr(module, 'lora_A', None)
            if lora_a is not None:
                for A_w in _iter_weights(lora_a):
                    # OPTIMIZATION: Avoid .detach().float() if already correct dtype/device
                    A = A_w.detach()
                    if A.dtype != torch.float32:
                        A = A.float()
                    if A.dim() >= 2 and A.shape[0] > 0:
                        if A.dim() > 2:
                            A = A.reshape(A.shape[0], -1)
                        # OPTIMIZATION: Use torch.matmul instead of @ for clarity
                        # and pre-allocate identity matrix if possible
                        AA_T = torch.matmul(A, A.t())
                        rows = AA_T.shape[0]
                        # OPTIMIZATION: Use torch.eye with device directly
                        ident = torch.eye(rows, device=device, dtype=torch.float32)
                        ortho_loss = ortho_loss + torch.norm(AA_T - ident, p='fro')
                        pair_count += 1

            # Process lora_B weights
            lora_b = getattr(module, 'lora_B', None)
            if lora_b is not None:
                for B_w in _iter_weights(lora_b):
                    B = B_w.detach()
                    if B.dtype != torch.float32:
                        B = B.float()
                    if B.dim() >= 2 and B.shape[-1] > 0:
                        if B.dim() > 2:
                            B = B.reshape(B.shape[0], -1)
                        BT_B = torch.matmul(B.t(), B)
                        cols = BT_B.shape[0]
                        ident = torch.eye(cols, device=device, dtype=torch.float32)
                        ortho_loss = ortho_loss + torch.norm(BT_B - ident, p='fro')
                        pair_count += 1

        if pair_count == 0:
            return torch.tensor(0.0, device=device, dtype=torch.float32)

        return weight * (ortho_loss / pair_count)

    # ── Strategy 4: Dynamic Dropout Scheduling ──
    _DROPOUT_WARNED = False  # class-level flag to emit warning only once
    _last_dropout_value = None  # Cache last applied dropout value to avoid redundant updates

    @staticmethod
    def update_dropout_schedule(model, epoch, epochs_total, 
                                  start_dropout=0.0, end_dropout=0.15,
                                  schedule_start_ratio=0.3) -> int:
        """
        Dynamically increase LoRA dropout rate as training progresses.
        
        In early phases, low dropout preserves gradient signal for learning.
        In later phases, higher dropout acts as regularizer preventing overfitting.

        Args:
            model: LoRA-enabled model
            epoch: Current epoch (0-indexed)
            epochs_total: Total number of training epochs
            start_dropout: Initial dropout rate
            end_dropout: Final dropout rate  
            schedule_start_ratio: When to start increasing (fraction of total)

        Returns:
            Number of dropout layers updated
        """
        # Sanity check: end must be >= start, otherwise the schedule is a no-op / decrease.
        if end_dropout < start_dropout:
            if not LoraTrainingStrategy._DROPOUT_WARNED:
                LOGGER.warning(
                    f"[LoRA-Strategy] ⚠️ lora_dropout_end={end_dropout} < lora_dropout={start_dropout}. "
                    f"Dynamic dropout schedule disabled (dropout would monotonically decrease)."
                )
                LoraTrainingStrategy._DROPOUT_WARNED = True
            return 0
        if not (0.0 <= start_dropout <= 1.0 and 0.0 <= end_dropout <= 1.0):
            if not LoraTrainingStrategy._DROPOUT_WARNED:
                LOGGER.warning(
                    f"[LoRA-Strategy] ⚠️ Invalid dropout range [{start_dropout}, {end_dropout}]. "
                    f"Must be within [0, 1]. Schedule disabled."
                )
                LoraTrainingStrategy._DROPOUT_WARNED = True
            return 0
        if not (0.0 <= schedule_start_ratio <= 1.0):
            if not LoraTrainingStrategy._DROPOUT_WARNED:
                LOGGER.warning(
                    f"[LoRA-Strategy] ⚠️ Invalid schedule_start_ratio={schedule_start_ratio}. "
                    f"Must be within [0, 1]. Schedule disabled."
                )
                LoraTrainingStrategy._DROPOUT_WARNED = True
            return 0

        schedule_start = int(epochs_total * schedule_start_ratio)
        if epoch < schedule_start:
            current_dropout = start_dropout
        else:
            # Linear interpolation after schedule starts
            progress = (epoch - schedule_start) / max(epochs_total - schedule_start, 1)
            current_dropout = start_dropout + (end_dropout - start_dropout) * min(progress, 1.0)

        # OPTIMIZATION: Skip redundant updates if dropout value hasn't changed
        if LoraTrainingStrategy._last_dropout_value is not None and \
           abs(LoraTrainingStrategy._last_dropout_value - current_dropout) < 1e-6:
            return 0  # No change needed
        
        LoraTrainingStrategy._last_dropout_value = current_dropout

        updated = 0
        for module in model.modules():
            # PEFT stores dropout as module.lora_dropout, which may be:
            #   - nn.Dropout directly
            #   - nn.ModuleDict containing a 'default' key → nn.Dropout
            drop_attr = getattr(module, 'lora_dropout', None)
            if drop_attr is None:
                continue

            if isinstance(drop_attr, torch.nn.Dropout):
                drop_attr.p = float(current_dropout)
                updated += 1
            elif hasattr(drop_attr, 'default') and isinstance(drop_attr.default, torch.nn.Dropout):
                drop_attr.default.p = float(current_dropout)
                updated += 1

        return updated


def get_lora_training_stats(model, svd_sample_ratio: float = 0.2, svd_max_layers: int = 20) -> Dict[str, Any]:
    """
    Gather comprehensive LoRA training statistics for monitoring.
    
    Returns a dict with metrics useful for TensorBoard/W&B logging.

    Args:
        model: LoRA-enabled model
        svd_sample_ratio: Fraction of LoRA layers to run SVD on for effective-rank
            estimation (default 0.2). Full-model SVD is expensive for large models.
        svd_max_layers: Hard cap on number of layers for SVD (default 20).
    """
    stats = {
        'lora_enabled': getattr(model, 'lora_enabled', False),
        'total_params': 0,
        'trainable_params': 0,
        'lora_params': 0,
        'frozen_params': 0,
        'lora_modules': 0,
        'effective_rank_avg': 0.0,
        'norm_A_frobenius': 0.0,
        'norm_B_frobenius': 0.0,
    }

    norm_A_sum = 0.0
    norm_B_sum = 0.0
    lora_module_count = 0

    for name, param in model.named_parameters():
        stats['total_params'] += param.numel()
        if param.requires_grad:
            stats['trainable_params'] += param.numel()
        else:
            stats['frozen_params'] += param.numel()
        if "lora_" in name:
            stats['lora_params'] += param.numel()

    # First pass: collect LoRA modules and cheap stats (Frobenius norms).
    # Handles both PEFT <0.18 (direct attr with .weight) and PEFT >=0.18 (ModuleDict).
    def _extract_weights(attr):
        """Return a list of (A_weight_tensor,) from either a direct LoRA layer or ModuleDict."""
        if attr is None:
            return []
        if isinstance(attr, nn.ModuleDict):
            return [child.weight for child in attr.values() if hasattr(child, 'weight')]
        if hasattr(attr, 'weight'):
            return [attr.weight]
        return []

    lora_layers = []
    for module in model.modules():
        a_weights = _extract_weights(getattr(module, 'lora_A', None))
        b_weights = _extract_weights(getattr(module, 'lora_B', None))

        if a_weights:
            for A in a_weights:
                A_det = A.detach()
                norm_A_sum += torch.norm(A_det, p='fro').item()
                if A_det.dim() >= 2:
                    lora_layers.append(A_det)
            lora_module_count += 1

        if b_weights:
            for B in b_weights:
                norm_B_sum += torch.norm(B.detach(), p='fro').item()

    stats['lora_modules'] = lora_module_count
    if lora_module_count > 0:
        stats['norm_A_frobenius'] = norm_A_sum / lora_module_count
        stats['norm_B_frobenius'] = norm_B_sum / lora_module_count

        # Second pass: sampled SVD for effective rank (expensive operation).
        # Evenly sample across depth rather than random-sample so results are reproducible.
        if lora_layers:
            n_sample = min(svd_max_layers, max(1, int(len(lora_layers) * svd_sample_ratio)))
            step = max(1, len(lora_layers) // n_sample)
            sampled = lora_layers[::step][:n_sample]

            rank_values = []
            for A in sampled:
                try:
                    _, S, _ = torch.linalg.svd(A.float(), full_matrices=False)
                    if S.numel() == 0 or S[0].item() == 0:
                        continue
                    effective_rank = (S > 0.01 * S[0]).sum().item()
                    rank_values.append((A.shape[0], A.shape[1], effective_rank))
                except Exception as e:
                    LOGGER.debug(f"[LoRA-Stats] SVD failed on layer shape {tuple(A.shape)}: {e}")
                    continue

            if rank_values:
                avg_eff_rank = sum(r[2] for r in rank_values) / len(rank_values)
                avg_theoretical = sum(min(r[0], r[1]) for r in rank_values) / len(rank_values)
                stats['effective_rank_avg'] = avg_eff_rank / avg_theoretical if avg_theoretical > 0 else 0

    return stats


# Convenience import for math used in strategies
import math


def suggest_lora_config_for_dataset(
    num_images: Optional[int] = None,
    num_classes: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Return a LoRA hyperparameter recipe tuned to dataset scale.

    Returns a dict with recommended keys (``lora_r``, ``lora_alpha``,
    ``lora_lr_mult``, ``lora_layer_decay``, ``lora_alpha_warmup``,
    ``lora_ortho_weight``, ``lora_dropout``) plus a human-readable ``notes``
    string explaining the rationale.

    Empirical baseline (from project experiments):
        - VOC (16K+ images, 50ep, batch=128)        : LoRA r=16 beats Full SFT
        - African Wildlife (~1K images, 20ep)       : Full SFT ~= LoRA r=32
        - COCO128 (128 images, <10ep)               : LoRA not recommended

    Args:
        num_images: Training set image count. If None, no sizing advice is given.
        num_classes: Class count; used to estimate per-class sample density.
        epochs: Planned total training epochs.
        batch_size: Planned batch size.

    Returns:
        Dict of recommended hyperparameters + ``notes``.
    """
    rec: Dict[str, Any] = {
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_lr_mult": 2.0,
        "lora_layer_decay": 0.9,
        "lora_alpha_warmup": 3,
        "lora_ortho_weight": 0.0,
        "lora_dropout": 0.05,
        "lora_dropout_end": 0.15,
    }
    notes = []

    if num_images is None:
        notes.append("No num_images provided - returning generic medium-dataset defaults.")
        rec["notes"] = " ".join(notes)
        return rec

    per_class = (num_images / num_classes) if num_classes else None

    if num_images < 500 or (per_class is not None and per_class < 5):
        rec.update({
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_lr_mult": 2.0,
            "lora_layer_decay": 0.0,
            "lora_alpha_warmup": 0,
            "lora_ortho_weight": 0.0,
            "lora_dropout": 0.02,
        })
        notes.append(
            "Small-dataset regime: LoRA often underperforms Full SFT here. "
            "If LoRA is still desired, use rank=32+ and compare against Full SFT baseline (lora_r=0)."
        )
    elif num_images < 5000:
        rec.update({
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_lr_mult": 2.0,
            "lora_layer_decay": 0.9,
            "lora_alpha_warmup": 3,
            "lora_ortho_weight": 1e-4,
            "lora_dropout": 0.05,
        })
        notes.append("Small/medium regime: rank=32 with orthogonal regularization recommended.")
    elif num_images < 20000:
        rec.update({
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_lr_mult": 2.0,
            "lora_layer_decay": 0.9,
            "lora_alpha_warmup": 3,
            "lora_ortho_weight": 1e-4,
        })
        notes.append("Medium regime: LoRA typically matches or exceeds Full SFT here.")
    else:
        rec.update({
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_lr_mult": 2.0,
            "lora_layer_decay": 0.85,
            "lora_alpha_warmup": 5,
            "lora_ortho_weight": 1e-4,
        })
        notes.append("Large regime: LoRA or DoRA recommended; adapter efficiency peaks here.")

    if epochs is not None and epochs < 20:
        notes.append(f"[warn] epochs={epochs} is below recommended 20+ for LoRA convergence.")
    if batch_size is not None and batch_size < 16:
        notes.append(f"[warn] batch={batch_size} is small; gradient noise will hurt LoRA more than Full SFT.")

    rec["notes"] = " ".join(notes)
    return rec


__all__ = [
    'apply_lora',
    'get_lora_param_groups',
    'resolve_adalora_total_step',
    'select_lora_backend',
    'resolve_effective_lora_request',
    'save_lora_adapters',
    'load_lora_adapters',
    'merge_lora_weights',
    'LoRAConfig',
    'PeftProxy',
    'LoRADetectionModel',
    '_get_mps_memory',
    # Training Strategies
    'LoraTrainingStrategy',
    'get_lora_training_stats',
    'suggest_lora_config_for_dataset',
]
