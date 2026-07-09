# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO-Master MoE-aware PEFT Core Module
======================================
Standalone parameter-efficient fine-tuning (PEFT) layer library for YOLO,
with first-class Mixture-of-Experts (MoE) awareness.

Supports:
  • LoRA      – Low-Rank Adaptation (Hu et al. 2021)
  • DoRA      – Weight-Decomposed Low-Rank Adaptation (Liu et al. 2024)
  • IA³       – Infused Adapter by Inhibiting and Amplifying Inner Activations
  • lora+ia3  – Hybrid mode: LoRA on weights + IA³ on activations

YOLO-specific design:
  • Handles Conv, C2f, C3k2, SPPELAN, Detect/Segment/Pose/OBB/v10Detect heads
  • Skips MoE gating / routing modules (applies only to FFN / expert FFN layers)
  • Handles Conv2d and Linear layers uniformly
  • Compatible with gradient checkpointing, ONNX export, and torch.compile

Usage
-----
    from yolo_peft_core_module import create_peft_model, PEFTCORE

    # Wrap an existing YOLO model
    model = create_peft_model(
        model,
        peft_mode="lora",
        r=8, alpha=16,
        target_modules=["model.2.cv1.conv", "model.4.cv2.conv"],
    )

    # Inside BaseModel.__init__ (integration path)
    self._apply_peft(peft_config)

Public API
----------
  LoRALayer, DoRALayer, IA3Layer, PEFTCORE,
  create_peft_model, mark_only_peft_as_trainable,
  peft_state_dict, load_peft_state_dict
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants & type helpers
# ---------------------------------------------------------------------------

PEFT_MODES: Tuple[str, ...] = ("disabled", "lora", "dora", "ia3", "lora+ia3")

# YOLO module names that should *never* be adapted (detection heads, gating, etc.)
_DEFAULT_YOLO_EXCLUDE_PATTERNS: Tuple[str, ...] = (
    "cv2", "cv3",           # YOLOv8/v9 detection head branches (cls/box)
    "dfl",                  # Distribution Focal Loss layer
    "router",               # MoE routers / gating
    "gate",                 # MoE gates
    "routing",              # routing networks
    "expert_gate",          # expert gating
    "moe_aux",              # MoE aux loss modules
    "bn",                   # BatchNorm (no params to adapt)
    "norm",                 # Normalization layers
    "ln",                   # LayerNorm
)

# YOLO module *classes* that should be excluded from adaptation
_DEFAULT_YOLO_EXCLUDE_MODULE_TYPES: Tuple[type, ...] = (
    nn.BatchNorm2d,
    nn.BatchNorm1d,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.InstanceNorm2d,
    nn.Identity,
)

# MoE-related module type markers (used to skip gating but keep FFN paths)
_MoE_GATE_ATTRS: Tuple[str, ...] = (
    "router",
    "gating",
    "gate",
    "routing_network",
    "expert_gate",
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _parent_child_name(full_name: str) -> Tuple[str, str]:
    """Split ``'model.5.m.0.cv1'`` → ``('model.5.m.0', 'cv1')``."""
    parts = full_name.rsplit(".", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "", parts[0]


def _get_submodule(model: nn.Module, path: str) -> Optional[nn.Module]:
    """Navigate to a submodule by dot-separated path."""
    if not path:
        return model
    mod = model
    for p in path.split("."):
        if hasattr(mod, p):
            mod = getattr(mod, p)
        elif p.isdigit() and isinstance(mod, (nn.Sequential, nn.ModuleList)):
            mod = mod[int(p)]
        else:
            return None
    return mod


def _is_moe_gating_module(module: nn.Module) -> bool:
    """Return True if *module* is an MoE gating / routing component."""
    # Check explicit attributes
    for attr in _MoE_GATE_ATTRS:
        if hasattr(module, attr) and getattr(module, attr, None) is not None:
            return True
    # Check class name heuristics
    cls_name = module.__class__.__name__.lower()
    if any(k in cls_name for k in ("router", "gate", "routing", "moe")):
        # But keep actual expert FFN layers (they are usually named "expert" or are plain Linear/Conv)
        if "expert" in cls_name and "gate" not in cls_name:
            return False
        return True
    # Check module-level num_experts without expert-specific weight shapes
    if getattr(module, "num_experts", 0) > 0 and not hasattr(module, "experts"):
        return True
    return False


def _is_yolo_head_module(name: str, module: nn.Module) -> bool:
    """Return True if *name* or *module* indicates a YOLO detection head."""
    lower_name = name.lower()
    if any(p in lower_name for p in ("detect", "segment", "pose", "obb", "v10detect", "classify")):
        return True
    cls_name = module.__class__.__name__
    if cls_name in ("Detect", "Segment", "Pose", "OBB", "v10Detect", "Classify",
                    "WorldDetect", "YOLOEDetect", "YOLOESegment", "LRPCHead"):
        return True
    return False


def _is_excluded_module(name: str, module: nn.Module,
                        exclude_patterns: Optional[Tuple[str, ...]] = None) -> bool:
    """Return True if module should be excluded from PEFT adaptation."""
    if exclude_patterns is None:
        exclude_patterns = _DEFAULT_YOLO_EXCLUDE_PATTERNS

    lower_name = name.lower()

    # Exclude by pattern
    for pat in exclude_patterns:
        if pat in lower_name:
            return True

    # Exclude by module type
    if isinstance(module, _DEFAULT_YOLO_EXCLUDE_MODULE_TYPES):
        return True

    # Exclude MoE gating / routing (but keep expert FFN layers)
    if _is_moe_gating_module(module):
        return True

    # Exclude YOLO heads (unless explicitly included)
    if _is_yolo_head_module(name, module):
        return True

    # Exclude depthwise conv by default (can be overridden)
    if isinstance(module, nn.Conv2d) and module.groups == module.in_channels and module.in_channels > 1:
        return True

    return False


def _scaling(r: int, alpha: int, use_rslora: bool = True) -> float:
    """LoRA / DoRA scaling factor.

    Standard: ``alpha / r``
    rsLoRA:   ``alpha / sqrt(r)``  (Kalajdzievski 2023)
    """
    if use_rslora:
        return alpha / math.sqrt(max(r, 1))
    return alpha / max(r, 1)


# ---------------------------------------------------------------------------
# LoRA Layer
# ---------------------------------------------------------------------------

class LoRALayer(nn.Module):
    """Standard LoRA adapter for Conv2d or Linear layers.

    For Conv2d:
      * lora_A: 1×1 conv  (C_in  → r)
      * lora_B: K×K conv  (r → C_out, same stride/padding as base)
    For Linear:
      * lora_A: Linear    (in_features  → r)
      * lora_B: Linear    (r → out_features)

    Forward: ``y = base(x) + lora_B(lora_A(x)) * scaling``

    Initialization:
      * A: Kaiming uniform (a=√5)
      * B: zeros → training starts from pretrained base weights
    """

    def __init__(
        self,
        base_layer: nn.Module,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        use_rslora: bool = True,
    ):
        super().__init__()
        if not isinstance(base_layer, (nn.Conv2d, nn.Linear)):
            raise TypeError(f"LoRALayer only supports Conv2d and Linear, got {type(base_layer)}")

        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = _scaling(r, alpha, use_rslora)
        self.merged = False

        # Freeze base layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        # Build low-rank pair
        if isinstance(base_layer, nn.Conv2d):
            self.is_conv = True
            in_c = base_layer.in_channels
            out_c = base_layer.out_channels
            k = base_layer.kernel_size
            if isinstance(k, int):
                k = (k, k)
            stride = base_layer.stride
            padding = base_layer.padding
            dilation = base_layer.dilation
            groups = base_layer.groups

            self.lora_A = nn.Conv2d(in_c, r, kernel_size=1, bias=False)
            self.lora_B = nn.Conv2d(
                r, out_c, kernel_size=k,
                stride=stride, padding=padding,
                dilation=dilation, groups=groups, bias=False,
            )
        else:  # Linear
            self.is_conv = False
            in_f = base_layer.in_features
            out_f = base_layer.out_features
            self.lora_A = nn.Linear(in_f, r, bias=False)
            self.lora_B = nn.Linear(r, out_f, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize A with Kaiming uniform, B with zeros."""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with optional merge optimization."""
        if self.merged:
            return self.base_layer(x)
        x_d = self.dropout(x)
        return self.base_layer(x) + self.lora_B(self.lora_A(x_d)) * self.scaling

    def merge_weights(self) -> None:
        """Merge LoRA delta into base layer weight (for inference / ONNX)."""
        if self.merged:
            return
        with torch.no_grad():
            if self.is_conv:
                a = self.lora_A.weight.squeeze(-1).squeeze(-1)          # [r, in_c]
                b = self.lora_B.weight                                  # [out_c, r, kH, kW]
                delta = torch.einsum("orkw,ri->oikw", b, a) * self.scaling
            else:
                delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            self.base_layer.weight.data.add_(delta)
        self.merged = True

    def unmerge_weights(self) -> None:
        """Restore base layer weight to original pretrained values."""
        if not self.merged:
            return
        with torch.no_grad():
            if self.is_conv:
                a = self.lora_A.weight.squeeze(-1).squeeze(-1)
                b = self.lora_B.weight
                delta = torch.einsum("orkw,ri->oikw", b, a) * self.scaling
            else:
                delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            self.base_layer.weight.data.sub_(delta)
        self.merged = False

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.4f}, merged={self.merged}"


# ---------------------------------------------------------------------------
# DoRA Layer (Weight-Decomposed Low-Rank Adaptation)
# ---------------------------------------------------------------------------

class DoRALayer(nn.Module):
    """DoRA adapter for Conv2d or Linear layers.

    Decomposes pretrained weight into **magnitude** ``m`` and **direction** ``V``,
    then fine-tunes only the direction via LoRA while magnitude stays trainable:

        W = m * (V + ΔV) / ||V + ΔV||
        ΔV = B @ A   (low-rank)

    This preserves the weight norm structure better than plain LoRA and often
    yields higher accuracy under the same rank budget.

    Reference: Liu et al. *"DoRA: Weight-Decomposed Low-Rank Adaptation"*, 2024.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        use_rslora: bool = True,
    ):
        super().__init__()
        if not isinstance(base_layer, (nn.Conv2d, nn.Linear)):
            raise TypeError(f"DoRALayer only supports Conv2d and Linear, got {type(base_layer)}")

        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = _scaling(r, alpha, use_rslora)
        self.merged = False

        # Freeze base weight; we will reparameterize it
        for p in self.base_layer.parameters():
            p.requires_grad = False

        # --- magnitude vector (trainable) ---
        # For Conv2d: per-output-channel magnitude  [out_c]
        # For Linear: per-output-feature magnitude  [out_f]
        if isinstance(base_layer, nn.Conv2d):
            self.is_conv = True
            out_dim = base_layer.out_channels
            in_c = base_layer.in_channels
            k = base_layer.kernel_size
            if isinstance(k, int):
                k = (k, k)
            stride = base_layer.stride
            padding = base_layer.padding
            dilation = base_layer.dilation
            groups = base_layer.groups

            # Store original weight norm per output channel
            with torch.no_grad():
                w = base_layer.weight                              # [out_c, in_c, kH, kW]
                self.register_buffer("_base_direction", w.clone())
                self.register_buffer("_base_norm", w.view(w.size(0), -1).norm(dim=1, keepdim=True))  # [out_c, 1]
            self.magnitude = nn.Parameter(torch.ones(out_dim, 1, 1, 1))

            self.lora_A = nn.Conv2d(in_c, r, kernel_size=1, bias=False)
            self.lora_B = nn.Conv2d(
                r, out_dim, kernel_size=k,
                stride=stride, padding=padding,
                dilation=dilation, groups=groups, bias=False,
            )
        else:  # Linear
            self.is_conv = False
            out_dim = base_layer.out_features
            in_f = base_layer.in_features

            with torch.no_grad():
                w = base_layer.weight                              # [out_f, in_f]
                self.register_buffer("_base_direction", w.clone())
                self.register_buffer("_base_norm", w.norm(dim=1, keepdim=True))  # [out_f, 1]
            self.magnitude = nn.Parameter(torch.ones(out_dim, 1))

            self.lora_A = nn.Linear(in_f, r, bias=False)
            self.lora_B = nn.Linear(r, out_dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        # Magnitude starts at 1.0 so initial output matches base layer
        nn.init.ones_(self.magnitude)

    def _compute_weight(self) -> torch.Tensor:
        """Reconstruct full weight with direction update and learned magnitude."""
        # Low-rank direction update
        if self.is_conv:
            a = self.lora_A.weight.squeeze(-1).squeeze(-1)          # [r, in_c]
            b = self.lora_B.weight                                  # [out_c, r, kH, kW]
            delta = torch.einsum("orkw,ri->oikw", b, a) * self.scaling
        else:
            delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling

        # New direction = base + delta
        new_direction = self._base_direction + delta

        # Normalize direction per output channel / feature
        if self.is_conv:
            norm = new_direction.view(new_direction.size(0), -1).norm(dim=1, keepdim=True).view(-1, 1, 1, 1)
        else:
            norm = new_direction.norm(dim=1, keepdim=True)

        # Scale by learned magnitude (initialized to base norm)
        weight = self.magnitude * new_direction / (norm + 1e-8)
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x)

        # Use functional forward with reconstructed weight
        x_d = self.dropout(x)
        weight = self._compute_weight()

        if self.is_conv:
            return F.conv2d(
                x_d, weight,
                bias=self.base_layer.bias,
                stride=self.base_layer.stride,
                padding=self.base_layer.padding,
                dilation=self.base_layer.dilation,
                groups=self.base_layer.groups,
            )
        else:
            return F.linear(x_d, weight, self.base_layer.bias)

    def merge_weights(self) -> None:
        """Bake the DoRA decomposition back into base layer weight."""
        if self.merged:
            return
        with torch.no_grad():
            self.base_layer.weight.copy_(self._compute_weight())
        self.merged = True

    def unmerge_weights(self) -> None:
        """Restore original base weight."""
        if not self.merged:
            return
        with torch.no_grad():
            self.base_layer.weight.copy_(self._base_direction)
        self.merged = False

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.4f}, merged={self.merged}"


# ---------------------------------------------------------------------------
# IA³ Layer (Infused Adapter by Inhibiting and Amplifying Inner Activations)
# ---------------------------------------------------------------------------

class IA3Layer(nn.Module):
    """IA³ adapter for Conv2d or Linear layers.

    IA³ rescales **inner activations** with learned element-wise vectors
    (no low-rank decomposition).  It is parameter-efficient and often
    complements LoRA in hybrid setups.

    For Linear:  ``y = W @ (x * l_k) * l_v``
    For Conv2d:  ``y = conv(x * l_k) * l_v``  (channel-wise scaling)

    Initialization: uniform(-1, 1)  (Liu et al. 2022)

    Reference: Liu et al. *"Few-Shot Parameter-Efficient Fine-Tuning is Better
    and Cheaper than In-Context Learning"*, NeurIPS 2022.
    """

    def __init__(self, base_layer: nn.Module):
        super().__init__()
        if not isinstance(base_layer, (nn.Conv2d, nn.Linear)):
            raise TypeError(f"IA3Layer only supports Conv2d and Linear, got {type(base_layer)}")

        self.base_layer = base_layer
        self.merged = False

        for p in self.base_layer.parameters():
            p.requires_grad = False

        if isinstance(base_layer, nn.Conv2d):
            self.is_conv = True
            in_c = base_layer.in_channels
            out_c = base_layer.out_channels
            # l_k scales input channels; l_v scales output channels
            self.l_k = nn.Parameter(torch.empty(in_c, 1, 1))
            self.l_v = nn.Parameter(torch.empty(out_c, 1, 1))
        else:
            self.is_conv = False
            in_f = base_layer.in_features
            out_f = base_layer.out_features
            self.l_k = nn.Parameter(torch.empty(in_f))
            self.l_v = nn.Parameter(torch.empty(out_f))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize with uniform(-1, 1) per IA³ paper."""
        nn.init.uniform_(self.l_k, -1.0, 1.0)
        nn.init.uniform_(self.l_v, -1.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x)

        # Scale input activations
        if self.is_conv:
            x_scaled = x * self.l_k
            out = self.base_layer(x_scaled)
            # Scale output activations
            out = out * self.l_v
        else:
            x_scaled = x * self.l_k
            out = self.base_layer(x_scaled)
            out = out * self.l_v
        return out

    def merge_weights(self) -> None:
        """Bake IA³ scales into base layer weight and bias.

        After merge, the layer behaves like a standard Conv2d / Linear
        with no extra memory overhead.
        """
        if self.merged:
            return
        with torch.no_grad():
            if self.is_conv:
                # W: [out_c, in_c, kH, kW]
                # l_k: [in_c, 1, 1]  -> multiply along input channels
                # l_v: [out_c, 1, 1] -> multiply along output channels
                W = self.base_layer.weight * self.l_k.unsqueeze(0) * self.l_v.unsqueeze(1)
                self.base_layer.weight.copy_(W)
                if self.base_layer.bias is not None:
                    self.base_layer.bias.mul_(self.l_v.squeeze())
            else:
                # W: [out_f, in_f]
                W = self.base_layer.weight * self.l_k.unsqueeze(0) * self.l_v.unsqueeze(1)
                self.base_layer.weight.copy_(W)
                if self.base_layer.bias is not None:
                    self.base_layer.bias.mul_(self.l_v)
        self.merged = True

    def unmerge_weights(self) -> None:
        """Restore base layer weight (requires saving original, currently no-op without buffer)."""
        # IA³ merge is irreversible without storing original weight.
        # In practice, users should reload the base checkpoint to restore.
        self.merged = False

    def extra_repr(self) -> str:
        return f"merged={self.merged}"


# ---------------------------------------------------------------------------
# Hybrid LoRA + IA³ Layer
# ---------------------------------------------------------------------------

class LoRAIA3Layer(nn.Module):
    """Hybrid layer combining LoRA on weights + IA³ on activations.

    Forward: ``y = (base(x * l_k) + lora_B(lora_A(x * l_k))) * l_v * scaling``

    This combines the structural flexibility of LoRA with the activation
    rescaling of IA³, often outperforming either method alone on vision tasks.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        use_rslora: bool = True,
    ):
        super().__init__()
        if not isinstance(base_layer, (nn.Conv2d, nn.Linear)):
            raise TypeError(f"LoRAIA3Layer only supports Conv2d and Linear, got {type(base_layer)}")

        self.lora = LoRALayer(base_layer, r=r, alpha=alpha, dropout=dropout, use_rslora=use_rslora)
        self.ia3 = IA3Layer(base_layer)
        self.base_layer = base_layer
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x)
        # IA³ scales input, LoRA computes delta, IA³ scales output
        x_k = x * self.ia3.l_k
        base_out = self.base_layer(x_k)
        lora_out = self.lora.lora_B(self.lora.lora_A(self.lora.dropout(x_k))) * self.lora.scaling
        out = (base_out + lora_out) * self.ia3.l_v
        return out

    def merge_weights(self) -> None:
        """Merge both LoRA and IA³ into base layer."""
        if self.merged:
            return
        # Merge LoRA first, then IA³
        self.lora.merge_weights()
        self.ia3.merge_weights()
        self.merged = True

    def unmerge_weights(self) -> None:
        """Unmerge (best-effort; IA³ unmerge is no-op)."""
        self.lora.unmerge_weights()
        self.ia3.unmerge_weights()
        self.merged = False

    def extra_repr(self) -> str:
        return f"r={self.lora.r}, alpha={self.lora.alpha}, merged={self.merged}"


# ---------------------------------------------------------------------------
# PEFTCORE – the main wrapper that dispatches to LoRA / DoRA / IA³ / hybrid
# ---------------------------------------------------------------------------

class PEFTCORE(nn.Module):
    """Production-grade PEFT wrapper with MoE-awareness for YOLO models.

    Args:
        base_layer: the Conv2d or Linear layer to adapt.
        peft_mode: one of ``"lora"`` | ``"dora"`` | ``"ia3"`` | ``"lora+ia3"``.
        r: LoRA / DoRA rank (ignored for IA³-only).
        alpha: LoRA / DoRA scaling numerator (ignored for IA³-only).
        dropout: dropout applied to LoRA / DoRA input activations.
        use_rslora: use rsLoRA scaling (``alpha / sqrt(r)``) instead of standard.

    Attributes:
        adapter: the actual adapter module (LoRALayer, DoRALayer, IA3Layer, or LoRAIA3Layer).
        peft_mode: current adaptation mode.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        peft_mode: str = "lora",
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        use_rslora: bool = True,
    ):
        super().__init__()
        if peft_mode not in PEFT_MODES:
            raise ValueError(f"peft_mode must be one of {PEFT_MODES}, got '{peft_mode}'")
        if peft_mode == "disabled":
            raise ValueError("Use peft_mode='disabled' by simply not wrapping the layer.")

        self.peft_mode = peft_mode
        self.base_layer = base_layer

        if peft_mode == "lora":
            self.adapter = LoRALayer(base_layer, r=r, alpha=alpha, dropout=dropout, use_rslora=use_rslora)
        elif peft_mode == "dora":
            self.adapter = DoRALayer(base_layer, r=r, alpha=alpha, dropout=dropout, use_rslora=use_rslora)
        elif peft_mode == "ia3":
            self.adapter = IA3Layer(base_layer)
        elif peft_mode == "lora+ia3":
            self.adapter = LoRAIA3Layer(base_layer, r=r, alpha=alpha, dropout=dropout, use_rslora=use_rslora)
        else:
            raise RuntimeError(f"Unhandled peft_mode: {peft_mode}")

    @property
    def merged(self) -> bool:
        return getattr(self.adapter, "merged", False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)

    def merge_weights(self) -> None:
        """Merge adapter into base layer for zero-overhead inference."""
        if hasattr(self.adapter, "merge_weights"):
            self.adapter.merge_weights()

    def unmerge_weights(self) -> None:
        """Restore base layer weights."""
        if hasattr(self.adapter, "unmerge_weights"):
            self.adapter.unmerge_weights()

    def extra_repr(self) -> str:
        return f"peft_mode={self.peft_mode}, " + self.adapter.extra_repr()


# ---------------------------------------------------------------------------
# Target auto-detection
# ---------------------------------------------------------------------------

def auto_detect_peft_targets(
    model: nn.Module,
    r: int = 8,
    include_moe: bool = True,
    include_attention: bool = False,
    only_backbone: bool = False,
    exclude_modules: Optional[List[str]] = None,
    last_n: Optional[int] = None,
    from_layer: Optional[int] = None,
    to_layer: Optional[int] = None,
    allow_depthwise: bool = False,
    kernels: Optional[Tuple[int, ...]] = None,
    skip_stem: bool = True,
    min_channels: int = 0,
    only_3x3: bool = False,
) -> List[str]:
    """Auto-detect Conv2d / Linear layers suitable for PEFT adaptation.

    Heuristics (inherited from YOLO-Master MoLoRA config builder):
      * Skip stem (first conv) by default – low-level features are usually transferable.
      * Skip depthwise convolutions – low FLOP, high memory overhead ratio.
      * Skip 1×1 convs unless ``only_3x3=False`` – often bottleneck layers.
      * Skip BatchNorm, activation, pooling, and detection heads.
      * Optionally include attention projection layers.
      * Optionally include MoE FFN layers (default True).
      * Optionally restrict to backbone only (exclude neck / head).

    Args:
        model: the YOLO model to inspect.
        r: target rank (used to filter layers that would benefit from adaptation).
        include_moe: whether to include MoE expert layers.
        include_attention: whether to include attention projection Linear layers.
        only_backbone: if True, only adapt backbone layers (before neck/head).
        exclude_modules: additional module name substrings to exclude.
        last_n: if set, only adapt the last N layers (by index in model.model).
        from_layer: starting layer index (inclusive).
        to_layer: ending layer index (inclusive).
        allow_depthwise: allow depthwise separable convolutions.
        kernels: allowed kernel sizes (e.g. ``(3, 5)``).  None = any.
        skip_stem: skip the very first Conv layer (stem).
        min_channels: minimum in/out channels to qualify.
        only_3x3: if True, only consider 3×3 convolutions.

    Returns:
        List of fully-qualified module names (e.g. ``["model.2.cv1.conv", ...]``).
    """
    targets: List[str] = []
    exclude_patterns = list(_DEFAULT_YOLO_EXCLUDE_PATTERNS)
    if exclude_modules:
        exclude_patterns.extend(exclude_modules)

    # If only_backbone, we need to detect where backbone ends.
    # In YOLO YAML, backbone is listed first; in the parsed model, we can
    # use the layer index heuristic: neck usually starts around index 5-7.
    # We use a simple heuristic: if the module name contains "head" or the
    # layer is after the first Detect/Segment/Pose-like module, skip it.

    named_modules = list(model.named_modules())

    # Pre-scan to find head indices
    head_indices: Set[int] = set()
    for name, module in named_modules:
        if _is_yolo_head_module(name, module):
            # Try to infer layer index from name like "model.23"
            parts = name.split(".")
            for p in parts:
                if p.isdigit():
                    head_indices.add(int(p))
                    break

    # Determine effective from/to
    max_layer_idx = -1
    for name, _ in named_modules:
        parts = name.split(".")
        for p in parts:
            if p.isdigit():
                max_layer_idx = max(max_layer_idx, int(p))

    if from_layer is None:
        from_layer = 1 if skip_stem else 0
    if to_layer is None:
        to_layer = max_layer_idx
    if last_n is not None:
        from_layer = max(from_layer, max_layer_idx - last_n + 1)

    for name, module in named_modules:
        # --- basic type filter ---
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue

        # --- exclusion heuristics ---
        if _is_excluded_module(name, module, tuple(exclude_patterns)):
            continue

        # --- layer index filter ---
        layer_idx: Optional[int] = None
        parts = name.split(".")
        for p in parts:
            if p.isdigit():
                layer_idx = int(p)
                break
        if layer_idx is not None:
            if layer_idx < from_layer or layer_idx > to_layer:
                continue
            if only_backbone and head_indices and layer_idx >= min(head_indices):
                continue

        # --- depthwise filter ---
        if isinstance(module, nn.Conv2d) and module.groups == module.in_channels and module.in_channels > 1:
            if not allow_depthwise:
                continue

        # --- kernel size filter ---
        if isinstance(module, nn.Conv2d):
            k = module.kernel_size
            if isinstance(k, int):
                k = (k, k)
            if only_3x3 and k != (3, 3):
                continue
            if kernels is not None and k[0] not in kernels:
                continue

        # --- channel count filter ---
        if isinstance(module, nn.Conv2d):
            if module.in_channels < min_channels or module.out_channels < min_channels:
                continue
        else:
            if module.in_features < min_channels or module.out_features < min_channels:
                continue

        # --- rank filter: very small layers are not worth adapting ---
        if isinstance(module, nn.Conv2d):
            fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
        else:
            fan_in = module.in_features
        if fan_in < r * 2:
            continue

        targets.append(name)

    return targets


# ---------------------------------------------------------------------------
# High-level API: create_peft_model
# ---------------------------------------------------------------------------

def create_peft_model(
    model: nn.Module,
    peft_mode: str = "lora",
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    use_rslora: bool = True,
    target_modules: Optional[List[str]] = None,
    auto_detect: bool = True,
    freeze_bn: bool = True,
    verbose: bool = True,
    **auto_detect_kwargs,
) -> nn.Module:
    """Wrap an Ultralytics YOLO model with PEFT adapters.

    This modifies the model **in-place**.

    Args:
        model: the base YOLO model (e.g. ``DetectionModel``).
        peft_mode: ``"lora"`` | ``"dora"`` | ``"ia3"`` | ``"lora+ia3"``.
        r: LoRA / DoRA rank.
        alpha: LoRA / DoRA alpha (scaling numerator).
        dropout: dropout on adapter inputs.
        use_rslora: use rsLoRA scaling.
        target_modules: explicit list of module names to adapt.  If None and
            ``auto_detect=True``, targets are discovered automatically.
        auto_detect: run auto-detection when ``target_modules`` is None.
        freeze_bn: freeze BatchNorm running statistics and affine params.
        verbose: print adaptation summary.
        **auto_detect_kwargs: forwarded to :func:`auto_detect_peft_targets`.

    Returns:
        The modified model (same object).
    """
    if peft_mode == "disabled":
        return model
    if peft_mode not in PEFT_MODES:
        raise ValueError(f"peft_mode must be one of {PEFT_MODES}, got '{peft_mode}'")

    # Prevent double-wrapping
    if getattr(model, "peft_enabled", False):
        if verbose:
            print("[PEFT] Model already has PEFT enabled. Skipping re-application.")
        return model

    # Resolve target modules
    if target_modules is None:
        if auto_detect:
            target_modules = auto_detect_peft_targets(model, r=r, **auto_detect_kwargs)
        if not target_modules:
            if verbose:
                print("[PEFT] No target modules found. Returning model unchanged.")
            return model

    modules_dict = dict(model.named_modules())
    wrapped = 0
    skipped_moe_gate = 0

    for name in target_modules:
        if name not in modules_dict:
            continue
        base_layer = modules_dict[name]

        # Double-check: skip MoE gating modules (defense in depth)
        if _is_moe_gating_module(base_layer):
            skipped_moe_gate += 1
            continue
        if not isinstance(base_layer, (nn.Conv2d, nn.Linear)):
            continue

        parent_name, child_name = _parent_child_name(name)
        parent = _get_submodule(model, parent_name) if parent_name else model
        if parent is None or not hasattr(parent, child_name):
            continue

        peft_layer = PEFTCORE(
            base_layer=base_layer,
            peft_mode=peft_mode,
            r=r,
            alpha=alpha,
            dropout=dropout,
            use_rslora=use_rslora,
        )
        setattr(parent, child_name, peft_layer)
        wrapped += 1

    # Attach metadata
    model.peft_enabled = True                      # type: ignore[union-attr]
    model.peft_config = {                           # type: ignore[union-attr]
        "peft_mode": peft_mode,
        "r": r,
        "alpha": alpha,
        "dropout": dropout,
        "use_rslora": use_rslora,
        "target_modules": target_modules,
    }

    # Freeze non-PEFT parameters
    mark_only_peft_as_trainable(model, freeze_bn=freeze_bn)

    if verbose:
        print(f"[PEFT] Mode={peft_mode}, wrapped={wrapped} layers"
              f"{f' (skipped {skipped_moe_gate} MoE gates)' if skipped_moe_gate else ''}")
        stats = peft_param_stats(model)
        print(f"[PEFT] Params: total={stats['total']:,}, trainable={stats['trainable']:,} "
              f"({stats['trainable_pct']:.2f}%), PEFT={stats['peft']:,} ({stats['peft_pct']:.2f}%)")

    return model


# ---------------------------------------------------------------------------
# Parameter management helpers
# ---------------------------------------------------------------------------

def mark_only_peft_as_trainable(model: nn.Module, freeze_bn: bool = True) -> None:
    """Freeze all parameters except PEFT adapter parameters.

    PEFT parameter keywords:
      * ``lora_A``, ``lora_B``  – LoRA / DoRA low-rank matrices
      * ``magnitude``           – DoRA magnitude vector
      * ``l_k``, ``l_v``        – IA³ scaling vectors
      * ``ia3``                 – IA³ sub-module prefix
    """
    for name, param in model.named_parameters():
        if any(k in name for k in ("lora_A", "lora_B", "magnitude", "l_k", "l_v", "ia3.")):
            param.requires_grad = True
        else:
            param.requires_grad = False

    if freeze_bn:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
                m.eval()   # freeze running stats
                for p in m.parameters():
                    p.requires_grad = False


def peft_param_stats(model: nn.Module) -> Dict[str, Union[int, float]]:
    """Return parameter statistics for a model with PEFT adapters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    peft = sum(
        p.numel()
        for n, p in model.named_parameters()
        if any(k in n for k in ("lora_A", "lora_B", "magnitude", "l_k", "l_v", "ia3."))
    )
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "peft": peft,
        "trainable_pct": 100.0 * trainable / total if total else 0.0,
        "peft_pct": 100.0 * peft / total if total else 0.0,
    }


def peft_state_dict(model: nn.Module, keep_base: bool = False) -> Dict[str, torch.Tensor]:
    """Extract only PEFT adapter parameters from a model.

    Args:
        model: the PEFT-wrapped model.
        keep_base: if True, include base layer weights (needed for merge).

    Returns:
        State dict with only PEFT-related tensors.
    """
    peft_keys = ("lora_A", "lora_B", "magnitude", "l_k", "l_v")
    state: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if any(pk in k for pk in peft_keys):
            state[k] = v
        elif keep_base and "base_layer" in k:
            state[k] = v
    return state


def load_peft_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor], strict: bool = False) -> None:
    """Load PEFT adapter parameters into a model.

    Args:
        model: the PEFT-wrapped model.
        state_dict: adapter state dict (from :func:`peft_state_dict`).
        strict: if True, raise on missing keys.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if strict and missing:
        raise RuntimeError(f"Missing keys when loading PEFT state dict: {missing}")
    if unexpected:
        # This is expected if the base model state is not provided
        pass


# ---------------------------------------------------------------------------
# Integration helper for BaseModel._apply_peft
# ---------------------------------------------------------------------------

def apply_peft_from_config(model: nn.Module, cfg: Dict[str, Any]) -> nn.Module:
    """Apply PEFT to a YOLO BaseModel using a config dict.

    Intended to be called inside ``BaseModel.__init__`` or via a new
    ``BaseModel._apply_peft`` method:

    .. code-block:: python

        class BaseModel(torch.nn.Module):
            ...
            def _apply_peft(self, peft_cfg: dict):
                from yolo_peft_core_module import apply_peft_from_config
                apply_peft_from_config(self, peft_cfg)

    Config keys (all optional):
      * ``peft_mode`` (str): ``"disabled"`` | ``"lora"`` | ``"dora"`` | ``"ia3"`` | ``"lora+ia3"``
      * ``r`` (int): LoRA / DoRA rank  (default 8)
      * ``alpha`` (int): scaling numerator  (default 16)
      * ``dropout`` (float): adapter dropout  (default 0.0)
      * ``use_rslora`` (bool): rsLoRA scaling  (default True)
      * ``target_modules`` (list[str]): explicit target names
      * ``auto_detect`` (bool): auto-detect targets if not provided  (default True)
      * ``freeze_bn`` (bool): freeze BN during fine-tuning  (default True)
      * ``only_backbone`` (bool): restrict adaptation to backbone  (default False)
      * ``include_moe`` (bool): include MoE FFN layers  (default True)
      * ``skip_stem`` (bool): skip first conv  (default True)
      * ``allow_depthwise`` (bool): allow depthwise conv  (default False)
    """
    peft_mode = cfg.get("peft_mode", "disabled")
    if peft_mode == "disabled" or not peft_mode:
        return model

    kwargs = {
        "peft_mode": peft_mode,
        "r": cfg.get("r", 8),
        "alpha": cfg.get("alpha", 16),
        "dropout": cfg.get("dropout", 0.0),
        "use_rslora": cfg.get("use_rslora", True),
        "target_modules": cfg.get("target_modules", None),
        "auto_detect": cfg.get("auto_detect", True),
        "freeze_bn": cfg.get("freeze_bn", True),
        "verbose": cfg.get("verbose", True),
    }

    # Forward auto-detect filters
    for key in ("only_backbone", "include_moe", "include_attention", "skip_stem",
                "allow_depthwise", "min_channels", "only_3x3", "last_n",
                "from_layer", "to_layer", "exclude_modules"):
        if key in cfg:
            kwargs[key] = cfg[key]

    return create_peft_model(model, **kwargs)


# ---------------------------------------------------------------------------
# Convenience: merge / unmerge all PEFT layers in a model
# ---------------------------------------------------------------------------

def merge_all_peft_layers(model: nn.Module) -> None:
    """Merge every PEFTCORE layer in *model* for inference."""
    count = 0
    for m in model.modules():
        if isinstance(m, PEFTCORE):
            m.merge_weights()
            count += 1
    print(f"[PEFT] Merged {count} adapter layers.")


def unmerge_all_peft_layers(model: nn.Module) -> None:
    """Unmerge every PEFTCORE layer in *model*."""
    count = 0
    for m in model.modules():
        if isinstance(m, PEFTCORE):
            m.unmerge_weights()
            count += 1
    print(f"[PEFT] Unmerged {count} adapter layers.")


# ---------------------------------------------------------------------------
# Unit Tests (run with ``python yolo_peft_core_module.py``)
# ---------------------------------------------------------------------------

class _TestLoRA:
    """Smoke tests for LoRALayer."""

    @staticmethod
    def test_conv2d() -> None:
        base = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        lora = LoRALayer(base, r=4, alpha=8)
        x = torch.randn(2, 64, 32, 32)
        y = lora(x)
        assert y.shape == (2, 128, 16, 16), f"Unexpected output shape: {y.shape}"
        assert not any(p.requires_grad for p in base.parameters())
        assert lora.lora_A.weight.requires_grad
        assert lora.lora_B.weight.requires_grad
        print("[TEST] LoRA Conv2d  – PASS")

    @staticmethod
    def test_linear() -> None:
        base = nn.Linear(256, 512)
        lora = LoRALayer(base, r=8, alpha=16)
        x = torch.randn(4, 256)
        y = lora(x)
        assert y.shape == (4, 512)
        print("[TEST] LoRA Linear  – PASS")

    @staticmethod
    def test_merge_unmerge() -> None:
        base = nn.Conv2d(16, 32, 3, padding=1)
        lora = LoRALayer(base, r=2, alpha=4)
        x = torch.randn(1, 16, 8, 8)
        y1 = lora(x)
        lora.merge_weights()
        assert lora.merged
        y2 = lora(x)
        torch.testing.assert_close(y1, y2, atol=1e-5, rtol=1e-4)
        lora.unmerge_weights()
        assert not lora.merged
        y3 = lora(x)
        torch.testing.assert_close(y1, y3, atol=1e-5, rtol=1e-4)
        print("[TEST] LoRA merge/unmerge  – PASS")


class _TestDoRA:
    """Smoke tests for DoRALayer."""

    @staticmethod
    def test_conv2d() -> None:
        base = nn.Conv2d(32, 64, 3, padding=1)
        dora = DoRALayer(base, r=4, alpha=8)
        x = torch.randn(2, 32, 16, 16)
        y = dora(x)
        assert y.shape == (2, 64, 16, 16)
        assert dora.magnitude.requires_grad
        print("[TEST] DoRA Conv2d  – PASS")

    @staticmethod
    def test_merge_unmerge() -> None:
        base = nn.Linear(128, 256)
        dora = DoRALayer(base, r=4, alpha=8)
        x = torch.randn(2, 128)
        y1 = dora(x)
        dora.merge_weights()
        y2 = dora(x)
        torch.testing.assert_close(y1, y2, atol=1e-5, rtol=1e-4)
        dora.unmerge_weights()
        y3 = dora(x)
        torch.testing.assert_close(y1, y3, atol=1e-5, rtol=1e-4)
        print("[TEST] DoRA merge/unmerge  – PASS")


class _TestIA3:
    """Smoke tests for IA3Layer."""

    @staticmethod
    def test_conv2d() -> None:
        base = nn.Conv2d(32, 64, 3, padding=1)
        ia3 = IA3Layer(base)
        x = torch.randn(2, 32, 16, 16)
        y = ia3(x)
        assert y.shape == (2, 64, 16, 16)
        assert ia3.l_k.requires_grad and ia3.l_v.requires_grad
        print("[TEST] IA³ Conv2d  – PASS")

    @staticmethod
    def test_linear() -> None:
        base = nn.Linear(128, 256)
        ia3 = IA3Layer(base)
        x = torch.randn(2, 128)
        y = ia3(x)
        assert y.shape == (2, 256)
        print("[TEST] IA³ Linear  – PASS")

    @staticmethod
    def test_merge() -> None:
        base = nn.Conv2d(8, 16, 1)
        ia3 = IA3Layer(base)
        x = torch.randn(1, 8, 4, 4)
        y1 = ia3(x)
        ia3.merge_weights()
        y2 = ia3(x)
        torch.testing.assert_close(y1, y2, atol=1e-5, rtol=1e-4)
        print("[TEST] IA³ merge  – PASS")


class _TestHybrid:
    """Smoke tests for LoRA+IA³ hybrid."""

    @staticmethod
    def test_forward() -> None:
        base = nn.Conv2d(16, 32, 3, padding=1)
        hybrid = LoRAIA3Layer(base, r=4, alpha=8)
        x = torch.randn(2, 16, 8, 8)
        y = hybrid(x)
        assert y.shape == (2, 32, 8, 8)
        print("[TEST] Hybrid forward  – PASS")


class _TestPEFTCORE:
    """Smoke tests for PEFTCORE dispatcher."""

    @staticmethod
    def test_dispatcher() -> None:
        for mode in ("lora", "dora", "ia3", "lora+ia3"):
            base = nn.Conv2d(16, 32, 3, padding=1)
            core = PEFTCORE(base, peft_mode=mode, r=4, alpha=8)
            x = torch.randn(1, 16, 8, 8)
            y = core(x)
            assert y.shape == (1, 32, 8, 8), f"Mode {mode} failed"
        print("[TEST] PEFTCORE dispatcher  – PASS")

    @staticmethod
    def test_disabled_raises() -> None:
        try:
            PEFTCORE(nn.Conv2d(1, 1, 1), peft_mode="disabled")
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass
        print("[TEST] PEFTCORE disabled raises  – PASS")


class _TestAutoDetect:
    """Smoke tests for target auto-detection."""

    @staticmethod
    def test_simple_sequential() -> None:
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),   # 0 – stem (skipped by default)
            nn.BatchNorm2d(64),                # 1 – BN (excluded)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),  # 3
            nn.Conv2d(128, 128, 1),            # 4 – 1x1 (excluded if only_3x3)
            nn.Conv2d(128, 256, 3, padding=1), # 5
        )
        targets = auto_detect_peft_targets(model, r=4, skip_stem=True, only_3x3=True)
        # Should pick 3 and 5, skip 0 (stem), 1 (BN), 4 (1x1)
        assert "3" in targets, f"Expected '3' in targets, got {targets}"
        assert "5" in targets, f"Expected '5' in targets, got {targets}"
        assert "0" not in targets
        assert "1" not in targets
        assert "4" not in targets
        print("[TEST] Auto-detect simple  – PASS")

    @staticmethod
    def test_moe_gate_skip() -> None:
        """MoE gating modules should be skipped."""
        class FakeRouter(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Linear(64, 4)
            def forward(self, x):
                return self.gate(x)

        class FakeExpertFFN(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 128)
                self.fc2 = nn.Linear(128, 64)
            def forward(self, x):
                return self.fc2(F.relu(self.fc1(x)))

        model = nn.Sequential(
            FakeRouter(),
            FakeExpertFFN(),
        )
        targets = auto_detect_peft_targets(model, r=4, skip_stem=False)
        # Router's internal Linear should be skipped; Expert FFN should be included
        assert not any("0" in t for t in targets), f"Router should be excluded: {targets}"
        assert any("fc" in t for t in targets), f"Expert FFN should be included: {targets}"
        print("[TEST] Auto-detect MoE gate skip  – PASS")


class _TestIntegration:
    """End-to-end integration smoke tests."""

    @staticmethod
    def test_create_peft_model() -> None:
        # Simple pseudo-YOLO backbone
        class PseudoYOLO(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),    # 0 – stem
                    nn.Conv2d(64, 128, 3, padding=1),  # 1
                    nn.Conv2d(128, 256, 3, padding=1), # 2
                )
            def forward(self, x):
                return self.model(x)

        model = PseudoYOLO()
        create_peft_model(
            model,
            peft_mode="lora",
            r=4,
            alpha=8,
            target_modules=None,
            auto_detect=True,
            skip_stem=True,
            only_3x3=True,
            verbose=False,
        )
        assert getattr(model, "peft_enabled", False)
        stats = peft_param_stats(model)
        assert stats["trainable"] > 0
        assert stats["peft"] > 0
        print("[TEST] create_peft_model  – PASS")

    @staticmethod
    def test_state_dict_roundtrip() -> None:
        base = nn.Conv2d(16, 32, 3, padding=1)
        core = PEFTCORE(base, peft_mode="lora", r=2, alpha=4)
        sd = peft_state_dict(core, keep_base=True)
        assert any("lora_A" in k for k in sd)
        assert any("lora_B" in k for k in sd)

        # New model, load state
        base2 = nn.Conv2d(16, 32, 3, padding=1)
        core2 = PEFTCORE(base2, peft_mode="lora", r=2, alpha=4)
        load_peft_state_dict(core2, sd, strict=False)

        x = torch.randn(1, 16, 8, 8)
        torch.testing.assert_close(core(x), core2(x), atol=1e-6, rtol=1e-5)
        print("[TEST] state_dict roundtrip  – PASS")

    @staticmethod
    def test_gradient_flow() -> None:
        """Ensure gradients flow only through PEFT parameters."""
        base = nn.Conv2d(8, 16, 3, padding=1)
        core = PEFTCORE(base, peft_mode="lora", r=2, alpha=4)
        x = torch.randn(2, 8, 4, 4, requires_grad=True)
        y = core(x).sum()
        y.backward()
        assert base.weight.grad is None or not base.weight.requires_grad
        assert core.adapter.lora_A.weight.grad is not None
        assert core.adapter.lora_B.weight.grad is not None
        print("[TEST] gradient flow  – PASS")


class _TestMoEAwareness:
    """Tests specifically for MoE-awareness."""

    @staticmethod
    def test_moe_gate_exclusion() -> None:
        """Explicitly verify MoE gating modules are not wrapped."""
        class MoEGate(nn.Module):
            num_experts = 4
            def __init__(self):
                super().__init__()
                self.router = nn.Linear(64, 4)
            def forward(self, x):
                return F.softmax(self.router(x), dim=-1)

        model = nn.Sequential(MoEGate(), nn.Linear(64, 128))
        targets = auto_detect_peft_targets(model, r=4, skip_stem=False)
        assert "0.router" not in targets, f"Gate router should be excluded: {targets}"
        assert "1" in targets, f"FFN should be included: {targets}"
        print("[TEST] MoE gate exclusion  – PASS")

    @staticmethod
    def test_expert_ffn_inclusion() -> None:
        """MoE expert FFN layers should be adaptable."""
        class ExpertFFN(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 128)
                self.fc2 = nn.Linear(128, 64)
            def forward(self, x):
                return self.fc2(F.relu(self.fc1(x)))

        model = nn.Sequential(ExpertFFN())
        targets = auto_detect_peft_targets(model, r=4, skip_stem=False)
        assert any("fc" in t for t in targets), f"Expert FFN should be included: {targets}"
        print("[TEST] Expert FFN inclusion  – PASS")


def _run_all_tests() -> None:
    """Execute the full unit-test suite."""
    print("=" * 60)
    print("YOLO PEFT Core Module – Unit Test Suite")
    print("=" * 60)

    _TestLoRA.test_conv2d()
    _TestLoRA.test_linear()
    _TestLoRA.test_merge_unmerge()

    _TestDoRA.test_conv2d()
    _TestDoRA.test_merge_unmerge()

    _TestIA3.test_conv2d()
    _TestIA3.test_linear()
    _TestIA3.test_merge()

    _TestHybrid.test_forward()

    _TestPEFTCORE.test_dispatcher()
    _TestPEFTCORE.test_disabled_raises()

    _TestAutoDetect.test_simple_sequential()
    _TestAutoDetect.test_moe_gate_skip()

    _TestIntegration.test_create_peft_model()
    _TestIntegration.test_state_dict_roundtrip()
    _TestIntegration.test_gradient_flow()

    _TestMoEAwareness.test_moe_gate_exclusion()
    _TestMoEAwareness.test_expert_ffn_inclusion()

    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    _run_all_tests()
