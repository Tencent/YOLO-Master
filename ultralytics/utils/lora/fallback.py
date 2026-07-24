# 🐧Please note that this file has been modified by Tencent on 2026/02/13. All Tencent Modifications are Copyright (C) 2026 Tencent.
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from ultralytics.nn.tasks import (
    ClassificationModel,
    DetectionModel,
    OBBModel,
    PoseModel,
    RTDETRDetectionModel,
    SegmentationModel,
    WorldModel,
)
from ultralytics.utils import LOGGER
from .api import (
    PEFT_AVAILABLE,
    PeftModel,
    _effective_peft_variant,
    _fast_parse_int_list,
    _fast_parse_str_list,
    _normalize_lora_init,
    _unfreeze_detection_head,
    build_lora_target_audit,
    resolve_effective_lora_request,
)


def _fallback_lora_scaling(alpha: int, r: int, use_rslora: bool) -> float:
    """Return the effective fallback LoRA scale; direct wrappers default to historical standard LoRA."""
    if r <= 0:
        raise ValueError(f"Fallback LoRA rank must be positive, got r={r}.")
    return alpha / math.sqrt(r) if use_rslora else alpha / r


class FewShotLoRAConv(nn.Module):
    """LoRA wrapper optimized for few-shot learning.

    Enhancements over ManualLoRAConv:
    - Scheduled DropConnect: curriculum-style rate scheduling (cosine/linear/exp)
    - Gradient-Importance Weighted DropConnect: Fisher-based connection importance
    - Knowledge distillation support: accepts teacher features for alignment
    - Adaptive rank scaling: adjusts effective rank based on data scarcity
    - Variational rank selection: Gumbel-Softmax based sparse rank (optional)
    """

    def __init__(self, conv: nn.Conv2d, r: int = 8, alpha: int = 16,
                 dropout: float = 0.0, dropconnect: float = 0.1,
                 adaptive_rank: bool = True,
                 dropconnect_schedule: str = "constant",
                 dropconnect_max: float = 0.3,
                 dropconnect_min: float = 0.0,
                 gradient_importance_weighted: bool = False,
                 variational_rank: bool = False,
                 rank_budget: float = 0.5,
                 use_rslora: bool = False):
        super().__init__()
        self.conv = conv
        self.r = r
        self.alpha = alpha
        self.use_rslora = bool(use_rslora)
        self.scaling = _fallback_lora_scaling(alpha, r, self.use_rslora)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.dropconnect_rate = dropconnect
        self.adaptive_rank = adaptive_rank

        # Scheduled DropConnect config
        self.dropconnect_schedule = dropconnect_schedule
        self.dropconnect_max = dropconnect_max
        self.dropconnect_min = dropconnect_min

        # Gradient-Importance Weighted DropConnect
        self.gradient_importance_weighted = gradient_importance_weighted
        self.importance_ema_decay = 0.9
        self.importance_A = None  # EMA of grad_A^2
        self.importance_B = None  # EMA of grad_B^2

        # Variational rank config
        self.variational_rank = variational_rank
        self.rank_budget = rank_budget

        groups = conv.groups
        if groups > 1 and (r % groups != 0):
            raise ValueError(
                f"FewShotLoRAConv: rank r={r} must be a multiple of groups={groups}"
            )
        self.groups = groups
        self.r_per_group = r // max(groups, 1)

        in_per_group = (conv.in_channels // groups) * conv.kernel_size[0] * conv.kernel_size[1]
        out_per_group = conv.out_channels // groups
        factory_kwargs = {"device": conv.weight.device, "dtype": conv.weight.dtype}

        self.lora_A = nn.Parameter(torch.zeros(groups, in_per_group, self.r_per_group, **factory_kwargs))
        self.lora_B = nn.Parameter(torch.zeros(groups, out_per_group, self.r_per_group, **factory_kwargs))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.01)
        nn.init.zeros_(self.lora_B)

        # Adaptive rank mask (learned during training)
        if adaptive_rank and not variational_rank:
            self.rank_mask = nn.Parameter(torch.ones(groups, self.r_per_group, **factory_kwargs))

        # Variational rank: Gumbel-Softmax logits
        if variational_rank:
            # Each rank dimension has a binary logit (keep vs drop)
            self.rank_logits = nn.Parameter(torch.zeros(groups, self.r_per_group, **factory_kwargs))
            self.gumbel_tau = 1.0  # Temperature for Gumbel-Softmax

        for param in self.conv.parameters():
            param.requires_grad = False

    def get_scheduled_dropconnect_rate(self, progress: float = 0.0) -> float:
        """Compute scheduled DropConnect rate based on training progress [0, 1]."""
        if self.dropconnect_schedule == "constant" or self.dropconnect_max <= self.dropconnect_min:
            return self.dropconnect_rate
        if self.dropconnect_schedule == "linear":
            rate = self.dropconnect_max - (self.dropconnect_max - self.dropconnect_min) * progress
        elif self.dropconnect_schedule == "cosine":
            rate = self.dropconnect_min + (self.dropconnect_max - self.dropconnect_min) * 0.5 * (1 + math.cos(math.pi * progress))
        elif self.dropconnect_schedule == "exponential":
            rate = self.dropconnect_min + (self.dropconnect_max - self.dropconnect_min) * math.exp(-5 * progress)
        else:
            rate = self.dropconnect_rate
        return max(self.dropconnect_min, min(self.dropconnect_max, rate))

    def _update_importance(self):
        """Update Fisher-information importance EMA for GIW-DC.
        
        NOTE: This must be called AFTER backward() but BEFORE optimizer step,
        when gradients are still available.
        """
        if not self.gradient_importance_weighted:
            return
        if self.lora_A.grad is None or self.lora_B.grad is None:
            return
        grad_A_sq = self.lora_A.grad.detach().pow(2)
        grad_B_sq = self.lora_B.grad.detach().pow(2)
        if self.importance_A is None:
            self.importance_A = grad_A_sq.clone()
            self.importance_B = grad_B_sq.clone()
        else:
            self.importance_A = self.importance_ema_decay * self.importance_A + (1 - self.importance_ema_decay) * grad_A_sq
            self.importance_B = self.importance_ema_decay * self.importance_B + (1 - self.importance_ema_decay) * grad_B_sq

    def _apply_dropconnect(self, tensor: torch.Tensor, is_A: bool = True,
                           progress: float = 0.0) -> torch.Tensor:
        """Apply DropConnect to LoRA matrices during training with optional scheduling and importance weighting."""
        if not self.training:
            return tensor

        rate = self.get_scheduled_dropconnect_rate(progress)
        if rate <= 0:
            return tensor

        if self.gradient_importance_weighted:
            # Gradient-Importance Weighted DropConnect
            importance = self.importance_A if is_A else self.importance_B
            if importance is None:
                # Fallback to random if importance not yet computed
                mask = torch.bernoulli(torch.full_like(tensor, 1 - rate)) / (1 - rate)
                return tensor * mask
            # Normalize importance per rank dimension
            importance_norm = importance / (importance.mean(dim=(0, 1), keepdim=True) + 1e-8)
            # Higher importance -> lower drop probability
            keep_prob = torch.clamp(1 - rate * (1.0 / (importance_norm + 0.1)), 0.0, 1.0)
            mask = torch.bernoulli(keep_prob) / (keep_prob + 1e-8)
            return tensor * mask
        else:
            # Standard random DropConnect
            mask = torch.bernoulli(
                torch.full_like(tensor, 1 - rate)
            ) / (1 - rate)
            return tensor * mask

    def _get_variational_rank_mask(self):
        """Get rank mask from variational Gumbel-Softmax distribution."""
        if not self.variational_rank or not self.training:
            # During eval, use hard threshold
            if self.variational_rank:
                return (torch.sigmoid(self.rank_logits) > 0.5).float()
            return None
        # Gumbel-Softmax sampling for binary mask
        # Use straight-through estimator
        logits = self.rank_logits
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        y_soft = torch.sigmoid((logits + gumbel_noise) / max(self.gumbel_tau, 0.1))
        # Straight-through: forward uses soft, backward passes through hard
        y_hard = (y_soft > 0.5).float()
        mask = y_hard - y_soft.detach() + y_soft
        return mask

    def get_rank_mask(self):
        """Get effective rank mask (adaptive or variational)."""
        if self.variational_rank:
            return self._get_variational_rank_mask()
        elif self.adaptive_rank and hasattr(self, 'rank_mask'):
            return self.rank_mask
        return None

    def forward(self, x, teacher_features=None, progress: float = 0.0):
        out = self.conv(x)

        k_h, k_w = self.conv.kernel_size
        
        # ── v3: 1x1 conv short-circuit ──
        # For 1x1 conv with zero padding, unfold is equivalent to reshape
        # This avoids expensive memory allocation for ~40% of YOLO conv layers
        is_1x1 = (k_h == 1 and k_w == 1)
        no_pad = (self.conv.padding == (0, 0) or self.conv.padding == 0)
        
        if is_1x1 and no_pad:
            # x: (B, C_in, H, W) -> (B, C_in, H*W)
            B_size, C_in, H, W = x.shape
            L = H * W
            out_h, out_w = H, W
            x_unfold = x.view(B_size, C_in, L)
        else:
            x_unfold = nn.functional.unfold(
                x, (k_h, k_w), padding=self.conv.padding,
                stride=self.conv.stride, dilation=self.conv.dilation
            )
            B_size, _, L = x_unfold.shape
            out_h, out_w = out.shape[2], out.shape[3]

        if self.lora_dropout is not None:
            x_unfold = self.lora_dropout(x_unfold)

        groups = getattr(self, "groups", getattr(self.conv, "groups", 1))

        # Update importance estimates (for GIW-DC)
        self._update_importance()

        # Apply DropConnect to LoRA matrices
        A = self._apply_dropconnect(self.lora_A, is_A=True, progress=progress)
        B = self._apply_dropconnect(self.lora_B, is_A=False, progress=progress)

        # Apply rank mask (adaptive or variational)
        rank_mask = self.get_rank_mask()
        if rank_mask is not None:
            A = A * rank_mask.unsqueeze(1)
            B = B * rank_mask.unsqueeze(1)

        if groups == 1 and A.dim() == 2 and B.dim() == 2:
            x_unfold = x_unfold.transpose(1, 2)
            lora = x_unfold @ A
            lora = lora @ B.t()
            lora = lora * self.scaling
            lora = lora.transpose(1, 2).reshape(B_size, self.conv.out_channels, out_h, out_w)
            out_lora = out + lora
            if teacher_features is not None and self.training:
                alignment_loss = self._compute_alignment_loss(out_lora, teacher_features)
                return out_lora, alignment_loss
            return out_lora

        if groups == 1:
            x_unfold = x_unfold.transpose(1, 2)
            lora = x_unfold @ A[0]
            lora = lora @ B[0].t()
            lora = lora * self.scaling
            lora = lora.transpose(1, 2).reshape(B_size, self.conv.out_channels, out_h, out_w)
            out_lora = out + lora
            if teacher_features is not None and self.training:
                alignment_loss = self._compute_alignment_loss(out_lora, teacher_features)
                return out_lora, alignment_loss
            return out_lora

        in_per_group = x_unfold.shape[1] // groups
        x_grouped = x_unfold.view(B_size, groups, in_per_group, L).permute(1, 0, 3, 2)
        lora = torch.bmm(
            x_grouped.reshape(groups, B_size * L, in_per_group), A
        )
        lora = torch.bmm(lora, B.transpose(1, 2))
        lora = lora * self.scaling
        out_per_group = B.shape[1]
        lora = lora.view(groups, B_size, L, out_per_group).permute(1, 0, 3, 2)
        lora = lora.reshape(B_size, groups * out_per_group, L)
        lora = lora.view(B_size, self.conv.out_channels, out_h, out_w)

        # Feature alignment with teacher (if provided)
        if teacher_features is not None and self.training:
            alignment_loss = self._compute_alignment_loss(out + lora, teacher_features)
            return out + lora, alignment_loss

        return out + lora

    def _compute_alignment_loss(self, student_feat, teacher_feat):
        """Compute feature alignment loss for knowledge distillation."""
        if teacher_feat is None:
            return torch.tensor(0.0, device=student_feat.device)
        # Match spatial dimensions
        if student_feat.shape != teacher_feat.shape:
            teacher_feat = nn.functional.adaptive_avg_pool2d(
                teacher_feat, student_feat.shape[2:]
            )
        return nn.functional.mse_loss(student_feat, teacher_feat)


class ManualLoRAConv(nn.Module):
    """Minimal manual LoRA wrapper for Conv2d fallback paths.

    Supports both dense Conv2d (groups=1) and grouped convolutions
    (groups>1, including depthwise where groups == in_channels == out_channels).
    For grouped convs we allocate one (A, B) pair per group, so the rank `r`
    MUST be divisible by `groups`.
    """

    def __init__(
        self,
        conv: nn.Conv2d,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        use_rslora: bool = False,
    ):
        super().__init__()
        self.conv = conv
        self.r = r
        self.alpha = alpha
        self.use_rslora = bool(use_rslora)
        self.scaling = _fallback_lora_scaling(alpha, r, self.use_rslora)
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
    variant = _effective_peft_variant(config)
    if variant == "dora":
        return bool(PEFT_AVAILABLE and getattr(config, "use_dora", False))
    return bool(PEFT_AVAILABLE and variant in {
        "lora", "adalora", "loha", "lokr",
        "ia3", "oft", "boft", "hra",
    })


def supports_fallback_request(config: "LoRAConfig") -> bool:
    """Return whether the in-repo fallback backend can satisfy the requested variant."""
    return getattr(config, "r", 0) > 0 and _effective_peft_variant(config) == "lora"


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
    
    v3: Supports layer-wise adaptive rank when few_shot_layerwise_rank=True.
    """
    replaced = 0
    base_r = getattr(config, "r", 0) or 0
    allow_depthwise = bool(getattr(config, "allow_depthwise", False))
    few_shot = getattr(config, "few_shot_mode", False)
    layerwise_rank = few_shot and getattr(config, "few_shot_layerwise_rank", False)

    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv2d):
            groups = child.groups
            # Compute per-layer rank if layerwise_rank is enabled
            r = base_r
            if layerwise_rank:
                r = _compute_layer_rank(child, base_r, full_name)
            
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
            # Use FewShotLoRAConv in few-shot mode
            if few_shot:
                lora_cls = FewShotLoRAConv
                lora_kwargs = {
                    "r": r, "alpha": max(r * 2, config.alpha),
                    "dropout": config.dropout,
                    "use_rslora": bool(getattr(config, "use_rslora", False)),
                    "dropconnect": getattr(config, "few_shot_dropconnect", 0.1),
                    "adaptive_rank": getattr(config, "few_shot_adaptive_rank", True),
                    "dropconnect_schedule": getattr(config, "few_shot_dropconnect_schedule", "constant"),
                    "dropconnect_max": getattr(config, "few_shot_dropconnect_max", 0.3),
                    "dropconnect_min": getattr(config, "few_shot_dropconnect_min", 0.0),
                    "gradient_importance_weighted": getattr(config, "few_shot_gradient_importance_weighted", False),
                    "variational_rank": getattr(config, "few_shot_variational_rank", False),
                    "rank_budget": getattr(config, "few_shot_rank_budget", 0.5),
                }
            else:
                lora_cls = ManualLoRAConv
                lora_kwargs = {
                    "r": r,
                    "alpha": max(r * 2, config.alpha),
                    "dropout": config.dropout,
                    "use_rslora": bool(getattr(config, "use_rslora", False)),
                }
            setattr(module, name, lora_cls(child, **lora_kwargs))
            replaced += 1
            continue
        replaced += _replace_conv_with_manual_lora(child, config, prefix=full_name, include_head=include_head)
    return replaced


def _compute_layer_rank(conv: nn.Conv2d, base_r: int, module_name: str, total_layers: int = 23) -> int:
    """Compute per-layer LoRA rank from depth, channel width, and capacity bound.

    Design goals:
      - Shallow layers (early feature extraction) get a larger rank.
      - Deep layers (semantic / task-specific) get a smaller rank.
      - Wider-channel layers get proportionally larger rank.
      - Capacity bound: rank never exceeds min(in_channels, out_channels) // 2,
        so LoRA stays genuinely low-rank and does not collapse to a full-rank
        reparameterization on narrow layers.
    """
    # Extract layer index from module name (e.g., "model.5.m.0.cv1" -> 5)
    layer_idx = 0
    for part in module_name.split("."):
        if part.isdigit():
            layer_idx = int(part)
            break

    # Depth factor: shallow layers (idx=0) -> 1.0, deep layers -> 0.5
    depth_factor = 1.0 - 0.5 * (layer_idx / max(total_layers, 1))

    # Channel factor: wider channels -> larger rank
    channels_factor = min(conv.out_channels / 64.0, 2.0)

    # Raw rank
    r = int(base_r * depth_factor * channels_factor)

    # Capacity bound: enforce r <= min(in, out) // 2 to keep low-rank semantics.
    # Without this, narrow layers (e.g. 16x8) can receive r=16 which is a full-rank
    # (or super-rank) reparameterization and wastes capacity.
    cap = max(1, min(conv.in_channels, conv.out_channels) // 2)
    r = min(r, cap)

    # Floor: keep at least rank 4 for any detected target (or `groups`, whichever larger)
    groups = max(conv.groups, 1)
    r = max(r, min(4, cap))

    # Ensure divisible by groups (required by PEFT Conv2d)
    r = (r // groups) * groups
    r = max(r, groups)

    return r


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
    model.lora_target_audit = build_lora_target_audit(
        valid_targets=model.lora_target_modules,
        selected_targets=model.lora_target_modules,
        requested_targets=getattr(config, "target_modules", None),
        exclude_modules=getattr(config, "exclude_modules", None),
        peft_type="lora",
        rank=getattr(config, "r", None),
    )
    model.lora_runtime_metadata = resolve_effective_lora_request(
        requested_backend=config.backend,
        effective_backend="fallback",
        requested_variant=config.variant,
        effective_variant="lora",
        requested_init_lora_weights=config.init_lora_weights,
        effective_init_lora_weights=config.init_lora_weights,
        requested_use_rslora=bool(getattr(config, "use_rslora", False)),
        effective_use_rslora=bool(getattr(config, "use_rslora", False)),
        include_head=include_head,
        freeze_bn=bool(getattr(config, "freeze_bn", False)),
        target_modules=model.lora_target_modules,
        target_audit=model.lora_target_audit,
    )

    _unfreeze_detection_head(model)

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
    """Collect serializable fallback LoRA adapter state from ManualLoRAConv or FewShotLoRAConv modules."""
    target_root = getattr(model, "model", model)
    modules = {}
    state = {}
    for name, module in target_root.named_modules():
        if not isinstance(module, (ManualLoRAConv, FewShotLoRAConv)):
            continue
        modules[name] = {
            "r": int(module.r),
            "alpha": int(module.alpha),
            "dropout": float(module.lora_dropout.p if module.lora_dropout is not None else 0.0),
            "use_rslora": bool(getattr(module, "use_rslora", False)),
        }
        state[name] = {
            "lora_A": module.lora_A.detach().cpu(),
            "lora_B": module.lora_B.detach().cpu(),
        }
        if isinstance(module, FewShotLoRAConv):
            modules[name]["few_shot"] = True
            modules[name]["dropconnect_schedule"] = getattr(module, "dropconnect_schedule", "constant")
            modules[name]["dropconnect_max"] = getattr(module, "dropconnect_max", 0.3)
            modules[name]["dropconnect_min"] = getattr(module, "dropconnect_min", 0.0)
            modules[name]["gradient_importance_weighted"] = getattr(module, "gradient_importance_weighted", False)
            modules[name]["variational_rank"] = getattr(module, "variational_rank", False)
            modules[name]["rank_budget"] = getattr(module, "rank_budget", 0.5)
            if hasattr(module, 'rank_mask'):
                state[name]["rank_mask"] = module.rank_mask.detach().cpu()
            if hasattr(module, 'rank_logits'):
                state[name]["rank_logits"] = module.rank_logits.detach().cpu()
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
        use_rslora = bool(config.get("use_rslora", False))
        if isinstance(original, (ManualLoRAConv, FewShotLoRAConv)):
            wrapped = original
        else:
            if not isinstance(original, nn.Conv2d):
                raise TypeError(f"Fallback adapter target is not Conv2d: {module_name}")
            is_few_shot = config.get("few_shot", False)
            lora_cls = FewShotLoRAConv if is_few_shot else ManualLoRAConv
            lora_kwargs = {
                "r": int(config.get("r", 0)),
                "alpha": int(config.get("alpha", 0)),
                "dropout": float(config.get("dropout", 0.0)),
                "use_rslora": use_rslora,
            }
            if is_few_shot:
                lora_kwargs["dropconnect"] = float(config.get("dropconnect", 0.1))
                lora_kwargs["adaptive_rank"] = bool(config.get("adaptive_rank", True))
                lora_kwargs["dropconnect_schedule"] = config.get("dropconnect_schedule", "constant")
                lora_kwargs["dropconnect_max"] = float(config.get("dropconnect_max", 0.3))
                lora_kwargs["dropconnect_min"] = float(config.get("dropconnect_min", 0.0))
                lora_kwargs["gradient_importance_weighted"] = bool(config.get("gradient_importance_weighted", False))
                lora_kwargs["variational_rank"] = bool(config.get("variational_rank", False))
                lora_kwargs["rank_budget"] = float(config.get("rank_budget", 0.5))
            wrapped = lora_cls(original, **lora_kwargs)
            _set_module_by_name(target_root, module_name, wrapped)

        wrapped.use_rslora = use_rslora
        wrapped.scaling = _fallback_lora_scaling(wrapped.alpha, wrapped.r, use_rslora)
        params = module_state.get(module_name, {})
        wrapped.lora_A.data.copy_(params["lora_A"])
        wrapped.lora_B.data.copy_(params["lora_B"])
        if "rank_mask" in params and hasattr(wrapped, 'rank_mask'):
            wrapped.rank_mask.data.copy_(params["rank_mask"])
        if "rank_logits" in params and hasattr(wrapped, 'rank_logits'):
            wrapped.rank_logits.data.copy_(params["rank_logits"])

    model = _wrap_top_level_lora_model(model, None)
    model.lora_enabled = True
    model.lora_backend = "fallback"
    model.lora_variant = payload.get("variant", "lora")
    model.lora_include_head = payload.get("include_head", False)
    model.lora_freeze_bn = payload.get("freeze_bn", False)
    model.lora_target_modules = payload.get("target_modules", sorted(module_configs))
    model.lora_target_audit = payload.get(
        "target_audit",
        build_lora_target_audit(
            valid_targets=model.lora_target_modules,
            selected_targets=model.lora_target_modules,
            peft_type=model.lora_variant,
        ),
    )
    model.lora_runtime_metadata = payload.get("runtime_metadata", {})
    return model


def _merge_manual_lora_conv(module) -> nn.Conv2d:
    """Materialize a ManualLoRAConv or FewShotLoRAConv adapter into a plain Conv2d with merged weights.

    Handles both dense (groups=1) and grouped (groups>1) convolutions. The stored
    shapes are:
        lora_A: (groups, in_per_group * kH * kW, r_per_group)
        lora_B: (groups, out_per_group, r_per_group)
    """
    conv = module.conv
    groups = getattr(module, "groups", 1)
    # Apply adaptive rank mask if present
    lora_A = module.lora_A
    lora_B = module.lora_B
    if hasattr(module, 'rank_mask'):
        lora_A = lora_A * module.rank_mask.unsqueeze(1)
        lora_B = lora_B * module.rank_mask.unsqueeze(1)
    # Per-group delta: (out_per_group, in_per_group * kH * kW)
    delta_per_group = torch.bmm(lora_B, lora_A.transpose(1, 2))
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
    """Recursively merge ManualLoRAConv/FewShotLoRAConv children back into Conv2d modules."""
    merged = 0
    for name, child in list(module.named_children()):
        if isinstance(child, (ManualLoRAConv, FewShotLoRAConv)):
            setattr(module, name, _merge_manual_lora_conv(child))
            merged += 1
            continue
        merged += _merge_fallback_modules(child)
    return merged


def _clear_lora_runtime_state(model: "DetectionModel") -> None:
    """Remove LoRA runtime markers after merge so the model no longer looks adapter-enabled."""
    for attr in (
        "lora_enabled",
        "lora_config",
        "lora_backend",
        "lora_variant",
        "lora_include_head",
        "lora_freeze_bn",
        "lora_target_modules",
        "lora_target_audit",
        "lora_runtime_metadata",
        "lora_original_class",
        "use_gradient_checkpointing",
    ):
        if hasattr(model, attr):
            try:
                delattr(model, attr)
            except AttributeError:
                pass


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
        """Helper to retrieve the underlying base model, handling nested PEFT wrappers.

        P1 PERF FIX: cache the resolved base module on first access so that hot
        paths (forward, __getitem__, __getattr__ fallback) do not re-walk the
        wrapper chain on every call. PyTorch's `nn.Module.__getattr__` is
        already non-trivial — repeating `hasattr(model, 'model')` lookups per
        layer access pushes 1-2% extra overhead in long training runs.
        """
        cached = self.__dict__.get("_cached_base_model")
        if cached is not None:
            return cached
        model = self.base_model
        # Traverse down if multiple wrappers exist (common in some PEFT versions)
        while hasattr(model, 'model') and not isinstance(model, nn.Sequential):
            model = model.model
        # Use __dict__ to bypass nn.Module's parameter registration machinery
        # — we only want to cache the reference, not register a submodule.
        self.__dict__["_cached_base_model"] = model
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

__all__ = [
    "FewShotLoRAConv",
    "ManualLoRAConv",
    "supports_peft_request",
    "supports_fallback_request",
    "apply_manual_lora",
    "_collect_fallback_adapter_state",
    "_load_fallback_adapter_state",
    "_merge_manual_lora_conv",
    "_merge_fallback_modules",
    "_clear_lora_runtime_state",
    "_filter_target_modules",
    "_freeze_batchnorm_layers",
    "_build_peft_exact_target_regex",
    "_validate_peft_init_compatibility",
    "PeftProxy",
    "LoRADetectionModel",
    "LoRADetectionModelWrapper",
    "LoRASegmentationModelWrapper",
    "LoRAPoseModelWrapper",
    "LoRAClassificationModelWrapper",
    "LoRAOBBModelWrapper",
    "LoRARTDETRDetectionModelWrapper",
    "LoRAWorldModelWrapper",
    "_wrap_top_level_lora_model",
]
