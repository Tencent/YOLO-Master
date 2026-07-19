# 🐧Please note that this file has been modified by Tencent on 2026/02/07. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Efficient routers for Mixture-of-Experts models"""
import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from urllib.request import Request, urlopen
from typing import Tuple, Optional, Dict
from .utils import FlopsUtils, get_safe_groups
from ultralytics.utils.errors import MoERouterError, ShapeMismatchError


def _get_router_in_channels(router) -> int:
    """Safely extract expected in_channels from a router module.

    Handles nn.Sequential, nn.Identity, and bare nn.Module cases.
    Returns -1 if unknown (skips channel check).
    """
    if hasattr(router, "__getitem__"):
        try:
            first = router[0]
            if hasattr(first, "in_channels"):
                return first.in_channels
        except (TypeError, IndexError, KeyError):
            pass
    if hasattr(router, "in_channels"):
        return router.in_channels
    return -1


def _validate_router_input(x: torch.Tensor, expected_channels: int, context: str = "") -> None:
    """Validate router input tensor before processing.

    Raises:
        MoERouterError: If x is not 4-D or contains NaN/Inf.
        ShapeMismatchError: If channel count does not match expected.
    """
    if x.dim() != 4:
        raise MoERouterError(
            f"Router input must be 4-D (NCHW), got {x.dim()}-D shape {tuple(x.shape)}"
            + (f" [{context}]" if context else "")
        )
    if expected_channels > 0 and x.shape[1] != expected_channels:
        raise ShapeMismatchError(
            expected=f"(N, {expected_channels}, H, W)",
            actual=tuple(x.shape),
            context=context or "router input",
        )
    if torch.isnan(x).any() or torch.isinf(x).any():
        #region debug-point router-nonfinite
        if os.getenv("ULTRA_DEBUG_NONFINITE", ""):
            try:
                finite = torch.isfinite(x)
                nonfinite = (~finite).sum().item()
                total = x.numel()
                if finite.any():
                    x_finite = x[finite]
                    x_min = float(x_finite.min().item())
                    x_max = float(x_finite.max().item())
                else:
                    x_min = None
                    x_max = None
                payload = {
                    "event": "router_input_nonfinite",
                    "context": context,
                    "shape": list(x.shape),
                    "dtype": str(x.dtype),
                    "device": str(x.device),
                    "nonfinite": int(nonfinite),
                    "total": int(total),
                    "min": x_min,
                    "max": x_max,
                }
                url = os.getenv("ULTRA_DEBUG_POST_URL", "")
                if url:
                    body = json.dumps(payload, separators=(",", ":")).encode()
                    req = Request(url, data=body, headers={"Content-Type": "application/json"})
                    urlopen(req, timeout=1.0).close()
                else:
                    from ultralytics.utils import LOGGER

                    LOGGER.warning(f"[debug-point router-nonfinite] {payload}")
            except Exception:
                pass
        #endregion debug-point router-nonfinite
        raise MoERouterError(
            f"Router input contains NaN/Inf values"
            + (f" [{context}]" if context else "")
        )


# ==========================================
# Ultra-lightweight Router (core optimization)
# ==========================================
class UltraEfficientRouter(nn.Module):
    """
    Ultra-efficient router:
    1) Depthwise-separable convolution instead of standard conv
    2) Aggressive downsampling (8x)
    3) Early channel compression
    4) Improved numerical stability

    Expected FLOPs reduction: ~95% vs a local router baseline.
    """

    def __init__(self, in_channels, num_experts, reduction=16, top_k=2,
                 noise_std=1.0, temperature: float = 1.0, pool_scale=8):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.temperature = max(float(temperature), 1e-3)
        self.pool_scale = pool_scale

        # More aggressive channel compression
        reduced_channels = max(in_channels // reduction, 4)

        # Depthwise-separable conv: compute ~ 1/(kernel_size^2) of standard conv
        self.router = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.GroupNorm(get_safe_groups(in_channels, 8), in_channels),
            nn.SiLU(inplace=False),
            # Pointwise compression
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.GroupNorm(get_safe_groups(reduced_channels, 4), reduced_channels),
            nn.SiLU(inplace=False),
            # Expert projection
            nn.Conv2d(reduced_channels, num_experts, 1, bias=True)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, top_k: Optional[int] = None) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        _validate_router_input(x, _get_router_in_channels(self.router), context="UltraEfficientRouter")
        B, C, H, W = x.shape
        current_top_k = max(1, min(int(self.top_k if top_k is None else top_k), self.num_experts))

        # 1) Aggressive downsampling (core optimization)
        if H > self.pool_scale and W > self.pool_scale:
            x_down = F.avg_pool2d(x, kernel_size=self.pool_scale, stride=self.pool_scale)
        else:
            x_down = x

        # 2) Lightweight convolutional routing
        logits = self.router(x_down)

        # 3) Noise injection (training only)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        # 4) Clamp the final logits that actually feed softmax
        logits_clamped = logits.clamp(-30.0, 30.0)

        # 5) Z-loss on the SAME logits that drive routing (post-noise, post-clamp),
        # so the regularizer constrains the real routing decision (was computed
        # on the pre-noise logits before, weakening its effect).
        scaled_logits = logits_clamped / self.temperature
        z_loss_metric = None
        if self.training:
            z_loss_metric = torch.logsumexp(scaled_logits.float(), dim=1).pow(2).mean()

        # 6) Softmax + TopK (fused operation)
        weights = F.softmax(scaled_logits.float(), dim=1).type_as(x)
        pooled_weights = weights.mean(dim=[2, 3], keepdim=True)
        
        topk_vals, topk_indices = torch.topk(pooled_weights, current_top_k, dim=1)
        
        # Out-of-place normalization preserves the Top-K autograd graph.
        topk_vals = topk_vals / topk_vals.sum(dim=1, keepdim=True).clamp_min(1e-6)

        if self.training:
            importance = pooled_weights.mean(dim=0).view(self.num_experts)

            # Optimization: use one_hot instead of scatter
            topk_indices_flat = topk_indices.view(B, current_top_k, 1, 1)[:, :, 0, 0]
            mask = F.one_hot(topk_indices_flat, num_classes=self.num_experts).float()
            usage_frequency = mask.sum(dim=[0, 1]) / (B * current_top_k)

            return topk_vals, topk_indices, usage_frequency, importance, z_loss_metric
        else:
            return topk_vals, topk_indices, None, None, None

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        h_down = max(H // self.pool_scale, 1)
        w_down = max(W // self.pool_scale, 1)

        flops = B * C * H * W  # AvgPool

        input_down_shape = (B, C, h_down, w_down)

        # Depthwise conv
        flops += FlopsUtils.count_conv2d(self.router[0], input_down_shape)
        # Pointwise conv
        flops += FlopsUtils.count_conv2d(self.router[3], (B, self.router[0].out_channels, h_down, w_down))
        # Expert projection
        flops += FlopsUtils.count_conv2d(self.router[6], (B, self.router[3].out_channels, h_down, w_down))

        return flops


class BaseRouter(nn.Module):
    """Base router with optional capacity factor support (P1-5 fix).

    Capacity factor controls the maximum number of tokens each expert can handle
    per step. When a batch has more tokens than ``capacity_factor * num_experts``,
    excess tokens are routed to an overflow bucket and handled by default experts.
    This prevents OOM when a single expert gets overloaded.
    """

    def __init__(self, num_experts, top_k, capacity_factor: Optional[float] = None):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor  # P1-5: optional token-level overflow guard
        self.softmax = nn.Softmax(dim=1)

    def _process_logits(self, logits: torch.Tensor, noise_std: float, training: bool,
                        top_k: Optional[int] = None) -> Tuple[
        torch.Tensor, torch.Tensor, Dict]:
        """Unified logic to process logits into Top-K selection.

        P1-5: When capacity_factor is set, excess tokens beyond the capacity limit
        are masked out of the routing and assigned to a default expert.
        """
        B = logits.shape[0]
        effective_top_k = self.top_k if top_k is None else max(1, min(int(top_k), self.num_experts))

        # Guard: detect NaN/Inf in logits early (catches upstream corruption)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise MoERouterError(
                f"Router logits contain NaN/Inf before softmax (B={logits.shape[0]})"
            )

        # 1) Add noise during training (simplified Gumbel-Softmax trick)
        if training and noise_std > 0:
            logits = logits + torch.randn_like(logits) * noise_std

        # 2) Keep routing probability math in fp32.  Under CUDA autocast the
        # router logits can be fp16; retaining fp32 through Top-K normalization
        # avoids quantized small probabilities and a low-precision reduction.
        probs = F.softmax(logits.float(), dim=1)

        # 3) Select Top-K in fp32
        topk_vals, topk_indices = torch.topk(probs, effective_top_k, dim=1)

        # P1-5: Apply capacity factor constraint if configured
        overflow_mask = None
        if self.capacity_factor is not None and training:
            max_tokens = int(self.capacity_factor * self.num_experts)
            if B > max_tokens:
                overflow_mask = torch.zeros(B, dtype=torch.bool, device=logits.device)
                # Randomly select which tokens get routed normally (first max_tokens)
                indices = torch.randperm(B, device=logits.device)[:max_tokens]
                overflow_mask[indices] = True
                # Overflow tokens are sent to the default expert (expert 0).
                topk_indices = topk_indices.clone()
                topk_indices[~overflow_mask] = 0
                # Mask routing probabilities before normalization so the
                # capacity decision cannot be undone by later weighting.
                topk_vals = topk_vals.masked_fill(~overflow_mask[:, None], 0)
                topk_vals[~overflow_mask, 0] = 1

        # 4) Normalize weights
        sum_vals = topk_vals.sum(dim=1, keepdim=True) + 1e-6
        topk_vals = topk_vals / sum_vals

        # 5) Collect loss-related info (train only)
        loss_dict = {}
        if training:
            loss_dict['router_logits'] = logits
            loss_dict['router_probs'] = probs
            loss_dict['topk_indices'] = topk_indices
            if overflow_mask is not None:
                loss_dict['overflow_count'] = int(B - max_tokens)

        return topk_vals, topk_indices, loss_dict


class EfficientSpatialRouter(BaseRouter):
    def __init__(self, in_channels, num_experts, reduction=8, top_k=2, noise_std=1.0, pool_scale=4):
        super().__init__(num_experts, top_k)
        self.noise_std = noise_std
        self.pool_scale = pool_scale
        reduced_channels = max(in_channels // reduction, 8)

        self.router = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.SiLU(inplace=False),
            nn.Conv2d(reduced_channels, num_experts, 1, bias=False),
            nn.BatchNorm2d(num_experts)  # numerical stability
        )

    def forward(self, x, top_k: Optional[int] = None):
        _validate_router_input(x, _get_router_in_channels(self.router), context="EfficientSpatialRouter")
        if not math.isfinite(float(self.noise_std)):
            raise MoERouterError("EfficientSpatialRouter noise_std must be finite")
        B, C, H, W = x.shape
        # Pre-pooling optimization
        if H > self.pool_scale and W > self.pool_scale:
            x_in = F.avg_pool2d(x, kernel_size=self.pool_scale, stride=self.pool_scale)
        else:
            x_in = x

        out = self.router(x_in)  # [B, E, H', W']
        if not torch.isfinite(out).all():
            raise MoERouterError("EfficientSpatialRouter internal output contains NaN/Inf")
        # Spatial reduction is sensitive to fp16 cancellation/underflow on
        # large feature maps.  Promote only this routing statistic; convolution
        # remains governed by the caller's autocast policy.
        global_logits = out.float().mean(dim=[2, 3])  # [B, E]
        if not torch.isfinite(global_logits).all():
            raise MoERouterError("EfficientSpatialRouter global logits contain NaN/Inf")

        return self._process_logits(global_logits, self.noise_std, self.training, top_k=top_k)

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        h_down, w_down = max(H // self.pool_scale, 1), max(W // self.pool_scale, 1)
        return FlopsUtils.count_conv2d(self.router, (B, C, h_down, w_down))


class AdaptiveRoutingLayer(BaseRouter):
    def __init__(self, in_channels, num_experts, reduction=8, top_k=2, noise_std=1.0):
        super().__init__(num_experts, top_k)
        self.noise_std = noise_std
        reduced_channels = max(in_channels // reduction, 8)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.SiLU(inplace=False),
            nn.Conv2d(reduced_channels, num_experts, 1, bias=False),
            nn.BatchNorm2d(num_experts)
        )

    def forward(self, x, top_k: Optional[int] = None):
        _validate_router_input(x, _get_router_in_channels(self.router), context="AdaptiveRoutingLayer")
        pooled = self.avg_pool(x)
        logits = self.router(pooled).squeeze(-1).squeeze(-1)  # [B, E]
        return self._process_logits(logits, self.noise_std, self.training, top_k=top_k)

    def compute_flops(self, input_shape):
        # FLOPs here are minimal
        return FlopsUtils.count_conv2d(self.router, (input_shape[0], input_shape[1], 1, 1))


class LocalRoutingLayer(BaseRouter):
    def __init__(self, in_channels, num_experts, reduction=8, top_k=2, noise_std=1.0):
        super().__init__(num_experts, top_k)
        self.noise_std = noise_std
        # Even for local routing, default to 2x downsampling to save FLOPs with minimal texture loss
        self.pool_scale = 2

        reduced_channels = max(in_channels // reduction, 8)
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.SiLU(inplace=False),
            nn.Conv2d(reduced_channels, num_experts, 1, bias=False),
            nn.BatchNorm2d(num_experts)
        )

    def forward(self, x, top_k: Optional[int] = None):
        _validate_router_input(x, _get_router_in_channels(self.router), context="LocalRoutingLayer")
        # Moderate downsampling to accelerate
        if x.shape[2] > self.pool_scale:
            x_in = F.avg_pool2d(x, kernel_size=self.pool_scale, stride=self.pool_scale)
        else:
            x_in = x

        out = self.router(x_in)
        global_logits = torch.mean(out, dim=[2, 3])
        return self._process_logits(global_logits, self.noise_std, self.training, top_k=top_k)

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        h_d, w_d = max(H // self.pool_scale, 1), max(W // self.pool_scale, 1)
        return FlopsUtils.count_conv2d(self.router, (B, C, h_d, w_d))


class AdvancedRoutingLayer(nn.Module):
    """Compatibility router used by some legacy checkpoints; behaves like a global average-pooling router.

    All projection layers are created in ``__init__`` so that the optimizer
    and DDP always see the full parameter set.  The ``_proj`` channel-adapter
    is pre-registered as ``nn.Identity`` and lazily replaced *only* during
    ``__init__`` (never in ``forward``), which keeps the layer tree stable
    for ONNX export and torch.compile.
    """

    def __init__(self, in_channels=64, num_experts=3, top_k=None):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = num_experts if top_k is None else min(top_k, num_experts)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self._expected_in = in_channels
        reduced = max(in_channels // 8, 8)
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, reduced, 1, bias=False),
            nn.SiLU(inplace=False),
            nn.Conv2d(reduced, num_experts, 1, bias=True),
        )
        # Pre-create identity adapter; replaced only if a mismatched channel
        # count is detected at first forward (rare legacy-checkpoint scenario).
        self._proj = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        pooled = self.avg_pool(x)
        expected_in = self.router[0].in_channels
        if expected_in != C:
            # Channel mismatch: use a pre-existing _proj Conv2d if present,
            # otherwise fall back to padding/truncation (tensor-only, no new
            # parameters created at runtime — safe for export & DDP).
            if isinstance(self._proj, nn.Conv2d) and self._proj.in_channels == C:
                pooled = self._proj(pooled)
            else:
                # Tensor-only adaptation: zero-pad or truncate channels.
                if C < expected_in:
                    pad = expected_in - C
                    pooled = F.pad(pooled, (0, 0, 0, 0, 0, pad))
                else:
                    pooled = pooled[:, :expected_in]
        logits = self.router(pooled)
        probs = F.softmax(logits.float(), dim=1).type_as(logits)
        E = probs.shape[1]
        k = max(1, min(getattr(self, "top_k", E), E))
        if k < E:
            vals, idx = torch.topk(probs, k, dim=1)
            vals = vals / (vals.sum(dim=1, keepdim=True) + 1e-6)
            weights = torch.zeros_like(probs)
            weights.scatter_(1, idx, vals)
        else:
            weights = probs
        return weights.repeat(1, 1, H, W)


class DynamicRoutingLayer(nn.Module):
    def __init__(self, in_channels, num_experts=3, reduction=8, top_k=None):
        """
        Args:
            top_k: Number of active experts; if None uses all experts (Softmax)
        """
        super(DynamicRoutingLayer, self).__init__()
        reduced_channels = max(in_channels // reduction, 8)

        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts) if top_k is not None else num_experts
        self.use_top_k = (top_k is not None)  # whether to enable Top-K

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Remove Softmax and control manually
        self.routing_network = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.SiLU(inplace=False),
            nn.Conv2d(reduced_channels, num_experts, kernel_size=1),
        )

    def forward(self, x):
        pooled = self.global_pool(x)
        routing_logits = self.routing_network(pooled)  # [B, num_experts, 1, 1]

        # Choose strategy based on Top-K enablement and train/infer mode
        # Note: Use unified path for ONNX export compatibility.
        # The soft/hard Top-K split via `if self.training` breaks ONNX tracing
        # because the control flow isn't fixed at export time.
        # Solution: always use soft Top-K (differentiable), which works for
        # both training and inference. Hard Top-K is only marginally faster
        # at inference but creates export incompatibility.
        if not self.use_top_k:
            # No Top-K: direct Softmax
            routing_weights = F.softmax(routing_logits.float(), dim=1).type_as(x)
        else:
            # Training / export: soft Top-K keeps gradient flow and a static
            # graph that traces cleanly for ONNX/TorchScript.
            # Eager-mode inference: hard Top-K gives true sparsity (non-selected
            # experts get exactly 0 weight, identical numerics to soft Top-K's
            # masked renormalisation) so callers can skip those experts.
            export = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()
            if self.training or export:
                routing_weights = self._soft_top_k(routing_logits)
            else:
                routing_weights = self._hard_top_k(routing_logits)

        return routing_weights.repeat(1, 1, x.size(2), x.size(3))

    def _soft_top_k(self, logits):
        """Soft Top-K during training to maintain gradient flow."""
        B, E, H, W = logits.shape
        logits_flat = logits.view(B, E, -1)

        # Compute softmax
        # Fix: Clamp logits to avoid overflow
        logits_flat = logits_flat.clamp(-30.0, 30.0)
        weights = F.softmax(logits_flat.float(), dim=1).type_as(logits)

        # Find Top-K and build mask
        _, topk_indices = torch.topk(weights, self.top_k, dim=1)
        idx = topk_indices.permute(0, 2, 1).contiguous()
        mask_one_hot = F.one_hot(idx, num_classes=E).sum(dim=2)
        mask_one_hot = mask_one_hot.permute(0, 2, 1).contiguous().to(weights.dtype)

        # Apply mask and re-normalize
        weights = weights * mask_one_hot
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
        
        return weights.view(B, E, H, W)

    def _hard_top_k(self, logits):
        """Inference Top-K with the same numerics as the differentiable path."""
        # Both paths must rank and normalize the same clamped logits. The only
        # inference-specific difference is that callers may skip zero-weight experts.
        return self._soft_top_k(logits)
