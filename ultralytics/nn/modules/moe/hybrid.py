# 🐧Please note that this file has been modified by Tencent on 2026/02/13. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Auto-generated MoE submodule — split from modules.py. Do not edit manually."""

from ._common import (
    autocast,
    MOE_LOSS_REGISTRY,
    _MOE_LOSS_REGISTRY_LOCK,
    _registry_set,
    _registry_get,
    _should_record_snapshot,
    _zero_aux_loss_like,
    _detached_zero_like,
    _get_moe_aux_loss,
    _flatten_moe_topk,
    _compute_usage_from_topk,
    _record_moe_snapshot,
    _robust_deepcopy,
)
# Standard library + third-party (imported directly, not via _common)
import os
import math
import copy
import weakref
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Union

from .utils import FlopsUtils, get_safe_groups, BatchedExpertComputation
from .experts import (
    OptimizedSimpleExpert, FusedGhostExpert, SimpleExpert, GhostExpert,
    InvertedResidualExpert, EfficientExpertGroup, SpatialExpert, SharedInvertedExpertGroup,
)
from .routers import (
    UltraEfficientRouter, EfficientSpatialRouter, LocalRoutingLayer,
    AdaptiveRoutingLayer, DynamicRoutingLayer, AdvancedRoutingLayer,
)
from ultralytics.nn.modules.block import ABlock, A2C2f, C3k
from .loss import MoELoss, gshard_balance_loss, weighted_gshard_balance_loss, differentiable_balance_loss, all_reduce_mean
from .scheduler import MoEDynamicScheduler, MoEDynamicSchedulerConfig

# Cross-submodule imports: hybrid classes inherit from advanced.py classes
from .advanced import (
    AdaptiveGateMoE,
    ZeroCostRouter,
    FusedExpertGroup,
    LowRankFusedExpertGroup,
    DualStreamGateRouter,
    DualStreamGateRouterV2,
    HyperSplitMoE,
    HyperFusedMoE,
)

# ---- Visual-enhanced + Hybrid MoE classes (split from modules.py) ----

class VisualDetailGate(nn.Module):
    """Lightweight detail gate for boundary and texture aware visual MoE."""

    def __init__(self, channels, num_groups=8, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.detail_filter = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(get_safe_groups(channels, num_groups), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.detail_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        detail = x - smooth
        gate = self.detail_filter(detail)
        return x * (1 + torch.tanh(self.detail_scale) * gate)

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        flops = FlopsUtils.count_conv2d(self.detail_filter, input_shape)
        flops += B * C * H * W * 2
        return flops


def _pool_to_size_mps_safe(x: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
    """Pool to a target spatial size without hitting MPS adaptive-pool limits."""
    h, w = output_size
    H, W = x.shape[-2:]
    if (H, W) == (h, w):
        return x
    if x.device.type != "mps":
        return F.adaptive_avg_pool2d(x, (h, w))

    if H % h == 0 and W % w == 0:
        kernel = (H // h, W // w)
        return F.avg_pool2d(x, kernel_size=kernel, stride=kernel)

    pad_h = ((H + h - 1) // h) * h - H
    pad_w = ((W + w - 1) // w) * w - W
    pooled_source = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate") if pad_h or pad_w else x
    H_pad, W_pad = pooled_source.shape[-2:]
    kernel = (H_pad // h, W_pad // w)
    return F.avg_pool2d(pooled_source, kernel_size=kernel, stride=kernel)


class PyramidContextMixer(nn.Module):
    """Pool-based multi-scale context mixer with a gated residual update."""

    def __init__(self, channels, num_groups=8, pool_scales=(2, 4)):
        super().__init__()
        self.pool_scales = tuple(pool_scales)
        self.local_context = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(get_safe_groups(channels, num_groups), channels),
            nn.SiLU(inplace=True),
        )
        self.pool_projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.GroupNorm(get_safe_groups(channels, num_groups), channels),
                nn.SiLU(inplace=True),
            )
            for _ in self.pool_scales
        )
        self.context_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.context_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, C, H, W = x.shape
        contexts = [self.local_context(x)]
        for scale, proj in zip(self.pool_scales, self.pool_projections):
            h = max(1, H // scale)
            w = max(1, W // scale)
            pooled = _pool_to_size_mps_safe(x, (h, w))
            contexts.append(F.interpolate(proj(pooled), size=(H, W), mode="nearest"))
        context = torch.stack(contexts, dim=0).mean(dim=0)
        return x + torch.tanh(self.context_scale) * context * self.context_gate(context)

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        flops = FlopsUtils.count_conv2d(self.local_context, input_shape)
        for scale, proj in zip(self.pool_scales, self.pool_projections):
            h = max(1, H // scale)
            w = max(1, W // scale)
            flops += FlopsUtils.count_conv2d(proj, (B, C, h, w))
        flops += FlopsUtils.count_conv2d(self.context_gate, input_shape)
        flops += B * C * H * W * 4
        return flops


def _run_visual_hybrid_moe_forward(module, x, detail_gate=None, context_mixer=None, refine_features=False):
    """Shared forward path for visual MoE variants."""
    B, C, H, W = x.shape

    if module.training:
        module._update_temperature()
        module.training_step += 1
        module._training_step_value += 1

    gate_weights = module.se_gate(x)
    gate_static = gate_weights[:, :module.static_channels].unsqueeze(-1).unsqueeze(-1)
    gate_dynamic = gate_weights[:, module.static_channels:].unsqueeze(-1).unsqueeze(-1)

    x_static = x[:, :module.static_channels, :, :] * gate_static
    x_dynamic = x[:, module.static_channels:, :, :] * gate_dynamic
    if detail_gate is not None:
        x_dynamic = detail_gate(x_dynamic)

    out_static = module.static_net(x_static)
    complexity = module._safe_complexity(x_dynamic)

    routing_weights, routing_indices, routing_stats = module.routing(x_dynamic)
    routing_weights, routing_indices, routing_stats, adaptive_top_k = module._apply_complexity_gate(
        routing_weights, routing_indices, routing_stats, complexity
    )
    out_dynamic = module.fused_experts(x_dynamic, routing_weights, routing_indices, adaptive_top_k)

    out_concat = module._channel_shuffle(torch.cat([out_static, out_dynamic], dim=1))
    if context_mixer is not None:
        out_concat = context_mixer(out_concat)
    if refine_features and hasattr(module, "_refine_features"):
        out_concat = module._refine_features(out_concat)

    out = module.proj(out_concat)
    out = module.bn(out) + x

    if module.training:
        router_probs = routing_stats.get('router_probs')
        router_logits = routing_stats.get('router_logits')
        topk_indices = routing_stats.get('topk_indices')
        if isinstance(router_probs, torch.Tensor) and isinstance(router_logits, torch.Tensor):
            aux_loss = module.moe_loss_fn(router_probs, router_logits, topk_indices)
            _registry_set(module, aux_loss)
            _record_moe_snapshot(
                module,
                expert_usage=routing_stats.get('expert_usage'),
                topk_indices=topk_indices,
                topk_weights=routing_weights,
                router_probs=router_probs,
                aux_loss=aux_loss,
            )

    return out


class FusedAdaptiveGateMoE(AdaptiveGateMoE):
    """
    v0.5 MoE: AdaptiveGateMoE with fully fused expert candidates.

    This variant keeps v0.4 dual-stream routing and gated static/dynamic
    feature processing, but replaces sparse per-expert projections with
    FusedExpertGroup. It is aimed at shallow and mid-level feature maps where
    reducing Python dispatch and small kernel launches is often more important
    than skipping every inactive expert.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        split_ratio: float = 0.5,
        num_groups: int = 8,
        initial_temperature: float = 1.0,
        final_temperature: float = 0.5,
        balance_loss_coeff: float = 1.0,
        router_z_loss_coeff: float = 1.0,
        entropy_loss_coeff: float = 0.01,
    ):
        super().__init__(
            in_channels,
            out_channels,
            num_experts,
            top_k,
            split_ratio,
            num_groups,
            initial_temperature,
            final_temperature,
            balance_loss_coeff,
            router_z_loss_coeff,
            entropy_loss_coeff,
        )
        self.expert_backend = "fused"
        self.fused_experts = FusedExpertGroup(self.dynamic_channels, self.out_dynamic, num_experts, num_groups, top_k=top_k)
        self._init_weights()  # re-init swapped-in experts


class HybridAdaptiveGateMoE(AdaptiveGateMoE):
    """
    v0.6 MoE: hybrid expert backend with lightweight channel mixing.

    Layers with fewer experts use the fused backend from v0.5 to amortize
    launch overhead. Layers with many experts use the shared inverted backend
    from v0.4 to avoid computing large inactive expert sets. A small channel
    shuffle before projection improves static/dynamic feature exchange.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        split_ratio: float = 0.5,
        num_groups: int = 8,
        initial_temperature: float = 1.2,
        final_temperature: float = 0.5,
        balance_loss_coeff: float = 1.0,
        router_z_loss_coeff: float = 1.0,
        entropy_loss_coeff: float = 0.01,
        fused_expert_threshold: int = 8,
        shuffle_groups: int = 2,
    ):
        super().__init__(
            in_channels,
            out_channels,
            num_experts,
            top_k,
            split_ratio,
            num_groups,
            initial_temperature,
            final_temperature,
            balance_loss_coeff,
            router_z_loss_coeff,
            entropy_loss_coeff,
        )
        self.fused_expert_threshold = fused_expert_threshold
        self.shuffle_groups = shuffle_groups if out_channels % shuffle_groups == 0 else 1
        if num_experts <= fused_expert_threshold:
            self.expert_backend = "fused"
            self.fused_experts = FusedExpertGroup(self.dynamic_channels, self.out_dynamic, num_experts, num_groups, top_k=top_k)
        else:
            self.expert_backend = "shared_inverted"
            self.fused_experts = SharedInvertedExpertGroup(
                self.dynamic_channels,
                self.out_dynamic,
                num_experts,
                top_k=top_k,
                weight_threshold=0.0,
            )
        self._init_weights()  # re-init swapped-in experts

    def _channel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        if self.shuffle_groups <= 1:
            return x
        B, C, H, W = x.shape
        return x.view(B, self.shuffle_groups, C // self.shuffle_groups, H, W).transpose(1, 2).reshape(B, C, H, W)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.training:
            self._update_temperature()
            self.training_step += 1
            self._training_step_value += 1

        gate_weights = self.se_gate(x)
        gate_static = gate_weights[:, :self.static_channels].unsqueeze(-1).unsqueeze(-1)
        gate_dynamic = gate_weights[:, self.static_channels:].unsqueeze(-1).unsqueeze(-1)

        x_static = x[:, :self.static_channels, :, :] * gate_static
        x_dynamic = x[:, self.static_channels:, :, :] * gate_dynamic

        out_static = self.static_net(x_static)

        complexity = self._safe_complexity(x_dynamic)

        routing_weights, routing_indices, routing_stats = self.routing(x_dynamic)
        routing_weights, routing_indices, routing_stats, adaptive_top_k = self._apply_complexity_gate(
            routing_weights, routing_indices, routing_stats, complexity
        )

        out_dynamic = self.fused_experts(x_dynamic, routing_weights, routing_indices, adaptive_top_k)

        out_concat = self._channel_shuffle(torch.cat([out_static, out_dynamic], dim=1))
        out = self.proj(out_concat)
        out = self.bn(out) + x

        if self.training:
            router_probs = routing_stats.get('router_probs')
            router_logits = routing_stats.get('router_logits')
            topk_indices = routing_stats.get('topk_indices')
            if isinstance(router_probs, torch.Tensor) and isinstance(router_logits, torch.Tensor):
                aux_loss = self.moe_loss_fn(router_probs, router_logits, topk_indices)
                _registry_set(self, aux_loss)
                _record_moe_snapshot(
                    self,
                    expert_usage=routing_stats.get('expert_usage'),
                    topk_indices=topk_indices,
                    topk_weights=routing_weights,
                    router_probs=router_probs,
                    aux_loss=aux_loss,
                )

        return out


class HybridAdaptiveGateMoEv2(HybridAdaptiveGateMoE):
    """
    v0.11 MoE: router-optimized successor to v0.6 ``HybridAdaptiveGateMoE``.

    The module-level ablation over v0.1-v0.10 showed the winning recipe is:
    SE-gated channel split + dual-stream routing + hybrid (fused / shared-
    inverted) experts + channel shuffle + complexity gate. Every added visual
    module afterwards (low-rank bottleneck v0.7, refine v0.8, detail v0.9,
    context v0.10) produced diminishing or negative mAP returns. v0.11 keeps
    the v0.6 core forward path completely intact and upgrades only the single
    most impactful component - the router - with two cheap, differentiable,
    DDP-safe refinements:

    1. Normalized dual-stream routing (``DualStreamGateRouterV2``): LayerNorm on
       channel statistics gives stable global routing logits.
    2. Learnable per-expert prior bias for auxiliary-loss-free-style load
       balancing. It is a plain parameter (gradient all-reduced by DDP), so it
       avoids the usage-based buffers that caused the v0.3 DDP-sync crash.

    The paired config (``yolo-master-v0_11.yaml``) additionally tunes
    ``split_ratio`` per insertion point - more dynamic capacity at the shallow
    P3 junction, more static capacity at the deep P5 stage - instead of a fixed
    0.5 everywhere, following the report's hyperparameter recommendation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        split_ratio: float = 0.5,
        num_groups: int = 8,
        initial_temperature: float = 1.2,
        final_temperature: float = 0.5,
        balance_loss_coeff: float = 1.0,
        router_z_loss_coeff: float = 1.0,
        entropy_loss_coeff: float = 0.01,
        fused_expert_threshold: int = 8,
        shuffle_groups: int = 2,
    ):
        super().__init__(
            in_channels,
            out_channels,
            num_experts,
            top_k,
            split_ratio,
            num_groups,
            initial_temperature,
            final_temperature,
            balance_loss_coeff,
            router_z_loss_coeff,
            entropy_loss_coeff,
            fused_expert_threshold,
            shuffle_groups,
        )
        # Drop-in upgrade of the router (same I/O contract as v0.6).
        self.routing = DualStreamGateRouterV2(
            self.dynamic_channels, num_experts, top_k,
            temperature=initial_temperature,
        )
        self._init_weights()  # re-init the swapped-in router


class LowRankHybridAdaptiveGateMoE(HybridAdaptiveGateMoE):
    """
    v0.7 MoE: hybrid routing with low-rank fused experts.

    Compared with v0.6, layers that use the fused backend first project the
    dynamic branch into a compact bottleneck before expert computation. Large
    expert-count layers still use `SharedInvertedExpertGroup` to avoid dense
    all-expert work.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        split_ratio: float = 0.5,
        num_groups: int = 8,
        initial_temperature: float = 1.2,
        final_temperature: float = 0.5,
        balance_loss_coeff: float = 1.0,
        router_z_loss_coeff: float = 1.0,
        entropy_loss_coeff: float = 0.01,
        fused_expert_threshold: int = 8,
        shuffle_groups: int = 2,
        bottleneck_ratio: float = 0.5,
    ):
        super().__init__(
            in_channels,
            out_channels,
            num_experts,
            top_k,
            split_ratio,
            num_groups,
            initial_temperature,
            final_temperature,
            balance_loss_coeff,
            router_z_loss_coeff,
            entropy_loss_coeff,
            fused_expert_threshold,
            shuffle_groups,
        )
        self.bottleneck_ratio = bottleneck_ratio
        if num_experts <= fused_expert_threshold:
            self.expert_backend = "low_rank_fused"
            self.fused_experts = LowRankFusedExpertGroup(
                self.dynamic_channels,
                self.out_dynamic,
                num_experts,
                num_groups,
                top_k=top_k,
                bottleneck_ratio=bottleneck_ratio,
            )
            self._init_weights()  # re-init swapped-in experts


class RefinedLowRankHybridAdaptiveGateMoE(LowRankHybridAdaptiveGateMoE):
    """
    v0.8 MoE: low-rank hybrid experts with lightweight feature refinement.

    This builds on v0.7 and adds a residual depthwise refinement block after
    static/dynamic channel mixing. The refinement is gated by global context so
    it can emphasize boundary/texture channels without forcing extra expert
    computation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        split_ratio: float = 0.5,
        num_groups: int = 8,
        initial_temperature: float = 1.2,
        final_temperature: float = 0.5,
        balance_loss_coeff: float = 1.0,
        router_z_loss_coeff: float = 1.0,
        entropy_loss_coeff: float = 0.01,
        fused_expert_threshold: int = 8,
        shuffle_groups: int = 2,
        bottleneck_ratio: float = 0.5,
        refine_reduction: int = 8,
    ):
        super().__init__(
            in_channels,
            out_channels,
            num_experts,
            top_k,
            split_ratio,
            num_groups,
            initial_temperature,
            final_temperature,
            balance_loss_coeff,
            router_z_loss_coeff,
            entropy_loss_coeff,
            fused_expert_threshold,
            shuffle_groups,
            bottleneck_ratio,
        )
        refine_hidden = max(out_channels // refine_reduction, 8)
        self.feature_refiner = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.GroupNorm(get_safe_groups(out_channels, num_groups), out_channels),
            nn.SiLU(inplace=True),
        )
        self.feature_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, refine_hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(refine_hidden, out_channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.refine_scale = nn.Parameter(torch.tensor(0.1))

    def _refine_features(self, x):
        return x + torch.tanh(self.refine_scale) * self.feature_refiner(x) * self.feature_gate(x)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.training:
            self._update_temperature()
            self.training_step += 1
            self._training_step_value += 1

        gate_weights = self.se_gate(x)
        gate_static = gate_weights[:, :self.static_channels].unsqueeze(-1).unsqueeze(-1)
        gate_dynamic = gate_weights[:, self.static_channels:].unsqueeze(-1).unsqueeze(-1)

        x_static = x[:, :self.static_channels, :, :] * gate_static
        x_dynamic = x[:, self.static_channels:, :, :] * gate_dynamic

        out_static = self.static_net(x_static)
        complexity = self._safe_complexity(x_dynamic)

        routing_weights, routing_indices, routing_stats = self.routing(x_dynamic)
        routing_weights, routing_indices, routing_stats, adaptive_top_k = self._apply_complexity_gate(
            routing_weights, routing_indices, routing_stats, complexity
        )
        out_dynamic = self.fused_experts(x_dynamic, routing_weights, routing_indices, adaptive_top_k)

        out_concat = self._channel_shuffle(torch.cat([out_static, out_dynamic], dim=1))
        out_concat = self._refine_features(out_concat)
        out = self.proj(out_concat)
        out = self.bn(out) + x

        if self.training:
            router_probs = routing_stats.get('router_probs')
            router_logits = routing_stats.get('router_logits')
            topk_indices = routing_stats.get('topk_indices')
            if isinstance(router_probs, torch.Tensor) and isinstance(router_logits, torch.Tensor):
                aux_loss = self.moe_loss_fn(router_probs, router_logits, topk_indices)
                _registry_set(self, aux_loss)
                _record_moe_snapshot(
                    self,
                    expert_usage=routing_stats.get('expert_usage'),
                    topk_indices=topk_indices,
                    topk_weights=routing_weights,
                    router_probs=router_probs,
                    aux_loss=aux_loss,
                )

        return out

    def get_gflops(self, input_shape):
        B, C, H, W = input_shape
        flops = super().get_gflops(input_shape)
        extra = FlopsUtils.count_conv2d(self.feature_refiner, (B, self.out_channels, H, W))
        hidden = self.feature_gate[1].out_channels
        extra += B * self.out_channels * hidden + B * hidden * self.out_channels
        flops['feature_refiner'] = extra / 1e9
        flops['total_gflops'] = sum(v for k, v in flops.items() if k != 'total_gflops')
        return flops


class DetailAwareLowRankHybridAdaptiveGateMoE(LowRankHybridAdaptiveGateMoE):
    """
    Visual MoE focused on boundaries, textures, and small-object details.

    The detail gate enhances the dynamic branch before routing, allowing the
    router and experts to see high-frequency residual cues without adding a
    heavy edge detector or task-specific supervision.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        split_ratio: float = 0.5,
        num_groups: int = 8,
        initial_temperature: float = 1.2,
        final_temperature: float = 0.5,
        balance_loss_coeff: float = 1.0,
        router_z_loss_coeff: float = 1.0,
        entropy_loss_coeff: float = 0.01,
        fused_expert_threshold: int = 8,
        shuffle_groups: int = 2,
        bottleneck_ratio: float = 0.5,
        detail_reduction: int = 8,
    ):
        super().__init__(
            in_channels,
            out_channels,
            num_experts,
            top_k,
            split_ratio,
            num_groups,
            initial_temperature,
            final_temperature,
            balance_loss_coeff,
            router_z_loss_coeff,
            entropy_loss_coeff,
            fused_expert_threshold,
            shuffle_groups,
            bottleneck_ratio,
        )
        self.detail_gate = VisualDetailGate(self.dynamic_channels, num_groups, detail_reduction)

    def forward(self, x):
        return _run_visual_hybrid_moe_forward(self, x, detail_gate=self.detail_gate)

    def get_gflops(self, input_shape):
        B, C, H, W = input_shape
        flops = super().get_gflops(input_shape)
        flops['detail_gate'] = self.detail_gate.compute_flops((B, self.dynamic_channels, H, W)) / 1e9
        flops['total_gflops'] = sum(v for k, v in flops.items() if k != 'total_gflops')
        return flops


class ContextRefinedLowRankHybridAdaptiveGateMoE(RefinedLowRankHybridAdaptiveGateMoE):
    """
    Visual MoE focused on multi-scale context aggregation.

    This variant adds pooled pyramid context after static/dynamic channel
    mixing, then applies the v0.8 refinement block. It is useful for detection
    and segmentation features where local evidence needs broader context.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        split_ratio: float = 0.5,
        num_groups: int = 8,
        initial_temperature: float = 1.2,
        final_temperature: float = 0.5,
        balance_loss_coeff: float = 1.0,
        router_z_loss_coeff: float = 1.0,
        entropy_loss_coeff: float = 0.01,
        fused_expert_threshold: int = 8,
        shuffle_groups: int = 2,
        bottleneck_ratio: float = 0.5,
        refine_reduction: int = 8,
    ):
        super().__init__(
            in_channels,
            out_channels,
            num_experts,
            top_k,
            split_ratio,
            num_groups,
            initial_temperature,
            final_temperature,
            balance_loss_coeff,
            router_z_loss_coeff,
            entropy_loss_coeff,
            fused_expert_threshold,
            shuffle_groups,
            bottleneck_ratio,
            refine_reduction,
        )
        self.context_mixer = PyramidContextMixer(out_channels, num_groups)

    def forward(self, x):
        return _run_visual_hybrid_moe_forward(
            self,
            x,
            context_mixer=self.context_mixer,
            refine_features=True,
        )

    def get_gflops(self, input_shape):
        B, C, H, W = input_shape
        flops = super().get_gflops(input_shape)
        flops['context_mixer'] = self.context_mixer.compute_flops((B, self.out_channels, H, W)) / 1e9
        flops['total_gflops'] = sum(v for k, v in flops.items() if k != 'total_gflops')
        return flops


class VisualEnhancedAdaptiveGateMoE(ContextRefinedLowRankHybridAdaptiveGateMoE):
    """
    Full visual MoE: detail-aware routing plus multi-scale refined fusion.

    It combines high-frequency detail conditioning before expert routing with
    pyramid context after static/dynamic fusion. This is the richest visual
    block in the current family and is intended for ablation against v0.8.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        split_ratio: float = 0.5,
        num_groups: int = 8,
        initial_temperature: float = 1.2,
        final_temperature: float = 0.5,
        balance_loss_coeff: float = 1.0,
        router_z_loss_coeff: float = 1.0,
        entropy_loss_coeff: float = 0.01,
        fused_expert_threshold: int = 8,
        shuffle_groups: int = 2,
        bottleneck_ratio: float = 0.5,
        refine_reduction: int = 8,
        detail_reduction: int = 8,
    ):
        super().__init__(
            in_channels,
            out_channels,
            num_experts,
            top_k,
            split_ratio,
            num_groups,
            initial_temperature,
            final_temperature,
            balance_loss_coeff,
            router_z_loss_coeff,
            entropy_loss_coeff,
            fused_expert_threshold,
            shuffle_groups,
            bottleneck_ratio,
            refine_reduction,
        )
        self.detail_gate = VisualDetailGate(self.dynamic_channels, num_groups, detail_reduction)

    def forward(self, x):
        return _run_visual_hybrid_moe_forward(
            self,
            x,
            detail_gate=self.detail_gate,
            context_mixer=self.context_mixer,
            refine_features=True,
        )

    def get_gflops(self, input_shape):
        B, C, H, W = input_shape
        flops = super().get_gflops(input_shape)
        flops['detail_gate'] = self.detail_gate.compute_flops((B, self.dynamic_channels, H, W)) / 1e9
        flops['total_gflops'] = sum(v for k, v in flops.items() if k != 'total_gflops')
        return flops

class AdaptiveBalanceController(nn.Module):
    """
    Adaptive Load Balancing Controller.

    Strategies:
    1. Early Training: High weight, forcing balance.
    2. Mid Training: Gradually decrease weight.
    3. Late Training: Low weight, allowing expert differentiation.
    """
    
    def __init__(
        self,
        num_experts,
        initial_coeff=1.0,
        final_coeff=0.1,
        decay_steps=50000,
        dynamic_scheduler=None,
        dynamic_scheduler_config=None,
    ):
        # NOTE(rev5): coeff raised from 0.1/0.001 -> 1.0/0.1 so the GShard-scale
        # balance term stays O(0.1..1), on par with other MoE blocks. The old
        # defaults shrank a ~1.0 balance to ~0.005 and got silently dominated
        # when summed with GShard-scale aux losses.
        super().__init__()
        self.num_experts = num_experts
        self.initial_coeff = initial_coeff
        self.final_coeff = final_coeff
        self.decay_steps = decay_steps
        self.dynamic_scheduler = dynamic_scheduler or (
            MoEDynamicScheduler(dynamic_scheduler_config) if dynamic_scheduler_config is not None else None
        )
        self.last_dynamic_schedule = None
        
        # Learnable expert importance weights
        self.expert_importance = nn.Parameter(torch.ones(num_experts))
    
    def forward(self, routing_stats, training_step):
        """Calculate adaptive load balancing loss."""
        expert_usage = routing_stats['expert_usage']  # [num_experts]
        
        # === 1. Dynamic Coefficient Decay ===
        progress = min(1.0, training_step.float() / self.decay_steps)
        current_coeff = self.initial_coeff * (1 - progress) + self.final_coeff * progress
        if self.dynamic_scheduler is not None:
            schedule_state = self.dynamic_scheduler.step(expert_usage, float(current_coeff))
            current_coeff = schedule_state.balance_loss_coeff
            self.last_dynamic_schedule = schedule_state.to_dict()
        
        # === 2. Differentiable Load Balancing (GShard scale, grad -> router) ===
        # importance = mean(router_probs) keeps the gradient path to the router;
        # the learnable expert_importance acts as a (soft) target prior. Falls
        # back to the usage-only weighted form if router_probs is missing.
        importance_weights = F.softmax(self.expert_importance, dim=0)
        router_probs = routing_stats.get('router_probs')
        if isinstance(router_probs, torch.Tensor):
            balance_loss = differentiable_balance_loss(
                router_probs, expert_usage, self.num_experts, target_usage=importance_weights
            )
        else:
            balance_loss = weighted_gshard_balance_loss(expert_usage, importance_weights, self.num_experts, reduce_ddp=True)

        # === 3. Entropy Regularization (Encourage Diversity, non-negative) ===
        # Penalize LOW entropy (collapse); max entropy = log(N) -> penalty 0.
        expert_usage_safe = expert_usage.clamp(min=1e-6)
        entropy = -(expert_usage_safe * torch.log(expert_usage_safe)).sum()
        max_entropy = math.log(max(self.num_experts, 2))
        entropy_penalty = (max_entropy - entropy).clamp_min(0.0) / max_entropy  # in [0,1]

        total_loss = current_coeff * (balance_loss + getattr(self, 'entropy_coeff', 0.1) * entropy_penalty)

        # Guard against NaN loss (graph-safe: keep grad_fn instead of new leaf)
        if not torch.isfinite(total_loss).all():
            total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)

        return total_loss

class OptimalHybridGateMoE(HybridAdaptiveGateMoEv2):
    """
    v0.12 MoE: the production-optimal synthesis of all v0.1-v0.11 findings.

    Design rationale (every choice is backed by the module-level ablation):
    ──────────────────────────────────────────────────────────────────────
    1. **v0.6 core forward path** — SE-gated split + dual-stream routing +
       hybrid (fused/shared-inverted) experts + channel shuffle + complexity
       gate. This is the single best-performing combination (mAP50-95=0.61017).
       Every module added afterwards (low-rank v0.7, refine v0.8, detail v0.9,
       context v0.10) produced diminishing or negative returns and is dropped.

    2. **v0.11 router upgrade** — DualStreamGateRouterV2 normalizes channel
       statistics with LayerNorm and adds a learnable per-expert prior bias
       for auxiliary-loss-free load balancing. Both are cheap, fully
       differentiable, and DDP-safe (no usage-based buffer updates that broke
       v0.3).

    3. **Layer-adaptive split_ratio** — Instead of a fixed 0.5 everywhere,
       the YAML config passes a per-insertion-point split_ratio. Shallow P3
       gets more dynamic capacity (split_ratio=0.5), deep P5 shifts to more
       static capacity (split_ratio=0.375) where feature maps are small and
       spatial redundancy is low. This follows the report's hyperparameter
       recommendation and avoids grid-search overhead.

    4. **Lightweight residual DW refinement** — A single depthwise 3×3 conv
       with a global SE gate is applied after channel mixing, *only* when
       ``refine=True``. This is far lighter than v0.8's full refine block
       (which added a separate GroupNorm + activation chain) and is the
       minimal viable enhancement: it gives the projection layer a slightly
       better-conditioned input without introducing the over-design that
       hurt v0.7-v0.10. The refine_scale starts at 0.1 so the block is nearly
       identity at init, avoiding training disruption.

    5. **Temperature schedule** — cosine annealing from 1.2 → 0.5 over 2000
       steps (inherited from v0.6, shorter than v0.1's 5000). The high initial
       temperature encourages exploration; the low final temperature sharpens
       routing for inference.

    DDP safety: all state is either nn.Parameter (auto-synced) or a Python int
    counter. No buffer-based training_step updates, no .item() sync points.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        split_ratio: float = 0.5,
        num_groups: int = 8,
        initial_temperature: float = 1.2,
        final_temperature: float = 0.5,
        balance_loss_coeff: float = 1.0,
        router_z_loss_coeff: float = 1.0,
        entropy_loss_coeff: float = 0.01,
        fused_expert_threshold: int = 8,
        shuffle_groups: int = 2,
        refine: bool = True,
        refine_reduction: int = 8,
    ):
        super().__init__(
            in_channels,
            out_channels,
            num_experts,
            top_k,
            split_ratio,
            num_groups,
            initial_temperature,
            final_temperature,
            balance_loss_coeff,
            router_z_loss_coeff,
            entropy_loss_coeff,
            fused_expert_threshold,
            shuffle_groups,
        )

        # ── Lightweight residual DW refinement ──
        # Single DW conv + global SE gate. Far lighter than v0.8's refine block.
        # refine_scale=0.1 → near-identity at init, safe for short schedules.
        self.refine = refine
        if refine:
            refine_hidden = max(out_channels // refine_reduction, 8)
            self.refine_dw = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1,
                          groups=out_channels, bias=False),
                nn.GroupNorm(get_safe_groups(out_channels, num_groups), out_channels),
            )
            self.refine_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, refine_hidden, 1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(refine_hidden, out_channels, 1, bias=True),
                nn.Sigmoid(),
            )
            self.refine_scale = nn.Parameter(torch.tensor(0.1))

        # Router noise decay schedule: linearly decay over 1000 steps (half of
        # the typical 2000-step cosine temperature schedule). After this, the
        # router is noise-free for the remaining training.
        self._noise_decay_steps = 1000
        self._init_weights()

    def _apply_refine(self, x: torch.Tensor) -> torch.Tensor:
        """Residual DW refinement: x + tanh(scale) * DW(x) * SE(x)."""
        refined = self.refine_dw(x) * self.refine_gate(x)
        return x + torch.tanh(self.refine_scale) * refined

    def forward(self, x):
        B, C, H, W = x.shape

        if self.training:
            self._update_temperature()
            self.training_step += 1
            self._training_step_value += 1
            # Advance router noise decay (linear, buffer-based, DDP-safe).
            if hasattr(self.routing, '_noise_progress'):
                progress = min(1.0, self._training_step_value / self._noise_decay_steps)
                self.routing._noise_progress.fill_(progress)

        # ── 1. SE-Gated Channel Allocation ──
        gate_weights = self.se_gate(x)
        gate_static = gate_weights[:, :self.static_channels].unsqueeze(-1).unsqueeze(-1)
        gate_dynamic = gate_weights[:, self.static_channels:].unsqueeze(-1).unsqueeze(-1)

        x_static = x[:, :self.static_channels, :, :] * gate_static
        x_dynamic = x[:, self.static_channels:, :, :] * gate_dynamic

        # ── 2. Static Path ──
        out_static = self.static_net(x_static)

        # ── 3. Complexity Estimation ──
        complexity = self._safe_complexity(x_dynamic)

        # ── 4. Dual-Stream V2 Routing (normalized + prior bias) ──
        routing_weights, routing_indices, routing_stats = self.routing(x_dynamic)
        routing_weights, routing_indices, routing_stats, adaptive_top_k = \
            self._apply_complexity_gate(
                routing_weights, routing_indices, routing_stats, complexity
            )

        # ── 5. Hybrid Expert Computation ──
        out_dynamic = self.fused_experts(
            x_dynamic, routing_weights, routing_indices, adaptive_top_k
        )

        # ── 6. Channel Shuffle + Optional Refinement ──
        out_concat = self._channel_shuffle(torch.cat([out_static, out_dynamic], dim=1))
        if self.refine:
            out_concat = self._apply_refine(out_concat)

        # ── 7. Projection + Residual ──
        out = self.proj(out_concat)
        out = self.bn(out) + x

        # ── 8. Auxiliary Loss ──
        if self.training:
            router_probs = routing_stats.get('router_probs')
            router_logits = routing_stats.get('router_logits')
            topk_indices = routing_stats.get('topk_indices')
            if isinstance(router_probs, torch.Tensor) and isinstance(router_logits, torch.Tensor):
                aux_loss = self.moe_loss_fn(router_probs, router_logits, topk_indices)
                _registry_set(self, aux_loss)
                _record_moe_snapshot(
                    self,
                    expert_usage=routing_stats.get('expert_usage'),
                    topk_indices=topk_indices,
                    topk_weights=routing_weights,
                    router_probs=router_probs,
                    aux_loss=aux_loss,
                )

        return out

    def get_gflops(self, input_shape):
        B, C, H, W = input_shape
        flops = super().get_gflops(input_shape)
        if self.refine:
            flops['refine_dw'] = FlopsUtils.count_conv2d(
                self.refine_dw, (B, self.out_channels, H, W)) / 1e9
            flops['refine_gate'] = FlopsUtils.count_conv2d(
                self.refine_gate, (B, self.out_channels, 1, 1)) / 1e9
            flops['total_gflops'] = sum(
                v for k, v in flops.items() if k != 'total_gflops'
            )
        return flops

