# 🐧Please note that this file has been modified by Tencent on 2026/02/13. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Mixture-of-Experts (MoE) modules, routing layers, and compatibility shims.

This module provides several MoE variants and routers optimized for inference efficiency,
plus backward-compatibility aliases so legacy checkpoints can be loaded without changes.
All public class/function names are preserved; only comments/docstrings have been clarified.

Architecture (split for maintainability):
  - ``_helpers.py``: shared registry, autocast wrapper, snapshot/diagnostic utils, deepcopy.
  - ``gated.py``: gated MoE family (DualStream routers, AdaptiveGate variants, fused experts).
  - ``modules.py`` (this file): base MoE classes + ultimate MoE classes + re-exports.

All symbols from ``_helpers`` and ``gated`` are re-exported here so that
``from .modules import X`` continues to work unchanged.
"""
import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Union

from .utils import FlopsUtils, get_safe_groups, BatchedExpertComputation
from .experts import (
    OptimizedSimpleExpert, FusedGhostExpert, SimpleExpert, GhostExpert,
    InvertedResidualExpert, EfficientExpertGroup, SpatialExpert, SharedInvertedExpertGroup
)
from .routers import (
    UltraEfficientRouter, EfficientSpatialRouter, LocalRoutingLayer,
    AdaptiveRoutingLayer, DynamicRoutingLayer, AdvancedRoutingLayer
)
from ultralytics.nn.modules.block import ABlock, A2C2f, C3k
from .loss import MoELoss, gshard_balance_loss, weighted_gshard_balance_loss, differentiable_balance_loss, all_reduce_mean, should_reduce_ddp
from .scheduler import MoEDynamicScheduler, MoEDynamicSchedulerConfig

# Re-export all helpers (backward compatibility for external imports)
from ._helpers import (
    autocast,
    MOE_LOSS_REGISTRY,
    MOE_SNAPSHOT_INTERVAL,
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

# Re-export all gated-family classes and functions
from .gated import (
    DualStreamGateRouter,
    DualStreamGateRouterV2,
    AdaptiveGateMoE,
    HyperSplitMoE,
    HyperFusedMoE,
    ZeroCostRouter,
    FusedExpertGroup,
    LowRankFusedExpertGroup,
    VisualDetailGate,
    PyramidContextMixer,
    FusedAdaptiveGateMoE,
    HybridAdaptiveGateMoE,
    HybridAdaptiveGateMoEv2,
    LowRankHybridAdaptiveGateMoE,
    RefinedLowRankHybridAdaptiveGateMoE,
    DetailAwareLowRankHybridAdaptiveGateMoE,
    ContextRefinedLowRankHybridAdaptiveGateMoE,
    VisualEnhancedAdaptiveGateMoE,
    AdaptiveBalanceController,
    OptimalHybridGateMoE,
    MultiHeadRouterV3,
    DiversifiedExpertGroup,
    CrossPathGate,
    MultiHeadRouterMoE,
    DiversifiedExpertMoE,
    GatedFusionMoE,
    UltraLightRouter,
    MatMulFusedExperts,
    _pool_to_size_mps_safe,
    _run_visual_hybrid_moe_forward
)

_MOE_FINITE_DIAGNOSTICS = os.environ.get("MOE_FINITE_DIAGNOSTICS", "0").lower() in {"1", "true", "yes", "on"}
_MOE_FINITE_DIAGNOSTIC_MAX_EVENTS = max(int(os.environ.get("MOE_FINITE_DIAGNOSTIC_MAX_EVENTS", "1")), 1)


# ==========================================
# Base MoE modules (UltraOptimizedMoE family)
# ==========================================

class UltraOptimizedMoE(nn.Module):
    """
    Ultra-optimized MoE with efficient routing, batched computation, and conditional execution.
    Features: Ultra-efficient router, batched experts, GroupNorm stability, and mixed-precision support.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_experts: int = 4,
            top_k: int = 2,
            expert_type: str = 'simple',  # 'simple', 'ghost', 'inverted'
            router_reduction: int = 16,
            router_pool_scale: int = 8,
            noise_std: float = 1.0,
            router_temperature: float = 1.0,
            balance_loss_coeff: float = 1.0,
            router_z_loss_coeff: float = 1.0,
            num_groups: int = 8,
            weight_threshold: float = 0.01  # conditional compute threshold
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_type = expert_type
        self.balance_loss_coeff = balance_loss_coeff
        self.router_z_loss_coeff = router_z_loss_coeff
        self.weight_threshold = weight_threshold

        # Ultra-lightweight router
        self.routing = UltraEfficientRouter(
            in_channels,
            num_experts,
            reduction=router_reduction,
            top_k=top_k,
            noise_std=noise_std,
            temperature=router_temperature,
            pool_scale=router_pool_scale
        )

        # Expert pool (optimized variants)
        self.experts = nn.ModuleList()
        if expert_type == 'ghost':
            for _ in range(num_experts):
                self.experts.append(FusedGhostExpert(in_channels, out_channels, num_groups=num_groups))
        elif expert_type == 'inverted':
            for _ in range(num_experts):
                self.experts.append(InvertedResidualExpert(in_channels, out_channels))
        else:
            for _ in range(num_experts):
                self.experts.append(OptimizedSimpleExpert(in_channels, out_channels, num_groups=num_groups))

        # Shared expert (with GroupNorm)
        self.shared_expert = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(get_safe_groups(out_channels, num_groups), out_channels),
            nn.SiLU(inplace=True)
        )

        self._init_weights()

        # Performance statistics
        self.last_aux_loss = 0.0
        self.last_balance_loss = 0.0
        self.last_z_loss = 0.0
        self.last_routing_snapshot = {}
        # self.aux_loss is now managed via MOE_LOSS_REGISTRY property

    def _init_weights(self):
        """Improved initialization strategy"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use He initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Router-specific init (enough variance for input-dependent routing)
        if hasattr(self.routing.router[-1], 'weight'):
            nn.init.normal_(self.routing.router[-1].weight, std=0.05)
            if self.routing.router[-1].bias is not None:
                nn.init.constant_(self.routing.router[-1].bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1) Routing computation (ultra-lightweight)
        routing_result = self.routing(x)
        routing_weights, routing_indices = routing_result[:2]

        # 2) Shared expert (parallel computation)
        shared_output = self.shared_expert(x)

        # 3) Batched sparse expert computation (key optimization)
        expert_output = BatchedExpertComputation.compute_sparse_experts_batched(
            x,
            self.experts,
            routing_weights,
            routing_indices,
            self.top_k,
            self.num_experts
        )

        # 4) Fuse outputs
        output = shared_output + expert_output

        # 5) Auxiliary loss computation
        if self.training:
            usage_freq, importance, z_loss_val = routing_result[2:]

            if importance is None:
                importance = torch.zeros(self.num_experts, device=x.device)
            if z_loss_val is None:
                z_loss_val = torch.tensor(0.0, device=x.device, dtype=x.dtype)

            # Standard GShard differentiable form N*sum(importance*usage),
            # same as loss.differentiable_balance_loss. importance keeps grad
            # to the router; usage is detached. DDP-average both so all ranks
            # optimise one global balance target (no-op on single GPU).
            balance_loss = differentiable_balance_loss(
                importance.unsqueeze(0),
                usage_freq,
                self.num_experts,
                reduce_ddp=should_reduce_ddp(self),
            )

            effective_balance_coeff = self.balance_loss_coeff
            if getattr(self, 'map_saturation_scheduler', None) is not None:
                effective_balance_coeff = self.map_saturation_scheduler.apply(effective_balance_coeff)

            aux_loss = (effective_balance_coeff * balance_loss) + (self.router_z_loss_coeff * z_loss_val)
            _registry_set(self, aux_loss)
            _record_moe_snapshot(
                self,
                expert_usage=usage_freq,
                topk_indices=routing_indices,
                topk_weights=routing_weights,
                aux_loss=aux_loss,
            )

            # Record statistics
            self.last_aux_loss = aux_loss.detach().item()
            self.last_balance_loss = balance_loss.detach().item()
            self.last_z_loss = z_loss_val.detach().item()

        return output

    @property
    def aux_loss(self):
        """Retrieve the auxiliary loss from the registry."""
        return _get_moe_aux_loss(self)

    def __deepcopy__(self, memo):
        return _robust_deepcopy(self, memo)

    def get_gflops(self, input_shape: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Compute GFLOPs"""
        B, C, H, W = input_shape
        flops_dict = {}

        # 1. Router FLOPs
        routing_flops = self.routing.compute_flops(input_shape)
        flops_dict['routing'] = routing_flops / 1e9

        # 2. Shared Expert FLOPs
        shared_flops = FlopsUtils.count_conv2d(self.shared_expert[0], input_shape)
        flops_dict['shared_expert'] = shared_flops / 1e9

        # 3. Sparse Experts FLOPs
        single_expert_flops = self.experts[0].compute_flops((1, C, H, W))
        total_sparse_flops = single_expert_flops * B * self.top_k
        flops_dict['sparse_experts'] = total_sparse_flops / 1e9

        # Total
        total_flops = routing_flops + shared_flops + total_sparse_flops
        flops_dict['total_gflops'] = total_flops / 1e9

        return flops_dict

    def get_efficiency_stats(self, input_shape: Tuple[int, int, int, int]) -> Dict[str, any]:
        """Get detailed efficiency statistics"""
        flops = self.get_gflops(input_shape)

        return {
            'gflops': flops,
            'router_percentage': flops['routing'] / flops['total_gflops'] * 100,
            'experts_percentage': flops['sparse_experts'] / flops['total_gflops'] * 100,
            'num_params': sum(p.numel() for p in self.parameters()) / 1e6,  # Millions
            'last_aux_loss': self.last_aux_loss,
            'last_balance_loss': self.last_balance_loss,
            'last_z_loss': self.last_z_loss
        }


# ==========================================
# Advanced optimization: dynamic expert capacity
# ==========================================

class AdaptiveCapacityMoE(UltraOptimizedMoE):
    """Complexity-adaptive MoE: scales the expert-mixture contribution by a
    learned input-complexity factor.

    Design note (rev: 2026-06-25)
    ─────────────────────────────
    The previous implementation tried to vary the *discrete* ``top_k`` by
    temporarily mutating ``self.routing.top_k`` inside ``forward``. That had
    three defects: (1) the parent forward reads ``self.top_k`` (not
    ``self.routing.top_k``) for expert computation, so the mutation had no real
    effect; (2) ``min(self.top_k, …)`` capped capacity at the base value, so it
    could only shrink and almost always saturated at ``top_k``, making the
    "adaptive" knob inert; (3) ``int(score.item())`` forced a GPU→CPU sync and
    mutating instance state mid-forward is not re-entrant / thread-safe.

    This version keeps ``top_k`` fixed and instead modulates the *output*
    expert contribution by a differentiable, sync-free complexity factor in a
    band around 1.0 (``[1/cf, cf]``). Higher complexity → larger expert
    contribution (more "capacity"); lower complexity → smaller. The factor is
    computed without any ``.item()`` call, is re-entrant, and genuinely varies.
    """

    def __init__(self, *args, capacity_factor: float = 1.5, **kwargs):
        super().__init__(*args, **kwargs)
        # capacity_factor > 1 defines the modulation band [1/cf, cf]
        self.capacity_factor = max(float(capacity_factor), 1.0)

        # Complexity estimator → scalar in (0, 1) per sample
        self.complexity_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.in_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Parent computes shared + sparse experts; scale only the sparse path.
        B, C, H, W = x.shape
        routing_result = self.routing(x)
        routing_weights, routing_indices = routing_result[:2]
        shared_output = self.shared_expert(x)
        expert_output = BatchedExpertComputation.compute_sparse_experts_batched(
            x,
            self.experts,
            routing_weights,
            routing_indices,
            self.top_k,
            self.num_experts,
        )
        if self.capacity_factor <= 1.0:
            result = shared_output + expert_output
        else:
            s = self.complexity_estimator(x).mean()
            scale = torch.exp((2.0 * s - 1.0) * math.log(self.capacity_factor))
            result = shared_output + expert_output * scale

        if self.training:
            usage_freq, importance, z_loss_val = routing_result[2:]
            if importance is None:
                importance = torch.zeros(self.num_experts, device=x.device)
            if z_loss_val is None:
                z_loss_val = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            balance_loss = differentiable_balance_loss(
                importance.unsqueeze(0),
                usage_freq,
                self.num_experts,
                reduce_ddp=should_reduce_ddp(self),
            )

            effective_balance_coeff = self.balance_loss_coeff
            if getattr(self, 'map_saturation_scheduler', None) is not None:
                effective_balance_coeff = self.map_saturation_scheduler.apply(effective_balance_coeff)

            aux_loss = (effective_balance_coeff * balance_loss) + (self.router_z_loss_coeff * z_loss_val)
            _registry_set(self, aux_loss)
            _record_moe_snapshot(
                self,
                expert_usage=usage_freq.detach(),
                topk_indices=routing_indices,
                topk_weights=routing_weights,
                aux_loss=aux_loss,
            )
            self.last_aux_loss = aux_loss.detach().item()
            self.last_balance_loss = balance_loss.detach().item()
            self.last_z_loss = z_loss_val.detach().item()

        return result


class ES_MOE(nn.Module):
    """General MoE block with a routing network and multiple expert branches."""

    def __init__(self, in_channels, out_channels=None, num_experts=3, reduction=8,
                 top_k=None, use_sparse_inference=True, dynamic_threshold=0.4):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels (defaults to in_channels)
            num_experts: Number of expert branches
            reduction: Channel reduction ratio for the routing network
            top_k: Number of active experts; None means use all experts
            use_sparse_inference: Enable sparse Top-K expert computation during inference
            dynamic_threshold: Threshold for pruning low-confidence experts during inference
        """
        super(ES_MOE, self).__init__()

        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts) if top_k is not None else num_experts
        self.use_top_k = (top_k is not None)
        self.use_sparse_inference = use_sparse_inference
        self.dynamic_threshold = dynamic_threshold

        # Dynamic routing (Top-K supported)
        self.routing = DynamicRoutingLayer(in_channels, num_experts, reduction, top_k)

        # Expert group (original design)
        default_kernel_sizes = [3, 5, 7]
        if num_experts <= len(default_kernel_sizes):
            ks = default_kernel_sizes[:num_experts]
        else:
            ks = [3 + 2 * i for i in range(num_experts)]
        self.experts = nn.ModuleList(
            [EfficientExpertGroup(in_channels, out_channels, kernel_size=k) for k in ks]
        )

        # Output normalization (original design)
        self.norm = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

        # Load-balancing loss (original design)
        # NOTE: these are NOT registered as nn.Module buffers to avoid DDP
        # _sync_buffers broadcasting them across ranks (they are pure statistics
        # and do not need gradient synchronisation).  Keeping them as plain
        # attributes prevents "No backend type associated with device type cpu"
        # when a buffer lands on CPU before the model is moved to CUDA.
        self.load_balancing_loss = torch.tensor(0.0)
        self.expert_usage_counts = torch.zeros(num_experts)
        self.last_routing_snapshot = {}
        # Expose balance_loss_coeff for GiniBalanceScheduler / apply_balance_loss_coeff
        self.balance_loss_coeff = 1.0

    def _ensure_compat_attrs(self):
        """One-time legacy checkpoint attribute repair (not per-forward)."""
        if not hasattr(self, "use_top_k"):
            self.use_top_k = False
        if not hasattr(self, "use_sparse_inference"):
            self.use_sparse_inference = True
        if not hasattr(self, "num_experts"):
            self.num_experts = len(self.experts) if hasattr(self, "experts") else 1
        if not hasattr(self, "top_k"):
            self.top_k = self.num_experts

    def forward(self, x):
        self._ensure_compat_attrs()
        # Get routing weights
        routing_weights = self.routing(x)

        # Compute load-balancing loss
        load_balance_loss = self._compute_load_balancing_loss(routing_weights)

        # Record routing snapshot for diagnostics (training only)
        if self.training:
            _record_moe_snapshot(
                self,
                expert_usage=routing_weights.mean(dim=(0, 2, 3)),
                router_probs=routing_weights,
                aux_loss=load_balance_loss,
            )

        # Dense forward only during training (gradients to all experts) or when
        # exporting/tracing (sparse control-flow breaks exporters). For normal
        # eval/inference use the Top-K sparse path to reclaim the MoE speedup.
        use_dense = (
            self.training
            or torch.onnx.is_in_onnx_export()
            or torch.jit.is_tracing()
            or not getattr(self, "use_sparse_inference", True)
        )
        if use_dense:
            final_output = self._dense_forward(x, routing_weights)
        else:
            final_output = self._sparse_forward(x, routing_weights)

        # ``self.norm`` is always built in __init__; a missing attribute can
        # only come from an old partial checkpoint. Build it eagerly here only
        # outside tracing (so export never bakes in a freshly-created layer).
        if not hasattr(self, "norm"):
            if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
                raise RuntimeError("ES_MOE.norm missing during export; reload a complete checkpoint.")
            self.norm = nn.Sequential(
                nn.BatchNorm2d(final_output.shape[1]),
                nn.SiLU(inplace=True),
            ).to(final_output.device, final_output.dtype)
        final_output = self.norm(final_output)

        return final_output

    @property
    def aux_loss(self):
        """Retrieve the auxiliary loss from the registry."""
        return _get_moe_aux_loss(self)

    def get_gflops(self, input_shape: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Estimate GFLOPs for a single forward pass.

        Args:
            input_shape: ``(B, C, H, W)`` tuple.

        Returns:
            Dict with per-component and ``total_gflops`` keys.
        """
        B, C, H, W = input_shape
        # Router: ~C*hidden + hidden*num_experts MACs (reduction=8 by default)
        hidden = max(C // 8, self.num_experts * 2)
        router_macs = B * (C * hidden + hidden * self.num_experts) * H * W
        # Experts: each expert is a depthwise-separable conv ≈ C*C*k + C*C*1
        expert_macs = 0
        for expert in self.experts:
            for m in expert.modules():
                if isinstance(m, nn.Conv2d):
                    macs = B * m.in_channels * m.out_channels * (H // m.stride[0]) * (W // m.stride[1])
                    macs *= (m.kernel_size[0] * m.kernel_size[1]) / max(m.groups, 1)
                    expert_macs += macs
        # Norm
        norm_macs = B * self.out_channels * H * W * 2  # BN+SiLU ≈ 2 ops per element
        total = (router_macs + expert_macs + norm_macs) / 1e9
        return {
            "router_gflops": router_macs / 1e9,
            "experts_gflops": expert_macs / 1e9,
            "norm_gflops": norm_macs / 1e9,
            "total_gflops": total,
        }

    def _dense_forward(self, x, routing_weights):
        """Dense forward: compute all experts (used during training)."""
        final_output = 0
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            weight = routing_weights[:, i:i + 1, :, :]
            final_output = final_output + expert_out * weight
        return final_output

    def _sparse_forward(self, x, routing_weights):
        """Sparse forward: compute only Top-K experts (used during inference)."""
        B, E, H, W = routing_weights.shape

        # Compute per-expert importance
        routing_weights_flat = routing_weights.view(B, E, -1)
        expert_importance = routing_weights_flat.mean(dim=2)

        # Find Top-K experts
        topk_values, topk_indices = torch.topk(expert_importance, self.top_k, dim=1)

        # Initialize output
        final_output = torch.zeros_like(x)

        # Iterate over experts (vectorized over batch)
        for expert_idx in range(self.num_experts):
            # Find batch samples that selected this expert (avoid .any() GPU sync)
            mask = (topk_indices == expert_idx)
            batch_indices, k_ranks = torch.where(mask)
            if batch_indices.numel() == 0:
                continue
            # === Dynamic Pruning ===
            if hasattr(self, 'dynamic_threshold') and self.dynamic_threshold > 0:
                current_weights = routing_weights[batch_indices, expert_idx:expert_idx + 1, :, :]
                # Keep if (rank == 0) OR (weight >= threshold)
                weight_means = current_weights.mean(dim=(1, 2, 3))
                keep_mask = (k_ranks == 0) | (weight_means >= self.dynamic_threshold)

                batch_indices = batch_indices[keep_mask]
                if batch_indices.numel() == 0:
                    continue
            # =======================

            # Compute expert output for selected samples
            expert_out = self.experts[expert_idx](x[batch_indices])
            weight = routing_weights[batch_indices, expert_idx:expert_idx + 1, :, :]

            # Accumulate
            # fp16-safe: cast accumulator source to match final_output dtype (P0-2 fix)
            final_output.index_add_(0, batch_indices, (expert_out * weight).to(final_output.dtype))

        return final_output

    def _compute_load_balancing_loss(self, routing_weights, eps=1e-6):
        """Compute load-balancing loss (GShard scale, ~1.0 at balance)."""
        expert_usage = routing_weights.mean(dim=(0, 2, 3))
        # reduce_ddp=should_reduce_ddp(self) → usage averaged across ranks so all GPUs share one
        # global balance target (matches MoELoss; no-op on single GPU).
        load_balance_loss = gshard_balance_loss(expert_usage, self.num_experts, reduce_ddp=should_reduce_ddp(self))

        # Guard against NaN loss (graph-safe: keep grad_fn instead of new leaf)
        if not torch.isfinite(load_balance_loss).all():
            load_balance_loss = torch.nan_to_num(load_balance_loss, nan=0.0, posinf=0.0, neginf=0.0)
            
        if not hasattr(self, "load_balancing_loss"):
            self.register_buffer("load_balancing_loss", torch.tensor(0.0), persistent=False)
        if not hasattr(self, "expert_usage_counts"):
            self.register_buffer("expert_usage_counts", torch.zeros_like(expert_usage), persistent=False)
        if self.load_balancing_loss.shape == torch.Size([]):
            self.load_balancing_loss = self.load_balancing_loss.to(load_balance_loss.device).reshape(())
        self.load_balancing_loss.copy_(load_balance_loss.detach())
        self.expert_usage_counts.copy_(expert_usage.detach())
        
        # Store in registry (training only — avoids leaving graph-detached eval
        # tensors in the global registry that the loss collector could pick up).
        if self.training:
            _registry_set(self, load_balance_loss)
        
        return load_balance_loss

    def get_load_balancing_loss(self):
        """Get load-balancing loss."""
        return self.load_balancing_loss

    def get_expert_usage_stats(self):
        """Get expert usage statistics."""
        if self.expert_usage_counts.numel() > 0:
            stats = {
                'expert_usage': self.expert_usage_counts.cpu().tolist(),
                'usage_variance': self.expert_usage_counts.var().item(),
                'max_usage': self.expert_usage_counts.max().item(),
                'min_usage': self.expert_usage_counts.min().item()
            }
            if self.use_top_k:
                stats['active_experts'] = f"{self.top_k}/{self.num_experts}"
                stats['theoretical_speedup'] = f"{self.num_experts / self.top_k:.2f}x"
            return stats
        return None

    def set_top_k(self, top_k):
        """Dynamically adjust Top-K value."""
        if top_k is not None:
            self.top_k = min(top_k, self.num_experts)
            self.routing.top_k = self.top_k
            self.use_top_k = True
            self.routing.use_top_k = True
        else:
            self.top_k = self.num_experts
            self.use_top_k = False
            self.routing.use_top_k = False

    def enable_sparse_inference(self, enable=True):
        """Enable/disable sparse inference."""
        self.use_sparse_inference = enable

    def __deepcopy__(self, memo):
        return _robust_deepcopy(self, memo)


class OptimizedMOE(nn.Module):
    """MoE variant using an efficient spatial router and a shared expert path."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_experts: int = 4,
            top_k: int = 2,
            expert_expand_ratio: int = 2,
            balance_loss_coeff: float = 1.0,
            z_loss_coeff: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.out_channels = out_channels
        self.balance_loss_coeff = balance_loss_coeff
        self.z_loss_coeff = z_loss_coeff

        # 1) Router
        self.router = EfficientSpatialRouter(in_channels, num_experts, top_k=top_k)

        # 2) Sparse expert pool
        self.experts = nn.ModuleList([
            SimpleExpert(in_channels, out_channels, expand_ratio=expert_expand_ratio)
            for _ in range(num_experts)
        ])

        # 3) Shared Expert (key optimization)
        # Regardless of routing, all data flows through here to stabilize gradients.
        self.shared_expert = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

        self._init_weights()
        self.moe_loss_fn = MoELoss(
            balance_loss_coeff=balance_loss_coeff, 
            z_loss_coeff=z_loss_coeff, 
            num_experts=num_experts, 
            top_k=top_k
        )
        self.last_routing_snapshot = {}

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # [Key] Router init:
        # Initialize with moderate std (0.05) for input-dependent routing
        # while keeping initial probabilities reasonably uniform.
        if isinstance(self.router.router[-2], nn.Conv2d):
            nn.init.normal_(self.router.router[-2].weight, std=0.05)

    def forward(self, x):
        B, C, H, W = x.shape

        # -------------------------------------------
        # Step 1: routing selection
        # -------------------------------------------
        # routing_weights: [B, k, 1, 1], routing_indices: [B, k, 1, 1]
        routing_weights, routing_indices, loss_info = self.router(x)

        # -------------------------------------------
        # Step 2: shared expert forward (shared path)
        # -------------------------------------------
        shared_out = self.shared_expert(x)

        # -------------------------------------------
        # Step 3: sparse expert forward (dispatch)
        # -------------------------------------------
        expert_output = torch.zeros(B, self.out_channels, H, W, device=x.device, dtype=x.dtype)

        # Flatten for processing
        flat_indices = routing_indices.view(B, self.top_k)  # [B, k]
        flat_weights = routing_weights.view(B, self.top_k)  # [B, k]

        if torch.onnx.is_in_onnx_export():
            # ONNX tracing cannot capture data-dependent ``if mask.any()``
            # skips. Use a dense path: compute all experts, gather Top-K, sum.
            all_outs = torch.stack(
                [self.experts[i](x) for i in range(self.num_experts)], dim=1
            )  # [B, E, out_C, H, W]
            for k in range(self.top_k):
                idx_k = flat_indices[:, k]                                        # [B]
                w_k = flat_weights[:, k]                                          # [B]
                idx_exp = idx_k.view(B, 1, 1, 1, 1).expand(B, 1, self.out_channels, H, W)
                selected = torch.gather(all_outs, 1, idx_exp).squeeze(1)          # [B, out_C, H, W]
                if selected.dtype != expert_output.dtype:
                    selected = selected.to(expert_output.dtype)
                if w_k.dtype != expert_output.dtype:
                    w_k = w_k.to(expert_output.dtype)
                expert_output = expert_output + selected * w_k.view(B, 1, 1, 1)
        else:
            # Iterate over all experts
            for i in range(self.num_experts):
                # Find samples in batch that selected expert i
                # mask shape: [B, k]
                mask = (flat_indices == i)

                if mask.any():
                    # batch_idx: which sample
                    # k_idx: which choice (top-1 or top-2)
                    batch_idx, k_idx = torch.where(mask)

                    # Extract per-sample input
                    inp = x[batch_idx]

                    # Expert compute
                    out = self.experts[i](inp)

                    # Extract weights and reshape for broadcast: [selected_count, 1, 1, 1]
                    w = flat_weights[batch_idx, k_idx].view(-1, 1, 1, 1)

                    # Accumulate results (index_add_ faster than per-loop assignment)
                    # P0-2 fix: fp16-safe — cast to expert_output.dtype before index_add_
                    if out.dtype != expert_output.dtype:
                        out = out.to(expert_output.dtype)
                    if w.dtype != expert_output.dtype:
                        w = w.to(expert_output.dtype)

                    expert_output.index_add_(0, batch_idx, (out * w).to(expert_output.dtype))

        # Guard against activation explosion on routing collapse (all tokens -> 1 expert)
        expert_output = expert_output.clamp_(-1e4, 1e4)

        # Final output = shared path + sparse path
        final_output = shared_out + expert_output

        # -------------------------------------------
        # Step 4: auxiliary loss computation (train-time only)
        # -------------------------------------------
        if self.training and loss_info:
            aux_loss = self.moe_loss_fn(loss_info['router_probs'], loss_info['router_logits'],
                                             loss_info['topk_indices'])
            _registry_set(self, aux_loss)
            _record_moe_snapshot(
                self,
                expert_usage=loss_info['router_probs'].detach().mean(dim=0) if isinstance(loss_info.get('router_probs'), torch.Tensor) else None,
                topk_indices=loss_info.get('topk_indices'),
                topk_weights=routing_weights,
                router_probs=loss_info.get('router_probs'),
                aux_loss=aux_loss,
            )

        return final_output

    @property
    def aux_loss(self):
        """Retrieve the auxiliary loss from the registry."""
        return _get_moe_aux_loss(self)

    def get_gflops(self, input_shape: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Compute GFLOPs"""
        B, C, H, W = input_shape
        flops = {}

        # Router
        flops['router'] = self.router.compute_flops(input_shape) / 1e9

        # Shared Expert
        flops['shared'] = FlopsUtils.count_conv2d(self.shared_expert, input_shape) / 1e9

        # Sparse Experts (estimate by routing only Top-K experts per sample)
        single_expert_flops = self.experts[0].compute_flops((1, C, H, W))
        flops['sparse'] = (single_expert_flops * B * self.top_k) / 1e9

        flops['total'] = flops['router'] + flops['shared'] + flops['sparse']
        return flops

    def __deepcopy__(self, memo):
        return _robust_deepcopy(self, memo)


class OptimizedMOEImproved(nn.Module):
    """Improved MoE with pluggable routers/experts and a shared expert for stability."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_experts: int = 4,
            top_k: int = 2,
            expert_type: str = 'simple',  # ['simple', 'ghost', 'inverted', 'spatial']
            router_type: str = 'efficient',  # ['efficient', 'local', 'adaptive']
            noise_std: float = 1.0,
            balance_loss_coeff: float = 1.0,
            router_z_loss_coeff: float = 1.0,
            expert_expand_ratio: float = 2.0,
            progressive_sparsity: bool = True,
            detach_routing: bool = False,
            add_residual: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_loss_coeff = balance_loss_coeff
        self.router_z_loss_coeff = router_z_loss_coeff
        self.progressive_sparsity = progressive_sparsity
        # When embedded in an outer block that owns the residual (e.g. ABlockMoE),
        # disable the internal residual to avoid an implicit double-add.
        self.add_residual = add_residual
        # True: isolate router from main-task grads (legacy); False (default): let them flow.
        self.detach_routing = detach_routing

        # Progressive Sparsity (Python ints — no GPU→CPU sync from .item())
        self._training_step = 0
        self._current_top_k = num_experts
        self.warmup_steps = 5000

        # 1) Instantiate Router
        if router_type == 'local':
            self.routing = LocalRoutingLayer(in_channels, num_experts, top_k=top_k, noise_std=noise_std)
        elif router_type == 'adaptive':
            self.routing = AdaptiveRoutingLayer(in_channels, num_experts, top_k=top_k, noise_std=noise_std)
        else:
            self.routing = EfficientSpatialRouter(in_channels, num_experts, top_k=top_k, noise_std=noise_std)

        # 2) Instantiate Experts
        self.experts = nn.ModuleList()
        kwargs = {}
        if expert_type == 'ghost':
            expert_cls = GhostExpert
            kwargs['ratio'] = int(expert_expand_ratio)
        elif expert_type == 'inverted':
            expert_cls = InvertedResidualExpert
            kwargs['expand_ratio'] = expert_expand_ratio
        elif expert_type == 'spatial':
            expert_cls = SpatialExpert
            kwargs['expand_ratio'] = expert_expand_ratio
        else:
            expert_cls = SimpleExpert
            kwargs['expand_ratio'] = expert_expand_ratio

        for _ in range(num_experts):
            self.experts.append(expert_cls(in_channels, out_channels, **kwargs))

        # 3) Shared expert (Always active)
        self.shared_expert = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

        self._init_weights()
        self.moe_loss_fn = MoELoss(
            balance_loss_coeff=balance_loss_coeff, 
            z_loss_coeff=router_z_loss_coeff, 
            num_experts=num_experts, 
            top_k=top_k
        )
        self.last_routing_snapshot = {}
        
        # Expert dropout: periodically disable experts to prevent uniform routing
        self.expert_dropout_rate = 0.15  # 15% dropout during training
        self.dropout_interval = 100  # Apply every 100 steps

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Robust router init: find the last Conv layer to initialize
        # Keep initial expert probabilities nearly uniform but with enough
        # variance to produce input-dependent routing (std=0.05, was 0.01)
        last_conv = None  # guard: routing may use a non-Conv router
        for m in self.routing.router.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is not None:
            nn.init.normal_(last_conv.weight, mean=0, std=0.05)
            if last_conv.bias is not None:
                nn.init.constant_(last_conv.bias, 0)

    def _update_sparsity(self):
        """Progressive Sparsity Scheduling"""
        if self._training_step < self.warmup_steps:
            progress = self._training_step / self.warmup_steps
            current_k = self.num_experts - progress * (self.num_experts - self.top_k)
            self._current_top_k = max(self.top_k, int(current_k))
        else:
            self._current_top_k = self.top_k

    def forward(self, x):
        B, C, H, W = x.shape

        if self.training and self.progressive_sparsity:
            self._update_sparsity()
            self._training_step += 1
            
        # Use current_top_k for routing
        adaptive_top_k = self._current_top_k if self.training and self.progressive_sparsity else self.top_k

        # 1) Routing (standardized interface) — pass top_k as parameter instead of
        #    mutating self.routing.top_k (thread-safe, ONNX-traceable).
        # loss_dict contains training loss inputs; empty during inference
        routing_weights, routing_indices, loss_dict = self.routing(x, top_k=adaptive_top_k)

        # 2) Shared expert compute (always active)
        shared_out = self.shared_expert(x)
        if not torch.isfinite(shared_out).all():
            raise RuntimeError("OptimizedMOEImproved shared expert output contains NaN/Inf")

        # 3) Sparse expert compute with STOP GRADIENT on routing weights
        # This prevents main task loss from dominating router learning direction.
        # Router should only learn from MoE auxiliary loss (balance + z-loss).
        accumulator_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        expert_output = torch.zeros(B, self.out_channels, H, W, device=x.device, dtype=accumulator_dtype)

        # Expert dropout: randomly disable experts to prevent collapse.
        # Only after warmup so it doesn't fight progressive-sparsity scheduling.
        active_experts = list(range(self.num_experts))
        _step = self._training_step
        ddp_active = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        if self.training and _step >= self.warmup_steps and _step % self.dropout_interval == 0:
            num_drop = max(1, int(self.num_experts * self.expert_dropout_rate))
            # Draw the drop set on a fixed-seed generator keyed by the global
            # step so every DDP rank disables the *same* experts. Without this,
            # ranks would skip different experts, leaving some experts with no
            # gradient on a subset of ranks → DDP "found unused parameters" /
            # gradient-bucket desync. (Previously dropout was disabled entirely
            # under DDP, which silently changed training dynamics vs single-GPU.)
            g = torch.Generator(device='cpu')
            g.manual_seed(_step)
            drop_indices = torch.randperm(self.num_experts, generator=g)[:num_drop].tolist()
            active_experts = [i for i in active_experts if i not in drop_indices]

        indices_flat = routing_indices.view(B, adaptive_top_k)
        weights_flat = routing_weights.view(B, adaptive_top_k)
        if getattr(self, "detach_routing", False):
            weights_flat = weights_flat.detach()

        if torch.onnx.is_in_onnx_export():
            # ONNX tracing cannot capture ``if mask.any()`` skips.
            # Dense path: compute all experts, gather Top-K, weighted-sum.
            all_outs = torch.stack(
                [self.experts[i](x) for i in range(self.num_experts)], dim=1
            )  # [B, E, out_C, H, W]
            for k in range(adaptive_top_k):
                idx_k = indices_flat[:, k]                                        # [B]
                w_k = weights_flat[:, k]                                          # [B]
                idx_exp = idx_k.view(B, 1, 1, 1, 1).expand(B, 1, self.out_channels, H, W)
                selected = torch.gather(all_outs, 1, idx_exp).squeeze(1)          # [B, out_C, H, W]
                expert_output = expert_output + selected * w_k.view(B, 1, 1, 1)
        else:
            for i in active_experts:
                # Find all samples assigned to expert i
                mask = (indices_flat == i)
                if mask.any():
                    batch_idx, k_idx = torch.where(mask)

                    # Select input and compute
                    inp = x[batch_idx]
                    out = self.experts[i](inp)

                    # Select weights and broadcast (no gradient to router)
                    w = weights_flat[batch_idx, k_idx].view(-1, 1, 1, 1)

                    # Accumulate results
                    expert_output.index_add_(0, batch_idx, out.to(expert_output.dtype) * w.to(expert_output.dtype))

        if not torch.isfinite(expert_output).all():
            raise RuntimeError("OptimizedMOEImproved sparse expert aggregation contains NaN/Inf")

        final_output = shared_out.to(expert_output.dtype) + expert_output
        if not torch.isfinite(final_output).all():
            raise RuntimeError("OptimizedMOEImproved final output contains NaN/Inf")
        final_output = final_output.to(x.dtype)
        if not torch.isfinite(final_output).all():
            raise RuntimeError("OptimizedMOEImproved final output overflowed during dtype conversion")

        # Add residual connection if dimensions match (skipped when the outer
        # block owns the residual, see add_residual)
        if self.add_residual and self.in_channels == self.out_channels:
            final_output = final_output + x
            if not torch.isfinite(final_output).all():
                raise RuntimeError("OptimizedMOEImproved residual output contains NaN/Inf")

        # 4) Compute and return Loss during training
        if self.training and loss_dict:
            aux_loss = self.moe_loss_fn(loss_dict['router_probs'], loss_dict['router_logits'],
                                             loss_dict['topk_indices'])
            _registry_set(self, aux_loss)
            _record_moe_snapshot(
                self,
                expert_usage=loss_dict.get('router_probs').detach().mean(dim=0) if isinstance(loss_dict.get('router_probs'), torch.Tensor) else None,
                topk_indices=loss_dict.get('topk_indices'),
                topk_weights=routing_weights,
                router_probs=loss_dict.get('router_probs'),
                aux_loss=aux_loss,
            )
        else:
            pass

        return final_output

    @property
    def aux_loss(self):
        """Retrieve the auxiliary loss from the registry."""
        return _get_moe_aux_loss(self)

    def get_gflops(self, input_shape: Tuple[int, int, int, int]) -> Dict[str, float]:
        """GFLOPs for router + shared expert + top_k sparse experts."""
        B, C, H, W = input_shape
        flops = {}
        flops['router'] = self.routing.compute_flops(input_shape) / 1e9
        flops['shared_expert'] = FlopsUtils.count_conv2d(self.shared_expert, input_shape) / 1e9
        single_expert_flops = self.experts[0].compute_flops((1, C, H, W))
        flops['sparse_experts'] = (single_expert_flops * B * self.top_k) / 1e9
        flops['total_gflops'] = flops['router'] + flops['shared_expert'] + flops['sparse_experts']
        return flops


class ABlockMoE(ABlock):
    """Area-attention block module with MoE-FFN for efficient feature extraction."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1, num_experts=4, top_k=2, expert_type='simple'):
        super().__init__(dim, num_heads, mlp_ratio, area)
        # Replace MLP with MoE
        self.mlp = OptimizedMOEImproved(
            in_channels=dim,
            out_channels=dim,
            num_experts=num_experts,
            top_k=top_k,
            expert_type=expert_type,
            expert_expand_ratio=mlp_ratio,
            progressive_sparsity=True,
            add_residual=False,  # ABlockMoE owns the MLP residual (see forward)
        )

    @staticmethod
    def _diagnostic_setting(name, default):
        """Read legacy ``moe.base`` debug overrides while keeping one class implementation."""
        legacy = sys.modules.get("ultralytics.nn.modules.moe.base")
        return getattr(legacy, name, globals().get(name, default)) if legacy is not None else globals().get(name, default)

    def _check_finite(self, value: torch.Tensor, boundary: str) -> None:
        """Fail at an opt-in residual boundary without changing activations."""
        if not self._diagnostic_setting("_MOE_FINITE_DIAGNOSTICS", False):
            return
        events = getattr(self, "_moe_nonfinite_events", 0)
        max_events = self._diagnostic_setting("_MOE_FINITE_DIAGNOSTIC_MAX_EVENTS", 1)
        if events >= max_events:
            return
        if not bool(torch.isfinite(value).all().item()):
            self._moe_nonfinite_events = events + 1
            raise RuntimeError(
                f"ABlockMoE non-finite tensor at {boundary} "
                f"(shape={tuple(value.shape)}, dtype={value.dtype})"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mirror ABlock semantics: residual around attn, then residual around mlp.
        # The inner MoE has add_residual=False, so the residual is applied here
        # exactly once (no double-add). Diagnostics fail fast without sanitizing.
        self._check_finite(x, "input")
        attn_out = self.attn(x)
        self._check_finite(attn_out, "attention output")
        x = x + attn_out
        self._check_finite(x, "attention residual")
        mlp_out = self.mlp(x)
        self._check_finite(mlp_out, "MoE output")
        output = x + mlp_out
        self._check_finite(output, "MoE residual")
        return output

    @property
    def aux_loss(self):
        """Delegate to the inner MoE MLP."""
        return self.mlp.aux_loss


class A2C2fMoE(A2C2f):
    """Area-Attention C2f module with MoE-FFN."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
        num_experts: int = 4,
        top_k: int = 2,
        expert_type: str = 'simple'
    ):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        c_ = int(c2 * e)
        # Re-initialize self.m with ABlockMoE
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlockMoE(c_, c_ // 32, mlp_ratio, area, num_experts, top_k, expert_type) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    @property
    def aux_loss(self):
        """Sum aux losses from inner ABlockMoE modules (registry is on inner MoE MLPs)."""
        total = _zero_aux_loss_like(self)
        for block_seq in self.m:
            modules = block_seq if hasattr(block_seq, "__iter__") else [block_seq]
            for block in modules:
                if hasattr(block, "aux_loss"):
                    total = total + block.aux_loss
        return total

    def get_gflops(self, input_shape: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Accurate GFLOPs calculation.

        A2C2fMoE has no `routing`/`shared_expert`/`experts` of its own; the MoE
        lives inside each ABlockMoE's `.mlp`. Sum over those sub-blocks.
        """
        total = 0.0
        for block_seq in self.m:
            modules = block_seq if hasattr(block_seq, "__iter__") else [block_seq]
            for block in modules:
                mlp = getattr(block, "mlp", None)
                if mlp is not None and hasattr(mlp, "get_gflops"):
                    sub = mlp.get_gflops(input_shape)
                    total += float(sub.get('total_gflops', 0.0))
        return {'total_gflops': total}

    def __deepcopy__(self, memo):
        return _robust_deepcopy(self, memo)

# ==========================================
# Ultimate MoE modules
# ==========================================

class HyperUltimateMoE(nn.Module):
    """
    HyperUltimateMoE: Integrates channel splitting, fused experts, and smart routing.
    Combines the best of UltimateMoE and HyperFusedMoEv2 for max efficiency.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        split_ratio: float = 0.5,
        num_groups: int = 8,
        use_routing_cache: bool = True,
        capacity_factor: float = 1.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        # Channel Splitting
        self.dynamic_channels = int(in_channels * split_ratio)
        self.static_channels = in_channels - self.dynamic_channels
        self.out_dynamic = int(out_channels * split_ratio)
        self.out_static = out_channels - self.out_dynamic
        
        # Static Path (Optimized with BN)
        self.static_net = nn.Sequential(
            nn.Conv2d(self.static_channels, self.static_channels, 3, 
                     padding=1, groups=self.static_channels, bias=False),
            nn.BatchNorm2d(self.static_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.static_channels, self.out_static, 1, bias=False),
            nn.BatchNorm2d(self.out_static),
            nn.SiLU(inplace=True)
        )
        
        # Ultra-light Routing
        self.routing = UltraLightRouter(
            self.dynamic_channels, num_experts, top_k,
            use_cache=use_routing_cache
        )
        
        # MatMul Fused Experts
        self.fused_experts = MatMulFusedExperts(
            self.dynamic_channels, self.out_dynamic, 
            num_experts, num_groups
        )
        
        # Adaptive Capacity Control
        self.complexity_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dynamic_channels, 1, 1),
            nn.Sigmoid()
        )
        
        # Progressive Sparsity
        self.register_buffer('training_step', torch.tensor(0), persistent=False)
        self.register_buffer('current_top_k', torch.tensor(num_experts))
        self.warmup_steps = 5000
        
        # Adaptive Load Balancing (rev5: GShard-scale coeffs, see controller note)
        self.balance_controller = AdaptiveBalanceController(
            num_experts,
            initial_coeff=1.0,
            final_coeff=0.1,
            decay_steps=50000
        )
        
        # Output Fusion Layer
        self.proj = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn = nn.GroupNorm(get_safe_groups(out_channels, num_groups), out_channels)
        
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal Initialization + Variance Scaling"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.shape[2] == 1 and m.weight.shape[3] == 1:
                    # 1x1 Conv using Orthogonal Initialization
                    # Ensure we don't squeeze batch/channel dims if they are 1
                    # Just squeeze spatial dims
                    w_view = m.weight.view(m.weight.size(0), m.weight.size(1))
                    if w_view.dim() >= 2 and w_view.size(0) > 1 and w_view.size(1) > 1:
                         nn.init.orthogonal_(w_view)
                    else:
                         # Fallback for shapes like [1, C] or [C, 1] where orthogonal might fail or not apply well
                         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Router Small Variance Initialization
        if hasattr(self.routing.router[-1], 'weight'):
            nn.init.normal_(self.routing.router[-1].weight, std=0.05)
    
    def _update_sparsity(self):
        """Progressive Sparsity Scheduling"""
        if self.training_step < self.warmup_steps:
            progress = self.training_step.float() / self.warmup_steps
            current_k = self.num_experts - progress * (self.num_experts - self.top_k)
            self.current_top_k.fill_(max(self.top_k, int(current_k)))
        else:
            self.current_top_k.fill_(self.top_k)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Progressive Sparsity
        if self.training:
            self._update_sparsity()
            self.training_step += 1
        
        # 1. Channel Split
        x_static, x_dynamic = torch.split(
            x, [self.static_channels, self.dynamic_channels], dim=1
        )
        
        # 2. Static Path (Parallel)
        out_static = self.static_net(x_static)
        
        # 3. Capacity selection (no per-forward GPU->CPU sync).
        # top_k is a fixed Python int (progressive sparsity already adjusts
        # current_top_k via a buffer); complexity now scales expert *weights*
        # rather than the discrete top_k, avoiding complexity_score.item() sync
        # that previously stalled the pipeline (esp. on multi-GPU).
        adaptive_top_k = int(self.current_top_k.item()) if self.training else self.top_k
        complexity_scale = self.complexity_estimator(x_dynamic).mean().clamp(0.3, 1.5)
        
        # 4. Routing Decision (Mixed Precision)
        with autocast(enabled=torch.cuda.is_available()):
            routing_weights, routing_indices, routing_stats = self.routing(
                x_dynamic, adaptive_top_k
            )
        routing_weights = routing_weights * complexity_scale
        
        # 5. MatMul Fused Expert Computation
        out_dynamic = self.fused_experts(
            x_dynamic, routing_weights, routing_indices, adaptive_top_k
        )
        
        # 6. Feature Fusion & Residual
        out_concat = torch.cat([out_static, out_dynamic], dim=1)
        out = self.proj(out_concat)
        out = self.bn(out) + x
        
        # 7. Adaptive Load Balancing Loss
        if self.training:
            balance_loss = self.balance_controller(routing_stats, self.training_step)
            _registry_set(self, balance_loss)
        
        return out
    
    @property
    def aux_loss(self):
        return _get_moe_aux_loss(self)
    
    def get_gflops(self, input_shape):
        """Accurate FLOPs Calculation"""
        B, C, H, W = input_shape
        flops = {}
        
        # 1. Static Path
        flops['static_path'] = FlopsUtils.count_conv2d(
            self.static_net, (B, self.static_channels, H, W)
        ) / 1e9
        
        # 2. Router
        flops['router'] = self.routing.compute_flops(
            (B, self.dynamic_channels, H, W)
        ) / 1e9
        
        # 3. Complexity Estimator
        flops['complexity_estimator'] = FlopsUtils.count_conv2d(
            self.complexity_estimator, (B, self.dynamic_channels, H, W)
        ) / 1e9
        
        # 4. MatMul Fused Experts (Consider Top-K Sparsity)
        # Note: MatMul computes all experts, but effectively uses Top-K.
        # Use the group's own compute_flops (attribute is `fused_conv`, not `fused_weight`).
        all_experts_flops = self.fused_experts.compute_flops(
            (B, self.dynamic_channels, H, W)
        )
        # Effective computation = all * (top_k / num_experts)
        flops['fused_experts'] = all_experts_flops / 1e9
        flops['effective_experts'] = all_experts_flops * (self.top_k / self.num_experts) / 1e9
        
        # 5. Projection Layer
        flops['projection'] = FlopsUtils.count_conv2d(
            self.proj, (B, self.out_channels, H, W)
        ) / 1e9
        
        # Total (Using effective computation)
        flops['total_gflops'] = (
            flops['static_path'] + 
            flops['router'] + 
            flops['complexity_estimator'] + 
            flops['effective_experts'] + 
            flops['projection']
        )
        
        return flops
    
    def __deepcopy__(self, memo):
        return _robust_deepcopy(self, memo)


class UltimateOptimizedMoE(nn.Module):
    """
    UltimateOptimizedMoE: Improved version based on HyperUltimateMoE.
    Enhancements: Dynamic temperature, entropy loss, AMP integration, and complexity-based skipping.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        split_ratio: float = 0.5,
        num_groups: int = 8,
        use_routing_cache: bool = True,
        capacity_factor: float = 1.5,
        initial_temperature: float = 2.0,  # New: Dynamic temperature start
        final_temperature: float = 0.5,    # New: Dynamic temperature end
        entropy_coeff: float = 0.01,       # New: Entropy loss coefficient
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.entropy_coeff = entropy_coeff
        
        # Channel Split
        self.dynamic_channels = int(in_channels * split_ratio)
        self.static_channels = in_channels - self.dynamic_channels
        self.out_dynamic = int(out_channels * split_ratio)
        self.out_static = out_channels - self.out_dynamic
        
        # Static Path (BN for speed)
        self.static_net = nn.Sequential(
            nn.Conv2d(self.static_channels, self.static_channels, 3, padding=1, groups=self.static_channels, bias=False),
            nn.BatchNorm2d(self.static_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.static_channels, self.out_static, 1, bias=False),
            nn.BatchNorm2d(self.out_static),
            nn.SiLU(inplace=True)
        )
        
        # Ultra-light Router (Supports cache + dynamic temperature)
        self.routing = UltraLightRouter(self.dynamic_channels, num_experts, top_k, temperature=initial_temperature, use_cache=use_routing_cache)
        
        # Fused Experts (GN for stability)
        self.fused_experts = MatMulFusedExperts(self.dynamic_channels, self.out_dynamic, num_experts, num_groups)
        
        # Complexity Estimator
        self.complexity_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dynamic_channels, 1, 1),
            nn.Sigmoid()
        )
        
        # Progressive Sparsity
        self.register_buffer('training_step', torch.tensor(0), persistent=False)
        self.register_buffer('current_top_k', torch.tensor(num_experts))
        self.warmup_steps = 5000
        
        # Adaptive Balancing (Add Entropy)
        self.balance_controller = AdaptiveBalanceController(num_experts, initial_coeff=1.0, final_coeff=0.1, decay_steps=50000)  # rev5: GShard-scale
        self.balance_controller.entropy_coeff = entropy_coeff  # New: Inject entropy coefficient

        # Trainer-injectable hyperparameter bridges.
        # The engine trainer injects MoE config by checking `balance_loss_coeff` /
        # `router_z_loss_coeff` attributes. Expose them here so YAML/CLI overrides
        # (moe_balance_loss, moe_router_z_loss) propagate into this module.
        self.balance_loss_coeff = self.balance_controller.initial_coeff
        self.router_z_loss_coeff = 0.0
        
        # Output Fusion
        self.proj = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn = nn.GroupNorm(get_safe_groups(out_channels, num_groups), out_channels)
        
        self._init_weights()
    
    def _init_weights(self):
        """Enhanced Initialization: Kaiming + Small Std + Diversity"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Router Small Std + Slight Noise Diversity
        if hasattr(self.routing.router[-1], 'weight'):
            nn.init.normal_(self.routing.router[-1].weight, std=0.05)
            self.routing.router[-1].weight.data += torch.randn_like(self.routing.router[-1].weight.data) * 0.001  # New: Slight noise
    
    def _update_sparsity_and_temperature(self):
        """Progressive Sparsity + Dynamic Temperature"""
        progress = min(1.0, self.training_step.float() / self.warmup_steps)
        # Sparsity
        current_k = self.num_experts - progress * (self.num_experts - self.top_k)
        self.current_top_k.fill_(max(self.top_k, int(current_k)))
        # Temperature
        if not getattr(self.routing, "_external_temperature_schedule", False):
            current_temp = self.initial_temperature * (1 - progress) + self.final_temperature * progress
            # Clamp temperature to avoid division by zero or explosion
            self.routing.temperature = max(current_temp, 0.1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        if self.training:
            self._update_sparsity_and_temperature()
            self.training_step += 1
        
        # Channel Split
        x_static, x_dynamic = torch.split(x, [self.static_channels, self.dynamic_channels], dim=1)
        
        # Complexity Estimation (graph-safe NaN guard, no extra sync).
        # complexity scales expert *weights*; top_k stays a fixed Python int so we
        # avoid the per-forward complexity_score.item() GPU->CPU sync.
        complexity_scale = self.complexity_estimator(x_dynamic).mean()
        complexity_scale = torch.nan_to_num(complexity_scale, nan=1.0, posinf=1.5, neginf=0.3).clamp(0.3, 1.5)
        out_static = self.static_net(x_static)
        
        adaptive_top_k = int(self.current_top_k.item()) if self.training else self.top_k
        
        # Routing (AMP Acceleration - only on CUDA)
        with autocast(enabled=torch.cuda.is_available()):  # New: Mixed Precision
            routing_weights, routing_indices, routing_stats = self.routing(x_dynamic, adaptive_top_k)
        routing_weights = routing_weights * complexity_scale
        
        # Fused Experts
        out_dynamic = self.fused_experts(x_dynamic, routing_weights, routing_indices, adaptive_top_k)
        
        # Fusion + Residual
        out_concat = torch.cat([out_static, out_dynamic], dim=1)
        out = self.proj(out_concat)
        out = self.bn(out) + x
        
        # Balancing Loss (With Entropy)
        if self.training:
            # Sync trainer-injected coefficient into the controller before computing loss
            self.balance_controller.initial_coeff = self.balance_loss_coeff
            balance_loss = self.balance_controller(routing_stats, self.training_step)
            _registry_set(self, balance_loss)
        
        return out
    
    @property
    def aux_loss(self):
        return _get_moe_aux_loss(self)
    
    def get_gflops(self, input_shape):
        B, C, H, W = input_shape
        flops = {}
        
        # Static path — always fully computed in forward(); no skipping mechanism
        # exists, so report the true FLOPs (the previous *0.9 "assume 10% skip"
        # factor was fictitious and understated real cost).
        flops['static_path'] = FlopsUtils.count_conv2d(self.static_net, (B, self.static_channels, H, W)) / 1e9
        
        # Router
        flops['router'] = self.routing.compute_flops((B, self.dynamic_channels, H, W)) / 1e9
        
        # Estimator
        flops['complexity_estimator'] = FlopsUtils.count_conv2d(self.complexity_estimator, (B, self.dynamic_channels, H, W)) / 1e9
        
        # Experts (effective computation)
        all_experts_flops = self.fused_experts.compute_flops((B, self.dynamic_channels, H, W))
        flops['effective_experts'] = all_experts_flops * (self.top_k / self.num_experts) / 1e9
        
        # Projection
        flops['projection'] = FlopsUtils.count_conv2d(self.proj, (B, self.out_channels, H, W)) / 1e9
        
        flops['total_gflops'] = sum(flops.values())
        return flops
    
    def get_efficiency_stats(self, input_shape):
        flops = self.get_gflops(input_shape)
        return {
            'gflops': flops,
            'num_params': sum(p.numel() for p in self.parameters()) / 1e6,
            'last_aux_loss': self.aux_loss.item() if self.training else 0.0,
            'current_temperature': self.routing.temperature,
            'current_top_k': self.current_top_k.item()
        }
    
    def __deepcopy__(self, memo):
        return _robust_deepcopy(self, memo)
        
# ---------------------------------------------------------------------------
# Backward-compatibility aliases
# ---------------------------------------------------------------------------
MOE = ES_MOE
EfficientSpatialRouterMoE = OptimizedMOE
ModularRouterExpertMoE = OptimizedMOEImproved

# Aliases for safe loading
if 'UltraOptimizedMoE' not in globals():
    UltraOptimizedMoE = UltimateOptimizedMoE  # Upgrade to the SOTA implementation

if __name__ == '__main__':
    # 1. Define a demo model
    model = OptimizedMOEImproved(in_channels=64, out_channels=64, num_experts=4, top_k=2)
    model.train()  # enable training mode

    # 2. Create dummy input
    x = torch.randn(2, 64, 32, 32)

    # 3. Forward pass
    output = model(x)

    print(f"Output Shape: {output.shape}")

    # 4. Compute FLOPs
    flops = model.get_gflops((1, 64, 32, 32))
    print(f"Total GFLOPs (Batch=1): {flops['total_gflops']:.4f}")
    print(f"  - Router: {flops['router']:.4f}")
    print(f"  - Shared: {flops['shared_expert']:.4f}")
    print(f"  - Sparse: {flops['sparse_experts']:.4f}")


__all__ = [
    "DualStreamGateRouter",
    "DualStreamGateRouterV2",
    "AdaptiveGateMoE",
    "HyperSplitMoE",
    "HyperFusedMoE",
    "ZeroCostRouter",
    "FusedExpertGroup",
    "LowRankFusedExpertGroup",
    "VisualDetailGate",
    "PyramidContextMixer",
    "FusedAdaptiveGateMoE",
    "HybridAdaptiveGateMoE",
    "HybridAdaptiveGateMoEv2",
    "LowRankHybridAdaptiveGateMoE",
    "RefinedLowRankHybridAdaptiveGateMoE",
    "DetailAwareLowRankHybridAdaptiveGateMoE",
    "ContextRefinedLowRankHybridAdaptiveGateMoE",
    "VisualEnhancedAdaptiveGateMoE",
    "AdaptiveBalanceController",
    "OptimalHybridGateMoE",
    "MultiHeadRouterV3",
    "DiversifiedExpertGroup",
    "CrossPathGate",
    "MultiHeadRouterMoE",
    "DiversifiedExpertMoE",
    "GatedFusionMoE",
    "UltraLightRouter",
    "MatMulFusedExperts",
    "_pool_to_size_mps_safe",
    "_run_visual_hybrid_moe_forward",
    "UltraOptimizedMoE",
    "AdaptiveCapacityMoE",
    "ES_MOE",
    "OptimizedMOE",
    "OptimizedMOEImproved",
    "ABlockMoE",
    "A2C2fMoE",
    "HyperUltimateMoE",
    "UltimateOptimizedMoE",
    "MOE",
    "EfficientSpatialRouterMoE",
    "ModularRouterExpertMoE",
    "autocast",
    "MOE_LOSS_REGISTRY",
    "MOE_SNAPSHOT_INTERVAL",
    "_registry_set",
    "_registry_get",
    "_should_record_snapshot",
    "_zero_aux_loss_like",
    "_detached_zero_like",
    "_get_moe_aux_loss",
    "_flatten_moe_topk",
    "_compute_usage_from_topk",
    "_record_moe_snapshot",
    "_robust_deepcopy"
]
