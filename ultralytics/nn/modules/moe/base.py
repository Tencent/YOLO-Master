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

# ---- Base MoE classes (split from modules.py) ----

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
                reduce_ddp=True,
            )

            aux_loss = (self.balance_loss_coeff * balance_loss) + (self.router_z_loss_coeff * z_loss_val)
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
                reduce_ddp=True,
            )
            aux_loss = (self.balance_loss_coeff * balance_loss) + (self.router_z_loss_coeff * z_loss_val)
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
        self.register_buffer('load_balancing_loss', torch.tensor(0.0), persistent=False)
        self.register_buffer('expert_usage_counts', torch.zeros(num_experts), persistent=False)
        self.last_routing_snapshot = {}

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
            final_output.index_add_(0, batch_indices, expert_out * weight)

        return final_output

    def _compute_load_balancing_loss(self, routing_weights, eps=1e-6):
        """Compute load-balancing loss (GShard scale, ~1.0 at balance)."""
        expert_usage = routing_weights.mean(dim=(0, 2, 3))
        # reduce_ddp=True → usage averaged across ranks so all GPUs share one
        # global balance target (matches MoELoss; no-op on single GPU).
        load_balance_loss = gshard_balance_loss(expert_usage, self.num_experts, reduce_ddp=True)

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
                    # Note: convert dtype if mismatched
                    if out.dtype != expert_output.dtype:
                        out = out.to(expert_output.dtype)
                    if w.dtype != expert_output.dtype:
                        w = w.to(expert_output.dtype)

                    expert_output.index_add_(0, batch_idx, out * w)

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

        # 3) Sparse expert compute with STOP GRADIENT on routing weights
        # This prevents main task loss from dominating router learning direction.
        # Router should only learn from MoE auxiliary loss (balance + z-loss).
        expert_output = torch.zeros(B, self.out_channels, H, W, device=x.device, dtype=x.dtype)

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

        # Guard against activation explosion on routing collapse (all tokens -> 1 expert)
        expert_output = expert_output.clamp_(-1e4, 1e4)

        final_output = shared_out + expert_output
        
        # Add residual connection if dimensions match (skipped when the outer
        # block owns the residual, see add_residual)
        if self.add_residual and self.in_channels == self.out_channels:
            final_output = final_output + x

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mirror ABlock semantics: residual around attn, then residual around mlp.
        # The inner MoE has add_residual=False, so the residual is applied here
        # exactly once (no double-add).
        x = x + self.attn(x)
        return x + self.mlp(x)

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
# Inverted Residual Expert & HyperSplitMoE
# ==========================================
