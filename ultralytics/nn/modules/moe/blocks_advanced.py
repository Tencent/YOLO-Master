# 🐧 YOLO-Master — Advanced MoE Blocks
# Copyright (C) 2026 Tencent. All rights reserved.
"""Advanced MoE block architectures.

Split from ``advanced.py`` for maintainability. Contains the three
high-level MoE blocks that combine routers, expert groups, and training
stabilization:

- ``AdaptiveGateMoE`` — dual-stream gated routing + SE-gated split
- ``HyperSplitMoE`` — channel-split static/dynamic paths
- ``HyperFusedMoE`` — zero-cost routing + fused experts + adaptive balance
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

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
from .utils import FlopsUtils, get_safe_groups, BatchedExpertComputation
from .experts import (
    OptimizedSimpleExpert, FusedGhostExpert, SimpleExpert, GhostExpert,
    InvertedResidualExpert, EfficientExpertGroup, SpatialExpert, SharedInvertedExpertGroup,
)
from .routers import (
    UltraEfficientRouter, EfficientSpatialRouter, LocalRoutingLayer,
    AdaptiveRoutingLayer, DynamicRoutingLayer, AdvancedRoutingLayer,
)
from .routers_advanced import (
    DualStreamGateRouter,
    DualStreamGateRouterV2,
    ZeroCostRouter,
)
from .experts_advanced import (
    FusedExpertGroup,
    LowRankFusedExpertGroup,
)
from .loss import MoELoss, gshard_balance_loss, weighted_gshard_balance_loss, differentiable_balance_loss, all_reduce_mean
from ultralytics.nn.modules.block import ABlock, A2C2f, C3k

__all__ = (
    "AdaptiveGateMoE",
    "HyperSplitMoE",
    "HyperFusedMoE",
)


class AdaptiveGateMoE(nn.Module):
    """
    AdaptiveGateMoE (v0.4): Dual-stream gated routing + SE-gated split +
    stabilized training.

    Key innovations over v0.3 (UltimateOptimizedMoE):
    ──────────────────────────────────────────────────
    1. DualStreamGateRouter: merges global-statistics stream (near-zero cost)
       with lightweight spatial stream for richer routing decisions.
    2. SE-Gated Split: Squeeze-and-Excitation block learns the optimal
       channel allocation ratio between static/dynamic paths (instead of
       fixed 0.5 split).
    3. StableComplexityEstimator: clamped + smoothed complexity scoring
       that eliminates NaN hazards from v0.3.
    4. Warmup-free training: removed progressive-sparsity warmup that
       conflicted with short training schedules (coco128 lesson).
    5. Direct MoELoss integration: uses the production-grade MoELoss with
       soft balancing, z-loss, and entropy regularization.
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
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_loss_coeff = balance_loss_coeff
        self.router_z_loss_coeff = router_z_loss_coeff
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature

        # ── SE-Gated Split ──
        # Instead of a fixed split, SE learns a soft allocation.
        # We still define nominal splits for structural allocation.
        self.nominal_split_ratio = split_ratio
        self.dynamic_channels = int(in_channels * split_ratio)
        self.static_channels = in_channels - self.dynamic_channels
        self.out_dynamic = int(out_channels * split_ratio)
        self.out_static = out_channels - self.out_dynamic

        # SE gate: decides how much of each channel goes dynamic vs static
        se_hidden = max(in_channels // 4, 4)
        self.se_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, se_hidden, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(se_hidden, in_channels, bias=True),
            nn.Sigmoid(),
        )

        # ── Static Path ──
        self.static_net = nn.Sequential(
            nn.Conv2d(self.static_channels, self.static_channels, 3,
                      padding=1, groups=self.static_channels, bias=False),
            nn.BatchNorm2d(self.static_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.static_channels, self.out_static, 1, bias=False),
            nn.BatchNorm2d(self.out_static),
            nn.SiLU(inplace=True),
        )

        # ── Dual-Stream Gate Router ──
        self.routing = DualStreamGateRouter(
            self.dynamic_channels, num_experts, top_k,
            temperature=initial_temperature,
        )

        # Shared feature extraction keeps inverted-residual spatial processing
        # cheap while preserving sparse expert-specific projections.
        self.fused_experts = SharedInvertedExpertGroup(
            self.dynamic_channels, self.out_dynamic, num_experts, top_k=top_k, weight_threshold=0.0
        )

        # ── Stable Complexity Estimator ──
        self.complexity_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dynamic_channels, 1, 1),
            nn.Sigmoid(),
        )

        # ── MoE Loss ──
        self.moe_loss_fn = MoELoss(
            balance_loss_coeff=balance_loss_coeff,
            z_loss_coeff=router_z_loss_coeff,
            entropy_loss_coeff=entropy_loss_coeff,
            num_experts=num_experts,
            top_k=top_k,
            use_soft_balancing=True,
        )

        # ── Output Fusion ──
        self.proj = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn = nn.GroupNorm(get_safe_groups(out_channels, num_groups), out_channels)

        # ── Training state ──
        self.register_buffer('training_step', torch.tensor(0), persistent=False)
        self._training_step_value = 0

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Router: small std for initially near-uniform routing
        if hasattr(self.routing, 'global_fc') and self.routing.global_fc is not None:
            nn.init.normal_(self.routing.global_fc.weight, std=0.05)

    def _safe_complexity(self, x_dynamic):
        """Compute complexity score with NaN/Inf protection."""
        raw = self.complexity_estimator(x_dynamic).mean()
        if torch.isnan(raw) or torch.isinf(raw):
            return torch.tensor(1.0, device=raw.device, dtype=raw.dtype)
        # Smooth clamping: keep in [0.3, 1.5] to avoid degenerate top_k
        return raw.clamp(0.3, 1.5)

    def _apply_complexity_gate(self, routing_weights, routing_indices, routing_stats, complexity):
        """Apply complexity-aware Top-K masking without CPU synchronization.

        Older versions converted the scalar complexity score to Python with
        `.item()` and then sliced Top-K tensors. That forces GPU/MPS sync and
        creates dynamic tensor shapes. Keeping the full Top-K shape while
        zeroing low-rank weights preserves the adaptive behavior with a much
        friendlier execution path.
        """
        top_k = routing_weights.shape[1]
        if top_k <= 1:
            return routing_weights, routing_indices, routing_stats, top_k

        safe_complexity = torch.nan_to_num(complexity, nan=1.0, posinf=1.0, neginf=1.0).clamp(0.3, 1.5)
        keep_count = torch.round(safe_complexity * top_k).clamp(1, top_k)
        expert_rank = torch.arange(1, top_k + 1, device=routing_weights.device, dtype=keep_count.dtype)
        mask = (expert_rank.view(1, top_k, 1, 1) <= keep_count).to(routing_weights.dtype)

        routing_weights = routing_weights * mask
        routing_weights = routing_weights / routing_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        if self.training and isinstance(routing_stats, dict):
            flat_indices = routing_indices.view(routing_indices.shape[0], top_k).to(torch.long)
            flat_weights = routing_weights.view(routing_weights.shape[0], top_k)
            usage = F.one_hot(flat_indices, num_classes=self.num_experts).to(flat_weights.dtype)
            usage = (usage * flat_weights.unsqueeze(-1)).sum(dim=(0, 1))
            routing_stats['expert_usage'] = usage / usage.sum().clamp_min(1e-6)
            routing_stats['effective_top_k'] = keep_count.detach()

        return routing_weights, routing_indices, routing_stats, top_k

    def _update_temperature(self):
        """Cosine annealing of router temperature over training."""
        # Short schedule: anneal over 2000 steps (not 5000)
        anneal_steps = 2000
        progress = min(1.0, self._training_step_value / anneal_steps)
        # Cosine annealing
        cos_val = 0.5 * (1 + math.cos(math.pi * progress))
        current_temp = self.final_temperature + (self.initial_temperature - self.final_temperature) * cos_val
        self.routing.temperature = max(current_temp, 0.1)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.training:
            self._update_temperature()
            self.training_step += 1
            self._training_step_value += 1

        # ── 1. SE-Gated Channel Allocation ──
        gate_weights = self.se_gate(x)                        # [B, C]
        # Separate gate for static and dynamic portions
        gate_static = gate_weights[:, :self.static_channels].unsqueeze(-1).unsqueeze(-1)   # [B, Cs, 1, 1]
        gate_dynamic = gate_weights[:, self.static_channels:].unsqueeze(-1).unsqueeze(-1)  # [B, Cd, 1, 1]

        x_static_raw = x[:, :self.static_channels, :, :]
        x_dynamic_raw = x[:, self.static_channels:, :, :]

        # Apply SE gates
        x_static = x_static_raw * gate_static
        x_dynamic = x_dynamic_raw * gate_dynamic

        # ── 2. Static Path ──
        out_static = self.static_net(x_static)

        # ── 3. Stable Complexity Estimation ──
        complexity = self._safe_complexity(x_dynamic)

        # ── 4. Dual-Stream Routing ──
        routing_weights, routing_indices, routing_stats = self.routing(x_dynamic)
        routing_weights, routing_indices, routing_stats, adaptive_top_k = self._apply_complexity_gate(
            routing_weights, routing_indices, routing_stats, complexity
        )

        # ── 5. Fused Expert Computation ──
        out_dynamic = self.fused_experts(
            x_dynamic, routing_weights, routing_indices, adaptive_top_k
        )

        # ── 6. Feature Fusion + Residual ──
        out_concat = torch.cat([out_static, out_dynamic], dim=1)
        out = self.proj(out_concat)
        out = self.bn(out) + x

        # ── 7. Auxiliary Loss ──
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

    @property
    def aux_loss(self):
        return _get_moe_aux_loss(self)

    def get_gflops(self, input_shape):
        B, C, H, W = input_shape
        flops = {}

        # SE gate
        se_hidden = max(C // 4, 4)
        flops['se_gate'] = (B * C * se_hidden + B * se_hidden * C) * 2 / 1e9

        # Static path
        flops['static_path'] = FlopsUtils.count_conv2d(
            self.static_net, (B, self.static_channels, H, W)) / 1e9

        # Router
        flops['router'] = self.routing.compute_flops(
            (B, self.dynamic_channels, H, W)) / 1e9

        # Complexity estimator
        flops['complexity_estimator'] = FlopsUtils.count_conv2d(
            self.complexity_estimator, (B, self.dynamic_channels, H, W)) / 1e9

        # Fused experts (effective)
        flops['effective_experts'] = self.fused_experts.compute_flops(
            (B, self.dynamic_channels, H, W)) / 1e9

        # Projection
        flops['projection'] = FlopsUtils.count_conv2d(
            self.proj, (B, self.out_channels, H, W)) / 1e9

        flops['total_gflops'] = sum(flops.values())
        return flops

    def get_efficiency_stats(self, input_shape):
        flops = self.get_gflops(input_shape)
        return {
            'gflops': flops,
            'num_params': sum(p.numel() for p in self.parameters()) / 1e6,
            'current_temperature': self.routing.temperature,
            'alpha_gate': torch.sigmoid(self.routing.alpha).item(),
        }

    def __deepcopy__(self, memo):
        return _robust_deepcopy(self, memo)


class HyperSplitMoE(nn.Module):
    """
    HyperSplitMoE: High-performance MoE based on channel splitting.
    Splits input into static (parallel) and dynamic (MoE) paths for speed and accuracy.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        split_ratio: float = 0.5,  # 动态路径占比
        router_reduction: int = 8,
        balance_loss_coeff: float = 1.0,
        router_z_loss_coeff: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_loss_coeff = balance_loss_coeff
        self.router_z_loss_coeff = router_z_loss_coeff

        # Calculate split channels
        self.dynamic_channels = int(in_channels * split_ratio)
        self.static_channels = in_channels - self.dynamic_channels

        # Ensure output channels alignment
        self.out_dynamic = int(out_channels * split_ratio)
        self.out_static = out_channels - self.out_dynamic

        # 1. Static Path - Process basic features with lightweight DW-Conv
        self.static_net = nn.Sequential(
            nn.Conv2d(self.static_channels, self.static_channels, 3, padding=1, groups=self.static_channels, bias=False),
            nn.BatchNorm2d(self.static_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.static_channels, self.out_static, 1, bias=False),
            nn.BatchNorm2d(self.out_static),
            nn.SiLU(inplace=True)
        )

        # 2. Dynamic Router (Global Pooling -> Conv -> Expert Scores)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dynamic_channels, self.dynamic_channels // router_reduction, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.dynamic_channels // router_reduction, num_experts, 1)
        )

        # 3. Expert Group (Inverted Residuals)
        self.experts = nn.ModuleList([
            InvertedResidualExpert(self.dynamic_channels, self.out_dynamic, expand_ratio=2)
            for _ in range(num_experts)
        ])

        # Auxiliary loss function
        self.moe_loss_fn = MoELoss(
            balance_loss_coeff=balance_loss_coeff,
            z_loss_coeff=router_z_loss_coeff,
            num_experts=num_experts,
            top_k=top_k
        )

        # Final fusion layer (1x1 Conv)
        self.proj = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Router initialization: Maintain initial balance
        if hasattr(self.router[-1], 'weight'):
            nn.init.normal_(self.router[-1].weight, std=0.05)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. Channel Split
        x_static, x_dynamic = torch.split(x, [self.static_channels, self.dynamic_channels], dim=1)

        # 2. Static Path Forward (Parallel)
        out_static = self.static_net(x_static)

        # 3. Dynamic Path Forward (MoE)
        # 3.1 Calculate routing logits
        # Sample-level routing: [B, num_experts, 1, 1]
        router_logits = self.router(x_dynamic)

        # 3.2 Top-K Selection
        router_probs = F.softmax(router_logits, dim=1)
        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=1)

        # 3.3 Calculate Load Balancing Loss (Training only)
        if self.training:
            # Record data for loss calculation
            loss_info = {
                'router_probs': router_probs,
                'router_logits': router_logits,
                'topk_indices': topk_indices
            }
            aux_loss = self.moe_loss_fn(router_probs, router_logits, topk_indices)
            _registry_set(self, aux_loss)

        # 3.4 Expert Computation (Batched Sparse Computation)
        # Reuse BatchedExpertComputation for maximum efficiency
        out_dynamic = BatchedExpertComputation.compute_sparse_experts_batched(
            x_dynamic,
            self.experts,
            topk_weights,
            topk_indices,
            self.top_k,
            self.num_experts
        )

        # 4. Feature Concatenation & Fusion
        out_concat = torch.cat([out_static, out_dynamic], dim=1)

        # 5. Channel Shuffle (Optional, enhances information flow) & Projection
        # Mix static and dynamic information (ShuffleNet-like)
        out = self.proj(out_concat)
        out = self.bn(out)

        return out + x  # Residual connection

    @property
    def aux_loss(self):
        return _get_moe_aux_loss(self)

    def __deepcopy__(self, memo):
        return _robust_deepcopy(self, memo)

    def get_gflops(self, input_shape: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Accurate GFLOPs calculation, demonstrating split strategy benefits."""
        B, C, H, W = input_shape
        flops = {}

        # 1. Static Path
        flops['static_path'] = FlopsUtils.count_conv2d(self.static_net, (B, self.static_channels, H, W)) / 1e9

        # 2. Router (Note: input is downsampled)
        flops['router'] = FlopsUtils.count_conv2d(self.router, (B, self.dynamic_channels, H, W)) / 1e9

        # 3. Experts (Top-K only)
        # Calculate single expert FLOPs
        single_expert_flops = self.experts[0].compute_flops((1, self.dynamic_channels, H, W))
        # Total Expert FLOPs = Single * Batch * TopK
        flops['sparse_experts'] = (single_expert_flops * B * self.top_k) / 1e9

        # 4. Projection
        flops['projection'] = FlopsUtils.count_conv2d(self.proj, (B, self.out_channels, H, W)) / 1e9

        flops['total_gflops'] = sum(flops.values())
        return flops


class HyperFusedMoE(nn.Module):
    """
    HyperFusedMoE: Optimizes accuracy and speed using zero-cost routing and fused experts.
    Features: Zero-cost feature reuse, fused kernels, adaptive balancing, and progressive sparsity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        num_groups: int = 8,
        use_zero_cost_routing: bool = True,
        adaptive_balance: bool = True,
        progressive_sparsity: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.adaptive_balance = adaptive_balance
        self.progressive_sparsity = progressive_sparsity

        # Zero-cost Routing or UltraEfficientRouter
        if use_zero_cost_routing:
            self.routing = ZeroCostRouter(in_channels, num_experts, top_k)
        else:
            self.routing = UltraEfficientRouter(in_channels, num_experts, top_k=top_k)

        # Fused Expert Group
        self.fused_experts = FusedExpertGroup(
            in_channels, out_channels, num_experts, num_groups, top_k=top_k
        )

        # Lightweight Shared Path
        self.shared_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False, groups=num_groups),
            nn.GroupNorm(get_safe_groups(out_channels, num_groups), out_channels),
            nn.SiLU(inplace=True)
        )

        # Adaptive Load Balancing
        if adaptive_balance:
            from .hybrid import AdaptiveBalanceController
            self.balance_controller = AdaptiveBalanceController(num_experts)

        # Progressive sparsity control
        self.register_buffer('training_step', torch.tensor(0), persistent=False)
        self.register_buffer('current_top_k', torch.tensor(num_experts))

        self._init_weights()

    def _init_weights(self):
        """Improved initialization strategy"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use variance scaling initialization
                fan_out = m.weight.size(0) * m.weight.size(2) * m.weight.size(3)
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # === Progressive Sparsity Scheduling ===
        if self.training and self.progressive_sparsity:
            self._update_sparsity()

        adaptive_top_k = int(self.current_top_k.item()) if self.training and self.progressive_sparsity else self.top_k

        # === 1. Zero-cost Routing ===
        # routing_weights: [B, k, 1, 1], routing_indices: [B, k, 1, 1]
        routing_weights, routing_indices, routing_stats = self.routing(x, adaptive_top_k)

        # === 2. Shared Path (Parallel Computation) ===
        shared_out = self.shared_path(x)

        # === 3. Fused Expert Computation (Key Optimization) ===
        # Check shapes
        # routing_indices is [B, top_k, 1, 1] from ZeroCostRouter

        expert_out = self.fused_experts(
            x, routing_weights, routing_indices,
            adaptive_top_k
        )

        # === 4. Output Fusion ===
        output = shared_out + expert_out

        # === 5. Adaptive Load Balancing ===
        if self.training:
            if self.adaptive_balance:
                balance_loss = self.balance_controller(
                    routing_stats, self.training_step
                )
            else:
                balance_loss = self._compute_static_balance_loss(routing_stats)

            _registry_set(self, balance_loss)
            self.training_step += 1

        return output

    def _update_sparsity(self):
        """Progressive Sparsity: Use more experts early in training, gradually sparse later."""
        warmup_steps = 5000
        if self.training_step < warmup_steps:
            # Linearly decrease from num_experts to top_k
            progress = self.training_step.float() / warmup_steps
            current_k = self.num_experts - progress * (self.num_experts - self.top_k)
            self.current_top_k.fill_(max(self.top_k, int(current_k)))
        else:
            self.current_top_k.fill_(self.top_k)

    def _compute_static_balance_loss(self, routing_stats):
        """Static load balancing loss (GShard scale).

        Uses differentiable importance (mean router_probs) so the gradient
        actually reaches the router; falls back to the (gradient-free) usage-only
        form only when router_probs is unavailable.
        """
        probs = routing_stats.get('router_probs')
        usage = routing_stats.get('expert_usage')
        if not isinstance(usage, torch.Tensor):
            # Defensive fallback: uniform usage (e.g. empty stats on H*W==1 input)
            dev = probs.device if isinstance(probs, torch.Tensor) else None
            usage = torch.full((self.num_experts,), 1.0 / self.num_experts, device=dev)
        if isinstance(probs, torch.Tensor):
            return differentiable_balance_loss(probs, usage, self.num_experts, reduce_ddp=True)
        return gshard_balance_loss(usage, self.num_experts, reduce_ddp=True)

    @property
    def aux_loss(self):
        return _get_moe_aux_loss(self)

    def __deepcopy__(self, memo):
        return _robust_deepcopy(self, memo)
