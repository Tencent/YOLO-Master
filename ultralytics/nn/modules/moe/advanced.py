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

# ---- Advanced routing / expert MoE classes (split from modules.py) ----

class DualStreamGateRouter(nn.Module):
    """
    Dual-Stream Gate Router for v0.4 AdaptiveGateMoE.

    Combines global context (channel statistics) with local spatial cues
    for richer routing decisions than ZeroCostRouter alone.

    Stream A (Global): AdaptiveAvgPool → FC → expert scores (near-zero cost)
    Stream B (Local):   Light DW-Conv → PW compress → expert scores
    Merge: learned scalar gate α ∈ [0,1] blends the two streams.

    This preserves the near-zero overhead of ZeroCostRouter while adding
    spatial awareness that was previously missing.
    """

    def __init__(self, in_channels, num_experts, top_k, temperature=1.0,
                 local_reduction=16, pool_scale=4):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = max(float(temperature), 1e-3)
        self.pool_scale = pool_scale

        # --- Stream A: Global (channel-statistics) ---
        stat_dim = 2 * in_channels  # mean + std
        self.global_fc = nn.Linear(stat_dim, num_experts, bias=False)
        nn.init.normal_(self.global_fc.weight, std=0.05)

        # --- Stream B: Local (spatial) ---
        reduced = max(in_channels // local_reduction, 4)
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.GroupNorm(get_safe_groups(in_channels, 8), in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, reduced, 1, bias=False),
            nn.GroupNorm(get_safe_groups(reduced, 4), reduced),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced, num_experts, 1, bias=True),
        )

        # --- Merge gate α ---
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B, C, H, W = x.shape

        # Stream A: global statistics
        mean = x.mean(dim=[2, 3])                          # [B, C]
        std = x.std(dim=[2, 3], unbiased=False) if H * W > 1 else torch.zeros_like(mean)
        stats = torch.cat([mean, std], dim=1)               # [B, 2C]
        global_logits = self.global_fc(stats)                # [B, E]

        # Stream B: local spatial cues (with optional downsampling)
        if H > self.pool_scale and W > self.pool_scale:
            x_local = F.avg_pool2d(x, kernel_size=self.pool_scale, stride=self.pool_scale)
        else:
            x_local = x
        local_map = self.local_conv(x_local)                # [B, E, h', w']
        local_logits = local_map.mean(dim=[2, 3])           # [B, E]

        # Merge with learned gate
        alpha = torch.sigmoid(self.alpha)
        logits = alpha * global_logits + (1 - alpha) * local_logits   # [B, E]

        # Numerical stability
        logits = logits.clamp(-30.0, 30.0)

        # Softmax + Top-K
        probs = F.softmax(logits / self.temperature, dim=1)  # [B, E]
        topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=1)
        topk_weights = topk_weights / (topk_weights.sum(dim=1, keepdim=True) + 1e-6)

        # Expand to spatial dims for downstream consumers
        routing_weights = topk_weights.view(B, self.top_k, 1, 1)
        routing_indices = topk_indices.view(B, self.top_k, 1, 1)

        routing_stats = {'topk_indices': topk_indices}
        if self.training:
            expert_usage = torch.zeros(self.num_experts, device=x.device)
            expert_usage.scatter_add_(0, topk_indices.view(-1),
                                      torch.ones_like(topk_indices.view(-1), dtype=torch.float32))
            expert_usage = expert_usage / (B * self.top_k)

            routing_stats = {
                'router_probs': probs,
                'router_logits': logits,
                'topk_indices': topk_indices,
                'expert_usage': expert_usage,
            }

        return routing_weights, routing_indices, routing_stats

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        # Stream A: negligible (linear on 2C)
        flops_a = B * 2 * C * self.num_experts
        # Stream B: DW-Conv + PW + PW
        h_d = max(H // self.pool_scale, 1)
        w_d = max(W // self.pool_scale, 1)
        down_shape = (B, C, h_d, w_d)
        flops_b = FlopsUtils.count_conv2d(self.local_conv, down_shape)
        return flops_a + flops_b


class DualStreamGateRouterV2(DualStreamGateRouter):
    """
    v0.11 router: normalized global statistics + learnable expert prior bias.

    Improvements over ``DualStreamGateRouter`` (used by v0.4-v0.10), each of
    which is cheap, fully differentiable and DDP-safe:

    1. LayerNorm on the concatenated [mean, std] channel statistics before the
       global FC. Raw first/second moments vary widely in scale across layers
       and batches, which makes the global routing logits noisy. Normalizing
       them stabilizes routing and lets the global stream learn faster at
       near-zero extra cost.
    2. Learnable per-expert prior bias added to the fused logits. This is an
       auxiliary-loss-free-style load-balancing prior: it is a plain
       ``nn.Parameter`` whose gradient is all-reduced by DDP automatically, so
       it avoids the usage-based buffer updates that broke v0.3 under DDP. The
       existing balance / z losses shape this prior to counteract expert
       under-use without any host-side synchronization.

    The output interface is identical to ``DualStreamGateRouter``, so this is a
    drop-in replacement for the AdaptiveGate family.
    """

    def __init__(self, in_channels, num_experts, top_k, temperature=1.0,
                 local_reduction=16, pool_scale=4, noise_std=0.1):
        super().__init__(in_channels, num_experts, top_k, temperature,
                         local_reduction, pool_scale)
        # Normalize channel statistics before the global stream FC.
        self.stat_norm = nn.LayerNorm(2 * in_channels)
        # Auxiliary-loss-free style learnable balancing prior (starts neutral).
        self.expert_prior = nn.Parameter(torch.zeros(num_experts))
        # Switch-Transformer-style router noise (training only). Prevents
        # expert collapse by perturbing logits before softmax, encouraging
        # exploration of under-utilized experts. Decays linearly to 0 over
        # the first 50% of training so late-stage routing is noise-free.
        self.noise_std_init = float(noise_std)
        self.register_buffer('_noise_progress', torch.tensor(0.0), persistent=False)

    def forward(self, x):
        B, C, H, W = x.shape

        # Stream A: normalized global statistics
        mean = x.mean(dim=[2, 3])                          # [B, C]
        std = x.std(dim=[2, 3], unbiased=False) if H * W > 1 else torch.zeros_like(mean)
        stats = self.stat_norm(torch.cat([mean, std], dim=1))  # [B, 2C]
        global_logits = self.global_fc(stats)               # [B, E]

        # Stream B: local spatial cues (with optional downsampling)
        if H > self.pool_scale and W > self.pool_scale:
            x_local = F.avg_pool2d(x, kernel_size=self.pool_scale, stride=self.pool_scale)
        else:
            x_local = x
        local_map = self.local_conv(x_local)                # [B, E, h', w']
        local_logits = local_map.mean(dim=[2, 3])           # [B, E]

        # Merge with learned gate + learnable balancing prior
        alpha = torch.sigmoid(self.alpha)
        logits = alpha * global_logits + (1 - alpha) * local_logits
        logits = logits + self.expert_prior.view(1, -1)

        # Switch-Transformer-style noise injection (training only).
        # Decays linearly from noise_std_init to 0 over the first half of
        # training. This is a plain tensor operation — no buffer sync, no
        # .item() — so it is fully DDP-safe and MPS-compatible.
        if self.training and self.noise_std_init > 0:
            decay = (1.0 - self._noise_progress).clamp(0.0, 1.0)
            noise = torch.randn_like(logits) * (self.noise_std_init * decay)
            logits = logits + noise

        # Numerical stability
        logits = logits.clamp(-30.0, 30.0)

        # Softmax + Top-K
        probs = F.softmax(logits / self.temperature, dim=1)
        topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=1)
        topk_weights = topk_weights / (topk_weights.sum(dim=1, keepdim=True) + 1e-6)

        routing_weights = topk_weights.view(B, self.top_k, 1, 1)
        routing_indices = topk_indices.view(B, self.top_k, 1, 1)

        routing_stats = {'topk_indices': topk_indices}
        if self.training:
            expert_usage = torch.zeros(self.num_experts, device=x.device)
            expert_usage.scatter_add_(0, topk_indices.view(-1),
                                      torch.ones_like(topk_indices.view(-1), dtype=torch.float32))
            expert_usage = expert_usage / (B * self.top_k)

            routing_stats = {
                'router_probs': probs,
                'router_logits': logits,
                'topk_indices': topk_indices,
                'expert_usage': expert_usage,
            }

        return routing_weights, routing_indices, routing_stats


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


class ZeroCostRouter(nn.Module):
    """
    Zero-cost Router: Reuses feature map statistics for routing decisions.

    Principles:
    1. Uses global average pooling and standard deviation as routing signals (already computed in BN).
    2. Requires only one 1x1 convolution to map statistics to expert scores.
    3. Reduces FLOPs by over 95%.
    """
    
    def __init__(self, in_channels, num_experts, top_k, temperature=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        
        # Statistics dimension: mean + std = 2 * in_channels
        stat_dim = 2 * in_channels
        
        # Ultra-lightweight mapping network
        self.router = nn.Sequential(
            nn.Linear(stat_dim, num_experts, bias=False),
            nn.Softmax(dim=1)
        )
        
        # Initialize with moderate variance for input-dependent routing
        nn.init.normal_(self.router[0].weight, std=0.05)
    
    def forward(self, x, top_k=None):
        B, C, H, W = x.shape
        current_top_k = max(1, min(int(self.top_k if top_k is None else top_k), self.num_experts))
        
        # === Zero-cost Feature Extraction ===
        # Global statistics (Overlaps with BN computation, near zero cost)
        mean = x.mean(dim=[2, 3])  # [B, C]
        # Use unbiased=False to avoid DoF warning when H*W <= 1 (e.g. classification head)
        std = x.std(dim=[2, 3], unbiased=False) if H * W > 1 else torch.zeros_like(mean)
        stats = torch.cat([mean, std], dim=1)  # [B, 2C]
        
        # === Routing Decision ===
        router_logits = self.router(stats) / self.temperature  # [B, num_experts]
        
        # Clamp logits for stability
        router_logits = router_logits.clamp(-30.0, 30.0)
        
        router_probs = F.softmax(router_logits, dim=1)
        
        # Top-K Selection
        topk_probs, topk_indices = torch.topk(router_probs, current_top_k, dim=1)
        
        # Renormalization
        topk_probs = topk_probs / (topk_probs.sum(dim=1, keepdim=True) + 1e-6)
        
        # Expand to spatial dimensions
        routing_weights = topk_probs.view(B, current_top_k, 1, 1)
        routing_indices = topk_indices.view(B, current_top_k, 1, 1)
        
        # Statistical Information
        expert_usage = torch.zeros(self.num_experts, device=x.device)
        expert_usage.scatter_add_(0, topk_indices.view(-1), 
                                  torch.ones_like(topk_indices.view(-1), dtype=torch.float32))
        expert_usage = expert_usage / (B * current_top_k)
        
        routing_stats = {
            'router_probs': router_probs,
            'router_logits': router_logits,
            'topk_indices': topk_indices,
            'expert_usage': expert_usage
        }
        
        return routing_weights, routing_indices, routing_stats
    
    def compute_flops(self, input_shape):
        """FLOPs calculation"""
        B, C, H, W = input_shape
        # Statistics computation (mean/std): 2 * B * C * H * W
        # Linear layer: B * (2*C) * num_experts
        flops = 2 * B * C * H * W + B * 2 * C * self.num_experts
        return flops


class FusedExpertGroup(nn.Module):
    """
    Fused Expert Group: Reduces memory access via kernel fusion.

    Optimization Strategies:
    1. Merges convolution kernels of multiple experts into a single large convolution.
    2. Uses grouped convolution for expert isolation.
    3. Uses dynamic slicing to extract Top-K expert outputs.
    """
    
    def __init__(self, in_channels, out_channels, num_experts, num_groups=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.out_channels = out_channels
        self.top_k = min(int(top_k), num_experts)
        fused_out_channels = num_experts * out_channels
        conv_groups = min(get_safe_groups(in_channels, num_groups), fused_out_channels)
        while conv_groups > 1 and (in_channels % conv_groups != 0 or fused_out_channels % conv_groups != 0):
            conv_groups -= 1
        self.num_groups = max(1, conv_groups)
        
        # === Fused Convolution: Merged weights of all experts ===
        # Output channels = num_experts * out_channels
        self.fused_conv = nn.Conv2d(
            in_channels,
            fused_out_channels,
            kernel_size=3,
            padding=1,
            groups=self.num_groups,
            bias=False
        )
        
        # Independent normalization affine parameters for each expert. Keeping
        # them as compact tables avoids stacking ModuleList parameters every
        # forward while preserving per-expert scaling.
        self.norm_groups = get_safe_groups(out_channels, num_groups)
        self.norm_eps = 1e-5
        self.expert_norm_weight = nn.Parameter(torch.ones(num_experts, out_channels))
        self.expert_norm_bias = nn.Parameter(torch.zeros(num_experts, out_channels))
        
        self.activation = nn.SiLU(inplace=True)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Map legacy per-expert GroupNorm keys to compact affine tables."""
        weight_key = prefix + "expert_norm_weight"
        bias_key = prefix + "expert_norm_bias"
        legacy_weight_keys = [prefix + f"expert_norms.{i}.weight" for i in range(self.num_experts)]
        legacy_bias_keys = [prefix + f"expert_norms.{i}.bias" for i in range(self.num_experts)]

        if weight_key not in state_dict and all(k in state_dict for k in legacy_weight_keys):
            state_dict[weight_key] = torch.stack([state_dict.pop(k) for k in legacy_weight_keys], dim=0)
        if bias_key not in state_dict and all(k in state_dict for k in legacy_bias_keys):
            state_dict[bias_key] = torch.stack([state_dict.pop(k) for k in legacy_bias_keys], dim=0)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    
    def forward(self, x, routing_weights, routing_indices, top_k):
        B, C, H, W = x.shape
        E, OC = self.num_experts, self.out_channels

        # === 1. Fused Forward Pass (Compute all experts in one convolution) ===
        fused_out = self.fused_conv(x)  # [B, E*OC, H, W]

        # === 2. Reshape to Expert Dimension ===
        fused_out = fused_out.view(B, E, OC, H, W)

        # === 3. Top-K gather FIRST (process only selected experts) ===
        # Gathering before normalization means we only run GroupNorm/activation on
        # top_k experts instead of all E experts -> big saving when E >> top_k.
        idx = routing_indices.view(B, top_k)              # [B, top_k]
        wts = routing_weights.view(B, top_k)              # [B, top_k]
        idx_exp = idx.view(B, top_k, 1, 1, 1).expand(B, top_k, OC, H, W)
        selected = torch.gather(fused_out, 1, idx_exp)    # [B, top_k, OC, H, W]

        # === 4. Vectorized per-expert GroupNorm (no Python loops / mask sync) ===
        w_sel = self.expert_norm_weight[idx].to(fused_out.dtype)  # [B, top_k, OC]
        b_sel = self.expert_norm_bias[idx].to(fused_out.dtype)    # [B, top_k, OC]

        # group_norm over channel dim per (sample, k) instance
        flat = selected.reshape(B * top_k, OC, H, W)
        normed = F.group_norm(flat, self.norm_groups, None, None, self.norm_eps).view(B, top_k, OC, H, W)
        normed = normed * w_sel.view(B, top_k, OC, 1, 1) + b_sel.view(B, top_k, OC, 1, 1)
        normed = self.activation(normed)

        # === 5. Weighted sum over top_k ===
        output = (normed * wts.view(B, top_k, 1, 1, 1)).sum(dim=1)  # [B, OC, H, W]

        return output
    
    def compute_flops(self, input_shape):
        """FLOPs calculation"""
        B, C, H, W = input_shape
        # FLOPs of fused convolution
        flops = FlopsUtils.count_conv2d(self.fused_conv, input_shape)
        # FLOPs of Top-K GroupNorm/activation (approximate)
        flops += B * self.top_k * self.out_channels * H * W * 10
        return flops


class LowRankFusedExpertGroup(nn.Module):
    """
    Low-rank fused expert group for large feature maps.

    It keeps the fused expert execution pattern from `FusedExpertGroup`, but
    first compresses the dynamic branch with a shared 1x1 bottleneck. This
    lowers the cost of the expert 3x3 convolution on P3/P4 while preserving
    per-expert spatial specialization.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_experts,
        num_groups=8,
        top_k=2,
        bottleneck_ratio=0.5,
        min_channels=16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = min(int(top_k), num_experts)
        self.bottleneck_channels = min(
            in_channels,
            max(min_channels, int(round(in_channels * bottleneck_ratio))),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, self.bottleneck_channels, 1, bias=False),
            nn.GroupNorm(get_safe_groups(self.bottleneck_channels, num_groups), self.bottleneck_channels),
            nn.SiLU(inplace=True),
        )
        self.fused = FusedExpertGroup(
            self.bottleneck_channels,
            out_channels,
            num_experts,
            num_groups,
            top_k=top_k,
        )

    def forward(self, x, routing_weights, routing_indices, top_k):
        return self.fused(self.bottleneck(x), routing_weights, routing_indices, top_k)

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        flops = FlopsUtils.count_conv2d(self.bottleneck, input_shape)
        flops += self.fused.compute_flops((B, self.bottleneck_channels, H, W))
        return flops

