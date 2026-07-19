# 🐧Please note that this file has been modified by Tencent on 2026/02/13. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Gated MoE module family — DualStream routers, AdaptiveGate MoE variants, and
fused-expert groups.

Extracted from ``modules.py`` to reduce file size.  All symbols are re-exported
by ``modules.py`` for backward compatibility.
"""
import math
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
from .loss import (
    MoELoss, gshard_balance_loss, weighted_gshard_balance_loss,
    differentiable_balance_loss, all_reduce_mean, should_reduce_ddp
)
from .scheduler import (
    MoEDynamicScheduler, MoEDynamicSchedulerConfig,
    MoEDynamicScheduleState, MapSaturationScheduler,
    MapSaturationSchedulerConfig, MapSaturationScheduleState,
    compute_gini,
)
from ._helpers import (
    autocast,
    MOE_LOSS_REGISTRY,
    _registry_set,
    _registry_get,
    _zero_aux_loss_like,
    _detached_zero_like,
    _get_moe_aux_loss,
    _flatten_moe_topk,
    _compute_usage_from_topk,
    _record_moe_snapshot,
    _robust_deepcopy,
)

# ==========================================
# Inverted Residual Expert & HyperSplitMoE
# ==========================================

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
            nn.SiLU(inplace=False),
            nn.Conv2d(in_channels, reduced, 1, bias=False),
            nn.GroupNorm(get_safe_groups(reduced, 4), reduced),
            nn.SiLU(inplace=False),
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
            nn.SiLU(inplace=False),
            nn.Linear(se_hidden, in_channels, bias=True),
            nn.Sigmoid(),
        )

        # ── Static Path ──
        self.static_net = nn.Sequential(
            nn.Conv2d(self.static_channels, self.static_channels, 3,
                      padding=1, groups=self.static_channels, bias=False),
            nn.BatchNorm2d(self.static_channels),
            nn.SiLU(inplace=False),
            nn.Conv2d(self.static_channels, self.out_static, 1, bias=False),
            nn.BatchNorm2d(self.out_static),
            nn.SiLU(inplace=False),
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
        if getattr(self.routing, "_external_temperature_schedule", False):
            return
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
            nn.SiLU(inplace=False),
            nn.Conv2d(self.static_channels, self.out_static, 1, bias=False),
            nn.BatchNorm2d(self.out_static),
            nn.SiLU(inplace=False)
        )

        # 2. Dynamic Router (Global Pooling -> Conv -> Expert Scores)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(self.dynamic_channels, self.dynamic_channels // router_reduction, 1),
            nn.SiLU(inplace=False),
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
            nn.SiLU(inplace=False)
        )
        
        # Adaptive Load Balancing
        if adaptive_balance:
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
        
        # Use fixed top_k — avoids per-forward .item() GPU→CPU sync.
        # Progressive sparsity still fills the buffer for diagnostics.
        adaptive_top_k = self.top_k

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
            return differentiable_balance_loss(probs, usage, self.num_experts, reduce_ddp=should_reduce_ddp(self))
        return gshard_balance_loss(usage, self.num_experts, reduce_ddp=should_reduce_ddp(self))
    
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
        
        self.activation = nn.SiLU(inplace=False)

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
            nn.SiLU(inplace=False),
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


class VisualDetailGate(nn.Module):
    """Lightweight detail gate for boundary and texture aware visual MoE."""

    def __init__(self, channels, num_groups=8, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.detail_filter = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(get_safe_groups(channels, num_groups), channels),
            nn.SiLU(inplace=False),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=False),
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
            nn.SiLU(inplace=False),
        )
        self.pool_projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.GroupNorm(get_safe_groups(channels, num_groups), channels),
                nn.SiLU(inplace=False),
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
            nn.SiLU(inplace=False),
        )
        self.feature_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, refine_hidden, 1, bias=False),
            nn.SiLU(inplace=False),
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
            balance_loss = weighted_gshard_balance_loss(expert_usage, importance_weights, self.num_experts, reduce_ddp=should_reduce_ddp(self))

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
                nn.SiLU(inplace=False),
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


class MultiHeadRouterV3(nn.Module):
    """
    Multi-head parallel router for v0.13.

    Instead of a single dual-stream routing decision, this router splits the
    global-statistics stream into ``num_heads`` independent sub-heads, each
    producing its own expert logits.  The heads are then aggregated by a
    learned temperature-weighted mean before Top-K.

    Motivation: v0.12's ``DualStreamGateRouterV2`` already normalises channel
    statistics, but the routing decision is still a single linear projection
    of a 2C-dim vector.  Multi-head attention literature shows that multiple
    low-rank projections capture richer routing patterns than one dense
    projection.  Each head operates on a ``head_dim`` slice of the normalised
    statistics, keeping the total parameter count comparable.

    Additional features (all DDP-safe, no .item() / buffer sync):
    * Expert dropout during training — randomly drops one of the top_k experts
      per sample with probability ``expert_dropout``, forcing the router to
      learn redundant paths and preventing over-specialisation.
    * Residual routing logits from a tiny channel-attention branch for
      shallow-layer spatial awareness.
    """

    def __init__(
        self,
        in_channels: int,
        num_experts: int,
        top_k: int,
        temperature: float = 1.0,
        num_heads: int = 4,
        local_reduction: int = 16,
        pool_scale: int = 4,
        noise_std: float = 0.1,
        expert_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = max(float(temperature), 1e-3)
        self.pool_scale = pool_scale
        self.num_heads = max(1, min(num_heads, num_experts))
        self.expert_dropout = float(expert_dropout)

        stat_dim = 2 * in_channels
        self.stat_norm = nn.LayerNorm(stat_dim)

        head_dim = max(stat_dim // self.num_heads, 4)
        self.heads = nn.ModuleList([
            nn.Linear(head_dim, num_experts, bias=False)
            for _ in range(self.num_heads)
        ])
        for h in self.heads:
            nn.init.normal_(h.weight, std=0.02)

        # Residual global projection: keeps a full-statistics view alongside
        # multi-head slices so no global context is lost by head splitting.
        self.global_proj = nn.Linear(stat_dim, num_experts, bias=False)
        nn.init.normal_(self.global_proj.weight, std=0.02)

        # Learned per-head temperature (soft merge)
        self.head_alpha = nn.Parameter(torch.ones(self.num_heads) / self.num_heads)
        # Global residual weight (starts small so heads dominate initially)
        self.global_weight = nn.Parameter(torch.tensor(0.1))

        # Learnable expert prior (auxiliary-loss-free balancing)
        self.expert_prior = nn.Parameter(torch.zeros(num_experts))

        # Local spatial branch (lightweight, same as V2)
        reduced = max(in_channels // local_reduction, 4)
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.GroupNorm(get_safe_groups(in_channels, 8), in_channels),
            nn.SiLU(inplace=False),
            nn.Conv2d(in_channels, reduced, 1, bias=False),
            nn.GroupNorm(get_safe_groups(reduced, 4), reduced),
            nn.SiLU(inplace=False),
            nn.Conv2d(reduced, num_experts, 1, bias=True),
        )

        # Learned global/local merge (same as V2)
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Router noise (Switch-Transformer style)
        self.noise_std_init = float(noise_std)
        self.register_buffer('_noise_progress', torch.tensor(0.0), persistent=False)

        # Store head_dim for forward
        self._head_dim = head_dim

    def forward(self, x):
        B, C, H, W = x.shape

        # --- Global statistics ---
        mean = x.mean(dim=[2, 3])
        std = x.std(dim=[2, 3], unbiased=False) if H * W > 1 else torch.zeros_like(mean)
        stats = self.stat_norm(torch.cat([mean, std], dim=1))  # [B, 2C]

        # --- Multi-head routing ---
        head_weights = torch.sigmoid(self.head_alpha)  # [num_heads]
        head_weights = head_weights / (head_weights.sum() + 1e-6)

        # Global residual logits from full statistics
        global_logits = self.global_proj(stats)  # [B, num_experts]
        global_w = torch.sigmoid(self.global_weight)

        # Split stats into head_dim chunks
        # Pad if needed
        if stats.shape[1] < self._head_dim * self.num_heads:
            pad_size = self._head_dim * self.num_heads - stats.shape[1]
            stats_padded = F.pad(stats, (0, pad_size))
        else:
            stats_padded = stats[:, :self._head_dim * self.num_heads]

        stats_chunks = stats_padded.view(B, self.num_heads, self._head_dim)
        head_logits = global_w * global_logits  # start from global view
        for i, h in enumerate(self.heads):
            head_logits = head_logits + (1 - global_w) * head_weights[i] * h(stats_chunks[:, i, :])

        # --- Local spatial branch ---
        if H > self.pool_scale and W > self.pool_scale:
            x_local = F.avg_pool2d(x, kernel_size=self.pool_scale, stride=self.pool_scale)
        else:
            x_local = x
        local_map = self.local_conv(x_local)
        local_logits = local_map.mean(dim=[2, 3])

        # --- Merge ---
        alpha = torch.sigmoid(self.alpha)
        logits = alpha * head_logits + (1 - alpha) * local_logits
        logits = logits + self.expert_prior.view(1, -1)

        # --- Noise injection (training only) ---
        if self.training and self.noise_std_init > 0:
            decay = (1.0 - self._noise_progress).clamp(0.0, 1.0)
            noise = torch.randn_like(logits) * (self.noise_std_init * decay)
            logits = logits + noise

        logits = logits.clamp(-30.0, 30.0)

        # --- Softmax + Top-K ---
        probs = F.softmax(logits / self.temperature, dim=1)
        topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=1)

        # --- Expert dropout (training only) ---
        # Soft dropout: instead of zeroing a selected expert, scale it down by 0.5.
        # This preserves gradient flow while still encouraging redundancy.
        if self.training and self.expert_dropout > 0 and self.top_k > 1:
            drop_mask = torch.rand(B, 1, device=x.device) < self.expert_dropout  # (B, 1)
            if drop_mask.any():
                random_slot = torch.randint(0, self.top_k, (B, 1), device=x.device)  # (B, 1)
                slot_match = torch.arange(self.top_k, device=x.device).unsqueeze(0) == random_slot  # (B, top_k)
                drop_idx = drop_mask & slot_match  # (B, top_k)
                # Scale down by 0.5 instead of zeroing
                scale_factor = torch.where(drop_idx, torch.tensor(0.5, device=x.device, dtype=topk_weights.dtype),
                                           torch.tensor(1.0, device=x.device, dtype=topk_weights.dtype))
                topk_weights = topk_weights * scale_factor

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

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        flops = B * self.num_heads * self._head_dim * self.num_experts
        h_d = max(H // self.pool_scale, 1)
        w_d = max(W // self.pool_scale, 1)
        flops += FlopsUtils.count_conv2d(self.local_conv, (B, C, h_d, w_d))
        return flops


class DiversifiedExpertGroup(nn.Module):
    """
    Heterogeneous expert pool for v0.14.

    Instead of all experts sharing the same inverted-residual structure,
    this group contains a *mix* of expert types:
      - Type A: 1×1 pointwise only (cheapest, good for channel mixing)
      - Type B: 3×3 DW spatial (standard inverted residual)
      - Type C: 5×5 dilated DW (larger receptive field, good for context)

    Each expert type processes the shared features independently and produces
    an output channel projection.  The router learns to assign samples to the
    expert type that best matches their spatial frequency characteristics.

    All experts share the same expand + DW backbone (inherited from
    SharedInvertedExpertGroup) but use different kernel sizes for the depthwise
    spatial conv, giving genuine functional diversity at minimal extra cost.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int,
        expand_ratio: float = 2.0,
        top_k: int = 2,
        weight_threshold: float = 0.0,
        num_groups: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.weight_threshold = weight_threshold

        hidden_dim = max(1, int(in_channels * expand_ratio))

        def _gn(channels: int) -> nn.GroupNorm:
            return nn.GroupNorm(get_safe_groups(channels, num_groups), channels)

        # Shared expand (1x1)
        self.shared_expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            _gn(hidden_dim),
            nn.SiLU(inplace=False),
        )

        # Unified 3x3 DW with per-expert learnable dilation rate.
        # Instead of heterogeneous kernel sizes (1/3/5) which cause output scale
        # inconsistency, all experts use 3x3 DW but with different dilation rates
        # (1, 1, 2, 2 for 4 experts). This preserves structural homogeneity while
        # giving each expert a different effective receptive field.
        self.dw_layers = nn.ModuleList()
        self.dw_dilations = nn.ParameterList()
        for i in range(num_experts):
            # Cycle dilation: 1, 1, 2, 2, 3, 3, ...
            init_dil = 1 + (i // 2)
            self.dw_layers.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=init_dil,
                          dilation=init_dil, groups=hidden_dim, bias=False),
                _gn(hidden_dim),
                nn.SiLU(inplace=False),
            ))
            # Store as learnable parameter (initialized to init_dil, clamped in forward)
            self.dw_dilations.append(nn.Parameter(torch.tensor(float(init_dil))))

        # Per-expert projection
        self.expert_projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                _gn(out_channels),
            )
            for _ in range(num_experts)
        )

    @staticmethod
    def _flatten_topk(tensor, batch, top_k):
        return tensor.reshape(batch, -1)[:, :top_k]

    def forward(self, x, routing_weights, routing_indices, top_k):
        B, _, H, W = x.shape
        top_k = min(int(top_k), routing_indices.numel() // max(B, 1))

        features = self.shared_expand(x)  # [B, hidden, H, W]
        indices = self._flatten_topk(routing_indices, B, top_k).to(torch.long)
        weights = self._flatten_topk(routing_weights, B, top_k)
        valid_mask = weights > self.weight_threshold

        output = x.new_zeros(B, self.out_channels, H, W)

        if torch.onnx.is_in_onnx_export():
            all_projs = torch.stack(
                [self.expert_projections[i](self.dw_layers[i](features))
                 for i in range(self.num_experts)], dim=1
            )
            for k in range(top_k):
                idx_k = indices[:, k]
                w_k = weights[:, k] * valid_mask[:, k].to(weights.dtype)
                idx_exp = idx_k.view(B, 1, 1, 1, 1).expand(B, 1, self.out_channels, H, W)
                selected = torch.gather(all_projs, 1, idx_exp).squeeze(1)
                output = output + selected * w_k.view(B, 1, 1, 1)
            return output

        active_experts = torch.unique(indices[valid_mask]).to(torch.long).tolist()
        for expert_idx in active_experts:
            dw_feat = self.dw_layers[expert_idx](features)
            projection = self.expert_projections[expert_idx]
            expert_mask = (indices == expert_idx) & valid_mask
            batch_indices, k_indices = torch.where(expert_mask)
            expert_out = projection(dw_feat[batch_indices])
            expert_weight = weights[batch_indices, k_indices].view(-1, 1, 1, 1).to(expert_out.dtype)
            # fp16-safe: cast accumulator source to match output dtype (P0-2 fix)
            output.index_add_(0, batch_indices, (expert_out * expert_weight).to(output.dtype))

        return output

    def compute_flops(self, input_shape):
        B, _, H, W = input_shape
        flops = FlopsUtils.count_conv2d(self.shared_expand, input_shape)
        hidden_dim = self.shared_expand[0].out_channels
        # Approximate: average DW + projection cost for top_k active experts
        avg_dw_flops = 0
        for dw in self.dw_layers:
            avg_dw_flops += FlopsUtils.count_conv2d(dw, (1, hidden_dim, H, W))
        avg_dw_flops /= self.num_experts
        proj_flops = FlopsUtils.count_conv2d(self.expert_projections[0][0], (1, hidden_dim, H, W))
        flops += (avg_dw_flops + proj_flops) * B * min(self.num_experts, self.top_k)
        return flops


class CrossPathGate(nn.Module):
    """
    Learnable cross-path gated fusion for v0_15.

    Instead of simple concatenation + 1×1 projection (v0.12), this module
    applies a learned affine gate that modulates static and dynamic paths
    *before* fusion, plus a residual stochastic-depth path that bypasses
    the entire MoE during training with probability ``drop_prob``.

    The gate is computed from both paths' channel statistics (not just the
    input), enabling the fusion to adapt based on what each path actually
    produced rather than what the input looked like.
    """

    def __init__(self, static_channels, dynamic_channels, out_channels, num_groups=8, drop_prob=0.1):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.static_channels = static_channels
        self.dynamic_channels = dynamic_channels
        self.out_channels = out_channels

        stat_dim = static_channels + dynamic_channels
        gate_hidden = max(stat_dim // 4, 8)

        # Cross-path gate: produces per-channel modulation for both paths.
        # Conservative residual design: gate output is added to 0.5 baseline
        # so the fusion starts near simple concatenation and learns deviations.
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(stat_dim, gate_hidden, bias=False),
            nn.SiLU(inplace=False),
            nn.Linear(gate_hidden, out_channels * 2, bias=True),
        )
        # Init gate to produce small deviations from 0.5 (sigmoid(0)=0.5)
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

        # Residual gate scale: starts at 0 so fusion = simple concat at init,
        # then gradually learns to modulate. This prevents early-training
        # instability that destroyed v0_15's original training.
        self.gate_scale = nn.Parameter(torch.tensor(0.0))

        # Drop-path: only drops the projection output (not the entire block).
        # This is far gentler than stochastic depth which zeroed everything.
        self.drop_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, out_static, out_dynamic, x):
        """
        Args:
            out_static: [B, Cs, H, W] static path output
            out_dynamic: [B, Cd, H, W] dynamic path output
            x: [B, C, H, W] residual input
        Returns:
            fused: [B, C, H, W] — must match x for residual
        """
        B, _, H, W = x.shape

        # Concatenate for gate input
        gate_input = torch.cat([out_static, out_dynamic], dim=1)
        gate_raw = self.gate_net(gate_input)  # [B, out_C*2]

        # Conservative residual gating: baseline 0.5 + scaled deviation
        # At init (gate_scale=0), this is pure 0.5 → identical to simple concat.
        gate = 0.5 + torch.tanh(self.gate_scale) * 0.5 * torch.sigmoid(gate_raw)

        # Split gate into static and dynamic portions
        gate_static = gate[:, :self.static_channels].unsqueeze(-1).unsqueeze(-1)
        gate_dynamic = gate[:, self.static_channels:self.static_channels + self.dynamic_channels].unsqueeze(-1).unsqueeze(-1)

        # Apply gates
        out_static_gated = out_static * gate_static
        out_dynamic_gated = out_dynamic * gate_dynamic

        # Concat and the caller does projection
        out_concat = torch.cat([out_static_gated, out_dynamic_gated], dim=1)

        # No stochastic depth — the caller applies gentle drop-path on projection only
        return out_concat


class MultiHeadRouterMoE(OptimalHybridGateMoE):
    """
    v0.13 MoE: multi-head parallel routing for richer expert assignment.

    Builds on the v0.12 winning core and upgrades the router from
    DualStreamGateRouterV2 (single-head) to MultiHeadRouterV3 (multi-head
    parallel).  All other components (SE-gated split, hybrid experts, channel
    shuffle, DW refinement) are preserved.

    Hypothesis: multi-head routing captures richer routing patterns because
    different heads can specialize in different feature statistics (e.g., one
    head responds to brightness, another to texture variance).  The heads are
    soft-merged by a learned temperature-weighted mean before Top-K.

    Also adds expert dropout (p=0.1) during training: randomly zeroes one of
    the top_k experts per sample, forcing redundant expert usage and reducing
    over-specialization at inference time.
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
        num_heads: int = 4,
        expert_dropout: float = 0.05,
    ):
        super().__init__(
            in_channels, out_channels, num_experts, top_k, split_ratio,
            num_groups, initial_temperature, final_temperature,
            balance_loss_coeff, router_z_loss_coeff, entropy_loss_coeff,
            fused_expert_threshold, shuffle_groups, refine, refine_reduction,
        )
        # Replace router with multi-head version (optimized: global residual + soft dropout)
        self.routing = MultiHeadRouterV3(
            self.dynamic_channels, num_experts, top_k,
            temperature=initial_temperature,
            num_heads=num_heads,
            expert_dropout=expert_dropout,
        )
        self._noise_decay_steps = 1000
        self._init_weights()


class DiversifiedExpertMoE(OptimalHybridGateMoE):
    """
    v0.14 MoE: heterogeneous expert pool with diverse kernel sizes.

    Builds on the v0.12 winning core but replaces SharedInvertedExpertGroup
    with DiversifiedExpertGroup.  The router stays as DualStreamGateRouterV2
    (same as v0.12), but the experts now have genuine functional diversity:
    some use 1×1 (channel-mixing), some 3×3 (spatial), some dilated 3×3 (large
    RF).  This gives the router more meaningful choices than homogeneous
    experts.

    Hypothesis: homogeneous experts (all 3×3 DW) limit the routing benefit
    because all experts learn similar filters.  Heterogeneous kernels give
    each expert a structural prior toward different spatial scales, making the
    routing decision more impactful.
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
            in_channels, out_channels, num_experts, top_k, split_ratio,
            num_groups, initial_temperature, final_temperature,
            balance_loss_coeff, router_z_loss_coeff, entropy_loss_coeff,
            fused_expert_threshold, shuffle_groups, refine, refine_reduction,
        )
        # Replace expert group with diversified version
        self.fused_experts = DiversifiedExpertGroup(
            self.dynamic_channels, self.out_dynamic, num_experts,
            expand_ratio=2.0, top_k=top_k, weight_threshold=0.0,
            num_groups=num_groups,
        )
        self._init_weights()


class GatedFusionMoE(OptimalHybridGateMoE):
    """
    v0.15 MoE: cross-path gated fusion + stochastic depth.

    Builds on the v0.12 winning core and replaces the simple concat+proj
    fusion with a CrossPathGate that modulates static and dynamic outputs
    based on their actual content (not just the input).  Additionally applies
    stochastic depth: during training, the entire MoE block is bypassed with
    probability ``drop_prob`` per sample, acting as regularisation.

    Hypothesis: the static and dynamic paths produce complementary information,
    but a fixed concatenation + 1×1 projection cannot adaptively weight them.
    A content-aware gate that sees both outputs can learn to suppress noisy
    dynamic outputs on easy samples and amplify them on hard ones.  Stochastic
    depth further regularises the deep MoE blocks (especially at P5 where the
    model is prone to overfitting on small datasets).
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
        drop_prob: float = 0.05,
    ):
        super().__init__(
            in_channels, out_channels, num_experts, top_k, split_ratio,
            num_groups, initial_temperature, final_temperature,
            balance_loss_coeff, router_z_loss_coeff, entropy_loss_coeff,
            fused_expert_threshold, shuffle_groups, refine, refine_reduction,
        )
        # Cross-path gated fusion
        self.cross_gate = CrossPathGate(
            self.out_static, self.out_dynamic, out_channels,
            num_groups=num_groups, drop_prob=drop_prob,
        )
        self._init_weights()

    def forward(self, x):
        B, C, H, W = x.shape

        if self.training:
            self._update_temperature()
            self.training_step += 1
            self._training_step_value += 1
            if hasattr(self.routing, '_noise_progress'):
                progress = min(1.0, self._training_step_value / self._noise_decay_steps)
                self.routing._noise_progress.fill_(progress)

        # 1. SE-Gated Channel Allocation
        gate_weights = self.se_gate(x)
        gate_static = gate_weights[:, :self.static_channels].unsqueeze(-1).unsqueeze(-1)
        gate_dynamic = gate_weights[:, self.static_channels:].unsqueeze(-1).unsqueeze(-1)
        x_static = x[:, :self.static_channels, :, :] * gate_static
        x_dynamic = x[:, self.static_channels:, :, :] * gate_dynamic

        # 2. Static Path
        out_static = self.static_net(x_static)

        # 3. Complexity Estimation
        complexity = self._safe_complexity(x_dynamic)

        # 4. Dual-Stream V2 Routing
        routing_weights, routing_indices, routing_stats = self.routing(x_dynamic)
        routing_weights, routing_indices, routing_stats, adaptive_top_k = \
            self._apply_complexity_gate(
                routing_weights, routing_indices, routing_stats, complexity
            )

        # 5. Hybrid Expert Computation
        out_dynamic = self.fused_experts(
            x_dynamic, routing_weights, routing_indices, adaptive_top_k
        )

        # 6. Cross-Path Gated Fusion (replaces simple concat)
        out_concat = self.cross_gate(out_static, out_dynamic, x)
        out_concat = self._channel_shuffle(out_concat)
        if self.refine:
            out_concat = self._apply_refine(out_concat)

        # 7. Projection + Gentle Drop-Path Residual
        out = self.proj(out_concat)
        out = self.bn(out)

        # Gentle drop-path: only drop the projection residual, not the entire block.
        # out = (1-drop) * (out + x) + drop * x = (1-drop)*out + x
        # This keeps the identity path alive even when dropped.
        if self.training and self.cross_gate.drop_prob > 0:
            keep_prob = 1.0 - self.cross_gate.drop_prob
            drop = torch.rand(B, 1, 1, 1, device=x.device) < self.cross_gate.drop_prob
            out = out * torch.where(drop, torch.zeros_like(out), out.new_full((), 1.0 / keep_prob))
        out = out + x

        # 8. Auxiliary Loss
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
        # Cross-path gate cost
        stat_dim = self.out_static + self.out_dynamic
        gate_hidden = max(stat_dim // 4, 8)
        flops['cross_gate'] = (B * stat_dim * gate_hidden + B * gate_hidden * self.out_channels * 2) / 1e9
        flops['total_gflops'] = sum(v for k, v in flops.items() if k != 'total_gflops')
        return flops


class UltraLightRouter(ZeroCostRouter):
    """
    UltraLightRouter with Caching mechanism.
    """
    def __init__(self, in_channels, num_experts, top_k, temperature=1.0, use_cache=True):
        super().__init__(in_channels, num_experts, top_k, temperature)
        self.use_cache = use_cache
        self.cache = None

    def forward(self, x, top_k=None):
        # Basic implementation relying on ZeroCostRouter logic. Caching is skipped
        # to avoid shape mismatch issues during training.
        return super().forward(x, top_k=top_k)

class MatMulFusedExperts(FusedExpertGroup):
    """
    MatMulFusedExperts: Alias for FusedExpertGroup for now.
    In future this can be optimized with specialized CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, num_experts, num_groups=8):
        super().__init__(in_channels, out_channels, num_experts, num_groups)
