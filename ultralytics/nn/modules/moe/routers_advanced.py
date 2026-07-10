# 🐧 YOLO-Master — Advanced MoE Routers
# Copyright (C) 2026 Tencent. All rights reserved.
"""Advanced router architectures for MoE blocks.

Split from ``advanced.py`` for maintainability. Contains routers that go
beyond the basic top-k gating in ``routers.py``:

- ``DualStreamGateRouter`` — global-statistics + local-spatial dual-stream
- ``DualStreamGateRouterV2`` — adds LayerNorm + learnable prior + noise
- ``ZeroCostRouter`` — reuses feature statistics for near-zero FLOPs routing
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import FlopsUtils, get_safe_groups

__all__ = (
    "DualStreamGateRouter",
    "DualStreamGateRouterV2",
    "ZeroCostRouter",
)


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
