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

# Cross-submodule imports: integration classes inherit from hybrid.py and advanced.py
from .hybrid import (
    AdaptiveGateMoE,
    HybridAdaptiveGateMoE,
    HybridAdaptiveGateMoEv2,
    OptimalHybridGateMoE,
    AdaptiveBalanceController,
    VisualDetailGate,
    PyramidContextMixer,
    _run_visual_hybrid_moe_forward,
)
from .advanced import (
    ZeroCostRouter,
    FusedExpertGroup,
    LowRankFusedExpertGroup,
    DualStreamGateRouter,
    HyperSplitMoE,
    HyperFusedMoE,
)
from .base import (
    ES_MOE,
    OptimizedMOE,
    OptimizedMOEImproved,
    UltraOptimizedMoE,
    AdaptiveCapacityMoE,
    ABlockMoE,
    A2C2fMoE,
)

# ---- Integration MoE classes (split from modules.py) ----

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
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, reduced, 1, bias=False),
            nn.GroupNorm(get_safe_groups(reduced, 4), reduced),
            nn.SiLU(inplace=True),
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
            nn.SiLU(inplace=True),
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
                nn.SiLU(inplace=True),
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
            output.index_add_(0, batch_indices, expert_out * expert_weight)

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
            nn.SiLU(inplace=True),
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

