# 🐧 YOLO-Master — Advanced Fused Expert Groups
# Copyright (C) 2026 Tencent. All rights reserved.
"""Fused expert architectures for efficient MoE computation.

Split from ``advanced.py`` for maintainability. Contains expert groups that
fuse multiple expert kernels into a single convolution for throughput:

- ``FusedExpertGroup`` — merged conv + per-expert GroupNorm + top-K gather
- ``LowRankFusedExpertGroup`` — bottleneck-compressed variant for large maps
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import FlopsUtils, get_safe_groups

__all__ = (
    "FusedExpertGroup",
    "LowRankFusedExpertGroup",
)


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
