# 🐧Please note that this file has been modified by Tencent on 2026/02/07. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Expert modules for Mixture-of-Experts models"""
import torch
import torch.nn as nn
import math
from .utils import FlopsUtils, get_safe_groups


# ==========================================
# Optimized expert modules
# ==========================================
class OptimizedSimpleExpert(nn.Module):
    """Use GroupNorm instead of BatchNorm to improve stability for small batches."""

    def __init__(self, in_channels, out_channels, expand_ratio=2, num_groups=8):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.GroupNorm(get_safe_groups(hidden_dim, num_groups), hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.GroupNorm(get_safe_groups(out_channels, num_groups), out_channels)
        )
        self.hidden_dim = hidden_dim

    def forward(self, x):
        return self.conv(x)

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        flops = FlopsUtils.count_conv2d(self.conv[0], (1, C, H, W))
        flops += FlopsUtils.count_conv2d(self.conv[3], (1, self.hidden_dim, H, W))
        return flops


class FusedGhostExpert(nn.Module):
    """Fused Ghost expert that reduces memory traffic by combining operations."""

    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2, num_groups=8):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        # Use GroupNorm to improve stability
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.GroupNorm(min(num_groups, init_channels), init_channels),
            nn.SiLU(inplace=True)
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, padding=1, groups=init_channels, bias=False),
            nn.GroupNorm(min(num_groups, new_channels), new_channels),
            nn.SiLU(inplace=True)
        )
        self.init_channels = init_channels

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        flops = FlopsUtils.count_conv2d(self.primary_conv[0], (1, C, H, W))
        flops += FlopsUtils.count_conv2d(self.cheap_operation[0], (1, self.init_channels, H, W))
        return flops


class SimpleExpert(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=2, num_groups=8):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        # GroupNorm (not BatchNorm): experts often see only 1 sample after Top-K
        # routing, where BN's running stats / n=1 variance are ill-defined.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.GroupNorm(get_safe_groups(hidden_dim, num_groups), hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.GroupNorm(get_safe_groups(out_channels, num_groups), out_channels)
        )

    def forward(self, x): return self.conv(x)

    def compute_flops(self, input_shape): return FlopsUtils.count_conv2d(self.conv, input_shape)


class SpatialExpert(nn.Module):
    """Expert network with 3x3 spatial convolution, enabling experts to learn spatial patterns."""
    def __init__(self, in_ch, out_ch, expand_ratio=2, num_groups=8):
        super().__init__()
        hid = int(in_ch * expand_ratio)
        # GroupNorm for small-batch / single-sample stability after Top-K routing.
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hid, 1, bias=False),
            nn.GroupNorm(get_safe_groups(hid, num_groups), hid),
            nn.SiLU(inplace=True),
            nn.Conv2d(hid, hid, 3, padding=1, groups=hid, bias=False),  # DW spatial conv
            nn.GroupNorm(get_safe_groups(hid, num_groups), hid),
            nn.SiLU(inplace=True),
            nn.Conv2d(hid, out_ch, 1, bias=False),
            nn.GroupNorm(get_safe_groups(out_ch, num_groups), out_ch),
        )

    def forward(self, x):
        return self.conv(x)

    def compute_flops(self, input_shape):
        return FlopsUtils.count_conv2d(self.conv, input_shape)


class GhostExpert(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2, num_groups=8):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        # GroupNorm for small-batch / single-sample stability after Top-K routing.
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.GroupNorm(get_safe_groups(init_channels, num_groups), init_channels),
            nn.SiLU(inplace=True)
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, padding=1, groups=init_channels, bias=False),
            nn.GroupNorm(get_safe_groups(new_channels, num_groups), new_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)[:, :self.out_channels, :, :]

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        flops = FlopsUtils.count_conv2d(self.primary_conv, input_shape)
        # Compute input shape to cheap op (output of primary conv)
        p_out = self.primary_conv[0].out_channels
        flops += FlopsUtils.count_conv2d(self.cheap_operation, (B, p_out, H, W))
        return flops


class InvertedResidualExpert(nn.Module):
    """
    Highly efficient expert module: Uses Inverted Residual structure (MobileNetV2 style).
    2-3x faster than standard convolution experts, fewer parameters, stronger non-linearity.
    """
    def __init__(self, in_channels, out_channels, expand_ratio=2, kernel_size=3, num_groups=8):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        # GroupNorm for small-batch / single-sample stability after Top-K routing.
        self.conv = nn.Sequential(
            # 1. Pointwise Expand
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.GroupNorm(get_safe_groups(hidden_dim, num_groups), hidden_dim),
            nn.SiLU(inplace=True),
            # 2. Depthwise Spatial
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2, 
                      groups=hidden_dim, bias=False),
            nn.GroupNorm(get_safe_groups(hidden_dim, num_groups), hidden_dim),
            nn.SiLU(inplace=True),
            # 3. Pointwise Project
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.GroupNorm(get_safe_groups(out_channels, num_groups), out_channels)
        )

    def forward(self, x):
        return self.conv(x)

    def compute_flops(self, input_shape):
        return FlopsUtils.count_conv2d(self.conv, input_shape)


class SharedInvertedExpertGroup(nn.Module):
    """
    Efficient expert group with shared inverted-residual feature extraction.

    The expensive expand + depthwise spatial processing is computed once for the
    dynamic branch. Experts specialize through lightweight pointwise projection
    heads, which are dispatched sparsely according to Top-K routing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int,
        expand_ratio: float = 2.0,
        kernel_size: int = 3,
        top_k: int = 2,
        weight_threshold: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.weight_threshold = weight_threshold
        hidden_dim = max(1, int(in_channels * expand_ratio))
        padding = kernel_size // 2

        def _gn(channels: int) -> nn.GroupNorm:
            return nn.GroupNorm(get_safe_groups(channels, 8), channels)

        self.shared_feature = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            _gn(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, groups=hidden_dim, bias=False),
            _gn(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.expert_projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                _gn(out_channels),
            )
            for _ in range(num_experts)
        )

    @staticmethod
    def _flatten_topk(tensor: torch.Tensor, batch: int, top_k: int) -> torch.Tensor:
        return tensor.reshape(batch, -1)[:, :top_k]

    def forward(self, x, routing_weights, routing_indices, top_k: int):
        B, _, H, W = x.shape
        top_k = min(int(top_k), routing_indices.numel() // max(B, 1))
        features = self.shared_feature(x)
        indices = self._flatten_topk(routing_indices, B, top_k).to(torch.long)
        weights = self._flatten_topk(routing_weights, B, top_k)
        valid_mask = weights > self.weight_threshold

        output = x.new_zeros(B, self.out_channels, H, W)

        if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
            # Export tracing cannot capture data-dependent ``torch.unique`` /
            # dynamic-length loops used by the sparse path. Dense path:
            # compute every expert projection, gather Top-K, weighted-sum.
            all_projs = torch.stack(
                [proj(features) for proj in self.expert_projections], dim=1
            )  # [B, E, out_C, H, W]
            for k in range(top_k):
                idx_k = indices[:, k]                                              # [B]
                w_k = weights[:, k] * valid_mask[:, k].to(weights.dtype)           # [B]
                idx_exp = idx_k.view(B, 1, 1, 1, 1).expand(B, 1, self.out_channels, H, W)
                selected = torch.gather(all_projs, 1, idx_exp).squeeze(1)          # [B, out_C, H, W]
                output = output + selected * w_k.view(B, 1, 1, 1)
            return output

        active_experts = torch.unique(indices[valid_mask]).to(torch.long).tolist()
        for expert_idx in active_experts:
            projection = self.expert_projections[expert_idx]
            expert_mask = (indices == expert_idx) & valid_mask

            batch_indices, k_indices = torch.where(expert_mask)
            expert_out = projection(features[batch_indices])
            expert_weight = weights[batch_indices, k_indices].view(-1, 1, 1, 1).to(expert_out.dtype)
            output.index_add_(0, batch_indices, expert_out * expert_weight)

        return output

    def compute_flops(self, input_shape):
        B, _, H, W = input_shape
        flops = FlopsUtils.count_conv2d(self.shared_feature, input_shape)
        hidden_dim = self.shared_feature[0].out_channels
        single_projection = FlopsUtils.count_conv2d(self.expert_projections[0][0], (1, hidden_dim, H, W))
        flops += single_projection * B * min(self.num_experts, self.top_k)
        return flops


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class EfficientExpertGroup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(EfficientExpertGroup, self).__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        # ``self.conv`` is always built in __init__; rebuild only as a legacy
        # checkpoint fallback and never while tracing (would break export).
        if not hasattr(self, "conv"):
            if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
                raise RuntimeError("EfficientExpertGroup.conv missing during export.")
            self.conv = DepthwiseSeparableConv(x.shape[1], x.shape[1], 3, 1).to(x.device, x.dtype)
        return self.conv(x)
