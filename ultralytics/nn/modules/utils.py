# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import copy
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import uniform_

__all__ = "inverse_sigmoid", "multi_scale_deformable_attn_pytorch", "get_safe_groups", "robust_deepcopy"


# ---------------------------------------------------------------------------
# MPS-safe grid_sample: MPS 不支持 grid_sample 反向传播。
# 前向用 MPS 原生基本算子手动实现 bilinear sampling，
# 反向由 PyTorch autograd 自动推导（全部为 MPS 兼容操作）。
# ---------------------------------------------------------------------------
def _mps_bilinear_sample(input, grid):
    """在 MPS 上用基本算子实现的 bilinear grid_sample (padding_mode='zeros')。

    完全避免 ``F.grid_sample``，只用 gather/index/arithmetic 等 MPS 原生算子。
    autograd 自动推导反向，无需手动实现 backward。

    关键设计：四个角点索引各自独立计算和 clamp，不允许从 clamp 后的值链式推导，
    否则会引入上一轮的 clamp 偏差（如 x0=-1→clamp→0，x1=x0+1=1，但真实角点是 0）。
    """
    N, C, H_in, W_in = input.shape
    _, H_out, W_out, _ = grid.shape
    device = input.device

    # 1. 坐标映射 (align_corners=False): px = ((gx+1)*W - 1)/2
    gx, gy = grid[..., 0], grid[..., 1]
    px = ((gx + 1.0) * W_in - 1.0) / 2.0  # [N, H_out, W_out]
    py = ((gy + 1.0) * H_in - 1.0) / 2.0

    # 2. 四个角点的原始 (float) 坐标 — 各自独立，不链式推导
    x0_f = px.floor()
    y0_f = py.floor()
    x1_f = x0_f + 1.0
    y1_f = y0_f + 1.0

    # 3. 每个角点独立 clamp 为安全索引
    x0_s = x0_f.long().clamp(0, W_in - 1)
    y0_s = y0_f.long().clamp(0, H_in - 1)
    x1_s = x1_f.long().clamp(0, W_in - 1)
    y1_s = y1_f.long().clamp(0, H_in - 1)

    # 4. 每个角点的越界 mask (padding_mode='zeros')
    m00 = ((x0_f >= 0) & (x0_f < W_in) & (y0_f >= 0) & (y0_f < H_in)).float().unsqueeze(1)
    m01 = ((x0_f >= 0) & (x0_f < W_in) & (y1_f >= 0) & (y1_f < H_in)).float().unsqueeze(1)
    m10 = ((x1_f >= 0) & (x1_f < W_in) & (y0_f >= 0) & (y0_f < H_in)).float().unsqueeze(1)
    m11 = ((x1_f >= 0) & (x1_f < W_in) & (y1_f >= 0) & (y1_f < H_in)).float().unsqueeze(1)

    # 5. 双线性权重 (dx/dy 通过 px/py 连接到 grid，保持可微)
    dx = px - x0_f
    dy = py - y0_f

    # 6. 构建索引网格
    n_idx = torch.arange(N, device=device).view(N, 1, 1, 1).expand(N, C, H_out, W_out)
    c_idx = torch.arange(C, device=device).view(1, C, 1, 1).expand(N, C, H_out, W_out)

    # 7. 采样 + mask
    v00 = input[n_idx, c_idx, y0_s.unsqueeze(1).expand(N, C, H_out, W_out),
               x0_s.unsqueeze(1).expand(N, C, H_out, W_out)] * m00
    v01 = input[n_idx, c_idx, y1_s.unsqueeze(1).expand(N, C, H_out, W_out),
               x0_s.unsqueeze(1).expand(N, C, H_out, W_out)] * m01
    v10 = input[n_idx, c_idx, y0_s.unsqueeze(1).expand(N, C, H_out, W_out),
               x1_s.unsqueeze(1).expand(N, C, H_out, W_out)] * m10
    v11 = input[n_idx, c_idx, y1_s.unsqueeze(1).expand(N, C, H_out, W_out),
               x1_s.unsqueeze(1).expand(N, C, H_out, W_out)] * m11

    # 8. 双线性插值
    wx0 = 1.0 - dx.unsqueeze(1)
    wy0 = 1.0 - dy.unsqueeze(1)
    wx1 = dx.unsqueeze(1)
    wy1 = dy.unsqueeze(1)

    output = wx0 * wy0 * v00 + wx0 * wy1 * v01 + wx1 * wy0 * v10 + wx1 * wy1 * v11

    return output


def _grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False):
    """MPS 兼容的 grid_sample。

    MPS 上用 _mps_bilinear_sample 完全替代 F.grid_sample，
    全部由基本算子实现，autograd 自动处理反向传播。
    """
    if input.device.type == "mps":
        return _mps_bilinear_sample(input, grid)
    return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
# ---------------------------------------------------------------------------


def get_safe_groups(channels: int, desired_groups: int = 8) -> int:
    """Return the largest ``num_groups`` <= ``desired_groups`` that evenly divides ``channels``."""
    if channels <= 0:
        return 1
    groups = min(desired_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return max(1, groups)


def robust_deepcopy(obj, memo):
    """Deep-copy a module while dropping transient graph tensors and stale property shadows."""

    def is_readonly_property(cls, name):
        return any(
            isinstance(base.__dict__.get(name), property) and base.__dict__[name].fset is None
            for base in cls.__mro__
        )

    def detached_zero(value):
        return value.detach().new_zeros(()) if isinstance(value, torch.Tensor) else torch.tensor(0.0)

    cls = obj.__class__
    new_obj = cls.__new__(cls)
    memo[id(obj)] = new_obj
    for name, value in obj.__dict__.items():
        if is_readonly_property(cls, name):
            continue
        if isinstance(value, torch.Tensor) and value.grad_fn is not None:
            setattr(new_obj, name, detached_zero(value))
            continue
        try:
            setattr(new_obj, name, copy.deepcopy(value, memo))
        except RuntimeError as exc:
            if "Only Tensors created explicitly" not in str(exc):
                raise
            logging.getLogger("ultralytics").warning(
                "Skipped deepcopy for attribute '%s' in %s due to a non-leaf tensor", name, cls.__name__
            )
            setattr(new_obj, name, detached_zero(value))
        except Exception:
            try:
                setattr(new_obj, name, value)
            except AttributeError:
                pass
    return new_obj


def _get_clones(module, n):
    """Create a list of cloned modules from the given module.

    Args:
        module (nn.Module): The module to be cloned.
        n (int): Number of clones to create.

    Returns:
        (nn.ModuleList): A ModuleList containing n clones of the input module.

    Examples:
        >>> import torch.nn as nn
        >>> layer = nn.Linear(10, 10)
        >>> clones = _get_clones(layer, 3)
        >>> len(clones)
        3
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def bias_init_with_prob(prior_prob=0.01):
    """Initialize conv/fc bias value according to a given probability value.

    This function calculates the bias initialization value based on a prior probability using the inverse sigmoid
    (logit)
    function. It's commonly used in object detection models to initialize classification layers with a specific positive
    prediction probability.

    Args:
        prior_prob (float, optional): Prior probability for bias initialization.

    Returns:
        (float): Bias initialization value calculated from the prior probability.

    Examples:
        >>> bias = bias_init_with_prob(0.01)
        >>> print(f"Bias initialization value: {bias:.4f}")
        Bias initialization value: -4.5951
    """
    return float(-np.log((1 - prior_prob) / prior_prob))  # return bias_init


def linear_init(module):
    """Initialize the weights and biases of a linear module.

    This function initializes the weights of a linear module using a uniform distribution within bounds calculated from
    the output dimension. If the module has a bias, it is also initialized.

    Args:
        module (nn.Module): Linear module to initialize.

    Examples:
        >>> import torch.nn as nn
        >>> linear = nn.Linear(10, 5)
        >>> linear_init(linear)
    """
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)
    if hasattr(module, "bias") and module.bias is not None:
        uniform_(module.bias, -bound, bound)


def inverse_sigmoid(x, eps=1e-5):
    """Calculate the inverse sigmoid function for a tensor.

    This function applies the inverse of the sigmoid function to a tensor, which is useful in various neural network
    operations, particularly in attention mechanisms and coordinate transformations.

    Args:
        x (torch.Tensor): Input tensor with values in range [0, 1].
        eps (float, optional): Small epsilon value to prevent numerical instability.

    Returns:
        (torch.Tensor): Tensor after applying the inverse sigmoid function.

    Examples:
        >>> x = torch.tensor([0.2, 0.5, 0.8])
        >>> inverse_sigmoid(x)
        tensor([-1.3863,  0.0000,  1.3863])
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: list,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Implement multi-scale deformable attention in PyTorch.

    Folds the (num_levels, num_points) axes into a single num_total_points axis so every traced tensor stays at rank <=
    5, the maximum rank supported by CoreML's MIL converter. Numerically equivalent to the rank-6 reference
    implementation on CUDA and CPU.

    Args:
        value (torch.Tensor): Value tensor with shape (bs, num_keys, num_heads, embed_dims).
        value_spatial_shapes (list): Per-level spatial shapes as [(H_0, W_0), ..., (H_{L-1}, W_{L-1})].
        sampling_locations (torch.Tensor): Sampling locations with shape (bs, num_queries, num_heads, num_levels *
            num_points, 2).
        attention_weights (torch.Tensor): Attention weights with shape (bs, num_queries, num_heads, num_levels *
            num_points).

    Returns:
        (torch.Tensor): Output tensor with shape (bs, num_queries, num_heads * embed_dims).

    References:
        https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_total_points, _ = sampling_locations.shape
    num_points = num_total_points // len(value_spatial_shapes)

    # (bs, num_keys, num_heads, embed_dims) -> tuple of (bs*num_heads, embed_dims, H*W) per level
    value_list = value.permute(0, 2, 3, 1).flatten(0, 1).split([h * w for h, w in value_spatial_shapes], dim=-1)
    # Map to grid_sample coords in [-1, 1] and split per level: tuple of (bs*num_heads, num_queries, num_points, 2)
    sampling_grids = (2 * sampling_locations - 1).permute(0, 2, 1, 3, 4).flatten(0, 1).split(num_points, dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value_list[level].reshape(bs * num_heads, embed_dims, h, w)
        sampling_value_list.append(
            _grid_sample(value_l, sampling_grids[level], mode="bilinear", padding_mode="zeros", align_corners=False)
        )
    attention_weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * num_heads, 1, num_queries, num_total_points)
    output = (
        (torch.cat(sampling_value_list, dim=-1) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()
