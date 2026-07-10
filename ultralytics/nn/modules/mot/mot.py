# 🐧 YOLO-Master MoT Module — Mixture of Transformers
# Copyright (C) 2026 Tencent. All rights reserved.
"""Mixture-of-Transformers (MoT) for YOLO-Master.

Design philosophy
─────────────────
MoA routes tokens to *different attention heads*;
MoE routes tokens to *different FFN experts*;
**MoT routes tokens to *different complete Transformer architectures*.**

Each "Transformer Expert" is a full Transformer block (Attn + FFN) with a
distinct inductive bias for a specific aspect of visual feature processing:

  Expert 0 — LocalConvTransformer   (Conv-Attn, DW-biased, best for texture/edges)
  Expert 1 — WindowTransformer      (Swin-style window partition, shifted, best for medium objects)
  Expert 2 — DeformableTransformer  (sparse deformable sampling, best for irregular/occluded objects)

A content-aware router (lightweight 1×1 MLP) assigns each spatial token a
*sparse Top-K weight* over these experts. The routing is **soft Top-K**:
all experts are computed every forward and blended by their (Top-K-masked,
renormalised) weights, so non-selected experts contribute 0 to the blend
while the graph stays static and ONNX/TorchScript trace-stable. This is a
deliberate trade-off — true sparse dispatch would save little here because
the three experts have distinct, individually-cheap compute graphs.

Key properties
──────────────
- CNN-native I/O: input [B,C,H,W] → output [B,C,H,W]
- Flash-Attention compatible (PyTorch ≥ 2.0)
- Load-balancing aux loss (z-loss style, optional) for stable expert utilization
- Drop-in for C3k2 / A2C2f via C2fMoT wrapper
- Designed to *complement* MoA (different routing granularity) and MoE (different expert type)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist
from ultralytics.nn.modules.conv import Conv
# P2 fix: remove MoT's compile-time dependency on the MoE package by
# inlining `all_reduce_mean` and `differentiable_balance_loss` locally,
# mirroring the same decoupling already applied in MoA.  This way MoE-
# internal refactors (file renames, signature changes) can no longer
# break MoT imports.  The copies are kept in sync with moe/loss.py.
from ultralytics.nn.modules.utils import get_safe_groups as _safe_groups
from ultralytics.utils import LOGGER


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a tensor across DDP ranks (gradient-preserving, no-op on 1 GPU).

    Local, dependency-free copy of MoE's ``moe.loss.all_reduce_mean`` —
    duplicated here (rather than imported) so MoT keeps zero compile-time
    dependency on the MoE package. Reduces in float32 to avoid precision
    loss when the model runs under fp16/bf16 AMP.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    world = dist.get_world_size()
    if world <= 1:
        return tensor
    orig_dtype = tensor.dtype
    out = tensor.float().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out = out / world
    return out.to(orig_dtype)


def differentiable_balance_loss(
    router_probs: torch.Tensor,
    expert_usage: torch.Tensor,
    num_experts: int,
    target_usage: Optional[torch.Tensor] = None,
    reduce_ddp: bool = False,
) -> torch.Tensor:
    """GShard balance loss with gradient flowing to the router via `importance`.

    Local copy of ``moe.loss.differentiable_balance_loss`` — kept in sync
    with the original. See the MoE version for full documentation.
    """
    probs = router_probs.reshape(
        router_probs.shape[0], router_probs.shape[1], -1
    ).mean(-1) if router_probs.dim() == 4 else router_probs.reshape(-1, num_experts)
    importance = probs.mean(dim=0)  # keeps grad
    importance = importance / importance.sum().clamp_min(1e-6)

    usage = expert_usage.reshape(-1).float().detach()
    usage = usage / usage.sum().clamp_min(1e-6)
    if reduce_ddp:
        importance = all_reduce_mean(importance)
        usage = all_reduce_mean(usage)

    if target_usage is not None:
        w = target_usage.reshape(-1).float()
        w = w / w.sum().clamp_min(1e-6)
        usage = usage * w * num_experts  # uniform w -> unchanged

    return num_experts * torch.sum(importance * usage)

# P2 fix: explicit __all__ so `from mot import *` only exposes public API
# symbols, not internal helpers like _LocalConvTransformerExpert, _MoTRouter, etc.
__all__ = (
    "MoTBlock",
    "C2fMoT",
    "collect_mot_aux_loss",
    "anneal_mot_temperature",
)

# Maximum token count for explicit O(N²) SDPA fallback (PyTorch < 2.0).
_SDPA_EXPLICIT_MAX_TOKENS = 4096

# P2 fix: use separate CUDA streams to overlap expert computation in the dense
# (training / non-sparse) path. Each expert's forward runs on its own stream,
# then results are gathered back to the default stream. This is a no-op on CPU.
# The threshold avoids stream overhead for tiny spatial maps where the launch
# cost dominates.
_EXPERT_STREAM_MIN_ELEMENTS = 1024  # H*W above which streams are worthwhile

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


# Query-chunk size used by the memory-bounded fallback so peak memory stays
# O(chunk·N) instead of O(N²) on PyTorch < 2.0.
_SDPA_FALLBACK_CHUNK = 1024


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
          scale: float, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Scaled dot-product attention.

    Uses ``F.scaled_dot_product_attention`` (Flash-Attention / memory-efficient
    kernels) on PyTorch ≥ 2.0. On older PyTorch it falls back to softmax
    attention. For large token counts (N > ``_SDPA_EXPLICIT_MAX_TOKENS``) the
    fallback automatically switches to a **query-chunked** computation that
    bounds peak memory to O(chunk·N) instead of materialising the full N×N
    matrix — so it no longer crashes at high resolution, only runs slower.
    """
    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    n_tokens = q.shape[-2]
    if n_tokens > _SDPA_EXPLICIT_MAX_TOKENS:
        # Memory-bounded chunked softmax-attention fallback (PyTorch < 2.0).
        LOGGER.warning(
            f"PyTorch < 2.0 SDPA fallback: N={n_tokens} > {_SDPA_EXPLICIT_MAX_TOKENS}; "
            f"using query-chunked attention (chunk={_SDPA_FALLBACK_CHUNK}). "
            "Upgrade to PyTorch ≥ 2.0 for fast memory-efficient SDPA."
        )
        outs = []
        for start in range(0, n_tokens, _SDPA_FALLBACK_CHUNK):
            end = min(start + _SDPA_FALLBACK_CHUNK, n_tokens)
            attn = (q[..., start:end, :] @ k.transpose(-2, -1)) * scale
            if mask is not None:
                attn = attn + (mask[..., start:end, :] if mask.dim() == q.dim() else mask)
            outs.append(attn.softmax(dim=-1) @ v)
        return torch.cat(outs, dim=-2)
    attn = (q @ k.transpose(-2, -1)) * scale
    if mask is not None:
        attn = attn + mask
    return attn.softmax(dim=-1) @ v


# ---------------------------------------------------------------------------
# Expert 0: LocalConvTransformer
# ─────────────────────────────
# Attention with depthwise-conv-biased QKV + DW-7×7 positional encoding.
# Best inductive bias for fine-grained texture, edges, and small-object detail.
# Complexity: O(N²) — suitable for P4/P5 where spatial resolution is small.
# ---------------------------------------------------------------------------

class _LocalConvTransformerExpert(nn.Module):
    """Transformer expert with convolutional inductive bias.

    Uses depthwise-conv pre-processing for QKV to bias attention toward
    spatially-local patterns.  The FFN is a standard Gated Linear Unit (GLU):
    ``out = (Sigmoid(W_g x)) ⊙ (W_v x)``. (This is GLU proper — *not* SwiGLU,
    which would gate with SiLU/Swish instead of Sigmoid.)
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0,
                 dropout: float = 0.0):
        super().__init__()
        # P1 fix: user-facing config validation must not be an `assert`
        # (stripped under `python -O`); a silent pass-through here would
        # later crash with an opaque reshape error deep in `to_heads()`.
        if dim % num_heads != 0:
            raise ValueError(f"_LocalConvTransformerExpert: dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # DW-3×3 pre-mixing before QKV projection
        self.dw_mix = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.pe = nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

        self.norm1 = nn.GroupNorm(_safe_groups(dim, 8), dim)
        self.norm2 = nn.GroupNorm(_safe_groups(dim, 8), dim)
        self.drop = nn.Dropout2d(dropout)

        # Gated FFN: standard GLU (Sigmoid gate × value), 2-conv split.
        ffn_hidden = int(dim * mlp_ratio)
        # split into gate + value projections
        self.ffn_gate = nn.Sequential(Conv(dim, ffn_hidden, 1), nn.Sigmoid())
        self.ffn_val  = Conv(dim, ffn_hidden, 1)
        self.ffn_out  = Conv(ffn_hidden, dim, 1, act=False)

        self.ls1 = nn.Parameter(torch.ones(dim, 1, 1) * 0.1)
        self.ls2 = nn.Parameter(torch.ones(dim, 1, 1) * 0.1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        nh, hd = self.num_heads, self.head_dim

        # ── Attention ────────────────────────────────────────────────────
        xn = self.norm1(x)
        qkv = self.qkv(self.dw_mix(xn)).flatten(2)       # [B, 3C, N]
        q, k, v = qkv.split(C, dim=1)

        def to_heads(t):
            return t.view(B, nh, hd, N).transpose(2, 3)  # [B,nh,N,hd]

        # Positional encoding on v
        v_2d = v.reshape(B, C, H, W)
        v_2d = v_2d + self.pe(v_2d)
        v = v_2d.flatten(2)

        out = _sdpa(to_heads(q), to_heads(k), to_heads(v), self.scale)
        out = out.transpose(2, 3).reshape(B, C, H, W)
        x = x + self.ls1 * self.drop(self.proj(out))

        # ── Gated FFN ─────────────────────────────────────────────────────
        xn = self.norm2(x)
        ffn = self.ffn_gate(xn) * self.ffn_val(xn)
        x = x + self.ls2 * self.ffn_out(ffn)
        return x


# ---------------------------------------------------------------------------
# Expert 1: WindowTransformer (Swin-style)
# ─────────────────────────────────────────
# Partitions the feature map into non-overlapping windows of size win×win,
# applies self-attention within each window, then applies shifted-window
# attention in the next call (alternating via a stride buffer).
# Best for medium objects with regular structure.
# Complexity: O(N · win²) — scales linearly with image size.
# ---------------------------------------------------------------------------

class _WindowTransformerExpert(nn.Module):
    """Swin-style window-partitioned Transformer expert.

    The cyclic-shift used for cross-window connectivity is fixed at
    construction time (``shift_size``), decided by the block's position in
    the network rather than by a runtime step counter. This keeps inference
    deterministic and the forward graph trace-stable (ONNX/TorchScript safe):
    pairing a non-shifted block with a shifted block over the stack provides
    both receptive fields, exactly as in Swin Transformer.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int = 7,
                 mlp_ratio: float = 2.0, dropout: float = 0.0,
                 shift_size: int = 0):
        super().__init__()
        # P1 fix: replace `assert` with an explicit exception (see
        # _LocalConvTransformerExpert for rationale).
        if dim % num_heads != 0:
            raise ValueError(f"_WindowTransformerExpert: dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.win = window_size
        # Fixed cyclic shift (0 = regular window, win//2 = shifted window).
        self.shift_size = (window_size // 2) if shift_size else 0

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        ffn_hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, dim),
        )
        self.drop = nn.Dropout(dropout)
        self.ls1 = nn.Parameter(torch.ones(dim) * 0.1)
        self.ls2 = nn.Parameter(torch.ones(dim) * 0.1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _pad_to_window(x: torch.Tensor, win: int) -> Tuple[torch.Tensor, int, int]:
        """Pad spatial dims to be divisible by win."""
        B, H, W, C = x.shape
        pad_h = (win - H % win) % win
        pad_w = (win - W % win) % win
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        return x, pad_h, pad_w

    @staticmethod
    def _window_partition(x: torch.Tensor, win: int) -> torch.Tensor:
        """[B, H, W, C] → [B*nH*nW, win*win, C]"""
        B, H, W, C = x.shape
        x = x.view(B, H // win, win, W // win, win, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win * win, C)

    @staticmethod
    def _window_reverse(windows: torch.Tensor, win: int, H: int, W: int) -> torch.Tensor:
        """[B*nH*nW, win*win, C] → [B, H, W, C]"""
        # P1 fix: use integer floor-division for the window grid instead of
        # float division + int() truncation. `H` and `W` here are always the
        # *padded* dims (divisible by `win`), so `//` is exact; the float
        # path could round down 1 short under floating-point error for large
        # H*W, silently corrupting the batch dimension.
        B = windows.shape[0] // ((H // win) * (W // win))
        C = windows.shape[2]
        x = windows.view(B, H // win, W // win, win, win, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H_orig, W_orig = x.shape
        win = self.win

        # NCHW → NHWC
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]

        # Pad
        x, pad_h, pad_w = self._pad_to_window(x, win)
        H, W = x.shape[1], x.shape[2]

        # Cyclic shift (fixed at construction → deterministic & trace-stable)
        shift = self.shift_size
        if shift > 0:
            x = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))

        # ── Window Attention ─────────────────────────────────────────────
        xn = self.norm1(x)
        windows = self._window_partition(xn, win)        # [Bw, win², C]
        Bw = windows.shape[0]
        nh, hd = self.num_heads, self.head_dim

        qkv = self.qkv(windows).reshape(Bw, win * win, 3, nh, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                         # each [Bw, nh, win², hd]

        attn_out = _sdpa(q, k, v, self.scale)            # [Bw, nh, win², hd]
        attn_out = attn_out.transpose(1, 2).reshape(Bw, win * win, C)
        attn_out = self.drop(self.proj(attn_out))

        # Reverse window partition
        attn_out = self._window_reverse(attn_out, win, H, W)   # [B, H, W, C]

        # Reverse shift
        if shift > 0:
            attn_out = torch.roll(attn_out, shifts=(shift, shift), dims=(1, 2))

        # Remove padding
        attn_out = attn_out[:, :H_orig, :W_orig, :]

        # Reverse shift on x BEFORE residual addition so spatial positions align.
        # Without this, the shifted input x is added to the un-shifted attention
        # output, causing a cyclic spatial misalignment in all shifted blocks.
        if shift > 0:
            x = torch.roll(x, shifts=(shift, shift), dims=(1, 2))
        x = x[:, :H_orig, :W_orig, :]
        x = x + self.ls1 * attn_out

        # ── FFN ──────────────────────────────────────────────────────────
        x = x + self.ls2 * self.ffn(self.norm2(x))

        return x.permute(0, 3, 1, 2).contiguous()   # NHWC → NCHW


# ---------------------------------------------------------------------------
# Expert 2: DeformableTransformerExpert
# ──────────────────────────────────────
# Sparse deformable attention: each query samples K learnable offset points
# instead of attending to all positions. Best for irregular shapes and
# occluded objects where the optimal attention region is content-dependent.
# Complexity: O(N · K) where K is fixed (default 4) — very efficient.
# ---------------------------------------------------------------------------

class _DeformableTransformerExpert(nn.Module):
    """Deformable-attention Transformer expert.

    Each query predicts `n_points` sampling offsets and attention weights,
    then aggregates sampled features.  This is a simplified single-scale
    variant of MS-Deformable-DETR, adapted for CNN feature maps.

    Reference:
        Zhu et al., "Deformable DETR" (ICLR 2021)
    """

    def __init__(self, dim: int, num_heads: int, n_points: int = 4,
                 mlp_ratio: float = 2.0, dropout: float = 0.0,
                 align_corners: bool = True):
        super().__init__()
        # P1 fix: replace `assert` with an explicit exception (see
        # _LocalConvTransformerExpert for rationale).
        if dim % num_heads != 0:
            raise ValueError(f"_DeformableTransformerExpert: dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.n_points = n_points
        self.align_corners = align_corners

        # Query projection
        self.q_proj = nn.Linear(dim, dim, bias=False)
        # Value projection
        self.v_proj = nn.Linear(dim, dim, bias=False)
        # Sampling offsets: [nh, n_points, 2]
        self.offset_proj = nn.Linear(dim, num_heads * n_points * 2, bias=True)
        # Attention weights over sampled points
        self.attn_proj = nn.Linear(dim, num_heads * n_points, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        ffn_hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, dim),
        )
        self.drop = nn.Dropout(dropout)
        self.ls1 = nn.Parameter(torch.ones(dim) * 0.1)
        self.ls2 = nn.Parameter(torch.ones(dim) * 0.1)

        self._init_weights()

    def _init_weights(self):
        # Zero-init offsets → initial grid sampling (identity-like)
        nn.init.zeros_(self.offset_proj.weight)
        nn.init.zeros_(self.offset_proj.bias)
        # Uniform attention weights initially
        nn.init.zeros_(self.attn_proj.weight)
        nn.init.zeros_(self.attn_proj.bias)
        for m in [self.q_proj, self.v_proj, self.out_proj]:
            nn.init.trunc_normal_(m.weight, std=0.02)
        for m in self.ffn:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _deform_attn(self, q: torch.Tensor, value: torch.Tensor,
                     H: int, W: int) -> torch.Tensor:
        """Core deformable attention.

        Args:
            q     : [B, N, C]    query tokens
            value : [B, H*W, C]  value feature map (flattened)
            H, W  : feature map spatial dims

        Returns:
            out   : [B, N, C]
        """
        B, N, C = q.shape
        nh, np_, hd = self.num_heads, self.n_points, self.head_dim
        # Token→coordinate mapping below assumes N == H*W with no padding.
        # P1 fix: this validates caller-supplied H/W against the actual
        # flattened token count, so it must survive `python -O`.
        if N != H * W:
            raise ValueError(f"deformable expert expects N==H*W, got N={N}, H*W={H * W}")

        # Predict sampling offsets & attention weights from query
        offsets = self.offset_proj(q)                         # [B, N, nh*np*2]
        offsets = offsets.reshape(B, N, nh, np_, 2)           # [B, N, nh, np, 2]
        offsets = offsets.tanh()                               # clamp to [-1,1]

        attn_w = self.attn_proj(q).reshape(B, N, nh, np_)     # [B, N, nh, np]
        attn_w = F.softmax(attn_w, dim=-1)                    # normalize over points

        # Build reference grid: each token's own position (normalised to [-1,1])
        # Token index → (row, col) → (x_norm, y_norm)
        idx = torch.arange(N, device=q.device)
        row = (idx // W).float() / max(H - 1, 1) * 2 - 1    # y in [-1,1]
        col = (idx %  W).float() / max(W - 1, 1) * 2 - 1    # x in [-1,1]
        ref = torch.stack([col, row], dim=-1)                 # [N, 2]
        ref = ref[None, :, None, None, :].expand(B, -1, nh, np_, -1)  # [B,N,nh,np,2]

        # Sampling locations = reference + learned offsets (scaled, clamped to valid range)
        sample_locs = (ref + offsets * 0.25).clamp(-1.0, 1.0)   # [B, N, nh, np, 2]

        # Value: [B, H*W, C] → reshape for grid_sample [B, C, H, W]
        v_4d = self.v_proj(value).permute(0, 2, 1).reshape(B, C, H, W)

        # Sample per head: split C into heads
        v_4d = v_4d.reshape(B * nh, hd, H, W)
        # sample_locs: [B, N, nh, np, 2] → [B*nh, N, np, 2]
        locs = sample_locs.permute(0, 2, 1, 3, 4).reshape(B * nh, N, np_, 2)

        # grid_sample expects [B, C, H_out, W_out] query grid
        # Here H_out=N, W_out=np  → treat N*np as 2D grid
        # Reshape locs → [B*nh, N, np, 2] already correct for grid_sample
        sampled = F.grid_sample(
            v_4d,                                              # [B*nh, hd, H, W]
            locs,                                              # [B*nh, N, np, 2]
            mode="bilinear", align_corners=self.align_corners, padding_mode="zeros"
        )                                                      # [B*nh, hd, N, np]

        # sampled: [B*nh, hd, N, np] → [B, nh, N, np, hd]
        sampled = sampled.reshape(B, nh, hd, N, np_).permute(0, 3, 1, 4, 2).contiguous()

        # Weighted sum over sampling points: [B, N, nh, hd]
        out = (attn_w.unsqueeze(-1) * sampled).sum(dim=3)    # [B, N, nh, hd]
        out = out.reshape(B, N, C)
        return self.out_proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        # NCHW → NLC
        x_flat = x.flatten(2).transpose(1, 2)  # [B, N, C]

        # ── Deformable Attention ─────────────────────────────────────────
        q = self.q_proj(self.norm1(x_flat))
        attn_out = self.drop(self._deform_attn(q, self.norm1(x_flat), H, W))
        x_flat = x_flat + self.ls1 * attn_out

        # ── FFN ──────────────────────────────────────────────────────────
        x_flat = x_flat + self.ls2 * self.ffn(self.norm2(x_flat))

        return x_flat.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# MoT Router
# ---------------------------------------------------------------------------

class _MoTRouter(nn.Module):
    """Content-aware router for MoT: assigns each token to Top-K experts.

    Architecture: global average pool → linear → softmax (soft) or
    top-k + renormalize (hard).

    For spatial routing (token-level), uses a lightweight 1×1 conv.
    For image-level routing (same assignment for all tokens), uses GAP.
    Default: token-level soft routing for stability; hard routing optional.

    Args:
        dim (int): Input channels.
        num_experts (int): Number of transformer experts (default 3).
        top_k (int): Number of active experts per token (soft Top-K mask).
        use_spatial (bool): Token-level vs. image-level routing.
        temperature (float): Softmax temperature for soft routing.
    """

    def __init__(self, dim: int, num_experts: int = 3, top_k: int = 2,
                 use_spatial: bool = True, temperature: float = 1.0,
                 exploration_eps: float = 0.02):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_spatial = use_spatial
        self.temperature = max(temperature, 0.1)
        self.exploration_eps = exploration_eps

        hidden = max(dim // 8, num_experts * 4)
        # P1 fix: drop `inplace=True` on the router's SiLU. In-place
        # activation on a tensor that also feeds the residual/`aux_loss`
        # graph (or that autograd otherwise needs unmodified for the
        # backward of the preceding GroupNorm/Linear) can raise
        # "a leaf Variable that requires grad is being used in an in-place
        # operation" or silently corrupt gradients depending on the autograd
        # version — the router is on the hot path for every MoT forward, so
        # correctness here matters more than the small memory saving.
        if use_spatial:
            self.router = nn.Sequential(
                nn.Conv2d(dim, hidden, 1, bias=False),
                nn.GroupNorm(_safe_groups(hidden, 4), hidden),
                nn.SiLU(inplace=False),
                nn.Conv2d(hidden, num_experts, 1, bias=True),
            )
        else:
            self.router = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(dim, hidden, bias=False),
                nn.SiLU(inplace=False),
                nn.Linear(hidden, num_experts, bias=True),
            )
        # init: near-uniform
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def _compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x)
        if not self.use_spatial:
            logits = logits.unsqueeze(-1).unsqueeze(-1)
        return logits

    @staticmethod
    def z_loss_from_logits(logits: torch.Tensor) -> torch.Tensor:
        log_z = torch.logsumexp(logits, dim=1)
        return (log_z ** 2).mean()

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            weights : [B, num_experts, H, W] or [B, num_experts, 1, 1]  (soft, sum-to-1)
            indices : [B, top_k, H, W] or [B, top_k, 1, 1]  (top-k expert ids)
        """
        logits = self._compute_logits(x)      # [B, E, H, W] or [B, E, 1, 1] after GAP

        # Use training temperature during train; fixed 1.0 at eval for stable routing.
        temp = self.temperature if self.training else 1.0

        # Soft weights (always computed for gradient flow)
        weights = F.softmax(logits / temp, dim=1)   # [B, E, H, W]
        dense_weights = weights

        # Top-K mask
        if self.top_k < self.num_experts:
            # get top-k indices [B, K, H, W]
            topk_vals, topk_idx = weights.topk(self.top_k, dim=1)
            # renormalize selected weights
            topk_weights = topk_vals / topk_vals.sum(dim=1, keepdim=True).clamp(min=1e-6)
            # scatter back to [B, E, H, W] sparse
            sparse_w = torch.zeros_like(weights)
            sparse_w.scatter_(1, topk_idx, topk_weights)
            weights = sparse_w
            if self.training and self.exploration_eps > 0:
                eps = min(max(self.exploration_eps, 0.0), 0.2)
                weights = weights * (1.0 - eps) + dense_weights * eps
            indices = topk_idx
        else:
            indices = torch.arange(self.num_experts, device=x.device).view(
                1, -1, 1, 1).expand(x.shape[0], -1, x.shape[2], x.shape[3])

        if return_logits:
            return weights, indices, logits
        return weights, indices

    @staticmethod
    def expert_usage_from_indices(indices: torch.Tensor, num_experts: int) -> torch.Tensor:
        """Discrete expert usage share from Top-K index tensor (any rank ≥ 2)."""
        flat = indices.reshape(-1).to(torch.long)
        counts = torch.bincount(flat, minlength=num_experts).float()
        return counts / max(flat.numel(), 1)

    def router_z_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Z-loss for load balance: encourages router logits to be small."""
        return self.z_loss_from_logits(self._compute_logits(x))


def _mot_router_aux_loss(
    weights: torch.Tensor,
    logits: torch.Tensor,
    indices: torch.Tensor,
    num_experts: int,
    balance_coeff: float,
    z_coeff: float,
    reduce_ddp: bool = True,
) -> torch.Tensor:
    """GShard balance + router z-loss for MoT (matches MoE/MoLoRA formulation).

    P1 fix: ``reduce_ddp`` defaults to True so that balance statistics are
    averaged across DDP ranks, matching MoE's ``MoELoss`` and MoLoRA's
    ``MoLoRALoss``.  On single-GPU/CPU the ``all_reduce_mean`` inside
    ``differentiable_balance_loss`` is a no-op.
    """
    if balance_coeff <= 0 and z_coeff <= 0:
        return weights.new_zeros(())

    probs = weights
    if probs.dim() == 4:
        probs = probs.reshape(probs.shape[0], num_experts, -1).mean(-1)
    probs = probs.reshape(-1, num_experts)

    usage = _MoTRouter.expert_usage_from_indices(indices, num_experts)
    balance = differentiable_balance_loss(probs, usage, num_experts, reduce_ddp=reduce_ddp)
    z_loss = _MoTRouter.z_loss_from_logits(logits)

    total = weights.new_zeros(())
    if balance_coeff > 0:
        total = total + balance_coeff * balance
    if z_coeff > 0:
        total = total + z_coeff * z_loss
    return total


# ---------------------------------------------------------------------------
# MoTBlock — core building block
# ---------------------------------------------------------------------------

class MoTBlock(nn.Module):
    """Mixture-of-Transformers Block.

    Routes each spatial token (or image) across K of E complete Transformer
    experts, then softly combines their outputs via learned routing weights.

    Experts:
      0: LocalConvTransformer  — Conv-biased attention + Gated FFN
      1: WindowTransformer     — Swin-style shifted window attention + FFN
      2: DeformableTransformer — Deformable sparse sampling attention + FFN

    Args:
        dim (int): Channel dimension.
        num_heads (int): Attention heads per expert.
        top_k (int): Active experts per forward (1 or 2 recommended).
        window_size (int): Window size for WindowTransformer expert.
        n_points (int): Deformable sampling points per head.
        mlp_ratio (float): FFN expansion ratio.
        temperature (float): Router softmax temperature.
        use_spatial_router (bool): Token-level vs image-level routing.
        balance_loss_coeff (float): Weight of router z-loss (0 to disable).
        dropout (float): Dropout for attention/FFN.
        exploration_eps (float): Training-only dense routing floor that keeps all experts trainable.

    Shape:
        Input:  [B, dim, H, W]
        Output: [B, dim, H, W]
    """

    NUM_EXPERTS: int = 3

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        top_k: int = 2,
        window_size: int = 7,
        n_points: int = 4,
        mlp_ratio: float = 2.0,
        temperature: float = 1.0,
        use_spatial_router: bool = True,
        balance_loss_coeff: float = 0.01,
        router_z_loss_coeff: float | None = None,
        dropout: float = 0.0,
        exploration_eps: float = 0.02,
        window_shift: bool = False,
        grid_align_corners: bool = True,
        sparse_train: bool = False,
    ):
        super().__init__()
        # P1 fix: replace `assert` with an explicit exception — this is a
        # user-supplied config invariant, not an internal-only sanity check.
        if not (1 <= top_k <= self.NUM_EXPERTS):
            raise ValueError(f"top_k must be in [1, {self.NUM_EXPERTS}], got {top_k}")
        self.top_k = top_k
        self.balance_loss_coeff = balance_loss_coeff
        self.sparse_train = sparse_train
        # P1 fix: `balance_loss_coeff` historically doubled as the z-loss
        # coefficient when `router_z_loss_coeff` was omitted (legacy
        # YAML/checkpoint compatibility). That implicit coupling is easy to
        # miss — a user who only sets `balance_loss_coeff` (expecting it to
        # control *only* the GShard balance term) silently also changes the
        # z-loss weight. Warn once so the coupling is discoverable without
        # reading the source.
        if router_z_loss_coeff is None and balance_loss_coeff > 0:
            LOGGER.warning(
                f"MoTBlock(dim={dim}): router_z_loss_coeff not set — defaulting to "
                f"balance_loss_coeff ({balance_loss_coeff}) for backward compatibility. "
                "Pass router_z_loss_coeff explicitly to decouple the two terms."
            )
        # Legacy YAML/checkpoints used balance_loss_coeff for z-loss only; when
        # router_z_loss_coeff is omitted, keep that behaviour for the z term.
        self.router_z_loss_coeff = balance_loss_coeff if router_z_loss_coeff is None else router_z_loss_coeff

        # Each expert uses the *full* dim with its own num_heads.
        # Find largest num_heads ≤ requested that divides dim.
        expert_heads = num_heads
        while dim % expert_heads != 0 and expert_heads > 1:
            expert_heads -= 1
        expert_heads = max(1, expert_heads)
        if expert_heads != num_heads:
            LOGGER.warning(
                f"MoTBlock(dim={dim}): num_heads {num_heads} reduced to {expert_heads} "
                f"(must divide dim and yield head_dim ≥ 1)"
            )

        # Three Transformer experts
        self.experts = nn.ModuleList([
            _LocalConvTransformerExpert(dim, expert_heads, mlp_ratio, dropout),
            _WindowTransformerExpert(dim, expert_heads, window_size, mlp_ratio, dropout,
                                     shift_size=window_size // 2 if window_shift else 0),
            _DeformableTransformerExpert(
                dim, expert_heads, n_points, mlp_ratio, dropout,
                align_corners=grid_align_corners,
            ),
        ])

        # Router
        self.router = _MoTRouter(
            dim, self.NUM_EXPERTS, top_k,
            use_spatial=use_spatial_router,
            temperature=temperature,
            exploration_eps=exploration_eps,
        )

        # Final output norm & projection
        self.out_norm = nn.GroupNorm(_safe_groups(dim, 8), dim)
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=False)

        self._init_weights()
        self.last_aux_loss: torch.Tensor | None = None

    def _init_weights(self):
        nn.init.trunc_normal_(self.out_proj.weight, std=0.02)

    def _blend_experts(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Blend expert outputs; sparse when eval or ``sparse_train`` (not during ONNX export).

        Sparse dispatch keys off discrete Top-K ``indices`` so training-time
        ``exploration_eps`` blending does not force every expert to run.

        P2: On CUDA with large spatial maps, the dense (training) path launches
        each expert on a separate CUDA stream so their attention/FFN kernels
        overlap. Streams are only used when (a) we're on CUDA, (b) not tracing
        for ONNX/TorchScript, and (c) the spatial map is large enough that the
        kernel launch overhead is amortised.
        """
        out = x.new_zeros(x.shape)
        use_sparse = (not self.training or self.sparse_train) and not torch.onnx.is_in_onnx_export()
        B = x.shape[0]
        if use_sparse:
            for e_idx, expert in enumerate(self.experts):
                if indices is not None:
                    active = (indices == e_idx).reshape(B, -1).any(dim=1)
                else:
                    active = weights[:, e_idx].reshape(B, -1).sum(dim=1) > 0
                batch_idx = torch.nonzero(active, as_tuple=True)[0]
                if batch_idx.numel() == 0:
                    continue
                w = weights[batch_idx, e_idx:e_idx + 1]
                expert_out = expert(x[batch_idx])
                # P0 fix: each Transformer expert must be shape-preserving
                # ([B,C,H,W] -> [B,C,H,W]); a misconfigured expert (e.g. a
                # window/deformable expert whose internal pad/crop drops a
                # row when H or W isn't a multiple of its window size) would
                # otherwise silently broadcast-fail or corrupt `out` via the
                # advanced-index assignment below. Fail loudly instead.
                if expert_out.shape != x[batch_idx].shape:
                    raise RuntimeError(
                        f"MoTBlock expert {e_idx} ({type(expert).__name__}) changed tensor shape: "
                        f"input {tuple(x[batch_idx].shape)} -> output {tuple(expert_out.shape)}. "
                        "All MoT experts must be shape-preserving."
                    )
                out[batch_idx] = out[batch_idx] + expert_out * w
        else:
            # P2: Determine whether CUDA stream parallelization is worthwhile.
            # Only use streams on CUDA, not during export, and when the spatial
            # map is large enough (≥ _EXPERT_STREAM_MIN_ELEMENTS per sample).
            use_streams = (
                x.is_cuda
                and not torch.jit.is_scripting()
                and not torch.onnx.is_in_onnx_export()
                and x.shape[2] * x.shape[3] >= _EXPERT_STREAM_MIN_ELEMENTS
            )
            if use_streams:
                expert_outs = [None] * len(self.experts)
                streams = [torch.cuda.Stream() for _ in self.experts]
                for e_idx, (expert, stream) in enumerate(zip(self.experts, streams)):
                    with torch.cuda.stream(stream):
                        expert_outs[e_idx] = expert(x)
                # Synchronize all streams before gathering results
                for stream in streams:
                    stream.synchronize()
                for e_idx, expert_out in enumerate(expert_outs):
                    if expert_out.shape != x.shape:
                        raise RuntimeError(
                            f"MoTBlock expert {e_idx} ({type(self.experts[e_idx]).__name__}) changed tensor shape: "
                            f"input {tuple(x.shape)} -> output {tuple(expert_out.shape)}. "
                            "All MoT experts must be shape-preserving."
                        )
                    w = weights[:, e_idx:e_idx + 1]
                    out = out + expert_out * w
            else:
                for e_idx, expert in enumerate(self.experts):
                    w = weights[:, e_idx:e_idx + 1]
                    expert_out = expert(x)
                    if expert_out.shape != x.shape:
                        raise RuntimeError(
                            f"MoTBlock expert {e_idx} ({type(expert).__name__}) changed tensor shape: "
                            f"input {tuple(x.shape)} -> output {tuple(expert_out.shape)}. "
                            "All MoT experts must be shape-preserving."
                        )
                    out = out + expert_out * w
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            out           : [B, C, H, W]
            aux_loss      : scalar (GShard balance + router z-loss, 0 if both coeffs==0)
        """
        # ── Routing weights ──────────────────────────────────────────────
        weights, indices, router_logits = self.router(x, return_logits=True)   # [B, E, H, W]

        # ── Expert computation ───────────────────────────────────────────
        out = self._blend_experts(x, weights, indices)
        out = self.out_norm(self.out_proj(out))

        # Residual (block-level shortcut)
        out = out + x

        # ── Auxiliary loss ───────────────────────────────────────────────
        if self.training and (self.balance_loss_coeff > 0 or self.router_z_loss_coeff > 0):
            aux = _mot_router_aux_loss(
                weights,
                router_logits,
                indices,
                self.NUM_EXPERTS,
                self.balance_loss_coeff,
                self.router_z_loss_coeff,
            )
        else:
            aux = x.new_zeros(())

        self.last_aux_loss = aux
        return out, aux


# ---------------------------------------------------------------------------
# C2fMoT — C2f-style wrapper for backbone/neck (YAML-compatible)
# ---------------------------------------------------------------------------

class C2fMoT(nn.Module):
    """C2f-style feature-flow wrapper around MoTBlock.

    Provides the same [c1, c2, n, ...] interface as C3k2 / A2C2f / C2fMoA,
    enabling direct YAML substitution.

    The aux_loss from each MoTBlock is accumulated and accessible via
    ``self.last_aux_loss`` after each forward call.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of stacked MoTBlock layers.
        num_heads (int): Attention heads per expert.
        top_k (int): Active experts per token (1 or 2).
        window_size (int): Window size for WindowTransformer.
        n_points (int): Deformable sampling points.
        mlp_ratio (float): FFN expansion ratio.
        temperature (float): Router temperature.
        balance_loss_coeff (float): Router balance loss weight.
        e (float): Internal channel expansion ratio.

    Shape:
        Input:  [B, c1, H, W]
        Output: [B, c2, H, W]
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        num_heads: int = 6,
        top_k: int = 2,
        window_size: int = 7,
        n_points: int = 4,
        mlp_ratio: float = 2.0,
        temperature: float = 1.0,
        balance_loss_coeff: float = 0.01,
        e: float = 0.5,
        sparse_train: bool = False,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # Clamp num_heads to valid range
        dim = self.c
        # Find largest num_heads ≤ requested that divides dim and gives head_dim ≥ 8
        eff_heads = num_heads
        while eff_heads > 1 and (dim % eff_heads != 0 or dim // eff_heads < 8):
            eff_heads -= 1
        eff_heads = max(1, eff_heads)
        if eff_heads != num_heads:
            LOGGER.warning(
                f"C2fMoT(c={dim}): num_heads {num_heads} reduced to {eff_heads} "
                f"(must divide channels and yield head_dim ≥ 8)"
            )

        # Alternate window shift by block index (Swin-style): even blocks use
        # regular windows, odd blocks use shifted windows. Fixed at build time
        # → deterministic inference and trace-stable export.
        self.m = nn.ModuleList(
            MoTBlock(
                dim=dim,
                num_heads=eff_heads,
                top_k=top_k,
                window_size=window_size,
                n_points=n_points,
                mlp_ratio=mlp_ratio,
                temperature=temperature,
                balance_loss_coeff=balance_loss_coeff,
                window_shift=bool(i % 2),
                sparse_train=sparse_train,
            )
            for i in range(n)
        )
        # Lazily set on first forward (always overwritten there). Kept as a
        # device-agnostic scalar so a pre-forward read never injects a stale,
        # wrong-device tensor; requires_grad=False ⇒ filtered by the collector.
        self.last_aux_loss: torch.Tensor = torch.zeros((), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        aux_total = x.new_zeros(())
        for m in self.m:
            out, aux = m(y[-1])
            y.append(out)
            aux_total = aux_total + aux
        self.last_aux_loss = aux_total
        return self.cv2(torch.cat(y, dim=1))


# ---------------------------------------------------------------------------
# Utility: collect aux losses from all MoT modules
# ---------------------------------------------------------------------------

def _aux_loss_device(model: nn.Module) -> torch.device:
    """Best-effort device lookup for zero aux-loss fallbacks."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def collect_mot_aux_loss(model: nn.Module) -> torch.Tensor:
    """Sum all MoT router aux losses in the model.

    Scans both C2fMoT wrappers and standalone MoTBlock instances.
    Uses id-based deduplication so blocks nested inside C2fMoT are not
    double-counted.

    Call in the training loss function:
        loss = det_loss + collect_mot_aux_loss(model)
    """
    total = None
    covered: set[int] = set()
    for m in model.modules():
        if isinstance(m, C2fMoT):
            l = m.last_aux_loss
            if isinstance(l, torch.Tensor) and l.requires_grad:
                total = l if total is None else total + l
            covered.update(id(child) for child in m.modules())
        elif isinstance(m, MoTBlock) and id(m) not in covered:
            l = getattr(m, 'last_aux_loss', None)
            if isinstance(l, torch.Tensor) and l.requires_grad:
                total = l if total is None else total + l
    return total if total is not None else torch.zeros(1, device=_aux_loss_device(model))


def anneal_mot_temperature(model: nn.Module, factor: float = 0.97,
                           min_temp: float = 0.3) -> None:
    """Multiplicatively anneal MoT router temperatures each epoch."""
    for m in model.modules():
        if isinstance(m, _MoTRouter):
            m.temperature = max(m.temperature * factor, min_temp)
