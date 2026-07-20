"""Transformer experts for Mixture-of-Transformer blocks."""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.utils import get_safe_groups as _safe_groups
from ultralytics.nn.modules.mot._constants import SDPA_EXPLICIT_MAX_TOKENS, SDPA_FALLBACK_CHUNK
from ultralytics.utils import LOGGER

_SDPA_EXPLICIT_MAX_TOKENS = SDPA_EXPLICIT_MAX_TOKENS
_SDPA_FALLBACK_CHUNK = SDPA_FALLBACK_CHUNK

def _roll_via_cat(x: torch.Tensor, shift: int, dims: tuple) -> torch.Tensor:
    """ONNX-compatible alternative to ``torch.roll`` for 2-D spatial shifts.

    ``torch.roll`` has inconsistent ONNX Runtime support across versions.
    This function replicates cyclic shift via slice-and-concatenate,
    which traces cleanly.  Supports symmetric shifts on dims (1, 2) and
    handles both positive and negative shift values.
    """
    if shift == 0:
        return x
    for dim in dims:
        n = x.size(dim)
        s = shift % n  # Python modulo handles negative shifts correctly
        if s == 0:
            continue
        x = torch.cat([x.narrow(dim, n - s, s), x.narrow(dim, 0, n - s)], dim=dim)
    return x

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
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by positive num_heads ({num_heads})")
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
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by positive num_heads ({num_heads})")
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
        windows_per_image = (H // win) * (W // win)
        B = windows.shape[0] // windows_per_image
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
        # Use cat-based roll for ONNX compatibility (torch.roll has limited
        # ONNX Runtime support across versions).
        shift = self.shift_size
        if shift > 0:
            if torch.jit.is_tracing() or torch.onnx.is_in_onnx_export():
                x = _roll_via_cat(x, -shift, dims=(1, 2))
            else:
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
            if torch.jit.is_tracing() or torch.onnx.is_in_onnx_export():
                attn_out = _roll_via_cat(attn_out, shift, dims=(1, 2))
            else:
                attn_out = torch.roll(attn_out, shifts=(shift, shift), dims=(1, 2))

        # Remove padding
        attn_out = attn_out[:, :H_orig, :W_orig, :]

        # Reverse shift on x BEFORE residual addition so spatial positions align.
        # Without this, the shifted input x is added to the un-shifted attention
        # output, causing a cyclic spatial misalignment in all shifted blocks.
        if shift > 0:
            if torch.jit.is_tracing() or torch.onnx.is_in_onnx_export():
                x = _roll_via_cat(x, shift, dims=(1, 2))
            else:
                x = torch.roll(x, shifts=(shift, shift), dims=(1, 2))
        x = x[:, :H_orig, :W_orig, :]
        x = x + self.ls1 * attn_out

        # ── FFN ──────────────────────────────────────────────────────────
        x = x + self.ls2 * self.ffn(self.norm2(x))

        return x.permute(0, 3, 1, 2).contiguous()

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
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by positive num_heads ({num_heads})")
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
        # Fall back to dense attention for callers that provide non-grid tokens.
        if N != H * W:
            value_proj = self.v_proj(value)
            q_heads = q.reshape(B, N, nh, hd).transpose(1, 2)
            value_heads = value_proj.reshape(B, value_proj.shape[1], nh, hd).transpose(1, 2)
            dense = F.scaled_dot_product_attention(q_heads, value_heads, value_heads)
            return self.out_proj(dense.transpose(1, 2).reshape(B, N, C))

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
        # fp16 safety: grid_sample has known precision issues with fp16
        # sampling coordinates; force float32 for the sampling operation
        # and cast back to the original dtype afterward.
        orig_dtype = v_4d.dtype
        if orig_dtype != torch.float32:
            v_4d = v_4d.float()
            locs = locs.float()
        sampled = F.grid_sample(
            v_4d,                                              # [B*nh, hd, H, W]
            locs,                                              # [B*nh, N, np, 2]
            mode="bilinear", align_corners=self.align_corners, padding_mode="zeros"
        )                                                      # [B*nh, hd, N, np]
        if orig_dtype != torch.float32:
            sampled = sampled.to(orig_dtype)

        # sampled: [B*nh, hd, N, np] → [B, nh, N, np, hd]
        sampled = sampled.reshape(B, nh, hd, N, np_).permute(0, 3, 1, 4, 2).contiguous()

        # Weighted sum over sampling points: [B, N, nh, hd]
        out = (attn_w.unsqueeze(-1) * sampled).sum(dim=3)    # [B, N, nh, hd]
        out = out.reshape(B, N, C)
        return self.out_proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # NCHW → NLC
        x_flat = x.flatten(2).transpose(1, 2)  # [B, N, C]

        # ── Deformable Attention ─────────────────────────────────────────
        # norm1 computed once and reused for both q and value (avoids 2-5%
        # redundant LayerNorm compute noted in P1 audit).
        xn = self.norm1(x_flat)
        q = self.q_proj(xn)
        attn_out = self.drop(self._deform_attn(q, xn, H, W))
        x_flat = x_flat + self.ls1 * attn_out

        # ── FFN ──────────────────────────────────────────────────────────
        x_flat = x_flat + self.ls2 * self.ffn(self.norm2(x_flat))

        return x_flat.transpose(1, 2).reshape(B, C, H, W)

__all__ = ("_DeformableTransformerExpert", "_LocalConvTransformerExpert", "_WindowTransformerExpert", "_roll_via_cat", "_sdpa")
