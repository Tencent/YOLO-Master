"""Attention heads for Mixture-of-Attention blocks."""
from __future__ import annotations
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ultralytics.nn.modules._numeric import fp_clamp_floor
from ultralytics.nn.modules.moa._constants import DEFAULT_RF_SEED, LINEAR_ATTN_ACTIVATION_LIMIT, LINEAR_ATTN_BLEND_WINDOW, LINEAR_ATTN_THRESHOLD
from ultralytics.nn.modules.utils import get_safe_groups as _safe_groups

_DEFAULT_RF_SEED = DEFAULT_RF_SEED
_fp_min = fp_clamp_floor

def _init_conv_weights(module: nn.Module) -> None:
    """Shared Conv2d weight initialisation for MoA blocks.

    Uses truncated-normal (std=0.02) for weights and zeros for bias,
    matching the original per-class ``_init_weights`` methods.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def _flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                scale: float) -> torch.Tensor:
    """Scaled dot-product attention; uses F.sdpa when available (torch ≥ 2.0)."""
    sdpa = getattr(F, "scaled_dot_product_attention", None)
    if callable(sdpa):
        try:
            accepts_scale = "scale" in inspect.signature(sdpa).parameters
        except (TypeError, ValueError):
            signature_text = (getattr(sdpa, "__text_signature__", None) or getattr(sdpa, "__doc__", "") or "")
            accepts_scale = "scale" in signature_text
        if accepts_scale:
            return sdpa(q, k, v, scale=scale)
        default_scale = q.shape[-1] ** -0.5
        return sdpa(q * (scale / default_scale), k, v)
    # fallback
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)
    return attn @ v

def _window_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    window_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Window-partitioned SDPA on [B, nh, N, hd] tokens (O(N·win²) complexity)."""
    B, nh, n_tokens, hd = q.shape
    if n_tokens != height * width:
        raise ValueError(
            f"window attention token count mismatch: N={n_tokens}, expected H*W={height * width}"
        )
    win = max(1, min(int(window_size), height, width))

    def to_spatial(t: torch.Tensor) -> torch.Tensor:
        return t.transpose(2, 3).reshape(B, nh, height, width, hd)

    qs, ks, vs = to_spatial(q), to_spatial(k), to_spatial(v)
    pad_h = (win - height % win) % win
    pad_w = (win - width % win) % win
    if pad_h or pad_w:
        pad = (0, 0, 0, pad_w, 0, pad_h)
        qs, ks, vs = F.pad(qs, pad), F.pad(ks, pad), F.pad(vs, pad)
    hp, wp = qs.shape[2], qs.shape[3]

    def partition(t: torch.Tensor) -> torch.Tensor:
        t = t.view(B, nh, hp // win, win, wp // win, win, hd)
        return t.permute(0, 1, 2, 4, 3, 5, 6).reshape(-1, win * win, hd)

    def reverse(windows: torch.Tensor) -> torch.Tensor:
        n_h, n_w = hp // win, wp // win
        t = windows.view(B, nh, n_h, n_w, win, win, hd)
        t = t.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, nh, hp, wp, hd)
        return t[:, :, :height, :width, :].reshape(B, nh, height * width, hd)

    out_w = _flash_attn(partition(qs), partition(ks), partition(vs), scale)
    return reverse(out_w)

class _LocalAttnHead(nn.Module):
    """Local attention head: DW-biased QKV + window-partitioned self-attention.

    Each token attends only within a fixed ``window_size × window_size`` neighbourhood
    (Swin-style), giving true O(N·win²) local context instead of global O(N²) SDPA.
    """

    def __init__(self, dim: int, num_heads: int, head_dim: Optional[int] = None,
                 window_size: int = 7):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or max(dim // num_heads, 16)
        self.window_size = max(1, int(window_size))
        inner = self.head_dim * num_heads
        # QKV with DW-3×3 for local bias
        self.qkv_dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.qkv_pw = nn.Conv2d(dim, inner * 3, 1, bias=False)
        self.proj = nn.Conv2d(inner, dim, 1, bias=False)
        # positional encoding (DW 7×7)
        self.pe = nn.Conv2d(inner, inner, 7, padding=3, groups=inner, bias=False)
        self.norm = nn.GroupNorm(_safe_groups(dim, 8), dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        nh, hd = self.num_heads, self.head_dim

        qkv = self.qkv_pw(self.qkv_dw(x))          # [B, 3*inner, H, W]
        inner = nh * hd
        q, k, v = qkv.split(inner, dim=1)            # each [B, inner, H, W]

        # PE on v
        v = v + self.pe(v)

        # reshape to [B, nh, N, hd]
        def to_heads(t):
            return t.flatten(2).view(B, nh, hd, N).transpose(2, 3)   # [B,nh,N,hd]

        out = _window_flash_attn(
            to_heads(q), to_heads(k), to_heads(v), self.scale, self.window_size, H, W
        )
        # [B, nh, N, hd] → [B, inner, H, W]
        out = out.transpose(2, 3).reshape(B, inner, H, W)
        return self.norm(self.proj(out))

class _RegionalAttnHead(nn.Module):
    """Regional attention head: pooled keys/values (stride-2 downsampling).

    Keys and values are computed on a 2× spatially downsampled feature map,
    giving each query a larger effective receptive field at lower cost.
    O(N · N/4) = O(N²/4) vs standard O(N²).

    P0-3 / P2-6 fixes:
      - Uses adaptive_avg_pool2d (was AvgPool2d) so H,W not divisible by stride
        no longer collapses information too aggressively.
      - Explicit pool_stride validation in __init__.
      - Guard against H=1 or W=1 feature maps (produces empty KV).
    """

    def __init__(self, dim: int, num_heads: int, head_dim: Optional[int] = None,
                 pool_stride: int = 2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or max(dim // num_heads, 16)
        inner = self.head_dim * num_heads

        # P0-3 fix: validate pool_stride at construction time.
        if pool_stride < 1:
            raise ValueError(f"pool_stride must be ≥ 1, got {pool_stride}")
        self.pool_stride = pool_stride

        self.q_proj = nn.Conv2d(dim, inner, 1, bias=False)
        # P2-6 fix: use adaptive_avg_pool2d instead of AvgPool2d so that when
        # H,W are not divisible by stride the output size is floor(H/stride) which
        # preserves spatial information better than hard-cropping.
        self.kv_proj = nn.Conv2d(dim, inner * 2, 1, bias=False)
        self.proj = nn.Conv2d(inner, dim, 1, bias=False)
        self.norm = nn.GroupNorm(_safe_groups(dim, 8), dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        nh, hd = self.num_heads, self.head_dim
        inner = nh * hd

        # P0-3 fix: guard against H=1 or W=1 feature maps which would produce
        # empty spatial KV after pooling. Fall back to identity (full-res KV) in
        # that edge case.
        if min(H, W) <= 1:
            kv = self.kv_proj(x)                    # conv-only, skip pool
        else:
            target_h = max(1, H // self.pool_stride)
            target_w = max(1, W // self.pool_stride)
            pooled = F.adaptive_avg_pool2d(x, (target_h, target_w))
            kv = self.kv_proj(pooled)               # [B, 2*inner, H', W']
        H2, W2 = kv.shape[2], kv.shape[3]

        # Guard: if pooling collapsed the spatial dim to zero (extreme edge case),
        # fall back to identity KV.
        if H2 * W2 == 0:
            k = self.q_proj(x).reshape(B, nh, hd, -1).transpose(2, 3)
            v = k.clone()
        else:
            k, v = kv.split(inner, dim=1)
            k = k.flatten(2).view(B, nh, hd, H2 * W2).transpose(2, 3)
            v = v.flatten(2).view(B, nh, hd, H2 * W2).transpose(2, 3)

        q = self.q_proj(x).flatten(2).view(B, nh, hd, H * W).transpose(2, 3)

        out = _flash_attn(q, k, v, self.scale)               # [B, nh, N, hd]
        out = out.transpose(2, 3).reshape(B, inner, H, W)
        return self.norm(self.proj(out))

class _GlobalAttnHead(nn.Module):
    """Global (linear) attention head using random-feature approximation.

    Based on the Performer-style kernel trick: replaces softmax attention
    (O(N²)) with an O(N) approximation via random Fourier features.
    Suitable for large spatial maps (e.g., P3 at stride 8).

    Falls back to standard attention when N is small (N ≤ 512).  The
    threshold was raised from 256 to 512 to eliminate the discontinuity at
    N=256/257 where the two attention modes produce visibly different outputs.
    A linear-blend transition window [512−64, 512] smoothly interpolates
    between exact and linear-attention so there is no abrupt mode switch.
    """

    # Token count above which linear-attention approximation is used.
    _LINEAR_ATTN_THRESHOLD: int = LINEAR_ATTN_THRESHOLD
    # Width of the smooth transition window (avoids hard mode switch).
    _BLEND_WINDOW: int = LINEAR_ATTN_BLEND_WINDOW

    def __init__(self, dim: int, num_heads: int, head_dim: Optional[int] = None,
                 nb_features: int = 64, rf_seed: int = _DEFAULT_RF_SEED):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or max(dim // num_heads, 16)
        inner = self.head_dim * num_heads
        self.nb_features = nb_features

        self.qkv = nn.Conv2d(dim, inner * 3, 1, bias=False)
        self.proj = nn.Conv2d(inner, dim, 1, bias=False)
        self.norm = nn.GroupNorm(_safe_groups(dim, 8), dim)
        self.scale = self.head_dim ** -0.5

        # Orthogonal random features for the Performer approximation.
        # Per-block seed keeps bases diverse across layers while remaining
        # deterministic for checkpoint resume and multi-process training.
        # NOTE: QR and Generator run only in __init__ (never in forward), so
        # they are invisible to ONNX tracing / TorchScript export.  The
        # resulting matrix is stored as a persistent buffer and simply loaded
        # at export time.
        eff_nb = min(self.nb_features, self.head_dim)  # QR yields ≤ head_dim cols
        with torch.no_grad():
            gen = torch.Generator().manual_seed(rf_seed)
            rf = torch.randn(self.head_dim, self.head_dim, generator=gen, dtype=torch.float32)
            try:
                rf, _ = torch.linalg.qr(rf)        # [hd, hd] orthogonal
            except RuntimeError:
                # Fallback: Gram-Schmidt via SVD if QR fails (rare, but
                # keeps construction robust on older CUDA / MPS drivers).
                u, _, _ = torch.linalg.svd(rf, full_matrices=False)
                rf = u
        self.register_buffer("_rf_matrix", rf[:eff_nb].contiguous(), persistent=True)

    def _get_rf(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return the (fixed) random-feature matrix on the right device/dtype."""
        return self._rf_matrix.to(device=device, dtype=dtype)

    @staticmethod
    def _relu_kernel(x: torch.Tensor) -> torch.Tensor:
        """ReLU kernel feature map (non-negative, stable)."""
        # dtype-aware epsilon prevents fp16 underflow (1e-6 -> 0.0)
        eps = _fp_min(1e-6, x.dtype)
        return F.relu(x) + eps

    def _linear_attn(self, q: torch.Tensor, k: torch.Tensor,
                     v: torch.Tensor) -> torch.Tensor:
        """O(N) linear attention via kernel trick.

        q, k, v: [B, nh, N, hd]
        """
        B, nh, N, hd = q.shape
        output_dtype = q.dtype
        if output_dtype in (torch.float16, torch.bfloat16):
            q, k, v = q.float(), k.float(), v.float()
        rf = self._get_rf(q.device, q.dtype)     # [eff_nb, hd]
        eff_nb = rf.shape[0]
        scale = eff_nb ** -0.5

        # Project to feature space: [B, nh, N, eff_nb]
        # Clamp kernel features to prevent float16 overflow in AMP training.
        q_feat = self._relu_kernel(q @ rf.T * scale).clamp(max=LINEAR_ATTN_ACTIVATION_LIMIT)
        k_feat = self._relu_kernel(k @ rf.T * scale).clamp(max=LINEAR_ATTN_ACTIVATION_LIMIT)

        # Reshape to [B*nh, N, eff_nb] and [B*nh, N, hd]
        k_flat = k_feat.reshape(B * nh, N, eff_nb)
        q_flat = q_feat.reshape(B * nh, N, eff_nb)
        v_flat = v.reshape(B * nh, N, hd)

        # kv = k^T @ v → [B*nh, eff_nb, hd]
        kv = k_flat.transpose(1, 2) @ v_flat
        # L2-normalize kv accumulator to keep matmul chain stable in float16
        # fp16-safe floor: 1e-6 underflows in half precision
        kv_norm = kv.norm(dim=-1, keepdim=True).clamp(min=_fp_min(1e-6, kv.dtype))
        kv = kv / kv_norm
        # normalizer: sum of k features over N → [B*nh, eff_nb]
        # fp16-safe: accumulate in float32 (P0-2 fix) — at N=25600 (1280×1280 P3),
        # raw k_sum can reach ~2.5e8 which overflows to inf in float16.
        k_sum_f32 = k_flat.float().sum(dim=1)
        # numerator: q @ kv → [B*nh, N, hd]
        numer = (q_flat @ kv).clamp(
            min=-LINEAR_ATTN_ACTIVATION_LIMIT, max=LINEAR_ATTN_ACTIVATION_LIMIT
        )
        # denominator: q @ k_sum^T → [B*nh, N, 1]; cast k_sum back to q.dtype for matmul
        denom = (q_flat @ k_sum_f32.to(q_flat.dtype).unsqueeze(-1)).clamp(min=_fp_min(1e-6, q_flat.dtype))

        return (numer / denom).reshape(B, nh, N, hd).to(output_dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        nh, hd = self.num_heads, self.head_dim
        inner = nh * hd

        qkv = self.qkv(x).flatten(2)                        # [B, 3*inner, N]
        q, k, v = qkv.split(inner, dim=1)

        def to_heads(t):
            return t.view(B, nh, hd, N).transpose(2, 3)     # [B, nh, N, hd]

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        if N <= self._LINEAR_ATTN_THRESHOLD:
            out = _flash_attn(q, k, v, self.scale)
            # Smooth blend in the transition window to avoid discontinuity.
            blend_start = self._LINEAR_ATTN_THRESHOLD - self._BLEND_WINDOW
            if N > blend_start:
                alpha = (N - blend_start) / self._BLEND_WINDOW
                linear_out = self._linear_attn(q, k, v)
                out = (1 - alpha) * out + alpha * linear_out
        else:
            out = self._linear_attn(q, k, v)

        out = out.transpose(2, 3).reshape(B, inner, H, W)
        return self.norm(self.proj(out))

__all__ = ("_GlobalAttnHead", "_LocalAttnHead", "_RegionalAttnHead", "_flash_attn", "_window_flash_attn")
