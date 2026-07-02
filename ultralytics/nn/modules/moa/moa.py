# 🐧 YOLO-Master MoA Module — Mixture of Attention
# Copyright (C) 2026 Tencent. All rights reserved.
"""Mixture-of-Attention (MoA) for YOLO-Master.

Design philosophy
─────────────────
Unlike MoE which routes *tokens to expert FFNs*, MoA routes tokens to
*different attention heads with different receptive fields*. For dense
prediction tasks (object detection), this provides:

1. **Local heads**  – depthwise-3×3-biased QKV, captures fine texture / edge detail
2. **Regional heads** – pooled key/value (stride=2), mid-range context
3. **Global heads** – linear attention (O(N) complexity), scene-level semantics

A lightweight 1×1 conv router assigns each spatial token a *soft probability*
over these head-groups. The router output is used to weight-sum the head outputs,
maintaining end-to-end differentiability (no hard dispatch, no load-balancing
overhead for small spatial maps).

Implementation notes
────────────────────
- Fully CNN-native: input [B,C,H,W], output [B,C,H,W]. Zero seq-dim reshape needed.
- Flash-attention compatible when PyTorch >= 2.0 (uses F.scaled_dot_product_attention).
- Drop-in for A2C2f and C3k2 blocks via C2fMoA wrapper.
- For FPN/PAN neck cross-scale fusion, NeckMoAFusion provides 2-scale cross-attention.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from ultralytics.nn.modules.moe.utils import get_safe_groups as _safe_groups


def _flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                scale: float) -> torch.Tensor:
    """Scaled dot-product attention; uses F.sdpa when available (torch ≥ 2.0)."""
    if hasattr(F, "scaled_dot_product_attention"):
        try:
            return F.scaled_dot_product_attention(q, k, v, scale=scale)
        except TypeError as e:
            if "scale" not in str(e):
                raise
            default_scale = q.shape[-1] ** -0.5
            return F.scaled_dot_product_attention(q * (scale / default_scale), k, v)
    # fallback
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)
    return attn @ v


# ---------------------------------------------------------------------------
# Sub-modules: three attention head variants
# ---------------------------------------------------------------------------

class _LocalAttnHead(nn.Module):
    """Local attention head: DW-3×3 biased QKV projection.

    Each token attends only within a local neighbourhood defined by the
    positional encoding, keeping computation well-localised.
    """

    def __init__(self, dim: int, num_heads: int, head_dim: Optional[int] = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or max(dim // num_heads, 16)
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

        out = _flash_attn(to_heads(q), to_heads(k), to_heads(v), self.scale)
        # [B, nh, N, hd] → [B, inner, H, W]
        out = out.transpose(2, 3).reshape(B, inner, H, W)
        return self.norm(self.proj(out))


class _RegionalAttnHead(nn.Module):
    """Regional attention head: pooled keys/values (stride-2 downsampling).

    Keys and values are computed on a 2× spatially downsampled feature map,
    giving each query a larger effective receptive field at lower cost.
    O(N · N/4) = O(N²/4) vs standard O(N²).
    """

    def __init__(self, dim: int, num_heads: int, head_dim: Optional[int] = None,
                 pool_stride: int = 2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or max(dim // num_heads, 16)
        inner = self.head_dim * num_heads
        self.pool_stride = pool_stride

        self.q_proj = nn.Conv2d(dim, inner, 1, bias=False)
        self.kv_pool = nn.Sequential(
            nn.AvgPool2d(pool_stride, pool_stride),
            nn.Conv2d(dim, inner * 2, 1, bias=False),
        )
        self.proj = nn.Conv2d(inner, dim, 1, bias=False)
        self.norm = nn.GroupNorm(_safe_groups(dim, 8), dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        nh, hd = self.num_heads, self.head_dim
        inner = nh * hd

        q = self.q_proj(x).flatten(2).view(B, nh, hd, H * W).transpose(2, 3)

        kv = self.kv_pool(x)                                 # [B, 2*inner, H', W']
        H2, W2 = kv.shape[2], kv.shape[3]
        k, v = kv.split(inner, dim=1)
        k = k.flatten(2).view(B, nh, hd, H2 * W2).transpose(2, 3)
        v = v.flatten(2).view(B, nh, hd, H2 * W2).transpose(2, 3)

        out = _flash_attn(q, k, v, self.scale)               # [B, nh, N, hd]
        out = out.transpose(2, 3).reshape(B, inner, H, W)
        return self.norm(self.proj(out))


class _GlobalAttnHead(nn.Module):
    """Global (linear) attention head using random-feature approximation.

    Based on the Performer-style kernel trick: replaces softmax attention
    (O(N²)) with an O(N) approximation via random Fourier features.
    Suitable for large spatial maps (e.g., P3 at stride 8).

    Falls back to standard attention when N is small (N ≤ 256).
    """

    def __init__(self, dim: int, num_heads: int, head_dim: Optional[int] = None,
                 nb_features: int = 64, rf_seed: int = 0x5F3759DF):
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
        eff_nb = min(self.nb_features, self.head_dim)  # QR yields ≤ head_dim cols
        gen = torch.Generator().manual_seed(rf_seed)
        rf = torch.randn(self.head_dim, self.head_dim, generator=gen, dtype=torch.float32)
        rf, _ = torch.linalg.qr(rf)        # [hd, hd] orthogonal
        self.register_buffer("_rf_matrix", rf[:eff_nb].contiguous(), persistent=True)

    def _get_rf(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return the (fixed) random-feature matrix on the right device/dtype."""
        return self._rf_matrix.to(device=device, dtype=dtype)

    @staticmethod
    def _relu_kernel(x: torch.Tensor) -> torch.Tensor:
        """ReLU kernel feature map (non-negative, stable)."""
        return F.relu(x) + 1e-6

    def _linear_attn(self, q: torch.Tensor, k: torch.Tensor,
                     v: torch.Tensor) -> torch.Tensor:
        """O(N) linear attention via kernel trick.

        q, k, v: [B, nh, N, hd]
        """
        B, nh, N, hd = q.shape
        rf = self._get_rf(q.device, q.dtype)     # [eff_nb, hd]
        eff_nb = rf.shape[0]
        scale = eff_nb ** -0.5

        # Project to feature space: [B, nh, N, eff_nb]
        # Clamp kernel features to prevent float16 overflow in AMP training.
        q_feat = self._relu_kernel(q @ rf.T * scale).clamp(max=1e4)
        k_feat = self._relu_kernel(k @ rf.T * scale).clamp(max=1e4)

        # Reshape to [B*nh, N, eff_nb] and [B*nh, N, hd]
        k_flat = k_feat.reshape(B * nh, N, eff_nb)
        q_flat = q_feat.reshape(B * nh, N, eff_nb)
        v_flat = v.reshape(B * nh, N, hd)

        # kv = k^T @ v → [B*nh, eff_nb, hd]
        kv = k_flat.transpose(1, 2) @ v_flat
        # L2-normalize kv accumulator to keep matmul chain stable in float16
        kv_norm = kv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        kv = kv / kv_norm
        # normalizer: sum of k features over N → [B*nh, eff_nb]
        k_sum = k_flat.sum(dim=1)
        # numerator: q @ kv → [B*nh, N, hd]
        numer = (q_flat @ kv).clamp(min=-1e4, max=1e4)
        # denominator: q @ k_sum^T → [B*nh, N, 1]
        denom = (q_flat @ k_sum.unsqueeze(-1)).clamp(min=1e-6)

        return (numer / denom).reshape(B, nh, N, hd)


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

        if N <= 256:
            out = _flash_attn(q, k, v, self.scale)
        else:
            out = self._linear_attn(q, k, v)

        out = out.transpose(2, 3).reshape(B, inner, H, W)
        return self.norm(self.proj(out))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class _MoARouter(nn.Module):
    """Lightweight soft-router: assigns each spatial token a weight over M head-groups.

    Complexity: O(H·W·C_in / reduction).
    Output: [B, M, H, W] soft gate probabilities (sum-to-one over M).
    """

    def __init__(self, dim: int, num_groups: int, reduction: int = 8,
                 temperature: float = 1.0):
        super().__init__()
        self.temperature = max(temperature, 0.1)
        hidden = max(dim // reduction, num_groups * 2)
        self.router = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.GroupNorm(_safe_groups(hidden, 4), hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, num_groups, 1, bias=True),
        )
        # init: near-uniform routing
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        temp = self.temperature if self.training else 1.0
        logits = self.router(x) / temp           # [B, M, H, W]
        probs = F.softmax(logits, dim=1)
        if return_logits:
            return probs, logits
        return probs


def _moa_router_aux_loss(weights: torch.Tensor, logits: torch.Tensor, coeff: float) -> torch.Tensor:
    """GShard-scale MoA router regularization with a small z/entropy stabilizer."""
    num_groups = weights.shape[1]
    importance = weights.float().mean(dim=(0, 2, 3))
    importance = importance / importance.sum().clamp_min(1e-6)
    balance_loss = num_groups * torch.sum(importance * importance)
    z_loss = torch.logsumexp(logits.float(), dim=1).pow(2).mean()
    entropy = -(importance * torch.log(importance.clamp_min(1e-6))).sum()
    max_entropy = math.log(max(num_groups, 2))
    entropy_deficit = (max_entropy - entropy).clamp_min(0.0) / max_entropy
    # Lower entropy weight (0.01) avoids over-constraining the router toward
    # uniform mixing when balance_loss already encourages load balance.
    return coeff * (balance_loss + 0.1 * z_loss + 0.01 * entropy_deficit)


# ---------------------------------------------------------------------------
# MoABlock — core building block
# ---------------------------------------------------------------------------

class MoABlock(nn.Module):
    """Mixture-of-Attention Block.

    Routes each spatial token softly over three attention head-groups:
      - local  (DW-biased, captures fine-grained detail)
      - regional (stride-2 pooled KV, mid-range context)
      - global (linear attention, scene semantics)

    A lightweight FFN follows the attention mixture.

    Args:
        dim (int): Channel dimension (input == output).
        num_heads (int): Total attention heads, split equally over groups.
        mlp_ratio (float): FFN expansion ratio.
        temperature (float): Router softmax temperature. Starts at 1.0 and
            can be annealed toward min_temp (default 0.3) via
            ``anneal_moa_temperature`` — note this must be called from the
            training loop; it is not hooked automatically.
        attn_drop (float): Dropout on attention output.
        shortcut (bool): Residual connection around the block.

    Shape:
        - Input:  [B, dim, H, W]
        - Output: [B, dim, H, W]
    """

    NUM_GROUPS: int = 3  # local / regional / global

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        temperature: float = 1.0,
        attn_drop: float = 0.0,
        shortcut: bool = True,
        aux_loss_coeff: float = 0.01,
        block_index: int = 0,
    ):
        super().__init__()
        assert num_heads % self.NUM_GROUPS == 0, (
            f"num_heads ({num_heads}) must be divisible by NUM_GROUPS ({self.NUM_GROUPS})"
        )
        self.shortcut = shortcut
        self.aux_loss_coeff = aux_loss_coeff
        head_dim = max(dim // num_heads, 16)
        heads_per_group = num_heads // self.NUM_GROUPS

        # Three attention head-groups (global head uses a per-block RF seed).
        global_rf_seed = block_index * 7919 + 2 * 65537
        self.local_head   = _LocalAttnHead(dim, heads_per_group, head_dim)
        self.region_head  = _RegionalAttnHead(dim, heads_per_group, head_dim)
        self.global_head  = _GlobalAttnHead(dim, heads_per_group, head_dim, rf_seed=global_rf_seed)

        # Router
        self.router = _MoARouter(dim, self.NUM_GROUPS, temperature=temperature)

        # Fusion conv (combine weighted head outputs)
        self.fusion = Conv(dim, dim, 1, act=False)
        self.attn_drop = nn.Dropout2d(attn_drop) if attn_drop > 0 else nn.Identity()

        # Layer-scale (initialised at 0.1: small residual contribution for
        # stable early training, à la CaiT LayerScale).
        self.ls_attn = nn.Parameter(torch.ones(dim, 1, 1) * 0.1)

        # FFN
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            Conv(dim, hidden, 1),
            Conv(hidden, dim, 1, act=False),
        )
        self.ls_ffn = nn.Parameter(torch.ones(dim, 1, 1) * 0.1)
        self.last_aux_loss: torch.Tensor = torch.zeros((), requires_grad=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # ── Routing weights ──────────────────────────────────────────────
        weights, router_logits = self.router(x, return_logits=True)   # [B, 3, H, W]
        if self.training and self.aux_loss_coeff > 0:
            self.last_aux_loss = _moa_router_aux_loss(weights, router_logits, self.aux_loss_coeff)
        else:
            self.last_aux_loss = x.new_zeros(())

        w_l = weights[:, 0:1]     # [B, 1, H, W]
        w_r = weights[:, 1:2]
        w_g = weights[:, 2:3]

        # ── Attention head outputs ────────────────────────────────────────
        out_l = self.local_head(x)
        out_r = self.region_head(x)
        out_g = self.global_head(x)

        # ── Soft mixture ─────────────────────────────────────────────────
        mixed = w_l * out_l + w_r * out_r + w_g * out_g   # [B, C, H, W]
        mixed = self.attn_drop(self.fusion(mixed))

        # ── Residual + layer-scale ────────────────────────────────────────
        # `shortcut` controls *all* block-level residual paths consistently:
        #   True  → pre-activation residual around both attention and FFN
        #   False → pure feed-forward transform (no residual anywhere), so the
        #           block fully replaces its input rather than refining it.
        if self.shortcut:
            x = x + self.ls_attn * mixed
            x = x + self.ls_ffn * self.ffn(x)
        else:
            x = self.ls_attn * mixed
            x = self.ls_ffn * self.ffn(x)

        return x


# ---------------------------------------------------------------------------
# C2fMoA — C2f-style wrapper for backbone/neck integration
# ---------------------------------------------------------------------------

class C2fMoA(nn.Module):
    """C2f-style feature-flow wrapper around MoABlock.

    Provides the same interface as C3k2 / A2C2f, enabling direct YAML
    substitution in the backbone or neck.

    Architecture:
        cv1 (1×1 split)
        ├── identity branch  (c2 // 2 channels, pass-through)
        └── n × MoABlock     (c2 // 2 channels)
        cv2 (1×1 fusion, (n+2) × c2//2 → c2)

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of stacked MoABlock layers.
        num_heads (int): Attention heads per MoABlock.
        mlp_ratio (float): FFN expansion ratio.
        temperature (float): Initial router temperature.
        shortcut (bool): Residual inside each MoABlock.
        e (float): Channel expansion ratio for internal width.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        num_heads: int = 6,
        mlp_ratio: float = 2.0,
        temperature: float = 1.0,
        shortcut: bool = True,
        e: float = 0.5,
        aux_loss_coeff: float = 0.01,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # ensure num_heads divisible by NUM_GROUPS=3
        eff_heads = num_heads
        while eff_heads % MoABlock.NUM_GROUPS != 0:
            eff_heads += 1
        # ensure head_dim ≥ 16
        while self.c // eff_heads < 16 and eff_heads > MoABlock.NUM_GROUPS:
            eff_heads -= MoABlock.NUM_GROUPS
        eff_heads = max(eff_heads, MoABlock.NUM_GROUPS)

        self.m = nn.ModuleList(
            MoABlock(self.c, num_heads=eff_heads,
                     mlp_ratio=mlp_ratio,
                     temperature=temperature,
                     shortcut=shortcut,
                     aux_loss_coeff=aux_loss_coeff,
                     block_index=i)
            for i in range(n)
        )
        self.last_aux_loss: torch.Tensor = torch.zeros((), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))   # [identity, dynamic]
        aux_total = x.new_zeros(())
        for m in self.m:
            y.append(m(y[-1]))
            l = m.last_aux_loss
            if isinstance(l, torch.Tensor):
                aux_total = aux_total + l
        self.last_aux_loss = aux_total
        return self.cv2(torch.cat(y, dim=1))


# ---------------------------------------------------------------------------
# NeckMoAFusion — cross-scale fusion for FPN/PAN neck
# ---------------------------------------------------------------------------

class NeckMoAFusion(nn.Module):
    """Cross-scale MoA fusion for FPN/PAN neck.

    Fuses a *high-resolution* feature map (finer scale) with a
    *low-resolution* contextual feature map using bidirectional
    cross-attention, replacing simple concatenation + conv in the neck.

    Routing decision is based on content similarity between scales,
    allowing the model to learn which queries need long-range context
    (→ global head) vs. local refinement (→ local head).

    Args:
        c_hi (int): Channels of high-resolution (fine) feature map.
        c_lo (int): Channels of low-resolution (coarse) feature map.
        c_out (int): Output channels.
        num_heads (int): Cross-attention heads.
        shortcut (bool): Residual path.

    Input:
        hi : [B, c_hi, H, W]        (fine-grained, e.g. P3 or P4)
        lo : [B, c_lo, H/2, W/2]    (semantic, e.g. P4 or P5 after upsample)

    Output: [B, c_out, H, W]
    """

    def __init__(
        self,
        c_hi: int,
        c_lo: int,
        c_out: int,
        num_heads: int = 4,
        shortcut: bool = True,
        aux_loss_coeff: float = 0.01,
    ):
        super().__init__()
        self.shortcut = shortcut
        self.aux_loss_coeff = aux_loss_coeff
        head_dim = max(c_hi // num_heads, 16)
        inner = head_dim * num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # Project hi-res → Q
        self.q_proj = nn.Conv2d(c_hi, inner, 1, bias=False)
        # Project lo-res → K, V (after upsample to match hi-res)
        self.kv_proj = nn.Conv2d(c_lo, inner * 2, 1, bias=False)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear",
                                    align_corners=False)

        # Router: decides how much cross-scale vs self-scale context to blend
        self.router = _MoARouter(c_hi, num_groups=2, temperature=1.0)

        # Self-attention fallback (for spatial-only refinement path)
        self.self_attn = _LocalAttnHead(c_hi, num_heads=max(num_heads // 2, 1),
                                        head_dim=head_dim)

        self.proj = nn.Conv2d(inner, c_out, 1, bias=False)
        self.norm = nn.GroupNorm(_safe_groups(c_out, 8), c_out)

        # Separate channel projections: self-attn path vs residual shortcut
        self.self_out_proj = (nn.Conv2d(c_hi, c_out, 1, bias=False)
                              if c_hi != c_out else nn.Identity())
        self.res_proj = (nn.Conv2d(c_hi, c_out, 1, bias=False)
                         if c_hi != c_out else nn.Identity())
        self.last_aux_loss: torch.Tensor = torch.zeros((), requires_grad=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, hi: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
        B, _, H, W = hi.shape
        nh, hd = self.num_heads, self.head_dim
        inner = nh * hd

        # Align lo-res to hi-res resolution
        lo_up = F.interpolate(lo, size=(H, W), mode="bilinear", align_corners=False) if lo.shape[2:] != hi.shape[2:] else lo

        # ── Cross-attention: hi queries lo context ──────────────────────
        q = self.q_proj(hi).flatten(2).view(B, nh, hd, H * W).transpose(2, 3)
        kv = self.kv_proj(lo_up)
        k, v = kv.split(inner, dim=1)
        k = k.flatten(2).view(B, nh, hd, H * W).transpose(2, 3)
        v = v.flatten(2).view(B, nh, hd, H * W).transpose(2, 3)

        cross_out = _flash_attn(q, k, v, self.scale)            # [B,nh,N,hd]
        cross_out = cross_out.transpose(2, 3).reshape(B, inner, H, W)
        cross_out = self.norm(self.proj(cross_out))              # [B, c_out, H, W]

        # ── Self-attention path ──────────────────────────────────────────
        # Align channels if needed for self-attn
        self_out = self.self_attn(hi)                            # [B, c_hi, H, W]
        if self_out.shape[1] != cross_out.shape[1]:
            self_out = self.self_out_proj(self_out)

        # ── Router blend ─────────────────────────────────────────────────
        weights, router_logits = self.router(hi, return_logits=True)         # [B, 2, H, W]
        if self.training and self.aux_loss_coeff > 0:
            self.last_aux_loss = _moa_router_aux_loss(weights, router_logits, self.aux_loss_coeff)
        else:
            self.last_aux_loss = hi.new_zeros(())
        w_cross = weights[:, 0:1]
        w_self  = weights[:, 1:2]

        assert self_out.shape[1] == cross_out.shape[1], (
            f"NeckMoAFusion channel mismatch before blend: "
            f"self_out C={self_out.shape[1]} vs cross_out C={cross_out.shape[1]}"
        )
        out = w_cross * cross_out + w_self * self_out

        if self.shortcut:
            out = out + self.res_proj(hi)

        return out


# ---------------------------------------------------------------------------
# Utility: update router temperature (call each epoch or step)
# ---------------------------------------------------------------------------

def anneal_moa_temperature(model: nn.Module, factor: float = 0.99,
                           min_temp: float = 0.3) -> None:
    """Multiplicatively anneal router temperatures in all MoA modules.

    Call at the end of each epoch:
        anneal_moa_temperature(model, factor=0.97, min_temp=0.3)
    """
    for m in model.modules():
        if isinstance(m, _MoARouter):
            m.temperature = max(m.temperature * factor, min_temp)


def _aux_loss_device(model: nn.Module) -> torch.device:
    """Best-effort device lookup for zero aux-loss fallbacks."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def collect_moa_aux_loss(model: nn.Module) -> torch.Tensor:
    """Sum graph-connected MoA router auxiliary losses without wrapper double-counting."""
    total = None
    covered: set[int] = set()
    for m in model.modules():
        if isinstance(m, C2fMoA):
            l = m.last_aux_loss
            if isinstance(l, torch.Tensor) and l.requires_grad:
                total = l if total is None else total + l
            covered.update(id(child) for child in m.modules())
        elif isinstance(m, NeckMoAFusion):
            l = m.last_aux_loss
            if isinstance(l, torch.Tensor) and l.requires_grad:
                total = l if total is None else total + l
        elif isinstance(m, MoABlock) and id(m) not in covered:
            l = m.last_aux_loss
            if isinstance(l, torch.Tensor) and l.requires_grad:
                total = l if total is None else total + l
    return total if total is not None else torch.zeros(1, device=_aux_loss_device(model))
