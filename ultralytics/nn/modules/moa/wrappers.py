"""YAML-facing wrappers and collection helpers for Mixture-of-Attention."""
from __future__ import annotations
from collections import OrderedDict
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import weakref
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules._numeric import should_reduce_ddp
from ultralytics.nn.modules.utils import get_safe_groups as _safe_groups, robust_deepcopy
from ultralytics.nn.modules.routing_protocol import collect_aux_loss, export_capabilities as _export_routing_capabilities, publish_aux_loss, routing_snapshot as _routing_snapshot
from .block import MoABlock
from .heads import _LocalAttnHead, _flash_attn, _init_conv_weights
from .router import _MoARouter, _moa_router_aux_loss

_MOA_TOPOLOGY_CACHE: weakref.WeakKeyDictionary[nn.Module, tuple[weakref.ReferenceType[nn.Module], ...]] = weakref.WeakKeyDictionary()


def _cached_moa_topology(model: nn.Module) -> tuple[nn.Module, ...]:
    """Cache weak module references so repeated aux collection avoids tree walks."""
    refs = _MOA_TOPOLOGY_CACHE.get(model)
    if refs is None:
        refs = tuple(weakref.ref(module) for module in model.modules())
        _MOA_TOPOLOGY_CACHE[model] = refs
    live = tuple(module for ref in refs if (module := ref()) is not None)
    return live

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
        local_window_size: int = 7,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # ensure num_heads divisible by NUM_GROUPS=3 (P1-6: add warnings for user visibility)
        eff_heads = num_heads
        _max_iter = 256
        while eff_heads % MoABlock.NUM_GROUPS != 0 and _max_iter > 0:
            eff_heads += 1
            _max_iter -= 1
        if eff_heads != num_heads:
            warnings.warn(
                f"C2fMoA(num_heads={num_heads}) adjusted to {eff_heads} "
                f"to be divisible by NUM_GROUPS={MoABlock.NUM_GROUPS}. "
                f"To avoid silent adjustment, set num_heads ≡ 0 (mod 3) directly.",
                stacklevel=2,
            )
        divisible_heads = eff_heads
        # ensure head_dim ≥ 16
        _max_iter = 256
        while self.c // eff_heads < 16 and eff_heads > MoABlock.NUM_GROUPS and _max_iter > 0:
            eff_heads -= MoABlock.NUM_GROUPS
            _max_iter -= 1
        if eff_heads != divisible_heads:
            warnings.warn(
                f"C2fMoA(num_heads={divisible_heads}) adjusted to {eff_heads} "
                f"to maintain head_dim ≥ 16 (c//heads ≥ 16).",
                stacklevel=2,
            )
        eff_heads = max(eff_heads, MoABlock.NUM_GROUPS)

        self.m = nn.ModuleList(
            MoABlock(self.c, num_heads=eff_heads,
                     mlp_ratio=mlp_ratio,
                     temperature=temperature,
                     shortcut=shortcut,
                     aux_loss_coeff=aux_loss_coeff,
                     block_index=i,
                     local_window_size=local_window_size)
            for i in range(n)
        )
        self.last_aux_loss: torch.Tensor = torch.zeros((), requires_grad=False)
        self.last_routing_snapshot: dict = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))   # [identity, dynamic]
        aux_total = x.new_zeros(())
        for m in self.m:
            y.append(m(y[-1]))
            aux_loss = m.last_aux_loss
            if isinstance(aux_loss, torch.Tensor):
                aux_total = aux_total + aux_loss
        self.last_aux_loss = aux_total
        publish_aux_loss(self, aux_total, kind="moa", training=self.training, covered_modules=self.m)

        # ── Routing snapshot (aggregated from child MoABlocks) ──────────
        with torch.no_grad():
            child_snaps = [getattr(m, "last_routing_snapshot", {}) for m in self.m]
            if child_snaps:
                usages = [s["expert_usage"] for s in child_snaps if "expert_usage" in s]
                mean_usage = sum(usages) / len(usages) if usages else x.new_zeros(3)
                self.last_routing_snapshot = {
                    "num_experts": MoABlock.NUM_GROUPS,
                    "top_k": MoABlock.NUM_GROUPS,
                    "expert_usage": mean_usage,
                    "mean_router_probs": mean_usage,
                    "aux_loss": float(aux_total.detach()),
                }
            else:
                self.last_routing_snapshot = {}

        return self.cv2(torch.cat(y, dim=1))

    # ── RoutedModule protocol ───────────────────────────────────────────
    @property
    def num_experts(self) -> int:
        return MoABlock.NUM_GROUPS

    @property
    def top_k(self) -> int:
        return MoABlock.NUM_GROUPS

    @property
    def aux_loss(self) -> torch.Tensor:
        return self.last_aux_loss

    def publish_aux_loss(self, *, step: int, training: bool) -> torch.Tensor:
        return publish_aux_loss(
            self, self.last_aux_loss, step=step, kind="moa", training=training, covered_modules=self.m
        )

    def routing_snapshot(self) -> dict:
        return _routing_snapshot(self)

    def export_capabilities(self) -> dict:
        return _export_routing_capabilities(self)

    def __deepcopy__(self, memo):
        return robust_deepcopy(self, memo)

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
        self.last_routing_snapshot: dict = {}
        self._lo_interpolate_cache: OrderedDict[
            tuple, tuple[weakref.ReferenceType, torch.Tensor]
        ] = OrderedDict()

        self._init_weights()

    def _init_weights(self):
        _init_conv_weights(self)

    def forward(self, hi: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
        B, _, H, W = hi.shape
        nh, hd = self.num_heads, self.head_dim
        inner = nh * hd

        # Align lo-res to hi-res resolution
        lo_up = self._align_low_resolution(lo, hi, H, W)

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
            self.last_aux_loss = _moa_router_aux_loss(
                weights, router_logits, self.aux_loss_coeff, reduce_ddp=should_reduce_ddp(self)
            )
        else:
            self.last_aux_loss = hi.new_zeros(())
        publish_aux_loss(self, self.last_aux_loss, kind="moa", training=self.training)
        w_cross = weights[:, 0:1]
        w_self  = weights[:, 1:2]

        # ── Routing snapshot ─────────────────────────────────────────────
        with torch.no_grad():
            mean_w = weights.detach().float().mean(dim=(0, 2, 3))  # [2]
            self.last_routing_snapshot = {
                "num_experts": 2,
                "top_k": 2,
                "expert_usage": mean_w,
                "mean_router_probs": mean_w,
                "aux_loss": float(self.last_aux_loss.detach()),
            }

        if self_out.shape[1] != cross_out.shape[1]:
            raise RuntimeError(
                "NeckMoAFusion channel mismatch before blend: "
                f"self_out C={self_out.shape[1]} vs cross_out C={cross_out.shape[1]}"
            )
        out = w_cross * cross_out + w_self * self_out

        if self.shortcut:
            out = out + self.res_proj(hi)

        return out

    def _align_low_resolution(self, lo: torch.Tensor, hi: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Cache only detached eval interpolation; never retain training/export graphs."""
        cacheable = (
            not self.training
            and not torch.is_grad_enabled()
            and not torch.jit.is_tracing()
            and not torch.onnx.is_in_onnx_export()
        )
        if not cacheable:
            self._lo_interpolate_cache.clear()
        if lo.shape[2:] == (height, width):
            return lo
        key = (id(lo), getattr(lo, "_version", 0), tuple(lo.shape), height, width, lo.dtype, lo.device)
        if cacheable and key in self._lo_interpolate_cache:
            source_ref, cached = self._lo_interpolate_cache[key]
            if source_ref() is lo:
                self._lo_interpolate_cache.move_to_end(key)
                return cached
            del self._lo_interpolate_cache[key]
        aligned = F.interpolate(lo, size=(height, width), mode="bilinear", align_corners=False)
        if cacheable:
            self._lo_interpolate_cache[key] = (weakref.ref(lo), aligned)
            while len(self._lo_interpolate_cache) > 4:
                self._lo_interpolate_cache.popitem(last=False)
        return aligned

    # ── RoutedModule protocol ───────────────────────────────────────────
    @property
    def num_experts(self) -> int:
        return 2

    @property
    def top_k(self) -> int:
        return 2

    @property
    def aux_loss(self) -> torch.Tensor:
        return self.last_aux_loss

    def publish_aux_loss(self, *, step: int, training: bool) -> torch.Tensor:
        return publish_aux_loss(self, self.last_aux_loss, step=step, kind="moa", training=training)

    def routing_snapshot(self) -> dict:
        return _routing_snapshot(self)

    def export_capabilities(self) -> dict:
        return _export_routing_capabilities(self)

    def __deepcopy__(self, memo):
        clone = robust_deepcopy(self, memo)
        clone._lo_interpolate_cache = OrderedDict()
        return clone

def _aux_loss_device(model: nn.Module) -> torch.device:
    """Best-effort device lookup for zero aux-loss fallbacks."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def collect_moa_aux_loss(model: nn.Module) -> torch.Tensor:
    """Sum graph-connected MoA router auxiliary losses without wrapper double-counting.

    P2-2 fix: NeckMoAFusion now also updates the ``covered`` set so that if it
    nests any MoABlock children they are not double-counted when collected later
    in the iteration order.  This mirrors the pattern used by C2fMoA and makes the
    function robust against future nested architectures.
    """
    modules = _cached_moa_topology(model)
    return collect_aux_loss(model, include_kinds=("moa",), device=_aux_loss_device(model), modules=modules)

__all__ = ("C2fMoA", "NeckMoAFusion", "collect_moa_aux_loss")
