#!/usr/bin/env python
"""Convert v26.08 mixture targets (MoA / MoT / MoLoRA-merged) to ncnn via pnnx.

Runs in /data/tmp/venv-v2608 (pnnx 20260526 + python ncnn 1.0.20260526) from the repo
root. Weight identity: rebuilds each architecture from cfg and loads the state_dict
that export_mixture.py --save-pt dumped, so outputs are comparable against the existing
<name>.ref.npz references.

ncnn has no TopK / Gather / Std / Where / Expression layers, and the v0_10 trunk's MoE
(moe/gated.py DualStreamGateRouter + AdaptiveGateMoE complexity gate + the fused expert
groups) emits exactly those ops under trace. This driver installs FAITHFUL SPARSE
EMULATION patches at trace time, edge-side only (the fork tree is never modified):

  - router top-k selection rebuilt from amax / clamp / ceil arithmetic with an index
    tiebreak (probs - arange(E)*1e-7: lowest index wins, matching torch.topk), producing
    FULL-WIDTH weights [B,E,1,1] with zeros at unselected experts;
  - the complexity gate reproduced exactly (round/clamp keep_count, rank-2 admission as
    floor(clamp(keep_count-1,0,1))), same renormalization sequence as the original;
  - expert dispatch rewritten as dense accumulation over all experts weighted by the
    full-width vector (4D-native, python-int indexing only): function-equal because
    zero-weight experts contribute nothing and GroupNorm is per-sample-per-expert.

The one documented deviation is the 1e-7 tiebreak epsilon (affects only exact or
near-exact routing ties). Every conversion is preceded by a full-model selftest:
patched forward vs original forward on the real weights must agree to <1e-4.
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import sys
import traceback
from collections import Counter
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# faithful sparse emulation (registered-ncnn-ops only)
# ---------------------------------------------------------------------------

def _topk_masks(pa: torch.Tensor, k: int):
    """Rank masks for the k largest entries of pa along dim 1, compare-free.

    Returns a list of k one-hot [B,E] masks (rank order). Requires entries of pa
    to be pairwise separated by more than 1e-9 (guaranteed by the index tiebreak).
    """
    masks = []
    remaining = pa
    for _ in range(k):
        m = remaining.amax(dim=1, keepdim=True)
        mk = torch.ceil((remaining - m + 1e-9).clamp(0.0, 1.0))
        masks.append(mk)
        remaining = remaining - mk * 1e30
    return masks


def _router_forward_emulated(self, x):
    """DualStreamGateRouter.forward with topk/std replaced; full-width weights out."""
    B, C, H, W = x.shape
    route_input = x.float()
    # std -> sqrt(E[x^2] - E[x]^2), identical to std(unbiased=False)
    mean = route_input.mean(dim=[2, 3])
    if H * W > 1:
        std = (route_input.pow(2).mean(dim=[2, 3]) - mean.pow(2)).clamp(min=0.0).sqrt()
    else:
        std = torch.zeros_like(mean)
    stats = torch.cat([mean, std], dim=1)
    global_logits = self.global_fc(stats)

    if H > self.pool_scale and W > self.pool_scale:
        x_local = F.avg_pool2d(route_input, kernel_size=self.pool_scale, stride=self.pool_scale)
    else:
        x_local = route_input
    local_logits = self.local_conv(x_local).mean(dim=[2, 3])

    alpha = torch.sigmoid(self.alpha)
    logits = (alpha * global_logits + (1 - alpha) * local_logits).clamp(-30.0, 30.0)
    probs = F.softmax(logits / self.temperature, dim=1)

    idxr = torch.arange(self.num_experts, dtype=probs.dtype, device=probs.device).view(1, -1)
    masks = _topk_masks(probs - idxr * 1e-7, self.top_k)
    sel = sum(masks)
    picked = probs * sel
    w_full = picked / (picked.sum(dim=1, keepdim=True) + 1e-6)   # same renorm as original

    routing_weights = w_full.to(dtype=x.dtype).view(B, self.num_experts, 1, 1)
    routing_stats = {"rank_masks": [m.view(B, self.num_experts, 1, 1) for m in masks]}
    return routing_weights, None, routing_stats


def _complexity_gate_emulated(self, routing_weights, routing_indices, routing_stats, complexity):
    """AdaptiveGateMoE._apply_complexity_gate on full-width weights.

    Reproduces: keep_count = round(top_k * clamp(c,0.3,1.5)).clamp(1, top_k); rank r is
    kept iff r <= keep_count; renormalize with clamp_min(1e-6). nan_to_num is identity
    on finite values so it is dropped.
    """
    top_k = int(self.top_k)
    if top_k <= 1:
        return routing_weights, routing_indices, routing_stats, top_k
    kc = torch.round(complexity.clamp(0.3, 1.5) * top_k).clamp(1.0, float(top_k))
    masks = routing_stats["rank_masks"]
    mask = masks[0]
    for r in range(1, top_k):
        admission = torch.floor((kc - float(r)).clamp(0.0, 1.0))   # 1 iff keep_count >= r+1
        mask = mask + masks[r] * admission
    w = routing_weights * mask
    w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return w, routing_indices, routing_stats, top_k


def _fused_group_forward_emulated(self, x, routing_weights, routing_indices, top_k):
    """FusedExpertGroup.forward, dense over all experts, no gather/index, 4D only."""
    B = x.shape[0]
    E, OC = self.num_experts, self.out_channels
    fused = self.fused_conv(x)                                   # [B, E*OC, H, W]
    w = routing_weights.view(B, E, 1, 1)
    out = None
    for e in range(E):
        fe = fused[:, e * OC:(e + 1) * OC]
        ne = F.group_norm(fe, self.norm_groups, None, None, self.norm_eps)
        ne = ne * self.expert_norm_weight[e].view(1, OC, 1, 1) + self.expert_norm_bias[e].view(1, OC, 1, 1)
        ne = self.activation(ne)
        term = ne * w[:, e:e + 1]
        out = term if out is None else out + term
    return out


def _shared_group_forward_emulated(self, x, routing_weights, routing_indices, top_k):
    """SharedInvertedExpertGroup.forward, dense weighted sum, no stack/gather.

    valid_mask (weights > 0.0) is dropped exactly: softmax-derived selected weights are
    strictly positive and unselected experts already carry weight zero.
    """
    B = x.shape[0]
    features = self.shared_feature(x)
    w = routing_weights.view(B, self.num_experts, 1, 1)
    out = None
    for e, proj in enumerate(self.expert_projections):
        term = proj(features) * w[:, e:e + 1]
        out = term if out is None else out + term
    return out


def _moa_window_attn_ncnn(q, k, v, scale, window_size, height, width):
    """moa/heads._window_flash_attn with windows folded into the HEADS dim.

    The original partitions to [B*nh*nWindows, win*win, hd]: the window count lands in
    the leading dim, which pnnx maps onto ncnn's nonexistent batch axis and the runtime
    dies (SIGFPE in reshape arithmetic). Windows never attend across each other, so
    each window can be its own attention head: [B, nh*nW, win*win, hd] is semantically
    identical and ncnn's SDPA represents heads natively as channels. All reshapes stay
    rank<=5 with batch kept at 1.
    """
    B, nh, n_tokens, hd = q.shape
    win = max(1, min(int(window_size), height, width))
    qs = q.reshape(B, nh, height, width, hd)
    ks = k.reshape(B, nh, height, width, hd)
    vs = v.reshape(B, nh, height, width, hd)
    pad_h = (win - height % win) % win
    pad_w = (win - width % win) % win
    if pad_h or pad_w:
        pad = (0, 0, 0, pad_w, 0, pad_h)
        qs, ks, vs = F.pad(qs, pad), F.pad(ks, pad), F.pad(vs, pad)
    hp, wp = qs.shape[2], qs.shape[3]
    nH, nW = hp // win, wp // win

    # every intermediate keeps dim0 == B == 1: pnnx folds any >1 leading dim onto
    # ncnn's nonexistent batch axis and silently drops the factor (runtime SIGFPE)
    def part(t):
        t = t.reshape(B, nh * nH, win, wp, hd)          # split hp, merge nh*nH
        t = t.permute(0, 1, 3, 2, 4)                    # [1, g, wp, win_h, hd]
        t = t.reshape(B, nh * nH * nW, win, win, hd)    # split wp, merge g*nW -> [1,G,win_w,win_h,hd]
        t = t.permute(0, 1, 3, 2, 4)                    # [1, G, win_h, win_w, hd]
        return t.reshape(B, nh * nH * nW, win * win, hd)

    out = F.scaled_dot_product_attention(part(qs), part(ks), part(vs), scale=scale)
    t = out.reshape(B, nh * nH * nW, win, win, hd)      # [1, G, win_h, win_w, hd]
    t = t.permute(0, 1, 3, 2, 4)                        # [1, G, win_w, win_h, hd]
    t = t.reshape(B, nh * nH, wp, win, hd)              # merge nW*win_w -> wp
    t = t.permute(0, 1, 3, 2, 4)                        # [1, g, win_h, wp, hd]
    t = t.reshape(B, nh, hp, wp, hd)
    return t[:, :, :height, :width, :].reshape(B, nh, height * width, hd)


def _mot_windows_to_heads(t, win, nh, hd):
    """NHWC [1,Hp,Wp,C] -> [1, nWindows*nh, win*win, hd]; every intermediate keeps dim0=1."""
    B, Hp, Wp, C = t.shape
    nH, nW = Hp // win, Wp // win
    t = t.reshape(B, nH, win, Wp, C).permute(0, 1, 3, 2, 4)
    t = t.reshape(B, nH * nW, win, win, C).permute(0, 1, 3, 2, 4)
    t = t.reshape(B, nH * nW, win * win, C)
    t = t.reshape(B, nH * nW, win * win, nh, hd).permute(0, 1, 3, 2, 4)
    return t.reshape(B, nH * nW * nh, win * win, hd)


def _mot_heads_to_windows(t, win, nh, hd, Hp, Wp, C):
    """Inverse of _mot_windows_to_heads -> NHWC [1,Hp,Wp,C]."""
    B = t.shape[0]
    nH, nW = Hp // win, Wp // win
    t = t.reshape(B, nH * nW, nh, win * win, hd).permute(0, 1, 3, 2, 4)
    t = t.reshape(B, nH * nW, win * win, C)
    t = t.reshape(B, nH * nW, win, win, C).permute(0, 1, 3, 2, 4)
    t = t.reshape(B, nH, Wp, win, C).permute(0, 1, 3, 2, 4)
    return t.reshape(B, Hp, Wp, C)


def _mot_window_expert_forward_ncnn(self, x):
    """_WindowTransformerExpert.forward with windows folded into SDPA heads and the
    cyclic shift via slicing. Faithful replica of the original otherwise."""
    import ultralytics.nn.modules.mot.experts as mot_experts
    B, C, H_orig, W_orig = x.shape
    win = self.win
    x = x.permute(0, 2, 3, 1)
    x, pad_h, pad_w = self._pad_to_window(x, win)
    H, W = x.shape[1], x.shape[2]
    shift = self.shift_size
    if shift > 0:
        x = mot_experts._roll_via_cat(x, -shift, dims=(1, 2))
    xn = self.norm1(x)
    nh, hd = self.num_heads, self.head_dim
    qkv = self.qkv(xn)                                   # pointwise Linear == per-window qkv
    q, k, v = qkv.split(C, dim=-1)
    out = F.scaled_dot_product_attention(
        _mot_windows_to_heads(q, win, nh, hd),
        _mot_windows_to_heads(k, win, nh, hd),
        _mot_windows_to_heads(v, win, nh, hd), scale=self.scale)
    attn_out = _mot_heads_to_windows(out, win, nh, hd, H, W, C)
    attn_out = self.drop(self.proj(attn_out))
    if shift > 0:
        attn_out = mot_experts._roll_via_cat(attn_out, shift, dims=(1, 2))
        x = mot_experts._roll_via_cat(x, shift, dims=(1, 2))
    attn_out = attn_out[:, :H_orig, :W_orig, :]
    x = x[:, :H_orig, :W_orig, :]
    x = x + self.ls1 * attn_out
    x = x + self.ls2 * self.ffn(self.norm2(x))
    return x.permute(0, 3, 1, 2).contiguous()


def _mot_localconv_forward_ncnn(self, x):
    """_LocalConvTransformerExpert.forward with the windowed branch heads-folded."""
    import ultralytics.nn.modules.mot.experts as mot_experts
    WTE = mot_experts._WindowTransformerExpert
    B, C, H, W = x.shape
    N = H * W
    nh, hd = self.num_heads, self.head_dim
    xn = self.norm1(x)
    qkv = self.qkv(self.dw_mix(xn)).flatten(2)
    q, k, v = qkv.split(C, dim=1)
    v_2d = v.reshape(B, C, H, W)
    v_2d = v_2d + self.pe(v_2d)
    v = v_2d.flatten(2)
    if self.local_window_size > 0 and N > self.local_window_size ** 2:
        win = self.local_window_size

        def nhwc(t):
            return t.reshape(B, C, H, W).permute(0, 2, 3, 1)

        q_n, _, _ = WTE._pad_to_window(nhwc(q), win)
        k_n, _, _ = WTE._pad_to_window(nhwc(k), win)
        v_n, _, _ = WTE._pad_to_window(nhwc(v), win)
        Hp, Wp = q_n.shape[1], q_n.shape[2]
        o = F.scaled_dot_product_attention(
            _mot_windows_to_heads(q_n, win, nh, hd),
            _mot_windows_to_heads(k_n, win, nh, hd),
            _mot_windows_to_heads(v_n, win, nh, hd), scale=self.scale)
        out = _mot_heads_to_windows(o, win, nh, hd, Hp, Wp, C)
        out = out[:, :H, :W, :].permute(0, 3, 1, 2).contiguous()
    else:
        def to_heads(t):
            return t.view(B, nh, hd, N).transpose(2, 3)
        out = F.scaled_dot_product_attention(to_heads(q), to_heads(k), to_heads(v), scale=self.scale)
        out = out.transpose(2, 3).reshape(B, C, H, W)
    x = x + self.ls1 * self.drop(self.proj(out))
    xn2 = self.norm2(x)
    ffn = self.ffn_gate(xn2) * self.ffn_val(xn2)
    x = x + self.ls2 * self.ffn_out(ffn)
    return x


def _mot_deform_attn_ncnn(self, q, value, H, W):
    """_DeformableTransformerExpert._deform_attn with grid_sample unrolled per head:
    the original samples on [B*nh, hd, H, W], and pnnx folds the nh factor onto the
    nonexistent ncnn batch axis. One GridSample per head keeps dim0 == 1."""
    B, N, C = q.shape
    nh, np_, hd = self.num_heads, self.n_points, self.head_dim
    if N != H * W:
        value_proj = self.v_proj(value)
        q_heads = q.reshape(B, N, nh, hd).transpose(1, 2)
        value_heads = value_proj.reshape(B, value_proj.shape[1], nh, hd).transpose(1, 2)
        dense = F.scaled_dot_product_attention(q_heads, value_heads, value_heads)
        return self.out_proj(dense.transpose(1, 2).reshape(B, N, C))
    offsets = self.offset_proj(q).reshape(B, N, nh, np_, 2).tanh()
    attn_w = F.softmax(self.attn_proj(q).reshape(B, N, nh, np_), dim=-1)
    idx = torch.arange(N, device=q.device)
    row = (idx // W).float() / max(H - 1, 1) * 2 - 1
    col = (idx % W).float() / max(W - 1, 1) * 2 - 1
    ref = torch.stack([col, row], dim=-1)[None, :, None, None, :]     # constant-folds at trace
    sample_locs = (ref + offsets * 0.25).clamp(-1.0, 1.0)             # [1, N, nh, np, 2]
    v_4d = self.v_proj(value).permute(0, 2, 1).reshape(B, C, H, W)
    outs = []
    for h in range(nh):
        s = F.grid_sample(v_4d[:, h * hd:(h + 1) * hd], sample_locs[:, :, h],
                          mode="bilinear", align_corners=self.align_corners,
                          padding_mode="zeros")                       # [1, hd, N, np]
        o_h = (s * attn_w[:, :, h].unsqueeze(1)).sum(dim=3)           # [1, hd, N]
        outs.append(o_h)
    out = torch.cat(outs, dim=1).permute(0, 2, 1)                     # [1, N, C], C == (nh, hd)
    return self.out_proj(out)


def _mixer_forward_emulated(self, x):
    """PyramidContextMixer.forward without torch.stack (pnnx lowers stack to a dim-0
    Concat and IGNORES the axis param - wrong semantics at runtime) and with the 0-d
    tanh scalar folded."""
    from ultralytics.nn.modules.moe.gated import _pool_to_size_mps_safe
    B, C, H, W = x.shape
    contexts = [self.local_context(x)]
    for scale, proj in zip(self.pool_scales, self.pool_projections):
        h = max(1, H // scale)
        w = max(1, W // scale)
        pooled = _pool_to_size_mps_safe(x, (h, w))
        contexts.append(F.interpolate(proj(pooled), size=(H, W), mode="nearest"))
    context = contexts[0]
    for c in contexts[1:]:
        context = context + c
    context = context / float(len(contexts))
    return x + float(self.context_scale.tanh()) * context * self.context_gate(context)


def _refine_features_folded(self, x):
    return x + float(self.refine_scale.tanh()) * self.feature_refiner(x) * self.feature_gate(x)


def _hook_apply_folded(self, module, stage, value):
    if stage == "post_fusion":
        return value + float(module.refine_scale.tanh()) * module.feature_refiner(value) * module.feature_gate(value)
    return value


@contextlib.contextmanager
def emulation_patches():
    """Install the trace patches on the moe/moa/mot classes; restore on exit."""
    from ultralytics.nn.modules.moe import gated, experts, hooks
    from ultralytics.nn.modules.moa import heads as moa_heads
    from ultralytics.nn.modules.mot import experts as mot_experts
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from export_coreml_mixture import (_moa_partition_r5, _moa_unpartition_r5,
                                       _mot_partition_r5, _mot_reverse_r5, _roll_via_slice)

    saved = [
        (gated.PyramidContextMixer, "forward", gated.PyramidContextMixer.forward),
        (moa_heads, "_window_flash_attn", moa_heads._window_flash_attn),
        (moa_heads, "_window_partition_2d", moa_heads._window_partition_2d),
        (moa_heads, "_window_unpartition_2d", moa_heads._window_unpartition_2d),
        (mot_experts._WindowTransformerExpert, "_window_partition",
         mot_experts._WindowTransformerExpert.__dict__["_window_partition"]),
        (mot_experts._WindowTransformerExpert, "_window_reverse",
         mot_experts._WindowTransformerExpert.__dict__["_window_reverse"]),
        (mot_experts, "_roll_via_cat", mot_experts._roll_via_cat),
        (mot_experts._WindowTransformerExpert, "forward", mot_experts._WindowTransformerExpert.forward),
        (mot_experts._LocalConvTransformerExpert, "forward", mot_experts._LocalConvTransformerExpert.forward),
        (mot_experts._DeformableTransformerExpert, "_deform_attn",
         mot_experts._DeformableTransformerExpert._deform_attn),
        (gated.DualStreamGateRouter, "forward", gated.DualStreamGateRouter.forward),
        (gated.AdaptiveGateMoE, "_apply_complexity_gate", gated.AdaptiveGateMoE._apply_complexity_gate),
        (gated.FusedExpertGroup, "forward", gated.FusedExpertGroup.forward),
        (experts.SharedInvertedExpertGroup, "forward", experts.SharedInvertedExpertGroup.forward),
        (hooks.FeatureRefinementHook, "apply", hooks.FeatureRefinementHook.apply),
    ]
    refine_owner = None
    for cls in vars(gated).values():
        if isinstance(cls, type) and "_refine_features" in vars(cls):
            refine_owner = cls
            saved.append((cls, "_refine_features", vars(cls)["_refine_features"]))
    gated.PyramidContextMixer.forward = _mixer_forward_emulated
    moa_heads._window_flash_attn = _moa_window_attn_ncnn
    moa_heads._window_partition_2d = _moa_partition_r5
    moa_heads._window_unpartition_2d = _moa_unpartition_r5
    mot_experts._WindowTransformerExpert._window_partition = staticmethod(_mot_partition_r5)
    mot_experts._WindowTransformerExpert._window_reverse = staticmethod(_mot_reverse_r5)
    mot_experts._roll_via_cat = _roll_via_slice          # aten::narrow has no pnnx/ncnn lowering
    mot_experts._WindowTransformerExpert.forward = _mot_window_expert_forward_ncnn
    mot_experts._LocalConvTransformerExpert.forward = _mot_localconv_forward_ncnn
    mot_experts._DeformableTransformerExpert._deform_attn = _mot_deform_attn_ncnn
    gated.DualStreamGateRouter.forward = _router_forward_emulated
    gated.AdaptiveGateMoE._apply_complexity_gate = _complexity_gate_emulated
    gated.FusedExpertGroup.forward = _fused_group_forward_emulated
    experts.SharedInvertedExpertGroup.forward = _shared_group_forward_emulated
    hooks.FeatureRefinementHook.apply = _hook_apply_folded
    if refine_owner is not None:
        refine_owner._refine_features = _refine_features_folded
    try:
        yield
    finally:
        for cls, name, fn in saved:
            setattr(cls, name, fn)


def selftest_topk_emulation():
    """Emulated selection == torch.topk on distinct probs; deterministic lowest-index
    order under exact ties (torch.topk's tie order is implementation-arbitrary, so no
    arithmetic emulation can match it there; the full-model selftest on real weights is
    the binding gate)."""
    torch.manual_seed(3)
    for E, k in ((4, 2), (8, 2), (16, 2), (6, 3)):
        probs = F.softmax(torch.randn(5, E), dim=1)
        idxr = torch.arange(E, dtype=probs.dtype).view(1, -1)
        masks = _topk_masks(probs - idxr * 1e-7, k)
        _, ti = torch.topk(probs, k, dim=1)
        for r in range(k):
            assert torch.equal(masks[r].argmax(dim=1), ti[:, r]), f"rank {r} mismatch E={E}"
            assert masks[r].sum(dim=1).eq(1).all(), "rank mask not one-hot"
        # exact-tie determinism: lowest indices win, ranks stay one-hot
        tied = torch.full((3, E), 1.0 / E)
        tmasks = _topk_masks(tied - idxr * 1e-7, k)
        for r in range(k):
            assert tmasks[r].argmax(dim=1).eq(r).all(), "tie order not lowest-index"
            assert tmasks[r].sum(dim=1).eq(1).all()
    print("[selftest] topk emulation matches torch.topk on distinct probs; lowest-index on ties")


class _DetOnly(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        y = self.m(x)
        while not torch.is_tensor(y):
            y = y[0]
        return y


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True, help="YOLO-Master v26.08 checkout (prepended to sys.path)")
    ap.add_argument("--out", default="models/mixture")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--only", default="")
    ap.add_argument("--tol", type=float, default=5e-4, help="patched-vs-original full-model maxdiff gate")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    import pnnx
    import ncnn as pyncnn
    from ultralytics import YOLO, __version__

    print(f"[env] ultralytics {__version__} torch {torch.__version__} ncnn {pyncnn.__version__}")
    assert not __version__.startswith("8.3."), "old-lineage ultralytics resolved - wrong sys.path"

    selftest_topk_emulation()

    out = Path(args.out)
    V010 = "ultralytics/cfg/models/master/v0_10/det"
    Y26 = "ultralytics/cfg/models/26"

    def build(cfg):
        return YOLO(str(Path(args.repo) / cfg))

    def build_molora_merged():
        from ultralytics.nn.peft.molora import MoLoRAConfig, MoLoRAModel
        y = build(f"{V010}/yolo-master-moa-n.yaml")
        mol = MoLoRAModel(y.model, MoLoRAConfig(r=8, alpha=16, num_experts=4, top_k=2))
        mol.merge(mode="uniform")
        y.model = mol.model
        return y

    # molora-routed excluded by decision (MoLoRA dispatch topk; merged is the answer);
    # yolo26 e2e is N/A on ncnn (upstream exporter itself disables end2end for ncnn).
    TARGETS = {
        "moa-n":                (lambda: build(f"{V010}/yolo-master-moa-n.yaml"), None),
        "mot-n":                (lambda: build(f"{V010}/yolo-master-mot-n.yaml"), None),
        "moa-mot-n":            (lambda: build(f"{V010}/yolo-master-moa-mot-n.yaml"), None),
        "yolo26-moa-n-anchors": (lambda: build(f"{Y26}/yolo26-master-moa-n.yaml"), False),
        "molora-merged":        (build_molora_merged, None),
    }
    only = {s for s in args.only.split(",") if s} or set(TARGETS)

    results = {}
    for name, (factory, e2e_flag) in TARGETS.items():
        if name not in only:
            continue
        print(f"\n=== {name} ===")
        try:
            y = factory()
            sd = torch.load(out / f"{name}.pt", map_location="cpu", weights_only=True)
            y.model.load_state_dict(sd, strict=True)
            model = y.model.eval()
            if e2e_flag is False:
                model.model[-1].end2end = False
            for p in model.parameters():
                p.requires_grad_(False)

            wrapped = _DetOnly(model)
            ex = torch.rand(1, 3, args.imgsz, args.imgsz)
            with torch.no_grad():
                ref = wrapped(ex)
                with emulation_patches():
                    got = wrapped(ex)
            diff = float((ref - got).abs().max())
            print(f"  full-model patched-vs-original maxdiff {diff:.2e} (gate {args.tol:.0e})")
            if diff > args.tol:
                raise RuntimeError(f"emulation selftest failed: maxdiff {diff:.2e}")

            dst = out / f"{name}_ncnn"
            dst.mkdir(parents=True, exist_ok=True)
            with emulation_patches(), \
                 mock.patch("torch.nan_to_num", lambda x, nan=0.0, posinf=None, neginf=None, out=None: x):
                pnnx.export(
                    wrapped,
                    str(dst / "model.pt"),
                    inputs=ex,
                    fp16=False,
                    device="cpu",
                    ncnnparam=str(dst / "model.ncnn.param"),
                    ncnnbin=str(dst / "model.ncnn.bin"),
                    ncnnpy=str(dst / "model_ncnn.py"),
                    pnnxparam=str(dst / "model.pnnx.param"),
                    pnnxbin=str(dst / "model.pnnx.bin"),
                    pnnxpy=str(dst / "model_pnnx.py"),
                    pnnxonnx=str(dst / "model.pnnx.onnx"),
                )
            for junk in ("model.pt", "model.pnnx.param", "model.pnnx.bin", "model_pnnx.py",
                         "model.pnnx.onnx", "debug.param", "debug.bin", "debug2.param", "debug2.bin"):
                (dst / junk).unlink(missing_ok=True)
            for junk in Path.cwd().glob("debug*.param"):
                junk.unlink(missing_ok=True)
            for junk in Path.cwd().glob("debug*.bin"):
                junk.unlink(missing_ok=True)

            # sidecar (read by the C++ runtime's read_ncnn_yaml)
            names = y.model.names if isinstance(y.model.names, dict) else dict(enumerate(y.model.names))
            with open(dst / "metadata.yaml", "w") as f:
                f.write(f"imgsz:\n- {args.imgsz}\n- {args.imgsz}\n")
                f.write("end2end: false\n")
                # Corroboration for the runtime's per-model precision policy: every export from this
                # script carries the emulated-router constants (1e-9 mask nudge, 1e30 expert mask)
                # that are unrepresentable in fp16, so the CPU path must stay fp32. The runtime scans
                # the .param itself (meta::scan_ncnn_param) and treats "any source says unsafe" as
                # unsafe; this key only makes the intent visible in the sidecar.
                f.write("fp16_safe: false\n")
                f.write("names:\n")
                for k in sorted(names):
                    f.write(f"  {k}: {names[k]}\n")

            # census: every layer type must be registered (or a known builtin token)
            types = Counter(l.split()[0] for l in (dst / "model.ncnn.param").read_text().splitlines()[2:] if l.strip())
            unregistered = {t: n for t, n in types.items()
                            if pyncnn.layer_to_index(t) == -1 and t not in ("Input",)}
            top = ", ".join(f"{t}:{n}" for t, n in types.most_common(8))
            print(f"  param: {sum(types.values())} layers, {len(types)} types | top: {top}")
            if unregistered:
                raise RuntimeError(f"unregistered ncnn layer types survived: {unregistered}")

            net = pyncnn.Net()
            rc_p = net.load_param(str(dst / "model.ncnn.param"))
            rc_b = net.load_model(str(dst / "model.ncnn.bin"))
            if rc_p != 0 or rc_b != 0:
                raise RuntimeError(f"ncnn load failed (param={rc_p}, bin={rc_b})")
            print(f"  saved {dst.name}  (loads clean, zero unregistered types)")
            results[name] = f"OK  selftest {diff:.1e}  layers={sum(types.values())}"
        except Exception as e:
            traceback.print_exc()
            results[name] = f"FAILED: {type(e).__name__}: {str(e)[:160]}"

    print("\n===== ncnn export summary =====")
    for name, r in results.items():
        print(f"  {name:24s} {r}")
    if any(r.startswith("FAILED") for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
