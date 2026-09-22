#!/usr/bin/env python
"""Export YOLO-Master DENSE checkpoints (v0.1-N det, v0.1-seg-N) to ncnn without the glue layers.

Runs in /data/tmp/venv-v2608 (pnnx 20260526 + python ncnn 1.0.20260526) with
PYTHONPATH=/data/YOLO-Master (the fork, ultralytics 8.4.101), from the repo root.

The stock ultralytics `format=ncnn` export of these models is 1.8x (x86) / 3-4x (phone) slower
than YOLO11 on ncnn although the FLOPs are comparable. The cost is not in the convs but in the
glue that pnnx has to emit for two modules whose tensor algebra it cannot pattern-match:

  * `AAttn.forward` (ultralytics/nn/modules/block.py:1705-1732): qkv is kept dim-major
    (B, heads, 3*hd, N), the QUERY is transposed and scaled (`(q*scale).T @ k`), then
    `v @ attn.T`. pnnx's SDPA fusion (pass_level5/fuse_scaled_dot_product_attention.cpp)
    only matches (batch, heads, size, feat) with the permute on the KEY, so nothing fuses:
    every attention block lowers to 2 MatMul + Softmax + a BinaryOp on the NxN attention
    matrix + ~7 Permute/Reshape, and the `area` split lands in the leading dim that ncnn does
    not have.
  * `DynamicRoutingLayer.forward` (moe/routers.py:496) returns the [B,E,1,1] router weights
    `.repeat(1,1,H,W)`: ncnn Tile + per-expert Slice + full-map Muls, although ncnn BinaryOp
    broadcasts a [1,1,1] operand natively (src/layer/binaryop.cpp).

This driver installs EXPORT-TIME REWRITES on the fork's classes (the fork tree is never
edited), traces with them active, and validates the result:

  R1  the `hd**-0.5` attention scale is folded into the q rows of the qkv 1x1 conv, so no
      BinaryOp is left on the attention matrix (SDPA is told scale=1.0);
  R2  `AAttn.forward` is replaced by a single row-permuted qkv conv -> channel split ->
      (1, heads*area, M, hd) -> `F.scaled_dot_product_attention` -> back to NCHW, with the
      positional-encoding depthwise conv fed by the v channels of the very same conv output
      (no second conv). Every intermediate keeps dim0 == 1: `area` is folded into the heads
      axis exactly as `_moa_window_attn_ncnn` in export_ncnn_mixture.py does. pnnx then
      emits ncnn's fused `SDPA` layer (pass_ncnn/F_scaled_dot_product_attention.cpp,
      F_scaled_dot_product_attention_fb);
  R3  `DynamicRoutingLayer.forward` returns the [B,E,1,1] weights unexpanded (tracing only);
      `ES_MOE._dense_forward` (moe/modules.py:650-657) already broadcasts per expert.

Every export is preceded by a full-model selftest on the real weights (patched forward vs
original forward, maxdiff gate) and followed by an ncnn-vs-PyTorch check on the bench probe
image (results/int8_bench/probe_coco.jpg letterboxed exactly like cpp/src/common.cpp).

Legacy checkpoint note: the released v0.1-N COCO weights predate `capacity_factor` on
EfficientSpatialRouter and `add_residual` on OptimizedMOEImproved; the current fork neither
defaults them nor repairs them, so the checkpoint cannot even run forward. The repair applied
here (`add_residual=False` where the attribute is ABSENT, `capacity_factor=None`) is the one
from /data/harvest/evals/coco_eval_models.py (a missing add_residual means the block was
trained WITHOUT the residual; defaulting it to True collapses class confidence to ~0.04).
The det routers pick top-k experts with data-dependent python control flow, so the trace bakes
the expert selection for the TRACE INPUT (zeros, exactly like the stock exporter); that is a
pre-existing property of every ncnn export of this model and is reported, not hidden.

Usage:
    PYTHONPATH=/data/YOLO-Master /data/tmp/venv-v2608/bin/python scripts/export_ncnn_dense.py \\
        --pt /data/coreml_export/YOLO-Master-v0.1-seg-N.pt --out models/v0.1-seg-n-sdpa_ncnn
    ... --pt /data/YOLO-Master/YOLO-Master-v0.1-N.pt --out models/v0.1-n_ncnn --no-rewrite
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

PROBE = Path("results/int8_bench/probe_coco.jpg")
PROBE_F32 = Path("results/int8_bench/probe_640.f32")


# ---------------------------------------------------------------------------
# R1 + R2: area attention as one conv + SDPA
# ---------------------------------------------------------------------------


def _qkv_row_permutation(num_heads: int, head_dim: int):
    """Row order of the rewritten qkv conv.

    The original `AAttn` views the qkv conv output as (N, heads, 3*hd) and splits the LAST axis
    into q|k|v, i.e. output channel `h*3*hd + s*hd + d` is stream s (0=q, 1=k, 2=v) of head h.
    The rewrite wants stream-major channels `s*C + h*hd + d` so that a single channel split
    yields q, k, v as plain NCHW tensors (v then feeds `pe` directly). Permuting the rows of a
    1x1 conv is exact. Returns new_row -> old_row.
    """
    C = num_heads * head_dim
    perm = [0] * (3 * C)
    for s in range(3):
        for h in range(num_heads):
            for d in range(head_dim):
                perm[s * C + h * head_dim + d] = h * 3 * head_dim + s * head_dim + d
    return perm


def install_aattn_convs(model: nn.Module) -> int:
    """Register `_ncnn_qkv` (row-permuted, q-scaled copy of the fused qkv conv) on every AAttn.

    Must run AFTER `model.fuse()` (the conv+BN fold): the rewrite copies `qkv.conv` weights and
    a still-present BN would be silently dropped. Registered as a real submodule before tracing
    so pnnx sees an nn.Conv2d (a weight tensor created inside forward would trace as a constant).
    """
    from ultralytics.nn.modules.block import AAttn

    n = 0
    for mod in model.modules():
        if not isinstance(mod, AAttn):
            continue
        if hasattr(mod.qkv, "bn"):
            raise RuntimeError("AAttn.qkv still has a BatchNorm: call model.fuse() before install_aattn_convs")
        conv = mod.qkv.conv
        C = mod.all_head_dim
        perm = torch.tensor(_qkv_row_permutation(mod.num_heads, mod.head_dim))
        w = conv.weight.detach()[perm].clone()
        b = conv.bias.detach()[perm].clone() if conv.bias is not None else torch.zeros(3 * C)
        # R1: `attn = (q * hd**-0.5).T @ k` -> scale the q rows once, at export time.
        w[:C] *= mod.head_dim**-0.5
        b[:C] *= mod.head_dim**-0.5
        new = nn.Conv2d(conv.in_channels, 3 * C, 1, bias=True)
        new.weight = nn.Parameter(w, requires_grad=False)
        new.bias = nn.Parameter(b, requires_grad=False)
        mod._ncnn_qkv = new
        n += 1
    return n


def _aattn_forward_ncnn(self, x: torch.Tensor) -> torch.Tensor:
    """AAttn.forward as conv -> split -> SDPA(heads*area) -> NCHW, batch axis pinned to 1.

    Layout derivation against the original (block.py:1705-1732), with C = heads*hd, N = H*W:
      * original: qkv (B,3C,H,W) -> flatten -> (B, N, 3C), token n = y*W + x (row-major);
        `area>1`: reshape (B*area, N/area, 3C) splits the FLATTENED token axis into `area`
        CONTIGUOUS blocks of M = N/area tokens: block a = tokens [a*M, (a+1)*M) = image rows
        [a*H/area, (a+1)*H/area) (requires H % area == 0, guarded below);
      * per (b, a, head): attn[i, j] = softmax_j(scale * q_i . k_j), out_i = sum_j attn[i, j] v_j
        which is exactly F.scaled_dot_product_attention on (M, hd) row-major q, k, v;
      * output tokens are put back at n = a*M + m, channel = h*hd + d, then viewed as (B,C,H,W).
    Here: q (1, C, H, W) with channel h*hd+d -> view (1, heads, hd, N) -> transpose -> (1, heads,
    N, hd) -> view (1, heads*area, M, hd): the contiguous N -> (area, M) split IS the original's
    block split, and merging `area` into the heads axis is legal because blocks never attend
    across each other. Head g = h*area + a. The inverse chain restores channel h*hd+d, token
    a*M+m. ncnn sees one Slice, 3x(Reshape, Permute(h<->w), Reshape), SDPA, Reshape, Permute,
    Reshape - no MatMul, no Softmax, no BinaryOp on the attention matrix, no batch axis.
    """
    B, _, H, W = x.shape
    area, nh, hd, C = self.area, self.num_heads, self.head_dim, self.all_head_dim
    N = H * W
    if H % area != 0 or N % area != 0:
        return _AATTN_ORIG_FORWARD(self, x)

    qkv = self._ncnn_qkv(x)  # (1, 3C, H, W): q rows already carry hd**-0.5
    q, k, v = qkv.split(C, dim=1)  # v is exactly the original's `v.permute(0,3,1,2)` NCHW view

    def heads(t):
        t = t.reshape(B, nh, hd, N).transpose(2, 3)
        return t if area == 1 else t.reshape(B, nh * area, N // area, hd)  # identity view -> no ncnn Reshape

    o = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), scale=1.0)
    o = o if area == 1 else o.reshape(B, nh, N, hd)
    o = o.transpose(2, 3).reshape(B, C, H, W)
    return self.proj(o + self.pe(v))


# ---------------------------------------------------------------------------
# R3: router weights stay [B,E,1,1]
# ---------------------------------------------------------------------------


def _router_forward_ncnn(self, x: torch.Tensor) -> torch.Tensor:
    """DynamicRoutingLayer.forward minus the trailing `.repeat(1, 1, H, W)`; tracing only.

    Same soft/no-top-k branch the original takes under `exporting` (routers.py:462-496); the
    eager path (diagnostics, hard top-k, channel validation) is untouched because this
    replacement defers to the original whenever no trace is active. The consumer
    `ES_MOE._dense_forward` does `expert_out * routing_weights[:, i:i+1]`, which broadcasts a
    [1,1,1,1] operand - ncnn BinaryOp handles that natively, so the Tile + full-map Muls go away.
    `_compute_load_balancing_loss` reduces over dims (0,2,3) and is dead code under trace.
    """
    if not torch.jit.is_tracing():
        return _ROUTER_ORIG_FORWARD(self, x)
    routing_logits = self.routing_network(self.global_pool(x))  # [B, E, 1, 1]
    if not self.use_top_k:
        return F.softmax(routing_logits.float().clamp(-30.0, 30.0), dim=1).type_as(x)
    return self._soft_top_k(routing_logits)


# ---------------------------------------------------------------------------
# det (v0.1-N) dispatch shim: baked top-k, applied to the stock graph and the rewritten one alike
# ---------------------------------------------------------------------------


def bake_det_routing(model: nn.Module, ex: torch.Tensor, probe: torch.Tensor) -> list[str]:
    """Record, per OptimizedMOEImproved block, the experts the router picks for the TRACE input.

    The stock `OptimizedMOEImproved.forward` (moe/modules.py:1069-1160) dispatches with
    `torch.topk` + `torch.where(mask)` + `x[batch_idx]` + `index_add_`: pnnx keeps those as
    torch.topk / aten::where / Tensor.index / aten::index_add layers that ncnn does not have
    (the stock ultralytics ncnn export of this checkpoint does not load; the older
    models/p03_v01n_ncnn is a PRUNED model whose 2 surviving experts made top-k trivial).
    A trace always freezes the data-dependent part - `torch.where` results are constants in the
    stock trace too - so the honest ncnn form is: expert SET fixed at export, expert WEIGHTS
    dynamic (softmax probabilities of the fixed experts, renormalised as `_process_logits` does).
    The probe routing is printed next to the baked one so a divergence is visible, not hidden.
    """
    notes = []
    blocks = [(n, m) for n, m in model.named_modules() if type(m).__name__ == "OptimizedMOEImproved"]
    picks = {}

    def hook(name):
        def _h(mod, inp, out):
            picks.setdefault(name, []).append([int(i) for i in out[1].view(-1).tolist()])

        return _h

    handles = [m.routing.register_forward_hook(hook(n)) for n, m in blocks]
    with torch.no_grad():
        model(ex)
        model(probe)
    for h in handles:
        h.remove()
    for n, m in blocks:
        baked, on_probe = picks[n]
        m._ncnn_baked_experts = baked
        notes.append(
            f"{n}: E={m.num_experts} k={m.top_k} baked experts {baked} | probe would pick {on_probe}"
            + ("" if set(baked) == set(on_probe) else "  <-- DIFFERENT")
        )
    return notes


def _v01_moe_forward_baked(self, x: torch.Tensor) -> torch.Tensor:
    """OptimizedMOEImproved.forward with the expert set frozen to `_ncnn_baked_experts` (tracing only).

    Router math is EfficientSpatialRouter.forward + BaseRouter._process_logits in eval mode
    (no noise, no capacity): avgpool(pool_scale) -> router convs -> spatial mean -> softmax;
    the weights of the baked experts are the softmax entries renormalised by their sum, which
    equals the stock top-k values whenever the runtime top-k set is the baked one.
    """
    if not torch.jit.is_tracing():
        return _V01_MOE_ORIG_FORWARD(self, x)
    B, _, H, W = x.shape
    r = self.routing
    x_in = (
        F.avg_pool2d(x, kernel_size=r.pool_scale, stride=r.pool_scale) if (H > r.pool_scale and W > r.pool_scale) else x
    )
    probs = F.softmax(r.router(x_in).mean(dim=[2, 3]), dim=1)  # [B, E]
    idx = self._ncnn_baked_experts
    vals = torch.cat([probs[:, e : e + 1] for e in idx], dim=1)  # [B, k] dynamic
    # _process_logits divides by sum.clamp_min(1e-6); the k largest of E softmax entries sum to
    # >= k/E >= 0.125 here, so the guard is a mathematical no-op - and pnnx spells it as an ncnn
    # Clip(1e-6, FLT_MAX) whose 3.4e38 literal makes the runtime's precision policy refuse fp16.
    vals = vals / vals.sum(dim=1, keepdim=True)
    out = self.shared_expert(x)
    for k, e in enumerate(idx):
        out = out + self.experts[e](x) * vals[:, k : k + 1].view(B, 1, 1, 1)
    if self.add_residual and self.in_channels == self.out_channels:
        out = out + x
    return out


_V01_MOE_ORIG_FORWARD = None


@contextlib.contextmanager
def det_dispatch_shim():
    global _V01_MOE_ORIG_FORWARD
    from ultralytics.nn.modules.moe.modules import OptimizedMOEImproved

    _V01_MOE_ORIG_FORWARD = OptimizedMOEImproved.forward
    OptimizedMOEImproved.forward = _v01_moe_forward_baked
    try:
        yield
    finally:
        OptimizedMOEImproved.forward = _V01_MOE_ORIG_FORWARD


_AATTN_ORIG_FORWARD = None
_ROUTER_ORIG_FORWARD = None


@contextlib.contextmanager
def dense_rewrites():
    """Install R1-R3 on the fork's classes; restore on exit."""
    global _AATTN_ORIG_FORWARD, _ROUTER_ORIG_FORWARD
    from ultralytics.nn.modules.block import AAttn
    from ultralytics.nn.modules.moe.routers import DynamicRoutingLayer

    _AATTN_ORIG_FORWARD = AAttn.forward
    _ROUTER_ORIG_FORWARD = DynamicRoutingLayer.forward
    AAttn.forward = _aattn_forward_ncnn
    DynamicRoutingLayer.forward = _router_forward_ncnn
    try:
        yield
    finally:
        AAttn.forward = _AATTN_ORIG_FORWARD
        DynamicRoutingLayer.forward = _ROUTER_ORIG_FORWARD


# ---------------------------------------------------------------------------
# model preparation (mirrors ultralytics Exporter.__call__, exporter.py:800-848)
# ---------------------------------------------------------------------------


def repair_legacy_v01_det(model: nn.Module) -> list[str]:
    """Make the Jan-2026 released v0.1-N checkpoint runnable on the 8.4.101 fork (see module doc)."""
    notes = []
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if cls == "OptimizedMOEImproved":
            if "add_residual" not in vars(mod):
                mod.add_residual = False
                notes.append(f"{name}.add_residual=False (absent in checkpoint)")
            # training-schedule bookkeeping read unconditionally by forward (modules.py:1098);
            # values are the __init__ defaults and are inert in eval mode
            for attr, default in (
                ("_training_step", 0),
                ("_current_top_k", mod.num_experts),
                ("warmup_steps", 5000),
                ("expert_dropout_rate", 0.15),
                ("dropout_interval", 100),
                ("progressive_sparsity", True),
                ("detach_routing", False),
            ):
                if not hasattr(mod, attr):
                    setattr(mod, attr, default)
                    notes.append(f"{name}.{attr}={default} (absent in checkpoint)")
        if cls in {"EfficientSpatialRouter", "AdaptiveRoutingLayer", "LocalRoutingLayer"} and not hasattr(
            mod, "capacity_factor"
        ):
            mod.capacity_factor = None
            notes.append(f"{name}.capacity_factor=None (absent in checkpoint)")
    return notes


def prepare_model(pt: str, imgsz: int):
    """Load, repair, fuse and put the head into export mode; returns (model, names, task)."""
    from ultralytics import YOLO
    from ultralytics.nn.modules import Detect

    y = YOLO(pt)
    model = deepcopy(y.model).float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for note in repair_legacy_v01_det(model):
        print(f"  [legacy] {note}")
    model = model.fuse()
    for m in model.modules():
        if isinstance(m, Detect):
            m.dynamic = False
            m.export = True
            m.format = "ncnn"
            m.max_det = min(300, sum(int(imgsz / s) ** 2 for s in model.stride.tolist()))
            m.agnostic_nms = False
            m.xyxy = False
            m.shape = None
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    return model, names, y.task


class _Outputs(nn.Module):
    """Flatten the head output to a tuple of tensors (det: out0; seg: out0, proto as out1)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        y = self.m(x)
        if torch.is_tensor(y):
            return y
        return tuple(t for t in y if torch.is_tensor(t))


def as_tuple(y):
    return (y,) if torch.is_tensor(y) else tuple(y)


# ---------------------------------------------------------------------------
# probe + validation
# ---------------------------------------------------------------------------


def load_probe(imgsz: int) -> np.ndarray:
    """probe_coco.jpg letterboxed like cpp/src/common.cpp (min-scale, gray 114, centered, RGB/255).

    Reuses scripts/make_probe_f32.py's `preprocess` (the tool that made the bench blob) and, at
    640, asserts bit-equality with results/int8_bench/probe_640.f32 so the PyTorch reference, the
    python-ncnn check and ncnn_bench all consume the identical input.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_probe_f32 import preprocess

    blob, _ = preprocess(PROBE, imgsz)
    if imgsz == 640 and PROBE_F32.exists():
        ref = np.fromfile(PROBE_F32, dtype=np.float32).reshape(1, 3, imgsz, imgsz)
        assert np.array_equal(blob, ref), "letterbox drifted from results/int8_bench/probe_640.f32"
    return blob


def run_ncnn(param: Path, binf: Path, blob: np.ndarray, n_out: int, threads: int = 4):
    import ncnn as pyncnn

    net = pyncnn.Net()
    net.opt.num_threads = threads
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    net.opt.use_bf16_storage = False
    if net.load_param(str(param)) != 0 or net.load_model(str(binf)) != 0:
        raise RuntimeError(f"ncnn load failed: {param}")
    ex = net.create_extractor()
    ex.input("in0", pyncnn.Mat(np.ascontiguousarray(blob[0])))
    outs = []
    for i in range(n_out):
        rc, m = ex.extract(f"out{i}")
        if rc != 0:
            raise RuntimeError(f"ncnn extract out{i} failed rc={rc}")
        outs.append(np.array(m)[None])
    return outs


def out_diff(a: np.ndarray, b: np.ndarray, nc: int, imgsz: int, is_out0: bool) -> tuple[str, float]:
    """Max abs diff of two outputs, reported per row group and reduced to one SCALED number.

    out0 rows 0..3 are xywh in PIXELS (magnitude up to imgsz: the original fp32 graph itself
    differs from an fp64 run by ~6e-3 there), rows 4..4+nc are class probabilities in [0,1],
    the rest are mask coefficients / proto. The gate is applied to max(box/imgsz, cls, rest),
    i.e. every group is judged at its own scale.
    """
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    if not is_out0:
        m = float(d.max())
        return f"{m:.2e}", m
    box, cls, rest = float(d[0, :4].max()), float(d[0, 4 : 4 + nc].max()), float(d[0, 4 + nc :].max(initial=0.0))
    return f"box {box:.2e} (px) cls {cls:.2e} rest {rest:.2e}", max(box / imgsz, cls, rest)


def det_count(out0: np.ndarray, nc: int, conf: float = 0.25) -> int:
    """Pre-NMS count of anchors whose best class score exceeds conf (rows 4..4+nc of out0)."""
    return int((out0[0, 4 : 4 + nc].max(axis=0) > conf).sum())


def layer_census(param: Path) -> Counter:
    return Counter(l.split()[0] for l in param.read_text().splitlines()[2:] if l.strip())


def write_metadata(dst: Path, names: dict, task: str, imgsz: int, stride: int, version: str):
    """Sidecar with the keys the C++ runtime's read_ncnn_yaml consumes (names / imgsz / end2end)."""
    with open(dst / "metadata.yaml", "w") as f:
        f.write(f"description: YOLO-Master {task} model, ncnn dense-rewrite export (scripts/export_ncnn_dense.py)\n")
        f.write(f"version: {version}\n")
        f.write(f"task: {task}\n")
        f.write(f"stride: {stride}\n")
        f.write("batch: 1\n")
        f.write(f"imgsz:\n- {imgsz}\n- {imgsz}\n")
        f.write("end2end: false\n")
        f.write("names:\n")
        f.writelines(f"  {k}: {names[k]}\n" for k in sorted(names))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pt", required=True, help="checkpoint (.pt)")
    ap.add_argument("--out", required=True, help="output ncnn dir, e.g. models/v0.1-seg-n-sdpa_ncnn")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-rewrite", action="store_true", help="stock graph (baseline export for comparison)")
    ap.add_argument("--ref", default=None, help="write/compare the fp32 PyTorch probe outputs here (.npz)")
    ap.add_argument("--compare", default=None, help="another ncnn dir whose probe outputs are diffed against this one")
    ap.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="patched-vs-original full-model gate on the SCALED maxdiff (see out_diff)",
    )
    ap.add_argument(
        "--trace-input",
        choices=("zeros", "probe"),
        default="zeros",
        help="trace input (zeros = stock exporter parity; matters only for top-k routed det models)",
    )
    args = ap.parse_args()

    import ncnn as pyncnn
    import pnnx
    from ultralytics import __version__

    print(f"[env] ultralytics {__version__} torch {torch.__version__} ncnn {pyncnn.__version__}")
    assert not __version__.startswith("8.3."), "old-lineage ultralytics resolved - wrong PYTHONPATH"

    model, names, task = prepare_model(args.pt, args.imgsz)
    nc = len(names)
    stride = int(max(model.stride))
    wrapped = _Outputs(model)
    probe = torch.from_numpy(load_probe(args.imgsz))
    ex = torch.zeros(1, 3, args.imgsz, args.imgsz) if args.trace_input == "zeros" else probe.clone()

    routed = any(type(m).__name__ == "OptimizedMOEImproved" for m in model.modules())
    if routed:
        for note in bake_det_routing(model, ex, probe):
            print(f"  [bake] {note}")

    # reference: fp32 PyTorch, BEFORE any patch
    with torch.no_grad():
        ref = as_tuple(wrapped(probe))
    print(f"[ref] {task} outputs: " + ", ".join(f"out{i}{tuple(t.shape)}" for i, t in enumerate(ref)))
    maxconf = float(ref[0][0, 4 : 4 + nc].max())
    print(f"[ref] probe max class conf {maxconf:.3f}, pre-NMS dets@0.25 = {det_count(ref[0].numpy(), nc)}")
    if maxconf < 0.5:
        raise RuntimeError("reference max confidence < 0.5: checkpoint compat repair is wrong (see module doc)")
    if args.ref:
        np.savez(args.ref, **{f"out{i}": t.numpy() for i, t in enumerate(ref)})

    if not args.no_rewrite:
        n = install_aattn_convs(model)
        print(f"[rewrite] {n} AAttn qkv convs re-registered (rows permuted, q rows scaled by hd**-0.5)")
        with torch.no_grad(), dense_rewrites():
            got = as_tuple(wrapped(probe))
        worst = 0.0
        for i, (a, b) in enumerate(zip(ref, got)):
            txt, scaled = out_diff(a.numpy(), b.numpy(), nc, args.imgsz, i == 0)
            worst = max(worst, scaled)
            print(f"[selftest] patched-vs-original eager out{i}: {txt}")
        print(f"[selftest] scaled maxdiff {worst:.2e} (gate {args.tol:.0e})")
        if worst > args.tol:
            raise RuntimeError("dense rewrite selftest failed")

    dst = Path(args.out)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    ctx = contextlib.ExitStack()
    if routed:
        ctx.enter_context(det_dispatch_shim())
    if not args.no_rewrite:
        ctx.enter_context(dense_rewrites())
    # pnnx compat shim (applies to the stock graph too): the det routers' `.clamp_min(1e-6)`
    # (routers.py BaseRouter._process_logits) traces to aten::clamp_min, which pnnx converts
    # but cannot spell in its generated model_pnnx.py, so pnnx.export dies at the import step
    # AFTER writing the ncnn files. clamp(min=) is the same op under a name pnnx can print.
    with ctx, mock.patch.object(torch.Tensor, "clamp_min", lambda t, v: t.clamp(min=v)):
        pnnx.export(
            wrapped,
            str(dst / "model.pt"),
            inputs=ex,
            fp16=False,  # fp32 weights; the ncnn runtime decides fp16 at load
            device="cpu",
            ncnnparam=str(dst / "model.ncnn.param"),
            ncnnbin=str(dst / "model.ncnn.bin"),
            ncnnpy=str(dst / "model_ncnn.py"),
            pnnxparam=str(dst / "model.pnnx.param"),
            pnnxbin=str(dst / "model.pnnx.bin"),
            pnnxpy=str(dst / "model_pnnx.py"),
            pnnxonnx=str(dst / "model.pnnx.onnx"),
        )
    for junk in (
        "model.pt",
        "model.pnnx.param",
        "model.pnnx.bin",
        "model_pnnx.py",
        "model_ncnn.py",
        "model.pnnx.onnx",
        "debug.param",
        "debug.bin",
        "debug2.param",
        "debug2.bin",
    ):
        (dst / junk).unlink(missing_ok=True)
    for junk in list(Path.cwd().glob("debug*.param")) + list(Path.cwd().glob("debug*.bin")):
        junk.unlink(missing_ok=True)
    shutil.rmtree(dst / "__pycache__", ignore_errors=True)
    write_metadata(dst, names, task, args.imgsz, stride, __version__)

    # census: every layer type must be registered; report the glue that matters
    types = layer_census(dst / "model.ncnn.param")
    unregistered = {t: n for t, n in types.items() if pyncnn.layer_to_index(t) == -1 and t != "Input"}
    if unregistered:
        raise RuntimeError(f"unregistered ncnn layer types: {unregistered}")
    watch = ("SDPA", "MatMul", "Softmax", "Tile", "Permute", "Reshape", "BinaryOp", "Slice", "Crop", "Split")
    print(
        f"[param] {sum(types.values())} layers, {len(types)} types | "
        + " ".join(f"{t}={types.get(t, 0)}" for t in watch)
    )
    n_1e30 = sum(1 for l in (dst / "model.ncnn.param").read_text().splitlines() if "e+30" in l)
    print(f"[param] fp16-overflow literals (e+30): {n_1e30}")

    # ncnn vs PyTorch on the probe (fp32 CPU)
    outs = run_ncnn(dst / "model.ncnn.param", dst / "model.ncnn.bin", probe.numpy(), len(ref))
    for i, (o, r) in enumerate(zip(outs, ref)):
        txt, scaled = out_diff(o.reshape(r.shape), r.numpy(), nc, args.imgsz, i == 0)
        print(f"[ncnn-vs-torch] out{i}{tuple(o.shape)}: {txt} | scaled {scaled:.2e}")
    print(
        f"[ncnn] pre-NMS dets@0.25 = {det_count(outs[0].reshape(ref[0].shape), nc)} "
        f"(torch {det_count(ref[0].numpy(), nc)})"
    )
    if args.compare:
        cmp_dir = Path(args.compare)
        outs_c = run_ncnn(cmp_dir / "model.ncnn.param", cmp_dir / "model.ncnn.bin", probe.numpy(), len(ref))
        for i, (o, c) in enumerate(zip(outs, outs_c)):
            txt, scaled = out_diff(o.reshape(c.shape), c, nc, args.imgsz, i == 0)
            print(f"[ncnn-vs-{cmp_dir.name}] out{i}: {txt} | scaled {scaled:.2e}")
        print(f"[ncnn-vs-{cmp_dir.name}] pre-NMS dets@0.25 = {det_count(outs_c[0].reshape(ref[0].shape), nc)}")
    print(f"[done] {dst}")


if __name__ == "__main__":
    main()
