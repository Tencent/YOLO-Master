#!/usr/bin/env python3
"""Export an Ultralytics / YOLO-Master detector or segmenter to a Core ML `.mlpackage`.

Produces the exact metadata the YOLO-Master CoreML Runner app reads (names, imgsz, output
tensor, task, and — for segmentation — proto/nm), converts to an mlprogram, and validates the
result. Works for plain YOLO (e.g. yolov12x) and YOLO-Master MoE models (v0.1 / EsMoE / UoMoE,
incl. P2 heads) and segmentation models.

Environment
-----------
Conversion runs on Linux (coremltools; only *prediction* needs macOS). Use an isolated env:

    conda create -n cmlexport python=3.11 -y
    pip install torch==2.5.1 torchvision==0.20.1 coremltools==9.0

Then install the ultralytics build that MATCHES the checkpoint:
  * YOLO-Master checkpoints  ->  the fork:   pip install -e /path/to/YOLO-Master --no-deps
  * stock YOLO checkpoints   ->  stock:      pip install ultralytics==8.3.240
Mixing them fails at trace time (e.g. stock yolov12's AAttn vs the fork's -> 'has no attribute qkv').
torch 2.11 also breaks coremltools' torch frontend (aten::Int) — pin torch 2.5.

Why the MoE tricks below (no-ops for plain models, required for YOLO-Master MoE)
------------------------------------------------------------------------------
1. mock is_in_onnx_export=True during trace -> forces the DENSE MoE path (sparse top-k routing is
   data-dependent and cannot be captured by a static trace).
2. torch.jit.freeze after trace -> constant-folds shape arithmetic; without it v0.1/UoMoE emit a
   dynamic aten::mul/floor_divide -> aten::Int that coremltools rejects
   ("only 0-dimensional arrays can be converted to Python scalars").
3. no-op ES_MOE._compute_load_balancing_loss -> EsMoE writes telemetry buffers in-place (aten::copy_)
   which coremltools' tensor-assignment pass rejects ("No matching select or slice").

LoRA note: merge adapters into the base BEFORE exporting (`YOLO(base).load_lora(dir, merge=True)`);
a merged LoRA is a static graph. Routed MoLoRA cannot be traced (same wall as the sparse MoE).

yolov12 note: the sunsmarterjie/yolov12 checkpoints use a split qk+v area-attention (AAttn) that
stock ultralytics doesn't have (it uses combined qkv) -> 'AAttn object has no attribute qkv'. Pass
--yolov12-aattn to monkeypatch stock's AAttn with the qk+v (CPU/dense) variant so those load & trace.
Area attention also reshapes by spatial size, which stays dynamic under trace -> the eager warmup
below bakes each layer's concrete spatial dims so the reshapes fold to static shapes for coremltools.
"""
import argparse, sys, traceback
from unittest import mock

import torch
import coremltools as ct
from ultralytics import YOLO


def _patch_yolov12_aattn():
    """Install sunsmarterjie/yolov12's split qk+v AAttn (CPU/dense path) onto stock ultralytics, so its
    checkpoints load and trace. No-op-safe if the module layout differs."""
    import torch.nn as nn
    import ultralytics.nn.modules.block as B
    from ultralytics.nn.modules.conv import Conv

    class AAttn(nn.Module):
        def __init__(self, dim, num_heads, area=1):
            super().__init__(); self.area = area; self.num_heads = num_heads
            self.head_dim = hd = dim // num_heads; ahd = hd * num_heads
            self.qk = Conv(dim, ahd * 2, 1, act=False); self.v = Conv(dim, ahd, 1, act=False)
            self.proj = Conv(ahd, dim, 1, act=False); self.pe = Conv(ahd, dim, 5, 1, 2, g=dim, act=False)

        def forward(self, x):
            if torch.jit.is_tracing() and hasattr(self, "_shp"):
                Bb, H, W = self._shp                      # concrete ints cached on the eager warmup
            else:
                Bb, _, H, W = x.shape; self._shp = (int(Bb), int(H), int(W))
            C = self.head_dim * self.num_heads; N = H * W
            qk = self.qk(x).flatten(2).transpose(1, 2); v = self.v(x); pp = self.pe(v)
            v = v.flatten(2).transpose(1, 2)
            if self.area > 1:
                qk = qk.reshape(Bb * self.area, N // self.area, C * 2)
                v = v.reshape(Bb * self.area, N // self.area, C); Bb = Bb * self.area; N = N // self.area
            q, k = qk.split([C, C], dim=2)
            q = q.transpose(1, 2).view(Bb, self.num_heads, self.head_dim, N)
            k = k.transpose(1, 2).view(Bb, self.num_heads, self.head_dim, N)
            v = v.transpose(1, 2).view(Bb, self.num_heads, self.head_dim, N)
            attn = ((q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)).softmax(dim=-1)
            x = (v @ attn.transpose(-2, -1)).permute(0, 3, 1, 2)
            if self.area > 1:
                x = x.reshape(Bb // self.area, N * self.area, C); Bb = Bb // self.area; N = N * self.area
            x = x.reshape(Bb, H, W, C).permute(0, 3, 1, 2)
            return self.proj(x + pp)

    B.AAttn = AAttn
    import ultralytics.nn.modules as _M
    if hasattr(_M, "AAttn"): _M.AAttn = AAttn


def _ane_topk_mask(p: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k membership of each column of `p` [B, E] with arithmetic only (no topk / sort / gather / int ops).

    rank_i = number of j with p_j > p_i, counted with step(p_j - p_i); mask_i = step(k - 0.5 - rank_i).
    step(z) = clamp(32768 z, 0, 1) is exact for gaps of 3e-5 and more (safe in fp16); two probabilities closer than
    that (an fp16 tie) get a proportional blend instead of an arbitrary winner.
    """
    d = p.unsqueeze(2) - p.unsqueeze(1)                     # [B, j, i] = p_j - p_i
    rank = (d * 32768.0).clamp(0.0, 1.0).sum(dim=1)          # [B, i]
    return ((float(k) - 0.5 - rank) * 32768.0).clamp(0.0, 1.0)


def _ane_router_probs(router, x: torch.Tensor) -> torch.Tensor:
    """The router's pooled softmax probabilities [B, E] in eval mode (what its top-k is taken over)."""
    import torch.nn.functional as F
    B, C, H, W = x.shape
    cls = type(router).__name__
    if cls == "EfficientSpatialRouter":
        x_in = F.avg_pool2d(x, kernel_size=router.pool_scale, stride=router.pool_scale) \
            if (H > router.pool_scale and W > router.pool_scale) else x
        return F.softmax(router.router(x_in).float().mean(dim=[2, 3]), dim=1)
    if cls == "UltraEfficientRouter":
        x_down = F.avg_pool2d(x, kernel_size=router.pool_scale, stride=router.pool_scale) \
            if (H > router.pool_scale and W > router.pool_scale) else x
        logits = router.router(x_down).clamp(-30.0, 30.0) / router.temperature
        return F.softmax(logits.float(), dim=1).mean(dim=[2, 3])
    raise RuntimeError(f"ANE-safe MoE export: unsupported router {cls}")


def _patch_ane_safe_moe() -> list[str]:
    """Export-time forward for OptimizedMOEImproved (v0.1 family) and OptimizedMOE that keeps the whole
    block on the Neural Engine.

    The stock export path stacks every expert's output and then GATHERS the router's top-k with
    torch.topk + torch.gather: MIL `topk`, `gather_along_axis` and int32 index tensors. The ANE has no
    kernels for those, so under cpuAndNeuralEngine Core ML runs each router on the CPU and pays a
    sync + copy on both sides of every MoE block (v0.1-N: 9.2 ms on the ANE against 4.1 ms for the
    router-free v0.1-seg-N graph, M4 Max). This path computes the same mixture as
    shared(x) + sum_i w_i * expert_i(x) with w = p * mask / sum(p * mask) and the mask from
    `_ane_topk_mask`: identical weights away from fp16 ties, no integer op anywhere.
    """
    import ultralytics.nn.modules.moe.modules as M
    notes = []

    def improved_forward(self, x):
        B = x.shape[0]
        p = _ane_router_probs(self.routing, x)
        w = p * _ane_topk_mask(p, int(self.top_k))
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        out = self.shared_expert(x)
        for i, expert in enumerate(self.experts):
            out = out + expert(x) * w[:, i].view(B, 1, 1, 1).to(x.dtype)
        if getattr(self, "add_residual", False) and self.in_channels == self.out_channels:
            out = out + x
        return out

    def optimized_forward(self, x):
        B = x.shape[0]
        p = _ane_router_probs(self.router, x)
        w = p * _ane_topk_mask(p, int(self.top_k))
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        mix = 0
        for i, expert in enumerate(self.experts):
            mix = mix + expert(x) * w[:, i].view(B, 1, 1, 1).to(x.dtype)
        return self.shared_expert(x) + mix.clamp(-1e4, 1e4)

    if hasattr(M, "OptimizedMOEImproved"):
        M.OptimizedMOEImproved.forward = improved_forward; notes.append("OptimizedMOEImproved")
    if hasattr(M, "OptimizedMOE"):
        M.OptimizedMOE.forward = optimized_forward; notes.append("OptimizedMOE")
    return notes


def _repair_legacy_checkpoint(model) -> list[str]:
    """Make the Jan-2026 released v0.1-N COCO checkpoint exportable on the 8.4.101 fork (the same
    repair scripts/export_ncnn_dense.py applies; no-op for current checkpoints).

    Two things that checkpoint predates: (1) `OptimizedMOEImproved.add_residual`, which the fork's
    `_ensure_compat_attrs()` defaults to True although those blocks were trained WITHOUT the residual;
    exporting them as-is keeps box regression alive and collapses every class score (mAP ~0, no
    error), so absence means False. (2) Router `capacity_factor` and the training-schedule
    bookkeeping the forward reads unconditionally (inert in eval mode, set to the __init__ defaults).
    """
    notes = []
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if cls == "OptimizedMOEImproved":
            if "add_residual" not in vars(mod):
                mod.add_residual = False
                notes.append(f"{name}.add_residual=False (absent in checkpoint)")
            for attr, default in (("_training_step", 0), ("_current_top_k", getattr(mod, "num_experts", 1)),
                                  ("warmup_steps", 5000), ("expert_dropout_rate", 0.15), ("dropout_interval", 100),
                                  ("progressive_sparsity", True), ("detach_routing", False)):
                if not hasattr(mod, attr):
                    setattr(mod, attr, default)
                    notes.append(f"{name}.{attr}={default} (absent in checkpoint)")
        if cls in {"EfficientSpatialRouter", "AdaptiveRoutingLayer", "LocalRoutingLayer"} and not hasattr(mod, "capacity_factor"):
            mod.capacity_factor = None
            notes.append(f"{name}.capacity_factor=None (absent in checkpoint)")
    return notes


def _silence_esmoe_aux_loss():
    """No-op EsMoE's in-place load-balancing telemetry (see docstring #3). Safe: training-only."""
    try:
        from ultralytics.nn.modules.moe.modules import ES_MOE
        ES_MOE._compute_load_balancing_loss = (
            lambda self, routing_weights, eps=1e-6: torch.zeros((), device=routing_weights.device))
    except Exception:
        pass  # plain ultralytics has no ES_MOE — nothing to patch.


def _deploy_target(name: str):
    return {"macos13": ct.target.macOS13, "macos14": ct.target.macOS14,
            "macos15": ct.target.macOS15, "ios16": ct.target.iOS16, "ios17": ct.target.iOS17}[name.lower()]


def export(weights: str, imgsz: int, out: str, target: str = "macos13",
           merge_lora_dir: str | None = None, yolov12_aattn: bool = False, ane_safe_moe: bool = True) -> dict:
    if yolov12_aattn:
        _patch_yolov12_aattn()
    if ane_safe_moe:
        for cls in _patch_ane_safe_moe():
            print(f"[ane] {cls}: top-k mixture exported as arithmetic (no topk / gather / int ops)")
    ym = YOLO(weights)
    if merge_lora_dir:
        if not ym.load_lora(merge_lora_dir, merge=True):
            raise RuntimeError(f"failed to load/merge LoRA adapters from {merge_lora_dir}")
    model = ym.model.eval()
    names = [ym.names[i] for i in sorted(ym.names)] if getattr(ym, "names", None) else []
    for note in _repair_legacy_checkpoint(model):
        print(f"[compat] {note}")

    for m in model.modules():                      # Detect/Segment head -> single concatenated tensor
        if hasattr(m, "export"): m.export = True
        if hasattr(m, "format"): m.format = "coreml"
        if hasattr(m, "use_sparse_inference"): m.use_sparse_inference = False   # force MoE dense path
    _silence_esmoe_aux_loss()

    ex = torch.zeros(1, 3, imgsz, imgsz)
    with torch.no_grad():
        _ = model(ex)                                                    # eager warmup -> bakes static spatial dims (area attention)
    with mock.patch("torch.onnx.is_in_onnx_export", return_value=True):   # dense MoE path during trace
        traced = torch.jit.trace(model, ex, strict=False, check_trace=False)
    traced = torch.jit.freeze(traced.eval())                             # constant-fold dynamic shapes
    try:
        torch.jit.run_frozen_optimizations(traced)                       # fold residual shape arithmetic
    except Exception:
        pass

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="images", shape=(1, 3, imgsz, imgsz), dtype=float)],
        minimum_deployment_target=_deploy_target(target),
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )

    # ---- metadata the app reads ----
    outs = [(o.name, list(o.type.multiArrayType.shape)) for o in mlmodel.get_spec().description.output]
    meta = mlmodel.user_defined_metadata
    if len(outs) > 1:                              # segmentation: detection [1,4+nc+nm,anchors] + protos [1,nm,mh,mw]
        det = next(o for o in outs if len(o[1]) == 3 and o[1][1] > 4)
        proto = next(o for o in outs if len(o[1]) == 4)
        out_name, task = det[0], "segment"
        meta["proto"] = proto[0]
        meta["nm"] = str(proto[1][1])
    else:
        out_name, task = outs[0][0], "detect"
    meta["task"] = task
    meta["output"] = out_name
    meta["imgsz"] = str(imgsz)
    meta["moe_export"] = "ane_safe_rank" if ane_safe_moe else "topk_gather"
    meta["precision"] = "fp16"                     # ML program default compute precision (weights and activations)
    if names: meta["names"] = ",".join(names)
    mlmodel.save(out)

    # ---- validate the emitted spec ----
    det_shape = next((s for n, s in outs if n == out_name), None)
    nc = (det_shape[1] - 4 - int(meta.get("nm", 0))) if det_shape else None
    if names and nc is not None and nc != len(names):
        raise RuntimeError(f"class-count mismatch: output implies nc={nc} but {len(names)} names")
    return {"out": out, "task": task, "output": out_name, "shapes": outs, "classes": len(names), "imgsz": imgsz}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export a YOLO(-Master) detector/segmenter to Core ML .mlpackage")
    ap.add_argument("--weights", required=True, help="path to the .pt checkpoint")
    ap.add_argument("--imgsz", type=int, default=640, help="square input size the model was trained/exported at")
    ap.add_argument("--out", required=True, help="output .mlpackage path")
    ap.add_argument("--target", default="macos13",
                    choices=["macos13", "macos14", "macos15", "ios16", "ios17"],
                    help="minimum deployment target (app floor is macOS 13; the app itself needs 14)")
    ap.add_argument("--merge-lora-dir", default=None, help="load + merge trained LoRA adapters before export")
    ap.add_argument("--yolov12-aattn", action="store_true",
                    help="monkeypatch stock ultralytics' AAttn with sunsmarterjie/yolov12's qk+v variant")
    ap.add_argument("--no-ane-safe-moe", action="store_true",
                    help="keep the stock topk + gather MoE export (router segments run on the CPU under the ANE)")
    a = ap.parse_args()
    try:
        r = export(a.weights, a.imgsz, a.out, target=a.target, merge_lora_dir=a.merge_lora_dir,
                   yolov12_aattn=a.yolov12_aattn, ane_safe_moe=not a.no_ane_safe_moe)
        print(f"OK  {r['out']}  task={r['task']}  output={r['output']}  classes={r['classes']}  "
              f"imgsz={r['imgsz']}  shapes={r['shapes']}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {a.weights}: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
