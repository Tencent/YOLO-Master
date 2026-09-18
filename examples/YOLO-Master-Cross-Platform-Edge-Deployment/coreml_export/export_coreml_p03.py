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
           merge_lora_dir: str | None = None, yolov12_aattn: bool = False) -> dict:
    if yolov12_aattn:
        _patch_yolov12_aattn()
    ym = YOLO(weights)
    if merge_lora_dir:
        if not ym.load_lora(merge_lora_dir, merge=True):
            raise RuntimeError(f"failed to load/merge LoRA adapters from {merge_lora_dir}")
    model = ym.model.eval()
    names = [ym.names[i] for i in sorted(ym.names)] if getattr(ym, "names", None) else []

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
    a = ap.parse_args()
    try:
        r = export(a.weights, a.imgsz, a.out, target=a.target, merge_lora_dir=a.merge_lora_dir,
                   yolov12_aattn=a.yolov12_aattn)
        print(f"OK  {r['out']}  task={r['task']}  output={r['output']}  classes={r['classes']}  "
              f"imgsz={r['imgsz']}  shapes={r['shapes']}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {a.weights}: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
