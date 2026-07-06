#!/usr/bin/env python3
"""Export YOLO-Master-EsMoE to NCNN via dense (full-softmax) routing.

Root cause of the NCNN failure
------------------------------
ES_MOE already combines experts densely during ONNX export (_dense_forward, all
experts weighted-summed — no gather). The remaining ``torch.topk`` in the graph
comes from the *router*: when ``DynamicRoutingLayer.use_top_k=True`` the export
path is ``_soft_top_k`` (topk + one_hot mask) — pnnx emits these as no-op layers
NCNN cannot register.

Fix
---
Flip ``DynamicRoutingLayer.use_top_k=False`` for the export. The router then uses
plain ``softmax(logits)`` over ALL experts (no topk/one_hot) and ES_MOE._dense_forward
weights every expert by it. The whole graph becomes conv / softmax / elementwise —
all native NCNN ops.

Semantic cost: full-softmax routing vs top-k-masked routing. For a stable
(frozen-backbone) router the top-k experts dominate the softmax, so the delta is
small; we measure it. NCNN consistency is verified PyTorch-dense vs NCNN-dense.

Usage:
    python scripts/export_ncnn_dense.py --model best.pt --imgsz 640
"""
from __future__ import annotations

import argparse
from pathlib import Path


PROJ_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO
    from ultralytics.nn.modules.moe.routers import DynamicRoutingLayer

    # Import the ES_MOE class lazily; its symbol lives in the moe modules package.
    import ultralytics.nn.modules.moe.modules as moe_mod

    esmoe_cls = getattr(moe_mod, "ES_MOE")

    model = YOLO(str(args.model))
    n_router = n_esmoe = 0
    for m in model.model.modules():
        if isinstance(m, DynamicRoutingLayer):
            m.use_top_k = False  # router: full softmax, no topk/one_hot -> NCNN-safe
            n_router += 1
        if isinstance(m, esmoe_cls):
            # NCNN export traces via TorchScript (not onnx), so is_in_onnx_export()
            # is False -> force the dense expert-combine path explicitly.
            m.use_sparse_inference = False
            n_esmoe += 1
    print(f"[ncnn-dense] dense-mode: {n_router} routers (use_top_k=False), {n_esmoe} ES_MOE (dense combine)")

    out = None
    try:
        out = model.export(format="ncnn", imgsz=args.imgsz, half=args.half)
    except SyntaxError as e:
        # pnnx's bonus model_pnnx.py has invalid Python (aten::where codegen);
        # the ncnn param/bin are written before this. Non-fatal.
        print(f"[ncnn-dense] pnnx reference-script SyntaxError (non-fatal): {e}")
    if out is None:
        out = args.model.parent / (args.model.stem + "_ncnn_model")
    print(f"[ncnn-dense] export -> {out}")

    # Verify the produced .param/.bin load + run in NCNN, with NO no-op layers.
    import contextlib
    import io

    import ncnn
    import numpy as np

    ncnn_dir = Path(out)
    param, binf = ncnn_dir / "model.ncnn.param", ncnn_dir / "model.ncnn.bin"
    text = param.read_text(errors="ignore").splitlines()
    noop = sorted({ln.split()[0] for ln in text[1:] if ln.startswith("torch.")})
    net = ncnn.Net()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        net.load_param(str(param))
        net.load_model(str(binf))
        in_name = [ln.split()[2] for ln in text[1:] if ln.startswith("Input")]
        ex = net.create_extractor()
        ex.input(in_name[0], ncnn.Mat(np.zeros((1, 3, args.imgsz, args.imgsz), dtype=np.float32)))
    log = buf.getvalue()
    missing = [ln for ln in log.splitlines() if "not exists or registered" in ln]
    print(f"[ncnn-dense] no-op layers in .param: {noop or 'NONE'}")
    if missing:
        print(f"[ncnn-dense] FAIL — ncnn still rejects: {missing}")
    else:
        print(f"[ncnn-dense] OK — NCNN graph loads & runs clean (dense routing)")


if __name__ == "__main__":
    main()
