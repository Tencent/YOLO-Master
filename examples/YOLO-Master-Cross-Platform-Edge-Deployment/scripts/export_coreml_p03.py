#!/usr/bin/env python
"""Project03 iPhone path: pruned v0.1-N COCO -> Core ML .mlpackage, fp16 + INT8-W8.

Runs in /data/tmp/venv-coreml (py3.11, torch 2.5.1, coremltools 9.0) with
PYTHONPATH=/data/YOLO-Master. Linux converts and weight-quantizes; ACTIVATION
quantization (A8) needs a Core ML runtime and therefore runs on the Mac side:
see mac/quantize_a8_p03.py, which consumes the fp16 package produced here.

Deployment context (from the phase-2 study): this model is activation-bound, so
W8 buys size/bandwidth (half the weight bytes, ANE-friendly) rather than big
latency; A8 is the experiment for actual speed. ANE is fp16-native, so fp16 is
the expected latency baseline on iPhone as it is on every GPU we measured.

Outputs (models/p03_coreml/):
  p03_v01n_coco_fp16.mlpackage      baseline, ANE-ready
  p03_v01n_coco_w8.mlpackage        INT8 weights, fp16 activations
  p03_ref.npz                       input blob + torch reference output (parity)

Usage:
  PYTHONPATH=/data/YOLO-Master /data/tmp/venv-coreml/bin/python \
      scripts/export_coreml_p03.py \
      --model runs/project03/pod-prune/v01-n-coco/YOLO-Master-v0.1-N_pruned_t0.10.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from unittest import mock

import numpy as np
import torch
import torch.nn as nn


class DenseMoE(nn.Module):
    """Exact dense equivalent of OptimizedMOEImproved at k==E (image routing).

    Ported from tempo-ncnn/export_ncnn_p03.py (numerically validated there).
    Why: the stock export branch stacks all experts and torch.gather-s the
    top-k, which lands 2 gather_along_axis ops per MoE layer in the MIL graph.
    Gathers do not stay ANE-resident, so every MoE block forces a partition
    split + ANE roundtrip - measured 2.2x model latency on iPhone (18.7ms vs
    8.4ms for a comparable gather-free graph). At k==E the gather is pointless:
    per-expert weight = router softmax, so the whole block reduces to convs +
    softmax + weighted sum - fully ANE-mappable.
    """

    def __init__(self, moe: nn.Module):
        super().__init__()
        for a in ("f", "i", "type", "np"):
            if hasattr(moe, a):
                setattr(self, a, getattr(moe, a))
        self.routing = moe.routing
        self.experts = moe.experts
        self.shared_expert = moe.shared_expert
        self.add_residual = bool(getattr(moe, "add_residual", False)) and \
            moe.in_channels == moe.out_channels
        self.E = len(moe.experts)
        self.top_k = self.E

    def forward(self, x):
        # inline router, WITHOUT topk: at k==E the sparse weights equal
        # softmax(logits)/(1+1e-6), paired to experts by index natively
        r = self.routing
        if hasattr(r, "pool_scale") and x.shape[2] > r.pool_scale and x.shape[3] > r.pool_scale:
            x_in = torch.nn.functional.avg_pool2d(x, r.pool_scale, r.pool_scale)
        elif hasattr(r, "avg_pool"):
            x_in = r.avg_pool(x)
        else:
            x_in = x
        logits = r.router(x_in).mean(dim=[2, 3])                 # [B, E]
        probs = torch.softmax(logits, dim=1)
        w = probs / (probs.sum(dim=1, keepdim=True) + 1e-6)      # renorm, exact
        out = self.shared_expert(x)
        for e in range(self.E):
            out = out + w[:, e].view(-1, 1, 1, 1) * self.experts[e](x)
        if self.add_residual:
            out = out + x
        return out


class DetectOnly(nn.Module):
    """Trace wrapper: single detection tensor out (ONNX-consistent semantics)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        y = self.net(x)
        return y[0] if isinstance(y, (list, tuple)) else y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="models/p03_coreml")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights)

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    net = (ck.get("model") if isinstance(ck, dict) else ck).float().eval()

    # swap every MoE block for its dense k==E equivalent, verified exact first
    moe_names = [n for n, m in net.named_modules()
                 if hasattr(m, "experts") and hasattr(m, "routing")]
    x = torch.randn(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        ref = net(x)
        ref = ref[0] if isinstance(ref, (list, tuple)) else ref
    for name in moe_names:
        parent = net.get_submodule(".".join(name.split(".")[:-1])) if "." in name else net
        moe = net.get_submodule(name)
        assert len(moe.experts) == moe.top_k, \
            f"{name}: k={moe.top_k} != E={len(moe.experts)} - dense rewrite not exact"
        setattr(parent, name.split(".")[-1], DenseMoE(moe))
    with torch.no_grad():
        got = net(x)
        got = got[0] if isinstance(got, (list, tuple)) else got
    err = float((ref - got).abs().max())
    print(f"[rewrite] {len(moe_names)} MoE modules -> dense; max|diff| {err:.2e}")
    assert err < 1e-2, "dense rewrite diverged from stock forward"

    head = net.model[-1]
    head.export = True
    head.format = "onnx"          # same single-tensor semantics as every other export

    wrapper = DetectOnly(net).eval()
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
    # the mock stays for any remaining traced-branch parity with the ncnn path;
    # the MoE gather branch it used to trigger is gone (DenseMoE above)
    with torch.no_grad(), mock.patch("torch.onnx.is_in_onnx_export", return_value=True):
        ref_out = wrapper(dummy).numpy()
        traced = torch.jit.trace(wrapper, dummy)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="images", shape=(1, 3, args.imgsz, args.imgsz))],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    # metadata the Swift Detector reads: without `names` the app labels
    # detections classN; without `output`/`task` it guesses
    names = getattr(net, "names", None) or {}
    names_csv = ",".join(str(names.get(i, i)) for i in range(len(names)))
    out_name = mlmodel.get_spec().description.output[0].name
    for k, v in {"task": "detect", "output": out_name,
                 "imgsz": str(args.imgsz), "names": names_csv}.items():
        mlmodel.user_defined_metadata[k] = v
    fp16_path = out / "p03_v01n_coco_fp16.mlpackage"
    mlmodel.save(str(fp16_path))
    print(f"[coreml] {fp16_path}")

    w8_cfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel"))
    w8 = linear_quantize_weights(mlmodel, config=w8_cfg)
    for k, v in mlmodel.user_defined_metadata.items():
        w8.user_defined_metadata[k] = v
    w8_path = out / "p03_v01n_coco_w8.mlpackage"
    w8.save(str(w8_path))
    print(f"[coreml] {w8_path}")

    # parity reference: fixed random image-like blob + torch output
    rng = np.random.default_rng(0)
    blob = rng.random((1, 3, args.imgsz, args.imgsz), dtype=np.float32)
    with torch.no_grad(), mock.patch("torch.onnx.is_in_onnx_export", return_value=True):
        ref = wrapper(torch.from_numpy(blob)).numpy()
    np.savez_compressed(out / "p03_ref.npz", input=blob, output=ref)
    print(f"[coreml] p03_ref.npz (zero-input ref max |out| {np.abs(ref_out).max():.3f})")


if __name__ == "__main__":
    main()
