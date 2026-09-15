#!/usr/bin/env python
"""Pruned v0.1-N COCO -> ncnn (fp32 master + INT8), for the Orin-CPU tempo bench.

Trace-time MoE rewrite: the pruned checkpoints keep k==E==2 experts with
IMAGE-level routing, so sparse dispatch reduces EXACTLY to
    out = w0 * expert0(x) + w1 * expert1(x) + shared_expert(x) [+ residual]
with per-image scalar weights - no topk/gather in the graph, plain rank-4 ops
that pnnx and ncnn handle natively. Verified bit-consistent against the stock
eval forward before tracing.

Steps (x86; pnnx is x86-only):
  1. torch.jit.trace of the rewritten model  (venv-v2608)
  2. pnnx -> p03_v01n.ncnn.param/.bin        (fp32 master)
  3. ncnn2table (entropy-ish KL, 256 COCO train images, letterbox preprocessing)
  4. strip SENSITIVE layers from the table   (stem pair, router convs, decode tail)
  5. ncnn2int8 -> p03_v01n-int8.param/.bin

Usage:
  PYTHONPATH=/data/YOLO-Master /data/tmp/venv-v2608/bin/python \
      tempo-ncnn/export_ncnn_p03.py \
      --model runs/project03/pod-prune/v01-n-coco/YOLO-Master-v0.1-N_pruned_t0.10.pt
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from unittest import mock

import os

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
SDK = REPO / "third_party" / "ncnn-20260526-ubuntu-2204-shared" / "bin"
SDK_ENV = {**os.environ,
           "LD_LIBRARY_PATH": str(SDK.parent / "lib") + ":"
           + os.environ.get("LD_LIBRARY_PATH", "")}


class DenseMoE(nn.Module):
    """Exact dense equivalent of OptimizedMOEImproved at k==E (image routing)."""

    def __init__(self, moe: nn.Module):
        super().__init__()
        # parse_model wiring attrs the ultralytics executor requires
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
        # replicate the router inline, WITHOUT topk: at k==E the sparse weights
        # equal softmax(logits)/(1+1e-6), paired to experts by index natively
        # (the previous version mispaired: topk sorts weights by magnitude)
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
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        y = self.net(x)
        return y[0] if isinstance(y, (list, tuple)) else y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--calib-n", type=int, default=256)
    ap.add_argument("--out", default="tempo-ncnn/models")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    net = (ck.get("model") if isinstance(ck, dict) else ck).float().eval()

    # sanity: dense rewrite must match the stock eval forward before we trade it in
    moe_names = [n for n, m in net.named_modules()
                 if hasattr(m, "experts") and hasattr(m, "routing")]
    x = torch.randn(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        ref = net(x)[0] if isinstance(net(x), (list, tuple)) else net(x)
    for name in moe_names:
        parent = net.get_submodule(".".join(name.split(".")[:-1])) if "." in name else net
        idx = name.split(".")[-1]
        moe = net.get_submodule(name)
        assert len(moe.experts) == moe.top_k, \
            f"{name}: k={moe.top_k} != E={len(moe.experts)} - dense rewrite not exact"
        setattr(parent, idx, DenseMoE(moe))
    with torch.no_grad():
        got = net(x)
        got = got[0] if isinstance(got, (list, tuple)) else got
    err = float((ref - got).abs().max())
    print(f"[rewrite] {len(moe_names)} MoE modules -> dense; max|diff| {err:.2e}")
    assert err < 1e-2, "dense rewrite diverged from stock forward"

    head = net.model[-1]
    head.export = True
    head.format = "onnx"
    wrapper = DetectOnly(net).eval()
    with torch.no_grad(), mock.patch("torch.onnx.is_in_onnx_export", return_value=True):
        traced = torch.jit.trace(wrapper, torch.zeros(1, 3, args.imgsz, args.imgsz))
    ts = out / "p03_v01n.pt"
    traced.save(str(ts))

    # 2) pnnx
    pnnx_bin = Path(sys.executable).parent / "pnnx"
    subprocess.run([str(pnnx_bin), str(ts.resolve()),
                    f"inputshape=[1,3,{args.imgsz},{args.imgsz}]"],
                   check=True, cwd=out.resolve())
    param = out / "p03_v01n.ncnn.param"
    binf = out / "p03_v01n.ncnn.bin"
    assert param.exists() and binf.exists(), "pnnx did not produce ncnn files"
    print(f"[pnnx] {param}")

    # 3) calibration list (letterboxed handled by ncnn2table mean/norm + shape)
    imgs = sorted(Path("/data/datasets/coco/images/train2017").glob("*.jpg"))
    idx = np.linspace(0, len(imgs) - 1, args.calib_n).astype(int)
    lst = out / "calib.txt"
    lst.write_text("\n".join(str(imgs[i]) for i in idx))
    table = out / "p03_v01n.table"
    subprocess.run([str(SDK / "ncnn2table"), str(param), str(binf), str(lst), str(table),
                    "mean=[0,0,0]", "norm=[0.003922,0.003922,0.003922]",
                    f"shape=[{args.imgsz},{args.imgsz},3]", "pixel=RGB",
                    "thread=8", "method=kl"], check=True, env=SDK_ENV)
    print(f"[table] {table}")

    # 4) strip sensitive layers. pnnx names layers generically (conv_16), so
    # identify them STRUCTURALLY from the param graph: the first two convs are
    # the stem pair; out-channels==2 convs are the router logit heads (E=2);
    # out-channels==1 1x1 is the DFL. (Router reducer convs stay quantized -
    # tiny layers, logit head protected.)
    conv_names, sensitive = [], set()
    for line in param.read_text().splitlines():
        parts = line.split()
        if len(parts) > 4 and parts[0] in ("Convolution", "ConvolutionDepthWise"):
            name = parts[1]
            outch = next((int(p[2:]) for p in parts if p.startswith("0=")), -1)
            conv_names.append(name)
            if outch in (1, 2):
                sensitive.add(name)
    sensitive.update(conv_names[:2])                      # stem pair
    keep, dropped = [], []
    for line in table.read_text().splitlines():
        lname = line.split()[0] if line.split() else ""
        base = lname[:-len("_param_0")] if lname.endswith("_param_0") else lname
        if base in sensitive:
            dropped.append(lname)
        else:
            keep.append(line)
    table_s = out / "p03_v01n_sensitive.table"
    table_s.write_text("\n".join(keep) + "\n")
    print(f"[table] stripped {len(dropped)} sensitive entries: {dropped[:6]}...")

    # 5) int8 model
    subprocess.run([str(SDK / "ncnn2int8"), str(param), str(binf),
                    str(out / "p03_v01n-int8.param"), str(out / "p03_v01n-int8.bin"),
                    str(table_s)], check=True, env=SDK_ENV)
    print(f"[int8] {out}/p03_v01n-int8.param")
    for f in sorted(out.glob("p03_v01n*")):
        print(f"  {f.name}  {f.stat().st_size/1e6:.1f}MB")


if __name__ == "__main__":
    main()
