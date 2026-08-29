#!/usr/bin/env python3
"""A3 P1 — 路由层量化位宽扫描（EsMoE-N）。

对每个 ES_MOE 块的路由层权重做 {16,8,4,2}-bit 对称量化（模拟），
对比 FP32 基线的 argmax 专家一致率与权重 MAE，验证「深层块在更激进
量化下最先失稳」的预判（深层 margin≈0.02，浅层 margin≈0.54）。

隔离变量：全程 fp32 计算，唯一变量是路由层权重精度。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.nn.modules.moe.modules import ES_MOE


def quantize_symmetric(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-tensor symmetric quantization to `bits` bits, then dequantize.

    int8 -> qmax 127, int4 -> 7, int2 -> 1 (values -1/0/+1).
    """
    w = w.float()
    if bits >= 16:
        return w.half().float() if bits == 16 else w
    qmax = 2 ** (bits - 1) - 1
    scale = torch.clamp(w.abs().max() / qmax, min=1e-8)
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    return q * scale


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="yolo_master_n.pt")
    ap.add_argument("--source", type=Path, default=Path("data/visdrone/VisDrone2019-DET-val/images"))
    ap.add_argument("--limit", type=int, default=548)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--bits", type=int, nargs="+", default=[16, 8, 4, 2])
    ap.add_argument("--out", type=Path, default=Path("runs/a3_router_bitwidth_sweep"))
    args = ap.parse_args()

    device = torch.device("cpu")
    yolo = YOLO(args.model)
    model = yolo.model.eval().to(device)

    esmoe_routings = [(n, m.routing) for n, m in model.named_modules() if isinstance(m, ES_MOE)]
    if not esmoe_routings:
        raise SystemExit("no ES_MOE blocks found")

    routing_params = {n: [p.data.clone() for p in m.parameters()] for n, m in esmoe_routings}
    captures: dict[str, list[torch.Tensor]] = {n: [] for n, _ in esmoe_routings}
    hooks = []
    for n, mod in esmoe_routings:
        def make_hook(name=n):
            def h(_m, _i, out):
                captures[name].append(out.detach())
            return h
        hooks.append(mod.register_forward_hook(make_hook(n)))

    _img_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in args.source.rglob("*") if p.suffix.lower() in _img_suffixes)[: args.limit]
    import cv2

    def preprocess(p: Path) -> torch.Tensor:
        im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(im).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    def run_pass(bits: int | None) -> dict[str, torch.Tensor] | None:
        """forward 一批图，返回 {block: stacked routing weights [N,E,H,W]}。bits=None 为 FP32 基线。"""
        # 应用量化
        for _n, mod in esmoe_routings:
            if bits is not None:
                for p in mod.parameters():
                    p.data = quantize_symmetric(p.data, bits)
        for k in captures:
            captures[k].clear()
        failed = False
        try:
            with torch.inference_mode():
                for p in images:
                    _ = model(preprocess(p).to(device))
        except Exception as exc:  # 极端量化可能触发路由 finite 检查
            print(f"    [bits={bits}] forward failed: {type(exc).__name__}: {exc}")
            failed = True
        # 恢复
        for n, mod in esmoe_routings:
            for p, orig in zip(mod.parameters(), routing_params[n]):
                p.data = orig
        if failed:
            return None
        return {n: torch.cat(captures[n], dim=0) for n in captures if captures[n]}

    baseline = run_pass(None)  # FP32
    if baseline is None:
        raise SystemExit("FP32 baseline forward failed — aborting")
    results: dict[str, dict[str, float]] = {}
    records: list[dict] = []

    for bits in args.bits:
        q = run_pass(bits)
        tag = f"int{bits}" if bits < 16 else "fp16"
        if q is None:
            print(f"[{tag}] FAILED — marked as NaN")
            for n in baseline:
                results.setdefault(n, {})[tag] = float("nan")
            continue
        for n in baseline:
            w32, wq = baseline[n], q[n]
            agree = (w32.argmax(1) == wq.argmax(1)).float().mean().item()
            mae = (w32 - wq).abs().mean().item()
            results.setdefault(n, {})[tag] = agree
            results.setdefault(n, {})[f"{tag}_mae"] = mae
            records.append({"block": n, "bits": tag, "argmax_agree": agree, "weight_mae": mae})

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bitwidth_sweep.json").write_text(
        json.dumps({"images": len(images), "blocks": results}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.out / "bitwidth_sweep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["block", "bits", "argmax_agree", "weight_mae"])
        writer.writeheader()
        writer.writerows(records)

    # 控制台表格
    tags = [f"int{b}" if b < 16 else "fp16" for b in args.bits]
    header = f"{'block':<12}" + "".join(f"{t:>14}" for t in tags)
    print("\nargmax 专家一致率 (FP32 baseline = 100%)")
    print(header)
    for n in baseline:
        row = f"{n:<12}"
        for t in tags:
            v = results[n].get(t, float("nan"))
            row += f"{v*100:>13.2f}%" if np.isfinite(v) else f"{'FAIL':>14}"
        print(row)
    print("\n权重 MAE")
    print(header)
    for n in baseline:
        row = f"{n:<12}"
        for t in tags:
            v = results[n].get(f"{t}_mae", float("nan"))
            row += f"{v:>14.2e}" if np.isfinite(v) else f"{'FAIL':>14}"
        print(row)
    print(f"\nwrote {args.out}")

    for h in hooks:
        h.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
