#!/usr/bin/env python3
"""A3 P1 — 路由决策 margin 分析（EsMoE-N 辅助）。

量化 argmax 零翻转的归因：若 softmax 的 top-1 与 top-2 权重差距（margin）
远大于量化扰动（~1e-4），则零翻转是结构性鲁棒；若 margin 与扰动量级相当，
则零翻转带有运气成分。本脚本只在 FP32 下 forward 一次，逐层统计 margin 分布。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.nn.modules.moe.modules import ES_MOE


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="yolo_master_n.pt")
    ap.add_argument("--source", type=Path, default=Path("data/visdrone/VisDrone2019-DET-val/images"))
    ap.add_argument("--limit", type=int, default=548)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", type=Path, default=Path("runs/a3_router_drift_visdrone548"))
    args = ap.parse_args()

    device = torch.device("cpu")
    yolo = YOLO(args.model)
    model = yolo.model.eval().to(device)

    esmoe_routings = [(n, m.routing) for n, m in model.named_modules() if isinstance(m, ES_MOE)]
    captures = {n: [] for n, _ in esmoe_routings}
    hooks = []
    for n, mod in esmoe_routings:
        def h(_m, _i, out, name=n):
            captures[name].append(out.detach())
        hooks.append(mod.register_forward_hook(h))

    images = sorted(args.source.glob("*"))[: args.limit]
    import cv2

    with torch.inference_mode():
        for p in images:
            im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
            x = torch.from_numpy(im).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            _ = model(x.to(device))

    for hk in hooks:
        hk.remove()

    summary = {}
    for name in captures:
        w = torch.cat(captures[name], dim=0)  # [N,E,H,W]
        N, E = w.shape[0], w.shape[1]
        top2 = torch.topk(w, 2, dim=1).values  # [N,2,H,W]
        margin = (top2[:, 0] - top2[:, 1])  # [N,H,W]
        m = margin.flatten().numpy()
        summary[name] = {
            "experts": E,
            "margin_mean": float(m.mean()),
            "margin_median": float(np.median(m)),
            "margin_min": float(m.min()),
            "margin_p1": float(np.percentile(m, 1)),
            "margin_p5": float(np.percentile(m, 5)),
            "frac_margin_below_1e-3": float((m < 1e-3).mean()),
            "frac_margin_below_1e-4": float((m < 1e-4).mean()),
        }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "router_margin_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n{'block':<12} {'margin_mean':>12} {'margin_median':>14} {'margin_min':>12} {'<1e-3':>8} {'<1e-4':>8}")
    for name, s in summary.items():
        print(
            f"{name:<12} {s['margin_mean']:>12.4f} {s['margin_median']:>14.4f} {s['margin_min']:>12.4e} "
            f"{s['frac_margin_below_1e-3']*100:>7.2f}% {s['frac_margin_below_1e-4']*100:>7.2f}%"
        )
    print(f"\nwrote {args.out / 'router_margin_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
