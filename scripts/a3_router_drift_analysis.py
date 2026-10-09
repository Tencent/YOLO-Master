#!/usr/bin/env python3
"""A3 P1 — 路由决策漂移分析（EsMoE-N，dense softmax 路由版）。

在锁定 EsMoE-N 权重上，对每个 ES_MOE 块的路由层权重做三档精度模拟
（FP32 基线 / FP16 / INT8），量化路由权重后 forward 同一批图，比较
路由 softmax 权重的分布漂移与 argmax 专家一致率。

设计要点：
- 隔离变量：全程 fp32 计算，唯一变量是「路由层权重的精度」。
- FP16 模拟 = w.half().float()；INT8 模拟 = per-tensor 对称量化再反量化。
- 只量化 routing 层（DynamicRoutingLayer），专家/其他层保持 FP32。
- 输出 per-block/per-image 的 argmax 一致率 + 权重 MAE + 汇总 JSON/CSV。
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


def quantize_int8(w: torch.Tensor) -> torch.Tensor:
    """Per-tensor symmetric int8 quantization then dequantize (simulate)."""
    w = w.float()
    scale = w.abs().max() / 127.0
    scale = torch.clamp(scale, min=1e-8)
    q = torch.clamp(torch.round(w / scale), -127, 127)
    return q * scale


def quantize_fp16(w: torch.Tensor) -> torch.Tensor:
    """Simulate fp16 storage precision (cast down then back to fp32)."""
    return w.float().half().float()


def collect_image_paths(data_yaml: Path | None, source: Path | None, limit: int) -> list[Path]:
    if source is not None:
        if source.is_file() and source.suffix == ".txt":
            root = source.parent
            return [Path(x.strip()) for x in source.read_text().splitlines() if x.strip()][:limit]
        if source.is_file():
            return [source]
        return sorted(x for x in source.rglob("*") if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})[:limit]
    # fall back to COCO8 val dir next to this script's repo root
    root = Path(__file__).resolve().parents[1]
    val_dir = root / "datasets" / "coco8" / "images" / "val"
    return sorted(val_dir.glob("*"))[:limit]


def preprocess(path: Path, imgsz: int, device: torch.device) -> torch.Tensor:
    import cv2

    im = cv2.imread(str(path))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(im).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return x.to(device)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="yolo_master_n.pt")
    ap.add_argument("--data", type=Path, help="optional dataset yaml (unused, kept for API symmetry)")
    ap.add_argument("--source", type=Path, help="image dir / file / txt")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("runs/a3_router_drift"))
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    yolo = YOLO(args.model)
    model = yolo.model.eval().to(device)

    # 定位 ES_MOE 路由层
    esmoe_routings: list[tuple[str, torch.nn.Module]] = [
        (name, mod.routing) for name, mod in model.named_modules() if isinstance(mod, ES_MOE)
    ]
    if not esmoe_routings:
        raise SystemExit("no ES_MOE blocks found")

    # 缓存路由层原权重
    routing_params = {name: [p.data.clone() for p in mod.parameters()] for name, mod in esmoe_routings}

    captures: dict[str, list[torch.Tensor]] = {}
    hooks = []
    for name, mod in esmoe_routings:
        captures[name] = []

        def make_hook(n=name):
            def hook(_m, _i, out):
                captures[n].append(out.detach())

            return hook

        hooks.append(mod.register_forward_hook(make_hook(name)))

    images = collect_image_paths(args.data, args.source, args.limit)
    if not images:
        raise SystemExit("no images found")

    precisions = {"fp32": None, "fp16": quantize_fp16, "int8": quantize_int8}

    records: list[dict] = []

    def run_pass(prec_name: str) -> dict[str, torch.Tensor]:
        """forward 一批图，返回 {block_name: 堆叠的 routing weights [N,E,H,W]}"""
        # 直接替换 routing 层参数
        for name, mod in esmoe_routings:
            fn = precisions[prec_name]
            for p in mod.parameters():
                if fn is not None:
                    p.data = fn(p.data)

        # 清空捕获
        for k in captures:
            captures[k].clear()

        with torch.inference_mode():
            for path in images:
                x = preprocess(path, args.imgsz, device)
                _ = model(x)

        # 恢复原权重
        for name, mod in esmoe_routings:
            for p, orig in zip(mod.parameters(), routing_params[name]):
                p.data = orig

        out: dict[str, torch.Tensor] = {}
        for name in captures:
            if captures[name]:
                out[name] = torch.cat([t.cpu() for t in captures[name]], dim=0)
        return out

    fp32 = run_pass("fp32")
    fp16 = run_pass("fp16")
    int8 = run_pass("int8")

    summary = {"model": args.model, "images": len(images), "blocks": {}}
    for name in fp32:
        w32 = fp32[name]  # [N,E,H,W]
        block = {"fp32_vs_fp16": {}, "fp32_vs_int8": {}}
        for tag, wq in (("fp32_vs_fp16", fp16[name]), ("fp32_vs_int8", int8[name])):
            argmax32 = w32.argmax(dim=1)
            argmax_q = wq.argmax(dim=1)
            agree = (argmax32 == argmax_q).float().mean().item()
            mae = (w32 - wq).abs().mean().item()
            # 逐图 argmax 一致率
            per_img = [
                (argmax32[i] == argmax_q[i]).float().mean().item() for i in range(w32.shape[0])
            ]
            block[tag] = {"argmax_agree": agree, "weight_mae": mae, "per_image_agree": per_img}
            records.append(
                {
                    "block": name,
                    "comparison": tag,
                    "argmax_agree": agree,
                    "weight_mae": mae,
                }
            )
        summary["blocks"][name] = block

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "router_drift_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.out / "router_drift_records.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["block", "comparison", "argmax_agree", "weight_mae"])
        writer.writeheader()
        writer.writerows(records)

    # 控制台摘要
    print(f"\n{'block':<12} {'FP16 agree':>12} {'FP16 MAE':>10} {'INT8 agree':>12} {'INT8 MAE':>10}")
    for name in summary["blocks"]:
        b = summary["blocks"][name]
        f16 = b["fp32_vs_fp16"]
        i8 = b["fp32_vs_int8"]
        print(f"{name:<12} {f16['argmax_agree']*100:>10.2f}% {f16['weight_mae']:>10.2e} {i8['argmax_agree']*100:>10.2f}% {i8['weight_mae']:>10.2e}")
    print(f"\nwrote {args.out}")

    for h in hooks:
        h.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
