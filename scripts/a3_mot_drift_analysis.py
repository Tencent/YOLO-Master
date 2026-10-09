#!/usr/bin/env python3
"""A3 P1 — MoT top-k 稀疏路由决策漂移分析。

MoT 是 A3 的稀疏路由主角：router 逐 token 输出 top-k 专家索引
indices[B,K,H,W]，是离散决策。本脚本对比「FP32 vs 量化路由权重」下
top-k 专家集合的一致率，验证量化对稀疏路由决策的冲击。

重要说明（方法学）：
- MoT 缺公开预训练权重，且其 router 末层被显式初始化为 0（logits 恒 0，
  路由退化）。因此本脚本注入非退化随机权重（kaiming），得到一个
  「未训练但非退化」的路由，测的是漂移现象与方法学口径，非真实精度数字。
  真实数字待 MoT 权重训出后重跑。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.nn.modules.mot import MoTBlock


def quantize_symmetric(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-tensor symmetric quantization to `bits` bits, then dequantize."""
    w = w.float()
    if bits >= 16:
        return w.half().float() if bits == 16 else w
    qmax = 2 ** (bits - 1) - 1
    scale = torch.clamp(w.abs().max() / qmax, min=1e-8)
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    return q * scale


def inject_nondegenerate_routers(model: nn.Module) -> None:
    """Replace MoT router's zero-init weights with non-degenerate random weights.

    MoT's _MoTRouter zero-inits its last conv, making logits ≡ 0 and routing
    degenerate. Inject kaiming weights so top-k decisions become meaningful.

    自动检测：仅当 router 末层权重全零（退化）时才注入；真实训练权重（非全零）
    会被跳过，避免污染。
    """
    injected = 0
    for name, mod in model.named_modules():
        if isinstance(mod, MoTBlock):
            params = list(mod.router.parameters())
            last_w = params[-2] if len(params) >= 2 else params[-1]  # 末层 conv weight
            if torch.count_nonzero(last_w) == 0:  # 退化（零初始化）才注入
                for p in mod.router.parameters():
                    if p.dim() >= 2:
                        nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")
                    else:
                        nn.init.normal_(p, mean=0.0, std=0.1)
                injected += 1
    if injected:
        print(f"[inject] {injected} degenerate MoT routers injected with random weights")
    else:
        print("[inject] all MoT routers non-degenerate — skipped (real weights preserved)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="ultralytics/cfg/models/26/yolo26-master-mot-n.yaml")
    ap.add_argument("--source", type=Path, default=Path("data/visdrone/VisDrone2019-DET-val/images"))
    ap.add_argument("--limit", type=int, default=548)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--bits", type=int, nargs="+", default=[16, 8, 4, 2])
    ap.add_argument("--out", type=Path, default=Path("runs/a3_mot_drift"))
    args = ap.parse_args()

    device = torch.device("cpu")
    yolo = YOLO(args.model)
    model = yolo.model.eval().to(device)
    inject_nondegenerate_routers(model)

    mot_routers = [(n, m.router) for n, m in model.named_modules() if isinstance(m, MoTBlock)]
    if not mot_routers:
        raise SystemExit("no MoTBlock found")

    routing_params = {n: [p.data.clone() for p in m.parameters()] for n, m in mot_routers}
    captures: dict[str, list[torch.Tensor]] = {n: [] for n, _ in mot_routers}
    hooks = []
    for n, mod in mot_routers:
        def make_hook(name=n):
            def h(_m, _i, out):
                # out = (weights, indices, logits) with return_logits=True
                captures[name].append(out[1].detach())  # indices [B,K,H,W]
            return h
        hooks.append(mod.register_forward_hook(make_hook(n)))

    _suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in args.source.rglob("*") if p.suffix.lower() in _suffixes)[: args.limit]
    import cv2

    def preprocess(p: Path) -> torch.Tensor:
        im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(im).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    def run_pass(bits: int | None) -> dict[str, torch.Tensor] | None:
        for _n, mod in mot_routers:
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
        except Exception as exc:
            print(f"    [bits={bits}] forward failed: {type(exc).__name__}: {exc}")
            failed = True
        for n, mod in mot_routers:
            for p, orig in zip(mod.parameters(), routing_params[n]):
                p.data = orig
        if failed:
            return None
        return {n: torch.cat(captures[n], dim=0) for n in captures if captures[n]}

    baseline = run_pass(None)
    if baseline is None:
        raise SystemExit("FP32 baseline failed")
    results: dict[str, dict[str, float]] = {}
    records: list[dict] = []
    topks: dict[str, int] = {}

    for bits in args.bits:
        q = run_pass(bits)
        tag = f"int{bits}" if bits < 16 else "fp16"
        if q is None:
            for n in baseline:
                results.setdefault(n, {})[tag] = float("nan")
            continue
        for n in baseline:
            i32, iq = baseline[n], q[n]  # [N,K,H,W]
            K = i32.shape[1]
            topks[n] = K
            # top-k 集合一致：对每个 token，K 个专家索引完全一致（topk 降序，直接比）
            agree = (i32 == iq).all(dim=1).float().mean().item()
            results.setdefault(n, {})[tag] = agree
            records.append({"block": n, "top_k": K, "bits": tag, "topk_agree": agree})

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "mot_drift.json").write_text(
        json.dumps({"images": len(images), "blocks": results, "top_k": topks}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (args.out / "mot_drift.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["block", "top_k", "bits", "topk_agree"])
        w.writeheader()
        w.writerows(records)

    tags = [f"int{b}" if b < 16 else "fp16" for b in args.bits]
    print("\ntop-k 专家集合一致率 (FP32 baseline = 100%)")
    print(f"{'block':<16}{'top_k':>7}" + "".join(f"{t:>12}" for t in tags))
    for n in baseline:
        row = f"{n:<16}{topks[n]:>7}"
        for t in tags:
            v = results[n].get(t, float("nan"))
            row += f"{v*100:>11.2f}%" if np.isfinite(v) else f"{'FAIL':>12}"
        print(row)
    print(f"\nwrote {args.out}")

    for h in hooks:
        h.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
