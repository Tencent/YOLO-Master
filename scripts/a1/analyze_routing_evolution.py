"""A1 MoE 路由训练演化:集中度 vs epoch。

对 B 格 save_period 断点(epoch0~90 + best)逐一跑路由分布检查,
输出每层专家使用集中度随 epoch 的演化,判定:
- 集中度仍在变化(未平台)→ 路由未收敛
- 集中度已平台 / 持续上升 → 收敛于失衡态

用法:
    python scripts/a1/analyze_routing_evolution.py --weights-dir <weights目录> [--n-imgs 100]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # YOLO-Master 仓库根
sys.path.insert(0, str(REPO_ROOT))

os.environ["WANDB_MODE"] = "disabled"
os.environ.setdefault("YOLO_VERBOSE", "false")

import cv2
import torch

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.nn.modules.moe.analysis import ExpertUsageTracker


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights-dir", required=True, help="B 格 weights 目录(含 epochN.pt/best.pt)")
    ap.add_argument("--imgs-dir", default="/root/gpufree-data/datasets/coco-train-2017/images/val")
    ap.add_argument("--n-imgs", type=int, default=100, help="采样图数(seed=0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0")
    return ap.parse_args()


def layer_concentration(tracker: ExpertUsageTracker) -> dict[str, float]:
    """每层专家使用集中度(最大份额 ÷ 理想份额)。"""
    out = {}
    for name, stats in tracker.usage_stats.items():
        n_experts = len(stats)
        hits = [stats[i].hits for i in range(n_experts)]
        total = sum(hits)
        if total <= 0 or n_experts == 0:
            continue
        share = [h / total for h in hits]
        out[name] = max(share) / (1.0 / n_experts)
    return out


def main() -> None:
    """逐断点采集集中度并打印演化表。"""
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)

    weights_dir = Path(args.weights_dir)
    ckpts = sorted(weights_dir.glob("epoch*.pt"), key=lambda p: int(p.stem.replace("epoch", "")))
    ckpts += [weights_dir / "best.pt"]
    print(f"断点: {[p.name for p in ckpts]}")

    images = sorted(Path(args.imgs_dir).glob("*.jpg"))
    rng = random.Random(args.seed)
    picked = [images[i] for i in sorted(rng.sample(range(len(images)), min(args.n_imgs, len(images))))]

    # 固定同一批图输入(跨断点可比)
    stride = 32
    lb = LetterBox((640, 640), auto=False, stride=stride)
    xs = []
    for p in picked:
        im = lb(image=cv2.imread(str(p)))
        im = im[..., ::-1].transpose(2, 0, 1)
        xs.append(torch.from_numpy(im.copy()).contiguous().float().unsqueeze(0) / 255.0)

    rows: dict[str, list] = {"epoch": []}
    for ckpt in ckpts:
        model = YOLO(str(ckpt)).model
        model.eval().to(device)
        tracker = ExpertUsageTracker(model)
        with torch.no_grad():
            for x in xs:
                model(x.to(device))
        conc = layer_concentration(tracker)
        epoch = int(ckpt.stem.replace("epoch", "")) if ckpt.stem.startswith("epoch") else 100
        rows["epoch"].append(epoch)
        for layer, c in conc.items():
            rows.setdefault(layer, []).append(round(c, 2))
        print(
            f"epoch {epoch:>4}: "
            + " | ".join(
                f"{l.split('model.')[1].split('.m')[0] if 'model.' in l else l}: {c:.2f}" for l, c in conc.items()
            )
        )
        del model
        torch.cuda.empty_cache()

    # 收敛判定:最后两个断点集中度差
    print("\n== 收敛判定(末段 Δ = 最后两断点集中度之差)== ")
    for layer, vals in rows.items():
        if layer == "epoch":
            continue
        delta = vals[-1] - vals[-2]
        print(f"{layer}: 末期集中度 {vals[-2]:.2f} → {vals[-1]:.2f}(Δ {delta:+.2f})")


if __name__ == "__main__":
    main()
