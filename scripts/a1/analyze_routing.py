"""A1 MoE 路由健康检查:B 格 20k 权重的专家选择分布。

用项目资产 ExpertUsageTracker(ultralytics/nn/modules/moe/analysis.py)在推理态采集:
- 每层专家被 top-k 选中的 token 数(hits)与累计权重
- hits 分布集中度(最大专家占比)与均衡度
判定:若 hits 高度集中在少数专家(集中度 ≫ 1/E)或路由熵极低 → 分工未形成/坍塌;
若接近均匀 → balance loss 强制均衡占主导,同样说明"分工红利"未兑现。

用法:
    python scripts/a1/analyze_routing.py --model <B格best.pt> [--n-imgs 200]
"""

from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--model", required=True, help="B 格(MoE)best.pt")
    ap.add_argument("--imgs-dir", default="/root/gpufree-data/datasets/coco-train-2017/images/val")
    ap.add_argument("--n-imgs", type=int, default=200, help="采样图数(seed=0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=None, help="JSON 输出路径(默认打印)")
    return ap.parse_args()


def main() -> None:
    """采集并报告路由分布。"""
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)

    yolo = YOLO(args.model)
    model = yolo.model
    model.eval().to(device)
    stride = int(max(model.stride.tolist()))

    # 固定 seed 抽样(与评测同机制)
    images = sorted(Path(args.imgs_dir).glob("*.jpg"))
    rng = random.Random(args.seed)
    picked = [images[i] for i in sorted(rng.sample(range(len(images)), min(args.n_imgs, len(images))))]
    print(f"采集 {len(picked)} 张图")

    tracker = ExpertUsageTracker(model)
    lb = LetterBox((640, 640), auto=False, stride=stride)
    with torch.no_grad():
        for p in picked:
            img0 = cv2.imread(str(p))
            im = lb(image=img0)
            im = im[..., ::-1].transpose(2, 0, 1)
            im = torch.from_numpy(im.copy()).contiguous().float().to(device).unsqueeze(0) / 255.0
            model(im)

    tracker.print_report()

    # 每层集中度统计
    report = {"total_tokens": tracker.total_tokens, "layers": {}}
    print("\n== 每层专家选择集中度 ==")
    for name, stats in tracker.usage_stats.items():
        n_experts = len(stats)
        hits = [stats[i].hits for i in range(n_experts)]
        weights = [stats[i].weighted_sum for i in range(n_experts)]
        total_h = sum(hits)
        if total_h <= 0:
            continue
        share = [h / total_h for h in hits]
        max_share = max(share)
        ideal = 1.0 / n_experts
        # 集中度 = 最大专家份额 / 理想份额(>1 越集中)
        concentration = max_share / ideal
        print(
            f"{name}: 专家数 {n_experts} | hits 份额 {['%.1f%%' % (s * 100) for s in share]} | "
            f"最大 {max_share * 100:.1f}%(理想 {ideal * 100:.1f}%) | 集中度 {concentration:.2f}"
        )
        report["layers"][name] = {
            "n_experts": n_experts,
            "hits_share": [round(s, 4) for s in share],
            "weight_share": [round(w / sum(weights), 4) for w in weights] if sum(weights) > 0 else [],
            "concentration": round(concentration, 3),
        }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\n报告已存: {args.out}")


if __name__ == "__main__":
    main()
