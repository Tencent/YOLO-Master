"""A1 MoE 专家行为差异检查:专家们是"分工"还是"分钱"。

方法:前向若干张图,hook 捕获每层 MoE MLP 的输入 x;随后把同一个 x 主动喂给
该层**所有**专家(绕过路由),计算:
1. 专家输出两两 cosine 相似度(平均)——高 ≈ 输出雷同("分钱"),低 ≈ 行为差异("分工")
2. 专家输出与输入 x 的 cosine 相似度——高 ≈ 专家近似恒等映射(没学到任何变换)

判定:
- 专家间相似度 ~1 且专家-输入相似度 ~1 → 分工完全未形成,参数纯冗余
- 专家间相似度明显 <1 → 分工存在,问题在路由使用失衡(结合 routing_report)

用法:
    python scripts/a1/analyze_expert_diversity.py --model <B格best.pt> [--n-imgs 50]
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
from ultralytics.nn.modules.moe.modules import OptimizedMOEImproved


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="B 格(MoE)best.pt")
    ap.add_argument("--imgs-dir", default="/root/gpufree-data/datasets/coco-train-2017/images/val")
    ap.add_argument("--n-imgs", type=int, default=50, help="采样图数(seed=0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0")
    return ap.parse_args()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """展平后 cosine 相似度。"""
    a_f = a.reshape(-1).float()
    b_f = b.reshape(-1).float()
    denom = a_f.norm() * b_f.norm()
    return float((a_f @ b_f) / denom) if denom > 0 else 0.0


def main() -> None:
    """采集各层输入并计算专家间/专家-输入相似度。"""
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)

    yolo = YOLO(args.model)
    model = yolo.model
    model.eval().to(device)
    stride = int(max(model.stride.tolist()))

    mlps = {n: m for n, m in model.named_modules() if isinstance(m, OptimizedMOEImproved)}
    captured: dict[str, list[torch.Tensor]] = {n: [] for n in mlps}

    def make_pre(n):
        def pre(_, inp):
            captured[n].append(inp[0].detach())

        return pre

    handles = [m.register_forward_pre_hook(make_pre(n)) for n, m in mlps.items()]

    images = sorted(Path(args.imgs_dir).glob("*.jpg"))
    rng = random.Random(args.seed)
    picked = [images[i] for i in sorted(rng.sample(range(len(images)), min(args.n_imgs, len(images))))]
    lb = LetterBox((640, 640), auto=False, stride=stride)
    with torch.no_grad():
        for p in picked:
            img0 = cv2.imread(str(p))
            im = lb(image=img0)
            im = im[..., ::-1].transpose(2, 0, 1)
            im = torch.from_numpy(im.copy()).contiguous().float().to(device).unsqueeze(0) / 255.0
            model(im)
    for h in handles:
        h.remove()

    print(f"{'层':<28} {'专家数':<5} {'专家间sim':<10} {'专家-输入sim':<12} {'专家-共享sim':<12} 判定")
    for name, mlp in mlps.items():
        xs = captured[name]
        if not xs:
            continue
        n_experts = len(mlp.experts)
        expert_outs = [[] for _ in range(n_experts)]
        shared_outs = []
        for x in xs:
            for i in range(n_experts):
                expert_outs[i].append(mlp.experts[i](x))
            shared_outs.append(mlp.shared_expert(x))
        # 专家间平均相似度(两两)
        sims = []
        for i in range(n_experts):
            for j in range(i + 1, n_experts):
                for oi, oj in zip(expert_outs[i], expert_outs[j]):
                    sims.append(cosine_sim(oi, oj))
        inter = sum(sims) / len(sims) if sims else 0.0
        # 专家输出 vs 输入
        in_sims = []
        for i in range(n_experts):
            for oi, x in zip(expert_outs[i], xs):
                in_sims.append(cosine_sim(oi, x))
        vs_input = sum(in_sims) / len(in_sims) if in_sims else 0.0
        # 专家输出 vs 共享专家输出
        sh_sims = []
        for i in range(n_experts):
            for oi, so in zip(expert_outs[i], shared_outs):
                sh_sims.append(cosine_sim(oi, so))
        vs_shared = sum(sh_sims) / len(sh_sims) if sh_sims else 0.0
        verdict = "分钱(雷同)" if inter > 0.9 else ("部分分工" if inter > 0.6 else "分工明显")
        print(f"{name:<28} {n_experts:<5} {inter:<10.3f} {vs_input:<12.3f} {vs_shared:<12.3f} {verdict}")


if __name__ == "__main__":
    main()
