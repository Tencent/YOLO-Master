"""路由分布对比脚本：合成随机输入 vs 真实 VisDrone 图像。

论文投稿用途：验证 dry-run 结论的稳健性。
若合成 vs 真实数据的 top-1 激活率差异 < 5%，则合成结论可靠。

用法：
  python scripts/compare_routing_synthetic_vs_real.py ^
      --model experiments_zviolin\runs\v08_mot6\weights\last.pt ^
      --data ultralytics\cfg\datasets\VisDrone.yaml ^
      --num-samples 50 ^
      --output experiments_zviolin\runs\routing_compare
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

# ── 修复 1: sys.path 指向 scripts/ 而非项目根 ─────────────────────────
# 原代码写的是 Path(__file__).resolve().parent.parent, 会把 YOLO-Master-new/
# 加入 sys.path, 但 diagnose_mot_routing.py 在 scripts/ 下, 找不到.
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from diagnose_mot_routing import (  # noqa: E402
    EXPERT_NAMES,
    load_model,
    normalize_torch_device,
)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# ── 修复 2: register_hooks 改为本地实现 ────────────────────────────────
# 原代码调用不存在的 register_hooks(), 这里自己注册 hook 收集 records.
# router.forward() 返回 (weights, indices, router_logits);
#  - weights  [B, E, H, W]  sparse, top-K 非零
#  - indices  [B, K, H, W]  top-K expert 索引
# 我们用 weights 的 argmax 算 top-1, 用 indices 算 top-K (K=2).
def register_hooks(model):
    """注册 hook 并返回 (records, handles). records 字段保持原算法期望的
    `activation_ratio` 与 `topk_activation_ratio`.
    """
    from ultralytics.nn.modules.mot import MoTBlock

    records: List[Dict] = []
    handles = []

    def make_hook(layer_name: str):
        def hook(_module, _inputs, output):
            if not isinstance(output, tuple) or len(output) < 2:
                return
            weights, indices = output[0], output[1]
            if weights.ndim != 4 or indices.ndim != 4:
                return
            weights = weights.detach().float().cpu()
            indices = indices.detach().long().cpu()
            B, E, H, W = weights.shape
            K = indices.shape[1]
            token_count = H * W

            for b in range(B):
                # Top-1: argmax 沿 expert 维
                top1 = weights[b].argmax(dim=0).reshape(-1)  # [H*W]
                # Top-K 累计: 每个 token 被 K 个 expert 覆盖, 统计每个 expert
                # 在 top-K 中出现的次数, 再除以 (K * H * W) 得到 top-K 占比
                for expert_id in range(E):
                    cnt_top1 = int((top1 == expert_id).sum().item())
                    cnt_topk = 0
                    for k in range(K):
                        top_at_k = indices[b, k].reshape(-1)
                        cnt_topk += int((top_at_k == expert_id).sum().item())
                    records.append({
                        "layer": layer_name,
                        "expert": EXPERT_NAMES[expert_id],
                        # 原算法期望的字段名 (保持不变)
                        "activation_ratio": cnt_top1 / token_count,
                        "topk_activation_ratio": cnt_topk / (K * token_count),
                    })

        return hook

    for name, module in model.named_modules():
        if isinstance(module, MoTBlock):
            handles.append(module.router.register_forward_hook(make_hook(name)))
    return records, handles


# ── 修复 3: 字段名保持原算法的 activation_ratio / topk_activation_ratio ──
# 原 aggregate() 直接照搬, 不变.
def aggregate(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    """聚合行数据为每个专家的 top1/top-k 激活率字典。"""
    agg = {e: {"top1": [], "topk": []} for e in EXPERT_NAMES}
    for r in rows:
        if r["expert"] in agg:
            agg[r["expert"]]["top1"].append(r["activation_ratio"])
            agg[r["expert"]]["topk"].append(r["topk_activation_ratio"])
    return {
        e: {
            "top1": sum(v["top1"]) / len(v["top1"]) if v["top1"] else 0.0,
            "topk": sum(v["topk"]) / len(v["topk"]) if v["topk"] else 0.0,
        }
        for e, v in agg.items()
    }


def run_synthetic(model, device: str, imgsz: int, runs: int = 10) -> List[Dict]:
    """合成输入：跑 N 次取平均（降低随机噪声）。"""
    all_rows = []
    for seed in range(runs):
        torch.manual_seed(seed)
        rows, hooks = register_hooks(model)
        with torch.no_grad():
            x = torch.randn(1, 3, imgsz, imgsz, device=device)
            _ = model(x)
        for h in hooks:
            h.remove()
        all_rows.extend(rows)
    return all_rows


def load_real_batch(data_cfg: str, num_samples: int, imgsz: int):
    """加载 VisDrone 真实图像（带 fallback）。

    简化版：直接读 VisDrone 图像目录，跳过 ultralytics dataset 包装（避免
    'mask_ratio' 等高版本参数依赖）。论文投稿足够使用。
    """
    try:
        from ultralytics.data.utils import check_det_dataset
        import cv2
    except Exception as e:
        print(f"[compare_routing] ⚠️ 无法导入依赖: {e}")
        return None
    try:
        data_dict = check_det_dataset(data_cfg)
        train_path = Path(data_dict["train"])
        if train_path.is_file() and train_path.suffix == ".txt":
            with open(train_path, encoding="utf-8") as f:
                image_paths = [line.strip() for line in f if line.strip()]
        else:
            image_paths = []
            for ext in ("*.jpg", "*.png", "*.jpeg"):
                image_paths.extend(sorted(train_path.glob(ext)))
        if not image_paths:
            print(f"[compare_routing] ⚠️ 未在 {train_path} 找到图像")
            return None

        samples = []
        for img_path in image_paths[:num_samples]:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.resize(img, (imgsz, imgsz))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            samples.append(img_t.unsqueeze(0))
        if not samples:
            return None
        return torch.cat(samples, dim=0)
    except Exception as e:
        print(f"[compare_routing] ⚠️ 加载真实数据失败: {e}")
        return None


def run_real(model, x: torch.Tensor) -> List[Dict]:
    """真实数据模式：每张图单独跑，聚合所有图。"""
    rows_all = []
    for i in range(x.shape[0]):
        rows, hooks = register_hooks(model)
        with torch.no_grad():
            _ = model(x[i:i + 1])
        for h in hooks:
            h.remove()
        rows_all.extend(rows)
    return rows_all


def main():
    parser = argparse.ArgumentParser(description="合成 vs 真实路由分布对比")
    parser.add_argument("--model", required=True, help=".pt 训练后权重")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/VisDrone.yaml")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "cuda:0", "0"])
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--synthetic-runs", type=int, default=10)
    parser.add_argument("--output", default="experiments_zviolin/runs/routing_compare")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="激活率差异阈值（< 5% 视为可靠）")
    args = parser.parse_args()

    device = normalize_torch_device(args.device)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ── 修复 2 续: build_model_from_pt → load_model ─────────────────
    # 原代码调不存在的 build_model_from_pt(args.model),
    # 实际 API 是 load_model(path, device, nc), nc 写死 80 (COCO 检测)
    model = load_model(Path(args.model), device, nc=80)
    model = model.to(device).eval()

    # 1) 合成输入
    print(f"\n[1/3] 合成随机输入（{args.synthetic_runs} 次取平均）...")
    synth_rows = run_synthetic(model, device, args.imgsz, args.synthetic_runs)
    synth_agg = aggregate(synth_rows)

    # 2) 真实数据
    print(f"[2/3] 加载 VisDrone 真实数据（{args.num_samples} 张）...")
    x = load_real_batch(args.data, args.num_samples, args.imgsz)
    if x is None:
        print("[3/3] 真实数据不可用，仅输出合成结果")
        real_agg = None
    else:
        print(f"[3/3] 真实数据推理（{x.shape[0]} 张）...")
        real_rows = run_real(model, x)
        real_agg = aggregate(real_rows)

    # 3) 写报告
    report = {
        "synthetic": {
            "runs": args.synthetic_runs,
            "imgsz": args.imgsz,
            "experts": synth_agg,
        },
    }
    if real_agg:
        report["real"] = {
            "num_samples": int(x.shape[0]),
            "imgsz": args.imgsz,
            "experts": real_agg,
        }
        # 计算差异
        diffs = {}
        for e in EXPERT_NAMES:
            d_top1 = abs(synth_agg[e]["top1"] - real_agg[e]["top1"])
            d_topk = abs(synth_agg[e]["topk"] - real_agg[e]["topk"])
            diffs[e] = {"top1_diff": d_top1, "topk_diff": d_topk}
        max_diff = max(d["top1_diff"] for d in diffs.values())
        report["diffs"] = diffs
        report["robustness"] = {
            "max_top1_diff": max_diff,
            "threshold": args.threshold,
            "is_robust": max_diff < args.threshold,
            "conclusion": (
                f"✓ 合成结论可靠 (max diff {max_diff:.3f} < {args.threshold})"
                if max_diff < args.threshold
                else f"✗ 合成结论需谨慎 (max diff {max_diff:.3f} >= {args.threshold})"
            ),
        }

    json_path = out / "routing_compare.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[compare_routing] wrote {json_path}")

    # 4) 绘制对比图
    if real_agg:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        x_pos = np.arange(len(EXPERT_NAMES))
        width = 0.35

        s_top1 = [synth_agg[e]["top1"] for e in EXPERT_NAMES]
        r_top1 = [real_agg[e]["top1"] for e in EXPERT_NAMES]
        axes[0].bar(x_pos - width / 2, s_top1, width, label="Synthetic", color="#4472c4")
        axes[0].bar(x_pos + width / 2, r_top1, width, label="Real (VisDrone)", color="#ed7d31")
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels([e.replace("Transformer", "") for e in EXPERT_NAMES])
        axes[0].set_ylabel("Top-1 Activation Ratio")
        axes[0].set_title("Top-1 Activation: Synthetic vs Real")
        axes[0].legend()
        axes[0].set_ylim(0, 1)
        axes[0].grid(axis="y", alpha=0.3)

        s_topk = [synth_agg[e]["topk"] for e in EXPERT_NAMES]
        r_topk = [real_agg[e]["topk"] for e in EXPERT_NAMES]
        axes[1].bar(x_pos - width / 2, s_topk, width, label="Synthetic", color="#4472c4")
        axes[1].bar(x_pos + width / 2, r_topk, width, label="Real (VisDrone)", color="#ed7d31")
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels([e.replace("Transformer", "") for e in EXPERT_NAMES])
        axes[1].set_ylabel("Top-2 Activation Ratio")
        axes[1].set_title("Top-2 Activation: Synthetic vs Real")
        axes[1].legend()
        axes[1].set_ylim(0, 1)
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        png_path = out / "routing_synthetic_vs_real.png"
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[compare_routing] wrote {png_path}")

    # 控制台摘要
    print("\n=== 路由分布对比 ===")
    print(f"{'Expert':<25s}  {'Synth top1':>10s}  {'Real top1':>10s}  {'Diff':>8s}")
    print("-" * 60)
    for e in EXPERT_NAMES:
        s = synth_agg[e]["top1"]
        if real_agg:
            r = real_agg[e]["top1"]
            d = abs(s - r)
            print(f"{e:<25s}  {s:>10.3f}  {r:>10.3f}  {d:>8.3f}")
        else:
            print(f"{e:<25s}  {s:>10.3f}  {'N/A':>10s}  {'N/A':>8s}")
    if real_agg:
        print(f"\n结论: {report['robustness']['conclusion']}")


if __name__ == "__main__":
    main()
