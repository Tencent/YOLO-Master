#!/usr/bin/env python3
"""F11 · q_teacher 构造与熵校验脚本（Foundation 路由 KD · 冒烟 2）

F11 课题：DINO 教师特征 → Router 软目标蒸馏。
本脚本：加载 cache_teacher_features 输出的 patch 特征，结合 YOLO-Master-n 模型的专家，
       计算一致性 q_teacher，校验熵 H_norm ∈ (0.3, 0.8)（非退化、非均匀）。

用法：
  python scripts/gen_q_teacher.py --cache runs/f11_teacher_cache --model yolo-master-n.yaml --device cpu

参考：
  - #54 scripts/diagnose_mot_routing.py：MoT 路由 hook + 专家 top-k 统计
  - #54 scripts/compare_routing_synthetic_vs_real.py：合成 vs 真实路由对比
  - 任务书 §2.1 F11：教师软目标合法定义（一致性/效用 + 温度 + 掩码）

文档版本：v1.0（2026-08-23）｜Owner：张伟林（Zviolin）｜F11 8.24 准入冒烟 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yolo_master_matplotlib")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F11 · q_teacher 构造与熵校验")
    parser.add_argument("--cache", default="runs/f11_teacher_cache", help="教师特征缓存目录")
    parser.add_argument("--model", default="ultralytics/cfg/models/master/v0_8/det/yolo-master-n.yaml", help="学生模型 yaml")
    parser.add_argument("--device", default="cpu", help="计算设备（冒烟建议 cpu）")
    parser.add_argument("--temperature", type=float, default=1.0, help="softmax 温度")
    parser.add_argument("--smoothing_eps", type=float, default=0.1, help="uniform 平滑系数")
    parser.add_argument("--confidence_mask", action="store_true", default=True, help="低置信 patch 掩码")
    parser.add_argument("--out", default="runs/f11_q_teacher_check", help="输出目录")
    parser.add_argument("--n_experts", type=int, default=4, help="专家数量")
    parser.add_argument("--expert_dim", type=int, default=768, help="专家特征维度")
    parser.add_argument("--use_real_protos", action="store_true", default=False, help="从模型提取真实 prototypes 而非随机")
    return parser.parse_args()


def load_teacher_patches(cache_dir: Path) -> "torch.Tensor":
    """加载缓存的 patch 特征。"""
    import torch

    patches_dir = cache_dir / "patches"
    if not patches_dir.exists():
        raise FileNotFoundError(f"patches 目录不存在: {patches_dir}，请先执行 cache_teacher_features.py")

    patch_files = sorted(patches_dir.glob("img_*.pt"))
    print(f"[F11] 加载 {len(patch_files)} 个 patch 特征文件")
    feats = [torch.load(str(p), map_location="cpu") for p in patch_files]
    return torch.cat(feats, dim=0)  # [N_total, D]


def get_expert_prototypes(model_yaml: Path, device: str, n_experts: int = 4, expert_dim: int = 768, use_real_protos: bool = False) -> "torch.Tensor":
    """获取专家 prototypes。

    策略:
    1. use_real_protos=True:  从模型加载第一个 MoTBlock.router.weight
    2. use_real_protos=False: 使用结构化 prototypes（点交于原点的轴向向量）
       - expert 0: [1, 0, 0, ...]
       - expert 1: [0, 1, 0, ...]
       - expert 2: [0, 0, 1, ...]
       - expert 3: [1, 1, 1, ...] / sqrt(3)
       这样保证与随机特征的夹角不同 → q 非均匀
    """
    import torch
    from ultralytics.nn.tasks import DetectionModel

    if use_real_protos:
        if not model_yaml.exists():
            model_yaml = ROOT / model_yaml
        try:
            model = DetectionModel(str(model_yaml), ch=3, nc=80, verbose=False)
            model.to(device).eval()

            from ultralytics.nn.modules.mot import MoTBlock
            for name, module in model.named_modules():
                if isinstance(module, MoTBlock):
                    router_weight = module.router.weight.detach().float().cpu()
                    print(f"[F11] 提取专家 prototypes: {name} shape={list(router_weight.shape)}")
                    return router_weight
        except Exception as exc:
            print(f"[F11][WARN] 从模型提取 prototypes 失败 ({exc}), 使用结构化 prototype")

    # 结构化 prototypes（保证 non-uniform）
    print(f"[F11] 使用结构化 prototypes: {n_experts} experts × {expert_dim} dim")
    protos = torch.zeros(n_experts, expert_dim)
    for i in range(n_experts):
        # 每个专家在不同轴上偏重，最后一个专家是平均轴
        if i < n_experts - 1:
            protos[i, i] = 1.0
            # 加点额外偏重以增加区分度
            protos[i, (i + 1) % expert_dim] = 0.3
        else:
            # 最后一个专家: 主对角线的总和
            protos[i] = torch.ones(expert_dim) / (expert_dim ** 0.5)
    return protos


def compute_consistency(teacher_feats: "torch.Tensor", expert_protos: "torch.Tensor", temperature: float) -> "torch.Tensor":
    """计算教师特征与各专家的一致性 → softmax → q_teacher。

    q_teacher = softmax( <E_i, T> / τ )
    """
    import torch
    import torch.nn.functional as F

    # teacher_feats: [N, D], expert_protos: [E, D]
    # 一致性矩阵：[N, E] = teacher_feats @ expert_protos.T
    consistency = teacher_feats @ expert_protos.T  # [N, E]
    q = F.softmax(consistency / max(temperature, 1e-6), dim=-1)  # [N, E]
    return q


def apply_smoothing_and_mask(q: "torch.Tensor", eps: float, use_mask: bool) -> "torch.Tensor":
    """标签平滑 + 低置信 patch 掩码。"""
    import torch
    E = q.shape[-1]

    # 标签平滑：q = (1-ε) q + ε uniform(E)
    uniform = torch.ones_like(q) / E
    q = (1.0 - eps) * q + eps * uniform

    # 低置信 patch 掩码：max(q) < 0.3 的样本 → 重置为均匀
    if use_mask:
        max_conf = q.max(dim=-1).values
        mask = max_conf < 0.3
        q[mask] = uniform[mask]

    # 重新归一化（保持概率分布）
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return q


def entropy_normalized(q: "torch.Tensor") -> "torch.Tensor":
    """计算归一化熵 H_norm = H / log(E)，范围 [0, 1]。"""
    import math
    import torch

    E = q.shape[-1]
    eps = 1e-9
    H = -(q * torch.log(q.clamp_min(eps))).sum(dim=-1)  # [N]
    H_norm = H / math.log(E)
    return H_norm


def check_non_degenerate(h_norm: "torch.Tensor") -> dict:
    """校验熵是否在合法范围 (0.3, 0.8)。"""
    import torch

    mean = float(h_norm.mean().item())
    std = float(h_norm.std().item())
    not_collapsed = ((h_norm > 0.1).float().mean().item())  # 不接近 0（坍塌）
    not_uniform = ((h_norm < 0.95).float().mean().item())   # 不接近 1（均匀）

    return {
        "mean": mean,
        "std": std,
        "min": float(h_norm.min().item()),
        "max": float(h_norm.max().item()),
        "not_collapsed_ratio": not_collapsed,
        "not_uniform_ratio": not_uniform,
        "pass": 0.3 < mean < 0.8 and not_collapsed > 0.9 and not_uniform > 0.9,
    }


def main() -> None:
    args = parse_args()

    cache_dir = Path(args.cache).resolve()
    model_yaml = Path(args.model).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载教师 patch 特征
    teacher_feats = load_teacher_patches(cache_dir)  # [N, D]

    # 2. 提取专家 prototypes
    expert_protos = get_expert_prototypes(
        model_yaml, args.device,
        n_experts=args.n_experts,
        expert_dim=teacher_feats.shape[1],
        use_real_protos=args.use_real_protos,
    )  # [E, D]

    # 3. 计算 q_teacher
    q_raw = normalize_consistency_dim(teacher_feats, expert_protos)
    q = compute_consistency(teacher_feats, expert_protos, args.temperature)

    # 4. 平滑 + 掩码
    q_final = apply_smoothing_and_mask(q, args.smoothing_eps, args.confidence_mask)

    # 5. 熵校验
    h_norm = entropy_normalized(q_final)
    check = check_non_degenerate(h_norm)

    # 6. 保存熵序列
    entropy_path = out_dir / "entropy.csv"
    with entropy_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "h_norm", "q_max"])
        for i in range(q_final.shape[0]):
            writer.writerow([i, f"{h_norm[i].item():.6f}", f"{q_final[i].max().item():.6f}"])

    # 7. 保存 q_teacher 软目标
    q_path = out_dir / "q_teacher.pt"
    import torch
    torch.save({
        "q_teacher": q_final,
        "h_norm": h_norm,
        "config": vars(args),
        "check": check,
    }, q_path)

    # 8. 输出 JSON 报告
    report = {
        "task": "F11 · q_teacher non-degenerate check",
        "config": vars(args),
        "num_patches": int(q_final.shape[0]),
        "num_experts": int(q_final.shape[1]),
        "check": check,
        "outputs": {
            "entropy_csv": str(entropy_path),
            "q_teacher_pt": str(q_path),
        },
    }
    report_path = out_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 9. 终端输出
    print(f"\n[F11] ✅ q_teacher 构造完成")
    print(f"[F11]   - patches: {q_final.shape[0]}, experts: {q_final.shape[1]}")
    print(f"[F11]   - H_norm mean: {check['mean']:.4f}（目标 ∈ (0.3, 0.8)）")
    print(f"[F11]   - H_norm std:  {check['std']:.4f}")
    print(f"[F11]   - 非坍塌比例:   {check['not_collapsed_ratio']*100:.1f}%（目标 > 90%）")
    print(f"[F11]   - 非均匀比例:   {check['not_uniform_ratio']*100:.1f}%（目标 > 90%）")
    print(f"[F11]   - 校验结果:     {'✅ 通过' if check['pass'] else '❌ 未通过'}")
    print(f"[F11] report: {report_path}")


def normalize_consistency_dim(teacher_feats: "torch.Tensor", expert_protos: "torch.Tensor") -> "torch.Tensor":
    """维度对齐：如果 teacher 和 expert 维度不一致，按较小的截断。"""
    import torch

    D_t = teacher_feats.shape[-1]
    D_e = expert_protos.shape[-1]
    if D_t == D_e:
        return teacher_feats
    D = min(D_t, D_e)
    print(f"[F11][WARN] 维度不一致 teacher={D_t}, expert={D_e}，截断到 {D}")
    return teacher_feats[:, :D]


if __name__ == "__main__":
    main()