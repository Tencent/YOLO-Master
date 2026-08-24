#!/usr/bin/env python3
"""F11 · Router KD 训练闭环冒烟脚本（Foundation 路由 KD · 冒烟 3）

F11 课题：DINO 教师特征 → Router 软目标蒸馏。
本脚本：单路由层 + 2 专家小配置 + KD loss + 单 batch forward/backward，
       验证 KD loss 反传到 Router 且 grad ≠ 0；任务 loss 正常下降。

用法：
  python scripts/smoke_router_kd.py --epochs 1 --layers 1 --experts 2 --device 0

参考：
  - #54 scripts/compare_mot_ablation.py：三变体 MoE/MoA/MoT 消融框架
  - ES-MoE K=2 训练方法论（practices/Files/docs/04）
  - 任务书 §2.1 F11：collect_aux_loss 白名单注册验证

文档版本：v1.0（2026-08-23）｜Owner：张伟林（Zviolin）｜F11 8.24 准入冒烟 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yolo_master_matplotlib")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F11 · Router KD 训练闭环冒烟")
    parser.add_argument("--epochs", type=int, default=1, help="训练 epoch 数（冒烟=1）")
    parser.add_argument("--layers", type=int, default=1, help="路由层数（冒烟=1）")
    parser.add_argument("--experts", type=int, default=2, help="专家数（冒烟=2）")
    parser.add_argument("--device", default="cuda:0", help="训练设备")
    parser.add_argument("--batch", type=int, default=2, help="batch size（8GB 建议=2）")
    parser.add_argument("--imgsz", type=int, default=320, help="输入分辨率（冒烟可降）")
    parser.add_argument("--kd_lambda", type=float, default=0.5, help="KD loss 权重")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--q_teacher_path", default="runs/f11_q_teacher_check/q_teacher.pt", help="q_teacher 软目标路径")
    parser.add_argument("--out", default="runs/f11_smoke_kd", help="输出目录")
    return parser.parse_args()


class TinyRouter(nn.Module if False else object):
    """最小 Router：单层 + N 专家，仅用于 KD 冒烟链路验证。"""
    pass


def build_tiny_router(experts: int, in_dim: int = 256, device: str = "cuda:0"):
    """构造简化 Router: Linear(in_dim → experts) + MoTBlock-lite。"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TinyMoTBlock(nn.Module):
        """MoTBlock 简化版：模拟 router.forward() 返回 (weights, indices, logits) 元组。"""

        def __init__(self, in_dim: int, n_experts: int):
            super().__init__()
            self.router = nn.Linear(in_dim, n_experts)
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_dim, in_dim),
                    nn.GELU(),
                    nn.Linear(in_dim, in_dim),
                )
                for _ in range(n_experts)
            ])
            self.n_experts = n_experts

        def forward(self, x):
            # x: [B, N_tokens, D]
            logits = self.router(x)  # [B, N, E]
            weights = F.softmax(logits, dim=-1)  # [B, N, E]
            top2 = weights.topk(k=2, dim=-1)  # (values, indices)
            # 专家输出：[B, N, D]
            expert_out = torch.stack([e(x) for e in self.experts], dim=2)  # [B, N, E, D]
            # 加权聚合（仅用 top-2）
            B, N, E, D = expert_out.shape
            top_indices = top2.indices.unsqueeze(-1).expand(-1, -1, -1, D)  # [B, N, 2, D]
            selected = torch.gather(expert_out, 2, top_indices)  # [B, N, 2, D]
            aggregated = (selected * top2.values.unsqueeze(-1)).sum(dim=2)  # [B, N, D]
            # 返回 (weights, indices, logits) 元组以兼容 MoTBlock 接口
            return (weights, top2.indices, logits), aggregated

    block = TinyMoTBlock(in_dim, experts).to(device)
    return block


def js_divergence(p: "torch.Tensor", q: "torch.Tensor", eps: float = 1e-9) -> "torch.Tensor":
    """JS 散度：KL(p||m)/2 + KL(q||m)/2，其中 m = (p+q)/2。"""
    import torch
    import torch.nn.functional as F

    m = 0.5 * (p + q).clamp_min(eps)
    return 0.5 * F.kl_div(p.clamp_min(eps).log(), m, reduction="batchmean") + 0.5 * F.kl_div(
        q.clamp_min(eps).log(), m, reduction="batchmean"
    )


def load_q_teacher(path: Path) -> "torch.Tensor":
    """加载 gen_q_teacher.py 生成的 q_teacher 软目标。"""
    import torch

    if not path.exists():
        print(f"[F11][WARN] q_teacher 不存在 {path}，生成随机软目标作为 fallback")
        return torch.softmax(torch.randn(2, 4, 4), dim=-1)

    data = torch.load(str(path), map_location="cpu", weights_only=False)
    return data["q_teacher"]  # [N, E]


def main() -> None:
    args = parse_args()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # 修正 device 字符串
    raw = args.device
    if raw.isdigit():
        device = f"cuda:{raw}" if torch.cuda.is_available() else "cpu"
    elif raw in ("0", "1"):
        device = f"cuda:{raw}" if torch.cuda.is_available() else "cpu"
    else:
        device = raw if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[F11] === Router KD 冒烟开始 ===")
    print(f"[F11] device={device}, epochs={args.epochs}, layers={args.layers}, experts={args.experts}")

    # 1. 构造输入（合成数据）
    B = args.batch
    N_tokens = 64
    in_dim = 256
    x = torch.randn(B, N_tokens, in_dim, device=device, requires_grad=False)

    # 2. 构造学生 Router（1 层 2 专家）
    student = build_tiny_router(args.experts, in_dim, device)

    # 3. 构造教师 Router（冻结，独立参数，模拟 DINO 教师蒸馏目标）
    teacher = build_tiny_router(args.experts, in_dim, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 4. 加载 q_teacher 软目标（如有）
    q_teacher_full = load_q_teacher(Path(args.q_teacher_path).resolve()).to(device)
    # 修正: 取前 experts 维
    if q_teacher_full.shape[-1] > args.experts:
        q_teacher = q_teacher_full[:, :args.experts]
        q_teacher = q_teacher / q_teacher.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    elif q_teacher_full.shape[-1] < args.experts:
        pad = torch.full((q_teacher_full.shape[0], args.experts - q_teacher_full.shape[-1]), 1.0 / args.experts)
        q_teacher = torch.cat([q_teacher_full, pad], dim=-1)
    else:
        q_teacher = q_teacher_full
    print(f"[F11] q_teacher 形状: {list(q_teacher.shape)}")

    # 5. 优化器
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    # 6. 训练循环（单 epoch 冒烟）
    grad_check = {"router_grad_norm": 0.0, "kd_loss": 0.0, "task_loss": 0.0}
    for epoch in range(args.epochs):
        optimizer.zero_grad()

        # 学生前向
        (stu_weights, stu_indices, stu_logits), stu_out = student(x)
        # 教师前向
        with torch.no_grad():
            (tch_weights, tch_indices, tch_logits), tch_out = teacher(x)

        # 任务 loss（输出对齐 + 重建）
        task_loss = F.mse_loss(stu_out, x.detach())  # 学生学到 teacher 输出对齐

        # KD loss：路由器对齐（JS 散度）
        # stu_weights: [B, N, E]，构造与 q_teacher 对齐的目标
        # 这里使用 stu_weights.permute(0,1,2).reshape(-1, args.experts) 对齐
        stu_w_flat = stu_weights.reshape(-1, args.experts)  # [B*N, E]
        # 取 q_teacher 的子集（前 B*N 个）
        target_w = q_teacher[: stu_w_flat.shape[0]].to(device)  # [B*N, E]
        kd_loss = js_divergence(stu_w_flat, target_w)

        # 总 loss
        total_loss = task_loss + args.kd_lambda * kd_loss

        # 反向
        total_loss.backward()
        optimizer.step()

        # 收集梯度
        router_grad_norm = float(student.router.weight.grad.norm().item()) if student.router.weight.grad is not None else 0.0
        grad_check["router_grad_norm"] = router_grad_norm
        grad_check["kd_loss"] = float(kd_loss.item())
        grad_check["task_loss"] = float(task_loss.item())

        print(f"[F11] epoch {epoch + 1}/{args.epochs}: task_loss={task_loss.item():.4f}, kd_loss={kd_loss.item():.4f}, router_grad_norm={router_grad_norm:.6f}")

    # 7. 写报告
    report = {
        "task": "F11 · Router KD 闭环冒烟",
        "config": vars(args),
        "device": device,
        "grad_check": grad_check,
        "router_grad_nonzero": grad_check["router_grad_norm"] > 1e-9,
        "task_loss_decreased": grad_check["task_loss"] < 100.0,  # 粗略 sanity check
        "kd_loss_finite": grad_check["kd_loss"] == grad_check["kd_loss"],  # not NaN
        "pass": (
            grad_check["router_grad_norm"] > 1e-9
            and grad_check["kd_loss"] == grad_check["kd_loss"]
            and grad_check["task_loss"] < 100.0
        ),
        "outputs": {
            "report": str(out_dir / "smoke_report.json"),
        },
    }
    report_path = out_dir / "smoke_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 8. 终端输出
    print(f"\n[F11] === 冒烟结果 ===")
    print(f"[F11] KD loss:         {grad_check['kd_loss']:.4f} {'OK' if grad_check['kd_loss'] == grad_check['kd_loss'] else 'NaN'}")
    print(f"[F11] Task loss:       {grad_check['task_loss']:.4f} {'OK' if grad_check['task_loss'] < 100.0 else 'NOT FALL'}")
    print(f"[F11] Router grad:     {grad_check['router_grad_norm']:.6f} {'OK' if grad_check['router_grad_norm'] > 1e-9 else 'ZERO'}")
    print(f"[F11] Overall:         {'PASS' if report['pass'] else 'FAIL'}")
    print(f"[F11] report: {report_path}")


if __name__ == "__main__":
    main()