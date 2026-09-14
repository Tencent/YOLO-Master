#!/usr/bin/env python3
"""F11 · T5 路由行为分析脚本(蒸馏前后对比)

依据 06 文档 §7(F11全部任务分析与设计):
  - top-k 一致率:     argtopk(student) == argtopk(teacher) 的比例
  - JS/KL 距离:       学生 vs 教师路由分布距离(单调下降)
  - 路由熵 H_router:  -Σ p_i log p_i(C 组应更均衡)
  - 专家负载:         top-1 选择频次(C 组 std 更小)
  - 切换率:           相邻样本 top-1 切换比例(C 组更平滑)

用法:
  # 冒烟: 合成数据验证链路可跑
  python scripts/analyze_f11_routing.py --synthetic --experts 4 --out runs/f11_routing_analysis

  # 真实: 对比 baseline 与 router_kd 训练产物
  python scripts/analyze_f11_routing.py \
      --student-a runs/f11_ablation/a-baseline/weights/best.pt \
      --student-c runs/f11_ablation/c-router-kd/weights/best.pt \
      --data coco8.yaml --out runs/f11_routing_analysis

参考:
  - ultralytics/nn/foundation/routing.py: FoundationTeacherRouter / routing_kd_loss
  - #54 scripts/diagnose_mot_routing.py: MoT 路由 hook 热力图
  - 任务书 §2.1 F11 + 06 文档 §7 判读线

文档版本：v1.0（2026-08-24）｜Owner：张伟林（Zviolin）｜F11 T5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yolo_master_matplotlib")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def normalize_device(device: str) -> str:
    """把数字设备字符串规范为完整设备字符串。"""
    if not device:
        return "cpu"
    if device.isdigit():
        return f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    return device


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """标准 JS 散度,返回 [N] 张量。"""
    p = p.clamp_min(1e-9)
    q = q.clamp_min(1e-9)
    m = 0.5 * (p + q)
    kl_psm = (p * (p / m).log()).sum(dim=-1)
    kl_qsm = (q * (q / m).log()).sum(dim=-1)
    return 0.5 * (kl_psm + kl_qsm)


def compute_router_metrics(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor | None,
    *,
    name: str,
) -> dict:
    """计算一组路由分布的全部 T5 指标。

    Args:
        student_probs: 学生路由分布 [N, E]
        teacher_probs: 教师路由分布 [N, E] 或 None(仅学生分析)
        name: 分析对象名称

    Returns:
        (dict): 指标字典
    """
    probs = student_probs.float()
    num_experts = probs.shape[-1]
    top1 = probs.argmax(dim=-1)  # [N]

    # 路由熵 H_router
    entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)  # [N]
    h_norm = entropy / np.log(num_experts)

    # 专家负载: top-1 选择频次
    counts = torch.bincount(top1, minlength=num_experts).float()
    load_ratio = counts / counts.sum().clamp_min(1e-9)
    load_std = float(load_ratio.std().item()) if num_experts > 1 else 0.0

    # 切换率: 相邻样本 top-1 切换比例
    switch_rate = float((top1[1:] != top1[:-1]).float().mean().item()) if probs.shape[0] > 1 else 0.0

    metrics = {
        "name": name,
        "num_tokens": int(probs.shape[0]),
        "num_experts": num_experts,
        "h_router": float(entropy.mean().item()),
        "h_norm": float(h_norm.mean().item()),
        "load_ratio": [float(v) for v in load_ratio.tolist()],
        "load_std": load_std,
        "top1_switch_rate": switch_rate,
    }

    if teacher_probs is not None:
        teacher = teacher_probs.float()
        # top-1 一致率
        teacher_top1 = teacher.argmax(dim=-1)
        top1_agree = float((top1 == teacher_top1).float().mean().item())
        # top-k 一致率(k=2): 学生 top-k 与教师 top-k 有交集的比例
        k = min(2, num_experts)
        stu_topk_tensor = probs.topk(k, dim=-1).indices  # [N, k]
        tea_topk_tensor = teacher.topk(k, dim=-1).indices  # [N, k]
        overlap = torch.zeros(probs.shape[0], dtype=torch.bool, device=probs.device)
        for i in range(k):
            overlap |= (stu_topk_tensor == tea_topk_tensor[:, i].unsqueeze(1)).any(dim=-1)
        topk_agree = float(overlap.float().mean().item())
        # JS 距离
        js = js_divergence(probs, teacher)
        metrics.update(
            {
                "top1_agree": top1_agree,
                f"top{k}_agree": topk_agree,
                "js_mean": float(js.mean().item()),
                "js_std": float(js.std().item()),
            }
        )
    return metrics


def generate_synthetic_logits(num_tokens: int, num_experts: int, seed: int) -> torch.Tensor:
    """生成合成路由 logits(冒烟链路验证用)。"""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(num_tokens, num_experts)).astype(np.float32)
    return torch.from_numpy(logits)


def run_synthetic(args) -> dict:
    """冒烟: 合成学生/教师分布,验证全部指标可计算。"""
    torch.manual_seed(args.seed)
    num_tokens = args.num_tokens
    num_experts = args.experts

    student_logits = generate_synthetic_logits(num_tokens, num_experts, args.seed)
    teacher_logits = generate_synthetic_logits(num_tokens, num_experts, args.seed + 1)

    student_probs = F.softmax(student_logits / args.temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / args.temperature, dim=-1)

    metrics = compute_router_metrics(
        student_probs,
        teacher_probs,
        name=f"synthetic-e{num_experts}",
    )
    return metrics


def _register_routing_hooks(model: torch.nn.Module) -> tuple[list, list]:
    """给模型注册路由分布 hook,支持 MoTBlock / MoE / MoA / 任意 .router 子模块。

    Returns:
        (router_probs, handles) —— router_probs 在推理后填入。
    """
    router_probs: list[torch.Tensor] = []
    handles: list = []

    try:
        from ultralytics.nn.modules.mot import MoTBlock
    except Exception:  # pragma: no cover
        MoTBlock = None

    def _safe_collect(probs_or_logits: torch.Tensor) -> torch.Tensor | None:
        """把任意形状的 [B, E, ...] 张量变成 [N, E] tensor。"""
        if probs_or_logits is None or not isinstance(probs_or_logits, torch.Tensor):
            return None
        if probs_or_logits.ndim == 2:  # [B, E]
            return probs_or_logits.detach().float().cpu()
        if probs_or_logits.ndim == 4:  # [B, E, H, W]
            t = probs_or_logits.permute(0, 2, 3, 1).reshape(-1, probs_or_logits.shape[1])
            return t.detach().float().cpu()
        if probs_or_logits.ndim == 3:  # [B, E, H]
            t = probs_or_logits.permute(0, 2, 1).reshape(-1, probs_or_logits.shape[1])
            return t.detach().float().cpu()
        return None

    def make_mot_hook():
        def hook(_module, _inputs, output):
            if isinstance(output, tuple) and len(output) >= 1:
                t = _safe_collect(output[0])
                if t is not None:
                    router_probs.append(t)
        return hook

    def make_router_attr_hook():
        """针对任意含 .router (nn.Module) 属性的 MoE-like 模块,hook 其自身 forward。"""

        def hook(_module, _inputs, output):
            # 许多 MoE 的 forward 返回 (out, routing_stats) 或包含 router_probs/logits 的元组
            if isinstance(output, tuple):
                last = output[-1] if len(output) >= 1 else None
                if isinstance(last, dict):
                    for key in ("router_probs", "router_logits"):
                        if key in last:
                            t = _safe_collect(last[key])
                            if t is not None:
                                router_probs.append(t)
                                return
                # 也可能直接是 [B, E, ...] 张量在 output[0]
                t = _safe_collect(output[0])
                if t is not None and t.shape[-1] >= 2 and t.shape[-1] <= 64:
                    router_probs.append(t)
        return hook

    def make_router_submodule_hook():
        """针对内部 .router / .routing 子模块本身的 forward hook。

        本项目 RefinedLowRankHybridAdaptiveGateMoE.routing 是 DualStreamGateRouter,
        forward 返回 (global_logits, local_logits, info_dict),shape 均为 [B, E, 1, 1]。
        对 global+local 取 softmax 后求平均作为综合路由分布(符合论文 dual-stream 融合假设)。
        """

        def hook(_module, _inputs, output):
            # 情况 1: DualStreamGateRouter 风格 —— tuple(logits_global, logits_local, info)
            if isinstance(output, tuple) and len(output) >= 2:
                g = _safe_collect(output[0])
                l = _safe_collect(output[1]) if len(output) >= 2 else None
                if g is not None and l is not None and g.shape == l.shape:
                    # softmax 后求平均(融合 dual-stream)
                    g_p = torch.softmax(g, dim=-1)
                    l_p = torch.softmax(l, dim=-1)
                    router_probs.append(0.5 * (g_p + l_p))
                    return
                if g is not None:
                    router_probs.append(torch.softmax(g, dim=-1))
                    return
            # 情况 2: 标准 MoE —— output 本身就是 [B, E] / [B, E, H, W] logits
            t = _safe_collect(output)
            if t is not None and 2 <= t.shape[-1] <= 64:
                router_probs.append(torch.softmax(t, dim=-1))
        return hook

    # MoE 模块的路由子模块名(脚本原版只认 ".router",本项目 RefinedLowRankHybridAdaptiveGateMoE 使用 ".routing")
    ROUTER_ATTRS: tuple[str, ...] = ("router", "routing")

    # Step 1: MoTBlock hook
    if MoTBlock is not None:
        for _, mod in model.named_modules():
            if isinstance(mod, MoTBlock):
                handles.append(mod.register_forward_hook(make_mot_hook()))

    # Step 2: 含 .router / .routing 属性的 MoE-style 模块 hook
    moe_module_types: tuple = ()
    try:
        from ultralytics.nn.modules.moe.gated import (
            RefinedLowRankHybridAdaptiveGateMoE,
            LowRankHybridAdaptiveGateMoE,
            HybridAdaptiveGateMoE,
            AdaptiveGateMoE,
            HyperSplitMoE,
            HyperFusedMoE,
            FusedAdaptiveGateMoE,
            MultiHeadRouterMoE,
            HyperUltimateMoE,
            DetailAwareLowRankHybridAdaptiveGateMoE,
            ContextRefinedLowRankHybridAdaptiveGateMoE,
            VisualEnhancedAdaptiveGateMoE,
        )
        moe_module_types = (
            RefinedLowRankHybridAdaptiveGateMoE,
            LowRankHybridAdaptiveGateMoE,
            HybridAdaptiveGateMoE,
            AdaptiveGateMoE,
            HyperSplitMoE,
            HyperFusedMoE,
            FusedAdaptiveGateMoE,
            MultiHeadRouterMoE,
            HyperUltimateMoE,
            DetailAwareLowRankHybridAdaptiveGateMoE,
            ContextRefinedLowRankHybridAdaptiveGateMoE,
            VisualEnhancedAdaptiveGateMoE,
        )
    except Exception:
        # 部分类可能在新版本被移除,不影响主流程
        moe_module_types = ()

    # 收集已 hook 过的路由子模块 id,防止重复
    hooked_router_ids: set[int] = set()

    for _, mod in model.named_modules():
        if isinstance(mod, moe_module_types):
            for attr in ROUTER_ATTRS:
                router = getattr(mod, attr, None)
                if isinstance(router, torch.nn.Module) and id(router) not in hooked_router_ids:
                    handles.append(mod.register_forward_hook(make_router_attr_hook()))
                    handles.append(router.register_forward_hook(make_router_submodule_hook()))
                    hooked_router_ids.add(id(router))
                    break  # 一个 MoE 块只取一个路由子模块

    # Step 3: heuristic —— 任何模块(.router / .routing 存在且末端 shape 暗示 num_experts)
    for _, mod in model.named_modules():
        if isinstance(mod, moe_module_types):  # 已在 step 2 hook,跳过
            continue
        for attr in ROUTER_ATTRS:
            router = getattr(mod, attr, None)
            if isinstance(router, torch.nn.Module) and id(router) not in hooked_router_ids:
                handles.append(mod.register_forward_hook(make_router_attr_hook()))
                handles.append(router.register_forward_hook(make_router_submodule_hook()))
                hooked_router_ids.add(id(router))
                break

    return router_probs, handles


def load_student_probs(ckpt_path: Path, data_yaml: str, device: str, batch: int, imgsz: int) -> torch.Tensor:
    """从训练权重加载学生模型,前向数据收集路由分布。支持 MoT/MoE/MoA。"""
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

    if not ckpt_path.exists():
        raise SystemExit(f"权重不存在: {ckpt_path}")

    model = YOLO(str(ckpt_path))
    data_info = check_det_dataset(data_yaml) if data_yaml else None
    names = data_info.get("names", {}) if data_info else {}
    nc = len(names) if names else 80
    model.model.nc = nc
    model.to(torch.device(device)).eval()

    # 用合成张量收集路由分布(避免依赖数据集下载)
    router_probs, handles = _register_routing_hooks(model.model)

    with torch.no_grad():
        x = torch.randn(batch, 3, imgsz, imgsz, device=device)
        _ = model.model(x)
        # 多次 batch 累计,扩大 token 数
        for _ in range(3):
            _ = model.model(torch.randn(batch, 3, imgsz, imgsz, device=device))
    for h in handles:
        h.remove()

    if not router_probs:
        raise SystemExit(
            f"{ckpt_path} 中未发现可 hook 的路由层(MoTBlock / MoE / MoA 都不存在)。"
        )

    # 合并所有 hook 收集的 token 级概率 [N, E]
    return torch.cat(router_probs, dim=0)


def write_report(metrics_list: list[dict], out_dir: Path) -> None:
    """写 JSON + CSV 报告。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "routing_analysis.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_list, f, ensure_ascii=False, indent=2)

    csv_path = out_dir / "routing_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for m in metrics_list for k in m if not isinstance(m[k], list)})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics_list:
            writer.writerow({k: m[k] for k in fieldnames if k in m})

    print(f"[F11][T5] JSON: {report_path}")
    print(f"[F11][T5] CSV:  {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F11 · T5 路由行为分析")
    parser.add_argument("--synthetic", action="store_true", help="合成数据冒烟(默认)")
    parser.add_argument("--student-a", default=None, help="A 组 baseline 权重 best.pt")
    parser.add_argument("--student-c", default=None, help="C 组 router_kd 权重 best.pt")
    parser.add_argument("--data", default="coco8.yaml", help="数据集 yaml")
    parser.add_argument("--device", default="cpu", help="分析设备(默认 cpu)")
    parser.add_argument("--batch", type=int, default=1, help="推理 batch")
    parser.add_argument("--imgsz", type=int, default=640, help="推理分辨率")
    parser.add_argument("--experts", type=int, default=4, help="专家数(合成冒烟)")
    parser.add_argument("--num-tokens", type=int, default=512, help="token 数(合成冒烟)")
    parser.add_argument("--temperature", type=float, default=1.0, help="softmax 温度")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--out", default="runs/f11_routing_analysis", help="输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out).resolve()
    device = normalize_device(args.device)

    metrics_list: list[dict] = []

    if args.student_a or args.student_c:
        # 真实权重对比模式
        if args.student_a:
            student_probs = load_student_probs(
                Path(args.student_a), args.data, device, args.batch, args.imgsz
            )
            metrics_list.append(
                compute_router_metrics(student_probs, None, name="A-baseline")
            )
        if args.student_c:
            student_probs_c = load_student_probs(
                Path(args.student_c), args.data, device, args.batch, args.imgsz
            )
            # 若同时有 A 组, 用 A 组作为"教师"对比 top-k 一致率
            teacher_probs = None
            if args.student_a and len(metrics_list) == 1:
                # 重新加载 A 组做对比
                teacher_probs = load_student_probs(
                    Path(args.student_a), args.data, device, args.batch, args.imgsz
                )
            metrics_list.append(
                compute_router_metrics(student_probs_c, teacher_probs, name="C-router-kd")
            )
    else:
        # 合成冒烟(默认)
        metrics_list.append(run_synthetic(args))

    for m in metrics_list:
        print(f"\n[F11][T5] ==== {m['name']} ====")
        print(f"[F11][T5] tokens={m['num_tokens']}, experts={m['num_experts']}")
        print(f"[F11][T5] H_router={m['h_router']:.4f}, H_norm={m['h_norm']:.4f}")
        print(f"[F11][T5] load_std={m['load_std']:.4f}, top1_switch={m['top1_switch_rate']:.4f}")
        for key in ("top1_agree", "top2_agree", "js_mean"):
            if key in m:
                print(f"[F11][T5] {key}={m[key]:.4f}")

    write_report(metrics_list, out_dir)


if __name__ == "__main__":
    main()
