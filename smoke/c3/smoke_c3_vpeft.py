#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C3 工业缺陷 V-PEFT 小样本实战 —— 准入检查 Smoke Test (2026-08-26)

目的(对应任务书 8.24 准入检查):
  1. 在 coco8 微型数据集上跑通一次 V-PEFT(planner_backend="vpeft")
  2. 保存并"解释一次 planner 输出"(status / targets / rank / budget)
  3. 记录最小训练日志、时间与内存占用,作为结果证据

用法(在仓库根目录):
  python smoke/c3/smoke_c3_vpeft.py

依赖:torch(MPS/CPU 均可)、ultralytics 本地包(pip install -e .)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ---- 环境:关闭 wandb / 自动安装,保证可复现 ----
os.environ.update(
    {
        "WANDB_MODE": "disabled",
        "WANDB_SILENT": "true",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "YOLO_AUTOINSTALL": "false",
    }
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import psutil
import torch

from ultralytics import YOLO
from ultralytics.utils import SETTINGS

SETTINGS["wandb"] = False

# 默认权重(准入阶段用 N 档,参数量最小)
WEIGHTS = REPO_ROOT / "YOLO-Master-EsMoE-N.pt"
DATA = "coco8.yaml"  # Ultralytics 内置 4 图微型数据集,首次运行自动下载


def device_choice():
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mem_record(tag: str) -> dict:
    """记录进程内存(GB)。MPS/CPU 无 nvidia-smi,统一用 RSS 作为 smoke 证据。"""
    rss_gb = psutil.Process().memory_info().rss / 1024**3
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return {"tag": tag, "rss_gb": round(rss_gb, 3), "cuda_alloc_gb": round(alloc, 3), "cuda_reserved_gb": round(reserved, 3)}
    return {"tag": tag, "rss_gb": round(rss_gb, 3)}


def explain_plan(plan: dict) -> str:
    """把一次 V-PEFT planner 输出翻译成可审计的解释文本。"""
    status = plan.get("status")
    backend = plan.get("planner_backend")
    solver = plan.get("solver")
    budget = plan.get("budget", {})
    max_params = budget.get("max_adapter_params")
    targets = plan.get("targets", [])
    n_targets = len(targets)
    total_rank = sum(int(t.get("rank", 0)) for t in targets)
    names = [t.get("name") for t in targets]

    meaning = {
        "ACCEPT": "约束下找到可行放置:按规划的 rank 注入 LoRA 适配器。",
        "ADAPT": "约束下做了适应性调整(如降秩/换变体)后接受。",
        "REFUSE": "不可行:预算/兼容性约束无法满足,拒绝 LoRA,回退到基线策略(如全参微调)。",
        "FALLBACK": "内部错误或配置不满足,兼容性回退。",
    }.get(status, "未知状态")

    lines = [
        f"[planner 输出解释] status={status} -> {meaning}",
        f"  planner_backend={backend}, solver={solver}",
        f"  adapter_budget(max_params)={max_params:,}",
        f"  放置适配器数={n_targets}, 合计 rank={total_rank}",
    ]
    if names:
        lines.append(f"  目标模块示例={names[:8]}{' ...' if n_targets > 8 else ''}")
    return "\n".join(lines)


def run_smoke(args):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "smoke" / "c3" / "runs" / f"c3_vpeft_smoke_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence = {"timestamp": datetime.now().isoformat(), "device": args.device, "cwd": str(REPO_ROOT)}
    print(f"== C3 Smoke: V-PEFT on {DATA} | device={args.device} | 输出目录={out_dir}")

    # ---- Step 1: 加载预训练模型 ----
    t0 = time.time()
    model = YOLO(str(WEIGHTS))
    evidence["load_weights_sec"] = round(time.time() - t0, 2)
    print(f"[1] 加载权重 {WEIGHTS.name} 完成, {time.time()-t0:.1f}s")

    # ---- Step 2: V-PEFT planner + 最小训练 ----
    t0 = time.time()
    mem_before = mem_record("before_train")
    print(f"[2] 开始 V-PEFT 训练: epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}")
    print(f"    planner_enabled=True, planner_backend=vpeft, adapter_budget={args.adapter_budget:,}")
    r = model.train(
        data=DATA,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        lora_planner_enabled=True,
        lora_planner_backend="vpeft",
        lora_adapter_budget=args.adapter_budget,
        lora_vpeft_strict=True,
        lora_r=args.rank,
        lora_variant="lora",
        lora_include_head=False,
        # V-PEFT planner 硬约束无 rank<=容量检查:stem '0.conv' 输入 3 通道
        # 无法承载 rank 8,strict 模式会抛 ValueError。受控排除 stem(不改算法,
        # 见 ADMISSION 文档 §7 风险与降级)。
        lora_exclude_modules="0.conv",
        project=str(out_dir / "train"),
        name="vpeft_on",
        verbose=True,
        workers=0,  # macOS 上避免多进程 DataLoader 问题
    )
    train_sec = time.time() - t0
    mem_after = mem_record("after_train")

    # ---- Step 3: 收集指标与 planner 输出 ----
    metrics = getattr(r, "results_dict", r) if not isinstance(r, dict) else r
    if isinstance(metrics, dict):
        m50 = metrics.get("metrics/mAP50(B)")
        m5095 = metrics.get("metrics/mAP50-95(B)")
    else:
        m50 = getattr(metrics, "mAP50", None)
        m5095 = getattr(metrics, "mAP50_95", None)

    # planner 决策:优先从运行时元数据取
    plan = None
    try:
        root = model.model
        meta = getattr(root, "lora_runtime_metadata", None) or getattr(root, "lora_placement_plan", None)
        if isinstance(meta, dict):
            plan = meta.get("placement_plan") or meta.get("planner_result") or meta
    except Exception:
        pass

    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

    evidence.update(
        {
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "train_sec": round(train_sec, 1),
            "mAP50": m50,
            "mAP50-95": m5095,
            "params_total": total_params,
            "params_trainable": trainable_params,
            "trainable_pct": round(trainable_params / total_params * 100, 4),
            "mem_before": mem_before,
            "mem_after": mem_after,
            "planner_plan": plan,
        }
    )
    print(f"[3] 训练耗时 {train_sec:.1f}s | mAP50={m50} | mAP50-95={m5095}")
    print(f"    总参数={total_params:,} | 可训练(适配器)={trainable_params:,} ({trainable_params/total_params*100:.3f}%)")
    print(f"    内存: before={mem_before['rss_gb']}GB after={mem_after['rss_gb']}GB")

    if plan:
        print("[4] " + explain_plan(plan))

    # ---- Step 4: 保存证据 ----
    ev_file = out_dir / "evidence.json"
    ev_file.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[5] 证据已保存: {ev_file}")
    print("SMOKE_OK" if m50 is not None or plan is not None else "SMOKE_WARN")
    return evidence


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="C3 V-PEFT 准入 smoke test")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--adapter-budget", type=int, default=2_000_000)
    ap.add_argument("--device", default=device_choice())
    args = ap.parse_args()
    run_smoke(args)
