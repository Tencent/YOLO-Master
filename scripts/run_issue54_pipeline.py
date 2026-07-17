#!/usr/bin/env python3
"""Issue #54 完整流水线：训练 → Benchmark → 路由分析 → 报告生成

一键运行:
    python scripts/run_issue54_pipeline.py --device 0

仅验证环境 + benchmark:
    python scripts/run_issue54_pipeline.py --device 0 --check-only

仅路由分析 (需要已有 checkpoint):
    python scripts/run_issue54_pipeline.py --device 0 --routing-only \
        --checkpoint runs/mot_ablation/v10_mot/weights/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── 模型变体 ──────────────────────────────────────────────────────────────
MODELS = ["v10", "v10_mot", "v10_moa", "v10_moa_mot"]

MODEL_LABELS = {
    "v10": "MoE 基线",
    "v10_mot": "MoT 实验组",
    "v10_moa": "MoA 对比组",
    "v10_moa_mot": "MoA+MoT 混合",
}


def run(cmd: list[str], desc: str = "") -> int:
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'='*60}")
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def step_check_build(args: argparse.Namespace) -> Path:
    """验证 4 个模型 YAML 可解析 + 输出 FLOPs/Params"""
    out = ROOT / "runs/mot_ablation/build_summary.csv"
    rc = run([
        "python", "scripts/compare_mot_ablation.py",
        "--check-build", "--models", *MODELS,
        "--imgsz", str(args.imgsz),
        "--device", args.device,
    ], "Step 1/5: 模型配置校验")
    if rc != 0:
        raise SystemExit(rc)
    return out


def step_benchmark(args: argparse.Namespace) -> Path:
    """CPU/GPU 延迟 Benchmark"""
    out = ROOT / f"runs/mot_ablation/latency_{args.device}_{args.imgsz}.csv"
    rc = run([
        "python", "scripts/compare_mot_ablation.py",
        "--benchmark", "--models", *MODELS,
        "--imgsz", str(args.imgsz),
        "--warmup", str(args.warmup),
        "--reps", str(args.reps),
        "--device", args.device,
    ], "Step 2/5: 延迟 Benchmark")
    if rc != 0:
        raise SystemExit(rc)
    return out


def step_train(args: argparse.Namespace) -> int:
    """训练 4 个模型变体"""
    return run([
        "python", "scripts/compare_mot_ablation.py",
        "--train", "--models", *MODELS,
        "--data", str(args.data),
        "--epochs", str(args.epochs),
        "--imgsz", str(args.imgsz),
        "--batch", str(args.batch),
        "--device", args.device,
        "--workers", "0",
        "--patience", str(args.patience),
        "--plots",
        "--exist-ok",
        "--resume",
    ], f"Step 3/5: 训练 {len(MODELS)} 个模型变体 ({args.epochs} epochs each)")


def step_routing_analysis(args: argparse.Namespace) -> int:
    """路由可解释性分析"""
    model_path = args.checkpoint or (
        ROOT / "runs/mot_ablation/v10_mot/weights/best.pt"
    )
    if not Path(model_path).exists():
        print(f"\n  ⚠️  Checkpoint 不存在: {model_path}")
        print(f"  先运行训练，或通过 --checkpoint 指定已有权重")
        return 1

    # 真实图片分析（如果数据集已下载）
    image_dir = args.image_dir
    synthetic_flag = []
    if not image_dir or not Path(image_dir).exists():
        print(f"\n  未找到图片目录，使用合成场景数据")
        synthetic_flag = ["--synthetic"]
        image_dir_flag = []
    else:
        image_dir_flag = ["--image-dir", str(image_dir)]

    return run([
        "python", "scripts/diagnose_mot_routing.py",
        "--model", str(model_path),
        "--imgsz", str(args.imgsz),
        "--batch", str(args.batch),
        "--max-images", str(args.max_images),
        "--device", args.device,
    ] + synthetic_flag + image_dir_flag,
        "Step 4/5: 路由可解释性分析")


def step_summary(args: argparse.Namespace) -> int:
    """生成汇总对比表"""
    return run([
        "python", "scripts/compare_mot_ablation.py",
        "--summary-only", "--models", *MODELS,
    ], "Step 5/5: 生成汇总 CSV")


def print_results_summary() -> None:
    """打印结果摘要"""
    project = ROOT / "runs/mot_ablation"

    # Build data
    build_csv = project / "build_summary.csv"
    if build_csv.exists():
        with open(build_csv) as f:
            rows = list(csv.DictReader(f))
        print("\n📊 模型对比")
        print(f"  {'模型':<18} {'Params':>10} {'GFLOPs':>10}")
        print(f"  {'─'*18} {'─'*10} {'─'*10}")
        for r in rows:
            print(f"  {r['label']:<18} {float(r['params_m']):>8.2f}M {float(r['flops_g']):>8.2f}")

    # Summary CSV (training results)
    summary_csv = project / "summary.csv"
    if summary_csv.exists():
        with open(summary_csv) as f:
            rows = list(csv.DictReader(f))
        print(f"\n📈 训练结果")
        headers = ["label", "epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
                    "nan_detected", "loss_diverged"]
        print(f"  {'模型':<18} {'Epoch':>6} {'mAP50':>8} {'mAP50-95':>8} {'NaN':>6} {'发散':>6}")
        print(f"  {'─'*18} {'─'*6} {'─'*8} {'─'*8} {'─'*6} {'─'*6}")
        for r in rows:
            print(f"  {r.get('label','')[:18]:<18} {r.get('epoch', '-'):>6} "
                  f"{r.get('metrics/mAP50(B)', '-'):>8} {r.get('metrics/mAP50-95(B)', '-'):>8} "
                  f"{r.get('nan_detected', '-'):>6} {r.get('loss_diverged', '-'):>6}")

    # Routing analysis
    routing_csv = project / "routing/mot_routing_scenarios.csv"
    if routing_csv.exists():
        print(f"\n🔬 路由分析: {routing_csv}")

    # Deformable check
    deformable_csv = project / "routing/mot_deformable_activation_check.csv"
    if deformable_csv.exists():
        with open(deformable_csv) as f:
            rows = list(csv.DictReader(f))
        significant = [r for r in rows if r.get("deformable_significantly_higher") == "True"]
        print(f"\n🎯 DeformableTransformer 遮挡场景激活检验:")
        if significant:
            for r in significant:
                print(f"  ✅ {r['metric']}: 遮挡场景显著高于 {r['baseline']}")
                print(f"     irregular={r['irregular_mean']} vs baseline={r['baseline_mean']}")
                print(f"     p={r['permutation_p_value_one_sided']}, lift={r['relative_lift']}")
        else:
            print(f"  ⚠️  合成数据上 Deformable 激活率无显著差异（模型未训练）")
            print(f"  用真实训练好的 checkpoint 重新运行 --routing-only 获取有意义结果")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="0", help="GPU device (0) or cpu")
    p.add_argument("--data", default="ultralytics/cfg/datasets/coco128.yaml")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--max-images", type=int, default=128)

    # Partial runs
    p.add_argument("--check-only", action="store_true", help="仅校验 + benchmark，不训练")
    p.add_argument("--routing-only", action="store_true", help="仅路由分析（需要已有 checkpoint）")
    p.add_argument("--checkpoint", help="路由分析用的 .pt checkpoint 路径")
    p.add_argument("--image-dir", help="路由分析用的真实图片目录（场景分类）")
    p.add_argument("--skip-train", action="store_true", help="跳过训练，只做 benchmark+分析")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.routing_only:
        step_routing_analysis(args)
        print_results_summary()
        return 0

    # 1. Check build
    step_check_build(args)

    # 2. Benchmark
    step_benchmark(args)

    if args.check_only:
        print_results_summary()
        return 0

    # 3. Train (if not skipped)
    if not args.skip_train:
        rc = step_train(args)
        if rc != 0:
            print(f"\n[FAIL] Training failed (exit {rc})")
            return rc

    # 4. Routing analysis on best MoT checkpoint
    mot_best = ROOT / "runs/mot_ablation/v10_mot/weights/best.pt"
    if mot_best.exists():
        args.checkpoint = str(mot_best)
    step_routing_analysis(args)

    # 5. Summary
    step_summary(args)

    print("\n" + "="*60)
    print("  ✅ Issue #54 全流程完成！")
    print("="*60)
    print_results_summary()

    print(f"\n📁 所有输出在: {ROOT / 'runs/mot_ablation/'}")
    print(f"   build_summary.csv     — 模型结构对比")
    print(f"   latency_*.csv         — 延迟 Benchmark")
    print(f"   v10*/results.csv      — 各变体训练日志")
    print(f"   summary.csv           — 最终汇总对比")
    print(f"   routing/              — 路由可解释性分析")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
