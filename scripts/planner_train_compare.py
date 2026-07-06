#!/usr/bin/env python3
"""Short training comparison to validate Planner effectiveness.

Runs 2-epoch training on COCO128 with Planner ON vs OFF, comparing mAP metrics.
Results are saved to runs/planner_train_compare/ for post-analysis.

Usage:
    python scripts/planner_train_compare.py --model yolo11s.pt --epochs 2
    python scripts/planner_train_compare.py --model yolo12s.pt --epochs 2 --variant dora
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO


def run_experiment(model_name: str, epochs: int, planner_enabled: bool, variant: str, rank: int):
    """Run a single training experiment."""
    label = "ON" if planner_enabled else "OFF"
    print(f"\n{'='*60}")
    print(f"  Experiment: {model_name} | Planner {label} | {variant} r={rank}")
    print(f"{'='*60}")

    t0 = time.time()
    model = YOLO(model_name)

    r = model.train(
        data="ultralytics/cfg/datasets/coco128.yaml",
        epochs=epochs,
        batch=8,
        imgsz=320,
        lora_r=rank,
        lora_type=variant,
        lora_planner_enabled=planner_enabled,
        device="mps",
        verbose=False,
        project="runs/planner_train_compare",
        name=f"{model_name.replace('.pt','')}_planner_{label}_{variant}_r{rank}_{datetime.now().strftime('%H%M%S')}",
    )

    elapsed = time.time() - t0
    # Ultralytics train() may return a dict or DetMetrics object depending on version
    metrics = getattr(r, "results_dict", r) if not isinstance(r, dict) else r
    if isinstance(metrics, dict):
        m50 = metrics.get("metrics/mAP50(B)")
        m5095 = metrics.get("metrics/mAP50-95(B)")
        prec = metrics.get("metrics/precision(B)")
        rec = metrics.get("metrics/recall(B)")
    else:
        # Fallback for DetMetrics object
        m50 = getattr(metrics, "mAP50", None)
        m5095 = getattr(metrics, "mAP50_95", None)
        prec = getattr(metrics, "precision", None)
        rec = getattr(metrics, "recall", None)

    results = {
        "model": model_name,
        "planner": label,
        "variant": variant,
        "rank": rank,
        "epochs": epochs,
        "elapsed_sec": elapsed,
        "mAP50": m50,
        "mAP50-95": m5095,
        "precision": prec,
        "recall": rec,
        "timestamp": datetime.now().isoformat(),
    }
    print(f"  mAP50={m50:.4f}, mAP50-95={m5095:.4f}, time={elapsed/60:.1f}min")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11s.pt", help="Model weights")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    parser.add_argument("--variant", default="lora", help="PEFT variant for OFF run")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    args = parser.parse_args()

    output_dir = Path("runs/planner_train_compare")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_file = output_dir / f"compare_{args.model.replace('.pt','')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Run OFF then ON
    off_results = run_experiment(args.model, args.epochs, False, args.variant, args.rank)
    on_results = run_experiment(args.model, args.epochs, True, args.variant, args.rank)

    report = {
        "model": args.model,
        "epochs": args.epochs,
        "off": off_results,
        "on": on_results,
        "delta_mAP50": on_results["mAP50"] - off_results["mAP50"],
        "delta_mAP50-95": on_results["mAP50-95"] - off_results["mAP50-95"],
    }

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print("  Comparison Summary")
    print(f"{'='*60}")
    print(f"  OFF mAP50:     {off_results['mAP50']:.4f}")
    print(f"  ON  mAP50:     {on_results['mAP50']:.4f}")
    print(f"  Δ mAP50:       {report['delta_mAP50']:+.4f}")
    print(f"  OFF mAP50-95:  {off_results['mAP50-95']:.4f}")
    print(f"  ON  mAP50-95:  {on_results['mAP50-95']:.4f}")
    print(f"  Δ mAP50-95:    {report['delta_mAP50-95']:+.4f}")
    print(f"\n  Report saved: {report_file}")


if __name__ == "__main__":
    main()
