#!/usr/bin/env python3
"""Issue #53: MoA vs MoE baseline training on VisDrone.

Quick validation (CPU, 1 epoch, imgsz=320):
    python scripts/issue53/train_visdrone_moa_vs_baseline.py --epochs 1 --imgsz 320 --device cpu

Full training (GPU, 50 epochs):
    python scripts/issue53/train_visdrone_moa_vs_baseline.py --epochs 50 --imgsz 640 --device 0
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from datetime import datetime

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

# Models
MOA_CFG = ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-n.yaml"
BASELINE_CFG = ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-n.yaml"

# Dataset
DATA_YAML = Path(r"F:\研究\VisDrone\VisDrone.yaml")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--project", type=Path, default=ROOT / "runs/issue53_visdrone")
    p.add_argument("--models", nargs="+", default=["moa", "baseline"],
                   choices=["moa", "baseline"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training, only write summary from existing runs")
    return p.parse_args()


def train_model(cfg_path: Path, name: str, args) -> Path:
    """Train one model and return run directory."""
    model = YOLO(str(cfg_path))
    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        project=str(args.project),
        name=name,
        exist_ok=True,
        pretrained=False,
        val=True,
        plots=True,
        patience=args.patience,
        amp=(args.device != "cpu"),
        verbose=True,
    )
    return args.project / name


def collect_metrics(run_dir: Path) -> dict:
    """Extract key metrics from results.csv."""
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return {"status": "no_results"}
    with open(results_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"status": "empty"}
    last = rows[-1]
    return {
        "epoch": int(float(last.get("epoch", 0))),
        "mAP50": float(last.get("metrics/mAP50(B)", 0)),
        "mAP50-95": float(last.get("metrics/mAP50-95(B)", 0)),
        "precision": float(last.get("metrics/precision(B)", 0)),
        "recall": float(last.get("metrics/recall(B)", 0)),
        "box_loss": float(last.get("train/box_loss", 0)),
        "cls_loss": float(last.get("train/cls_loss", 0)),
        "dfl_loss": float(last.get("train/dfl_loss", 0)),
        "val_box_loss": float(last.get("val/box_loss", 0)),
        "val_cls_loss": float(last.get("val/cls_loss", 0)),
        "val_dfl_loss": float(last.get("val/dfl_loss", 0)),
    }


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg_map = {
        "moa": (MOA_CFG, "yolo_master_moa_n"),
        "baseline": (BASELINE_CFG, "yolo_master_n_baseline"),
    }

    if not DATA_YAML.exists():
        raise SystemExit(f"Dataset YAML not found: {DATA_YAML}")

    results = {}
    for model_key in args.models:
        cfg_path, name = cfg_map[model_key]
        run_name = f"{name}_{timestamp}"

        if args.skip_train:
            # Find latest run
            existing = sorted(args.project.glob(f"{name}_*"))
            if existing:
                run_dir = existing[-1]
                print(f"[skip] Using existing run: {run_dir}")
            else:
                print(f"[skip] No existing run for {name}, skipping")
                continue
        else:
            print(f"\n{'='*60}")
            print(f"Training: {name}")
            print(f"Config: {cfg_path}")
            print(f"Epochs: {args.epochs}, imgsz: {args.imgsz}, batch: {args.batch}")
            print(f"Device: {args.device}")
            print(f"{'='*60}\n")

            run_dir = train_model(cfg_path, run_name, args)

        metrics = collect_metrics(run_dir)
        metrics["model"] = model_key
        metrics["run_dir"] = str(run_dir)
        results[model_key] = metrics

        print(f"\n--- {name} Results ---")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    # Write summary
    summary_path = args.project / f"comparison_summary_{timestamp}.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    all_keys = set()
    for r in results.values():
        all_keys.update(r.keys())
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
        writer.writeheader()
        for r in results.values():
            writer.writerow(r)
    print(f"\nSummary written to: {summary_path}")

    # Comparison
    if "moa" in results and "baseline" in results:
        moa_map = results["moa"].get("mAP50-95", 0)
        bl_map = results["baseline"].get("mAP50-95", 0)
        delta = moa_map - bl_map
        print(f"\n=== MoA vs Baseline Comparison ===")
        print(f"MoA mAP50-95:      {moa_map:.4f}")
        print(f"Baseline mAP50-95: {bl_map:.4f}")
        print(f"Delta (MoA - BL):  {delta:+.4f}")


if __name__ == "__main__":
    main()
