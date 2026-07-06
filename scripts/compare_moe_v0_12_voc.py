#!/usr/bin/env python3
"""
VOC comparison training: v0_6 (HybridAdaptiveGateMoE) vs v0_12 (OptimalHybridGateMoE).

Runs both versions on the same VOC subset (3000 train / 800 val) with identical
hyperparameters, then produces a comparison table + plots.

Usage:
  python scripts/compare_moe_v0_12_voc.py --epochs 50 --batch 16
  python scripts/compare_moe_v0_12_voc.py --epochs 50 --batch 16 --device mps
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VOC_DATA_ROOT = Path("/Users/gatilin/Downloads/.session_tmps/bed05d0f-229c-45d5-a6c5-2cc4abe4350e/datasets/VOC")

VERSIONS = {
    "v0_6": {
        "cfg": "master/v0_6/det/yolo-master-n.yaml",
        "name": "moe_v0_6_voc",
    },
    "v0_12": {
        "cfg": "master/v0_12/det/yolo-master-n.yaml",
        "name": "moe_v0_12_voc",
    },
}


def make_voc_yaml():
    """Create a temporary VOC yaml pointing at the local dataset."""
    tmp = ROOT / "scripts" / "_voc_local.yaml"
    lines = [
        f"path: {VOC_DATA_ROOT}",
        "train: voc_sub_train.txt",
        "val: voc_sub_val.txt",
        "test: voc_sub_val.txt",
        "",
        "names:",
    ]
    names = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
        "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    ]
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")
    tmp.write_text("\n".join(lines) + "\n")
    return str(tmp)


def run_training(version_key, cfg_path, data_yaml, epochs, batch, device, imgsz):
    """Train a single version and return the results dict."""
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"  Training {version_key} | cfg={cfg_path}")
    print(f"  epochs={epochs} batch={batch} device={device} imgsz={imgsz}")
    print(f"{'='*60}\n")

    model = YOLO(cfg_path)
    name = VERSIONS[version_key]["name"]

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        device=device,
        imgsz=imgsz,
        project=str(ROOT / "runs" / "moe_voc_compare"),
        name=name,
        exist_ok=True,
        patience=0,           # no early stopping — full epoch range
        cos_lr=True,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        workers=4,
        amp=True,
        verbose=True,
        # Save every epoch so we can compare convergence
        save_period=1,
    )

    # Load results.csv
    results_csv = ROOT / "runs" / "moe_voc_compare" / name / "results.csv"
    metrics = {}
    if results_csv.exists():
        import csv
        with open(results_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                last = rows[-1]
                metrics = {
                    "epoch": int(float(last.get("                  epoch", 0))),
                    "mAP50": float(last.get("       metrics/mAP50(B)", 0)),
                    "mAP50-95": float(last.get("    metrics/mAP50-95(B)", 0)),
                    "precision": float(last.get("   metrics/precision(B)", 0)),
                    "recall": float(last.get("      metrics/recall(B)", 0)),
                    "box_loss": float(last.get("         train/box_loss", 0)),
                    "cls_loss": float(last.get("         train/cls_loss", 0)),
                    "dfl_loss": float(last.get("         train/dfl_loss", 0)),
                }
                # Training time from the directory timestamps
                train_dir = ROOT / "runs" / "moe_voc_compare" / name
                # Sum per-epoch wall times from results if available
                all_times = []
                for r in rows:
                    t = r.get("                train/box_loss", None)
                    # Try time column if present
                    for key in r:
                        if "time" in key.lower():
                            try:
                                all_times.append(float(r[key]))
                            except (ValueError, TypeError):
                                pass
                            break
                metrics["total_rows"] = len(rows)
                # Check for time column
                for key in last:
                    if "time" in key.lower():
                        try:
                            total_time = sum(
                                float(row[key]) for row in rows
                                if row.get(key)
                            )
                            metrics["train_time_hours"] = total_time / 3600.0
                        except (ValueError, TypeError):
                            pass
                        break

    return metrics


def main():
    parser = argparse.ArgumentParser(description="MoE v0_6 vs v0_12 VOC comparison")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps/cpu)")
    parser.add_argument("--skip-v0_6", action="store_true", help="Skip v0_6 training")
    parser.add_argument("--skip-v0_12", action="store_true", help="Skip v0_12 training")
    args = parser.parse_args()

    data_yaml = make_voc_yaml()
    print(f"VOC data yaml: {data_yaml}")
    print(f"VOC data root: {VOC_DATA_ROOT}")

    all_metrics = {}

    # Train v0_6 (baseline)
    if not args.skip_v0_6:
        cfg_v0_6 = str(ROOT / "ultralytics/cfg/models" / VERSIONS["v0_6"]["cfg"])
        t0 = time.time()
        m = run_training("v0_6", cfg_v0_6, data_yaml,
                         args.epochs, args.batch, args.device, args.imgsz)
        m["wall_time_hours"] = (time.time() - t0) / 3600.0
        all_metrics["v0_6"] = m

    # Train v0_12 (new)
    if not args.skip_v0_12:
        cfg_v0_12 = str(ROOT / "ultralytics/cfg/models" / VERSIONS["v0_12"]["cfg"])
        t0 = time.time()
        m = run_training("v0_12", cfg_v0_12, data_yaml,
                         args.epochs, args.batch, args.device, args.imgsz)
        m["wall_time_hours"] = (time.time() - t0) / 3600.0
        all_metrics["v0_12"] = m

    # Save and print comparison
    output_path = ROOT / "runs" / "moe_voc_compare" / "comparison_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n{'='*60}")
    print("  COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"{'Version':<10} {'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8} {'Time(h)':>8}")
    print("-" * 60)
    for ver, m in all_metrics.items():
        print(f"{ver:<10} {m.get('mAP50', 0):>8.5f} {m.get('mAP50-95', 0):>10.5f} "
              f"{m.get('precision', 0):>8.5f} {m.get('recall', 0):>8.5f} "
              f"{m.get('wall_time_hours', 0):>8.2f}")

    if "v0_6" in all_metrics and "v0_12" in all_metrics:
        d = all_metrics["v0_12"]
        b = all_metrics["v0_6"]
        print("-" * 60)
        print(f"{'delta':<10} {d.get('mAP50',0)-b.get('mAP50',0):>+8.5f} "
              f"{d.get('mAP50-95',0)-b.get('mAP50-95',0):>+10.5f} "
              f"{'':>8} {'':>8} "
              f"{d.get('wall_time_hours',0)-b.get('wall_time_hours',0):>+8.2f}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
