#!/usr/bin/env python3
"""
VOC comparison training: v0_12 (OptimalHybridGateMoE) vs v0_13 (MultiHeadRouterMoE)
vs v0_14 (DiversifiedExpertMoE) vs v0_15 (GatedFusionMoE).

Runs all four versions on the same VOC subset (3000 train / 800 val) with
identical hyperparameters, then produces a comparison table + plots.

Usage:
  python scripts/compare_moe_v0_13_15_voc.py --epochs 30 --batch 16
  python scripts/compare_moe_v0_13_15_voc.py --epochs 30 --batch 16 --device mps
  python scripts/compare_moe_v0_13_15_voc.py --epochs 30 --batch 16 --skip v0_12 --skip v0_14
"""
import argparse
import csv
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
    "v0_12": {
        "cfg": "master/v0_12/det/yolo-master-n.yaml",
        "name": "moe_v0_12_voc",
    },
    "v0_13": {
        "cfg": "master/v0_13/det/yolo-master-n.yaml",
        "name": "moe_v0_13_voc",
    },
    "v0_14": {
        "cfg": "master/v0_14/det/yolo-master-n.yaml",
        "name": "moe_v0_14_voc",
    },
    "v0_15": {
        "cfg": "master/v0_15/det/yolo-master-n.yaml",
        "name": "moe_v0_15_voc",
    },
}


def make_voc_yaml():
    """Create a temporary VOC yaml pointing at the local dataset."""
    tmp = ROOT / "scripts" / "_voc_local_v0_13_15.yaml"
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


def count_params(cfg_path):
    """Count model parameters from YAML config."""
    from ultralytics import YOLO
    model = YOLO(cfg_path)
    total = sum(p.numel() for p in model.model.parameters()) / 1e6
    return total


def run_training(version_key, cfg_path, data_yaml, epochs, batch, device, imgsz):
    """Train a single version and return the results dict."""
    from ultralytics import YOLO

    print(f"\n{'='*70}")
    print(f"  Training {version_key} | cfg={cfg_path}")
    print(f"  epochs={epochs} batch={batch} device={device} imgsz={imgsz}")
    print(f"{'='*70}\n")

    model = YOLO(cfg_path)
    name = VERSIONS[version_key]["name"]

    # Count parameters before training
    num_params = sum(p.numel() for p in model.model.parameters()) / 1e6
    print(f"  Model parameters: {num_params:.3f}M")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        device=device,
        imgsz=imgsz,
        project=str(ROOT / "runs" / "moe_voc_v0_13_15"),
        name=name,
        exist_ok=True,
        patience=0,
        cos_lr=True,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        workers=4,
        amp=True,
        verbose=True,
        save_period=1,
    )

    # Load results.csv
    results_csv = ROOT / "runs" / "moe_voc_v0_13_15" / name / "results.csv"
    metrics = {"version": version_key, "params_M": num_params}
    if results_csv.exists():
        with open(results_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                last = rows[-1]
                metrics["epoch"] = int(float(last.get("                  epoch", 0)))
                metrics["mAP50"] = float(last.get("       metrics/mAP50(B)", 0))
                metrics["mAP50-95"] = float(last.get("    metrics/mAP50-95(B)", 0))
                metrics["precision"] = float(last.get("   metrics/precision(B)", 0))
                metrics["recall"] = float(last.get("      metrics/recall(B)", 0))
                metrics["box_loss"] = float(last.get("         train/box_loss", 0))
                metrics["cls_loss"] = float(last.get("         train/cls_loss", 0))
                metrics["dfl_loss"] = float(last.get("         train/dfl_loss", 0))
                metrics["total_epochs"] = len(rows)

                # Extract per-epoch metrics for convergence plots
                epoch_metrics = []
                for row in rows:
                    try:
                        epoch_metrics.append({
                            "epoch": int(float(row.get("                  epoch", 0))),
                            "mAP50": float(row.get("       metrics/mAP50(B)", 0)),
                            "mAP50-95": float(row.get("    metrics/mAP50-95(B)", 0)),
                            "box_loss": float(row.get("         train/box_loss", 0)),
                            "cls_loss": float(row.get("         train/cls_loss", 0)),
                        })
                    except (ValueError, TypeError):
                        pass
                metrics["epoch_metrics"] = epoch_metrics

                # Sum time column
                for key in last:
                    if "time" in key.lower():
                        try:
                            total_time = sum(float(row[key]) for row in rows if row.get(key))
                            metrics["train_time_hours"] = total_time / 3600.0
                        except (ValueError, TypeError):
                            pass
                        break

    return metrics


def main():
    parser = argparse.ArgumentParser(description="MoE v0_12 vs v0_13 vs v0_14 vs v0_15 VOC comparison")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps/cpu)")
    parser.add_argument("--skip", action="append", default=[], help="Skip version (e.g., --skip v0_12)")
    args = parser.parse_args()

    data_yaml = make_voc_yaml()
    print(f"VOC data yaml: {data_yaml}")
    print(f"VOC data root: {VOC_DATA_ROOT}")

    all_metrics = {}

    for ver_key, ver_info in VERSIONS.items():
        if ver_key in args.skip:
            print(f"\n  Skipping {ver_key} (--skip)")
            continue

        cfg_path = str(ROOT / "ultralytics/cfg/models" / ver_info["cfg"])
        t0 = time.time()
        try:
            m = run_training(ver_key, cfg_path, data_yaml,
                             args.epochs, args.batch, args.device, args.imgsz)
            m["wall_time_hours"] = (time.time() - t0) / 3600.0
            all_metrics[ver_key] = m
        except Exception as e:
            print(f"\n  ERROR training {ver_key}: {e}")
            import traceback
            traceback.print_exc()
            all_metrics[ver_key] = {"error": str(e)}

    # Save and print comparison
    output_path = ROOT / "runs" / "moe_voc_v0_13_15" / "comparison_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        # Remove epoch_metrics for the summary JSON
        summary = {}
        for k, v in all_metrics.items():
            summary[k] = {kk: vv for kk, vv in v.items() if kk != "epoch_metrics"}
        json.dump(summary, f, indent=2)

    # Save full metrics with epoch-level data
    full_path = ROOT / "runs" / "moe_voc_v0_13_15" / "full_metrics.json"
    with open(full_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print("  COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"{'Version':<10} {'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8} {'Params':>8} {'Time(h)':>8}")
    print("-" * 80)
    for ver, m in all_metrics.items():
        if "error" in m:
            print(f"{ver:<10} ERROR: {m['error'][:40]}")
            continue
        print(f"{ver:<10} {m.get('mAP50', 0):>8.5f} {m.get('mAP50-95', 0):>10.5f} "
              f"{m.get('precision', 0):>8.5f} {m.get('recall', 0):>8.5f} "
              f"{m.get('params_M', 0):>7.3f}M {m.get('wall_time_hours', 0):>8.2f}")

    # Deltas vs v0_12
    if "v0_12" in all_metrics and "error" not in all_metrics["v0_12"]:
        b = all_metrics["v0_12"]
        print("-" * 80)
        for ver, m in all_metrics.items():
            if ver == "v0_12" or "error" in m:
                continue
            print(f"{'d'+ver:<10} {m.get('mAP50',0)-b.get('mAP50',0):>+8.5f} "
                  f"{m.get('mAP50-95',0)-b.get('mAP50-95',0):>+10.5f} "
                  f"{'':>8} {'':>8} "
                  f"{m.get('params_M',0)-b.get('params_M',0):>+7.3f}M "
                  f"{m.get('wall_time_hours',0)-b.get('wall_time_hours',0):>+8.2f}")

    print(f"\nResults saved to: {output_path}")
    print(f"Full metrics saved to: {full_path}")


if __name__ == "__main__":
    main()
