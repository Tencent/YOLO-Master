#!/usr/bin/env python3
"""Sync existing training results (results.csv) to Weights & Biases.

This script imports metrics from existing training runs into WandB,
useful when training was done with --no-wandb but WandB links are needed.

Usage:
    python scripts/reproduce/sync_results_to_wandb.py --project yolo-master-reproduce
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sync_to_wandb(project: str, entity: str | None, run_name: str, results_csv: Path):
    """Sync results.csv data to a new WandB run."""
    try:
        import wandb
    except ImportError:
        print("Error: wandb is not installed. Run `pip install wandb` first.")
        return None

    if not results_csv.exists():
        print(f"Error: {results_csv} not found")
        return None

    # Read results.csv
    with results_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(f"Error: {results_csv} is empty")
        return None

    print(f"[sync] {run_name}: {len(rows)} epochs found")

    # Initialize WandB run
    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            reinit=True,
            config={
                "run_name": run_name,
                "results_source": str(results_csv),
                "total_epochs": len(rows),
            },
        )
    except Exception as e:
        print(f"Error: wandb init failed - {e}")
        print("Try running `wandb login` first, or check your API key.")
        return None

    # Log each epoch
    for row in rows:
        try:
            epoch = int(float(row.get("epoch", 0)))
            log = {"epoch": epoch}
            
            # mAP metrics
            for key in ["metrics/precision(B)", "metrics/recall(B)", 
                        "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
                val = row.get(key)
                if val:
                    try:
                        log[key] = float(val)
                    except ValueError:
                        pass
            
            # Loss metrics
            for key in ["train/box_loss", "train/cls_loss", "train/dfl_loss", "train/moe_loss",
                        "val/box_loss", "val/cls_loss", "val/dfl_loss", "val/moe_loss"]:
                val = row.get(key)
                if val:
                    try:
                        log[key] = float(val)
                    except ValueError:
                        pass
            
            run.log(log, step=epoch)
        except Exception as e:
            print(f"Warning: Failed to log epoch {epoch}: {e}")
            continue

    run.finish()
    print(f"[sync] {run_name} -> {run.url}")
    return run.url


def main():
    parser = argparse.ArgumentParser(description="Sync results.csv to WandB")
    parser.add_argument("--project", default="yolo-master-reproduce", help="W&B project name")
    parser.add_argument("--entity", default="", help="W&B entity/team (optional)")
    parser.add_argument("--dry-run", action="store_true", help="List runs without syncing")
    args = parser.parse_args()

    # Define runs to sync
    runs = [
        # VisDrone
        ("VisDrone_v0.1-N", ROOT / "runs/reproduce/visdrone/VisDrone_v0.1-N/results.csv"),
        ("VisDrone_EsMoE-N", ROOT / "runs/reproduce/visdrone/VisDrone_EsMoE-N/results.csv"),
        # SKU-110K
        ("SKU-110K_v0.1-N", ROOT / "runs/reproduce/sku110k/SKU-110K_v0.1-N/results.csv"),
        ("SKU-110K_EsMoE-N", ROOT / "runs/reproduce/sku110k/SKU-110K_EsMoE-N/results.csv"),
    ]

    print(f"Found {len(runs)} runs to sync")
    for run_name, csv_path in runs:
        exists = "✓" if csv_path.exists() else "✗"
        print(f"  {exists} {run_name}: {csv_path}")

    if args.dry_run:
        return

    urls = []
    for run_name, csv_path in runs:
        url = sync_to_wandb(args.project, args.entity or None, run_name, csv_path)
        if url:
            urls.append((run_name, url))

    print("\n=== WandB Links ===")
    for run_name, url in urls:
        print(f"- {run_name}: {url}")


if __name__ == "__main__":
    main()
