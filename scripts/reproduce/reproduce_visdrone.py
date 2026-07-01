#!/usr/bin/env python3
"""
VisDrone reproduction script — YOLO-Master-v0.1-N & YOLO-Master-EsMoE-N
=======================================================================

Trains both model variants on the VisDrone2019-DET dataset (aerial dense
small-object detection, 10 classes) and logs metrics to Weights & Biases.

Usage
-----
    # Default: 300 epochs, batch 16, single GPU
    python scripts/reproduce/reproduce_visdrone.py

    # Custom settings
    python scripts/reproduce/reproduce_visdrone.py --epochs 100 --batch 32 --device 0,1

    # Train only one variant
    python scripts/reproduce/reproduce_visdrone.py --variants esmoe-n

Dataset
-------
    VisDrone2019-DET is automatically downloaded on first run (~2.3 GB).
    - 6 471 train / 548 val / 1 610 test images
    - 10 classes: pedestrian, people, bicycle, car, van, truck, tricycle,
                  awning-tricycle, bus, motor
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO, settings

# ---------------------------------------------------------------------------
# Model variant registry
# ---------------------------------------------------------------------------
MODEL_VARIANTS = {
    "esmoe-n": {
        "cfg": "ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml",
        "name": "esmoe-n",
        "desc": "YOLO-Master-EsMoE-N (v0, ES_MOE)",
    },
    "master-v01-n": {
        "cfg": "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml",
        "name": "master-v01-n",
        "desc": "YOLO-Master-v0.1-N (ModularRouterExpertMoE / OptimizedMOEImproved)",
    },
}

# ---------------------------------------------------------------------------
# Training hyper-parameters (following paper recommendations)
# ---------------------------------------------------------------------------
TRAIN_ARGS_DEFAULTS = dict(
    data="VisDrone.yaml",
    imgsz=640,
    epochs=300,
    batch=16,
    device="0",
    workers=8,
    # Augmentations
    scale=0.5,
    mosaic=1.0,
    mixup=0.0,             # disabled — less effective for dense small objects
    copy_paste=0.1,
    close_mosaic=15,       # disable mosaic for last 15 epochs
    # MoE loss coefficient (default 0.15 in ultralytics/cfg/default.yaml)
    # Override here if needed:  moe=0.15
    # Misc
    exist_ok=True,         # overwrite existing experiment directory
    seed=42,
    deterministic=True,
    single_cls=False,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce YOLO-Master models on VisDrone2019-DET",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--project",
        type=str,
        default="reproduce/visdrone",
        help="Parent directory for training outputs and W&B project name.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["esmoe-n", "master-v01-n"],
        choices=list(MODEL_VARIANTS.keys()),
        help="Which model variant(s) to train.",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint (per variant).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # --- Weights & Biases ---------------------------------------------------
    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", "yolo-master-reproduce-visdrone")
        settings.update({"wandb": True})
        print("📊 W&B logging enabled — visit https://wandb.ai to track runs")
    else:
        settings.update({"wandb": False})

    # --- Common training kwargs ----------------------------------------------
    train_kwargs = {**TRAIN_ARGS_DEFAULTS}
    train_kwargs.update(
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        resume=args.resume,
    )

    print("=" * 70)
    print("VisDrone reproduction — YOLO-Master v0.1 & EsMoE")
    print(f"GPU(s): {args.device}  |  Epochs: {args.epochs}  |  Batch: {args.batch}")
    print(f"Variants: {args.variants}")
    print("=" * 70)

    results_all = {}

    for key in args.variants:
        var = MODEL_VARIANTS[key]
        print(f"\n{'─' * 60}")
        print(f"▶  Training {var['desc']}")
        print(f"   Config: {var['cfg']}")
        print(f"   Run name: {var['name']}")
        print(f"{'─' * 60}")

        model = YOLO(var["cfg"])
        results = model.train(name=var["name"], **train_kwargs)
        results_all[key] = results

        print(f"✅ {var['desc']} training finished.\n")

    # --- Summary ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Training completed for all variants.")
    print("Results summary:")
    for key, res in results_all.items():
        print(f"  {key}: {res}")
    print("=" * 70)


if __name__ == "__main__":
    main()
