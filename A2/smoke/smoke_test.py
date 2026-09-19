#!/usr/bin/env python3
"""Run the STAL small-object label-assignment smoke test end-to-end.

Reproducible entry point for the "STAL 式小目标自适应标签分配" admission smoke test.
It (1) generates a tiny VisDrone-format subset (10 classes, small 8-24 px objects),
(2) trains YOLO-Master v0.1-N from scratch for one epoch end-to-end (data -> assigner -> loss -> val),
and (3) prints the metric log line (box/cls/dfl loss + per-class mAP) and the run dir.

The assigner exercised here is ``TaskAlignedAssigner`` in ``ultralytics/utils/tal.py``.
The loss/config injection points are ``v8DetectionLoss`` in ``ultralytics/utils/loss.py``
and ``DetectionModel.init_criterion`` in ``ultralytics/nn/tasks.py``.

Examples:
    python A2/smoke/smoke_test.py --device 0
    python A2/smoke/smoke_test.py --device cpu --imgsz 320 --epochs 1
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
# Ensure this repo's ``ultralytics`` is imported when ``run_smoke`` does its local import (a stale
# editable install elsewhere would shadow it).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET_DIR = REPO_ROOT / "datasets" / "VisDrone-smoke"
DATASET_YAML = REPO_ROOT / "datasets" / "VisDrone-smoke.yaml"
# YOLO-Master v0.1-N (MoE) — the reproduction-protocol baseline model (NOT stock yolo11n.yaml).
MODEL_YAML = REPO_ROOT / "ultralytics" / "cfg" / "models" / "master" / "v0_1" / "det" / "yolo-master-n.yaml"

# VisDrone2019-DET class order (matches ultralytics/cfg/datasets/VisDrone.yaml).
VISDRONE_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def _write_dataset_yaml() -> Path:
    """Write the VisDrone-smoke dataset config with an absolute root path."""
    lines = [f"path: {DATASET_DIR}", "train: images/train", "val: images/val", "names:"]
    lines.extend(f"  {i}: {name}" for i, name in enumerate(VISDRONE_NAMES))
    DATASET_YAML.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return DATASET_YAML


def make_subset(seed: int = 0, num_train: int = 6, num_val: int = 2) -> Path:
    """Generate the tiny VisDrone-format subset (idempotent). Returns the YAML path."""
    random.seed(seed)
    np.random.seed(seed)
    for split, count in (("train", num_train), ("val", num_val)):
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            size = 640
            img = Image.fromarray(np.random.randint(0, 60, (size, size, 3), dtype=np.uint8))
            draw = ImageDraw.Draw(img)
            labels = []
            for _ in range(random.randint(3, 6)):  # small objects: 8-24 px side
                cls = random.randint(0, len(VISDRONE_NAMES) - 1)
                bw = random.randint(8, 24)
                bh = random.randint(8, 24)
                x = random.randint(0, size - bw)
                y = random.randint(0, size - bh)
                draw.rectangle([x, y, x + bw, y + bh], outline=255, width=2)
                labels.append(
                    f"{cls} {(x + bw / 2) / size:.6f} {(y + bh / 2) / size:.6f} {bw / size:.6f} {bh / size:.6f}"
                )
            img.save(img_dir / f"{split}_{i:03d}.jpg")
            (lbl_dir / f"{split}_{i:03d}.txt").write_text("\n".join(labels), encoding="utf-8")
    return _write_dataset_yaml()


def run_smoke(model_cfg: str, data_yaml: str, epochs: int, imgsz: int, batch: int, device: str) -> None:
    """Train one epoch from scratch and print the resulting metric log line."""
    from ultralytics import YOLO  # local import: only needed at runtime

    model = YOLO(model_cfg)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=0,
        pretrained=False,
        verbose=True,
    )
    results_csv = Path(model.trainer.save_dir) / "results.csv"
    print("\n=== STAL smoke test metric log ===")
    if results_csv.exists():
        print(results_csv.read_text(encoding="utf-8").strip())
    print(f"Run artifacts: {model.trainer.save_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="STAL small-object label-assignment smoke test")
    parser.add_argument("--model", default=str(MODEL_YAML), help="model config (from-scratch, no .pt)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0", help="device, e.g. '0' (GPU) or 'cpu'")
    args = parser.parse_args()

    data_yaml = make_subset()
    print(f"Generated VisDrone-smoke subset: {DATASET_DIR} ({data_yaml})")
    run_smoke(args.model, str(data_yaml), args.epochs, args.imgsz, args.batch, args.device)


if __name__ == "__main__":
    main()
