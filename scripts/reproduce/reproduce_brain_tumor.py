#!/usr/bin/env python3
"""Reproduce YOLO-Master-v0.1-N and YOLO-Master-EsMoE-N baselines on brain-tumor.

brain-tumor (medical imaging: negative/positive tumor detection, 2 classes),
built-in config brain-tumor.yaml. Add --no-sparse-eval to opt into the
corrected dense evaluation for EsMoE-N (train==eval); v0.1-N is unaffected.

Examples:
    python scripts/reproduce/reproduce_brain_tumor.py --check-build
    python scripts/reproduce/reproduce_brain_tumor.py --epochs 200 --batch 32
    python scripts/reproduce/reproduce_brain_tumor.py --model EsMoE-N --no-sparse-eval
    python scripts/reproduce/reproduce_brain_tumor.py --model v0.1-N --no-wandb
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _reproduce_common import DatasetSpec, run_dataset  # noqa: E402

DATASET = DatasetSpec(
    name="brain-tumor",
    data="brain-tumor.yaml",
    project="runs/reproduce/brain-tumor",
)


if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET))
