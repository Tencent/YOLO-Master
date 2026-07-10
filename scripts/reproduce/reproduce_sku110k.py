#!/usr/bin/env python3
"""Reproduce YOLO-Master-v0.1-N and YOLO-Master-EsMoE-N baselines on SKU-110K.

SKU-110K (retail, dense products, single class), built-in config SKU-110K.yaml.
By default EsMoE-N keeps the shipped sparse evaluation path. Add
--no-sparse-eval for train/eval-consistent dense validation; v0.1-N is
unaffected.

Examples:
    python scripts/reproduce/reproduce_sku110k.py --check-build
    python scripts/reproduce/reproduce_sku110k.py --epochs 300 --batch 64
    python scripts/reproduce/reproduce_sku110k.py --model EsMoE-N --no-sparse-eval
    python scripts/reproduce/reproduce_sku110k.py --model v0.1-N --no-wandb
    python scripts/reproduce/reproduce_sku110k.py --wandb-project my-proj --wandb-mode offline
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _reproduce_common import DatasetSpec, run_dataset  # noqa: E402

DATASET = DatasetSpec(
    name="SKU-110K",
    data="SKU-110K.yaml",
    project="runs/reproduce/sku110k",
)


if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET))
