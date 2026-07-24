#!/usr/bin/env python3
"""Reproduce YOLO-Master-v0.1-N and YOLO-Master-EsMoE-N baselines on construction-ppe.

construction-ppe (industrial safety: helmet/vest/mask/goggles, 4 classes),
built-in config construction-ppe.yaml. Add --no-sparse-eval to opt into the
corrected dense evaluation for EsMoE-N (train==eval); v0.1-N is unaffected.

Examples:
    python scripts/reproduce/reproduce_construction_ppe.py --check-build
    python scripts/reproduce/reproduce_construction_ppe.py --epochs 100 --batch 32
    python scripts/reproduce/reproduce_construction_ppe.py --model EsMoE-N --no-sparse-eval
    python scripts/reproduce/reproduce_construction_ppe.py --model v0.1-N --no-wandb
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _reproduce_common import DatasetSpec, run_dataset  # noqa: E402

DATASET = DatasetSpec(
    name="construction-ppe",
    data="construction-ppe.yaml",
    project="runs/reproduce/ppe",
)


if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET))
