#!/usr/bin/env python3
"""Reproduce YOLO-Master-v0.1-N and YOLO-Master-EsMoE-N baselines on VisDrone.

VisDrone (aerial, dense small objects), built-in config VisDrone.yaml.
By default the models are reproduced as-is (EsMoE-N keeps its sparse eval, which
collapses mAP). Add --no-sparse-eval to opt into the corrected dense evaluation
for EsMoE-N (train==eval); v0.1-N is unaffected.

Examples:
    python scripts/reproduce/reproduce_visdrone.py --check-build
    python scripts/reproduce/reproduce_visdrone.py --epochs 300 --batch 64                 # as-is
    python scripts/reproduce/reproduce_visdrone.py --model EsMoE-N --no-sparse-eval        # corrected
    python scripts/reproduce/reproduce_visdrone.py --model v0.1-N --no-wandb
    python scripts/reproduce/reproduce_visdrone.py --wandb-project my-proj --wandb-mode offline
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _reproduce_common import DatasetSpec, run_dataset  # noqa: E402

DATASET = DatasetSpec(
    name="VisDrone",
    data="VisDrone.yaml",
    project="runs/reproduce/visdrone",
)


if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET))
