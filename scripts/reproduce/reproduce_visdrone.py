#!/usr/bin/env python3
"""Reproduce YOLO-Master nano baselines on VisDrone."""

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

