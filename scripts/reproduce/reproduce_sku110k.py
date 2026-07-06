#!/usr/bin/env python3
"""Reproduce YOLO-Master nano baselines on SKU-110K."""

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

