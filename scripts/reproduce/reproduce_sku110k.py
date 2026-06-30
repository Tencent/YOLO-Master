#!/usr/bin/env python3
"""Reproduce YOLO-Master-v0.1-N vs YOLO-Master-EsMoE-N on SKU-110K."""

from _domain_baseline_common import DatasetSpec, run_reproduction


if __name__ == "__main__":
    run_reproduction(
        DatasetSpec(
            name="sku110k",
            data="SKU-110K.yaml",
            project="runs/reproduce/sku110k",
            max_det=1000,
            batch=16,
        )
    )
