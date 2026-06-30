#!/usr/bin/env python3
"""Reproduce YOLO-Master-v0.1-N vs YOLO-Master-EsMoE-N on VisDrone."""

from _domain_baseline_common import DatasetSpec, run_reproduction


if __name__ == "__main__":
    run_reproduction(
        DatasetSpec(
            name="visdrone",
            data="VisDrone.yaml",
            project="runs/reproduce/visdrone",
            max_det=500,
            batch=16,
        )
    )
