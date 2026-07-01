#!/usr/bin/env python3
"""Reproduce issue #49 baselines on VisDrone."""

from __future__ import annotations

import argparse
from pathlib import Path

from reproduce_common import ROOT, add_common_args, run_reproduction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser,
        dataset_name="visdrone",
        default_data=ROOT / "ultralytics/cfg/datasets/VisDrone.yaml",
        default_root=ROOT / "datasets/VisDrone",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_reproduction(parse_args(), dataset_name="visdrone"))
