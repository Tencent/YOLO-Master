#!/usr/bin/env python3
"""Compare saved backend output tensors for edge deployment validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from edge_utils import compare_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Reference PyTorch output .npy")
    parser.add_argument("--candidate", type=Path, required=True, help="Exported backend output .npy")
    parser.add_argument("--tolerance", type=float, default=0.005, help="Max absolute error tolerance")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference = np.load(args.reference)
    candidate = np.load(args.candidate)
    report = compare_arrays(reference, candidate, args.tolerance)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
