#!/usr/bin/env python3
"""Fit teacher-agreement reliability weights on a disjoint GT calibration split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ov_distill.reliability import calibrate_reliability


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned", required=True, help="box-level JSONL from ov_distill clean")
    parser.add_argument("--gt-labels", required=True, help="YOLO-format COCO-80 GT label directory")
    parser.add_argument(
        "--calibrate-manifest", required=True, help="JSONL/path manifest for labeled calibration images"
    )
    parser.add_argument(
        "--train-manifest", required=True, help="training image manifest; must be disjoint from calibration"
    )
    parser.add_argument("--out", required=True, help="output calibration JSON used by convert --weight-mode calibrated")
    parser.add_argument("--class-split", default="", help="optional shared COCO-80 split; calibrate only seen prompts")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a pseudo box to match one GT box")
    parser.add_argument("--bins", type=int, default=10, help="number of equal-width bins for calibration metrics")
    parser.add_argument("--min-samples", type=int, default=50, help="minimum pseudo boxes needed to fit the mapping")
    parser.add_argument("--min-positives", type=int, default=5, help="minimum matched boxes needed")
    parser.add_argument("--min-negatives", type=int, default=5, help="minimum unmatched boxes needed")
    args = parser.parse_args()

    result = calibrate_reliability(
        cleaned=Path(args.cleaned),
        gt_labels=Path(args.gt_labels),
        calibrate_manifest=Path(args.calibrate_manifest),
        train_manifest=Path(args.train_manifest),
        out=Path(args.out),
        iou_thr=args.iou,
        bins=args.bins,
        min_samples=args.min_samples,
        min_positives=args.min_positives,
        min_negatives=args.min_negatives,
        class_split=Path(args.class_split) if args.class_split else None,
    )
    print(json.dumps({"calibrator": args.out, "statistics": result["statistics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
