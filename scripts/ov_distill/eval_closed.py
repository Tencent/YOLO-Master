#!/usr/bin/env python3
"""OV-Distill: closed-set evaluation on a YOLO-format validation set (e.g. val1k.yaml).

Complements eval_ov_holdout.py (unseen classes): this script measures detection
quality on the closed COCO 80-class protocol. The YOLOE validator applies
``set_classes`` from the yaml's names automatically, so no external class
setup is required.

Usage:
  python scripts/ov_distill/eval_closed.py \
      --weights runs/ov_s2_train/s5k_q16/weights/best.pt \
      --data runs/ov_s2_val1k/val1k.yaml --device mps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Closed-set evaluation on a YOLO data yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="runs/ov_s2_val1k/val1k.yaml")
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="", help="optional path to write the JSON report")
    args = parser.parse_args()

    from ultralytics import YOLOE

    model = YOLOE(args.weights)
    r = model.val(
        data=str((ROOT / args.data).resolve()),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        verbose=False,
        plots=False,
    )
    report = {
        "weights": args.weights,
        "data": args.data,
        "mAP50": round(float(r.box.map50), 4),
        "mAP50-95": round(float(r.box.map), 4),
        "precision": round(float(r.box.mp), 4),
        "recall": round(float(r.box.mr), 4),
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
