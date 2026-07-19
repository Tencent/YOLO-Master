#!/usr/bin/env python3
"""Quick GPU-memory probe for Issue 53 VisDrone training."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402


MODEL_CFGS = {
    "v10": ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-n.yaml",
    "v10_moa": ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-n.yaml",
}
CLUSTER_ROOT = Path(os.environ.get("YOLO_ISSUE53_ROOT", Path.home() / "yolo-master-issue53"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_CFGS), required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--data", type=Path, default=CLUSTER_ROOT / "datasets/VisDrone/visdrone.yaml")
    parser.add_argument("--project", type=Path, default=CLUSTER_ROOT / "runs/issue53_batch_probe")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--fraction", type=float, default=0.03)
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MODEL_CFGS[args.model]
    name = f"{args.model}_b{args.batch}"
    model = YOLO(str(cfg))
    model.train(
        data=str(args.data),
        epochs=1,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=42,
        deterministic=True,
        project=str(args.project),
        name=name,
        exist_ok=True,
        pretrained=False,
        val=False,
        plots=False,
        cache=False,
        patience=0,
        amp=args.amp,
        fraction=args.fraction,
        verbose=False,
    )


if __name__ == "__main__":
    main()
