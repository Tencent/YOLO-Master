"""
Reproduce YOLO-Master on visdrone dataset
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
from ultralytics.utils import SETTINGS

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATASET = "VisDrone.yaml"
DATASET_TAG = "visdrone"
MODELS = {
    "v0.1":  ("ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml", "v0_1_n"),
    "esmoe": ("ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml",   "esmoe_n"),
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", choices=list(MODELS)+["all"], default=["all"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", default="yolo-master-reproduce")
    return p.parse_args()

def main():
    args = parse_args()

    SETTINGS.update(wandb=True)
    os.environ.setdefault("WANDB_PROJECT", args.project)
    from ultralytics import YOLO

    selected = list(MODELS) if "all" in args.models else args.models
    print(f"[reproduce] dataset={DATASET} device={args.device} epochs={args.epochs} models={selected}")
    summary = []
    for key in selected:
        cfg, suffix = MODELS[key]
        run_name = f"{DATASET_TAG}_{suffix}"
        print(f"\n[reproduce] === Training {key} ({cfg}) -> {run_name} ===")
        model = YOLO(str(ROOT / cfg))
        model.train(
            data=DATASET,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=run_name,
            scale=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.1,
        )
        metrics = model.val(data=DATASET)
        summary.append((key, run_name, metrics.box.map50, metrics.box.map))

    print("\n[reproduce] ===== Final results (VisDrone) =====")
    print(f"{'model':<8} {'run':<20} {'mAP50':>8} {'mAP50-95':>10}")
    for key, run_name, map50, mAP in summary:
        print(f"{key:<8} {run_name:<20} {map50:>8.4f} {mAP:>10.4f}")

if __name__ == "__main__":
    main()