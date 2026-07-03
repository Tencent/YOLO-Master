#!/usr/bin/env python3
"""Fine-tune YOLO-Master-EsMoE-N on the VisDrone vertical-domain dataset.

The COCO-pretrained EsMoE-N checkpoint (nc=80) is reused; ultralytics rebuilds the
Detect head for VisDrone's 10 classes while keeping the ES-MoE backbone weights.
Run from the project root:
    python scripts/train_visdrone.py --epochs 50 --imgsz 640 --device 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJ_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJ_ROOT.parent / "weights" / "YOLO-Master-EsMoE-N.pt"
DEFAULT_DATA = PROJ_ROOT / "configs" / "visdrone.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune EsMoE-N on VisDrone")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="pretrained .pt checkpoint")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="dataset yaml")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=-1, help="-1 = auto batch")
    p.add_argument("--device", type=str, default="0", help="cuda device, e.g. 0 or 0,1")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="freeze first N layers (13 = full backbone incl. all ES_MOE, preserves "
        "pretrained eval-mode MoE routing while adapting the head to VisDrone)",
    )
    p.add_argument("--name", type=str, default="esmoe_n_visdrone")
    p.add_argument(
        "--dense",
        action="store_true",
        help="train in dense-routing mode (use_top_k=False). Makes train/eval/export "
        "use identical full-softmax dense features -> export faithful AND no BN drift.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[train] model={args.model} data={args.data}")
    assert args.model.exists(), f"checkpoint not found: {args.model}"

    model = YOLO(str(args.model))
    if args.dense:
        # Force full-softmax dense routing during training so the model's native
        # inference mode matches the export path (and avoids train/eval BN drift).
        from ultralytics.nn.modules.moe.routers import DynamicRoutingLayer
        import ultralytics.nn.modules.moe.modules as moe_mod
        esmoe_cls = getattr(moe_mod, "ES_MOE")
        nr = ne = 0
        for m in model.model.modules():
            if isinstance(m, DynamicRoutingLayer):
                m.use_top_k = False
                nr += 1
            if isinstance(m, esmoe_cls):
                m.use_sparse_inference = False
                ne += 1
        print(f"[train] dense mode: use_top_k=False on {nr} routers, dense combine on {ne} ES_MOE")
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(PROJ_ROOT.parent / "runs" / "train"),
        name=args.name,
        # VisDrone-specific augmentation bias: small objects benefit from
        # stronger scale/flip augmentation; mosaic/mixup help density.
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        patience=30,
        freeze=args.freeze,
        seed=42,
        verbose=True,
    )
    print(f"[train] done. results_dir={results.save_dir if hasattr(results, 'save_dir') else 'n/a'}")

    # Validate the best checkpoint and report mAP50-95 (the metric used for consistency checks).
    best = PROJ_ROOT.parent / "runs" / "train" / args.name / "weights" / "best.pt"
    if best.exists():
        print(f"[train] validating best checkpoint: {best}")
        best_model = YOLO(str(best))
        metrics = best_model.val(data=str(args.data), imgsz=args.imgsz, device=args.device)
        print(f"[train] mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}")
    else:
        print(f"[train] WARNING best.pt not found at {best}")


if __name__ == "__main__":
    main()
