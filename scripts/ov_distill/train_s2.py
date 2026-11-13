#!/usr/bin/env python3
"""S2 open-vocabulary pretraining: yoloe-26-master-n x val1k four-teacher voted pseudo-labels (grounding JSON).

Differences from smoke_train_yoloe_master.py:
  - targets the production data pool (default runs/ov_s2_val1k/dataset_grounding/grounding_train.json, 1000 images);
  - imgsz/batch/device/epochs are configurable;
  - ``--init-from`` supports weight relay (resume=True crashes with dict-form data; see ADR-0007 known issues):
    each chunk initializes from the previous chunk's best/last weights and continues training;
  - the validation set defaults to runs/ov_s2_val1k/val1k.yaml (val2017-subset GT, closed-set protocol).

Usage (chunked relay, each chunk <= 4.5 minutes):
  python scripts/ov_distill/train_s2.py --epochs 1 --imgsz 384 --device mps --name s2_e1
  python scripts/ov_distill/train_s2.py --epochs 1 --imgsz 384 --device mps --name s2_e2 \
      --init-from runs/ov_s2_train/s2_e1/weights/best.pt

Checkpoint insurance: ``--save-steps 50`` writes ``weights/last_step.pt`` every 50 steps;
when a chunk is interrupted by timeout, relay with ``--init-from <chunk_dir>/weights/last_step.pt``
so the loss shrinks from "the whole chunk" to "the last 50 steps".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLOE  # noqa: E402
from ultralytics.models.yolo.yoloe import YOLOETrainerFromScratch  # noqa: E402

RUN_DIR = ROOT / "runs/ov_s2_train"


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="S2 OV pretraining (chunked relay)")
    parser.add_argument("--json-file", default="runs/ov_s2_val1k/dataset_grounding/grounding_train.json")
    parser.add_argument("--img-root", default="datasets/coco/val2017", help="grounding image root directory")
    parser.add_argument("--val-yaml", default="runs/ov_s2_val1k/val1k.yaml")
    parser.add_argument("--model-cfg", default="ultralytics/cfg/models/26/yoloe-26-master-n.yaml")
    parser.add_argument("--init-from", default="", help="relay initialization weights (best.pt/last.pt); empty trains from scratch")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr0", type=float, default=0.002, help="initial learning rate (a constant-level value is recommended for chunked relay)")
    parser.add_argument("--optimizer", default="AdamW", help="must be set explicitly; auto would refit lr0 by nc and ignore this argument")
    parser.add_argument("--warmup-epochs", type=float, default=0.0, help="must be 0 for chunked relay, otherwise the whole chunk is stuck in low-LR warmup")
    parser.add_argument("--lrf", type=float, default=1.0, help="OneCycle/linear decay final factor; 1.0 = constant LR for chunked relay")
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--name", default="s2_e1")
    parser.add_argument("--no-val", action="store_true", help="skip end-of-chunk validation (saves time)")
    parser.add_argument(
        "--save-steps",
        type=int,
        default=0,
        help="additionally save weights/last_step.pt every N training steps (0 = disabled). "
        "When a chunk is interrupted by timeout, relay from it instead of losing the whole chunk "
        "(last/best.pt are only written at epoch end)",
    )
    args = parser.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    model_cfg = ROOT / args.model_cfg
    if args.init_from:
        # Weight relay: build the model from the config first, then load the previous chunk's
        # weights (optimizer state is not relayed; LR restarts per chunk)
        model = YOLOE(str(model_cfg))
        model.load(args.init_from)
        print(f"[s2] init from {args.init_from}")
    else:
        model = YOLOE(str(model_cfg))
    print(f"[s2] model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

    data = {
        "train": {
            "yolo_data": [],
            "grounding_data": [
                {
                    "img_path": str((ROOT / args.img_root).resolve()),
                    "json_file": str((ROOT / args.json_file).resolve()),
                }
            ],
        },
        "val": {"yolo_data": [str((ROOT / args.val_yaml).resolve())]},
    }

    if args.save_steps > 0:
        state = {"calls": 0}

        def _save_step_ckpt(trainer):
            """Serialize the current weights to last_step.pt every save_steps batches (a serialization failure never aborts training)."""
            state["calls"] += 1
            if state["calls"] % args.save_steps:
                return
            try:
                payload = trainer._serialize_checkpoint()
                step_path = trainer.wdir / "last_step.pt"
                step_path.write_bytes(payload)
            except Exception as exc:  # noqa: BLE001 - fail-safe logic; no exception may kill training
                print(f"[s2] warn: step checkpoint save failed: {exc}")

        model.add_callback("on_train_batch_end", _save_step_ckpt)
        print(f"[s2] step checkpoint enabled: every {args.save_steps} steps -> weights/last_step.pt")

    model.train(
        data=data,
        trainer=YOLOETrainerFromScratch,
        epochs=args.epochs,
        lr0=args.lr0,
        optimizer=args.optimizer,
        warmup_epochs=args.warmup_epochs,
        cos_lr=False,
        lrf=args.lrf,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=False,
        save=True,
        val=not args.no_val,
        project=str(RUN_DIR),
        name=args.name,
        exist_ok=True,
        verbose=False,
    )
    print(f"[s2] chunk {args.name} finished OK")


if __name__ == "__main__":
    main()
