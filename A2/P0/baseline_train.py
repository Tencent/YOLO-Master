#!/usr/bin/env python3
"""Train the P0 VisDrone baseline (YOLO-Master v0.1-N from scratch) and log per-epoch positive-sample stats.

Registers two callbacks (without editing ``ultralytics/utils/callbacks/``, per project rules):

- ``on_train_epoch_end``: reads the criterion's ``fg_count`` / ``fg_count_by_stride`` and writes
  them into ``trainer.lr`` so they land in ``results.csv`` as ``train/fg_sum`` and
  ``train/fg_s8`` / ``train/fg_s16`` / ``train/fg_s32`` columns (FPN stride = area proxy).
- ``on_fit_epoch_end``: resets the counters. Validation reuses the training criterion (see
  ``engine/validator.py`` ``model.loss(...)``), so the count must be cleared *after* validate to
  avoid polluting the next epoch's total.

Usage:
    python A2/P0/baseline_train.py --epochs 120 --imgsz 800 --device 0              # full baseline (v0.1-N protocol)
    python A2/P0/baseline_train.py --epochs 1 --imgsz 320 --batch 8 --device 0      # short-cycle link check
    python A2/P0/baseline_train.py --resume --epochs 120 --imgsz 800 --device 0     # resume from weights/last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure this repo's ``ultralytics`` is imported (a stale editable install elsewhere would shadow it).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO
from ultralytics.utils.torch_utils import unwrap_model

DATA_YAML = REPO_ROOT / "datasets" / "VisDrone.yaml"
# YOLO-Master v0.1-N (MoE) — the reproduction-protocol baseline model (NOT stock yolo11n.yaml).
MODEL_YAML = REPO_ROOT / "ultralytics" / "cfg" / "models" / "master" / "v0_1" / "det" / "yolo-master-n.yaml"


def _criterion(trainer):
    """Return the native v8DetectionLoss (unwrapping a CompositeCriterion if present)."""
    model = unwrap_model(trainer.model)
    crit = getattr(model, "criterion", None)
    return getattr(crit, "native_criterion", crit) if crit is not None else None


def on_train_epoch_end(trainer) -> None:
    """Write per-epoch positive-sample counts into trainer.lr (survives into results.csv)."""
    crit = _criterion(trainer)
    if crit is None:
        return
    trainer.lr["train/fg_sum"] = int(crit.fg_count)
    for stride, count in sorted(crit.fg_count_by_stride.items()):
        trainer.lr[f"train/fg_s{int(stride)}"] = int(count)
    # STAL P0: per-area-tier coverage (COCO-style 32²/96²). avg_pos = pos/gt, zero_ratio = zero/gt.
    for tier in ("small", "medium", "large"):
        pos = crit.fg_tier_pos.get(tier, 0)
        n_gt = crit.fg_tier_gt.get(tier, 0)
        n_zero = crit.fg_tier_zero.get(tier, 0)
        trainer.lr[f"train/fg_{tier}"] = int(pos)
        trainer.lr[f"train/gt_{tier}"] = int(n_gt)
        trainer.lr[f"train/zero_{tier}"] = int(n_zero)
        trainer.lr[f"train/avg_pos_{tier}"] = float(pos / n_gt) if n_gt else 0.0
        trainer.lr[f"train/zero_ratio_{tier}"] = float(n_zero / n_gt) if n_gt else 0.0


def on_fit_epoch_end(trainer) -> None:
    """Reset counters after validation so the next epoch starts clean."""
    crit = _criterion(trainer)
    if crit is None:
        return
    crit.fg_count = 0
    crit.fg_count_by_stride = {}
    crit.fg_tier_pos = {}
    crit.fg_tier_gt = {}
    crit.fg_tier_zero = {}


def main() -> None:
    parser = argparse.ArgumentParser(description="P0 VisDrone baseline training")
    parser.add_argument("--model", default=str(MODEL_YAML), help="model config (from-scratch, no .pt)")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--amp",
        type=lambda s: str(s).lower() in {"1", "true", "yes"},
        default=None,
        help="override AMP (default: keep default.yaml amp=True for fresh runs; pass --amp false to resume in FP32)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0, help="early-stop patience (0 disables, fixed-epoch)")
    parser.add_argument("--project", default=str(REPO_ROOT / "runs" / "a2" / "p0"))
    parser.add_argument("--name", default="visdrone-baseline-v01n")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from weights/last.pt in project/name (pause/resume across sessions)",
    )
    # STAL overrides for the three-way comparison (none=fixed=adaptive). None = keep default.yaml's stal_mode=fixed.
    parser.add_argument("--stal-mode", default=None, help="STAL mode override: none|fixed|adaptive")
    parser.add_argument(
        "--stal-min-positive",
        type=lambda s: str(s).lower() in {"1", "true", "yes"},
        default=None,
        help="force >=1 positive anchor per recoverable GT (adaptive mode)",
    )
    parser.add_argument("--stal-topk-small", type=int, default=None, help="adaptive top-k for small GTs")
    parser.add_argument(
        "--stal-expand", type=float, default=None, help="candidate-region expansion radius (stride[0] units)"
    )
    parser.add_argument(
        "--tal-alpha", type=float, default=None, help="TAL metric classification exponent (score^alpha)"
    )
    parser.add_argument("--tal-beta", type=float, default=None, help="TAL metric localization exponent (iou^beta)")
    parser.add_argument(
        "--optimizer",
        default=None,
        help="optimizer override (e.g. MuSGD/AdamW). default=auto picks MuSGD only when iterations>10000 "
        "(~98 epochs at batch=6/nbs=64), so 30-epoch screening silently falls back to AdamW unless forced here.",
    )
    parser.add_argument(
        "--warmup-bias-lr",
        type=float,
        default=None,
        help="bias LR during warmup. default.yaml=0.1, but the auto-optimizer path forces 0.0; "
        "pass 0.0 to match a forced-MuSGD screening run to the auto-MuSGD champion.",
    )
    args = parser.parse_args()

    # Pause/resume: when --resume is set, reload the run's last.pt and continue from the saved epoch.
    # train() is given exist_ok=True so the run directory (checkpoints + results.csv) stays contiguous.
    run_dir = Path(args.project) / args.name
    last_pt = run_dir / "weights" / "last.pt"
    if args.resume:
        if not last_pt.exists():
            raise SystemExit(f"--resume requested but no checkpoint found at {last_pt}")
        model = YOLO(str(last_pt))
        print(f"Resuming from {last_pt}")
    else:
        model = YOLO(args.model)

    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    train_kwargs = {
        "data": str(DATA_YAML),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "seed": args.seed,
        "patience": args.patience,
        "workers": 0,
        "pretrained": False,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "resume": args.resume,
    }
    if args.amp is not None:
        train_kwargs["amp"] = args.amp
    for key, val in (
        ("stal_mode", args.stal_mode),
        ("stal_min_positive", args.stal_min_positive),
        ("stal_topk_small", args.stal_topk_small),
        ("stal_expand", args.stal_expand),
        ("tal_alpha", args.tal_alpha),
        ("tal_beta", args.tal_beta),
        ("optimizer", args.optimizer),
        ("warmup_bias_lr", args.warmup_bias_lr),
    ):
        if val is not None:
            train_kwargs[key] = val
    model.train(**train_kwargs)
    print(f"\nRun artifacts: {Path(args.project) / args.name}")


if __name__ == "__main__":
    main()
