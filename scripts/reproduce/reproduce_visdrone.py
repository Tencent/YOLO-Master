#!/usr/bin/env python3
"""Reproduce YOLO-Master-v0.1-N and YOLO-Master-EsMoE-N baselines on VisDrone.

Self-contained VisDrone reproduction entry (no dependency on ``_reproduce_common``).
Trains from YAML configs, writes per-epoch ``results.csv``, optional W&B logging
(mAP50, mAP50-95, box/cls/moe or mixture_aux loss), and ``summary.csv``.

EsMoE-N: default keeps sparse eval (as-shipped). Pass ``--no-sparse-eval`` for
corrected dense evaluation (train==eval).

Examples:
    python scripts/reproduce/reproduce_visdrone.py --check-build
    python scripts/reproduce/reproduce_visdrone.py --model v0.1-N --epochs 100 --batch 8
    python scripts/reproduce/reproduce_visdrone.py --model EsMoE-N --epochs 100 --batch 16 --no-sparse-eval
    python scripts/reproduce/reproduce_visdrone.py --epochs 100 --batch 8 --no-sparse-eval
"""

from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "train/moe_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "val/moe_loss",
)

_WANDB_METRICS = {
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
    "train/box_loss": "train/box_loss",
    "train/cls_loss": "train/cls_loss",
    "train/moe_loss": "train/moe_loss",
    "train/mixture_aux_loss": "train/mixture_aux_loss",
    "val/box_loss": "val/box_loss",
    "val/cls_loss": "val/cls_loss",
    "val/moe_loss": "val/moe_loss",
    "val/mixture_aux_loss": "val/mixture_aux_loss",
}


@dataclass(frozen=True)
class ModelSpec:
    """One model config to reproduce."""

    name: str
    cfg: str
    uses_esmoe: bool = False


MODELS = (
    ModelSpec("v0.1-N", "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml", uses_esmoe=False),
    ModelSpec("EsMoE-N", "ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml", uses_esmoe=True),
)

# Optional nano variants (same set as the shared reproduce suite).
VARIANTS = (
    ModelSpec("EsMoE-P2-N", "ultralytics/cfg/models/master/v0/det/yolo-master-n-p2.yaml", uses_esmoe=True),
    ModelSpec("v0.1-P2-N", "ultralytics/cfg/models/master/v0_1/det/yolo-master-n-p2.yaml", uses_esmoe=False),
    ModelSpec("UoMoE-N", "ultralytics/cfg/models/master/v0_1/det/yolo-master-n-uomoe.yaml", uses_esmoe=False),
    ModelSpec("UoMoE-P2-N", "ultralytics/cfg/models/master/v0_1/det/yolo-master-n-uomoe-p2.yaml", uses_esmoe=False),
)

ALL_MODELS = MODELS + VARIANTS

DATASET_NAME = "VisDrone"
DATASET_DATA = "VisDrone.yaml"
DEFAULT_PROJECT = "runs/reproduce/visdrone"


def _make_dense_inference_callback():
    """Flip ES_MOE.use_sparse_inference=False on model + EMA for dense val."""
    from ultralytics.nn.modules.moe.modules import ES_MOE
    from ultralytics.utils import LOGGER

    state = {"logged": False}

    def _apply(trainer):
        targets = []
        model = getattr(trainer, "model", None)
        if model is not None:
            targets.append(model)
        ema = getattr(trainer, "ema", None)
        if ema is not None and getattr(ema, "ema", None) is not None:
            targets.append(ema.ema)

        count = 0
        for target in targets:
            for module in target.modules():
                if isinstance(module, ES_MOE):
                    module.use_sparse_inference = False
                    count += 1
        if count and not state["logged"]:
            LOGGER.info(
                f"[reproduce] EsMoE dense validation enabled: "
                f"use_sparse_inference=False on {count} ES_MOE module(s)"
            )
            state["logged"] = True

    return _apply


def _make_wandb_callbacks(run_name: str, spec: ModelSpec, args: argparse.Namespace, dense_val: bool) -> dict:
    """Stream per-epoch metrics to Weights & Biases (independent of Ultralytics SETTINGS)."""
    from ultralytics.utils import LOGGER

    state = {"run": None}

    def on_train_start(trainer):
        try:
            import wandb
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(f"[reproduce] wandb unavailable ({exc}); continuing without it.")
            return
        try:
            state["run"] = wandb.init(
                project=args.wandb_project,
                entity=(args.wandb_entity or None),
                name=run_name,
                mode=args.wandb_mode,
                reinit=True,
                config={
                    "model": spec.name,
                    "cfg": spec.cfg,
                    "dataset": DATASET_NAME,
                    "data": DATASET_DATA,
                    "epochs": args.epochs,
                    "imgsz": args.imgsz,
                    "batch": args.batch,
                    "seed": args.seed,
                    "dense_val": dense_val,
                },
            )
            url = getattr(state["run"], "url", None)
            LOGGER.info(f"[reproduce] wandb run '{run_name}' [{args.wandb_mode}] -> {url}")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                f"[reproduce] wandb init failed ({exc}); continuing without wandb. "
                f"For a live URL run `wandb login` first, or use --wandb-mode offline."
            )
            state["run"] = None

    def on_fit_epoch_end(trainer):
        run = state["run"]
        if run is None:
            return
        data = {}
        try:
            data.update(trainer.label_loss_items(trainer.tloss, prefix="train"))
        except Exception:  # noqa: BLE001
            pass
        try:
            data.update(trainer.metrics or {})
        except Exception:  # noqa: BLE001
            pass
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        log = {"epoch": epoch}
        for out_key, src_key in _WANDB_METRICS.items():
            v = data.get(src_key)
            if v is not None:
                try:
                    log[out_key] = float(v)
                except (TypeError, ValueError):
                    pass
        try:
            run.log(log, step=epoch)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(f"[reproduce] wandb log failed at epoch {epoch}: {exc}")

    def on_train_end(trainer):
        run = state["run"]
        if run is not None:
            try:
                run.finish()
            except Exception:  # noqa: BLE001
                pass
            state["run"] = None

    return {
        "on_train_start": on_train_start,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }


def _read_last_metrics(results_csv: Path) -> dict[str, str]:
    if not results_csv.exists():
        return {}
    with results_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k.strip(): v for k, v in rows[-1].items()} if rows else {}


def _float_or_blank(value: str | None) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.6g}"
    except ValueError:
        return value


def _completed_epoch(run_dir: Path) -> int | None:
    val = _read_last_metrics(run_dir / "results.csv").get("epoch")
    try:
        return int(float(val)) if val not in (None, "") else None
    except ValueError:
        return None


def write_summary(project: Path, models: tuple[ModelSpec, ...], sparse_eval: bool) -> Path:
    """Write aggregated ``summary.csv`` under ``project``."""
    project.mkdir(parents=True, exist_ok=True)
    out = project / "summary.csv"
    fieldnames = ["dataset", "model", "cfg", "run_dir", "dense_eval", "epoch", *METRIC_KEYS]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for spec in models:
            run_dir = project / f"{DATASET_NAME}_{spec.name}"
            res = _read_last_metrics(run_dir / "results.csv")
            row = {
                "dataset": DATASET_NAME,
                "model": spec.name,
                "cfg": spec.cfg,
                "run_dir": str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
                "dense_eval": (spec.uses_esmoe and not sparse_eval) if spec.uses_esmoe else "n/a",
                "epoch": res.get("epoch", ""),
            }
            for k in METRIC_KEYS:
                row[k] = _float_or_blank(res.get(k))
            w.writerow(row)
    return out


def train_one(args: argparse.Namespace, spec: ModelSpec, project: Path) -> dict:
    """Train one VisDrone model variant."""
    from ultralytics import YOLO

    run_name = f"{DATASET_NAME}_{spec.name}"
    run_dir = project / run_name
    last_pt = run_dir / "weights" / "last.pt"
    best_pt = run_dir / "weights" / "best.pt"
    done = _completed_epoch(run_dir)

    if best_pt.exists() and done is not None and done + 1 >= args.epochs:
        print(f"[skip] {run_name}: complete at epoch {done}", flush=True)
        return {"model": spec.name, "status": "skipped"}

    dense_eval = spec.uses_esmoe and not args.sparse_eval
    if last_pt.exists() and done is not None:
        print(f"[resume] {run_name}: {last_pt} epoch={done} -> {args.epochs}", flush=True)
        model = YOLO(str(last_pt))
        resume = True
    else:
        print(
            f"[train] {run_name}: cfg={spec.cfg} data={DATASET_DATA} "
            f"sparse_eval={args.sparse_eval} dense_eval={dense_eval}",
            flush=True,
        )
        model = YOLO(str(ROOT / spec.cfg))
        resume = False

    if dense_eval:
        cb = _make_dense_inference_callback()
        model.add_callback("on_pretrain_routine_end", cb)
        model.add_callback("on_train_start", cb)

    if args.wandb and args.wandb_mode != "disabled":
        for event, fn in _make_wandb_callbacks(run_name, spec, args, dense_eval).items():
            model.add_callback(event, fn)

    start = time.time()
    model.train(
        data=DATASET_DATA,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        project=str(project),
        name=run_name,
        exist_ok=True,
        pretrained=False,
        lora_r=0,  # disable default LoRA so this is a full from-scratch baseline
        optimizer="auto",  # SGD@0.01 for long runs (avoid AdamW@0.01 instability)
        val=True,
        plots=True,
        cache=args.cache,
        patience=args.patience,
        amp=args.amp,
        resume=resume,
        verbose=args.verbose,
    )
    return {
        "model": spec.name,
        "status": "resumed" if resume else "ok",
        "duration_s": f"{time.time() - start:.1f}",
    }


def build_parser(models: tuple[ModelSpec, ...] = ALL_MODELS) -> argparse.ArgumentParser:
    """CLI for VisDrone reproduction."""
    p = argparse.ArgumentParser(
        description="Reproduce YOLO-Master nano baselines on VisDrone.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=300, help="Recommended 100~300 (adjust to GPU budget).")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default="0")
    p.add_argument(
        "--workers",
        type=int,
        default=0 if platform.system() == "Windows" else 16,
        help="DataLoader workers (0 on Windows by default).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=0, help="0 disables early stopping.")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--cache",
        nargs="?",
        const="ram",
        default=False,
        help="Cache images: --cache / --cache ram / --cache disk; omit to disable.",
    )
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument(
        "--model",
        choices=[m.name for m in models] + ["both", "v01", "moe"],
        default="both",
        help="Which model(s) to train. Default 'both' = v0.1-N + EsMoE-N.",
    )
    p.add_argument(
        "--include-variants",
        action="store_true",
        help="Also include P2/UoMoE nano variants when --model both.",
    )
    p.add_argument(
        "--baseline-only",
        action="store_true",
        help=argparse.SUPPRESS,  # kept for backward compatibility; default is already baselines-only
    )
    p.add_argument(
        "--sparse-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ES_MOE sparse inference at val. Pass --no-sparse-eval for corrected dense eval.",
    )
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb-project", default="yolo-master-reproduce")
    p.add_argument("--wandb-entity", default="", help="W&B entity/team (optional).")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--check-build", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--stop-on-failure", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    """Entry point."""
    # Prefer keeping Ultralytics built-in wandb off: it uses local `project` path as
    # W&B project name on Windows and raises UsageError. This script has its own logger.
    try:
        from ultralytics.utils import SETTINGS

        # Avoid Ultralytics wb callback using local save path as W&B project name (Windows).
        SETTINGS["wandb"] = False
    except Exception:  # noqa: BLE001
        pass

    args = build_parser().parse_args()
    # Default: task baselines only (v0.1-N + EsMoE-N). Opt into variants with --include-variants.
    catalog = ALL_MODELS if args.include_variants else MODELS
    project = Path(args.project) if Path(args.project).is_absolute() else ROOT / args.project
    aliases = {"v01": "v0.1-N", "moe": "EsMoE-N"}
    selected = aliases.get(args.model, args.model)
    if selected == "both":
        specs = list(catalog)
    else:
        specs = [m for m in ALL_MODELS if m.name == selected]
        if not specs:
            print(f"[error] unknown model: {args.model}", flush=True)
            return 2

    wandb_desc = "off" if (not args.wandb or args.wandb_mode == "disabled") else args.wandb_mode
    print(
        f"[reproduce:{DATASET_NAME}] data={DATASET_DATA}  project={project}  "
        f"sparse_eval={args.sparse_eval}  wandb={wandb_desc}"
    )
    for s in specs:
        dense = s.uses_esmoe and not args.sparse_eval
        note = f"dense_eval={dense}" if s.uses_esmoe else "no ES_MOE (sparse-eval n/a)"
        print(f"  - {s.name:<8} cfg={s.cfg}  {note}")

    if args.dry_run:
        return 0
    if args.check_build:
        from ultralytics.nn.tasks import DetectionModel

        for s in specs:
            m = DetectionModel(str(ROOT / s.cfg), ch=3, nc=80, verbose=False)
            print(f"[build-ok] {s.name}: {sum(p.numel() for p in m.parameters()) / 1e6:.3f}M  ({s.cfg})")
        return 0
    if args.summary_only:
        print("[summary]", write_summary(project, tuple(specs), sparse_eval=args.sparse_eval))
        return 0

    project.mkdir(parents=True, exist_ok=True)
    statuses = []
    for s in specs:
        try:
            statuses.append(train_one(args, s, project))
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {s.name}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            statuses.append({"model": s.name, "status": "failed", "error": str(exc)})
            if args.stop_on_failure:
                break
        finally:
            try:
                write_summary(project, tuple(specs), sparse_eval=args.sparse_eval)
            except OSError as e:
                print(f"[summary-warn] {e}", flush=True)

    print(f"\n[reproduce:{DATASET_NAME}] DONE")
    for st in statuses:
        print("  ", st)
    ok = {"ok", "resumed", "skipped"}
    return 0 if all(st.get("status") in ok for st in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
