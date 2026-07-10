#!/usr/bin/env python3
"""Shared logic for the per-dataset YOLO-Master baseline reproduction scripts.

Both reproduce_visdrone.py and reproduce_sku110k.py train the two nano release
variants from their YAML configs (from scratch) and log per-epoch metrics
(mAP50, mAP50-95, box/cls/dfl/moe_loss) to each run's results.csv, plus an
aggregated summary.csv.

Models
------
  - YOLO-Master-v0.1-N  -> ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml
      (MoE block: OptimizedMOEImproved -- train/eval-consistent, always-on shared
       expert; no sparse-inference issue.)
  - YOLO-Master-EsMoE-N -> ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml
      (MoE block: ES_MOE. The shipped eval path uses sparse inference, while
       --no-sparse-eval enables train/eval-consistent dense validation.)

Sparse vs dense evaluation (EsMoE-N only)
-----------------------------------------
By default the scripts reproduce the model exactly as shipped: ES_MOE keeps
`use_sparse_inference=True`, so validation uses its sparse routing path.

Pass ``--no-sparse-eval`` for the train/eval-consistent reproduction path. It is
an explicit flag so the evaluation mode is recorded in the invocation. It
registers a training callback that flips
`ES_MOE.use_sparse_inference=False` on both the live model and its EMA at
`on_pretrain_routine_end` (before validation and before checkpoints are written
from the EMA), so per-epoch validation, saved checkpoints, and final evaluation
use the same dense forward as training. v0.1-N has no ES_MOE modules, so the
flag is a no-op there.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from contextlib import contextmanager
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


@dataclass(frozen=True)
class ModelSpec:
    name: str
    cfg: str
    uses_esmoe: bool = False  # True if the model contains ES_MOE blocks (sparse-eval sensitive)


@dataclass(frozen=True)
class DatasetSpec:
    name: str          # short tag, e.g. "VisDrone"
    data: str          # dataset yaml, e.g. "VisDrone.yaml"
    project: str       # e.g. "runs/reproduce/visdrone"


# Both datasets train the same two models.
MODELS = (
    ModelSpec("v0.1-N", "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml", uses_esmoe=False),
    ModelSpec("EsMoE-N", "ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml", uses_esmoe=True),
)


# --------------------------------------------------------------------------- #
# Dense-validation callback for ES_MOE                                         #
# --------------------------------------------------------------------------- #
def _make_dense_inference_callback():
    """Return a trainer callback that sets ES_MOE.use_sparse_inference=False.

    Applied to both trainer.model and trainer.ema.ema so per-epoch validation
    (which runs on the EMA), the EMA-derived checkpoints, and the final eval all
    take the dense forward path that matches training.
    """
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
            LOGGER.info(f"[reproduce] EsMoE dense validation enabled: "
                        f"use_sparse_inference=False on {count} ES_MOE module(s)")
            state["logged"] = True

    return _apply


# --------------------------------------------------------------------------- #
# Real-time W&B per-epoch logging                                              #
# --------------------------------------------------------------------------- #
# Metrics logged every epoch: mAP50, mAP50-95, box_loss, cls_loss, moe_loss
# (train + val variants where available). One W&B run per (dataset, model).
_WANDB_METRICS = {
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
    "train/box_loss": "train/box_loss",
    "train/cls_loss": "train/cls_loss",
    "train/moe_loss": "train/moe_loss",
    "val/box_loss": "val/box_loss",
    "val/cls_loss": "val/cls_loss",
    "val/moe_loss": "val/moe_loss",
}


def _make_wandb_callbacks() -> dict:
    """Return trainer callbacks that stream per-epoch metrics to Weights & Biases.

    If wandb is unavailable or initialization fails, training continues after
    emitting a warning.
    """
    from ultralytics.utils import LOGGER

    state = {"run": None}

    def on_train_start(trainer):
        from ultralytics.utils import RANK

        if RANK not in {-1, 0}:
            return
        try:
            import wandb
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(f"[reproduce] wandb unavailable ({exc}); continuing without it.")
            return
        try:
            state["run"] = wandb.init(
                project=os.getenv("WANDB_PROJECT"),
                entity=(os.getenv("WANDB_ENTITY") or None),
                name=os.getenv("WANDB_NAME"),
                mode=os.getenv("WANDB_MODE", "online"),
                reinit=True,
                config={
                    "model": Path(str(trainer.args.model)).stem,
                    "data": trainer.args.data,
                    "epochs": trainer.args.epochs,
                    "imgsz": trainer.args.imgsz,
                    "batch": trainer.args.batch,
                    "seed": trainer.args.seed,
                    "eval": "nosparse" if os.getenv("YOLO_MASTER_REPRO_DENSE_EVAL") == "1" else "default",
                },
            )
            url = getattr(state["run"], "url", None)
            LOGGER.info(
                f"[reproduce] wandb run '{os.getenv('WANDB_NAME')}' "
                f"[{os.getenv('WANDB_MODE', 'online')}] -> {url}"
            )
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

    return {"on_train_start": on_train_start,
            "on_fit_epoch_end": on_fit_epoch_end,
            "on_train_end": on_train_end}


# --------------------------------------------------------------------------- #
# Summary CSV                                                                  #
# --------------------------------------------------------------------------- #
def _read_last_metrics(results_csv: Path) -> dict[str, str]:
    if not results_csv.exists():
        return {}
    with results_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {k.strip(): v for k, v in rows[-1].items()} if rows else {}


def _float_or_blank(value: str | None) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.6g}"
    except ValueError:
        return value


def _eval_mode(spec: ModelSpec, sparse_eval: bool) -> str:
    return "nosparse" if spec.uses_esmoe and not sparse_eval else "default"


def _run_name(dataset: DatasetSpec, spec: ModelSpec, sparse_eval: bool) -> str:
    name = f"{dataset.name}_{spec.name}"
    return f"{name}_{_eval_mode(spec, sparse_eval)}" if spec.uses_esmoe else name


def write_summary(project: Path, dataset: DatasetSpec) -> Path:
    project.mkdir(parents=True, exist_ok=True)
    out = project / "summary.csv"
    fieldnames = ["dataset", "model", "cfg", "eval", "run_dir", "epoch", *METRIC_KEYS]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for spec in MODELS:
            eval_modes = (True, False) if spec.uses_esmoe else (True,)
            for sparse_eval in eval_modes:
                run_dir = project / _run_name(dataset, spec, sparse_eval)
                res = _read_last_metrics(run_dir / "results.csv")
                if not res:
                    continue
                row = {
                    "dataset": dataset.name,
                    "model": spec.name,
                    "cfg": spec.cfg,
                    "eval": _eval_mode(spec, sparse_eval),
                    "run_dir": str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
                    "epoch": res.get("epoch", ""),
                }
                for key in METRIC_KEYS:
                    row[key] = _float_or_blank(res.get(key))
                w.writerow(row)
    return out


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def _completed_epoch(run_dir: Path) -> int | None:
    val = _read_last_metrics(run_dir / "results.csv").get("epoch")
    try:
        return int(float(val)) if val not in (None, "") else None
    except ValueError:
        return None


@contextmanager
def _reproduction_environment(args: argparse.Namespace, run_name: str, dense_eval: bool):
    """Expose script-only settings to Ultralytics DDP worker processes."""
    script_dir = str(Path(__file__).resolve().parent)
    pythonpath = os.environ.get("PYTHONPATH", "")
    updates = {
        "PYTHONPATH": os.pathsep.join(filter(None, (script_dir, pythonpath))),
        "YOLO_MASTER_REPRO_DENSE_EVAL": "1" if dense_eval else "0",
        "YOLO_MASTER_REPRO_WANDB": "1" if args.wandb and args.wandb_mode != "disabled" else "0",
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_MODE": args.wandb_mode,
        "WANDB_NAME": run_name,
    }
    if args.wandb_entity:
        updates["WANDB_ENTITY"] = args.wandb_entity
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def train_one(args: argparse.Namespace, dataset: DatasetSpec, spec: ModelSpec, project: Path) -> dict:
    from ultralytics import YOLO
    from _reproduce_trainer import ReproductionTrainer

    run_name = _run_name(dataset, spec, args.sparse_eval)
    run_dir = project / run_name
    last_pt = run_dir / "weights" / "last.pt"
    best_pt = run_dir / "weights" / "best.pt"
    done = _completed_epoch(run_dir)

    if best_pt.exists() and done is not None and done >= args.epochs:
        print(f"[skip] {run_name}: complete at epoch {done}", flush=True)
        return {"model": spec.name, "status": "skipped"}

    # Shipped sparse evaluation is the default for ES_MOE. Dense validation is
    # opt-in via --no-sparse-eval and is a no-op for v0.1-N.
    dense_eval = spec.uses_esmoe and not args.sparse_eval
    if last_pt.exists() and done is not None:
        print(f"[resume] {run_name}: {last_pt} epoch={done} -> {args.epochs}", flush=True)
        model = YOLO(str(last_pt))
        resume = True
    else:
        print(f"[train] {run_name}: cfg={spec.cfg} data={dataset.data} "
              f"sparse_eval={args.sparse_eval} dense_eval={dense_eval}", flush=True)
        model = YOLO(str(ROOT / spec.cfg))
        resume = False

    start = time.time()
    with _reproduction_environment(args, run_name, dense_eval):
        model.train(
            trainer=ReproductionTrainer,
            data=dataset.data,
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
            val=True,
            plots=True,
            cache=args.cache,
            patience=args.patience,
            amp=args.amp,
            resume=resume,
            lora_r=0,
            verbose=args.verbose,
        )
    return {"model": spec.name, "status": "resumed" if resume else "ok",
            "duration_s": f"{time.time() - start:.1f}"}


def build_parser(dataset: DatasetSpec) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"Reproduce YOLO-Master v0.1-N and EsMoE-N baselines on {dataset.name}.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=300, help="Recommended ~300 (adjust to GPU budget).")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=0, help="0 disables early stopping.")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cache", action="store_true")
    p.add_argument("--project", default=dataset.project)
    p.add_argument("--model", choices=[m.name for m in MODELS] + ["both"], default="both",
                   help="Which model to train: v0.1-N, EsMoE-N, or both (default).")
    p.add_argument("--sparse-eval", action=argparse.BooleanOptionalAction, default=True,
                   help="ES_MOE sparse inference at validation/inference. Default True "
                        "reproduces the shipped model path. Pass --no-sparse-eval for the "
                        "train/eval-consistent dense validation workaround. No-op for v0.1-N.")
    # --- Weights & Biases real-time per-epoch logging ---
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True,
                   help="Stream mAP50/mAP50-95/box/cls/moe loss to W&B each epoch (default on). Use --no-wandb to disable.")
    p.add_argument("--wandb-project", default="yolo-master-reproduce", help="W&B project name.")
    p.add_argument("--wandb-entity", default="", help="W&B entity/team (optional).")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online",
                   help="online needs `wandb login`; offline logs locally (sync later); disabled turns it off.")
    p.add_argument("--check-build", action="store_true", help="Instantiate both models and exit.")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    p.add_argument("--summary-only", action="store_true", help="Only (re)write summary.csv from existing runs.")
    p.add_argument("--download-only", action="store_true",
                   help="Download/validate the dataset and exit.")
    p.add_argument("--stop-on-failure", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def run_dataset(dataset: DatasetSpec) -> int:
    """Entry point used by the per-dataset scripts."""
    args = build_parser(dataset).parse_args()
    project = Path(args.project) if Path(args.project).is_absolute() else ROOT / args.project
    specs = list(MODELS) if args.model == "both" else [m for m in MODELS if m.name == args.model]

    wandb_desc = "off" if (not args.wandb or args.wandb_mode == "disabled") else args.wandb_mode
    print(f"[reproduce:{dataset.name}] data={dataset.data}  project={project}  "
          f"sparse_eval={args.sparse_eval}  wandb={wandb_desc}")
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
        print("[summary]", write_summary(project, dataset))
        return 0
    if args.download_only:
        from ultralytics.data.utils import check_det_dataset
        data = check_det_dataset(dataset.data, autodownload=True)
        print(f"[dataset-ok] {dataset.name}: {data.get('path', dataset.data)}")
        return 0

    project.mkdir(parents=True, exist_ok=True)
    statuses = []
    for s in specs:
        try:
            statuses.append(train_one(args, dataset, s, project))
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {s.name}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            statuses.append({"model": s.name, "status": "failed", "error": str(exc)})
            if args.stop_on_failure:
                break
        finally:
            try:
                write_summary(project, dataset)
            except OSError as e:
                print(f"[summary-warn] {e}", flush=True)

    print(f"\n[reproduce:{dataset.name}] DONE")
    for st in statuses:
        print("  ", st)
    ok = {"ok", "resumed", "skipped"}
    return 0 if all(st.get("status") in ok for st in statuses) else 1
