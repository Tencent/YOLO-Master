#!/usr/bin/env python3
"""Shared helpers for dense-dataset YOLO-Master reproduction scripts."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

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
    uses_esmoe: bool


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    data: str
    project: str


MODELS = (
    ModelSpec("v0.1-N", "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml", False),
    ModelSpec("EsMoE-N", "ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml", True),
)


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


def _completed_epoch(run_dir: Path) -> int | None:
    value = _read_last_metrics(run_dir / "results.csv").get("epoch")
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _make_dense_eval_callback():
    from ultralytics.nn.modules.moe.modules import ES_MOE
    from ultralytics.utils import LOGGER

    state = {"logged": False}

    def apply_dense_eval(trainer):
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
            LOGGER.info(f"[reproduce] ES_MOE dense eval enabled on {count} module(s)")
            state["logged"] = True

    return apply_dense_eval


def _make_wandb_callbacks(run_name: str, dataset: DatasetSpec, spec: ModelSpec, args: argparse.Namespace, dense_eval: bool):
    from ultralytics.utils import LOGGER

    state = {"run": None}
    metric_map = {
        "mAP50": "metrics/mAP50(B)",
        "mAP50-95": "metrics/mAP50-95(B)",
        "train/box_loss": "train/box_loss",
        "train/cls_loss": "train/cls_loss",
        "train/moe_loss": "train/moe_loss",
        "val/box_loss": "val/box_loss",
        "val/cls_loss": "val/cls_loss",
        "val/moe_loss": "val/moe_loss",
    }

    def on_train_start(trainer):
        try:
            import wandb

            state["run"] = wandb.init(
                project=args.wandb_project,
                entity=(args.wandb_entity or None),
                name=run_name,
                mode=args.wandb_mode,
                reinit=True,
                config={
                    "dataset": dataset.name,
                    "data": dataset.data,
                    "model": spec.name,
                    "cfg": spec.cfg,
                    "dense_eval": dense_eval,
                    "epochs": args.epochs,
                    "imgsz": args.imgsz,
                    "batch": args.batch,
                    "seed": args.seed,
                },
            )
            LOGGER.info(f"[reproduce] wandb run '{run_name}' -> {getattr(state['run'], 'url', None)}")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(f"[reproduce] wandb disabled: {exc}")
            state["run"] = None

    def on_fit_epoch_end(trainer):
        run = state["run"]
        if run is None:
            return
        values = {}
        try:
            values.update(trainer.label_loss_items(trainer.tloss, prefix="train"))
        except Exception:
            pass
        try:
            values.update(trainer.metrics or {})
        except Exception:
            pass

        epoch = int(getattr(trainer, "epoch", 0)) + 1
        payload = {"epoch": epoch}
        for out_key, src_key in metric_map.items():
            value = values.get(src_key)
            if value is not None:
                try:
                    payload[out_key] = float(value)
                except (TypeError, ValueError):
                    pass
        run.log(payload, step=epoch)

    def on_train_end(trainer):
        run = state["run"]
        if run is not None:
            run.finish()
            state["run"] = None

    return {
        "on_train_start": on_train_start,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }


def write_summary(project: Path, dataset: DatasetSpec, specs: list[ModelSpec], sparse_eval: bool) -> Path:
    project.mkdir(parents=True, exist_ok=True)
    out = project / "summary.csv"
    fieldnames = ["dataset", "model", "cfg", "run_dir", "dense_eval", "epoch", *METRIC_KEYS]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for spec in specs:
            run_dir = project / f"{dataset.name}_{spec.name}"
            results = _read_last_metrics(run_dir / "results.csv")
            row = {
                "dataset": dataset.name,
                "model": spec.name,
                "cfg": spec.cfg,
                "run_dir": str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
                "dense_eval": (spec.uses_esmoe and not sparse_eval) if spec.uses_esmoe else "n/a",
                "epoch": results.get("epoch", ""),
            }
            for key in METRIC_KEYS:
                row[key] = _float_or_blank(results.get(key))
            writer.writerow(row)
    return out


def train_one(args: argparse.Namespace, dataset: DatasetSpec, spec: ModelSpec, project: Path) -> dict[str, str]:
    from ultralytics import YOLO

    run_name = f"{dataset.name}_{spec.name}"
    run_dir = project / run_name
    done = _completed_epoch(run_dir)
    last_pt = run_dir / "weights" / "last.pt"

    if done is not None and done + 1 >= args.epochs:
        print(f"[skip] {run_name}: already completed epoch={done}")
        return {"model": spec.name, "status": "skipped"}

    dense_eval = spec.uses_esmoe and not args.sparse_eval
    if last_pt.exists() and done is not None:
        print(f"[resume] {run_name}: epoch={done} -> {args.epochs}")
        model = YOLO(str(last_pt))
        resume = True
    else:
        print(f"[train] {run_name}: cfg={spec.cfg} data={dataset.data} dense_eval={dense_eval}")
        model = YOLO(str(ROOT / spec.cfg))
        resume = False

    if dense_eval:
        callback = _make_dense_eval_callback()
        model.add_callback("on_pretrain_routine_end", callback)
        model.add_callback("on_train_start", callback)

    if args.wandb and args.wandb_mode != "disabled":
        for event, fn in _make_wandb_callbacks(run_name, dataset, spec, args, dense_eval).items():
            model.add_callback(event, fn)

    start = time.time()
    model.train(
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
        plots=args.plots,
        cache=args.cache,
        patience=args.patience,
        amp=args.amp,
        resume=resume,
        verbose=args.verbose,
    )
    return {"model": spec.name, "status": "resumed" if resume else "ok", "duration_s": f"{time.time() - start:.1f}"}


def build_parser(dataset: DatasetSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Reproduce YOLO-Master baselines on {dataset.name}.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--project", default=dataset.project)
    parser.add_argument("--model", choices=["v0.1-N", "EsMoE-N", "both"], default="both")
    parser.add_argument("--sparse-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="yolo-master-reproduce")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--check-build", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def run_dataset(dataset: DatasetSpec) -> int:
    args = build_parser(dataset).parse_args()
    project = Path(args.project)
    if not project.is_absolute():
        project = ROOT / project

    specs = list(MODELS) if args.model == "both" else [m for m in MODELS if m.name == args.model]

    print(f"[reproduce:{dataset.name}] data={dataset.data} project={project} sparse_eval={args.sparse_eval}")
    for spec in specs:
        dense_eval = spec.uses_esmoe and not args.sparse_eval
        print(f"  - {spec.name}: cfg={spec.cfg} dense_eval={dense_eval}")

    if args.check_build:
        from ultralytics.nn.tasks import DetectionModel

        for spec in specs:
            model = DetectionModel(str(ROOT / spec.cfg), ch=3, nc=80, verbose=False)
            print(f"[build-ok] {spec.name}: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")
        return 0

    if args.summary_only:
        print("[summary]", write_summary(project, dataset, specs, args.sparse_eval))
        return 0

    project.mkdir(parents=True, exist_ok=True)
    statuses = []
    for spec in specs:
        try:
            statuses.append(train_one(args, dataset, spec, project))
        finally:
            write_summary(project, dataset, specs, args.sparse_eval)

    print(f"[reproduce:{dataset.name}] DONE")
    for status in statuses:
        print(" ", status)
    return 0

