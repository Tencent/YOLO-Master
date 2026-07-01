#!/usr/bin/env python3
"""Shared helpers for issue #49 reproduction scripts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_DATASETS_DIR = ROOT.parent / "data"

METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "val/moe_loss",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "train/moe_loss",
)

SUMMARY_FIELDS = (
    "dataset",
    "model",
    "model_name",
    "model_cfg",
    "data_yaml",
    "run_dir",
    "epoch",
    "epochs_requested",
    "imgsz",
    "batch",
    "seed",
    "wandb_url",
    *METRIC_KEYS,
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    folder_name: str
    data_yaml: Path
    default_project: Path


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_name: str
    cfg: Path


DATASETS = {
    "visdrone": DatasetSpec(
        key="visdrone",
        name="VisDrone",
        folder_name="VisDrone",
        data_yaml=ROOT / "ultralytics/cfg/datasets/VisDrone.yaml",
        default_project=ROOT / "runs/issue49_visdrone",
    ),
    "sku110k": DatasetSpec(
        key="sku110k",
        name="SKU-110K",
        folder_name="SKU-110K",
        data_yaml=ROOT / "ultralytics/cfg/datasets/SKU-110K.yaml",
        default_project=ROOT / "runs/issue49_sku110k",
    ),
}

MODEL_ORDER = ("esmoe_n", "v0_1_n")
MODELS = {
    "esmoe_n": ModelSpec(
        key="esmoe_n",
        model_name="YOLO-Master-EsMoE-N",
        cfg=ROOT / "ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml",
    ),
    "v0_1_n": ModelSpec(
        key="v0_1_n",
        model_name="YOLO-Master-v0.1-N",
        cfg=ROOT / "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml",
    ),
}

MODEL_ALIASES = {
    "all": "all",
    "esmoe": "esmoe_n",
    "esmoe-n": "esmoe_n",
    "esmoe_n": "esmoe_n",
    "yolo-master-esmoe-n": "esmoe_n",
    "v0.1": "v0_1_n",
    "v0_1": "v0_1_n",
    "v0.1-n": "v0_1_n",
    "v0_1_n": "v0_1_n",
    "yolo-master-v0.1-n": "v0_1_n",
}


def dataset_spec(key: str) -> DatasetSpec:
    try:
        return DATASETS[key]
    except KeyError as exc:
        raise ValueError(f"unknown dataset key: {key}") from exc


def normalize_model_key(value: str) -> str:
    key = value.strip().lower()
    normalized = MODEL_ALIASES.get(key)
    if normalized is None:
        valid = ", ".join(sorted(MODEL_ALIASES))
        raise ValueError(f"unknown model {value!r}; valid values: {valid}")
    return normalized


def discover_model_specs(values: Iterable[str]) -> list[ModelSpec]:
    selected: list[str] = []
    for value in values:
        key = normalize_model_key(value)
        if key == "all":
            for item in MODEL_ORDER:
                if item not in selected:
                    selected.append(item)
            continue
        if key not in selected:
            selected.append(key)
    return [MODELS[key] for key in selected]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run_name(spec: ModelSpec, run_suffix: str) -> str:
    suffix = run_suffix.strip("_")
    return f"{spec.key}_{suffix}" if suffix else spec.key


def read_last_metrics(results_csv: Path) -> dict[str, str]:
    if not results_csv.exists():
        return {}
    with results_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    return {key.strip(): value for key, value in rows[-1].items()}


def float_or_blank(value: str | None) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6g}"
    except ValueError:
        return value


def read_wandb_url(run_dir: Path) -> str:
    marker = run_dir / "wandb_url.txt"
    if not marker.exists():
        return ""
    return marker.read_text(encoding="utf-8").strip().splitlines()[0]


def collect_summary(
    dataset_name: str,
    data_yaml: Path,
    project: Path,
    specs: list[ModelSpec],
    run_suffix: str,
    epochs_requested: int | str = "",
    imgsz: int | str = "",
    batch: int | str = "",
    seed: int | str = "",
) -> list[dict[str, str]]:
    rows = []
    for spec in specs:
        directory = project / run_name(spec, run_suffix)
        metrics = read_last_metrics(directory / "results.csv")
        row = {
            "dataset": dataset_name,
            "model": spec.key,
            "model_name": spec.model_name,
            "model_cfg": relative_or_absolute(spec.cfg),
            "data_yaml": relative_or_absolute(data_yaml),
            "run_dir": relative_or_absolute(directory),
            "epoch": metrics.get("epoch", ""),
            "epochs_requested": str(epochs_requested),
            "imgsz": str(imgsz),
            "batch": str(batch),
            "seed": str(seed),
            "wandb_url": read_wandb_url(directory),
        }
        for key in METRIC_KEYS:
            row[key] = float_or_blank(metrics.get(key))
        rows.append(row)
    return rows


def markdown_cell(value: str) -> str:
    return str(value).replace("\n", "<br>").replace("|", "\\|")


def write_markdown_table(rows: list[dict[str, str]], output: Path) -> None:
    lines = ["# Issue #49 Result Summary", ""]
    lines.append("| " + " | ".join(SUMMARY_FIELDS) + " |")
    lines.append("| " + " | ".join(["---"] * len(SUMMARY_FIELDS)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(markdown_cell(row.get(field, "")) for field in SUMMARY_FIELDS) + " |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    dataset_name: str,
    data_yaml: Path,
    project: Path,
    specs: list[ModelSpec],
    run_suffix: str,
    epochs_requested: int | str = "",
    imgsz: int | str = "",
    batch: int | str = "",
    seed: int | str = "",
) -> Path:
    rows = collect_summary(dataset_name, data_yaml, project, specs, run_suffix, epochs_requested, imgsz, batch, seed)
    project.mkdir(parents=True, exist_ok=True)
    output = project / "summary.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown_table(rows, project / "summary.md")
    return output


def safe_write_summary(args: argparse.Namespace, spec: DatasetSpec, model_specs: list[ModelSpec], project: Path) -> None:
    try:
        output = write_summary(
            dataset_name=spec.name,
            data_yaml=spec.data_yaml,
            project=project,
            specs=model_specs,
            run_suffix=args.run_suffix,
            epochs_requested=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            seed=args.seed,
        )
        print(f"[summary] updated {relative_or_absolute(output)} and {relative_or_absolute(project / 'summary.md')}")
    except OSError as exc:
        print(f"[summary-warning] failed to write summary: {exc}")


def configure_datasets_dir(datasets_dir: Path) -> None:
    from ultralytics.utils import SETTINGS

    SETTINGS.update({"datasets_dir": str(datasets_dir)})


def validate_dataset_dir(spec: DatasetSpec, datasets_dir: Path) -> None:
    dataset_dir = datasets_dir / spec.folder_name
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"{spec.name} was not found at {dataset_dir}. "
            f"Pass --datasets-dir or download the dataset before training."
        )


def completed_epoch(run_dir: Path) -> int | None:
    metrics = read_last_metrics(run_dir / "results.csv")
    value = metrics.get("epoch")
    if value in {None, ""}:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def check_build(model_specs: list[ModelSpec]) -> None:
    from ultralytics.nn.tasks import DetectionModel

    for spec in model_specs:
        model = DetectionModel(str(spec.cfg), ch=3, nc=80, verbose=False)
        params = sum(param.numel() for param in model.parameters())
        print(f"[build-ok] {spec.key:<8} params={params / 1e6:.3f}M cfg={relative_or_absolute(spec.cfg)}")


def train_one(args: argparse.Namespace, spec: ModelSpec, dataset: DatasetSpec, project: Path) -> dict[str, str]:
    from ultralytics import YOLO

    start = time.time()
    directory = project / run_name(spec, args.run_suffix)
    target_epochs = args.epochs
    last_pt = directory / "weights/last.pt"
    resume = False

    if args.skip_existing and (directory / "results.csv").exists() and not args.resume_existing:
        print(f"[skip] {spec.key}: existing {relative_or_absolute(directory / 'results.csv')}")
        return {"model": spec.key, "status": "skipped", "duration_s": "0"}

    if args.resume_existing and last_pt.exists() and completed_epoch(directory) is not None:
        print(f"[resume] {spec.key}: {relative_or_absolute(last_pt)} -> target epochs={target_epochs}")
        model = YOLO(str(last_pt))
        resume = True
    else:
        if args.resume_existing:
            print(f"[resume-miss] {spec.key}: no completed checkpoint, starting from cfg")
        print(f"[train] {spec.key}: cfg={relative_or_absolute(spec.cfg)} data={relative_or_absolute(dataset.data_yaml)}")
        model = YOLO(str(spec.cfg))

    model.train(
        data=str(dataset.data_yaml),
        epochs=target_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        project=str(project),
        name=run_name(spec, args.run_suffix),
        exist_ok=args.exist_ok,
        pretrained=False,
        val=True,
        plots=args.plots,
        cache=args.cache,
        patience=args.patience,
        amp=args.amp,
        resume=resume,
        verbose=args.verbose,
    )
    duration = time.time() - start
    return {"model": spec.key, "status": "resumed" if resume else "ok", "duration_s": f"{duration:.2f}"}


def parse_args(dataset: DatasetSpec, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Reproduce issue #49 training on {dataset.name}.")
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--models", nargs="+", default=["all"], help="Models: all, esmoe_n, v0_1_n.")
    parser.add_argument("--project", type=Path, default=dataset.default_project)
    parser.add_argument("--run-suffix", default="", help="Suffix appended to run names, e.g. e100 or smoke.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), help="Set WANDB_MODE for this run.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-build", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def run_dataset(dataset_key: str, argv: list[str] | None = None) -> int:
    dataset = dataset_spec(dataset_key)
    args = parse_args(dataset, argv)
    try:
        model_specs = discover_model_specs(args.models)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    datasets_dir = args.datasets_dir.expanduser().resolve()
    project = resolve_path(args.project)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    print(f"[dataset] {dataset.name}")
    print(f"[data-yaml] {relative_or_absolute(dataset.data_yaml)}")
    print(f"[datasets-dir] {datasets_dir}")
    print(f"[project] {relative_or_absolute(project)}")
    for spec in model_specs:
        print(f"[model] {spec.key}: {relative_or_absolute(spec.cfg)} run={run_name(spec, args.run_suffix)}")

    if args.dry_run:
        return 0

    if args.check_build:
        configure_datasets_dir(datasets_dir)
        check_build(model_specs)
        write_summary(
            dataset.name,
            dataset.data_yaml,
            project,
            model_specs,
            args.run_suffix,
            args.epochs,
            args.imgsz,
            args.batch,
            args.seed,
        )
        return 0

    if args.summary_only:
        output = write_summary(
            dataset.name,
            dataset.data_yaml,
            project,
            model_specs,
            args.run_suffix,
            args.epochs,
            args.imgsz,
            args.batch,
            args.seed,
        )
        print(f"[summary] wrote {relative_or_absolute(output)} and {relative_or_absolute(project / 'summary.md')}")
        return 0

    validate_dataset_dir(dataset, datasets_dir)
    configure_datasets_dir(datasets_dir)

    statuses = []
    project.mkdir(parents=True, exist_ok=True)
    for model_spec in model_specs:
        try:
            statuses.append(train_one(args, model_spec, dataset, project))
        except Exception as exc:
            print(f"[fail] {model_spec.key}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            statuses.append({"model": model_spec.key, "status": "failed", "error": str(exc)})
            if args.stop_on_failure:
                break
        finally:
            safe_write_summary(args, dataset, model_specs, project)

    with (project / "status.json").open("w", encoding="utf-8") as handle:
        json.dump(statuses, handle, indent=2, ensure_ascii=False)
    success_states = {"ok", "skipped", "resumed"}
    return 0 if all(status.get("status") in success_states for status in statuses) else 1
