#!/usr/bin/env python3
"""Shared helpers for Rhino-Bird issue #49 reproduction scripts."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = Path(os.environ.get("YOLO_MASTER_REPRO_RUNTIME_DIR", ROOT / "runs/reproduce/_runtime"))
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME_DIR / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_DIR / "matplotlib"))
sys.path.insert(0, str(ROOT))

from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.utils import SETTINGS  # noqa: E402


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
    key: str
    label: str
    cfg: Path

    @property
    def run_suffix(self) -> str:
        return self.key.replace("_", "-")


MODEL_SPECS = {
    "v01": ModelSpec(
        key="v01",
        label="YOLO-Master-v0.1-N",
        cfg=ROOT / "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml",
    ),
    "esmoe": ModelSpec(
        key="esmoe",
        label="YOLO-Master-EsMoE-N",
        cfg=ROOT / "ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml",
    ),
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def add_common_args(parser: argparse.ArgumentParser, *, dataset_name: str, default_data: Path, default_root: Path) -> None:
    parser.add_argument("--model", choices=("all", "v01", "esmoe"), default="all", help="Model(s) to run.")
    parser.add_argument("--data", type=Path, default=default_data, help="Base dataset YAML.")
    parser.add_argument("--dataset-root", type=Path, default=default_root, help="Absolute or repo-relative dataset root.")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "reproduce" / dataset_name.lower())
    parser.add_argument("--name-prefix", default=f"{dataset_name.lower()}_", help="Run directory prefix.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="Ultralytics device string, e.g. 0 or 0,1.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from each run's weights/last.pt if present.")
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--download-dataset", action="store_true", help="Allow Ultralytics to download/convert missing data.")
    parser.add_argument("--prepare-data-only", action="store_true", help="Prepare/check dataset files and exit without training.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings without training.")
    parser.add_argument("--check-build", action="store_true", help="Instantiate selected models and exit.")
    parser.add_argument("--summary-only", action="store_true", help="Only aggregate existing results.csv files.")
    parser.add_argument("--wandb", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--wandb-project", default="yolo-master-issue-49", help="W&B project name.")
    parser.add_argument("--wandb-entity", help="Optional W&B entity/team.")
    parser.add_argument("--wandb-group", help="Optional W&B group. Defaults to the dataset name.")
    parser.add_argument("--wandb-tags", default="", help="Comma-separated W&B tags.")
    parser.add_argument("--init-v01", type=Path, help="Optional pretrained/fine-tune weight for YOLO-Master-v0.1-N.")
    parser.add_argument("--init-esmoe", type=Path, help="Optional pretrained/fine-tune weight for YOLO-Master-EsMoE-N.")


def selected_specs(model_arg: str) -> list[ModelSpec]:
    if model_arg == "all":
        return [MODEL_SPECS["v01"], MODEL_SPECS["esmoe"]]
    return [MODEL_SPECS[model_arg]]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def configure_wandb(mode: str) -> None:
    if mode == "disabled":
        SETTINGS.update({"wandb": False})
        os.environ.setdefault("WANDB_DISABLED", "true")
        return
    SETTINGS.update({"wandb": True})
    os.environ.pop("WANDB_DISABLED", None)
    os.environ["WANDB_MODE"] = "offline" if mode == "offline" else "online"


def prepared_data_yaml(base_yaml: Path, dataset_root: Path, project: Path) -> Path:
    base_yaml = resolve(base_yaml)
    dataset_root = resolve(dataset_root)
    out_dir = project / "_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{base_yaml.stem}.local.yaml"

    lines = base_yaml.read_text(encoding="utf-8").splitlines()
    rewritten = []
    replaced = False
    for line in lines:
        if line.startswith("path:"):
            rewritten.append(f"path: {dataset_root.as_posix()}")
            replaced = True
        else:
            rewritten.append(line)
    if not replaced:
        rewritten.insert(0, f"path: {dataset_root.as_posix()}")
    out.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return out


def dataset_ready(dataset_root: Path, dataset_name: str) -> bool:
    dataset_root = resolve(dataset_root)
    if dataset_name.lower() == "visdrone":
        required = [
            dataset_root / "images/train",
            dataset_root / "images/val",
            dataset_root / "labels/train",
            dataset_root / "labels/val",
        ]
    elif dataset_name.lower() == "sku-110k":
        required = [
            dataset_root / "train.txt",
            dataset_root / "val.txt",
            dataset_root / "images",
            dataset_root / "labels",
        ]
    else:
        required = [dataset_root]
    return all(p.exists() for p in required)


def read_result_rows(results_csv: Path) -> list[dict[str, str]]:
    if not results_csv.exists():
        return []
    with results_csv.open(newline="", encoding="utf-8") as f:
        return [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]


def metric_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def best_row(rows: list[dict[str, str]], key: str) -> dict[str, str]:
    if not rows:
        return {}
    return max(rows, key=lambda row: metric_float(row, key))


def infer_spec_from_run(run_dir: Path, specs: list[ModelSpec]) -> ModelSpec | None:
    name = run_dir.name.lower().replace("-", "_")
    for spec in specs:
        suffix = spec.run_suffix.lower().replace("-", "_")
        if name.endswith(suffix) or f"_{suffix}" in name:
            return spec
    return None


def summary_rows(project: Path, specs: list[ModelSpec], name_prefix: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[Path] = set()
    seen_specs: set[str] = set()

    for results_csv in sorted(project.glob("*/results.csv")):
        run_dir = results_csv.parent
        spec = infer_spec_from_run(run_dir, specs)
        if spec is None:
            continue
        result_rows = read_result_rows(results_csv)
        metrics = result_rows[-1] if result_rows else {}
        best_map50 = best_row(result_rows, "metrics/mAP50(B)")
        best_map = best_row(result_rows, "metrics/mAP50-95(B)")
        row = {
            "model": spec.label,
            "cfg": rel(spec.cfg),
            "run_dir": rel(run_dir),
            "last_epoch": metrics.get("epoch", ""),
            "best_mAP50_epoch": best_map50.get("epoch", ""),
            "best_mAP50": best_map50.get("metrics/mAP50(B)", ""),
            "best_mAP50-95_at_best_mAP50": best_map50.get("metrics/mAP50-95(B)", ""),
            "best_mAP50-95_epoch": best_map.get("epoch", ""),
            "best_mAP50-95": best_map.get("metrics/mAP50-95(B)", ""),
            "mAP50_at_best_mAP50-95": best_map.get("metrics/mAP50(B)", ""),
        }
        for key in METRIC_KEYS:
            row[f"last/{key}"] = metrics.get(key, "")
        rows.append(row)
        seen.add(run_dir.resolve())
        seen_specs.add(spec.key)

    # Keep expected rows visible before a first run has produced results.csv.
    for spec in specs:
        if spec.key in seen_specs:
            continue
        run_dir = project / f"{name_prefix}{spec.run_suffix}"
        if run_dir.resolve() in seen:
            continue
        row = {
            "model": spec.label,
            "cfg": rel(spec.cfg),
            "run_dir": rel(run_dir),
            "last_epoch": "",
            "best_mAP50_epoch": "",
            "best_mAP50": "",
            "best_mAP50-95_at_best_mAP50": "",
            "best_mAP50-95_epoch": "",
            "best_mAP50-95": "",
            "mAP50_at_best_mAP50-95": "",
        }
        for key in METRIC_KEYS:
            row[f"last/{key}"] = ""
        rows.append(row)

    return rows


def write_summary(project: Path, specs: list[ModelSpec], name_prefix: str) -> tuple[Path, Path]:
    project.mkdir(parents=True, exist_ok=True)
    rows = summary_rows(project, specs, name_prefix)

    csv_path = project / "summary.csv"
    fields = [
        "model",
        "cfg",
        "run_dir",
        "last_epoch",
        "best_mAP50_epoch",
        "best_mAP50",
        "best_mAP50-95_at_best_mAP50",
        "best_mAP50-95_epoch",
        "best_mAP50-95",
        "mAP50_at_best_mAP50-95",
        *[f"last/{key}" for key in METRIC_KEYS],
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_path = project / "summary.md"
    md_lines = [
        "# Reproduction Summary",
        "",
        "| model | last epoch | best mAP50 | best mAP50 epoch | best mAP50-95 | best mAP50-95 epoch | last mAP50 | last mAP50-95 | last train/box | last train/cls | last train/moe | run_dir |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        md_lines.append(
            "| {model} | {epoch} | {best_map50} | {best_map50_epoch} | {best_map} | {best_map_epoch} | {last_map50} | {last_map} | {tbox} | {tcls} | {tmoe} | {run_dir} |".format(
                model=row["model"],
                epoch=row["last_epoch"],
                best_map50=row["best_mAP50"],
                best_map50_epoch=row["best_mAP50_epoch"],
                best_map=row["best_mAP50-95"],
                best_map_epoch=row["best_mAP50-95_epoch"],
                last_map50=row["last/metrics/mAP50(B)"],
                last_map=row["last/metrics/mAP50-95(B)"],
                tbox=row["last/train/box_loss"],
                tcls=row["last/train/cls_loss"],
                tmoe=row["last/train/moe_loss"],
                run_dir=row["run_dir"],
            )
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def start_wandb_run(args: argparse.Namespace, spec: ModelSpec, dataset_name: str, run_name: str, config: dict) -> None:
    if args.wandb == "disabled":
        return

    import wandb

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    tags.extend([dataset_name, spec.key, "rhino-bird-issue-49"])
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group or dataset_name,
        name=run_name,
        tags=tags,
        config={
            "dataset": dataset_name,
            "model_label": spec.label,
            "model_key": spec.key,
            "model_cfg": rel(spec.cfg),
            **config,
        },
        reinit="finish_previous",
    )
    print(f"[wandb] mode={args.wandb} project={args.wandb_project} run={run.name} url={run.url}")


def check_build(specs: list[ModelSpec]) -> None:
    from ultralytics import YOLO

    for spec in specs:
        model = YOLO(str(spec.cfg))
        params = sum(p.numel() for p in model.model.parameters())
        print(f"[build-ok] {spec.label}: params={params / 1e6:.3f}M cfg={rel(spec.cfg)}")


def init_weight_for(args: argparse.Namespace, spec: ModelSpec) -> Path | None:
    value = args.init_v01 if spec.key == "v01" else args.init_esmoe
    return resolve(value) if value else None


def train_one(args: argparse.Namespace, spec: ModelSpec, data_yaml: Path, project: Path, dataset_name: str) -> None:
    from ultralytics import YOLO

    run_name = f"{args.name_prefix}{spec.run_suffix}"
    run_dir = project / run_name
    last_pt = run_dir / "weights/last.pt"
    init_weight = init_weight_for(args, spec)

    if args.resume and last_pt.exists():
        model_source = last_pt
        resume = True
        print(f"[resume] {spec.label}: {rel(last_pt)}")
    elif init_weight:
        model_source = init_weight
        resume = False
        print(f"[finetune] {spec.label}: init={rel(init_weight)}")
    else:
        model_source = spec.cfg
        resume = False
        print(f"[train] {spec.label}: cfg={rel(spec.cfg)}")

    model = YOLO(str(model_source))
    kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        project=str(project),
        name=run_name,
        exist_ok=args.exist_ok,
        val=True,
        plots=args.plots,
        cache=args.cache,
        patience=args.patience,
        amp=args.amp,
        resume=resume,
        save_period=args.save_period,
    )
    if not init_weight and not resume:
        kwargs["pretrained"] = False
    start_wandb_run(args, spec, dataset_name, run_name, kwargs)

    start = time.time()
    model.train(**kwargs)
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        raise RuntimeError(f"Training finished without results.csv: {rel(results_csv)}")
    print(f"[done] {spec.label}: {(time.time() - start) / 3600:.2f} h")


def run_reproduction(args: argparse.Namespace, *, dataset_name: str) -> int:
    configure_wandb(args.wandb)
    specs = selected_specs(args.model)
    project = resolve(args.project)
    data_yaml = prepared_data_yaml(args.data, args.dataset_root, project)
    ready = dataset_ready(args.dataset_root, dataset_name)

    print(f"[dataset] {dataset_name}")
    print(f"[data-yaml] {rel(data_yaml)}")
    print(f"[dataset-root] {rel(resolve(args.dataset_root))}")
    print(f"[dataset-ready] {ready}")
    print(f"[project] {rel(project)}")
    for spec in specs:
        print(f"[model] {spec.label}: {rel(spec.cfg)}")

    if args.dry_run:
        return 0
    if args.check_build:
        check_build(specs)
        write_summary(project, specs, args.name_prefix)
        return 0
    if args.summary_only:
        csv_path, md_path = write_summary(project, specs, args.name_prefix)
        print(f"[summary] {rel(csv_path)}")
        print(f"[summary] {rel(md_path)}")
        return 0
    if args.prepare_data_only:
        check_det_dataset(str(data_yaml), autodownload=args.download_dataset)
        print(f"[dataset-ready] {dataset_ready(args.dataset_root, dataset_name)}")
        return 0
    if not ready and not args.download_dataset:
        print("[blocked] Dataset files are missing. Re-run with --download-dataset or prepare the dataset root first.")
        return 2

    for spec in specs:
        train_one(args, spec, data_yaml, project, dataset_name)
        csv_path, md_path = write_summary(project, specs, args.name_prefix)
        print(f"[summary] {rel(csv_path)}")
        print(f"[summary] {rel(md_path)}")
    return 0
