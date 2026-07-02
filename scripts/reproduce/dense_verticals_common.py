#!/usr/bin/env python3
"""Shared runner for dense vertical dataset reproduction jobs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


WEIGHTS_DIR = ROOT / "weights"
V01_N_URL = "https://github.com/Tencent/YOLO-Master/releases/download/YOLO-Master-v26.02/YOLO-Master-v0.1-N.pt"
ESMOE_N_URL = "https://github.com/Tencent/YOLO-Master/releases/download/YOLO-Master-v26.02/YOLO-Master-EsMoE-N.pt"

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
class DatasetSpec:
    key: str
    label: str
    data_yaml: Path
    default_project: Path


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    cfg: Path
    weight_name: str
    weight_url: str
    uses_esmoe: bool = False


MODEL_SPECS = {
    "esmoe_n": ModelSpec(
        key="esmoe_n",
        label="YOLO-Master-EsMoE-N",
        cfg=ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-n.yaml",
        weight_name="YOLO-Master-EsMoE-N.pt",
        weight_url=ESMOE_N_URL,
        uses_esmoe=True,
    ),
    "v01_n": ModelSpec(
        key="v01_n",
        label="YOLO-Master-v0.1-N",
        cfg=ROOT / "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml",
        weight_name="YOLO-Master-v0.1-N.pt",
        weight_url=V01_N_URL,
        uses_esmoe=False,
    ),
}


class Tee:
    """Write stream output to console and a log file."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def dataset_spec(key: str) -> DatasetSpec:
    specs = {
        "visdrone": DatasetSpec(
            key="visdrone",
            label="VisDrone2019-DET",
            data_yaml=ROOT / "ultralytics/cfg/datasets/VisDrone.yaml",
            default_project=ROOT / "runs/reproduce_visdrone",
        ),
        "sku110k": DatasetSpec(
            key="sku110k",
            label="SKU-110K",
            data_yaml=ROOT / "ultralytics/cfg/datasets/SKU-110K.yaml",
            default_project=ROOT / "runs/reproduce_sku110k",
        ),
    }
    if key not in specs:
        raise ValueError(f"unknown dataset key: {key}")
    return specs[key]


def existing_weight_or_url(spec: ModelSpec) -> str:
    local = WEIGHTS_DIR / spec.weight_name
    return str(local) if local.exists() else spec.weight_url


def read_last_metrics(results_csv: Path) -> dict[str, str]:
    if not results_csv.exists():
        return {}
    with results_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return {k.strip(): v for k, v in rows[-1].items()}


def completed_epoch(run_dir: Path) -> int | None:
    value = read_last_metrics(run_dir / "results.csv").get("epoch")
    if value in {None, ""}:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def completed_epochs(run_dir: Path) -> int:
    done = completed_epoch(run_dir)
    return 0 if done is None else done + 1


def _set_esmoe_sparse_inference(module, enabled: bool) -> int:
    count = 0
    for submodule in module.modules():
        if submodule.__class__.__name__ != "ES_MOE":
            continue
        if hasattr(submodule, "enable_sparse_inference"):
            submodule.enable_sparse_inference(enabled)
        else:
            submodule.use_sparse_inference = enabled
        count += 1
    return count


def _make_dense_inference_callback():
    state = {"configured": False}

    def _callback(trainer) -> None:
        if state["configured"]:
            return
        touched = []
        model = getattr(trainer, "model", None)
        if model is not None:
            count = _set_esmoe_sparse_inference(model, enabled=False)
            if count:
                touched.append(f"model={count}")
        ema = getattr(getattr(trainer, "ema", None), "ema", None)
        if ema is not None:
            count = _set_esmoe_sparse_inference(ema, enabled=False)
            if count:
                touched.append(f"ema={count}")
        if touched:
            print(f"[dense-eval] disabled sparse inference on ES_MOE modules ({', '.join(touched)})", flush=True)
        state["configured"] = True

    return _callback


def _make_wandb_callbacks(
    run_name: str,
    dataset: DatasetSpec,
    spec: ModelSpec,
    args: argparse.Namespace,
    dense_eval: bool,
):
    def _on_pretrain_routine_start(trainer) -> None:
        try:
            import wandb
        except ImportError:
            return
        if not wandb.run:
            return
        wandb.config.update(
            {
                "dense_eval": dense_eval,
                "sparse_eval": args.sparse_eval,
                "dataset_key": dataset.key,
                "dataset_label": dataset.label,
                "model_key": spec.key,
                "model_label": spec.label,
                "run_name": run_name,
            },
            allow_val_change=True,
        )

    return {"on_pretrain_routine_start": _on_pretrain_routine_start}


def float_or_blank(value: str | None) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6g}"
    except ValueError:
        return value


def select_models(keys: list[str]) -> list[ModelSpec]:
    specs = []
    for key in keys:
        if key not in MODEL_SPECS:
            raise SystemExit(f"unknown model key: {key}. Choices: {', '.join(MODEL_SPECS)}")
        spec = MODEL_SPECS[key]
        if not spec.cfg.exists():
            raise SystemExit(f"missing model config: {spec.cfg}")
        specs.append(spec)
    return specs


def configure_wandb(args: argparse.Namespace, dataset: DatasetSpec) -> None:
    if args.dry_run or args.summary_only:
        return
    from ultralytics.utils import SETTINGS

    if not args.wandb:
        SETTINGS["wandb"] = False
        return
    SETTINGS["wandb"] = True
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project or f"YOLO-Master-{dataset.key}")
    if args.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
    if args.wandb_mode:
        os.environ.setdefault("WANDB_MODE", args.wandb_mode)


def check_build(specs: list[ModelSpec], nc: int) -> None:
    from ultralytics.nn.tasks import DetectionModel

    for spec in specs:
        model = DetectionModel(str(spec.cfg), ch=3, nc=nc, verbose=False)
        params = sum(p.numel() for p in model.parameters())
        print(f"[build-ok] {spec.label:<22} params={params / 1e6:.3f}M cfg={spec.cfg.relative_to(ROOT)}")


def train_one(args: argparse.Namespace, dataset: DatasetSpec, spec: ModelSpec, project: Path) -> dict[str, str]:
    from ultralytics import YOLO

    start = time.time()
    run_name = f"{dataset.key}_{spec.key}"
    run_dir = project / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    done = completed_epoch(run_dir)
    done_epochs = completed_epochs(run_dir)
    target_epochs = args.epochs
    if args.extra_epochs is not None and done is not None:
        target_epochs = done_epochs + args.extra_epochs

    last_pt = run_dir / "weights/last.pt"
    best_pt = run_dir / "weights/best.pt"

    resume_requested = args.resume_existing or args.extra_epochs is not None
    if args.skip_existing and (run_dir / "results.csv").exists() and not resume_requested:
        print(f"[skip] {spec.label}: existing {run_dir / 'results.csv'}")
        return {"model": spec.label, "status": "skipped", "duration_s": "0"}

    if best_pt.exists() and done is not None and done_epochs >= target_epochs:
        print(f"[skip] {spec.label}: complete at epoch {done}, target={target_epochs}")
        return {"model": spec.label, "status": "skipped", "duration_s": "0"}

    dense_eval = spec.uses_esmoe and not args.sparse_eval
    resume = last_pt.exists() and done is not None

    with log_path.open("a", encoding="utf-8") as log_file, redirect_stdout(Tee(sys.stdout, log_file)), redirect_stderr(
        Tee(sys.stderr, log_file)
    ):
        print(f"[run] dataset={dataset.label} model={spec.label} target_epochs={target_epochs}", flush=True)
        print(f"[run] data={dataset.data_yaml.relative_to(ROOT)} project={project} name={run_name}")
        print(f"[run] wandb={args.wandb} project={os.environ.get('WANDB_PROJECT', '')}")
        if resume:
            print(f"[resume] checkpoint={last_pt} completed_epoch={done} -> target={target_epochs}", flush=True)
            model = YOLO(str(last_pt))
        else:
            model_source = str(spec.cfg) if args.from_cfg else existing_weight_or_url(spec)
            print(
                f"[train] source={model_source} data={dataset.data_yaml} "
                f"sparse_eval={args.sparse_eval} dense_eval={dense_eval}",
                flush=True,
            )
            model = YOLO(model_source)

        if dense_eval:
            cb = _make_dense_inference_callback()
            model.add_callback("on_pretrain_routine_end", cb)
            model.add_callback("on_train_start", cb)
        elif spec.uses_esmoe:
            print(
                "[warn] ES_MOE sparse eval is enabled; this reproduces the legacy sparse-validation path "
                "and can collapse VisDrone mAP. Use --no-sparse-eval for the corrected dense evaluation.",
                flush=True,
            )

        if args.wandb and args.wandb_mode != "disabled":
            for event, fn in _make_wandb_callbacks(run_name, dataset, spec, args, dense_eval).items():
                model.add_callback(event, fn)

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
            name=run_name,
            exist_ok=True,
            pretrained=False if args.from_cfg else True,
            val=True,
            plots=args.plots,
            cache=args.cache,
            patience=args.patience,
            amp=args.amp,
            resume=resume,
            verbose=args.verbose,
        )

    duration = time.time() - start
    return {"model": spec.label, "status": "resumed" if resume else "ok", "duration_s": f"{duration:.2f}"}


def collect_summary(project: Path, dataset: DatasetSpec, specs: list[ModelSpec]) -> list[dict[str, str]]:
    rows = []
    for spec in specs:
        run_name = f"{dataset.key}_{spec.key}"
        run_dir = project / run_name
        metrics = read_last_metrics(run_dir / "results.csv")
        row = {
            "dataset": dataset.key,
            "model": spec.label,
            "run_dir": str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
            "epoch": metrics.get("epoch", ""),
        }
        for key in METRIC_KEYS:
            row[key] = float_or_blank(metrics.get(key))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "model", "run_dir", "epoch", *METRIC_KEYS]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]], wandb_url: str = "") -> None:
    headers = ["Dataset", "Model", "Epoch", "mAP50", "mAP50-95", "box_loss", "cls_loss", "moe_loss", "W&B"]
    lines = [
        "# Dense vertical reproduction results",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join([":--"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["model"],
                    row["epoch"],
                    row["metrics/mAP50(B)"],
                    row["metrics/mAP50-95(B)"],
                    row["train/box_loss"] or row["val/box_loss"],
                    row["train/cls_loss"] or row["val/cls_loss"],
                    row["train/moe_loss"] or row["val/moe_loss"],
                    wandb_url,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Metrics are read from each run's final `results.csv` row.",
            "- Keep the full per-epoch CSV and `train.log` files in the run directories for PR review.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(project: Path, dataset: DatasetSpec, specs: list[ModelSpec], wandb_url: str = "") -> Path:
    rows = collect_summary(project, dataset, specs)
    csv_path = project / "summary.csv"
    md_path = project / "summary.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, wandb_url=wandb_url)
    print(f"[summary] wrote {csv_path}")
    print(f"[summary] wrote {md_path}")
    return csv_path


def parse_common_args(dataset: DatasetSpec, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Reproduce YOLO-Master dense detection on {dataset.label}.")
    parser.add_argument("--models", nargs="+", default=["esmoe_n", "v01_n"], choices=tuple(MODEL_SPECS))
    parser.add_argument("--project", type=Path, default=dataset.default_project)
    parser.add_argument("--epochs", type=int, default=100, help="Use 100-300 for paper-style reproduction.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="", help="Ultralytics device string: '', cpu, 0, 0,1, mps ...")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--from-cfg", action="store_true", help="Train from YAML instead of release checkpoints.")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-mode", default="", choices=("", "online", "offline", "disabled"))
    parser.add_argument("--wandb-url", default="", help="Public W&B report/run URL to copy into summary.md.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--extra-epochs", type=int, help="Additional epochs from each run's weights/last.pt.")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-build", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--sparse-eval", action=argparse.BooleanOptionalAction, default=False,
                    help="ES_MOE sparse inference at validation/inference. Use --no-sparse-eval "
                        "with YOLO-Master-EsMoE-N to keep the corrected dense evaluation path "
                        "(train==eval). Pass --sparse-eval only if you explicitly want to "
                        "reproduce the legacy sparse-validation behavior. No-op for v0.1-N.")
    return parser.parse_args(argv)


def run_dataset(dataset_key: str, argv: list[str] | None = None) -> int:
    dataset = dataset_spec(dataset_key)
    args = parse_common_args(dataset, argv)
    specs = select_models(args.models)
    project = args.project if args.project.is_absolute() else ROOT / args.project

    configure_wandb(args, dataset)
    print(f"[dataset] {dataset.label}: {dataset.data_yaml.relative_to(ROOT)}")
    print(f"[project] {project}")
    for spec in specs:
        source = spec.cfg if args.from_cfg else existing_weight_or_url(spec)
        print(f"  - {spec.label}: {source}")

    if args.dry_run:
        return 0
    if args.check_build:
        nc = 1 if dataset.key == "sku110k" else 10
        check_build(specs, nc=nc)
        write_summary(project, dataset, specs, wandb_url=args.wandb_url)
        return 0
    if args.summary_only:
        write_summary(project, dataset, specs, wandb_url=args.wandb_url)
        return 0

    statuses = []
    project.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        try:
            statuses.append(train_one(args, dataset, spec, project))
        except Exception as exc:
            print(f"[fail] {spec.label}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            statuses.append({"model": spec.label, "status": "failed", "error": str(exc)})
            if args.stop_on_failure:
                break
        finally:
            write_summary(project, dataset, specs, wandb_url=args.wandb_url)

    with (project / "status.json").open("w", encoding="utf-8") as f:
        json.dump(statuses, f, indent=2, ensure_ascii=False)
    success_states = {"ok", "skipped", "resumed"}
    return 0 if all(s.get("status") in success_states for s in statuses) else 1
