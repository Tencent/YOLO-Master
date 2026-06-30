#!/usr/bin/env python3
"""Shared helpers for YOLO-Master vertical-domain baseline reproduction."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

MODEL_SPECS = {
    "v0.1-n": "https://github.com/Tencent/YOLO-Master/releases/download/YOLO-Master-v26.02/YOLO-Master-v0.1-N.pt",
    "esmoe-n": "https://github.com/Tencent/YOLO-Master/releases/download/YOLO-Master-v26.02/YOLO-Master-EsMoE-N.pt",
}

METRIC_KEYS = [
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "train/box_loss",
    "train/cls_loss",
    "train/moe_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/moe_loss",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    data: str
    project: str
    max_det: int
    batch: int


def read_last_metrics(results_csv: Path) -> dict[str, str]:
    if not results_csv.exists():
        return {}
    with results_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return {key.strip(): value for key, value in rows[-1].items()}


def metric_value(metrics: dict[str, str], key: str) -> str:
    value = metrics.get(key, "")
    if value == "":
        return ""
    try:
        return f"{float(value):.6g}"
    except ValueError:
        return value


def train_command(args: argparse.Namespace, dataset: DatasetSpec, model_name: str, model: str) -> list[str]:
    run_name = f"{dataset.name}_{model_name}"
    cmd = [
        "yolo",
        "train",
        f"model={model}",
        f"data={dataset.data}",
        f"imgsz={args.imgsz}",
        f"epochs={args.epochs}",
        f"batch={args.batch}",
        f"device={args.device}",
        f"workers={args.workers}",
        f"project={args.project}",
        f"name={run_name}",
        f"max_det={dataset.max_det}",
        f"seed={args.seed}",
        "deterministic=True",
        "val=True",
        "plots=True",
        f"cache={args.cache}",
        f"amp={args.amp}",
        f"exist_ok={args.exist_ok}",
    ]
    if args.resume:
        cmd.append("resume=True")
    return cmd


def write_summary(args: argparse.Namespace, dataset: DatasetSpec, rows: list[dict[str, str]]) -> Path:
    out_dir = ROOT / args.project
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{dataset.name}_summary.csv"
    fieldnames = [
        "dataset",
        "model",
        "run_dir",
        "epochs",
        "imgsz",
        "duration_min",
        *METRIC_KEYS,
        "wandb_url",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def run_reproduction(dataset: DatasetSpec) -> None:
    parser = argparse.ArgumentParser(description=f"Reproduce {dataset.name} YOLO-Master baselines.")
    parser.add_argument("--epochs", type=int, default=100, help="Recommended range: 100-300.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=dataset.batch)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default=dataset.project)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", default="False")
    parser.add_argument("--amp", default="True")
    parser.add_argument("--exist-ok", default="False")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wandb-project", default="", help="Optional W&B project name.")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    os.environ.setdefault("WANDB_MODE", args.wandb_mode)

    rows: list[dict[str, str]] = []
    for model_name, model in MODEL_SPECS.items():
        cmd = train_command(args, dataset, model_name, model)
        run_dir = ROOT / args.project / f"{dataset.name}_{model_name}"
        print(" ".join(cmd))
        if args.dry_run:
            continue

        start = time.perf_counter()
        subprocess.run(cmd, check=True, cwd=ROOT)
        duration_min = (time.perf_counter() - start) / 60.0
        metrics = read_last_metrics(run_dir / "results.csv")
        rows.append(
            {
                "dataset": dataset.name,
                "model": model_name,
                "run_dir": str(run_dir.relative_to(ROOT)),
                "epochs": str(args.epochs),
                "imgsz": str(args.imgsz),
                "duration_min": f"{duration_min:.2f}",
                **{key: metric_value(metrics, key) for key in METRIC_KEYS},
                "wandb_url": "fill from W&B run page if enabled",
            }
        )

    if rows:
        out = write_summary(args, dataset, rows)
        print(f"[summary] wrote {out.relative_to(ROOT)}")
