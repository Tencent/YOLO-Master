#!/usr/bin/env python3
"""Run issue #52 dynamic MoE schedule comparison on a selected dataset.

The script compares three groups:

- `baseline`: fixed MoE balance coefficient.
- `dynamic`: Gini-driven dynamic balance coefficient.
- `ablation`: fixed low balance coefficient.

Example:

    python scripts/run_moe_dynamic_schedule_ablation.py --dry-run
    python scripts/run_moe_dynamic_schedule_ablation.py --variant dynamic --epochs 100 --wandb offline
    python scripts/run_moe_dynamic_schedule_ablation.py --summary-only
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs/reproduce/_runtime/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs/reproduce/_runtime/matplotlib"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import SETTINGS  # noqa: E402


METRIC_KEY = "metrics/mAP50-95(B)"
MAP50_KEY = "metrics/mAP50(B)"


@dataclass(frozen=True)
class Variant:
    key: str
    name: str
    extra_args: dict[str, Any]


VARIANTS = {
    "baseline": Variant(
        key="baseline",
        name="issue52_fixed_balance",
        extra_args={"moe_dynamic_schedule": "none", "moe_balance_loss": 1.0},
    ),
    "dynamic": Variant(
        key="dynamic",
        name="issue52_gini_balance",
        extra_args={
            "moe_dynamic_schedule": "gini_balance",
            "moe_balance_loss": 1.0,
            "moe_dynamic_gini_target": 0.25,
            "moe_dynamic_gini_alpha": 1.0,
            "moe_dynamic_gini_beta": 0.8,
            "moe_dynamic_balance_min": 0.5,
            "moe_dynamic_balance_max": 2.0,
        },
    ),
    "ablation": Variant(
        key="ablation",
        name="issue52_low_balance",
        extra_args={"moe_dynamic_schedule": "none", "moe_balance_loss": 0.3},
    ),
}


def default_model_cfg() -> Path:
    return ROOT / "ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml"


def default_data_yaml() -> Path:
    local = ROOT / "runs/reproduce/visdrone/_data/VisDrone.local.yaml"
    return local if local.exists() else ROOT / "ultralytics/cfg/datasets/VisDrone.yaml"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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


def read_results(results_csv: Path) -> list[dict[str, str]]:
    if not results_csv.exists():
        return []
    with results_csv.open(newline="", encoding="utf-8") as handle:
        return [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(handle)]


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def first_epoch_at(rows: list[dict[str, str]], target: float) -> int | None:
    for index, row in enumerate(rows, start=1):
        if as_float(row, METRIC_KEY) >= target:
            return index
    return None


def summarize(project: Path, variants: list[Variant]) -> Path:
    rows_by_variant = {variant.key: read_results(project / variant.name / "results.csv") for variant in variants}
    baseline_rows = rows_by_variant.get("baseline", [])
    baseline_final = as_float(baseline_rows[-1], METRIC_KEY) if baseline_rows else float("nan")
    target = baseline_final * 0.95 if baseline_final == baseline_final else float("nan")
    baseline_epoch = first_epoch_at(baseline_rows, target) if target == target else None

    summary_rows = []
    for variant in variants:
        rows = rows_by_variant[variant.key]
        final = rows[-1] if rows else {}
        best = max(rows, key=lambda row: as_float(row, METRIC_KEY)) if rows else {}
        reach_epoch = first_epoch_at(rows, target) if target == target else None
        summary_rows.append(
            {
                "variant": variant.key,
                "run_dir": rel(project / variant.name),
                "epochs": len(rows),
                "final_mAP50-95": final.get(METRIC_KEY, ""),
                "final_mAP50": final.get(MAP50_KEY, ""),
                "best_mAP50-95": best.get(METRIC_KEY, ""),
                "best_mAP50": best.get(MAP50_KEY, ""),
                "target_95pct_baseline_mAP50-95": target if target == target else "",
                "epoch_to_target": reach_epoch or "",
                "convergence_epoch_ratio": (reach_epoch / baseline_epoch) if reach_epoch and baseline_epoch else "",
            }
        )

    out = project / "dynamic_schedule_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "variant",
        "run_dir",
        "epochs",
        "final_mAP50-95",
        "final_mAP50",
        "best_mAP50-95",
        "best_mAP50",
        "target_95pct_baseline_mAP50-95",
        "epoch_to_target",
        "convergence_epoch_ratio",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    return out


def selected_variants(value: str) -> list[Variant]:
    if value == "all":
        return [VARIANTS["baseline"], VARIANTS["dynamic"], VARIANTS["ablation"]]
    return [VARIANTS[value]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("all", "baseline", "dynamic", "ablation"), default="all")
    parser.add_argument("--model", type=Path, default=default_model_cfg())
    parser.add_argument("--data", type=Path, default=default_data_yaml())
    parser.add_argument("--project", type=Path, default=ROOT / "runs/reproduce/issue52_dynamic_schedule")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--wandb", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.model = resolve(args.model)
    args.data = resolve(args.data)
    args.project = resolve(args.project)
    configure_wandb(args.wandb)
    variants = selected_variants(args.variant)

    print(f"[issue52-dynamic] model={rel(args.model)}")
    print(f"[issue52-dynamic] data={rel(args.data)}")
    print(f"[issue52-dynamic] project={rel(args.project)}")
    for variant in variants:
        print(f"  - {variant.key}: {variant.name} {variant.extra_args}")

    if args.dry_run:
        return 0

    if args.summary_only:
        summary = summarize(args.project, [VARIANTS["baseline"], VARIANTS["dynamic"], VARIANTS["ablation"]])
        print(f"[summary] {rel(summary)}")
        return 0

    args.project.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        model = YOLO(str(args.model))
        model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            seed=args.seed,
            patience=args.patience,
            amp=args.amp,
            plots=args.plots,
            project=str(args.project),
            name=variant.name,
            exist_ok=args.exist_ok,
            save_period=args.save_period,
            pretrained=True,
            **variant.extra_args,
        )

    summary = summarize(args.project, [VARIANTS["baseline"], VARIANTS["dynamic"], VARIANTS["ablation"]])
    print(f"[summary] {rel(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
