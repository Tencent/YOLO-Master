#!/usr/bin/env python3
"""Create a reproducible MoE pruning threshold sweep plan.

The script writes a manifest and CSV skeleton for the issue-52 experiment:
thresholds, direct inference, optional LoRA 10-epoch recovery, and the metric
columns expected by the analysis/plotting step. Use ``--execute-prune`` to run
the pruning command for each threshold; validation/LoRA recovery commands are
left in the manifest so they can be launched on the target GPU box.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30)
METRIC_FIELDS = (
    "mAP50-95",
    "mAP50",
    "gflops",
    "latency_ms",
    "params_m",
    "experts_per_layer",
    "expert_usage_gini",
    "convergence_epoch_95pct",
)


def fmt_threshold(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def build_prune_command(model: Path, dataset: str, output_model: Path, threshold: float) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "ultralytics/nn/modules/moe/pruning.py"),
        str(model),
        "--output",
        str(output_model),
        "--threshold",
        f"{threshold:.2f}",
        "--dataset",
        dataset,
    ]


def build_val_command(model: Path, dataset: str, device: str, batch: int, imgsz: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "ultralytics",
        "val",
        f"model={model}",
        f"data={dataset}",
        f"device={device}",
        f"batch={batch}",
        f"imgsz={imgsz}",
    ]


def build_lora_command(model: Path, dataset: str, out_dir: Path, device: str, batch: int, imgsz: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "ultralytics",
        "train",
        f"model={model}",
        f"data={dataset}",
        "epochs=10",
        f"project={out_dir}",
        "name=lora_recovery",
        f"device={device}",
        f"batch={batch}",
        f"imgsz={imgsz}",
        "lora=True",
    ]


def shell_join(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Trained YOLO-Master-EsMoE-N checkpoint.")
    parser.add_argument("--dataset", default="coco.yaml", help="Dataset YAML, e.g. coco.yaml or VisDrone.yaml.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/moe_pruning_sweep")
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--execute-prune", action="store_true", help="Run pruning for every threshold.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    manifest: dict[str, object] = {
        "model": str(args.model),
        "dataset": args.dataset,
        "thresholds": args.thresholds,
        "points": [],
    }

    for threshold in args.thresholds:
        tag = fmt_threshold(threshold)
        point_dir = out_dir / f"threshold_{tag}"
        point_dir.mkdir(parents=True, exist_ok=True)
        pruned_model = point_dir / f"pruned_{tag}.pt"
        prune_cmd = build_prune_command(args.model, args.dataset, pruned_model, threshold)
        direct_val_cmd = build_val_command(pruned_model, args.dataset, args.device, args.batch, args.imgsz)
        lora_cmd = build_lora_command(pruned_model, args.dataset, point_dir, args.device, args.batch, args.imgsz)

        if args.execute_prune:
            subprocess.run(prune_cmd, check=True, cwd=ROOT)

        for recovery in ("direct", "lora10"):
            row = {
                "threshold": f"{threshold:.2f}",
                "recovery": recovery,
                "model": str(pruned_model),
                "prune_command": shell_join(prune_cmd),
                "eval_command": shell_join(direct_val_cmd if recovery == "direct" else lora_cmd),
            }
            row.update({field: "" for field in METRIC_FIELDS})
            rows.append(row)

        manifest["points"].append(
            {
                "threshold": threshold,
                "pruned_model": str(pruned_model),
                "prune_command": prune_cmd,
                "direct_eval_command": direct_val_cmd,
                "lora10_recovery_command": lora_cmd,
            }
        )

    csv_path = out_dir / "moe_pruning_sweep.csv"
    fieldnames = ["threshold", "recovery", "model", "prune_command", "eval_command", *METRIC_FIELDS]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = out_dir / "moe_pruning_sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[moe-sweep] wrote {csv_path}")
    print(f"[moe-sweep] wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
