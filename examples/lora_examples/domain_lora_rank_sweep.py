#!/usr/bin/env python3
"""Run LoRA rank sweeps for YOLO-Master domain adaptation configs.

The script launches one `yolo train` run per rank and writes a compact CSV
summary with mAP50-95, trainable parameter count placeholder, wall-clock time,
and peak CUDA memory when available from the run artifacts/logs.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path


SCENARIOS = {
    "visdrone": {
        "cfg": "examples/lora_examples/yolo_master_visdrone_lora.yaml",
        "name_prefix": "yolo_master_visdrone_lora",
        "epochs": 30,
    },
    "brain_tumor": {
        "cfg": "examples/lora_examples/yolo_master_brain_tumor_lora.yaml",
        "name_prefix": "yolo_master_brain_tumor_lora",
        "epochs": 40,
    },
}


def latest_metric(results_csv: Path, key: str) -> str:
    if not results_csv.exists():
        return ""
    rows = list(csv.DictReader(results_csv.open(newline="")))
    if not rows:
        return ""
    row = rows[-1]
    for candidate in (key, f"metrics/{key}", f"metrics/{key}(B)"):
        if candidate in row and row[candidate] != "":
            return row[candidate]
    return ""


def write_summary_row(path: Path, row: dict[str, str]) -> None:
    fieldnames = [
        "scenario",
        "rank",
        "alpha",
        "epochs",
        "mAP50-95",
        "trainable_params",
        "train_time_min",
        "peak_mem_gb",
        "run_dir",
    ]
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO-Master LoRA rank sweeps.")
    parser.add_argument("--scenario", choices=SCENARIOS.keys(), required=True)
    parser.add_argument("--ranks", default="4,8,16", help="Comma-separated LoRA ranks.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/lora_examples")
    parser.add_argument("--summary", default="examples/lora_examples/domain_lora_rank_results.csv")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching training.")
    args = parser.parse_args()

    scenario = SCENARIOS[args.scenario]
    ranks = [int(r.strip()) for r in args.ranks.split(",") if r.strip()]
    summary_path = Path(args.summary)

    for rank in ranks:
        alpha = rank * 2
        run_name = f"{scenario['name_prefix']}_r{rank}"
        cmd = [
            "yolo",
            "train",
            f"cfg={scenario['cfg']}",
            f"device={args.device}",
            f"project={args.project}",
            f"name={run_name}",
            f"epochs={scenario['epochs']}",
            f"lora_r={rank}",
            f"lora_alpha={alpha}",
        ]

        if args.dry_run:
            print(" ".join(cmd))
            continue

        start = time.perf_counter()
        subprocess.run(cmd, check=True)
        elapsed_min = (time.perf_counter() - start) / 60.0

        run_dir = Path(args.project) / run_name
        results_csv = run_dir / "results.csv"
        write_summary_row(
            summary_path,
            {
                "scenario": args.scenario,
                "rank": str(rank),
                "alpha": str(alpha),
                "epochs": str(scenario["epochs"]),
                "mAP50-95": latest_metric(results_csv, "mAP50-95"),
                "trainable_params": "see args.yaml / trainer log",
                "train_time_min": f"{elapsed_min:.2f}",
                "peak_mem_gb": latest_metric(results_csv, "GPU_mem"),
                "run_dir": str(run_dir),
            },
        )


if __name__ == "__main__":
    main()
