#!/usr/bin/env python3
"""Run YOLO-Master LoRA rank sweeps for VisDrone and brain-tumor examples."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover - user-facing dependency check
    raise SystemExit("PyYAML is required to parse Ultralytics args.yaml files.") from exc


SCENES = {
    "visdrone": {
        "cfg": "examples/lora_examples/yolo_master_visdrone_lora.yaml",
        "base_name": "yolo_master_visdrone_lora",
        "epochs": 30,
        "fraction": 0.2,
    },
    "brain_tumor": {
        "cfg": "examples/lora_examples/yolo_master_brain_tumor_lora.yaml",
        "base_name": "yolo_master_brain_tumor_lora",
        "epochs": 40,
        "fraction": 1.0,
    },
}


def run_command(cmd: list[str], dry_run: bool) -> float:
    if dry_run:
        print(" ".join(cmd))
        return 0.0
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    return (time.perf_counter() - start) / 60.0


def run_command_with_log(cmd: list[str], log_path: Path, dry_run: bool) -> tuple[float, int]:
    if dry_run:
        print(" ".join(cmd))
        return 0.0, 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return_code = proc.wait()
    return (time.perf_counter() - start) / 60.0, return_code


def read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_results(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    best = max(rows, key=lambda row: float(row.get("metrics/mAP50-95(B)", 0.0) or 0.0))
    return {
        "best_epoch": best.get("epoch", ""),
        "map50_95": best.get("metrics/mAP50-95(B)", ""),
        "completed_epochs": len(rows),
    }


def _parse_gpu_mem(value: str) -> float:
    match = re.search(r"([0-9]*\.?[0-9]+)\s*G", str(value))
    return float(match.group(1)) if match else 0.0


def _peak_gpu_mem(rows: list[dict]) -> str:
    peak = max((_parse_gpu_mem(row.get("GPU_mem", "")) for row in rows), default=0.0)
    return f"{peak:.3f}" if peak else ""


def parse_log(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    trainable = ""
    adapter_params = ""
    match = re.search(r"Trainable:\s*([0-9,]+).*?Adapter Params:\s*([0-9,]+)", text, re.S)
    if match:
        trainable = match.group(1).replace(",", "")
        adapter_params = match.group(2).replace(",", "")
    peak_vram = _peak_gpu_mem_from_log(text)
    completed = bool(re.search(r"\b\d+\s+epochs completed\b", text))
    return {
        "trainable_params": trainable,
        "adapter_params": adapter_params,
        "peak_vram_gb": peak_vram,
        "completed": completed,
    }


def _peak_gpu_mem_from_log(text: str) -> str:
    values = [float(match.group(1)) for match in re.finditer(r"\s([0-9]+(?:\.[0-9]+)?)G\s+", text)]
    return f"{max(values):.3f}" if values else ""


def summarize_run(scene: str, rank: int, run_dir: Path, minutes: float, return_code: int, log_path: Path) -> dict:
    args = read_yaml(run_dir / "args.yaml")
    metrics = read_results(run_dir / "results.csv")
    log_info = parse_log(log_path)
    return {
        "scene": scene,
        "rank": rank,
        "alpha": rank * 2,
        "epochs": args.get("epochs", ""),
        "fraction": args.get("fraction", ""),
        "map50_95": metrics.get("map50_95", ""),
        "best_epoch": metrics.get("best_epoch", ""),
        "trainable_params": log_info.get("trainable_params", ""),
        "adapter_params": log_info.get("adapter_params", ""),
        "train_time_min": f"{minutes:.2f}" if minutes else "",
        "peak_vram_gb": log_info.get("peak_vram_gb", ""),
        "status": "completed" if log_info.get("completed") else "incomplete",
        "return_code": return_code,
        "log": str(log_path),
        "run_dir": str(run_dir),
    }


def write_summary(rows: Iterable[dict], output: Path) -> None:
    rows = list(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene",
        "rank",
        "alpha",
        "epochs",
        "fraction",
        "map50_95",
        "best_epoch",
        "trainable_params",
        "adapter_params",
        "train_time_min",
        "peak_vram_gb",
        "status",
        "return_code",
        "log",
        "run_dir",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=[*SCENES.keys(), "all"], default="all")
    parser.add_argument("--ranks", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/lora_rank_sweeps")
    parser.add_argument("--output", default="examples/lora_examples/yolo_master_lora_rank_sweep_results.csv")
    parser.add_argument("--log-dir", default="runs/lora_rank_sweeps/logs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = SCENES if args.scene == "all" else {args.scene: SCENES[args.scene]}
    rows = []
    for scene, spec in selected.items():
        for rank in args.ranks:
            name = f"{spec['base_name']}_r{rank}"
            run_dir = Path(args.project) / name
            cmd = [
                "yolo",
                "train",
                f"cfg={spec['cfg']}",
                f"lora_r={rank}",
                f"lora_alpha={rank * 2}",
                f"device={args.device}",
                f"epochs={spec['epochs']}",
                f"fraction={spec['fraction']}",
                f"project={args.project}",
                f"name={name}",
                "exist_ok=True",
            ]
            log_path = Path(args.log_dir) / f"{name}.log"
            minutes, return_code = run_command_with_log(cmd, log_path, args.dry_run)
            rows.append(summarize_run(scene, rank, run_dir, minutes, return_code, log_path))
            write_summary(rows, Path(args.output))
            if return_code != 0:
                raise SystemExit(return_code)

    write_summary(rows, Path(args.output))
    print(f"Wrote summary to {args.output}")


if __name__ == "__main__":
    main()
