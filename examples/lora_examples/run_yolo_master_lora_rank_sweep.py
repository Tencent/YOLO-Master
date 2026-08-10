#!/usr/bin/env python3
"""Run YOLO-Master LoRA rank sweeps for VisDrone and brain-tumor examples.

Usage:
  python examples/lora_examples/run_yolo_master_lora_rank_sweep.py --scene visdrone --device 0
  python examples/lora_examples/run_yolo_master_lora_rank_sweep.py --scene brain_tumor --device 0
  python examples/lora_examples/run_yolo_master_lora_rank_sweep.py --scene all --device 0
  python examples/lora_examples/run_yolo_master_lora_rank_sweep.py --scene all --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
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
    },
    "brain_tumor": {
        "cfg": "examples/lora_examples/yolo_master_brain_tumor_lora.yaml",
        "base_name": "yolo_master_brain_tumor_lora",
    },
}

SUMMARY_FIELDS = [
    "protocol_id",
    "dataset",
    "rank",
    "alpha",
    "epochs",
    "fraction",
    "amp",
    "batch",
    "imgsz",
    "mAP50_95",
    "best_epoch",
    "trainable_params",
    "adapter_params",
    "train_time_min",
    "peak_gpu_memory_gb",
    "status",
    "return_code",
    "log",
    "run_dir",
]


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
        "map50": best.get("metrics/mAP50(B)", ""),
        "precision": best.get("metrics/precision(B)", ""),
        "recall": best.get("metrics/recall(B)", ""),
        "completed_epochs": len(rows),
    }


def _parse_gpu_mem(value: str) -> float:
    match = re.search(r"([0-9]*\.?[0-9]+)\s*G", str(value))
    return float(match.group(1)) if match else 0.0


def parse_log(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    # Extract trainable and adapter parameter counts
    trainable = ""
    adapter_params = ""
    lora_module_count = ""
    match = re.search(
        r"Trainable:\s*([0-9,]+).*?Adapter Params:\s*([0-9,]+)",
        text, re.S
    )
    if match:
        trainable = match.group(1).replace(",", "")
        adapter_params = match.group(2).replace(",", "")
    # Count LoRA modules
    lora_match = re.search(r"Final Targets Passed to PEFT[:\s]*(\d+)", text)
    if lora_match:
        lora_module_count = lora_match.group(1)
    # Peak VRAM (from various log formats)
    peak_vram = _peak_gpu_mem_from_log(text)
    completed = bool(re.search(r"\b\d+\s+epochs completed\b", text))
    return {
        "trainable_params": trainable,
        "adapter_params": adapter_params,
        "lora_module_count": lora_module_count,
        "peak_vram_gb": peak_vram,
        "completed": completed,
    }


def _peak_gpu_mem_from_log(text: str) -> str:
    values = [
        float(match.group(1))
        for match in re.finditer(r"\s([0-9]+(?:\.[0-9]+)?)G\s+", text)
    ]
    return f"{max(values):.2f}" if values else ""


def summarize_run(
    scene: str, rank: int, run_dir: Path, minutes: float, return_code: int, log_path: Path
) -> dict:
    args = read_yaml(run_dir / "args.yaml")
    metrics = read_results(run_dir / "results.csv")
    log_info = parse_log(log_path)
    return {
        "protocol_id": protocol_id,
        "dataset": scene,
        "rank": rank,
        "alpha": rank * 2,
        "lora_module_count": log_info.get("lora_module_count", ""),
        "trainable_params": log_info.get("trainable_params", ""),
        "adapter_params": log_info.get("adapter_params", ""),
        "best_epoch": metrics.get("best_epoch", ""),
        "map50": metrics.get("map50", ""),
        "map50_95": metrics.get("map50_95", ""),
        "precision": metrics.get("precision", ""),
        "recall": metrics.get("recall", ""),
        "train_time_min": f"{minutes:.1f}" if minutes else "",
        "peak_vram_gb": log_info.get("peak_vram_gb", ""),
        "epochs_requested": args.get("epochs", ""),
        "fraction": args.get("fraction", ""),
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
        "lora_module_count",
        "trainable_params",
        "adapter_params",
        "best_epoch",
        "map50",
        "map50_95",
        "precision",
        "recall",
        "train_time_min",
        "peak_vram_gb",
        "epochs_requested",
        "fraction",
        "status",
        "return_code",
        "log",
        "run_dir",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=[*SCENES.keys(), "all"], default="all")
    parser.add_argument("--ranks", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/lora_examples")
    parser.add_argument(
        "--output",
        default="examples/lora_examples/yolo_master_lora_rank_sweep_results.csv",
    )
    parser.add_argument("--log-dir", default="runs/lora_examples/logs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    safe_protocol = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.protocol_id).strip("._")
    if not safe_protocol:
        parser.error("--protocol-id must contain at least one filename-safe character")
    output = Path(args.output or f"examples/lora_examples/yolo_master_lora_rank_sweep_results_{safe_protocol}.csv")
    if output.exists() and not args.overwrite:
        parser.error(f"output already exists: {output}; pass --overwrite to replace it explicitly")

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
                f"project={args.project}",
                f"name={name}",
                "exist_ok=True",
            ]
            log_path = Path(args.log_dir) / f"{name}.log"
            print(f"\n{'='*60}")
            print(f"  Scene: {scene}  |  Rank: r={rank}  |  Alpha: {rank*2}")
            print(f"  Log: {log_path}")
            print(f"{'='*60}")
            minutes, return_code = run_command_with_log(cmd, log_path, args.dry_run)
            row = summarize_run(scene, rank, run_dir, minutes, return_code, log_path)
            rows.append(row)
            write_summary(rows, Path(args.output))
            if return_code != 0:
                print(f"  ❌ Failed with exit code {return_code}, continuing...")
                continue

    write_summary(rows, Path(args.output))
    print(f"\n✅ Summary written to {args.output}")


if __name__ == "__main__":
    main()
