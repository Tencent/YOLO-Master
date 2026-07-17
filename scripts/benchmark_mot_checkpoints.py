#!/usr/bin/env python3
"""Benchmark trained MoT ablation checkpoints on identical real images."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_mot_routing import image_paths_from_data, letterbox_image  # noqa: E402
from scripts.compare_mot_ablation import (  # noqa: E402
    SPECS,
    count_modules,
    normalize_torch_device,
    percentile,
    sync_device,
)
from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules.moa import C2fMoA, MoABlock  # noqa: E402
from ultralytics.nn.modules.mot import C2fMoT, MoTBlock  # noqa: E402

import cv2  # noqa: E402


def load_image_tensor(image_path: Path, imgsz: int, device: str) -> torch.Tensor:
    """Load one image with detector-style letterboxing; preprocessing is excluded from latency."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = letterbox_image(image, imgsz)
    tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float().unsqueeze(0) / 255.0
    return tensor.to(torch.device(device))


def profile_image_flops(model: torch.nn.Module, tensor: torch.Tensor) -> float:
    """Profile actual operations executed for one routed input and return GFLOPs."""
    with torch.inference_mode(), torch.profiler.profile(with_flops=True) as profile:
        model(tensor)
    return float(sum(event.flops for event in profile.key_averages()) / 1e9)


def benchmark_model(
    model: torch.nn.Module,
    image_paths: list[Path],
    device: str,
    imgsz: int,
    warmup: int,
    flops_images: int,
) -> tuple[list[dict[str, str | float]], list[dict[str, str | float]]]:
    """Measure synchronized model-only latency for every image and FLOPs for a fixed prefix."""
    if not image_paths:
        raise ValueError("benchmark requires at least one image")
    if warmup < 0 or flops_images < 0:
        raise ValueError("warmup and flops_images must be non-negative")
    first = load_image_tensor(image_paths[0], imgsz, device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(first)
            sync_device(device)

    latency_rows: list[dict[str, str | float]] = []
    flops_rows: list[dict[str, str | float]] = []
    for index, image_path in enumerate(image_paths):
        tensor = first if index == 0 else load_image_tensor(image_path, imgsz, device)
        sync_device(device)
        started = time.perf_counter()
        with torch.inference_mode():
            model(tensor)
        sync_device(device)
        latency_rows.append({"image": str(image_path), "latency_ms": (time.perf_counter() - started) * 1000.0})
        if index < flops_images:
            flops_rows.append({"image": str(image_path), "flops_g": profile_image_flops(model, tensor)})
    return latency_rows, flops_rows


def summarize_benchmark(
    key: str,
    checkpoint: Path,
    model: torch.nn.Module,
    latency_rows: list[dict[str, str | float]],
    flops_rows: list[dict[str, str | float]],
    device: str,
    imgsz: int,
    warmup: int,
) -> dict[str, str | float | int]:
    """Build one auditable summary row while retaining raw samples in separate tables."""
    times = [float(row["latency_ms"]) for row in latency_rows]
    flops = [float(row["flops_g"]) for row in flops_rows]
    torch_device = torch.device(device)
    return {
        "key": key,
        "label": SPECS[key].label,
        "checkpoint": str(checkpoint),
        "images": len(times),
        "warmup": warmup,
        "imgsz": imgsz,
        "batch_size": 1,
        "input_dtype": "float32",
        "latency_scope": "model_forward_only_perf_counter_with_device_sync",
        "latency_ms_mean": sum(times) / len(times),
        "latency_ms_p50": percentile(times, 0.50),
        "latency_ms_p95": percentile(times, 0.95),
        "latency_ms_p99": percentile(times, 0.99),
        "latency_ms_min": min(times),
        "latency_ms_max": max(times),
        "flops_images": len(flops),
        "flops_g_mean": sum(flops) / len(flops) if flops else "",
        "flops_g_min": min(flops) if flops else "",
        "flops_g_max": max(flops) if flops else "",
        "flops_method": "torch_profiler_actual_routed_input",
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "params_m": sum(parameter.numel() for parameter in model.parameters()) / 1e6,
        "moablocks": count_modules(model, MoABlock),
        "c2fmoa": count_modules(model, C2fMoA),
        "motblocks": count_modules(model, MoTBlock),
        "c2fmot": count_modules(model, C2fMoT),
        "device": device,
        "device_name": torch.cuda.get_device_name(torch_device) if device.startswith("cuda") else device,
        "torch_version": torch.__version__,
        "cuda_version": str(torch.version.cuda or ""),
        "cudnn_version": str(torch.backends.cudnn.version() or ""),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a nonempty row collection to CSV."""
    if not rows:
        raise ValueError(f"refusing to write empty benchmark table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True, help="Project containing <model>/weights/best.pt.")
    parser.add_argument("--models", nargs="+", choices=tuple(SPECS), default=["v10", "v10_mot", "v10_moa", "v10_moa_mot"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0, help="0 benchmarks the complete split.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--flops-images", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = normalize_torch_device(args.device)
    data = args.data.resolve()
    images = image_paths_from_data(data, args.split, args.limit)
    if not images:
        raise SystemExit(f"no images found for {data}:{args.split}")

    summaries = []
    all_latency_rows: list[dict[str, object]] = []
    all_flops_rows: list[dict[str, object]] = []
    for key in args.models:
        checkpoint = args.project.resolve() / key / "weights" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = YOLO(str(checkpoint)).model.eval().float().to(torch.device(device))
        latency_rows, flops_rows = benchmark_model(
            model, images, device, args.imgsz, args.warmup, args.flops_images
        )
        summaries.append(
            summarize_benchmark(key, checkpoint, model, latency_rows, flops_rows, device, args.imgsz, args.warmup)
        )
        all_latency_rows.extend({"key": key, **row} for row in latency_rows)
        all_flops_rows.extend({"key": key, **row} for row in flops_rows)

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "checkpoint_benchmark_summary.csv", summaries)
    write_csv(args.out / "checkpoint_latency_samples.csv", all_latency_rows)
    if all_flops_rows:
        write_csv(args.out / "checkpoint_flops_samples.csv", all_flops_rows)
    (args.out / "checkpoint_benchmark_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
