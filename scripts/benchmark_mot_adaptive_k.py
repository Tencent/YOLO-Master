#!/usr/bin/env python3
"""Benchmark baseline, utility, and adaptive-K MoT routing end to end.

The image set is taken from a detection utility matrix so routing quality,
detection mAP, latency percentiles, and actual expert-sample dispatch counts
refer to exactly the same samples.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_mot_cross_domain import load_model, normalize_torch_device, write_csv
from scripts.build_mot_detection_utility import build_validation_loader, resolve_mot_layer
from scripts.evaluate_mot_utility_router import parse_thresholds
from scripts.train_mot_utility_router import file_sha256, read_utility_matrix, verify_matrix_manifest
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.modules.mot import UtilityRouterDeployment


def latency_percentiles(values_ms: list[float]) -> dict[str, float]:
    """Summarize per-image model-forward latency."""
    if not values_ms or not np.isfinite(values_ms).all():
        raise ValueError("latency values must be non-empty and finite")
    values = np.asarray(values_ms, dtype=np.float64)
    return {
        "latency_p50_ms": float(np.quantile(values, 0.50)),
        "latency_p95_ms": float(np.quantile(values, 0.95)),
        "latency_p99_ms": float(np.quantile(values, 0.99)),
        "latency_mean_ms": float(values.mean()),
    }


def subset_loader_factory(
    model: nn.Module,
    data: Path,
    split: str,
    image_ids: list[str],
    *,
    device: torch.device,
    imgsz: int,
    workers: int,
):
    """Return a fresh-loader factory and matching validator preprocessor."""
    preprocessor, base_loader = build_validation_loader(
        model,
        data,
        split,
        device=device,
        imgsz=imgsz,
        workers=workers,
    )
    dataset = base_loader.dataset
    index_by_name = {Path(path).name: index for index, path in enumerate(dataset.im_files)}
    missing = sorted(image_id for image_id in image_ids if image_id not in index_by_name)
    if missing:
        raise ValueError(f"{len(missing)} benchmark images are absent from split {split}: {missing[:3]}")
    subset_indices = [index_by_name[image_id] for image_id in image_ids]
    close = getattr(base_loader, "close", None)
    if callable(close):
        close()

    def factory() -> DataLoader:
        return DataLoader(
            Subset(dataset, subset_indices),
            batch_size=1,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            collate_fn=dataset.collate_fn,
        )

    return factory, preprocessor


def scalar_metrics(results: dict[str, Any]) -> dict[str, float]:
    """Keep JSON-safe scalar detection metrics."""
    output = {}
    for key, value in results.items():
        if isinstance(value, (int, float, np.number)) and math.isfinite(float(value)):
            output[str(key)] = float(value)
    return output


def aggregate_timing_rounds(rows: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate repeated timing rounds without hiding run-to-run spread."""
    if not rows:
        raise ValueError("at least one timing round is required")
    keys = (
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "mean_selected_k",
        "mean_expert_sample_calls",
        "expert_sample_saving_vs_dense",
    )
    output = {key: float(np.median([float(row[key]) for row in rows])) for key in keys}
    for key in ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms"):
        values = [float(row[key]) for row in rows]
        output[f"{key}_run_min"] = min(values)
        output[f"{key}_run_max"] = max(values)
    return output


def evaluate_detection_metrics(
    model: nn.Module,
    loader: DataLoader,
    data: Path,
    split: str,
    *,
    device: str,
    imgsz: int,
    save_dir: Path,
) -> dict[str, float]:
    """Run the standard Ultralytics detection validator on a fixed subset."""
    validator = DetectionValidator(
        dataloader=loader,
        save_dir=save_dir,
        args={
            "data": str(data),
            "split": split,
            "imgsz": imgsz,
            "batch": 1,
            "workers": 0,
            "device": device,
            "rect": False,
            "plots": False,
            "verbose": False,
            "save": False,
            "save_json": False,
        },
    )
    return scalar_metrics(validator(model=model))


def benchmark_forward(
    model: nn.Module,
    block: nn.Module,
    loader: DataLoader,
    preprocessor: DetectionValidator,
    *,
    device: torch.device,
    warmup: int,
    min_warmup_seconds: float,
) -> dict[str, float]:
    """Measure model-only latency and actual target-layer sparse dispatch."""
    model.eval()
    warmup_batch = next(iter(loader))
    warmup_batch = preprocessor.preprocess(warmup_batch)
    warmup_started = time.perf_counter()
    warmup_calls = 0
    with torch.inference_mode():
        while warmup_calls < warmup or time.perf_counter() - warmup_started < min_warmup_seconds:
            _ = model(warmup_batch["img"])
            warmup_calls += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    latencies = []
    selected_k = []
    expert_sample_calls = []
    dense_expert_sample_calls = []
    with torch.inference_mode():
        for batch in loader:
            batch = preprocessor.preprocess(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            _ = model(batch["img"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - start) * 1000.0)
            stats = block._last_dispatch_stats
            selected_k_tensor = stats.get("selected_k")
            selected_k.append(
                float(selected_k_tensor.detach().float().mean())
                if isinstance(selected_k_tensor, torch.Tensor)
                else float(block.top_k)
            )
            expert_sample_calls.append(float(stats.get("expert_sample_calls", block.num_experts)))
            dense_expert_sample_calls.append(float(stats.get("dense_expert_sample_calls", block.num_experts)))
    summary = latency_percentiles(latencies)
    summary.update(
        mean_selected_k=float(np.mean(selected_k)),
        mean_expert_sample_calls=float(np.mean(expert_sample_calls)),
        expert_sample_saving_vs_dense=1.0
        - float(np.sum(expert_sample_calls)) / max(float(np.sum(dense_expert_sample_calls)), 1.0),
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--router-bundle", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--layer", default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--blend-alpha", type=float, default=0.4)
    parser.add_argument("--adaptive-k-thresholds", type=parse_thresholds, default=parse_thresholds("0.35,0.36,0.38"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--min-warmup-seconds", type=float, default=2.0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.blend_alpha <= 1:
        raise SystemExit("--blend-alpha must be in [0,1]")
    if args.warmup < 0 or args.workers < 0 or args.imgsz <= 0:
        raise SystemExit("--warmup and --workers must be non-negative; --imgsz must be positive")
    if args.min_warmup_seconds < 0 or args.rounds <= 0:
        raise SystemExit("--min-warmup-seconds must be non-negative and --rounds must be positive")
    model_path = args.model.expanduser().resolve()
    data_path = args.data.expanduser().resolve()
    matrix_path = args.matrix.expanduser().resolve()
    bundle_path = args.router_bundle.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device_name = normalize_torch_device(args.device)
    device = torch.device(device_name)
    model = load_model(model_path, device_name, nc=10)
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    if bundle.get("format") != "yolo-master-mot-utility-router-v1":
        raise SystemExit("unsupported utility-router bundle format")
    layer_name = args.layer or str(bundle.get("layer_name", ""))
    layer_name, block = resolve_mot_layer(model, layer_name)
    model_sha = file_sha256(model_path)
    if bundle.get("base_checkpoint_sha256") != model_sha or bundle.get("layer_name") != layer_name:
        raise SystemExit("router bundle does not match the model checkpoint and target layer")
    records, expert_names = read_utility_matrix(matrix_path)
    if tuple(bundle.get("expert_names", ())) != expert_names:
        raise SystemExit("router bundle and matrix expert order differ")
    verify_matrix_manifest(matrix_path, model_sha, layer_name, args.split)
    image_ids = [record.image_id for record in records]
    loader_factory, preprocessor = subset_loader_factory(
        model,
        data_path,
        args.split,
        image_ids,
        device=device,
        imgsz=args.imgsz,
        workers=args.workers,
    )

    variants = [("baseline_fixed_k", False, None), ("utility_fixed_k", False, None)]
    variants.extend((f"utility_adaptive_k_{threshold:g}", True, threshold) for threshold in args.adaptive_k_thresholds)

    def deployment_context(variant_name: str, adaptive_k: bool, threshold: float | None):
        return (
            nullcontext()
            if variant_name == "baseline_fixed_k"
            else UtilityRouterDeployment(
                block.router,
                bundle["router_state_dict"],
                alpha=args.blend_alpha,
                adaptive_k=adaptive_k,
                adaptive_k_threshold=threshold or 0.5,
            )
        )

    detection_by_variant = {}
    for variant_name, adaptive_k, threshold in variants:
        with deployment_context(variant_name, adaptive_k, threshold):
            detection_by_variant[variant_name] = evaluate_detection_metrics(
                model,
                loader_factory(),
                data_path,
                args.split,
                device=device_name,
                imgsz=args.imgsz,
                save_dir=output / "validator" / variant_name,
            )
        print(
            f"[adaptive-benchmark] {variant_name} "
            f"mAP50-95={detection_by_variant[variant_name].get('metrics/mAP50-95(B)', float('nan')):.5f}",
            flush=True,
        )
    write_csv(
        output / "detection_metrics.csv",
        [{"variant": name, **metrics} for name, metrics in detection_by_variant.items()],
    )

    timing_rounds = []
    for round_index in range(args.rounds):
        rotated = variants[round_index % len(variants) :] + variants[: round_index % len(variants)]
        for order_index, (variant_name, adaptive_k, threshold) in enumerate(rotated):
            with deployment_context(variant_name, adaptive_k, threshold):
                timing = benchmark_forward(
                    model,
                    block,
                    loader_factory(),
                    preprocessor,
                    device=device,
                    warmup=args.warmup,
                    min_warmup_seconds=args.min_warmup_seconds,
                )
            timing_rounds.append(
                {
                    "round": round_index + 1,
                    "order": order_index + 1,
                    "variant": variant_name,
                    **timing,
                }
            )
            write_csv(output / "latency_rounds.csv", timing_rounds)
            print(
                f"[adaptive-benchmark] round={round_index + 1} order={order_index + 1} "
                f"{variant_name} P50={timing['latency_p50_ms']:.3f}ms "
                f"calls={timing['mean_expert_sample_calls']:.3f}",
                flush=True,
            )

    rows = []
    for variant_name, adaptive_k, threshold in variants:
        variant_timings = [row for row in timing_rounds if row["variant"] == variant_name]
        row = {
            "variant": variant_name,
            "blend_alpha": 0.0 if variant_name == "baseline_fixed_k" else args.blend_alpha,
            "adaptive_k": adaptive_k,
            "adaptive_k_threshold": threshold if threshold is not None else "",
            **detection_by_variant[variant_name],
            **aggregate_timing_rounds(variant_timings),
        }
        rows.append(row)

    baseline_calls = rows[0]["mean_expert_sample_calls"]
    for row in rows:
        row["expert_sample_saving_vs_baseline_fixed_dispatch"] = (
            1.0 - row["mean_expert_sample_calls"] / baseline_calls if baseline_calls > 0 else 0.0
        )

    manifest = {
        "protocol": "same-image end-to-end routing, detection, latency, and dispatch benchmark",
        "model_sha256": model_sha,
        "matrix_sha256": file_sha256(matrix_path),
        "router_bundle_sha256": file_sha256(bundle_path),
        "data_config_name": data_path.name,
        "split": args.split,
        "layer": layer_name,
        "images": len(records),
        "image_size": args.imgsz,
        "blend_alpha": args.blend_alpha,
        "warmup_forwards": args.warmup,
        "minimum_warmup_seconds": args.min_warmup_seconds,
        "timing_rounds": args.rounds,
        "variant_order_rotated_each_round": True,
        "latency_scope": "model forward only, batch=1, synchronized device",
    }
    write_csv(output / "adaptive_k_benchmark.csv", rows)
    write_csv(output / "latency_rounds.csv", timing_rounds)
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[adaptive-benchmark] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
