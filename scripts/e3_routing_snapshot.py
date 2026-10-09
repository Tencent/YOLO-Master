#!/usr/bin/env python3
"""Run the Rhino-Bird E3 cross-family routing admission smoke on one COCO8 image."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.routing_interpreter import _load_batch, _load_model
from ultralytics import __version__ as ultralytics_version
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.routing_interpreter import RoutingHeatmap, RoutingInterpreter
from ultralytics.utils.routing_telemetry import normalize_routing_layer, routing_module_family

DEFAULT_MODELS = {
    "moe": ROOT / "ultralytics/cfg/models/26/yolo26-master-n.yaml",
    "mot": ROOT / "ultralytics/cfg/models/26/yolo26-master-mot-n.yaml",
    "latent": ROOT / "ultralytics/cfg/models/26/yolo26-master-latent-n.yaml",
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    """Build the E3 smoke command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="coco8.yaml", help="dataset YAML used to resolve a real validation image")
    parser.add_argument("--image", type=Path, help="optional image override; defaults to the first validation image")
    parser.add_argument("--imgsz", type=int, default=320, help="square inference size")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or an explicit torch device")
    parser.add_argument("--warmup", type=int, default=1, help="unmeasured forward passes per family")
    parser.add_argument("--iterations", type=int, default=3, help="timed forward passes per family")
    parser.add_argument("--seed", type=int, default=0, help="explicit model-initialization seed")
    parser.add_argument("--output", type=Path, default=Path("runs/e3_routing_observability"), help="artifact directory")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="FAMILY=PATH",
        help="override a family model YAML; may be repeated",
    )
    return parser


def _git(*args: str) -> str:
    """Return one Git command result without shell interpolation."""
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    """Return a file SHA-256 checksum."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_device(requested: str) -> torch.device:
    """Resolve an explicit device or select the best available local backend."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _synchronize(device: torch.device) -> None:
    """Synchronize asynchronous accelerators before reading a timer."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _timed(callable_, *, iterations: int, device: torch.device) -> list[float]:
    """Measure repeated calls in milliseconds with accelerator synchronization."""
    samples = []
    for _ in range(iterations):
        _synchronize(device)
        started = time.perf_counter_ns()
        callable_()
        _synchronize(device)
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    return samples


def _timing_summary(samples: list[float]) -> dict[str, float]:
    """Return JSON-safe latency statistics."""
    return {
        "mean_ms": statistics.fmean(samples),
        "std_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _parse_models(overrides: list[str]) -> dict[str, Path]:
    """Merge validated FAMILY=PATH overrides into the three required profiles."""
    models = dict(DEFAULT_MODELS)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--model must use FAMILY=PATH, got {item!r}")
        family, raw_path = item.split("=", 1)
        family = family.strip().lower()
        if family not in models:
            raise ValueError(f"unsupported family {family!r}; expected one of {sorted(models)}")
        models[family] = Path(raw_path).expanduser()
    resolved = {family: (path if path.is_absolute() else ROOT / path).resolve() for family, path in models.items()}
    missing = [str(path) for path in resolved.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"model YAMLs not found: {missing}")
    return resolved


def _resolve_image(data: str, override: Path | None) -> tuple[Path, dict[str, str]]:
    """Resolve one genuine dataset image and retain the dataset paths as evidence."""
    dataset = check_det_dataset(data, autodownload=True)
    evidence = {key: str(dataset[key]) for key in ("path", "train", "val") if key in dataset}
    if override is not None:
        image = override.expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(f"image not found: {image}")
        return image, evidence

    val = dataset.get("val")
    candidates: list[Path] = []
    values = val if isinstance(val, (list, tuple)) else [val]
    for value in values:
        path = Path(str(value))
        if path.is_dir():
            candidates.extend(file for file in path.rglob("*") if file.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"no validation images resolved from {data!r}: {val!r}")
    return min(candidates).resolve(), evidence


def _layer_payload(
    name: str,
    module: torch.nn.Module,
    heatmap: RoutingHeatmap,
) -> dict[str, Any]:
    """Normalize one routed layer to the shared E3 schema."""
    return normalize_routing_layer(name, module, probabilities=heatmap.probabilities)


def _save_family_figure(family: str, layers: list[dict[str, Any]], path: Path) -> None:
    """Save a compact layer-by-expert usage matrix for one routing family."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    columns = max(layer["num_experts"] for layer in layers)
    matrix = np.full((len(layers), columns), np.nan, dtype=float)
    for row, layer in enumerate(layers):
        matrix[row, : layer["num_experts"]] = layer["expert_usage"]
    width = max(6.4, columns * 1.0)
    height = max(3.4, len(layers) * 0.52 + 1.8)
    figure, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    axis.set_xticks(range(columns), [f"E{index}" for index in range(columns)])
    axis.set_yticks(range(len(layers)), [layer["layer_name"] for layer in layers])
    axis.set_xlabel("Expert")
    axis.set_ylabel("Routed layer")
    axis.set_title(f"E3 routing snapshot - {family.upper()} expert usage")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if not np.isnan(value):
                color = "white" if value < 0.25 or value > 0.75 else "black"
                axis.text(column, row, f"{value:.3f}", ha="center", va="center", color=color, fontsize=8)
    figure.colorbar(image, ax=axis, label="Mean routing probability")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _portable_dataset_evidence(dataset: dict[str, str], image: Path) -> tuple[dict[str, str], str]:
    """Remove host-specific prefixes while retaining dataset-relative provenance."""
    dataset_root = Path(dataset["path"]).resolve()
    dataset_parent = dataset_root.parent
    portable = {}
    for key, value in dataset.items():
        path = Path(value).resolve()
        try:
            portable[key] = str(path.relative_to(dataset_parent))
        except ValueError:
            portable[key] = path.name
    try:
        sample = str(image.resolve().relative_to(dataset_root))
    except ValueError:
        sample = image.name
    return portable, sample


def _repo_relative_or_name(path: Path) -> str:
    """Return a repository-relative path or only the basename for external paths."""
    try:
        return str(path.absolute().relative_to(ROOT))
    except ValueError:
        return path.name


def _environment(device: torch.device, dataset: dict[str, str], image: Path) -> dict[str, Any]:
    """Collect the minimum pinned environment evidence required by the admission gate."""
    portable_dataset, portable_image = _portable_dataset_evidence(dataset, image)
    executable = Path(sys.executable)
    module_path = Path(sys.modules["ultralytics"].__file__).resolve()
    return {
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "remote": _git("remote", "get-url", "origin"),
        },
        "python": {
            "version": platform.python_version(),
            "executable": _repo_relative_or_name(executable),
        },
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "torch": {
            "version": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "ultralytics": {
            "version": ultralytics_version,
            "module": _repo_relative_or_name(module_path),
        },
        "dataset": portable_dataset,
        "sample": {"path": portable_image, "sha256": _sha256(image)},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the three required families and return the unified snapshot payload."""
    if args.imgsz <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("imgsz and iterations must be positive; warmup must be non-negative")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = _select_device(args.device)
    models = _parse_models(args.model)
    image, dataset = _resolve_image(args.data, args.image)
    environment = _environment(device, dataset, image)
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")

    payload: dict[str, Any] = {
        "schema_version": "e3.routing_snapshot.v1",
        "scope": "Rhino-Bird 2026-08-24 admission smoke",
        "dataset": args.data,
        "sample": environment["sample"],
        "imgsz": args.imgsz,
        "device": str(device),
        "seed": args.seed,
        "weights": "random_initialization_from_yaml",
        "families": {},
        "limitations": [
            "This smoke validates collection, schema, and visualization plumbing; random YAML initialization is not a trained routing-quality result.",
            "Auxiliary routing losses are training-only and are reported as configured_inactive_eval during this inference smoke.",
            "This admission snapshot records only the selected runtime device.",
        ],
    }

    for family, model_path in models.items():
        print(f"[E3] family={family} model={model_path.relative_to(ROOT)} device={device}", flush=True)
        torch.manual_seed(args.seed)
        network = _load_model(model_path, device, half=False)
        batch = _load_batch(image, args.imgsz, device, torch.float32)
        interpreter = RoutingInterpreter(network)

        def baseline_forward(current_network=network, current_batch=batch):
            """Run one inference-only baseline forward for timing."""
            with torch.no_grad():
                return current_network(current_batch)

        def captured_forward(current_interpreter=interpreter, current_batch=batch):
            """Run one routing-captured forward for timing."""
            return current_interpreter.capture_routing(current_batch)

        for _ in range(args.warmup):
            baseline_forward()
        baseline_samples = _timed(baseline_forward, iterations=args.iterations, device=device)
        capture_samples = _timed(captured_forward, iterations=args.iterations, device=device)
        heatmaps = interpreter.capture_routing(batch)
        modules = {name or "<root>": module for name, module in network.named_modules()}
        layers = [
            _layer_payload(name, modules[name], heatmap)
            for name, heatmap in heatmaps.items()
            if name in modules and routing_module_family(modules[name]) == family
        ]
        if not layers:
            captured = sorted({routing_module_family(modules[name]) for name in heatmaps if name in modules})
            raise RuntimeError(f"{family} profile captured no {family} layers; captured families={captured}")
        figure_path = output / f"{family}_expert_usage.png"
        _save_family_figure(family, layers, figure_path)
        baseline = _timing_summary(baseline_samples)
        instrumented = _timing_summary(capture_samples)
        overhead = 100.0 * (instrumented["mean_ms"] - baseline["mean_ms"]) / max(baseline["mean_ms"], 1e-12)
        payload["families"][family] = {
            "model": str(model_path.relative_to(ROOT)),
            "model_sha256": _sha256(model_path),
            "captured_layers": len(layers),
            "layers": layers,
            "static_figure": figure_path.name,
            "overhead": {
                "method": "same-process warm forward vs hook registration + captured forward",
                "warmup": args.warmup,
                "iterations": args.iterations,
                "baseline": baseline,
                "instrumented": instrumented,
                "overhead_percent": overhead,
                "visualization_io_excluded": True,
            },
        }
        del interpreter, network, batch, heatmaps
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    report_path = output / "route_stats.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[E3] snapshot={report_path}", flush=True)
    return payload


def main(argv: list[str] | None = None) -> int:
    """Run the command-line entry point."""
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
