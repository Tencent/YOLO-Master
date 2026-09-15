#!/usr/bin/env python3
"""Run paired baseline/E3 training-overhead benchmarks for MoE, MoT, and Latent routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = {
    "moe": ROOT / "ultralytics/cfg/models/26/yolo26-master-n.yaml",
    "mot": ROOT / "ultralytics/cfg/models/26/yolo26-master-mot-n.yaml",
    "latent": ROOT / "ultralytics/cfg/models/26/yolo26-master-latent-n.yaml",
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sequence."""
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * quantile
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def paired_bootstrap_ci(values: list[float], *, iterations: int = 10_000, seed: int = 0) -> list[float]:
    """Return a deterministic percentile bootstrap CI over paired overhead percentages."""
    if not values or iterations <= 0:
        raise ValueError("bootstrap requires values and a positive iteration count")
    generator = random.Random(seed)
    samples = [statistics.fmean(generator.choice(values) for _ in values) for _ in range(iterations)]
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def summarize_pairs(
    runs: list[dict[str, Any]], *, threshold_percent: float = 10.0, bootstrap_iterations: int = 10_000, seed: int = 0
) -> dict[str, Any]:
    """Summarize paired run records and evaluate the formal overhead gate."""
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(int(run["pair"]), {})[str(run["condition"])] = run
    if not grouped or any(set(pair) != {"baseline", "e3"} for pair in grouped.values()):
        raise ValueError("every pair must contain exactly one baseline and one e3 run")
    pairs = []
    for pair_index, pair in sorted(grouped.items()):
        baseline = float(pair["baseline"]["mean_milliseconds"])
        e3 = float(pair["e3"]["mean_milliseconds"])
        if baseline <= 0:
            raise ValueError("baseline mean must be positive")
        pairs.append(
            {
                "pair": pair_index,
                "baseline_mean_milliseconds": baseline,
                "e3_mean_milliseconds": e3,
                "overhead_percent": (e3 - baseline) / baseline * 100.0,
            }
        )
    overheads = [pair["overhead_percent"] for pair in pairs]
    ci = paired_bootstrap_ci(overheads, iterations=bootstrap_iterations, seed=seed)
    mean_overhead = statistics.fmean(overheads)
    return {
        "pairs": pairs,
        "mean_overhead_percent": mean_overhead,
        "paired_bootstrap_95_ci_percent": ci,
        "threshold_percent": threshold_percent,
        "passed": mean_overhead < threshold_percent and ci[1] <= threshold_percent,
    }


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label_for(image: Path) -> Path:
    parts = list(image.parts)
    try:
        parts[parts.index("images")] = "labels"
    except ValueError as exc:
        raise ValueError(f"dataset image is not under an images directory: {image}") from exc
    return Path(*parts).with_suffix(".txt")


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_repeated_dataset(data: str, output: Path, *, images: int) -> Path:
    """Create a unique-path COCO-style dataset large enough for one exact training cycle."""
    from ultralytics.data.utils import check_det_dataset

    dataset = check_det_dataset(data, autodownload=True)
    train = Path(str(dataset["train"]))
    sources = sorted(path for path in train.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not sources:
        raise FileNotFoundError(f"no training images found under {train}")
    dataset_root = output / "generated_dataset"
    image_dir = dataset_root / "images/train"
    label_dir = dataset_root / "labels/train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for index in range(images):
        source = sources[index % len(sources)]
        label = _label_for(source)
        stem = f"sample_{index:06d}"
        _link_or_copy(source, image_dir / f"{stem}{source.suffix.lower()}")
        if label.is_file():
            _link_or_copy(label, label_dir / f"{stem}.txt")
    yaml_path = dataset_root / "benchmark.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {"path": str(dataset_root), "train": "images/train", "val": "images/train", "names": dataset["names"]}
        ),
        encoding="utf-8",
    )
    return yaml_path


def _image_files(source: str | list[str]) -> list[Path]:
    """Resolve image files from a dataset directory, text manifest, or source list."""
    sources = source if isinstance(source, list) else [source]
    images: list[Path] = []
    for item in sources:
        path = Path(item)
        if path.is_dir():
            images.extend(candidate for candidate in path.rglob("*") if candidate.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file() and path.suffix.lower() == ".txt":
            base = path.parent
            for line in path.read_text(encoding="utf-8").splitlines():
                candidate = Path(line.strip())
                if not candidate.is_absolute():
                    candidate = (base / candidate).resolve()
                if candidate.suffix.lower() in IMAGE_SUFFIXES:
                    images.append(candidate)
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
    return sorted(set(images))


def prepare_benchmark_dataset(data: str, output: Path, *, images: int, native: bool) -> tuple[Path, int]:
    """Return a benchmark dataset and its real training-image count."""
    if not native:
        return prepare_repeated_dataset(data, output, images=images), images

    from ultralytics.data.utils import check_det_dataset

    dataset = check_det_dataset(data, autodownload=True)
    sources = dataset.get("train")
    if not isinstance(sources, (str, list)):
        raise TypeError("dataset train split must resolve to a path or list of paths")
    train_images = _image_files(sources)
    if not train_images:
        raise FileNotFoundError(f"no training images resolved from native dataset {data!r}")
    source_yaml = Path(data)
    if not source_yaml.is_file():
        source_yaml = ROOT / "ultralytics/cfg/datasets" / data
    if not source_yaml.is_file():
        raise FileNotFoundError(f"native benchmark requires a concrete dataset YAML: {data}")
    return source_yaml.resolve(), len(train_images)


def parse_model_overrides(overrides: list[str]) -> dict[str, Path]:
    """Return the three family model paths after applying FAMILY=PATH overrides."""
    models = dict(DEFAULT_MODELS)
    for raw in overrides:
        family, separator, value = raw.partition("=")
        if not separator or family not in models or not value:
            raise ValueError(f"model override must be FAMILY=PATH for one of {sorted(models)}: {raw!r}")
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        models[family] = path
    return models


def _run_summary(
    telemetry_path: Path, *, family: str, pair: int, condition: str, event_files: list[Path]
) -> dict[str, Any]:
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    rank = telemetry["ranks"][0]
    steps = rank["instrumented_steps"]
    raw = steps["raw_milliseconds"]
    routing = rank["routing"]
    last = routing.get("last", {})
    families = sorted({layer.get("family") for layer in last.get("layers", {}).values()})
    gate_errors = []
    if len(raw or []) != int(steps["count"]):
        gate_errors.append("raw step count mismatch")
    if condition == "baseline" and routing["observations"] != 0:
        gate_errors.append("baseline produced routing observations")
    if condition == "e3":
        if routing["observations"] <= 0:
            gate_errors.append("E3 produced no routing observations")
        if family not in families:
            gate_errors.append(f"target family {family!r} absent from {families}")
        if int(last.get("invalid_layers", -1)) != 0 or int(last.get("unsupported_layers", -1)) != 0:
            gate_errors.append("E3 snapshot contains invalid or unsupported layers")
    if not event_files:
        gate_errors.append("TensorBoard event file missing")
    return {
        "family": family,
        "pair": pair,
        "condition": condition,
        "count": steps["count"],
        "samples": steps["samples"],
        "mean_milliseconds": steps["mean_milliseconds"],
        "median_milliseconds": steps["p50_milliseconds"],
        "p95_milliseconds": steps["p95_milliseconds"],
        "samples_per_second": steps["samples_per_second"],
        "raw_milliseconds": raw,
        "memory": rank["memory"],
        "routing_observations": routing["observations"],
        "routing_families": families,
        "invalid_layers": last.get("invalid_layers") if condition == "e3" else None,
        "unsupported_layers": last.get("unsupported_layers") if condition == "e3" else None,
        "snapshot_bytes_max": routing["snapshot_bytes_max"],
        "telemetry_sha256": _sha256(telemetry_path),
        "tensorboard_event_sha256": [_sha256(path) for path in event_files],
        "gate_errors": gate_errors,
    }


def _train_command(args: argparse.Namespace, *, model: Path, data: Path, name: str, epochs: int) -> list[str]:
    return [
        str(ROOT / ".venv/bin/yolo"),
        "train",
        f"model={model}",
        f"data={data}",
        f"epochs={epochs}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"device={args.device}",
        f"workers={args.workers}",
        f"seed={args.seed}",
        f"amp={args.amp}",
        "pretrained=False",
        "val=False",
        "plots=False",
        "save=False",
        "verbose=False",
        f"project={args.output / 'runs'}",
        f"name={name}",
        "exist_ok=True",
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the requested paired benchmark matrix and persist raw and summarized evidence."""
    if args.rounds <= 0 or args.warmup < 0 or args.steps <= 0 or args.batch <= 0 or args.routing_interval <= 0:
        raise ValueError("rounds, steps, batch, and routing interval must be positive; warmup must be non-negative")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    args.output = output
    dirty = bool(_git("status", "--porcelain"))
    if dirty and not args.allow_dirty:
        raise RuntimeError("formal benchmark requires a clean worktree; use --allow-dirty only for a dry-run")
    total_steps = args.warmup + args.steps
    dataset, train_images = prepare_benchmark_dataset(
        args.data, output, images=total_steps * args.batch, native=args.native_data
    )
    steps_per_epoch = math.ceil(train_images / args.batch)
    epochs = math.ceil(total_steps / steps_per_epoch)
    models = parse_model_overrides(args.model)
    config_dir = output / "ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "YOLO_CONFIG_DIR": str(config_dir)}
    subprocess.run(
        [str(ROOT / ".venv/bin/yolo"), "settings", "tensorboard=True", "wandb=False"],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    records = []
    for family in args.families:
        model = models[family]
        for pair in range(args.rounds):
            order = ("baseline", "e3") if pair % 2 == 0 else ("e3", "baseline")
            for condition in order:
                name = f"{family}_pair{pair + 1:02d}_{condition}"
                run_dir = output / "runs" / name
                run_env = {
                    **environment,
                    "YOLO_TRAIN_TELEMETRY": "1",
                    "YOLO_TRAIN_TELEMETRY_ROUTING_ENABLED": "1" if condition == "e3" else "0",
                    "YOLO_TRAIN_TELEMETRY_ROUTING_INTERVAL": str(args.routing_interval),
                    "YOLO_TRAIN_TELEMETRY_WARMUP_STEPS": str(args.warmup),
                    "YOLO_TRAIN_TELEMETRY_MAX_STEPS": str(args.steps),
                    "YOLO_TRAIN_TELEMETRY_RAW_STEPS": "1",
                }
                completed = subprocess.run(
                    _train_command(args, model=model, data=dataset, name=name, epochs=epochs),
                    cwd=ROOT,
                    env=run_env,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                (output / f"{name}.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
                if completed.returncode:
                    raise RuntimeError(
                        f"{name} failed with exit code {completed.returncode}; see {output / f'{name}.log'}"
                    )
                telemetry_path = run_dir / "telemetry.json"
                record = _run_summary(
                    telemetry_path,
                    family=family,
                    pair=pair + 1,
                    condition=condition,
                    event_files=sorted(run_dir.glob("events.out.tfevents.*")),
                )
                records.append(record)
                if record["gate_errors"]:
                    raise RuntimeError(f"{name} evidence gate failed: {record['gate_errors']}")
                print(f"completed {name}: {record['mean_milliseconds']:.3f} ms")
    family_summaries = {
        family: summarize_pairs(
            [record for record in records if record["family"] == family],
            threshold_percent=args.threshold,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        for family in args.families
    }
    payload = {
        "schema_version": "e3.overhead_benchmark.v1",
        "git": {"commit": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current"), "dirty": dirty},
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "device": args.device,
            "data": args.data,
            "native_data": args.native_data,
            "train_images": train_images,
            "epochs": epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "amp": args.amp,
            "workers": args.workers,
            "seed": args.seed,
            "warmup_steps": args.warmup,
            "measured_steps": args.steps,
            "routing_interval_steps": args.routing_interval,
            "rounds": args.rounds,
            "tensorboard": True,
        },
        "runs": records,
        "families": family_summaries,
        "passed": all(summary["passed"] for summary in family_summaries.values()),
    }
    (output / "raw_runs.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="coco8.yaml")
    parser.add_argument("--families", nargs="+", choices=sorted(DEFAULT_MODELS), default=sorted(DEFAULT_MODELS))
    parser.add_argument("--device", default="0")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--routing-interval", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("runs/e3_overhead_cuda"))
    parser.add_argument("--native-data", action="store_true", help="use the dataset train split without duplication")
    parser.add_argument(
        "--model", action="append", default=[], metavar="FAMILY=PATH", help="override a family model or checkpoint"
    )
    parser.add_argument(
        "--allow-dirty", action="store_true", help="permit an uncommitted worktree for non-formal dry-runs"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args)
    print(json.dumps({"passed": payload["passed"], "families": payload["families"]}, indent=2))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
