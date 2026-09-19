"""Run the real MoE/MoT/Latent P0 evidence workflow on a fixed coco8 sample set."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from .aggregation import aggregate_events, compare_batch_aggregates, compare_repeated_runs
from .collector import RoutingCollector
from .io_utils import environment, sha256_file, write_json, write_jsonl, write_manifest
from .plotting import (
    save_batch_consistency_plot,
    save_cross_family_plot,
    save_family_plot,
    save_sample_variation_plot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _load_config(config_path: Path, run_id_override: str | None = None) -> dict[str, Any]:
    """Load config and apply a path-safe run id override."""

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"Config must contain a mapping: {config_path}")
    config = dict(loaded)
    run_id = str(run_id_override if run_id_override is not None else config.get("run_id", "")).strip()
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must be 1-128 characters and contain only letters, digits, '.', '_' or '-' "
            "without path separators"
        )
    config["run_id"] = run_id
    indices = config.get("sample_indices")
    if indices is None and "sample_index" in config:
        indices = [config["sample_index"]]
    if not isinstance(indices, list) or not indices:
        raise ValueError("sample_indices must be a non-empty list")
    config["sample_indices"] = [int(index) for index in indices]
    if len(set(config["sample_indices"])) != len(config["sample_indices"]):
        raise ValueError("sample_indices must not contain duplicates")
    primary_batch_size = int(config.get("primary_batch_size", 1))
    boundary_batch_sizes = [int(value) for value in config.get("boundary_batch_sizes", [])]
    if primary_batch_size != 1:
        raise ValueError("primary_batch_size must remain 1 so per-sample evidence is retained")
    if any(value < 1 for value in boundary_batch_sizes):
        raise ValueError("boundary_batch_sizes must contain positive integers")
    config["primary_batch_size"] = primary_batch_size
    config["boundary_batch_sizes"] = sorted(set(boundary_batch_sizes))
    if int(config.get("repeat_primary_passes", 2)) < 2:
        raise ValueError("repeat_primary_passes must be at least 2")
    return config


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("e3_p0")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(path, encoding="utf-8", mode="w")
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def _resolve_images(dataset_name: str, split: str, sample_indices: list[int]) -> tuple[list[Path], dict[str, Any]]:
    from ultralytics.data.utils import check_det_dataset

    dataset = check_det_dataset(dataset_name, autodownload=True)
    roots = dataset[split] if isinstance(dataset[split], list) else [dataset[split]]
    extensions = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    images: list[Path] = []
    for item in roots:
        root = Path(item)
        if root.is_file() and root.suffix.lower() == ".txt":
            images.extend(Path(line.strip()) for line in root.read_text(encoding="utf-8").splitlines() if line.strip())
        elif root.is_file():
            images.append(root)
        elif root.is_dir():
            images.extend(path for path in root.rglob("*") if path.suffix.lower() in extensions)
    images = sorted(path.resolve() for path in images)
    if not images:
        raise FileNotFoundError(f"No image found for dataset={dataset_name!r}, split={split!r}")
    invalid = [index for index in sample_indices if not 0 <= index < len(images)]
    if invalid:
        raise IndexError(f"sample_indices={invalid} outside [0, {len(images) - 1}]")
    return [images[index] for index in sample_indices], {
        "dataset_root": str(dataset.get("path")),
        "split_image_count": len(images),
        "selected_indices": sample_indices,
    }


def _load_tensor(path: Path, image_size: int, torch_module: Any) -> tuple[Any, dict[str, Any]]:
    image = Image.open(path).convert("RGB")
    original = list(image.size)
    resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch_module.from_numpy(array).contiguous(), {
        "path": str(path),
        "name": path.name,
        "sha256": sha256_file(path),
        "original_size_wh": original,
        "input_size_wh": [image_size, image_size],
        "normalization": "RGB uint8 / 255.0",
    }


def _validate_events(events: list[dict[str, Any]]) -> None:
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError("jsonschema is required for P0 contract validation") from exc
    schema = json.loads((PROJECT_ROOT / "schemas" / "routing-event.schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    errors = []
    for index, event in enumerate(events):
        errors.extend(f"event[{index}] {error.message}" for error in validator.iter_errors(event))
    if errors:
        raise RuntimeError("Schema validation failed:\n" + "\n".join(errors))


def _batched_indices(count: int, batch_size: int) -> list[list[int]]:
    return [list(range(start, min(start + batch_size, count))) for start in range(0, count, batch_size)]


def _collect_pass(
    *,
    model: Any,
    family: str,
    run_id: str,
    tensors: list[Any],
    input_meta: list[dict[str, Any]],
    sample_indices: list[int],
    batch_size: int,
    pass_name: str,
    device: str,
    predicate: Any,
    torch_module: Any,
) -> tuple[list[dict[str, Any]], list[str], list[float], str]:
    collector = RoutingCollector(family=family, run_id=run_id)
    registered = collector.register(model, predicate=predicate)
    if not registered:
        raise RuntimeError(f"No routed modules discovered for family={family}")
    durations = []
    output_type = "unknown"
    try:
        for batch_index, positions in enumerate(_batched_indices(len(tensors), batch_size)):
            batch = torch_module.stack([tensors[position] for position in positions], dim=0).to(device)
            selected_indices = [sample_indices[position] for position in positions]
            collector.set_context(
                pass_name=pass_name,
                batch_index=batch_index,
                batch_size=len(positions),
                sample_indices=selected_indices,
                input_ids=[input_meta[position]["sha256"][:16] for position in positions],
            )
            started = time.perf_counter()
            with torch_module.inference_mode():
                output = model(batch)
            durations.append((time.perf_counter() - started) * 1000.0)
            output_type = type(output).__name__
            del batch, output
    finally:
        collector.remove()
    if collector.handles:
        raise RuntimeError(f"Hook leak detected for family={family} pass={pass_name}")
    if not collector.events:
        raise RuntimeError(f"No routing events captured for family={family} pass={pass_name}")
    _validate_events(collector.events)
    return collector.events, registered, durations, output_type


def _input_set_sha256(input_meta: list[dict[str, Any]]) -> str:
    payload = "\n".join(f"{item['sample_index']}:{item['sha256']}" for item in input_meta).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run(config_path: Path, *, run_id: str | None = None, update_latest: bool = True) -> Path:
    config = _load_config(config_path, run_id)
    run_dir = PROJECT_ROOT / "artifacts" / "p0" / str(config["run_id"])
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty evidence directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _logger(run_dir / "full.log")
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (run_dir / "command.txt").write_text("run_p0.cmd\n", encoding="utf-8")

    source_root = (PROJECT_ROOT / config["runtime_root"]).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"runtime_root does not exist: {source_root}")
    sys.path.insert(0, str(source_root))
    os.environ["MOE_SNAPSHOT_INTERVAL"] = str(int(config.get("moe_snapshot_interval", 1)))
    os.chdir(PROJECT_ROOT)

    import torch
    from ultralytics import YOLO
    from ultralytics.nn.modules.routing_protocol import is_routed_module

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info("scope=E3 P0 real three-family multi-sample evidence run")
    logger.info("source_ref=%s source_tree=%s", config["official_runtime_ref"], config["official_runtime_tree"])
    logger.info("device=%s seed=%d selection_policy=%s", config["device"], seed, config["selection_policy"])

    paths, dataset_meta = _resolve_images(config["dataset"], config["dataset_split"], config["sample_indices"])
    tensors = []
    inputs = []
    for sample_index, path in zip(config["sample_indices"], paths):
        tensor, metadata = _load_tensor(path, int(config["image_size"]), torch)
        metadata["sample_index"] = sample_index
        tensors.append(tensor)
        inputs.append(metadata)
        logger.info("input_index=%d input=%s sha256=%s", sample_index, path, metadata["sha256"])
    input_record = {
        **dataset_meta,
        "dataset": config["dataset"],
        "split": config["dataset_split"],
        "selected_image_count": len(inputs),
        "input_set_sha256": _input_set_sha256(inputs),
        "images": inputs,
    }
    write_json(run_dir / "input.json", input_record)

    primary_events: dict[str, list[dict[str, Any]]] = {}
    repeat_events: dict[str, list[dict[str, Any]]] = {}
    boundary_events: dict[int, dict[str, list[dict[str, Any]]]] = {
        batch_size: {} for batch_size in config["boundary_batch_sizes"]
    }
    family_summaries: dict[str, Any] = {}
    reproducibility: dict[str, Any] = {}
    subtitle = (
        f"coco8 val indices {config['sample_indices']}, random initialization, CPU, "
        f"seed={seed}, imgsz={config['image_size']}"
    )

    for family, profile in config["profiles"].items():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model_config = source_root / profile["model_config"]
        wrapper = YOLO(model_config)
        model = wrapper.model.to(config["device"]).eval()
        primary, registered, primary_durations, output_type = _collect_pass(
            model=model,
            family=family,
            run_id=config["run_id"],
            tensors=tensors,
            input_meta=inputs,
            sample_indices=config["sample_indices"],
            batch_size=config["primary_batch_size"],
            pass_name="primary",
            device=config["device"],
            predicate=is_routed_module,
            torch_module=torch,
        )
        repeated, repeat_registered, repeat_durations, _ = _collect_pass(
            model=model,
            family=family,
            run_id=config["run_id"],
            tensors=tensors,
            input_meta=inputs,
            sample_indices=config["sample_indices"],
            batch_size=config["primary_batch_size"],
            pass_name="repeat-2",
            device=config["device"],
            predicate=is_routed_module,
            torch_module=torch,
        )
        if registered != repeat_registered:
            raise RuntimeError(f"Routed module discovery changed between repeated passes for family={family}")
        repeat_result = compare_repeated_runs(primary, repeated)
        reproducibility[family] = repeat_result
        if repeat_result["status"] != "PASS":
            raise RuntimeError(f"Deterministic repeat failed for family={family}: {repeat_result['mismatches'][:1]}")

        primary_events[family] = primary
        repeat_events[family] = repeated
        write_jsonl(run_dir / f"routing-{family}.jsonl", primary)
        write_jsonl(run_dir / f"routing-repeat-{family}.jsonl", repeated)
        save_family_plot(family, primary, run_dir / f"routing-{family}.png", subtitle)

        for batch_size in config["boundary_batch_sizes"]:
            candidate, candidate_registered, _, _ = _collect_pass(
                model=model,
                family=family,
                run_id=config["run_id"],
                tensors=tensors,
                input_meta=inputs,
                sample_indices=config["sample_indices"],
                batch_size=batch_size,
                pass_name=f"batch-{batch_size}",
                device=config["device"],
                predicate=is_routed_module,
                torch_module=torch,
            )
            if registered != candidate_registered:
                raise RuntimeError(f"Routed module discovery changed for family={family} batch_size={batch_size}")
            boundary_events[batch_size][family] = candidate
            write_jsonl(run_dir / f"routing-batch{batch_size}-{family}.jsonl", candidate)

        parameters = sum(parameter.numel() for parameter in model.parameters())
        aggregates = aggregate_events(primary)
        family_summaries[family] = {
            "status": "PASS",
            "model_config": profile["model_config"],
            "model_parameters": parameters,
            "registered_modules": registered,
            "routed_module_count": len(registered),
            "selected_image_count": len(inputs),
            "primary_forward_count": len(primary_durations),
            "primary_captured_events": len(primary),
            "repeat_captured_events": len(repeated),
            "hooks_removed": True,
            "output_type": output_type,
            "primary_forward_ms_observation_only": primary_durations,
            "repeat_forward_ms_observation_only": repeat_durations,
            "entropy_normalized_range": [
                min(item["per_forward_metric_stats"]["entropy_normalized"]["min"] for item in aggregates),
                max(item["per_forward_metric_stats"]["entropy_normalized"]["max"] for item in aggregates),
            ],
            "load_gini_range": [
                min(item["per_forward_metric_stats"]["load_gini"]["min"] for item in aggregates),
                max(item["per_forward_metric_stats"]["load_gini"]["max"] for item in aggregates),
            ],
            "deterministic_repeat": repeat_result["status"],
        }
        logger.info(
            "family=%s modules=%d images=%d primary_events=%d repeat_match=%s params=%d",
            family,
            len(registered),
            len(inputs),
            len(primary),
            repeat_result["status"],
            parameters,
        )
        del model, wrapper

    combined = [event for family in config["profiles"] for event in primary_events[family]]
    repeated_combined = [event for family in config["profiles"] for event in repeat_events[family]]
    _validate_events(combined)
    _validate_events(repeated_combined)
    write_jsonl(run_dir / "routing-all.jsonl", combined)
    write_jsonl(run_dir / "routing-repeat-all.jsonl", repeated_combined)
    aggregates = aggregate_events(combined)
    write_json(run_dir / "routing-aggregate.json", aggregates)
    save_cross_family_plot(primary_events, run_dir / "routing-cross-family.png", subtitle)
    save_sample_variation_plot(primary_events, run_dir / "routing-sample-variation.png", subtitle)

    tolerance = float(config["batch_consistency_tolerance"])
    batch_comparisons = []
    for batch_size in config["boundary_batch_sizes"]:
        candidate_combined = [event for family in config["profiles"] for event in boundary_events[batch_size][family]]
        comparison = compare_batch_aggregates(
            combined,
            candidate_combined,
            candidate_batch_size=batch_size,
            tolerance=tolerance,
        )
        batch_comparisons.append(comparison)
        if comparison["status"] != "PASS":
            raise RuntimeError(
                f"Batch consistency failed for batch_size={batch_size}: max_delta={comparison['max_abs_load_delta']}"
            )
    write_json(run_dir / "reproducibility.json", reproducibility)
    write_json(run_dir / "batch-consistency.json", batch_comparisons)
    save_batch_consistency_plot(batch_comparisons, run_dir / "routing-batch-consistency.png")
    write_json(
        run_dir / "environment.json",
        environment(torch, source_root, PROJECT_ROOT, config["official_runtime_ref"], config["official_runtime_tree"]),
    )
    summary = {
        "status": "PASS",
        "scope": "E3 P0 multi-sample unified routing evidence for real MoE/MoT/Latent forwards",
        "run_id": config["run_id"],
        "schema_version": config["schema_version"],
        "official_locked_base_ref": config["official_locked_base_ref"],
        "official_runtime_ref": config["official_runtime_ref"],
        "official_runtime_tree": config["official_runtime_tree"],
        "official_source_archive_url": config["official_source_archive_url"],
        "official_source_archive_sha256": config["official_source_archive_sha256"],
        "input": input_record,
        "coverage": {
            "selected_images": len(inputs),
            "sample_indices": config["sample_indices"],
            "primary_batch_size": config["primary_batch_size"],
            "boundary_batch_sizes": config["boundary_batch_sizes"],
            "primary_passes": config["repeat_primary_passes"],
        },
        "families": family_summaries,
        "total_events": len(combined),
        "total_repeat_events": len(repeated_combined),
        "aggregate_module_records": len(aggregates),
        "schema_validation": "PASS",
        "deterministic_repetition": {
            "status": "PASS",
            "matching_events": sum(item["matching_events"] for item in reproducibility.values()),
            "compared_events": sum(item["reference_events"] for item in reproducibility.values()),
        },
        "batch_consistency": {
            "status": "PASS",
            "tolerance": tolerance,
            "comparisons": [
                {
                    "candidate_batch_size": item["candidate_batch_size"],
                    "max_abs_load_delta": item["max_abs_load_delta"],
                }
                for item in batch_comparisons
            ],
        },
        "limitations": [
            "Models are randomly initialized; routing distributions are pipeline evidence, not trained-quality claims.",
            "The four coco8 validation images improve coverage but do not support population-level specialization claims.",
            "Forward durations remain observations only and are not the P1 paired training-overhead benchmark.",
            "P0 covers MoE, MoT and Latent; realtime panels and image overlays remain P1/P2 work.",
        ],
    }
    write_json(run_dir / "summary.json", summary)
    write_manifest(run_dir)
    if update_latest:
        (PROJECT_ROOT / "artifacts" / "p0" / "LATEST.txt").write_text(config["run_id"] + "\n", encoding="utf-8")
    logger.info(
        "result=PASS primary_events=%d repeat_events=%d batch_consistency=PASS artifacts=%s",
        len(combined),
        len(repeated_combined),
        run_dir,
    )
    write_manifest(run_dir)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "p0" / "p0.yaml")
    parser.add_argument(
        "--run-id",
        help="Path-safe output id. Overrides config run_id; existing non-empty output directories are never overwritten.",
    )
    parser.add_argument(
        "--no-update-latest",
        action="store_true",
        help="Do not change artifacts/p0/LATEST.txt. Use this for local reproduction runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config.resolve(), run_id=args.run_id, update_latest=not args.no_update_latest)
