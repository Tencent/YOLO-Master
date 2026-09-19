"""Generate five-family feasibility evidence and truthful MoT/MoA overlays."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from .capture import SpatialRouterCollector, max_output_delta, routing_diagnostics, routing_metric_fields, tensor_shapes
from .geometry import LetterboxMeta, letterbox
from .io_utils import environment, sha256_file, write_json, write_manifest
from .plotting import (
    save_dominant_overlay,
    save_ground_truth_overlay,
    save_metric_overlay,
    save_overview,
    save_probability_overlay,
)
from .regions import (
    YoloBox,
    aggregate_region_diagnostics,
    label_path_for_image,
    parse_yolo_labels,
    region_routing_diagnostics,
    token_region_masks,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _load_config(config_path: Path, run_id_override: str | None = None) -> dict[str, Any]:
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"config must contain a mapping: {config_path}")
    config = dict(loaded)
    run_id = str(run_id_override if run_id_override is not None else config.get("run_id", "")).strip()
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must be path-safe and contain at most 128 letters, digits, dots, dashes or underscores"
        )
    config["run_id"] = run_id
    indices = config.get("sample_indices")
    if not isinstance(indices, list) or not indices:
        raise ValueError("sample_indices must be a non-empty list")
    config["sample_indices"] = [int(index) for index in indices]
    if len(set(config["sample_indices"])) != len(config["sample_indices"]):
        raise ValueError("sample_indices must not contain duplicates")
    if set(config.get("spatial_profiles", {})) != {"mot", "moa"}:
        raise ValueError("P2 spatial_profiles must be exactly MoT and MoA")
    if set(config.get("non_spatial_profiles", {})) != {"moe", "latent", "molora"}:
        raise ValueError("P2 non_spatial_profiles must be exactly MoE, Latent and MoLoRA")
    batch_sizes = config.get("batch_equivalence_sizes", [])
    if not isinstance(batch_sizes, list) or any(int(size) < 2 for size in batch_sizes):
        raise ValueError("batch_equivalence_sizes must contain integers >= 2")
    config["batch_equivalence_sizes"] = [int(size) for size in batch_sizes]
    region_analysis = config.get("region_analysis")
    expected_region_analysis = {
        "enabled": True,
        "label_format": "yolo_detection_normalized_xywh",
        "assignment_rule": "valid_token_center_inside_any_ground_truth_box",
        "exclude_letterbox_padding": True,
    }
    if region_analysis != expected_region_analysis:
        raise ValueError(f"region_analysis must preserve the audited contract: {expected_region_analysis}")
    return config


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("e3_p2")
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


def _verify_source(source_root: Path, expected: dict[str, str]) -> list[dict[str, Any]]:
    checks = []
    for relative, digest in expected.items():
        path = source_root / relative
        actual = sha256_file(path) if path.is_file() else None
        checks.append({"path": relative, "expected_sha256": digest, "actual_sha256": actual, "match": actual == digest})
    mismatches = [item for item in checks if not item["match"]]
    if mismatches:
        raise RuntimeError(f"runtime source fingerprint mismatch: {mismatches[:1]}")
    return checks


def _verify_project_source_state(require_committed: bool) -> dict[str, Any]:
    """Bind a formal evidence run to committed implementation files."""

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

    commit_result = git("rev-parse", "HEAD")
    tree_result = git("rev-parse", "HEAD^{tree}")
    tracked_status = git(
        "status",
        "--porcelain",
        "--",
        "src",
        "tests",
        "configs",
        "pyproject.toml",
        "run_p2.cmd",
        "run_robustness.cmd",
        "run_appearance.cmd",
        "run_layer_drilldown.cmd",
        "run_image_scale.cmd",
        "run_image_driver.cmd",
        "run_dose_response.cmd",
        "run_output_coupling.cmd",
        "run_dose_holdout.cmd",
        "run_blur_residual.cmd",
        "run_demo.cmd",
        "run_tests.cmd",
    )
    state = {
        "commit": commit_result.stdout.strip() if commit_result.returncode == 0 else None,
        "tree": tree_result.stdout.strip() if tree_result.returncode == 0 else None,
        "implementation_status_porcelain": tracked_status.stdout.splitlines(),
        "implementation_clean": tracked_status.returncode == 0 and not tracked_status.stdout.strip(),
    }
    if require_committed and (not state["commit"] or not state["implementation_clean"]):
        raise RuntimeError(f"formal run requires committed implementation files: {state}")
    return state


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
    invalid = [index for index in sample_indices if not 0 <= index < len(images)]
    if not images or invalid:
        raise IndexError(f"dataset contains {len(images)} images; invalid sample indices={invalid}")
    return [images[index] for index in sample_indices], {
        "dataset": dataset_name,
        "split": split,
        "dataset_root": str(dataset.get("path")),
        "split_image_count": len(images),
    }


def _prepare_inputs(
    paths: list[Path], indices: list[int], image_size: int, torch_module: Any
) -> tuple[list[Any], list[Any], list[dict[str, Any]]]:
    tensors, originals, metadata = [], [], []
    for sample_index, path in zip(indices, paths):
        image = Image.open(path).convert("RGB")
        canvas, geometry = letterbox(image, image_size)
        array = canvas.astype(np.float32).transpose(2, 0, 1) / 255.0
        tensors.append(torch_module.from_numpy(array).unsqueeze(0).contiguous())
        originals.append(image.copy())
        label_path = label_path_for_image(path)
        boxes = parse_yolo_labels(label_path)
        metadata.append(
            {
                "sample_index": sample_index,
                "name": path.name,
                "path": str(path),
                "sha256": sha256_file(path),
                "label_path": str(label_path),
                "label_sha256": sha256_file(label_path),
                "ground_truth_box_count": len(boxes),
                "ground_truth_boxes": [box.to_dict() for box in boxes],
                "geometry": geometry.to_dict(),
                "normalization": "RGB uint8 / 255.0",
            }
        )
    return tensors, originals, metadata


def _detach_tree(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "shape"):
        return value.detach().cpu().clone()
    if isinstance(value, tuple):
        return tuple(_detach_tree(child) for child in value)
    if isinstance(value, list):
        return [_detach_tree(child) for child in value]
    if isinstance(value, dict):
        return {key: _detach_tree(child) for key, child in value.items()}
    return value


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def _archive_input_previews(
    paths: list[Path], originals: list[Image.Image], inputs: list[dict[str, Any]], run_dir: Path
) -> None:
    destination_root = run_dir / "inputs"
    destination_root.mkdir(parents=True, exist_ok=True)
    for path, original, metadata in zip(paths, originals, inputs):
        relative = Path("inputs") / f"sample-{metadata['sample_index']}--{path.name}"
        shutil.copyfile(path, run_dir / relative)
        metadata["artifact_path"] = relative.as_posix()
        label_path = Path(metadata["label_path"])
        label_relative = Path("inputs") / f"sample-{metadata['sample_index']}--{label_path.name}"
        shutil.copyfile(label_path, run_dir / label_relative)
        metadata["label_artifact_path"] = label_relative.as_posix()
        annotated_relative = Path("inputs") / f"sample-{metadata['sample_index']}--ground-truth.png"
        boxes = [YoloBox(**box) for box in metadata["ground_truth_boxes"]]
        save_ground_truth_overlay(
            original,
            boxes,
            LetterboxMeta(**metadata["geometry"]),
            str(run_dir / annotated_relative),
        )
        metadata["annotated_artifact_path"] = annotated_relative.as_posix()


def _aggregate_spatial_diagnostics(captures: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        diagnostics = [item["diagnostics"] for item in items]
        token_count = sum(item["token_count"] for item in diagnostics)
        expert_count = len(diagnostics[0]["dominant_token_count"])
        dominant_counts = [
            sum(item["dominant_token_count"][expert] for item in diagnostics) for expert in range(expert_count)
        ]
        def weighted(key: str) -> float:
            return sum(item[key]["mean"] * item["token_count"] for item in diagnostics) / token_count
        return {
            "capture_count": len(items),
            "token_count": token_count,
            "dominant_token_count": dominant_counts,
            "dominant_token_fraction": [count / token_count for count in dominant_counts],
            "active_dominant_experts": sum(count > 0 for count in dominant_counts),
            "normalized_entropy_mean": weighted("normalized_entropy"),
            "top1_margin_mean": weighted("top1_margin"),
            "neighbor_probability_l1_mean": sum(
                item["neighbor_probability_l1_mean"] * item["token_count"] for item in diagnostics
            )
            / token_count,
        }

    families = sorted({item["family"] for item in captures})
    modules = sorted({(item["family"], item["module"]) for item in captures})
    return {
        "capture_count": len(captures),
        "by_family": {
            family: summarize([item for item in captures if item["family"] == family]) for family in families
        },
        "by_module": {
            f"{family}:{module}": summarize(
                [item for item in captures if item["family"] == family and item["module"] == module]
            )
            for family, module in modules
        },
    }


def _validate_demo_assets(
    run_dir: Path, entries: list[dict[str, Any]], inputs: list[dict[str, Any]]
) -> dict[str, Any]:
    expected_sizes = {
        item["sample_index"]: (item["geometry"]["original_width"], item["geometry"]["original_height"])
        for item in inputs
    }
    paths = [entry["path"] for entry in entries]
    if len(paths) != len(set(paths)):
        raise RuntimeError("demo contains duplicate rendered asset paths")
    checked = 0
    for entry in entries:
        path = run_dir / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"demo asset is missing: {path}")
        with Image.open(path) as image:
            if image.size != expected_sizes[entry["sample_index"]]:
                raise RuntimeError(
                    f"demo asset size mismatch: {entry['path']} has {image.size}, "
                    f"expected {expected_sizes[entry['sample_index']]}"
                )
        original_path = run_dir / entry["original_path"]
        if not original_path.is_file():
            raise FileNotFoundError(f"demo original asset is missing: {original_path}")
        annotated_path = run_dir / entry["annotated_original_path"]
        if not annotated_path.is_file():
            raise FileNotFoundError(f"demo annotated original asset is missing: {annotated_path}")
        with Image.open(annotated_path) as image:
            if image.size != expected_sizes[entry["sample_index"]]:
                raise RuntimeError(
                    f"demo annotated asset size mismatch: {entry['annotated_original_path']} has {image.size}, "
                    f"expected {expected_sizes[entry['sample_index']]}"
                )
        checked += 1
    return {
        "status": "PASS",
        "entry_count": checked,
        "unique_rendered_asset_count": len(set(paths)),
        "all_rendered_assets_match_original_size": True,
        "all_original_assets_present": True,
        "all_annotated_original_assets_present_and_match_original_size": True,
    }


def _batch_equivalence(
    *,
    model: Any,
    family: str,
    router_class: str,
    tensors: list[Any],
    records_per_sample: list[list[Any]],
    batch_sizes: list[int],
    torch_module: Any,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Prove that the batch axis maps back to the same per-sample router grids."""

    results = []
    for batch_size in batch_sizes:
        if batch_size > len(tensors):
            raise ValueError(f"batch equivalence size {batch_size} exceeds {len(tensors)} selected samples")
        weight_delta = 0.0
        logit_delta = 0.0
        indices_equal = True
        compared_samples = 0
        compared_records = 0
        for start in range(0, len(tensors), batch_size):
            stop = min(start + batch_size, len(tensors))
            if stop - start != batch_size:
                continue
            collector = SpatialRouterCollector(family=family, router_class=router_class)
            registered = collector.register(model)
            try:
                with torch_module.inference_mode():
                    model(torch_module.cat(tensors[start:stop], dim=0))
            finally:
                collector.remove()
            if collector.handles or len(collector.records) != len(registered):
                raise RuntimeError(f"batch capture count or hook cleanup failed for family={family}, size={batch_size}")
            for module_index, batched in enumerate(collector.records):
                for offset, sample_index in enumerate(range(start, stop)):
                    single = records_per_sample[sample_index][module_index]
                    if batched.module_name != single.module_name:
                        raise RuntimeError("batch and single module order differ")
                    weight_delta = max(
                        weight_delta,
                        float(np.max(np.abs(batched.weights[offset] - single.weights[0]))),
                    )
                    logit_delta = max(
                        logit_delta,
                        float(np.max(np.abs(batched.logits[offset] - single.logits[0]))),
                    )
                    if batched.indices is not None:
                        indices_equal = indices_equal and bool(
                            np.array_equal(batched.indices[offset], single.indices[0])
                        )
                    compared_records += 1
            compared_samples += stop - start
        if not compared_samples:
            raise RuntimeError(f"batch size {batch_size} produced no complete comparison group")
        if weight_delta > tolerance or logit_delta > tolerance or not indices_equal:
            raise RuntimeError(
                f"single-vs-batch routing mismatch for family={family}, size={batch_size}: "
                f"weights={weight_delta}, logits={logit_delta}, indices_equal={indices_equal}"
            )
        results.append(
            {
                "batch_size": batch_size,
                "compared_samples": compared_samples,
                "compared_router_records": compared_records,
                "max_weight_abs_delta": weight_delta,
                "max_logit_abs_delta": logit_delta,
                "indices_equal": indices_equal if family == "mot" else None,
                "tolerance": tolerance,
                "status": "PASS",
            }
        )
    return {"status": "PASS", "sizes": results}


def _run_spatial_family(
    *,
    family: str,
    profile: dict[str, Any],
    source_root: Path,
    tensors: list[Any],
    originals: list[Image.Image],
    inputs: list[dict[str, Any]],
    run_dir: Path,
    alpha: float,
    torch_module: Any,
    yolo_class: Any,
    logger: logging.Logger,
    batch_equivalence_sizes: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray], list[dict[str, str]]]:
    model_config = source_root / profile["model_config"]
    wrapper = yolo_class(model_config)
    model = wrapper.model.to("cpu").eval()
    baseline_outputs = []
    with torch_module.inference_mode():
        for tensor in tensors:
            baseline_outputs.append(_detach_tree(model(tensor)))

    collector = SpatialRouterCollector(family=family, router_class=profile["router_class"])
    registered = collector.register(model)
    records_per_sample = []
    deltas = []
    try:
        for tensor, baseline in zip(tensors, baseline_outputs):
            before = len(collector.records)
            with torch_module.inference_mode():
                observed = model(tensor)
            deltas.append(max_output_delta(baseline, observed))
            records_per_sample.append(collector.records[before:])
    finally:
        collector.remove()
    if collector.handles:
        raise RuntimeError(f"hook leak for family={family}")
    if any(len(records) != len(registered) for records in records_per_sample):
        raise RuntimeError(f"capture count mismatch for family={family}")
    if max(deltas, default=0.0) != 0.0:
        raise RuntimeError(f"hook changed model output for family={family}: max_delta={max(deltas)}")

    repeat_collector = SpatialRouterCollector(family=family, router_class=profile["router_class"])
    repeat_registered = repeat_collector.register(model)
    try:
        with torch_module.inference_mode():
            for tensor in tensors:
                model(tensor)
    finally:
        repeat_collector.remove()
    if repeat_registered != registered or len(repeat_collector.records) != len(collector.records):
        raise RuntimeError(f"repeat discovery/capture mismatch for family={family}")
    repeat_max_delta = 0.0
    repeat_indices_equal = True
    for first, second in zip(collector.records, repeat_collector.records):
        if first.module_name != second.module_name:
            raise RuntimeError(f"repeat module order changed for family={family}")
        repeat_max_delta = max(
            repeat_max_delta,
            float(np.max(np.abs(first.weights - second.weights))),
            float(np.max(np.abs(first.logits - second.logits))),
        )
        if first.indices is not None:
            repeat_indices_equal = repeat_indices_equal and bool(np.array_equal(first.indices, second.indices))
    if repeat_max_delta != 0.0 or not repeat_indices_equal:
        raise RuntimeError(
            f"deterministic repeat failed for family={family}: delta={repeat_max_delta}, "
            f"indices_equal={repeat_indices_equal}"
        )

    batch_equivalence = _batch_equivalence(
        model=model,
        family=family,
        router_class=profile["router_class"],
        tensors=tensors,
        records_per_sample=records_per_sample,
        batch_sizes=batch_equivalence_sizes,
        torch_module=torch_module,
    )

    arrays: dict[str, np.ndarray] = {}
    demo_entries: list[dict[str, Any]] = []
    overview_cards: list[dict[str, str]] = []
    capture_metadata = []
    overlay_root = run_dir / "overlays" / family
    overlay_root.mkdir(parents=True, exist_ok=True)
    expert_labels = profile["expert_labels"]
    for sample_position, (sample_meta, original, records) in enumerate(zip(inputs, originals, records_per_sample)):
        geometry = LetterboxMeta(**sample_meta["geometry"])
        for layer_index, record in enumerate(records):
            weights = record.weights[0]
            module_slug = _slug(record.module_name)
            key = f"sample{sample_meta['sample_index']}__{family}__{module_slug}"
            arrays[f"{key}__weights"] = record.weights
            arrays[f"{key}__logits"] = record.logits
            if record.indices is not None:
                arrays[f"{key}__indices"] = record.indices
            metric_fields = routing_metric_fields(weights)
            diagnostics = routing_diagnostics(weights)
            masks = token_region_masks(
                [YoloBox(**box) for box in sample_meta["ground_truth_boxes"]],
                geometry,
                int(weights.shape[1]),
                int(weights.shape[2]),
            )
            region_diagnostics = region_routing_diagnostics(weights, masks)
            arrays[f"{key}__foreground_mask"] = masks["foreground"]
            arrays[f"{key}__background_mask"] = masks["background"]
            arrays[f"{key}__padding_mask"] = masks["padding"]
            arrays[f"{key}__normalized_entropy"] = metric_fields["normalized_entropy"]
            arrays[f"{key}__top1_margin"] = metric_fields["top1_margin"]
            dominant_relative = (
                Path("overlays") / family / f"sample-{sample_meta['sample_index']}--{module_slug}--dominant.png"
            )
            counts = save_dominant_overlay(
                original,
                weights,
                geometry,
                str(run_dir / dominant_relative),
                alpha=alpha,
            )
            demo_entries.append(
                {
                    "family": family,
                    "sample_index": sample_meta["sample_index"],
                    "sample_name": sample_meta["name"],
                    "module": record.module_name,
                    "layer_index": layer_index,
                    "view": "dominant",
                    "expert_index": None,
                    "expert_label": "dominant expert",
                    "path": dominant_relative.as_posix(),
                    "source_shape": record.validation["shape"],
                    "token_counts": counts,
                }
            )
            if sample_position == 0 and layer_index < 2:
                overview_cards.append(
                    {
                        "path": str(run_dir / dominant_relative),
                        "caption": f"{family.upper()} | {record.module_name} | dominant",
                    }
                )
            expert_stats = []
            for expert_index in range(weights.shape[0]):
                expert_label = (
                    expert_labels[expert_index] if expert_index < len(expert_labels) else f"expert-{expert_index}"
                )
                relative = (
                    Path("overlays")
                    / family
                    / f"sample-{sample_meta['sample_index']}--{module_slug}--expert-{expert_index}-{_slug(expert_label)}.png"
                )
                stats = save_probability_overlay(
                    original,
                    weights[expert_index],
                    geometry,
                    str(run_dir / relative),
                    expert_index=expert_index,
                    alpha=alpha,
                )
                expert_stats.append({"expert_index": expert_index, "expert_label": expert_label, **stats})
                demo_entries.append(
                    {
                        "family": family,
                        "sample_index": sample_meta["sample_index"],
                        "sample_name": sample_meta["name"],
                        "module": record.module_name,
                        "layer_index": layer_index,
                        "view": "probability",
                        "expert_index": expert_index,
                        "expert_label": expert_label,
                        "path": relative.as_posix(),
                        "source_shape": record.validation["shape"],
                        "stats": stats,
                    }
                )
            metric_stats = {}
            metric_specs = {
                "normalized_entropy": ("entropy", "normalized routing entropy", (166, 108, 255)),
                "top1_margin": ("margin", "top-1 routing margin", (0, 229, 255)),
            }
            for field_name, (view_name, label, color) in metric_specs.items():
                relative = (
                    Path("overlays")
                    / family
                    / f"sample-{sample_meta['sample_index']}--{module_slug}--{view_name}.png"
                )
                stats = save_metric_overlay(
                    original,
                    metric_fields[field_name],
                    geometry,
                    str(run_dir / relative),
                    color=color,
                    alpha=alpha,
                )
                metric_stats[field_name] = stats
                demo_entries.append(
                    {
                        "family": family,
                        "sample_index": sample_meta["sample_index"],
                        "sample_name": sample_meta["name"],
                        "module": record.module_name,
                        "layer_index": layer_index,
                        "view": view_name,
                        "expert_index": None,
                        "expert_label": label,
                        "path": relative.as_posix(),
                        "source_shape": record.validation["shape"],
                        "stats": stats,
                    }
                )
            capture_metadata.append(
                {
                    "family": family,
                    "sample_index": sample_meta["sample_index"],
                    "module": record.module_name,
                    "module_type": record.module_type,
                    "weights_key": f"{key}__weights",
                    "logits_key": f"{key}__logits",
                    "indices_key": f"{key}__indices" if record.indices is not None else None,
                    "validation": record.validation,
                    "expert_stats": expert_stats,
                    "metric_stats": metric_stats,
                    "diagnostics": diagnostics,
                    "region_mask_keys": {
                        "foreground": f"{key}__foreground_mask",
                        "background": f"{key}__background_mask",
                        "padding": f"{key}__padding_mask",
                    },
                    "region_diagnostics": region_diagnostics,
                }
            )
    grids = sorted({tuple(item["validation"]["shape"][2:]) for item in capture_metadata})
    summary = {
        "status": "SUPPORTED",
        "token_overlay": "supported",
        "evidence_kind": "full_detector_forward",
        "model_config": profile["model_config"],
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "router_class": profile["router_class"],
        "registered_modules": registered,
        "registered_module_count": len(registered),
        "samples": len(tensors),
        "capture_count": len(capture_metadata),
        "feature_grids_hw": [list(grid) for grid in grids],
        "expert_labels": expert_labels,
        "probability_validation": "PASS",
        "hook_output_equivalence": "PASS",
        "max_output_abs_delta": max(deltas, default=0.0),
        "deterministic_repeat": "PASS",
        "repeat_max_weight_or_logit_abs_delta": repeat_max_delta,
        "repeat_indices_equal": repeat_indices_equal,
        "single_vs_batch_equivalence": batch_equivalence,
        "hooks_removed": True,
    }
    logger.info(
        "family=%s supported modules=%d samples=%d captures=%d grids=%s output_delta=%g repeat_delta=%g",
        family,
        len(registered),
        len(tensors),
        len(capture_metadata),
        grids,
        summary["max_output_abs_delta"],
        repeat_max_delta,
    )
    del model, wrapper, baseline_outputs
    return summary, capture_metadata, arrays, overview_cards


def _audit_moe(source_root: Path, tensor: Any, torch_module: Any, yolo_class: Any) -> dict[str, Any]:
    model_config = source_root / "ultralytics/cfg/models/26/yolo26-master-n.yaml"
    wrapper = yolo_class(model_config)
    model = wrapper.model.eval()
    captures: list[dict[str, Any]] = []
    handles = []
    for name, module in model.named_modules():
        module_path = module.__class__.__module__.lower()
        routing = getattr(module, "routing", None)
        if routing is None or ".moe." not in module_path or not hasattr(module, "num_experts"):
            continue

        def capture(current: Any, inputs: Any, output: Any, *, owner_name: str = name) -> None:
            del current, inputs
            captures.append({"module": owner_name, "router_output_shapes": tensor_shapes(output)})

        handles.append(routing.register_forward_hook(capture))
    try:
        with torch_module.inference_mode():
            model(tensor)
    finally:
        for handle in handles:
            handle.remove()
    shapes = [shape for item in captures for shape in item["router_output_shapes"]]
    spatial = [shape for shape in shapes if len(shape) == 4 and shape[-2] > 1 and shape[-1] > 1]
    if spatial:
        raise RuntimeError(f"MoE audit unexpectedly found spatial router outputs: {spatial}")
    result = {
        "status": "UNSUPPORTED",
        "token_overlay": "unsupported",
        "evidence_kind": "full_detector_nested_router_forward",
        "model_config": "ultralytics/cfg/models/26/yolo26-master-n.yaml",
        "reason": "router decisions are image/sample-level; singleton H/W axes are broadcast containers, not token maps",
        "captured_modules": captures,
        "observed_shapes": shapes,
        "spatial_shape_count": 0,
    }
    del model, wrapper
    return result


def _audit_latent(source_root: Path, tensor: Any, torch_module: Any, yolo_class: Any) -> dict[str, Any]:
    model_config = source_root / "ultralytics/cfg/models/26/yolo26-master-latent-n.yaml"
    wrapper = yolo_class(model_config)
    model = wrapper.model.eval()
    with torch_module.inference_mode():
        model(tensor)
    captures = []
    for name, module in model.named_modules():
        if module.__class__.__name__ not in {"LatentMixture", "MultiScaleLatentMixture"}:
            continue
        probs = getattr(module, "_last_routing_probs", None)
        snapshot = getattr(module, "last_routing_snapshot", {})
        captures.append(
            {
                "module": name,
                "routing_probs_shape": [int(item) for item in probs.shape] if probs is not None else None,
                "routing_axis": snapshot.get("routing_axis") if isinstance(snapshot, dict) else None,
            }
        )
    if not captures:
        raise RuntimeError("Latent audit found no routing modules")
    result = {
        "status": "UNSUPPORTED",
        "token_overlay": "unsupported",
        "evidence_kind": "full_detector_runtime_state",
        "model_config": "ultralytics/cfg/models/26/yolo26-master-latent-n.yaml",
        "reason": "routing axes are expert or scale-by-expert after spatial pooling; no reversible H/W token axis exists",
        "captured_modules": captures,
        "spatial_shape_count": 0,
    }
    del model, wrapper
    return result


def _audit_molora(torch_module: Any) -> dict[str, Any]:
    from ultralytics.nn.peft.molora.router import HybridRouter, LinearRouter, SpatialRouter

    probe = torch_module.linspace(-1.0, 1.0, steps=16 * 8 * 8).reshape(1, 16, 8, 8)
    captures = []
    for router_class in (LinearRouter, SpatialRouter, HybridRouter):
        router = router_class(16, 4).eval()
        with torch_module.inference_mode():
            output = router(probe)
        captures.append(
            {"router_type": router_class.__name__, "input_shape": [1, 16, 8, 8], "output_shape": list(output.shape)}
        )
    if any(item["output_shape"] != [1, 4] for item in captures):
        raise RuntimeError(f"unexpected MoLoRA router contract: {captures}")
    return {
        "status": "UNSUPPORTED",
        "token_overlay": "unsupported",
        "evidence_kind": "isolated_official_router_contract",
        "reason": "all official MoLoRA router variants reduce spatial features and return one [B,E] decision per image",
        "captured_router_variants": captures,
        "spatial_shape_count": 0,
    }


def _demo_html() -> str:
    return """<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><link rel=\"icon\" href=\"data:,\">
<title>E3 P2 Routing Lens</title><style>
:root{color-scheme:dark;--bg:#061022;--panel:#0b1932;--line:#1d4264;--cyan:#00e5ff;--violet:#a66cff;--text:#eaf4ff;--muted:#89a4c2}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% 0,#112b4d 0,var(--bg) 42%);font:15px system-ui;color:var(--text)}
.shell{max-width:1440px;margin:auto;padding:28px}.hero{border:1px solid var(--line);background:linear-gradient(100deg,#07172cdd,#120c2add);padding:22px 26px;border-radius:18px;box-shadow:0 0 40px #00e5ff14}
.eyebrow{color:var(--cyan);letter-spacing:.18em;font:700 12px ui-monospace}.hero h1{margin:8px 0 6px;font-size:30px}.hero p{margin:0;color:var(--muted)}
.notice{margin-top:14px;padding:11px 14px;border-left:3px solid #ffc400;background:#ffc40012;color:#ffe7a0}.grid{display:grid;grid-template-columns:310px 1fr;gap:18px;margin-top:18px}
.panel{border:1px solid var(--line);background:#09162bcc;border-radius:16px;padding:18px}.control{margin-bottom:14px}.control label{display:block;margin-bottom:6px;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}
select,button,a.button{width:100%;border:1px solid #285474;background:#0d203b;color:var(--text);padding:10px;border-radius:9px;text-decoration:none;display:block}button,a.button{cursor:pointer;text-align:center;margin-top:9px;background:linear-gradient(90deg,#006f8c,#53359a);font-weight:700}.toggle{display:flex;gap:8px;align-items:center;color:var(--muted);margin:2px 0 12px}.toggle input{accent-color:var(--cyan)}
.compare{display:grid;grid-template-columns:1fr 1fr;gap:12px}.frame{margin:0}.frame figcaption{color:var(--muted);font:700 11px ui-monospace;letter-spacing:.1em;margin:0 0 7px}.frame img{display:block;width:100%;height:430px;object-fit:contain;background:#050b16;border-radius:12px;border:1px solid #173653}
.meta{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}.chip{background:#102748;border:1px solid #214b70;border-radius:999px;padding:6px 10px;color:#bcd5ee;font-size:12px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:12px}.stat{padding:10px 12px;border:1px solid #214b70;background:#08172d;border-radius:10px}.stat b{display:block;color:var(--cyan);font:700 17px ui-monospace}.stat small{color:var(--muted)}
.matrix{margin-top:18px;display:grid;grid-template-columns:repeat(5,1fr);gap:10px}.family{padding:12px;border:1px solid var(--line);border-radius:12px;background:#08172d}.ok{color:#42e58c}.no{color:#ff889c}.status{margin-top:10px;color:#7ee8b2;font-size:12px}pre{max-height:210px;overflow:auto;white-space:pre-wrap;color:#aac4df;font-size:11px}@media(max-width:900px){.grid{grid-template-columns:1fr}.matrix{grid-template-columns:repeat(2,1fr)}}@media(max-width:620px){.shell{padding:14px}.hero h1{font-size:24px}.matrix,.compare,.stats{grid-template-columns:1fr}.frame img{height:auto}}
</style></head><body><main class=\"shell\"><section class=\"hero\"><div class=\"eyebrow\">E3://SPATIAL ROUTING LENS</div><h1>Token routing, mapped back without distortion</h1><p>Original and overlay stay side by side. Switch sample, family, layer, probability, entropy or margin without losing the evidence source.</p>
<div class=\"notice\">Interpretation boundary: this run uses random initialization. Uniform maps validate capture, batch identity and geometry; they do not demonstrate learned expert specialization.</div></section>
<section class=\"matrix\" id=\"matrix\" aria-label=\"Five-family feasibility\"></section><section class=\"grid\"><aside class=\"panel\"><div class=\"control\"><label for=\"family\">Family</label><select id=\"family\"></select></div><div class=\"control\"><label for=\"sample\">Sample</label><select id=\"sample\"></select></div><div class=\"control\"><label for=\"layer\">Layer</label><select id=\"layer\"></select></div><div class=\"control\"><label for=\"view\">View / expert</label><select id=\"view\"></select></div><label class=\"toggle\"><input id=\"groundTruth\" type=\"checkbox\">Show archived ground-truth boxes</label><a class=\"button\" id=\"download\" download>Export current PNG</a><button id=\"copy\">Copy evidence metadata</button><div class=\"status\" id=\"status\" role=\"status\">Loading evidence…</div><pre id=\"detail\"></pre></aside><section class=\"panel\"><div class=\"compare\"><figure class=\"frame\"><figcaption>ORIGINAL / GROUND TRUTH</figcaption><img id=\"original\" alt=\"original coco8 input\"></figure><figure class=\"frame\"><figcaption>ROUTING OVERLAY</figcaption><img id=\"image\" alt=\"routing overlay\"></figure></div><div class=\"meta\" id=\"chips\"></div><div class=\"stats\" id=\"stats\"></div></section></section></main>
<script>
let entries;const q=id=>document.getElementById(id);const unique=xs=>[...new Set(xs)];
function options(el,values,label){const old=el.value;el.innerHTML=values.map(v=>`<option value=\"${v}\">${label(v)}</option>`).join('');if(values.includes(old))el.value=old;}
function viewLabel(x){if(x.view==='dominant')return'Dominant expert';if(x.view==='entropy')return'Normalized routing entropy';if(x.view==='margin')return'Top-1 routing margin';return`Expert ${x.expert_index} · ${x.expert_label}`;}
function cascade(source){let family=q('family').value;if(source==='family'||!family){options(q('family'),unique(entries.map(x=>x.family)),x=>x.toUpperCase());family=q('family').value}let pool=entries.filter(x=>x.family===family);options(q('sample'),unique(pool.map(x=>String(x.sample_index))),x=>`${x} · ${pool.find(y=>String(y.sample_index)===x).sample_name}`);pool=pool.filter(x=>String(x.sample_index)===q('sample').value);options(q('layer'),unique(pool.map(x=>x.module)),x=>x);pool=pool.filter(x=>x.module===q('layer').value);options(q('view'),pool.map((_,i)=>String(i)),i=>viewLabel(pool[Number(i)]));render(pool[Number(q('view').value||0)]);}
function renderStats(x){let values;if(x.stats){values=[['min',x.stats.feature_min],['mean',x.stats.feature_mean],['max',x.stats.feature_max]]}else{const d=x.diagnostics;values=[['entropy',d.normalized_entropy.mean],['top-1 margin',d.top1_margin.mean],['active experts',d.active_dominant_experts]]}const r=x.region_diagnostics;if(r){values.push(['foreground tokens',r.foreground.token_count],['background tokens',r.background.token_count],['FG/BG TV',r.contrast?r.contrast.total_variation_distance:'n/a'])}q('stats').innerHTML=values.map(([label,value])=>`<div class=\"stat\"><b>${typeof value==='number'?value.toFixed(6):value}</b><small>${label}</small></div>`).join('');}
function renderOriginal(x){q('original').src=q('groundTruth').checked?x.annotated_original_path:x.original_path;}
function render(x){if(!x)return;renderOriginal(x);q('image').src=x.path;q('download').href=x.path;q('chips').innerHTML=[x.family.toUpperCase(),x.sample_name,x.module,`shape ${x.source_shape.join('×')}`,x.expert_label].map(v=>`<span class=\"chip\">${v}</span>`).join('');renderStats(x);q('detail').textContent=JSON.stringify(x,null,2);q('status').textContent=`Ready · ${viewLabel(x)} · original-size export`;q('copy').onclick=async()=>{try{await navigator.clipboard.writeText(q('detail').textContent);q('status').textContent='Evidence metadata copied'}catch(error){q('status').textContent=`Copy failed: ${error.message}`}};q('groundTruth').onchange=()=>renderOriginal(x);}
fetch('demo-index.json').then(response=>{if(!response.ok)throw new Error(`HTTP ${response.status}`);return response.json()}).then(data=>{entries=data.entries;q('matrix').innerHTML=Object.entries(data.feasibility).map(([key,value])=>`<div class=\"family\"><b>${key.toUpperCase()}</b><div class=\"${value.token_overlay==='supported'?'ok':'no'}\">${value.token_overlay}</div><small>${value.reason||value.evidence_kind}</small></div>`).join('');['family','sample','layer','view'].forEach(id=>q(id).onchange=()=>cascade(id));cascade('family');document.body.dataset.ready='true'}).catch(error=>{q('status').textContent=`Evidence failed to load: ${error.message}`;q('status').style.color='#ff889c'});
</script></body></html>"""


def run(config_path: Path, *, run_id: str | None = None, update_latest: bool = True) -> Path:
    config = _load_config(config_path, run_id)
    project_source = _verify_project_source_state(bool(config.get("require_committed_source", False)))
    run_dir = PROJECT_ROOT / "artifacts" / "p2" / config["run_id"]
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty evidence directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _logger(run_dir / "full.log")
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (run_dir / "command.txt").write_text("run_p2.cmd\n", encoding="utf-8")
    started = time.perf_counter()

    source_root = (PROJECT_ROOT / config["runtime_root"]).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"runtime_root does not exist: {source_root}")
    checks = _verify_source(source_root, config["source_fingerprints"])
    sys.path.insert(0, str(source_root))
    os.chdir(PROJECT_ROOT)
    import torch
    from ultralytics import YOLO

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info("scope=E3 P2 five-family feasibility and truthful spatial overlay")
    logger.info("source_ref=%s source_tree=%s", config["official_runtime_ref"], config["official_runtime_tree"])
    logger.info("device=cpu cuda_available=%s seed=%d", torch.cuda.is_available(), seed)

    paths, dataset_meta = _resolve_images(config["dataset"], config["dataset_split"], config["sample_indices"])
    tensors, originals, inputs = _prepare_inputs(paths, config["sample_indices"], int(config["image_size"]), torch)
    _archive_input_previews(paths, originals, inputs, run_dir)
    input_record = {**dataset_meta, "selected_image_count": len(inputs), "images": inputs}
    input_record["input_set_sha256"] = hashlib.sha256(
        "\n".join(f"{item['sample_index']}:{item['sha256']}" for item in inputs).encode("utf-8")
    ).hexdigest()
    input_record["annotation_set_sha256"] = hashlib.sha256(
        "\n".join(f"{item['sample_index']}:{item['label_sha256']}" for item in inputs).encode("utf-8")
    ).hexdigest()
    input_record["image_and_annotation_set_sha256"] = hashlib.sha256(
        "\n".join(
            f"{item['sample_index']}:{item['sha256']}:{item['label_sha256']}" for item in inputs
        ).encode("utf-8")
    ).hexdigest()
    write_json(run_dir / "input.json", input_record)
    write_json(run_dir / "source-fingerprint-checks.json", checks)

    feasibility: dict[str, Any] = {}
    capture_metadata: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    overview_cards: list[dict[str, str]] = []
    for family, profile in config["spatial_profiles"].items():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        family_summary, family_metadata, family_arrays, family_cards = _run_spatial_family(
            family=family,
            profile=profile,
            source_root=source_root,
            tensors=tensors,
            originals=originals,
            inputs=inputs,
            run_dir=run_dir,
            alpha=float(config["overlay_alpha"]),
            torch_module=torch,
            yolo_class=YOLO,
            logger=logger,
            batch_equivalence_sizes=config["batch_equivalence_sizes"],
        )
        feasibility[family] = family_summary
        capture_metadata.extend(family_metadata)
        arrays.update(family_arrays)
        overview_cards.extend(family_cards)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    feasibility["moe"] = _audit_moe(source_root, tensors[0], torch, YOLO)
    logger.info("family=moe unsupported observed_shapes=%s", feasibility["moe"]["observed_shapes"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    feasibility["latent"] = _audit_latent(source_root, tensors[0], torch, YOLO)
    logger.info("family=latent unsupported captures=%s", feasibility["latent"]["captured_modules"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    feasibility["molora"] = _audit_molora(torch)
    logger.info("family=molora unsupported variants=%s", feasibility["molora"]["captured_router_variants"])

    np.savez_compressed(run_dir / "spatial-routing-raw.npz", **arrays)
    write_json(run_dir / "spatial-captures.json", capture_metadata)
    write_json(run_dir / "spatial-diagnostics.json", _aggregate_spatial_diagnostics(capture_metadata))
    region_analysis = aggregate_region_diagnostics(capture_metadata)
    write_json(run_dir / "region-routing-analysis.json", region_analysis)
    write_json(run_dir / "family-feasibility.json", feasibility)
    save_overview(overview_cards, str(run_dir / "routing-overview.png"))

    demo_entries = []
    for metadata in capture_metadata:
        family = metadata["family"]
        sample_index = metadata["sample_index"]
        sample_name = next(item["name"] for item in inputs if item["sample_index"] == sample_index)
        module = metadata["module"]
        module_slug = _slug(module)
        shape = metadata["validation"]["shape"]
        dominant_path = Path("overlays") / family / f"sample-{sample_index}--{module_slug}--dominant.png"
        dominant_entry = {
            "family": family,
            "sample_index": sample_index,
            "sample_name": sample_name,
            "module": module,
            "view": "dominant",
            "expert_index": None,
            "expert_label": "dominant expert",
            "path": dominant_path.as_posix(),
            "source_shape": shape,
            "original_path": next(
                item["artifact_path"] for item in inputs if item["sample_index"] == sample_index
            ),
            "annotated_original_path": next(
                item["annotated_artifact_path"] for item in inputs if item["sample_index"] == sample_index
            ),
            "diagnostics": metadata["diagnostics"],
            "region_diagnostics": metadata["region_diagnostics"],
        }
        if dominant_path.is_file() or (run_dir / dominant_path).is_file():
            demo_entries.append(dominant_entry)
        labels = config["spatial_profiles"][family]["expert_labels"]
        for stats in metadata["expert_stats"]:
            expert_index = stats["expert_index"]
            label = labels[expert_index]
            relative = (
                Path("overlays")
                / family
                / f"sample-{sample_index}--{module_slug}--expert-{expert_index}-{_slug(label)}.png"
            )
            demo_entries.append(
                {
                    "family": family,
                    "sample_index": sample_index,
                    "sample_name": sample_name,
                    "module": module,
                    "view": "probability",
                    "expert_index": expert_index,
                    "expert_label": label,
                    "path": relative.as_posix(),
                    "source_shape": shape,
                    "stats": stats,
                    "original_path": next(
                        item["artifact_path"] for item in inputs if item["sample_index"] == sample_index
                    ),
                    "annotated_original_path": next(
                        item["annotated_artifact_path"]
                        for item in inputs
                        if item["sample_index"] == sample_index
                    ),
                    "region_diagnostics": metadata["region_diagnostics"],
                }
            )
        metric_specs = {
            "normalized_entropy": ("entropy", "normalized routing entropy"),
            "top1_margin": ("margin", "top-1 routing margin"),
        }
        for field_name, (view_name, label) in metric_specs.items():
            relative = Path("overlays") / family / f"sample-{sample_index}--{module_slug}--{view_name}.png"
            demo_entries.append(
                {
                    "family": family,
                    "sample_index": sample_index,
                    "sample_name": sample_name,
                    "module": module,
                    "view": view_name,
                    "expert_index": None,
                    "expert_label": label,
                    "path": relative.as_posix(),
                    "source_shape": shape,
                    "stats": metadata["metric_stats"][field_name],
                    "original_path": next(
                        item["artifact_path"] for item in inputs if item["sample_index"] == sample_index
                    ),
                    "annotated_original_path": next(
                        item["annotated_artifact_path"]
                        for item in inputs
                        if item["sample_index"] == sample_index
                    ),
                    "region_diagnostics": metadata["region_diagnostics"],
                }
            )
    demo_asset_validation = _validate_demo_assets(run_dir, demo_entries, inputs)
    demo_index = {
        "schema_version": config["schema_version"],
        "run_id": config["run_id"],
        "interpretation": "random initialization validates capture and geometry only; it does not prove learned specialization",
        "feasibility": feasibility,
        "evidence_counts": {
            "spatial_captures": len(capture_metadata),
            "raw_arrays": len(arrays),
            "demo_views": len(demo_entries),
        },
        "asset_validation": demo_asset_validation,
        "entries": demo_entries,
    }
    write_json(run_dir / "demo-index.json", demo_index)
    (run_dir / "demo.html").write_text(_demo_html(), encoding="utf-8")
    write_json(run_dir / "environment.json", environment(torch, source_root, PROJECT_ROOT))

    supported = sorted(family for family, item in feasibility.items() if item["token_overlay"] == "supported")
    unsupported = sorted(family for family, item in feasibility.items() if item["token_overlay"] == "unsupported")
    summary = {
        "status": "PASS",
        "scope": "E3 P2 five-family feasibility audit plus true MoT/MoA spatial overlays and local demo",
        "run_id": config["run_id"],
        "schema_version": config["schema_version"],
        "official_locked_base_ref": config["official_locked_base_ref"],
        "official_runtime_ref": config["official_runtime_ref"],
        "official_runtime_tree": config["official_runtime_tree"],
        "official_source_archive_url": config["official_source_archive_url"],
        "official_source_archive_sha256": config["official_source_archive_sha256"],
        "input": input_record,
        "source_fingerprint_validation": "PASS",
        "source_fingerprint_count": len(checks),
        "tool_source": project_source,
        "families_audited": sorted(feasibility),
        "token_overlay_supported": supported,
        "token_overlay_unsupported": unsupported,
        "true_spatial_capture_count": len(capture_metadata),
        "raw_array_count": len(arrays),
        "demo_entry_count": len(demo_entries),
        "demo_asset_validation": demo_asset_validation,
        "single_vs_batch_equivalence": {
            family: feasibility[family]["single_vs_batch_equivalence"] for family in supported
        },
        "hook_output_equivalence": "PASS",
        "geometry_contract": "letterbox upsample -> exact integer unpad -> original-size bilinear mapping",
        "region_analysis": {
            "status": "PASS",
            "artifact": "region-routing-analysis.json",
            "assignment_rule": region_analysis["method"]["assignment_rule"],
            "padding_policy": region_analysis["method"]["padding_policy"],
            "primary_conclusion_basis": region_analysis["method"]["primary_conclusion_basis"],
        },
        "interpretation_boundary": "random initialization; pipeline evidence only, not learned specialization",
        "duration_seconds_observation_only": time.perf_counter() - started,
    }
    write_json(run_dir / "summary.json", summary)
    write_manifest(run_dir)
    logger.info(
        "status=PASS supported=%s unsupported=%s captures=%d demo_entries=%d",
        supported,
        unsupported,
        len(capture_metadata),
        len(demo_entries),
    )
    for handler in logger.handlers:
        handler.flush()
    write_manifest(run_dir)
    if update_latest:
        latest = PROJECT_ROOT / "artifacts" / "p2" / "LATEST.txt"
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(config["run_id"] + "\n", encoding="utf-8")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "p2" / "p2.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--no-latest", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.config, run_id=args.run_id, update_latest=not args.no_latest))


if __name__ == "__main__":
    main()
