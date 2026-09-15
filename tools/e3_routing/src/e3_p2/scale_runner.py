"""CPU-only image-level scaling study for MoA appearance routing."""

from __future__ import annotations

import hashlib
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from .appearance_runner import _apply_transform, _font, _summarize_input_effects, _tensor
from .capture import max_output_delta
from .detector_output import detector_output_comparison
from .geometry import LetterboxMeta, letterbox
from .io_utils import environment, sha256_file, write_json, write_manifest
from .layer_drilldown import margin_bin_summary
from .regions import label_path_for_image, parse_yolo_labels, token_region_masks
from .robustness_runner import _capture_once
from .runner import (
    PROJECT_ROOT,
    RUN_ID_PATTERN,
    _detach_tree,
    _logger,
    _resolve_images,
    _slug,
    _verify_project_source_state,
    _verify_source,
)
from .stability import probability_map_comparison, restore_probability_stack


def _load_config(path: Path, run_id_override: str | None) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"config must contain a mapping: {path}")
    config = dict(loaded)
    run_id = str(run_id_override if run_id_override is not None else config.get("run_id", "")).strip()
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must be path-safe and at most 128 characters")
    config["run_id"] = run_id
    if config.get("device") != "cpu" or int(config.get("resolution", 0)) < 32:
        raise ValueError("scale study requires CPU and resolution >= 32")
    if config.get("family") != "moa" or not str(config.get("target_module", "")).strip():
        raise ValueError("scale study is intentionally scoped to MoA and requires target_module")
    seeds = [int(value) for value in config.get("seeds", [])]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain at least two unique integers")
    config["seeds"] = seeds
    image_count = int(config.get("selected_image_count", 0))
    dataset_count = int(config.get("expected_dataset_image_count", 0))
    if image_count < 16 or dataset_count < image_count:
        raise ValueError("selected_image_count must be >=16 and <= expected_dataset_image_count")
    if not str(config.get("selection_salt", "")):
        raise ValueError("selection_salt is required")
    transforms = config.get("transformations")
    study_kind = str(config.get("study_kind", "image_scale"))
    if not isinstance(transforms, list) or len(transforms) < 4:
        raise ValueError("scale study requires identity plus at least three perturbations")
    names = [str(item.get("name", "")) for item in transforms if isinstance(item, dict)]
    if len(names) != len(transforms) or len(set(names)) != len(names) or names[0] != "identity":
        raise ValueError("transform names must be unique and identity must be first")
    expected_transforms = [
        {"name": "identity", "kind": "identity"},
        {"name": "brightness_090", "kind": "brightness", "factor": 0.9},
        {"name": "contrast_090", "kind": "contrast", "factor": 0.9},
        {"name": "gaussian_blur_075", "kind": "gaussian_blur", "radius": 0.75},
    ]
    if study_kind == "image_scale" and transforms != expected_transforms:
        raise ValueError(f"transform contract must remain fixed: {expected_transforms}")
    if study_kind not in {"image_scale", "dose_response", "output_coupling"}:
        raise ValueError(f"unsupported scale study_kind: {study_kind}")
    if study_kind == "dose_response" and len(transforms) < 10:
        raise ValueError("dose_response requires identity plus at least nine predeclared conditions")
    if study_kind == "output_coupling" and (len(transforms) != 4 or not config.get("record_detector_outputs")):
        raise ValueError("output_coupling requires identity plus three conditions and detector-output recording")
    if int(config.get("bootstrap_draws", 0)) < 1000:
        raise ValueError("bootstrap_draws must be at least 1000")
    if int(config.get("max_evidence_bytes", 0)) < 1_000_000:
        raise ValueError("max_evidence_bytes must be at least 1 MB")
    return config


def deterministic_image_selection(
    paths: list[Path], count: int, salt: str
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Select a stable subset without looking at image contents or model outputs."""

    if count <= 0 or count > len(paths) or not salt:
        raise ValueError("invalid deterministic selection request")
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("dataset image names must be unique for path-independent selection")
    ranked = []
    for dataset_index, path in enumerate(paths):
        key = hashlib.sha256(f"{salt}\0{path.name}".encode()).hexdigest()
        ranked.append((key, path.name, dataset_index, path))
    ranked.sort()
    selected = ranked[:count]
    records = [
        {
            "sample_index": sample_index,
            "dataset_sorted_index": dataset_index,
            "name": name,
            "selection_sha256": key,
        }
        for sample_index, (key, name, dataset_index, _) in enumerate(selected)
    ]
    return [item[3] for item in selected], records


def bootstrap_mean_interval(values: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 2 or not np.isfinite(array).all() or draws < 100:
        raise ValueError("bootstrap requires finite values, at least two units and at least 100 draws")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(draws, array.size))
    estimates = array[indices].mean(axis=1)
    return {
        "unit_count": int(array.size),
        "draws": draws,
        "seed": seed,
        "observed_mean": float(array.mean()),
        "percentile_95_interval": [float(value) for value in np.percentile(estimates, [2.5, 97.5])],
        "bootstrap_median": float(np.median(estimates)),
    }


def bootstrap_layer_share_interval(
    image_by_layer: np.ndarray, target_index: int, draws: int, seed: int
) -> dict[str, Any]:
    matrix = np.asarray(image_by_layer, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or not 0 <= target_index < matrix.shape[1]:
        raise ValueError("image_by_layer must be [image,layer] with a valid target index")
    if not np.isfinite(matrix).all() or (matrix < 0.0).any():
        raise ValueError("layer values must be finite and non-negative")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, matrix.shape[0], size=(draws, matrix.shape[0]))
    estimates = matrix[indices].mean(axis=1)
    denominators = estimates.sum(axis=1)
    if (denominators <= 0.0).any():
        raise ValueError("bootstrap produced a non-positive layer sum")
    shares = estimates[:, target_index] / denominators
    ranks_first = estimates[:, target_index] >= estimates.max(axis=1)
    observed = matrix.mean(axis=0)
    return {
        "image_unit_count": int(matrix.shape[0]),
        "draws": draws,
        "seed": seed,
        "observed_target_share": float(observed[target_index] / observed.sum()),
        "percentile_95_interval": [float(value) for value in np.percentile(shares, [2.5, 97.5])],
        "bootstrap_median": float(np.median(shares)),
        "target_rank_one_draw_fraction": float(ranks_first.mean()),
    }


def summarize_image_level_attribution(
    comparisons: list[dict[str, Any]],
    transforms: list[str],
    target_module: str,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Keep images as the inferential unit for ranks, leave-one-out and bootstrap."""

    selected = [
        item
        for item in comparisons
        if item["family"] == "moa"
        and item["comparison_type"].removeprefix("appearance_") in transforms
    ]
    images = sorted({int(item["sample_index"]) for item in selected})
    seeds = sorted({int(item["seed"]) for item in selected})
    modules = sorted({item["module"] for item in selected})
    if target_module not in modules or not images or not seeds:
        raise ValueError("comparison matrix is missing images, seeds or target module")
    lookup = {
        (
            int(item["sample_index"]),
            item["comparison_type"].removeprefix("appearance_"),
            int(item["seed"]),
            item["module"],
        ): float(item["metrics"]["probability_mae"])
        for item in selected
    }
    expected = len(images) * len(transforms) * len(seeds) * len(modules)
    if len(lookup) != expected:
        raise ValueError(f"incomplete image-level comparison matrix: {len(lookup)} != {expected}")
    target_index = modules.index(target_module)

    case_rankings = []
    image_transform_rankings = []
    image_rankings = []
    for image in images:
        for transform in transforms:
            for seed in seeds:
                values = np.asarray([lookup[(image, transform, seed, module)] for module in modules])
                ranking = np.argsort(-values, kind="stable")
                case_rankings.append(
                    {
                        "sample_index": image,
                        "transform": transform,
                        "seed": seed,
                        "target_rank": int(np.where(ranking == target_index)[0][0]) + 1,
                        "rank_one_module": modules[int(ranking[0])],
                    }
                )
            means = np.asarray(
                [np.mean([lookup[(image, transform, seed, module)] for seed in seeds]) for module in modules]
            )
            ranking = np.argsort(-means, kind="stable")
            image_transform_rankings.append(
                {
                    "sample_index": image,
                    "transform": transform,
                    "target_rank": int(np.where(ranking == target_index)[0][0]) + 1,
                    "rank_one_module": modules[int(ranking[0])],
                    "target_share_of_layer_mean_mae_sum": float(means[target_index] / means.sum()),
                }
            )
        means = np.asarray(
            [
                np.mean([lookup[(image, transform, seed, module)] for transform in transforms for seed in seeds])
                for module in modules
            ]
        )
        ranking = np.argsort(-means, kind="stable")
        image_rankings.append(
            {
                "sample_index": image,
                "target_rank": int(np.where(ranking == target_index)[0][0]) + 1,
                "rank_one_module": modules[int(ranking[0])],
                "target_share_of_layer_mean_mae_sum": float(means[target_index] / means.sum()),
            }
        )

    leave_one_out = []
    bootstrap = {}
    for transform_index, transform in enumerate(transforms):
        matrix = np.asarray(
            [
                [np.mean([lookup[(image, transform, seed, module)] for seed in seeds]) for module in modules]
                for image in images
            ],
            dtype=np.float64,
        )
        for omitted_index, image in enumerate(images):
            means = np.delete(matrix, omitted_index, axis=0).mean(axis=0)
            ranking = np.argsort(-means, kind="stable")
            leave_one_out.append(
                {
                    "transform": transform,
                    "omitted_sample_index": image,
                    "target_rank": int(np.where(ranking == target_index)[0][0]) + 1,
                    "rank_one_module": modules[int(ranking[0])],
                    "target_share_of_layer_mean_mae_sum": float(means[target_index] / means.sum()),
                }
            )
        bootstrap[transform] = bootstrap_layer_share_interval(
            matrix, target_index, bootstrap_draws, bootstrap_seed + transform_index
        )

    return {
        "method": {
            "metric": "original-coordinate probability MAE",
            "primary_unit": "image",
            "within_image_aggregation": "equal mean across configured seeds and/or transforms",
            "bootstrap": "non-parametric resampling of images with replacement; descriptive for selected coco128 subset",
            "rank_tie_policy": "stable module order",
        },
        "image_count": len(images),
        "seed_count": len(seeds),
        "transform_count": len(transforms),
        "module_order": modules,
        "case_rankings": case_rankings,
        "target_rank_one_case_count": sum(item["target_rank"] == 1 for item in case_rankings),
        "case_count": len(case_rankings),
        "image_transform_rankings": image_transform_rankings,
        "target_rank_one_image_transform_count": sum(
            item["target_rank"] == 1 for item in image_transform_rankings
        ),
        "image_transform_count": len(image_transform_rankings),
        "image_rankings": image_rankings,
        "target_rank_one_image_count": sum(item["target_rank"] == 1 for item in image_rankings),
        "leave_one_image_out": leave_one_out,
        "target_rank_one_leave_one_out_count": sum(item["target_rank"] == 1 for item in leave_one_out),
        "leave_one_out_count": len(leave_one_out),
        "bootstrap_target_share_by_transform": bootstrap,
    }


def summarize_image_level_switches(
    cases: list[dict[str, Any]], transforms: list[str], draws: int, seed: int
) -> dict[str, Any]:
    images = sorted({int(item["sample_index"]) for item in cases})
    by_transform = {}
    for index, transform in enumerate(transforms):
        image_values = []
        for image in images:
            values = [
                item["dominant_switch_fraction"]
                for item in cases
                if item["transform"] == transform and item["sample_index"] == image
            ]
            if not values:
                raise ValueError(f"missing target-layer switch cases for {transform}, image={image}")
            image_values.append(float(np.mean(values)))
        by_transform[transform] = {
            "image_level_mean_switch_fraction": bootstrap_mean_interval(
                np.asarray(image_values), draws, seed + index
            ),
            "image_min": float(np.min(image_values)),
            "image_max": float(np.max(image_values)),
            "zero_switch_image_count": sum(value == 0.0 for value in image_values),
            "image_values": [
                {"sample_index": image, "mean_across_seeds": value}
                for image, value in zip(images, image_values)
            ],
        }
    return {
        "method": {
            "unit": "image",
            "within_image_aggregation": "equal mean across seeds",
            "bootstrap": "images resampled with replacement",
        },
        "by_transform": by_transform,
    }


def _prepare_inputs(
    paths: list[Path], transforms: list[dict[str, Any]], resolution: int
) -> tuple[dict[tuple[int, str], dict[str, Any]], list[dict[str, Any]]]:
    prepared = {}
    audit = []
    for sample_index, path in enumerate(paths):
        with Image.open(path) as opened:
            original = opened.convert("RGB")
        for spec in transforms:
            transformed = _apply_transform(original, spec)
            canvas, meta = letterbox(transformed, resolution)
            record = {
                "sample_index": sample_index,
                "sample_name": path.name,
                "transform": spec["name"],
                "spec": spec,
                "geometry": meta.to_dict(),
                "model_input_rgb_bytes_sha256": hashlib.sha256(canvas.tobytes()).hexdigest(),
                "model_input_shape_hwc": [int(value) for value in canvas.shape],
                "normalization": "RGB uint8 / 255.0",
            }
            prepared[(sample_index, spec["name"])] = {"canvas": canvas, "meta": meta}
            audit.append(record)
    return prepared, audit


def archive_selected_inputs(paths: list[Path], run_dir: Path) -> list[dict[str, Any]]:
    """Archive selected images and explicitly retain label-missing background samples."""

    destination = run_dir / "inputs"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for sample_index, path in enumerate(paths):
        label = label_path_for_image(path)
        image_relative = Path("inputs") / f"sample-{sample_index}--{path.name}"
        shutil.copyfile(path, run_dir / image_relative)
        with Image.open(path) as image:
            width, height = image.size
        label_exists = label.is_file()
        label_relative = Path("inputs") / f"sample-{sample_index}--{label.name}" if label_exists else None
        if label_relative is not None:
            shutil.copyfile(label, run_dir / label_relative)
        boxes = parse_yolo_labels(label) if label_exists else []
        records.append(
            {
                "sample_index": sample_index,
                "name": path.name,
                "original_width": width,
                "original_height": height,
                "sha256": sha256_file(path),
                "label_status": "PRESENT" if label_exists else "MISSING_DATASET_LABEL",
                "label_sha256": sha256_file(label) if label_exists else None,
                "ground_truth_box_count": len(boxes),
                "image_artifact": image_relative.as_posix(),
                "label_artifact": label_relative.as_posix() if label_relative is not None else None,
            }
        )
    return records


def _save_overview(
    attribution: dict[str, Any], switches: dict[str, Any], target_cases: list[dict[str, Any]], output: Path
) -> None:
    width, height = 1800, 1160
    canvas = Image.new("RGB", (width, height), "#050c1b")
    draw = ImageDraw.Draw(canvas)
    title, section, body, small, metric = _font(38), _font(24), _font(18), _font(15), _font(30, mono=True)
    draw.rounded_rectangle((40, 34, width - 40, height - 34), radius=30, fill="#09172d", outline="#21678d", width=3)
    draw.text((80, 68), "E3 P2  /  COCO128 IMAGE-LEVEL AUDIT", fill="#00e5ff", font=title)
    draw.text((82, 120), "32 hash-selected images · 3 seeds · 3 appearance families · CPU", fill="#9bb8d6", font=body)
    cards = [
        ("32", "image-level units"),
        (
            f"{attribution['target_rank_one_case_count']}/{attribution['case_count']}",
            "model.16 rank #1 · image×seed×transform",
        ),
        (
            f"{attribution['target_rank_one_leave_one_out_count']}/{attribution['leave_one_out_count']}",
            "rank #1 · leave-one-image-out",
        ),
        ("10,000", "image-bootstrap draws / transform"),
    ]
    for index, (value, label) in enumerate(cards):
        left = 82 + index * 410
        draw.rounded_rectangle((left, 170, left + 375, 286), radius=18, fill="#0d223e", outline="#204d70", width=2)
        draw.text((left + 20, 191), value, fill="#63e6a7", font=metric)
        draw.text((left + 20, 247), label, fill="#a9c1da", font=small)

    transforms = list(attribution["bootstrap_target_share_by_transform"])
    draw.text((82, 340), "Model.16 share · image bootstrap", fill="#eaf4ff", font=section)
    draw.text((82, 377), "Share of four layer-level image-mean MAEs; 95% percentile interval", fill="#829fbd", font=small)
    for row, transform in enumerate(transforms):
        item = attribution["bootstrap_target_share_by_transform"][transform]
        y = 438 + row * 116
        mean = item["observed_target_share"]
        low, high = item["percentile_95_interval"]
        draw.text((82, y), transform.replace("_", " "), fill="#b5cce2", font=body)
        draw.rounded_rectangle((305, y - 2, 925, y + 40), radius=9, fill="#10233c")
        draw.rounded_rectangle((305, y - 2, 305 + int(620 * mean), y + 40), radius=9, fill="#05bdd4")
        draw.text((330, y + 8), f"{mean * 100:5.2f}%", fill="#ffffff", font=body)
        draw.text((305, y + 58), f"95% [{low * 100:.2f}%, {high * 100:.2f}%]", fill="#d5b7ff", font=body)

    draw.text((1050, 340), "Target-layer switch rate · image bootstrap", fill="#eaf4ff", font=section)
    draw.text((1050, 377), "Per image: equal mean across three seeds", fill="#829fbd", font=small)
    max_rate = max(
        item["image_level_mean_switch_fraction"]["percentile_95_interval"][1]
        for item in switches["by_transform"].values()
    )
    for row, transform in enumerate(transforms):
        item = switches["by_transform"][transform]["image_level_mean_switch_fraction"]
        y = 438 + row * 116
        mean = item["observed_mean"]
        low, high = item["percentile_95_interval"]
        draw.text((1050, y), transform.replace("_", " "), fill="#b5cce2", font=body)
        draw.rounded_rectangle((1280, y - 2, 1660, y + 40), radius=9, fill="#10233c")
        if max_rate > 0.0:
            draw.rounded_rectangle((1280, y - 2, 1280 + int(380 * mean / max_rate), y + 40), radius=9, fill="#d65c8a")
        draw.text((1300, y + 8), f"{mean * 100:5.2f}%", fill="#ffffff", font=body)
        draw.text((1280, y + 58), f"95% [{low * 100:.2f}%, {high * 100:.2f}%]", fill="#d5b7ff", font=body)

    worst = sorted(target_cases, key=lambda item: item["dominant_switch_fraction"], reverse=True)[:5]
    draw.text((82, 815), "Highest target-layer case switch rates", fill="#eaf4ff", font=section)
    for index, item in enumerate(worst):
        left = 82 + index * 330
        draw.rounded_rectangle((left, 865, left + 300, 974), radius=14, fill="#0d223e", outline="#204d70", width=2)
        draw.text((left + 15, 883), item["transform"].replace("_", " "), fill="#d5b7ff", font=small)
        draw.text((left + 15, 914), f"image {item['sample_index']} · seed {item['seed']}", fill="#dcecff", font=small)
        draw.text((left + 15, 943), f"switch {item['dominant_switch_fraction'] * 100:.2f}%", fill="#63e6a7", font=body)
    draw.rounded_rectangle((82, 1018, 1710, 1095), radius=16, fill="#081326", outline="#213c5a", width=2)
    draw.text(
        (108, 1041),
        "Guardrail: deterministic coco128 subset and random initialization; intervals describe image heterogeneity, not learned accuracy.",
        fill="#ffcc66",
        font=small,
    )
    canvas.save(output)


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
    (run_dir / "command.txt").write_text("run_image_scale.cmd\n", encoding="utf-8")
    started = time.perf_counter()
    source_root = (PROJECT_ROOT / config["runtime_root"]).resolve()
    checks = _verify_source(source_root, config["source_fingerprints"])
    sys.path.insert(0, str(source_root))
    os.chdir(PROJECT_ROOT)
    import torch
    from ultralytics import YOLO

    expected_count = int(config["expected_dataset_image_count"])
    all_paths, dataset_meta = _resolve_images(config["dataset"], config["dataset_split"], list(range(expected_count)))
    if dataset_meta["split_image_count"] != expected_count:
        raise RuntimeError(
            f"dataset image count drift: {dataset_meta['split_image_count']} != {expected_count}"
        )
    paths, selection = deterministic_image_selection(
        all_paths, int(config["selected_image_count"]), config["selection_salt"]
    )
    sample_indices = list(range(len(paths)))
    archived_inputs = archive_selected_inputs(paths, run_dir)
    prepared, canvas_audit = _prepare_inputs(paths, config["transformations"], int(config["resolution"]))
    transform_names = [item["name"] for item in config["transformations"]]
    candidate_names = transform_names[1:]
    effects = _summarize_input_effects(prepared, sample_indices, transform_names)
    for item, selected in zip(archived_inputs, selection):
        item.update(
            {
                "dataset_sorted_index": selected["dataset_sorted_index"],
                "selection_sha256": selected["selection_sha256"],
            }
        )
    input_record = {
        **dataset_meta,
        "selection_method": "ascending SHA-256(selection_salt + NUL + unique filename)",
        "selection_salt": config["selection_salt"],
        "selected_image_count": len(paths),
        "selection": selection,
        "images": archived_inputs,
        "selected_image_and_label_set_sha256": hashlib.sha256(
            "\n".join(
                f"{item['sample_index']}:{item['sha256']}:{item['label_sha256'] or 'NO_LABEL'}"
                for item in archived_inputs
            ).encode("utf-8")
        ).hexdigest(),
    }
    write_json(run_dir / "input.json", input_record)
    write_json(run_dir / "model-input-audit.json", canvas_audit)
    write_json(run_dir / "transformation-effect.json", effects)
    write_json(run_dir / "source-fingerprint-checks.json", checks)

    family = config["family"]
    target_module = config["target_module"]
    resolution = int(config["resolution"])
    profile = config["profile"]
    arrays: dict[str, np.ndarray] = {}
    captures = []
    comparisons = []
    target_cases = []
    detector_comparisons = []
    all_margins: dict[str, list[np.ndarray]] = defaultdict(list)
    all_switches: dict[str, list[np.ndarray]] = defaultdict(list)
    invariants = {}
    module_order: list[str] | None = None
    logger.info(
        "scope=MoA image-level scale dataset=%s images=%d transforms=%s seeds=%s",
        config["dataset"], len(paths), transform_names, config["seeds"],
    )

    for seed in config["seeds"]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        wrapper = YOLO(source_root / profile["model_config"])
        model = wrapper.model.to("cpu").eval()
        representative = _tensor(prepared[(0, "identity")]["canvas"], torch)
        with torch.inference_mode():
            baseline_output = _detach_tree(model(representative))
        hooked_output, first_records, current_order = _capture_once(
            model, family, profile["router_class"], representative, torch
        )
        _, repeat_records, repeat_order = _capture_once(
            model, family, profile["router_class"], representative, torch
        )
        hook_delta = max_output_delta(baseline_output, hooked_output)
        repeat_weight_delta = max(
            float(np.max(np.abs(first.weights - second.weights)))
            for first, second in zip(first_records, repeat_records)
        )
        repeat_logit_delta = max(
            float(np.max(np.abs(first.logits - second.logits)))
            for first, second in zip(first_records, repeat_records)
        )
        if hook_delta != 0.0 or repeat_weight_delta != 0.0 or repeat_logit_delta != 0.0 or current_order != repeat_order:
            raise RuntimeError(f"representative invariant failure for seed={seed}")
        if target_module not in current_order or (module_order is not None and current_order != module_order):
            raise RuntimeError(f"router module order or target changed for seed={seed}")
        module_order = current_order
        invariants[f"seed-{seed}"] = {
            "hook_output_max_abs_delta": hook_delta,
            "repeat_weight_max_abs_delta": repeat_weight_delta,
            "repeat_logit_max_abs_delta": repeat_logit_delta,
            "module_order": current_order,
            "hook_cleanup": True,
            "status": "PASS",
        }

        for sample_index, path in enumerate(paths):
            restored = {}
            raw_weights = {}
            detector_outputs = {}
            for transform in transform_names:
                prepared_item = prepared[(sample_index, transform)]
                tensor = _tensor(prepared_item["canvas"], torch)
                current_output, records, registered = _capture_once(
                    model, family, profile["router_class"], tensor, torch
                )
                if config.get("record_detector_outputs"):
                    detector_outputs[transform] = current_output
                if registered != module_order:
                    raise RuntimeError("module order changed during image-level capture")
                meta: LetterboxMeta = prepared_item["meta"]
                for record in records:
                    weights = record.weights[0]
                    aligned, restoration = restore_probability_stack(weights, meta)
                    restored[(transform, record.module_name)] = aligned
                    raw_weights[(transform, record.module_name)] = weights
                    key = (
                        f"{family}__seed-{seed}__sample-{sample_index}__{transform}__"
                        f"{_slug(record.module_name)}__weights"
                    )
                    arrays[key] = record.weights
                    captures.append(
                        {
                            "family": family,
                            "seed": seed,
                            "sample_index": sample_index,
                            "sample_name": path.name,
                            "resolution": resolution,
                            "transformation": transform,
                            "module": record.module_name,
                            "module_type": record.module_type,
                            "source_shape": record.validation["shape"],
                            "geometry": meta.to_dict(),
                            "router_validation": record.validation,
                            "restoration_validation": restoration,
                            "raw_weights_key": key,
                        }
                    )
            for module in module_order:
                reference = restored[("identity", module)]
                for transform in candidate_names:
                    comparisons.append(
                        {
                            "comparison_type": f"appearance_{transform}",
                            "family": family,
                            "seed": seed,
                            "sample_index": sample_index,
                            "module": module,
                            "reference_resolution": resolution,
                            "candidate_resolution": resolution,
                            "alignment": "same geometry; both maps restored to original-image pixels",
                            "metrics": probability_map_comparison(reference, restored[(transform, module)]),
                        }
                    )
            if config.get("record_detector_outputs"):
                for transform in candidate_names:
                    detector_comparisons.append(
                        {
                            "seed": seed,
                            "sample_index": sample_index,
                            "transform": transform,
                            "reference": "identity",
                            "metrics": detector_output_comparison(
                                detector_outputs["identity"], detector_outputs[transform]
                            ),
                        }
                    )
            reference_target = raw_weights[("identity", target_module)]
            target_meta: LetterboxMeta = prepared[(sample_index, "identity")]["meta"]
            label_path = label_path_for_image(path)
            boxes = parse_yolo_labels(label_path) if label_path.is_file() else []
            masks = token_region_masks(
                boxes, target_meta, reference_target.shape[1], reference_target.shape[2]
            )
            valid = ~masks["padding"]
            if seed == config["seeds"][0]:
                arrays[f"{family}__sample-{sample_index}__{_slug(target_module)}__mask-padding"] = masks[
                    "padding"
                ]
            ordered = np.sort(reference_target.astype(np.float64), axis=0)
            margin = ordered[-1] - ordered[-2]
            for transform in candidate_names:
                candidate_target = raw_weights[(transform, target_module)]
                difference = np.abs(reference_target.astype(np.float64) - candidate_target.astype(np.float64))
                switched = np.argmax(reference_target, axis=0) != np.argmax(candidate_target, axis=0)
                valid_margin = margin[valid]
                valid_switched = switched[valid]
                threshold_30 = float(np.quantile(valid_margin, 0.30))
                low_30 = valid_margin <= threshold_30
                record = {
                    "transform": transform,
                    "seed": seed,
                    "sample_index": sample_index,
                    "valid_token_count": int(valid.sum()),
                    "padding_token_count": int(masks["padding"].sum()),
                    "probability_mae": float(difference[:, valid].mean()),
                    "dominant_switch_count": int(valid_switched.sum()),
                    "dominant_switch_fraction": float(valid_switched.mean()),
                    "reference_margin_mean": float(valid_margin.mean()),
                    "within_case_margin_p30": threshold_30,
                    "switch_count_at_or_below_margin_p30": int(valid_switched[low_30].sum()),
                    "switch_count_above_margin_p30": int(valid_switched[~low_30].sum()),
                }
                target_cases.append(record)
                all_margins[transform].append(valid_margin)
                all_switches[transform].append(valid_switched)
                all_margins["overall"].append(valid_margin)
                all_switches["overall"].append(valid_switched)
            if (sample_index + 1) % 8 == 0:
                logger.info("seed=%d progress=%d/%d", seed, sample_index + 1, len(paths))
        del model, wrapper

    np.savez_compressed(run_dir / "image-scale-routing-raw.npz", **arrays)
    write_json(run_dir / "spatial-captures.json", captures)
    write_json(run_dir / "image-scale-comparisons.json", comparisons)
    write_json(run_dir / "target-layer-cases.json", target_cases)
    if config.get("record_detector_outputs"):
        write_json(run_dir / "detector-output-comparisons.json", detector_comparisons)
    attribution = summarize_image_level_attribution(
        comparisons,
        candidate_names,
        target_module,
        int(config["bootstrap_draws"]),
        int(config["bootstrap_seed"]),
    )
    switch_summary = summarize_image_level_switches(
        target_cases, candidate_names, int(config["bootstrap_draws"]), int(config["bootstrap_seed"]) + 100
    )
    margin_deciles = {
        key: margin_bin_summary(np.concatenate(all_margins[key]), np.concatenate(all_switches[key]))
        for key in [*candidate_names, "overall"]
    }
    write_json(run_dir / "image-level-attribution.json", attribution)
    write_json(run_dir / "image-level-switches.json", switch_summary)
    write_json(
        run_dir / "margin-deciles.json",
        {
            "method": {
                "scope": "target-module raw tokens; letterbox padding excluded",
                "unit": "token-comparison exposure; descriptive localization only",
            },
            "by_transform_and_overall": margin_deciles,
        },
    )
    write_json(run_dir / "invariants.json", invariants)
    write_json(run_dir / "environment.json", environment(torch, source_root, PROJECT_ROOT))
    _save_overview(attribution, switch_summary, target_cases, run_dir / "image-level-overview.png")
    summary = {
        "status": "PASS",
        "scope": "CPU-only 32-image MoA appearance audit with image-level resampling",
        "run_id": config["run_id"],
        "tool_source": project_source,
        "official_locked_base_ref": config["official_locked_base_ref"],
        "official_runtime_ref": config["official_runtime_ref"],
        "source_fingerprint_validation": "PASS",
        "source_fingerprint_count": len(checks),
        "device": "cpu",
        "dataset": config["dataset"],
        "dataset_split_image_count": dataset_meta["split_image_count"],
        "selected_image_count": len(paths),
        "selection_salt": config["selection_salt"],
        "seeds": config["seeds"],
        "resolution": resolution,
        "transformations": config["transformations"],
        "transformation_effect": effects["by_transform"],
        "spatial_capture_count": len(captures),
        "raw_array_count": len(arrays),
        "aligned_comparison_count": len(comparisons),
        "target_case_count": len(target_cases),
        "detector_output_comparison_count": len(detector_comparisons),
        "image_level_attribution": {
            "target_rank_one_case_count": attribution["target_rank_one_case_count"],
            "case_count": attribution["case_count"],
            "target_rank_one_image_count": attribution["target_rank_one_image_count"],
            "image_count": attribution["image_count"],
            "target_rank_one_leave_one_out_count": attribution["target_rank_one_leave_one_out_count"],
            "leave_one_out_count": attribution["leave_one_out_count"],
        },
        "representative_invariants": invariants,
        "restoration_validation": {
            "max_pre_normalization_expert_sum_error": max(
                item["restoration_validation"]["pre_normalization_max_expert_sum_error"] for item in captures
            ),
            "max_post_normalization_expert_sum_error": max(
                item["restoration_validation"]["post_normalization_max_expert_sum_error"] for item in captures
            ),
        },
        "interpretation_boundary": "deterministic coco128 subset and random initialization; no learned accuracy claim",
        "duration_seconds_observation_only": time.perf_counter() - started,
    }
    write_json(run_dir / "summary.json", summary)
    logger.info(
        "status=PASS images=%d captures=%d comparisons=%d target_cases=%d",
        len(paths), len(captures), len(comparisons), len(target_cases),
    )
    for handler in logger.handlers:
        handler.flush()
    evidence_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    if evidence_bytes > int(config["max_evidence_bytes"]):
        raise RuntimeError(f"evidence budget exceeded: {evidence_bytes} > {config['max_evidence_bytes']}")
    summary["evidence_bytes_before_manifest"] = evidence_bytes
    write_json(run_dir / "summary.json", summary)
    final_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    if final_bytes > int(config["max_evidence_bytes"]):
        raise RuntimeError(f"final evidence budget exceeded: {final_bytes} > {config['max_evidence_bytes']}")
    write_manifest(run_dir)
    if update_latest:
        latest = PROJECT_ROOT / "artifacts" / "p2" / "IMAGE_SCALE_LATEST.txt"
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(config["run_id"] + "\n", encoding="utf-8")
    return run_dir
