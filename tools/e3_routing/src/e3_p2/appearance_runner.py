"""CPU-only appearance-perturbation diagnostics for P2 spatial routers."""

from __future__ import annotations

import hashlib
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from .capture import max_output_delta
from .geometry import LetterboxMeta, letterbox
from .io_utils import environment, sha256_file, write_json, write_manifest
from .regions import label_path_for_image, parse_yolo_labels, token_region_masks
from .robustness_runner import _archive_inputs, _capture_once, _logger
from .runner import (
    PROJECT_ROOT,
    RUN_ID_PATTERN,
    _detach_tree,
    _resolve_images,
    _slug,
    _verify_project_source_state,
    _verify_source,
)
from .stability import aggregate_stability_comparisons, probability_map_comparison, restore_probability_stack


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
        raise ValueError("appearance diagnostics require CPU and resolution >= 32")
    seeds = [int(value) for value in config.get("seeds", [])]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain at least two unique integers")
    config["seeds"] = seeds
    indices = [int(value) for value in config.get("sample_indices", [])]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("sample_indices must be non-empty and unique")
    config["sample_indices"] = indices
    if set(config.get("spatial_profiles", {})) != {"mot", "moa"}:
        raise ValueError("spatial_profiles must be exactly MoT and MoA")
    transforms = config.get("transformations")
    if not isinstance(transforms, list) or len(transforms) < 2:
        raise ValueError("transformations must contain identity and at least one perturbation")
    names = [str(item.get("name", "")) for item in transforms if isinstance(item, dict)]
    if len(names) != len(transforms) or len(set(names)) != len(names) or names[0] != "identity":
        raise ValueError("transform names must be unique and identity must be first")
    for index, spec in enumerate(transforms):
        kind = spec.get("kind")
        if index == 0 and spec != {"name": "identity", "kind": "identity"}:
            raise ValueError("identity transform contract changed")
        if kind in {"brightness", "contrast"}:
            factor = float(spec.get("factor", 0.0))
            if not 0.5 <= factor <= 1.5 or factor == 1.0:
                raise ValueError(f"{kind} factor must be in [0.5,1.5] and not equal one")
        elif kind == "gaussian_blur":
            radius = float(spec.get("radius", -1.0))
            if not 0.0 < radius <= 3.0:
                raise ValueError("gaussian blur radius must be in (0,3]")
        elif kind != "identity":
            raise ValueError(f"unsupported appearance transform: {kind}")
    return config


def _apply_transform(image: Image.Image, spec: dict[str, Any]) -> Image.Image:
    kind = spec["kind"]
    rgb = image.convert("RGB")
    if kind == "identity":
        return rgb.copy()
    if kind == "brightness":
        return ImageEnhance.Brightness(rgb).enhance(float(spec["factor"]))
    if kind == "contrast":
        return ImageEnhance.Contrast(rgb).enhance(float(spec["factor"]))
    if kind == "gaussian_blur":
        return rgb.filter(ImageFilter.GaussianBlur(radius=float(spec["radius"])))
    raise ValueError(f"unsupported appearance transform: {kind}")


def _prepare_inputs(
    paths: list[Path], sample_indices: list[int], transforms: list[dict[str, Any]], resolution: int, run_dir: Path
) -> tuple[dict[tuple[int, str], dict[str, Any]], list[dict[str, Any]]]:
    transformed_dir = run_dir / "transformed-inputs"
    canvas_dir = run_dir / "model-inputs"
    transformed_dir.mkdir(parents=True, exist_ok=True)
    canvas_dir.mkdir(parents=True, exist_ok=True)
    prepared: dict[tuple[int, str], dict[str, Any]] = {}
    audit = []
    for sample_index, path in zip(sample_indices, paths):
        with Image.open(path) as opened:
            original = opened.convert("RGB")
        for spec in transforms:
            name = spec["name"]
            transformed = _apply_transform(original, spec)
            canvas, meta = letterbox(transformed, resolution)
            transformed_relative = Path("transformed-inputs") / f"sample-{sample_index}--{name}.png"
            canvas_relative = Path("model-inputs") / f"sample-{sample_index}--{name}--{resolution}px.png"
            transformed.save(run_dir / transformed_relative)
            Image.fromarray(canvas).save(run_dir / canvas_relative)
            record = {
                "sample_index": sample_index,
                "sample_name": path.name,
                "transform": name,
                "spec": spec,
                "geometry": meta.to_dict(),
                "transformed_artifact": transformed_relative.as_posix(),
                "transformed_artifact_sha256": sha256_file(run_dir / transformed_relative),
                "model_input_artifact": canvas_relative.as_posix(),
                "model_input_artifact_sha256": sha256_file(run_dir / canvas_relative),
                "model_input_rgb_bytes_sha256": hashlib.sha256(canvas.tobytes()).hexdigest(),
                "model_input_shape_hwc": [int(value) for value in canvas.shape],
                "normalization": "RGB uint8 / 255.0",
            }
            audit.append(record)
            prepared[(sample_index, name)] = {"canvas": canvas, "meta": meta, "audit": record}
    return prepared, audit


def _tensor(canvas: np.ndarray, torch_module: Any) -> Any:
    array = canvas.astype(np.float32).transpose(2, 0, 1) / 255.0
    return torch_module.from_numpy(array).unsqueeze(0).contiguous()


def _summarize_input_effects(
    prepared: dict[tuple[int, str], dict[str, Any]], sample_indices: list[int], transform_names: list[str]
) -> dict[str, Any]:
    by_transform = {}
    records = []
    for transform_name in transform_names[1:]:
        current = []
        for sample_index in sample_indices:
            reference = prepared[(sample_index, "identity")]["canvas"].astype(np.int16)
            candidate = prepared[(sample_index, transform_name)]["canvas"].astype(np.int16)
            difference = np.abs(reference - candidate)
            record = {
                "sample_index": sample_index,
                "transform": transform_name,
                "rgb_mae_0_255": float(difference.mean()),
                "rgb_max_abs_error_0_255": int(difference.max()),
                "changed_channel_fraction": float((difference > 0).mean()),
                "model_input_changed": bool(np.any(difference)),
            }
            current.append(record)
            records.append(record)
        changed = sum(item["model_input_changed"] for item in current)
        if changed != len(sample_indices):
            raise RuntimeError(
                f"transform {transform_name} was a no-op for {len(sample_indices) - changed} configured samples"
            )
        by_transform[transform_name] = {
            "sample_count": len(current),
            "changed_sample_count": changed,
            "rgb_mae_0_255_mean": float(np.mean([item["rgb_mae_0_255"] for item in current])),
            "rgb_mae_0_255_min": float(np.min([item["rgb_mae_0_255"] for item in current])),
            "rgb_mae_0_255_max": float(np.max([item["rgb_mae_0_255"] for item in current])),
            "changed_channel_fraction_mean": float(
                np.mean([item["changed_channel_fraction"] for item in current])
            ),
        }
    return {
        "reference": "identity model-input uint8 canvas",
        "scale": "absolute RGB difference on [0,255]",
        "all_candidate_samples_changed": True,
        "by_transform": by_transform,
        "records": records,
    }


def _aggregate_region_comparisons(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        grouped[(item["comparison_type"], item["family"], item["region"])].append(item)
    metric_names = (
        "probability_mae",
        "mean_total_variation_distance",
        "mean_jensen_shannon_divergence_nats",
        "dominant_expert_agreement_fraction",
        "reference_top1_margin_mean",
    )
    result = {}
    for key, items in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "comparison_count": len(items),
            "token_count_total": sum(item["token_count"] for item in items),
        }
        for metric_name in metric_names:
            values = np.asarray([item["metrics"][metric_name] for item in items], dtype=np.float64)
            summary[metric_name] = {
                "mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())
            }
        result[f"{key[0]}:{key[1]}:{key[2]}"] = summary
    return {
        "method": {
            "unit": "one seed x sample x router-module x region comparison",
            "weighting": "each non-empty region comparison receives equal weight",
            "regions": "foreground/background from 128px feature-cell centers; letterbox padding excluded",
            "interpretation_boundary": "descriptive cold-start sensitivity, not learned robustness",
        },
        "comparison_count": len(records),
        "by_type_family_region": result,
        "records": records,
    }


def _font(size: int, *, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = ["C:/Windows/Fonts/consola.ttf"] if mono else ["C:/Windows/Fonts/segoeui.ttf"]
    candidates += ["DejaVuSansMono.ttf"] if mono else ["DejaVuSans.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _save_overview(aggregate: dict[str, Any], region: dict[str, Any], transform_names: list[str], path: Path) -> None:
    canvas = Image.new("RGB", (1800, 1120), "#050c1b")
    draw = ImageDraw.Draw(canvas)
    title, section, body = _font(38), _font(23), _font(18)
    metric, small, mono = _font(31, mono=True), _font(15), _font(16, mono=True)
    draw.rounded_rectangle((42, 36, 1758, 1084), radius=30, fill="#09172d", outline="#21678d", width=3)
    draw.text((82, 70), "E3 P2  /  APPEARANCE SENSITIVITY", fill="#00e5ff", font=title)
    draw.text((84, 122), "128px · 3 seeds · 4 images · brightness / contrast / blur", fill="#9bb8d6", font=body)
    draw.rounded_rectangle((1195, 66, 1708, 141), radius=16, fill="#221938", outline="#8f65db", width=2)
    draw.text((1220, 83), "COLD-START DIAGNOSTIC", fill="#d5b7ff", font=section)
    draw.text((1220, 113), "Not learned robustness", fill="#ffcc66", font=small)
    cards = [("576", "true spatial captures"), ("480", "aligned comparisons"), ("960", "FG/BG comparisons")]
    for index, (value, label) in enumerate(cards):
        left = 82 + index * 340
        draw.rounded_rectangle((left, 180, left + 310, 292), radius=18, fill="#0d223e", outline="#204d70", width=2)
        draw.text((left + 22, 201), value, fill="#63e6a7", font=metric)
        draw.text((left + 22, 252), label, fill="#a9c1da", font=small)

    draw.text((82, 335), "MoA original-coordinate agreement", fill="#eaf4ff", font=section)
    draw.text((82, 371), "Overall versus pixels at/above the reference 90th-margin percentile", fill="#829fbd", font=small)
    rows = []
    for name in transform_names:
        item = aggregate["by_type_family_resolution"][f"appearance_{name}:moa:128"]
        rows.append((name, item))
    for index, (name, item) in enumerate(rows):
        y = 420 + index * 70
        agreement = item["dominant_expert_agreement_fraction"]["mean"]
        high_margin = item["agreement_at_or_above_reference_margin_percentiles"]["90"]["agreement_mean"]
        draw.text((82, y + 8), name.replace("_", " "), fill="#b5cce2", font=body)
        draw.rounded_rectangle((285, y, 930, y + 38), radius=9, fill="#10233c")
        draw.rounded_rectangle((285, y, 285 + int(645 * agreement), y + 38), radius=9, fill="#4d55d8")
        draw.text((300, y + 8), f"all {agreement * 100:5.2f}%", fill="#ffffff", font=mono)
        draw.text((580, y + 8), f"P90+ {high_margin * 100:5.2f}%", fill="#d5b7ff", font=mono)
        draw.text((950, y + 8), f"MAE {item['probability_mae']['mean']:.2e}", fill="#9bb8d6", font=mono)

    draw.text((1110, 335), "MoA region-conditioned probability MAE", fill="#eaf4ff", font=section)
    draw.text((1110, 371), "Equal-weight capture summaries; padding excluded", fill="#829fbd", font=small)
    region_data = region["by_type_family_region"]
    max_mae = max(
        region_data[f"appearance_{name}:moa:{area}"]["probability_mae"]["mean"]
        for name in transform_names
        for area in ("foreground", "background")
    )
    scale = max(max_mae, 1e-12)
    for index, name in enumerate(transform_names):
        y = 425 + index * 88
        foreground = region_data[f"appearance_{name}:moa:foreground"]["probability_mae"]["mean"]
        background = region_data[f"appearance_{name}:moa:background"]["probability_mae"]["mean"]
        draw.text((1110, y), name.replace("_", " "), fill="#b5cce2", font=small)
        draw.rounded_rectangle((1300, y, 1690, y + 24), radius=6, fill="#10233c")
        draw.rounded_rectangle((1300, y, 1300 + int(390 * foreground / scale), y + 24), radius=6, fill="#d65c8a")
        draw.rounded_rectangle((1300, y + 32, 1690, y + 56), radius=6, fill="#10233c")
        draw.rounded_rectangle((1300, y + 32, 1300 + int(390 * background / scale), y + 56), radius=6, fill="#0e9fb0")
        draw.text((1310, y + 4), f"FG {foreground:.2e}", fill="#ffffff", font=small)
        draw.text((1310, y + 36), f"BG {background:.2e}", fill="#ffffff", font=small)

    draw.rounded_rectangle((82, 835, 1710, 998), radius=18, fill="#081326", outline="#213c5a", width=2)
    draw.text((110, 860), "Interpretation guardrails", fill="#a66cff", font=section)
    notes = [
        "All transformed originals and exact 128x128 model-input canvases are archived and hashed.",
        "Agreement is read with probability error and margin; near-tie argmax changes are not large distribution shifts.",
        "MoT constant-map equality remains a cold-start property, not evidence of trained appearance robustness.",
    ]
    for index, note in enumerate(notes):
        draw.text((115, 907 + index * 28), f"• {note}", fill="#c8d8e8", font=small)
    draw.text((84, 1036), "SHA-256 manifest · locked source · raw NPZ · exact appearance-input audit", fill="#63e6a7", font=small)
    canvas.save(path)


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
    (run_dir / "command.txt").write_text("run_appearance.cmd\n", encoding="utf-8")
    started = time.perf_counter()

    source_root = (PROJECT_ROOT / config["runtime_root"]).resolve()
    checks = _verify_source(source_root, config["source_fingerprints"])
    sys.path.insert(0, str(source_root))
    os.chdir(PROJECT_ROOT)
    import torch
    from ultralytics import YOLO

    paths, dataset_meta = _resolve_images(config["dataset"], config["dataset_split"], config["sample_indices"])
    archived_inputs = _archive_inputs(paths, config["sample_indices"], run_dir)
    prepared, transform_audit = _prepare_inputs(
        paths, config["sample_indices"], config["transformations"], int(config["resolution"]), run_dir
    )
    write_json(run_dir / "transformation-audit.json", transform_audit)
    input_effects = _summarize_input_effects(
        prepared,
        config["sample_indices"],
        [item["name"] for item in config["transformations"]],
    )
    write_json(run_dir / "transformation-effect.json", input_effects)
    input_record = {
        **dataset_meta,
        "selected_image_count": len(paths),
        "images": archived_inputs,
        "image_and_annotation_set_sha256": hashlib.sha256(
            "\n".join(f"{item['sample_index']}:{item['sha256']}:{item['label_sha256']}" for item in archived_inputs).encode("utf-8")
        ).hexdigest(),
    }
    write_json(run_dir / "input.json", input_record)
    write_json(run_dir / "source-fingerprint-checks.json", checks)

    resolution = int(config["resolution"])
    transform_names = [item["name"] for item in config["transformations"]]
    candidate_names = transform_names[1:]
    first_seed = config["seeds"][0]
    arrays: dict[str, np.ndarray] = {}
    capture_metadata: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    region_comparisons: list[dict[str, Any]] = []
    invariants: dict[str, Any] = {}
    logger.info(
        "scope=CPU appearance sensitivity resolution=%d transforms=%s seeds=%s samples=%s",
        resolution, transform_names, config["seeds"], config["sample_indices"],
    )

    for family, profile in config["spatial_profiles"].items():
        for seed in config["seeds"]:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            wrapper = YOLO(source_root / profile["model_config"])
            model = wrapper.model.to("cpu").eval()
            representative = _tensor(prepared[(config["sample_indices"][0], "identity")]["canvas"], torch)
            with torch.inference_mode():
                baseline_output = _detach_tree(model(representative))
            hooked_output, first_records, module_order = _capture_once(
                model, family, profile["router_class"], representative, torch
            )
            _, repeat_records, repeat_order = _capture_once(model, family, profile["router_class"], representative, torch)
            hook_delta = max_output_delta(baseline_output, hooked_output)
            repeat_weight_delta = max(
                float(np.max(np.abs(first.weights - second.weights)))
                for first, second in zip(first_records, repeat_records)
            )
            repeat_logit_delta = max(
                float(np.max(np.abs(first.logits - second.logits)))
                for first, second in zip(first_records, repeat_records)
            )
            repeat_indices_equal = all(
                first.indices is None or np.array_equal(first.indices, second.indices)
                for first, second in zip(first_records, repeat_records)
            )
            if hook_delta != 0.0 or module_order != repeat_order or repeat_weight_delta != 0.0 or repeat_logit_delta != 0.0 or not repeat_indices_equal:
                raise RuntimeError(f"representative invariants failed for family={family}, seed={seed}")
            invariants[f"{family}:seed-{seed}"] = {
                "hook_output_max_abs_delta": hook_delta,
                "repeat_weight_max_abs_delta": repeat_weight_delta,
                "repeat_logit_max_abs_delta": repeat_logit_delta,
                "repeat_indices_equal": repeat_indices_equal if family == "mot" else None,
                "module_order": module_order,
                "hook_cleanup": True,
                "status": "PASS",
            }

            for sample_index, path in zip(config["sample_indices"], paths):
                boxes = parse_yolo_labels(label_path_for_image(path))
                restored: dict[tuple[str, str], np.ndarray] = {}
                raw_weights: dict[tuple[str, str], np.ndarray] = {}
                for transform_name in transform_names:
                    prepared_item = prepared[(sample_index, transform_name)]
                    tensor = _tensor(prepared_item["canvas"], torch)
                    _, records, registered = _capture_once(model, family, profile["router_class"], tensor, torch)
                    if registered != module_order:
                        raise RuntimeError(f"module order changed for family={family}, transform={transform_name}")
                    meta: LetterboxMeta = prepared_item["meta"]
                    for record in records:
                        weights = record.weights[0]
                        aligned, restoration_validation = restore_probability_stack(weights, meta)
                        restored[(transform_name, record.module_name)] = aligned
                        raw_weights[(transform_name, record.module_name)] = weights
                        prefix = (
                            f"{family}__seed-{seed}__sample-{sample_index}__{transform_name}__"
                            f"{_slug(record.module_name)}"
                        )
                        weight_key, logit_key = f"{prefix}__weights", f"{prefix}__logits"
                        arrays[weight_key], arrays[logit_key] = record.weights, record.logits
                        index_key = None
                        if record.indices is not None:
                            index_key = f"{prefix}__indices"
                            arrays[index_key] = record.indices
                        capture_metadata.append(
                            {
                                "family": family,
                                "seed": seed,
                                "sample_index": sample_index,
                                "sample_name": path.name,
                                "resolution": resolution,
                                "transformation": transform_name,
                                "module": record.module_name,
                                "module_type": record.module_type,
                                "source_shape": record.validation["shape"],
                                "geometry": meta.to_dict(),
                                "router_validation": record.validation,
                                "restoration_validation": restoration_validation,
                                "raw_keys": {"weights": weight_key, "logits": logit_key, "indices": index_key},
                            }
                        )
                for module in module_order:
                    reference = restored[("identity", module)]
                    reference_weights = raw_weights[("identity", module)]
                    meta = prepared[(sample_index, "identity")]["meta"]
                    masks = token_region_masks(boxes, meta, reference_weights.shape[1], reference_weights.shape[2])
                    if seed == first_seed:
                        mask_prefix = f"{family}__sample-{sample_index}__{_slug(module)}"
                        for region_name in ("foreground", "background", "padding"):
                            arrays[f"{mask_prefix}__mask-{region_name}"] = masks[region_name]
                    for transform_name in candidate_names:
                        comparison_type = f"appearance_{transform_name}"
                        comparisons.append(
                            {
                                "comparison_type": comparison_type,
                                "family": family,
                                "seed": seed,
                                "sample_index": sample_index,
                                "module": module,
                                "reference_resolution": resolution,
                                "candidate_resolution": resolution,
                                "alignment": "same geometry; both maps restored to original-image pixels",
                                "metrics": probability_map_comparison(reference, restored[(transform_name, module)]),
                            }
                        )
                        candidate_weights = raw_weights[(transform_name, module)]
                        for region_name in ("foreground", "background"):
                            mask = masks[region_name]
                            if not mask.any():
                                continue
                            region_comparisons.append(
                                {
                                    "comparison_type": comparison_type,
                                    "family": family,
                                    "seed": seed,
                                    "sample_index": sample_index,
                                    "module": module,
                                    "region": region_name,
                                    "token_count": int(mask.sum()),
                                    "metrics": probability_map_comparison(
                                        reference_weights[:, mask].reshape(reference_weights.shape[0], 1, -1),
                                        candidate_weights[:, mask].reshape(candidate_weights.shape[0], 1, -1),
                                    ),
                                }
                            )
                logger.info("family=%s seed=%d sample=%d captures=%d", family, seed, sample_index, len(module_order) * len(transform_names))
            del model, wrapper

    np.savez_compressed(run_dir / "appearance-routing-raw.npz", **arrays)
    write_json(run_dir / "spatial-captures.json", capture_metadata)
    write_json(run_dir / "appearance-stability-comparisons.json", comparisons)
    aggregate = aggregate_stability_comparisons(comparisons)
    write_json(run_dir / "appearance-stability-aggregate.json", aggregate)
    region_aggregate = _aggregate_region_comparisons(region_comparisons)
    write_json(run_dir / "appearance-region-analysis.json", region_aggregate)
    write_json(run_dir / "invariants.json", invariants)
    write_json(run_dir / "environment.json", environment(torch, source_root, PROJECT_ROOT))
    _save_overview(aggregate, region_aggregate, candidate_names, run_dir / "appearance-overview.png")
    summary = {
        "status": "PASS",
        "scope": "CPU-only brightness, contrast and blur sensitivity for true MoT/MoA spatial router probabilities",
        "run_id": config["run_id"],
        "official_locked_base_ref": config["official_locked_base_ref"],
        "official_runtime_ref": config["official_runtime_ref"],
        "tool_source": project_source,
        "source_fingerprint_validation": "PASS",
        "source_fingerprint_count": len(checks),
        "device": "cpu",
        "seeds": config["seeds"],
        "resolution": resolution,
        "transformations": config["transformations"],
        "transformation_effect": input_effects["by_transform"],
        "sample_count": len(paths),
        "spatial_capture_count": len(capture_metadata),
        "raw_array_count": len(arrays),
        "aligned_comparison_count": len(comparisons),
        "region_comparison_count": len(region_comparisons),
        "representative_invariants": invariants,
        "restoration_validation": {
            "max_pre_normalization_expert_sum_error": max(
                item["restoration_validation"]["pre_normalization_max_expert_sum_error"] for item in capture_metadata
            ),
            "max_post_normalization_expert_sum_error": max(
                item["restoration_validation"]["post_normalization_max_expert_sum_error"] for item in capture_metadata
            ),
        },
        "interpretation_boundary": "random initialization; appearance sensitivity and pipeline evidence only",
        "status_semantics": "PASS means input audit, capture, alignment, determinism and evidence integrity checks completed",
        "duration_seconds_observation_only": time.perf_counter() - started,
    }
    write_json(run_dir / "summary.json", summary)
    for handler in logger.handlers:
        handler.flush()
    write_manifest(run_dir)
    if update_latest:
        latest = PROJECT_ROOT / "artifacts" / "p2" / "APPEARANCE_LATEST.txt"
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(config["run_id"] + "\n", encoding="utf-8")
    logger.info(
        "status=PASS captures=%d comparisons=%d region_comparisons=%d arrays=%d",
        len(capture_metadata), len(comparisons), len(region_comparisons), len(arrays),
    )
    for handler in logger.handlers:
        handler.flush()
    write_manifest(run_dir)
    return run_dir
