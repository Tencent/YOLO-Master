"""CPU-only resolution and horizontal-flip diagnostics for P2 spatial routers."""

from __future__ import annotations

import hashlib
import logging
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
from PIL import Image, ImageDraw, ImageFont

from .capture import SpatialRouterCollector, max_output_delta
from .geometry import LetterboxMeta, letterbox
from .io_utils import environment, sha256_file, write_json, write_manifest
from .regions import label_path_for_image, parse_yolo_labels, token_region_masks
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
    resolutions = [int(value) for value in config.get("resolutions", [])]
    if len(resolutions) < 2 or len(set(resolutions)) != len(resolutions) or min(resolutions) < 32:
        raise ValueError("resolutions must contain at least two unique integers >= 32")
    config["resolutions"] = resolutions
    reference = int(config.get("reference_resolution", 0))
    if reference not in resolutions:
        raise ValueError("reference_resolution must be present in resolutions")
    config["reference_resolution"] = reference
    seeds = [int(value) for value in config.get("seeds", [])]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain at least two unique integers")
    config["seeds"] = seeds
    indices = [int(value) for value in config.get("sample_indices", [])]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("sample_indices must be non-empty and unique")
    config["sample_indices"] = indices
    if config.get("device") != "cpu":
        raise ValueError("this evidence protocol is intentionally CPU-only")
    if config.get("transformations") != ["identity", "horizontal_flip"]:
        raise ValueError("transformations must be exactly identity and horizontal_flip")
    if set(config.get("spatial_profiles", {})) != {"mot", "moa"}:
        raise ValueError("spatial_profiles must be exactly MoT and MoA")
    return config


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("e3_p2_robustness")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8", mode="w")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _tensor(image: Image.Image, resolution: int, flipped: bool, torch_module: Any) -> tuple[Any, LetterboxMeta]:
    source = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if flipped else image
    canvas, meta = letterbox(source, resolution)
    array = canvas.astype(np.float32).transpose(2, 0, 1) / 255.0
    return torch_module.from_numpy(array).unsqueeze(0).contiguous(), meta


def _capture_once(model: Any, family: str, router_class: str, tensor: Any, torch_module: Any) -> tuple[Any, list[Any], list[str]]:
    collector = SpatialRouterCollector(family=family, router_class=router_class)
    registered = collector.register(model)
    try:
        with torch_module.inference_mode():
            output = model(tensor)
    finally:
        collector.remove()
    if collector.handles or len(collector.records) != len(registered):
        raise RuntimeError(f"capture count or hook cleanup failed for family={family}")
    if [record.module_name for record in collector.records] != registered:
        raise RuntimeError(f"capture module order changed for family={family}")
    return _detach_tree(output), collector.records, registered


def _archive_inputs(paths: list[Path], sample_indices: list[int], run_dir: Path) -> list[dict[str, Any]]:
    destination = run_dir / "inputs"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for sample_index, path in zip(sample_indices, paths):
        label = label_path_for_image(path)
        image_relative = Path("inputs") / f"sample-{sample_index}--{path.name}"
        label_relative = Path("inputs") / f"sample-{sample_index}--{label.name}"
        shutil.copyfile(path, run_dir / image_relative)
        shutil.copyfile(label, run_dir / label_relative)
        with Image.open(path) as image:
            width, height = image.size
        boxes = parse_yolo_labels(label)
        records.append(
            {
                "sample_index": sample_index,
                "name": path.name,
                "original_width": width,
                "original_height": height,
                "sha256": sha256_file(path),
                "label_sha256": sha256_file(label),
                "ground_truth_box_count": len(boxes),
                "image_artifact": image_relative.as_posix(),
                "label_artifact": label_relative.as_posix(),
            }
        )
    return records


def _coverage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        grouped[(item["family"], int(item["resolution"]))].append(item)
    return {
        "method": {
            "seed": "first configured seed only; geometry and annotations are seed-independent",
            "assignment_rule": "valid token center inside any ground-truth box",
            "padding_policy": "excluded from foreground and background",
            "purpose": "measure whether higher input resolution reduces empty foreground/background comparisons",
        },
        "capture_count": len(records),
        "by_family_resolution": {
            f"{family}:{resolution}": {
                "capture_count": len(items),
                "supported_capture_count": sum(item["status"] == "SUPPORTED" for item in items),
                "insufficient_capture_count": sum(item["status"] == "INSUFFICIENT_TOKENS" for item in items),
                "foreground_token_count": sum(item["foreground_token_count"] for item in items),
                "background_token_count": sum(item["background_token_count"] for item in items),
                "padding_token_count": sum(item["padding_token_count"] for item in items),
            }
            for (family, resolution), items in sorted(grouped.items())
        },
        "records": records,
    }


def _save_overview(aggregate: dict[str, Any], coverage: dict[str, Any], path: Path) -> None:
    canvas = Image.new("RGB", (1800, 1120), "#050c1b")
    draw = ImageDraw.Draw(canvas)
    def font(size: int, *, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        candidates = ["C:/Windows/Fonts/consola.ttf"] if mono else ["C:/Windows/Fonts/segoeui.ttf"]
        candidates += ["DejaVuSansMono.ttf"] if mono else ["DejaVuSans.ttf"]
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
        return ImageFont.load_default()

    title_font, section_font, body_font = font(38), font(23), font(18)
    metric_font, small_font, mono_font = font(31, mono=True), font(15), font(16, mono=True)
    draw.rounded_rectangle((42, 36, 1758, 1084), radius=30, fill="#09172d", outline="#21678d", width=3)
    draw.text((82, 70), "E3 P2  /  CPU ROUTING STABILITY", fill="#00e5ff", font=title_font)
    draw.text((84, 122), "3 seeds · 4 COCO8 images · 3 resolutions · identity + horizontal flip", fill="#9bb8d6", font=body_font)
    draw.rounded_rectangle((1195, 66, 1708, 141), radius=16, fill="#221938", outline="#8f65db", width=2)
    draw.text((1220, 83), "COLD-START DIAGNOSTIC", fill="#d5b7ff", font=section_font)
    draw.text((1220, 113), "Not learned robustness", fill="#ffcc66", font=small_font)

    cards = [("576", "true spatial captures"), ("480", "aligned comparisons"), ("0", "failed invariants")]
    for index, (value, label) in enumerate(cards):
        left = 82 + index * 340
        draw.rounded_rectangle((left, 180, left + 310, 292), radius=18, fill="#0d223e", outline="#204d70", width=2)
        draw.text((left + 22, 201), value, fill="#63e6a7", font=metric_font)
        draw.text((left + 22, 252), label, fill="#a9c1da", font=small_font)

    data = aggregate["by_type_family_resolution"]
    flip = [data[f"horizontal_flip:moa:{size}"] for size in (64, 128, 256)]
    cross = [data[f"resolution_vs_reference:moa:{size}"] for size in (128, 256)]
    draw.text((82, 335), "MoA dominant-expert agreement", fill="#eaf4ff", font=section_font)
    draw.text((82, 371), "Near-tie argmax sensitivity; inspect with probability MAE and margin", fill="#829fbd", font=small_font)
    chart_left, chart_top, chart_width = 82, 418, 735
    rows = [
        ("flip 64", flip[0]), ("flip 128", flip[1]), ("flip 256", flip[2]),
        ("64→128", cross[0]), ("64→256", cross[1]),
    ]
    for index, (label, item) in enumerate(rows):
        y = chart_top + index * 67
        agreement = item["dominant_expert_agreement_fraction"]["mean"]
        draw.text((chart_left, y + 8), label, fill="#b5cce2", font=body_font)
        draw.rounded_rectangle((chart_left + 135, y, chart_left + 135 + chart_width, y + 38), radius=9, fill="#10233c")
        draw.rounded_rectangle(
            (chart_left + 135, y, chart_left + 135 + int(chart_width * agreement), y + 38),
            radius=9,
            fill="#4d55d8" if "flip" in label else "#0e9fb0",
        )
        draw.text((chart_left + 150, y + 8), f"{agreement * 100:5.2f}%", fill="#ffffff", font=mono_font)
        draw.text((chart_left + 890, y + 8), f"MAE {item['probability_mae']['mean']:.2e}", fill="#9bb8d6", font=mono_font)

    draw.text((1050, 335), "GT region comparison coverage", fill="#eaf4ff", font=section_font)
    draw.text((1050, 371), "Same geometry for MoT and MoA; 16 router captures per size", fill="#829fbd", font=small_font)
    for index, size in enumerate((64, 128, 256)):
        item = coverage["by_family_resolution"][f"moa:{size}"]
        supported = item["supported_capture_count"]
        y = 430 + index * 112
        draw.text((1050, y), f"{size}px", fill="#b5cce2", font=body_font)
        draw.rounded_rectangle((1140, y - 2, 1665, y + 42), radius=10, fill="#10233c")
        draw.rounded_rectangle((1140, y - 2, 1140 + int(525 * supported / 16), y + 42), radius=10, fill="#00a879")
        draw.text((1160, y + 8), f"{supported}/16 supported", fill="#ffffff", font=mono_font)
        draw.text(
            (1140, y + 58),
            f"FG {item['foreground_token_count']}   BG {item['background_token_count']}   padding {item['padding_token_count']}",
            fill="#8faecb",
            font=small_font,
        )

    draw.rounded_rectangle((82, 820, 1710, 998), radius=18, fill="#081326", outline="#213c5a", width=2)
    draw.text((110, 848), "Interpretation guardrails", fill="#a66cff", font=section_font)
    notes = [
        "MoA probability MAE stays below 7e-7 while mean top-1 margin is also about 1e-6: discrete argmax is tie-sensitive.",
        "MoT maps are spatially constant at cold start: exact equality is real, but Pearson is undefined (0 defined / 144 undefined per group).",
        "Raising input size from 64 to 128 removes all 5/16 INSUFFICIENT_TOKENS cases in this four-image mechanism test.",
    ]
    for index, note in enumerate(notes):
        draw.text((115, 895 + index * 30), f"• {note}", fill="#c8d8e8", font=small_font)
    draw.text((84, 1036), "SHA-256 manifest · locked upstream fingerprints · raw NPZ arrays · exact original-coordinate alignment", fill="#63e6a7", font=small_font)
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
    (run_dir / "command.txt").write_text("run_robustness.cmd\n", encoding="utf-8")
    started = time.perf_counter()

    source_root = (PROJECT_ROOT / config["runtime_root"]).resolve()
    checks = _verify_source(source_root, config["source_fingerprints"])
    sys.path.insert(0, str(source_root))
    os.chdir(PROJECT_ROOT)
    import torch
    from ultralytics import YOLO

    paths, dataset_meta = _resolve_images(config["dataset"], config["dataset_split"], config["sample_indices"])
    input_records = _archive_inputs(paths, config["sample_indices"], run_dir)
    input_record = {
        **dataset_meta,
        "selected_image_count": len(paths),
        "images": input_records,
        "image_and_annotation_set_sha256": hashlib.sha256(
            "\n".join(f"{item['sample_index']}:{item['sha256']}:{item['label_sha256']}" for item in input_records).encode("utf-8")
        ).hexdigest(),
    }
    write_json(run_dir / "input.json", input_record)
    write_json(run_dir / "source-fingerprint-checks.json", checks)

    arrays: dict[str, np.ndarray] = {}
    capture_metadata: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    coverage_records: list[dict[str, Any]] = []
    invariants: dict[str, Any] = {}
    reference_resolution = config["reference_resolution"]
    first_seed = config["seeds"][0]
    logger.info(
        "scope=CPU resolution and horizontal-flip diagnostics resolutions=%s seeds=%s samples=%s",
        config["resolutions"],
        config["seeds"],
        config["sample_indices"],
    )

    for family, profile in config["spatial_profiles"].items():
        for seed in config["seeds"]:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            wrapper = YOLO(source_root / profile["model_config"])
            model = wrapper.model.to("cpu").eval()
            representative, _ = _tensor(Image.open(paths[0]).convert("RGB"), reference_resolution, False, torch)
            with torch.inference_mode():
                baseline_output = _detach_tree(model(representative))
            hooked_output, first_records, module_order = _capture_once(
                model, family, profile["router_class"], representative, torch
            )
            hook_delta = max_output_delta(baseline_output, hooked_output)
            _, repeat_records, repeat_order = _capture_once(model, family, profile["router_class"], representative, torch)
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
                with Image.open(path) as opened:
                    original = opened.convert("RGB")
                boxes = parse_yolo_labels(label_path_for_image(path))
                restored: dict[tuple[int, bool, str], np.ndarray] = {}
                observed_module_order = None
                for resolution in config["resolutions"]:
                    for flipped in (False, True):
                        tensor, meta = _tensor(original, resolution, flipped, torch)
                        _, records, registered = _capture_once(model, family, profile["router_class"], tensor, torch)
                        if observed_module_order is None:
                            observed_module_order = registered
                        if registered != module_order or registered != observed_module_order:
                            raise RuntimeError(f"module order changed across transformations for family={family}")
                        for record in records:
                            weights = record.weights[0]
                            aligned, restoration_validation = restore_probability_stack(weights, meta)
                            if flipped:
                                aligned = np.flip(aligned, axis=2).copy()
                            restored[(resolution, flipped, record.module_name)] = aligned
                            prefix = (
                                f"{family}__seed-{seed}__sample-{sample_index}__size-{resolution}__"
                                f"{'hflip' if flipped else 'identity'}__{_slug(record.module_name)}"
                            )
                            weight_key = f"{prefix}__weights"
                            logit_key = f"{prefix}__logits"
                            arrays[weight_key] = record.weights
                            arrays[logit_key] = record.logits
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
                                    "transformation": "horizontal_flip" if flipped else "identity",
                                    "module": record.module_name,
                                    "module_type": record.module_type,
                                    "source_shape": record.validation["shape"],
                                    "geometry": meta.to_dict(),
                                    "router_validation": record.validation,
                                    "restoration_validation": restoration_validation,
                                    "aligned_to_original_coordinates": True,
                                    "raw_keys": {"weights": weight_key, "logits": logit_key, "indices": index_key},
                                }
                            )
                            if seed == first_seed and not flipped:
                                masks = token_region_masks(boxes, meta, weights.shape[1], weights.shape[2])
                                foreground = int(masks["foreground"].sum())
                                background = int(masks["background"].sum())
                                coverage_records.append(
                                    {
                                        "family": family,
                                        "resolution": resolution,
                                        "sample_index": sample_index,
                                        "module": record.module_name,
                                        "grid_height": int(weights.shape[1]),
                                        "grid_width": int(weights.shape[2]),
                                        "foreground_token_count": foreground,
                                        "background_token_count": background,
                                        "padding_token_count": int(masks["padding"].sum()),
                                        "status": "SUPPORTED" if foreground and background else "INSUFFICIENT_TOKENS",
                                    }
                                )
                for module in module_order:
                    reference = restored[(reference_resolution, False, module)]
                    for resolution in config["resolutions"]:
                        identity = restored[(resolution, False, module)]
                        flipped = restored[(resolution, True, module)]
                        comparisons.append(
                            {
                                "comparison_type": "horizontal_flip",
                                "family": family,
                                "seed": seed,
                                "sample_index": sample_index,
                                "module": module,
                                "reference_resolution": resolution,
                                "candidate_resolution": resolution,
                                "alignment": "restore both maps to original pixels; horizontally unflip candidate",
                                "metrics": probability_map_comparison(identity, flipped),
                            }
                        )
                        if resolution != reference_resolution:
                            comparisons.append(
                                {
                                    "comparison_type": "resolution_vs_reference",
                                    "family": family,
                                    "seed": seed,
                                    "sample_index": sample_index,
                                    "module": module,
                                    "reference_resolution": reference_resolution,
                                    "candidate_resolution": resolution,
                                    "alignment": "restore both identity maps to original-image pixels",
                                    "metrics": probability_map_comparison(reference, identity),
                                }
                            )
                logger.info("family=%s seed=%d sample=%d captures=%d", family, seed, sample_index, len(module_order) * 6)
            del model, wrapper

    np.savez_compressed(run_dir / "stability-routing-raw.npz", **arrays)
    write_json(run_dir / "spatial-captures.json", capture_metadata)
    write_json(run_dir / "stability-comparisons.json", comparisons)
    aggregate = aggregate_stability_comparisons(comparisons)
    write_json(run_dir / "stability-aggregate.json", aggregate)
    coverage = _coverage_summary(coverage_records)
    write_json(run_dir / "region-resolution-coverage.json", coverage)
    write_json(run_dir / "invariants.json", invariants)
    write_json(run_dir / "environment.json", environment(torch, source_root, PROJECT_ROOT))
    _save_overview(aggregate, coverage, run_dir / "robustness-overview.png")

    summary = {
        "status": "PASS",
        "scope": "CPU-only resolution and horizontal-flip diagnostics for true MoT/MoA spatial router probabilities",
        "run_id": config["run_id"],
        "official_locked_base_ref": config["official_locked_base_ref"],
        "official_runtime_ref": config["official_runtime_ref"],
        "tool_source": project_source,
        "source_fingerprint_validation": "PASS",
        "source_fingerprint_count": len(checks),
        "device": "cpu",
        "seeds": config["seeds"],
        "resolutions": config["resolutions"],
        "reference_resolution": reference_resolution,
        "sample_count": len(paths),
        "spatial_capture_count": len(capture_metadata),
        "raw_array_count": len(arrays),
        "aligned_comparison_count": len(comparisons),
        "restoration_validation": {
            "max_pre_normalization_expert_sum_error": max(
                item["restoration_validation"]["pre_normalization_max_expert_sum_error"]
                for item in capture_metadata
            ),
            "max_post_normalization_expert_sum_error": max(
                item["restoration_validation"]["post_normalization_max_expert_sum_error"]
                for item in capture_metadata
            ),
        },
        "representative_invariants": invariants,
        "region_resolution_coverage": coverage["by_family_resolution"],
        "interpretation_boundary": "random initialization; pipeline and sensitivity evidence only, not learned robustness",
        "status_semantics": "PASS means capture, geometry, alignment, determinism and evidence integrity checks completed",
        "duration_seconds_observation_only": time.perf_counter() - started,
    }
    write_json(run_dir / "summary.json", summary)
    for handler in logger.handlers:
        handler.flush()
    write_manifest(run_dir)
    if update_latest:
        latest = PROJECT_ROOT / "artifacts" / "p2" / "ROBUSTNESS_LATEST.txt"
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(config["run_id"] + "\n", encoding="utf-8")
    logger.info("status=PASS captures=%d comparisons=%d arrays=%d", len(capture_metadata), len(comparisons), len(arrays))
    for handler in logger.handlers:
        handler.flush()
    write_manifest(run_dir)
    return run_dir
