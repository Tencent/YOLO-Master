"""Integrity-bound held-middle validation for the formal three-level dose evidence."""

from __future__ import annotations

import itertools
import json
import logging
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from .image_driver_analysis import spearman
from .io_utils import write_json, write_manifest
from .layer_drilldown import verify_parent_evidence
from .runner import PROJECT_ROOT, RUN_ID_PATTERN, _verify_project_source_state
from .scale_runner import bootstrap_mean_interval

ENDPOINTS = {
    "probability_mae": ("target_probability_mae_mean_across_seeds", "probability_mae"),
    "dominant_switch_fraction": (
        "target_dominant_switch_fraction_mean_across_seeds",
        "dominant_switch_fraction",
    ),
}


def _load_config(path: Path, run_id_override: str | None) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"config must contain a mapping: {path}")
    config = dict(loaded)
    run_id = str(run_id_override if run_id_override is not None else config.get("run_id", "")).strip()
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must be path-safe and at most 128 characters")
    config["run_id"] = run_id
    if int(config.get("expected_image_count", 0)) < 4:
        raise ValueError("expected_image_count must be at least four")
    seeds = [int(value) for value in config.get("expected_seeds", [])]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("expected_seeds must contain at least two unique integers")
    config["expected_seeds"] = seeds
    if int(config.get("bootstrap_draws", 0)) < 1000:
        raise ValueError("bootstrap_draws must be at least 1000")
    if int(config.get("max_evidence_bytes", 0)) < 100_000:
        raise ValueError("max_evidence_bytes must be at least 100 KB")
    digest = str(config.get("expected_parent_manifest_sha256", ""))
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("expected_parent_manifest_sha256 must be a lowercase SHA-256 digest")
    families = config.get("families")
    if not isinstance(families, list) or len(families) != 3:
        raise ValueError("families must contain exactly three transformation families")
    seen_names: set[str] = set()
    seen_transforms: set[str] = set()
    normalized = []
    for item in families:
        name = str(item.get("name", "")).strip()
        transforms = [
            str(item.get("low_transform", "")).strip(),
            str(item.get("middle_transform", "")).strip(),
            str(item.get("high_transform", "")).strip(),
        ]
        if not name or name in seen_names or any(not value for value in transforms):
            raise ValueError("family names and transforms must be non-empty and unique")
        if len(set(transforms)) != 3 or any(value in seen_transforms for value in transforms):
            raise ValueError("dose holdout transforms must be globally unique")
        seen_names.add(name)
        seen_transforms.update(transforms)
        normalized.append({"name": name, "transforms": transforms})
    config["families"] = normalized
    return config


def build_holdout_records(
    parent_records: list[dict[str, Any]],
    families: list[dict[str, Any]],
    expected_images: int,
    expected_seeds: list[int],
) -> list[dict[str, Any]]:
    """Build one held-middle prediction record per family and image."""

    lookup: dict[tuple[str, str, int], dict[str, Any]] = {}
    for item in parent_records:
        key = (str(item["family"]), str(item["transform"]), int(item["sample_index"]))
        if key in lookup:
            raise ValueError(f"duplicate parent dose record: {key}")
        lookup[key] = item
    expected_keys = {
        (family["name"], transform, image)
        for family in families
        for transform in family["transforms"]
        for image in range(expected_images)
    }
    if set(lookup) != expected_keys:
        raise ValueError(
            f"parent dose matrix mismatch: missing={len(expected_keys - set(lookup))}, "
            f"extra={len(set(lookup) - expected_keys)}"
        )

    expected_seed_set = set(expected_seeds)
    output = []
    for family in families:
        for sample_index in range(expected_images):
            levels = [lookup[(family["name"], transform, sample_index)] for transform in family["transforms"]]
            x = [float(item["input_rgb_mae_0_255"]) for item in levels]
            if not all(np.isfinite(x)) or not x[0] < x[1] < x[2]:
                raise ValueError(f"RGB distance must be strictly ordered: {family['name']}, image={sample_index}")
            weight = (x[1] - x[0]) / (x[2] - x[0])
            endpoints: dict[str, Any] = {}
            for endpoint, (mean_field, seed_field) in ENDPOINTS.items():
                y = [float(item[mean_field]) for item in levels]
                if not all(np.isfinite(y)):
                    raise ValueError(f"non-finite endpoint: {family['name']}, image={sample_index}, {endpoint}")
                prediction = y[0] + weight * (y[2] - y[0])
                midpoint = (y[0] + y[2]) / 2.0
                residual = y[1] - prediction
                span = abs(y[2] - y[0])
                seed_records = []
                for seed in expected_seeds:
                    seed_values = []
                    for level in levels:
                        values = level.get("seed_values", [])
                        seeds = [int(item["seed"]) for item in values]
                        if (
                            len(values) != len(expected_seeds)
                            or set(seeds) != expected_seed_set
                            or len(seeds) != len(set(seeds))
                        ):
                            raise ValueError(
                                f"incomplete parent seed matrix: {family['name']}, image={sample_index}"
                            )
                        seed_values.append(float(next(item[seed_field] for item in values if int(item["seed"]) == seed)))
                    seed_prediction = seed_values[0] + weight * (seed_values[2] - seed_values[0])
                    seed_records.append(
                        {
                            "seed": seed,
                            "low": seed_values[0],
                            "actual_middle": seed_values[1],
                            "high": seed_values[2],
                            "predicted_middle": seed_prediction,
                            "residual": seed_values[1] - seed_prediction,
                            "low_to_high_slope_per_rgb_mae": (seed_values[2] - seed_values[0]) / (x[2] - x[0]),
                        }
                    )
                endpoints[endpoint] = {
                    "low": y[0],
                    "actual_middle": y[1],
                    "high": y[2],
                    "predicted_middle_input_weighted": prediction,
                    "predicted_middle_naive_midpoint": midpoint,
                    "signed_residual_input_weighted": residual,
                    "absolute_residual_input_weighted": abs(residual),
                    "absolute_residual_naive_midpoint": abs(y[1] - midpoint),
                    "endpoint_span": span,
                    "normalized_absolute_residual": abs(residual) / span if span > 0.0 else None,
                    "actual_middle_within_endpoints": min(y[0], y[2]) <= y[1] <= max(y[0], y[2]),
                    "seed_records": seed_records,
                }
            output.append(
                {
                    "family": family["name"],
                    "sample_index": sample_index,
                    "transforms": family["transforms"],
                    "input_rgb_mae_0_255": {"low": x[0], "middle": x[1], "high": x[2]},
                    "input_interpolation_weight": weight,
                    "endpoints": endpoints,
                }
            )
    return output


def _slope_association(x: np.ndarray, y: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    observed = spearman(x, y)
    base = {"x_unique_value_count": int(np.unique(x).size), "y_unique_value_count": int(np.unique(y).size)}
    if observed is None:
        return {
            **base,
            "status": "UNDEFINED_CONSTANT_VECTOR",
            "observed_spearman_rho": None,
            "bootstrap": None,
            "leave_one_image_out": None,
        }
    rng = np.random.default_rng(seed)
    bootstrap_values = []
    undefined_bootstrap = 0
    for _ in range(draws):
        indices = rng.integers(0, x.size, size=x.size)
        value = spearman(x[indices], y[indices])
        if value is None:
            undefined_bootstrap += 1
        else:
            bootstrap_values.append(value)
    defined_fraction = len(bootstrap_values) / draws
    bootstrap_array = np.asarray(bootstrap_values, dtype=np.float64)
    bootstrap = {
        "draws": draws,
        "seed": seed,
        "defined_draw_count": len(bootstrap_values),
        "undefined_draw_count": undefined_bootstrap,
        "defined_draw_fraction": defined_fraction,
        "interval_status": "DEFINED" if defined_fraction >= 0.95 else "INSUFFICIENT_DEFINED_DRAWS",
        "percentile_95_interval": (
            [float(value) for value in np.quantile(bootstrap_array, [0.025, 0.975])]
            if defined_fraction >= 0.95
            else None
        ),
        "bootstrap_median": float(np.median(bootstrap_array)) if bootstrap_values else None,
    }
    leave_records = []
    for omitted in range(x.size):
        keep = np.arange(x.size) != omitted
        value = spearman(x[keep], y[keep])
        leave_records.append(
            {
                "omitted_position": omitted,
                "status": "DEFINED" if value is not None else "UNDEFINED_CONSTANT_VECTOR",
                "spearman_rho": value,
            }
        )
    defined_leave = [item["spearman_rho"] for item in leave_records if item["spearman_rho"] is not None]
    return {
        **base,
        "status": "DEFINED",
        "observed_spearman_rho": observed,
        "bootstrap": bootstrap,
        "leave_one_image_out": {
            "count": len(leave_records),
            "defined_count": len(defined_leave),
            "undefined_count": len(leave_records) - len(defined_leave),
            "minimum": min(defined_leave) if defined_leave else None,
            "maximum": max(defined_leave) if defined_leave else None,
            "records": leave_records,
        },
    }


def analyze_holdout(
    records: list[dict[str, Any]],
    families: list[dict[str, Any]],
    expected_seeds: list[int],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    by_family: dict[str, Any] = {}
    for family_index, family in enumerate(families):
        selected = [item for item in records if item["family"] == family["name"]]
        endpoint_results: dict[str, Any] = {}
        for endpoint_index, endpoint in enumerate(ENDPOINTS):
            values = [item["endpoints"][endpoint] for item in selected]
            weighted_error = np.asarray([item["absolute_residual_input_weighted"] for item in values])
            midpoint_error = np.asarray([item["absolute_residual_naive_midpoint"] for item in values])
            signed_error = np.asarray([item["signed_residual_input_weighted"] for item in values])
            normalized = np.asarray(
                [item["normalized_absolute_residual"] for item in values if item["normalized_absolute_residual"] is not None]
            )
            base_seed = seed + family_index * 10000 + endpoint_index * 1000
            seed_associations = []
            for pair_index, (left_seed, right_seed) in enumerate(itertools.combinations(expected_seeds, 2)):
                left = np.asarray(
                    [
                        next(value for value in item["seed_records"] if value["seed"] == left_seed)[
                            "low_to_high_slope_per_rgb_mae"
                        ]
                        for item in values
                    ]
                )
                right = np.asarray(
                    [
                        next(value for value in item["seed_records"] if value["seed"] == right_seed)[
                            "low_to_high_slope_per_rgb_mae"
                        ]
                        for item in values
                    ]
                )
                seed_associations.append(
                    {
                        "seed_pair": [left_seed, right_seed],
                        **_slope_association(left, right, draws, base_seed + 100 + pair_index),
                    }
                )
            endpoint_results[endpoint] = {
                "image_count": len(values),
                "input_weighted_absolute_error": bootstrap_mean_interval(weighted_error, draws, base_seed),
                "naive_midpoint_absolute_error": bootstrap_mean_interval(midpoint_error, draws, base_seed + 1),
                "paired_naive_minus_weighted_absolute_error": bootstrap_mean_interval(
                    midpoint_error - weighted_error, draws, base_seed + 2
                ),
                "input_weighted_signed_error": bootstrap_mean_interval(signed_error, draws, base_seed + 3),
                "normalized_absolute_error": {
                    "defined_image_count": int(normalized.size),
                    "zero_span_image_count": len(values) - int(normalized.size),
                    "mean": float(normalized.mean()) if normalized.size else None,
                    "median": float(np.median(normalized)) if normalized.size else None,
                    "p90": float(np.quantile(normalized, 0.9)) if normalized.size else None,
                    "fraction_at_or_below_0_10": float(np.mean(normalized <= 0.10)) if normalized.size else None,
                    "fraction_at_or_below_0_20": float(np.mean(normalized <= 0.20)) if normalized.size else None,
                },
                "actual_middle_within_endpoints_count": sum(
                    item["actual_middle_within_endpoints"] for item in values
                ),
                "cross_seed_low_to_high_slope_associations": seed_associations,
            }
        by_family[family["name"]] = {"endpoints": endpoint_results}
    return {
        "method": {
            "unit": "image after equal averaging across seeds",
            "held_level": "middle",
            "predictor": "linear interpolation in per-image actual RGB MAE using low/high route endpoints",
            "baseline": "unweighted arithmetic midpoint of low/high route endpoints",
            "primary_endpoint": "probability_mae",
            "secondary_endpoint": "dominant_switch_fraction",
            "bootstrap": f"{draws:,} image resamples; descriptive for the fixed selected subset",
            "pass_gate": "integrity and completeness only; predictive performance is not gated",
        },
        "by_family": by_family,
    }


def _font(size: int, *, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidate = "C:/Windows/Fonts/consola.ttf" if mono else "C:/Windows/Fonts/segoeui.ttf"
    try:
        return ImageFont.truetype(candidate, size)
    except OSError:
        return ImageFont.load_default()


def _save_overview(records: list[dict[str, Any]], result: dict[str, Any], output: Path) -> None:
    canvas = Image.new("RGB", (1800, 1050), "#071426")
    draw = ImageDraw.Draw(canvas)
    title, subtitle, section, body, small = _font(46), _font(22), _font(27), _font(18), _font(15, mono=True)
    draw.text((70, 48), "CAN LOW + HIGH PREDICT THE HELD MIDDLE DOSE?", fill="#f2f7ff", font=title)
    draw.text(
        (72, 110),
        "32 image units · interpolation weighted by actual RGB distance · 10,000 image bootstrap draws",
        fill="#8eabc8",
        font=subtitle,
    )
    families = list(result["by_family"])
    for column, family in enumerate(families):
        left = 70 + column * 570
        draw.rounded_rectangle((left, 170, left + 530, 925), radius=24, fill="#0b213a", outline="#226287", width=2)
        draw.text((left + 28, 196), family.replace("_", " ").upper(), fill="#ffcf66", font=section)
        selected = [item for item in records if item["family"] == family]
        actual = np.asarray([item["endpoints"]["probability_mae"]["actual_middle"] for item in selected])
        predicted = np.asarray(
            [item["endpoints"]["probability_mae"]["predicted_middle_input_weighted"] for item in selected]
        )
        low = float(min(actual.min(), predicted.min()))
        high = float(max(actual.max(), predicted.max()))
        span = high - low or 1.0
        plot_left, plot_top, plot_right, plot_bottom = left + 70, 275, left + 480, 635
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#496783", width=2)
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#496783", width=2)
        draw.line((plot_left, plot_bottom, plot_right, plot_top), fill="#315e72", width=2)
        for prediction, observation in zip(predicted, actual):
            px = plot_left + int((plot_right - plot_left) * (prediction - low) / span)
            py = plot_bottom - int((plot_bottom - plot_top) * (observation - low) / span)
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill="#18d6c4", outline="#ffffff")
        draw.text((plot_left, plot_bottom + 10), "predicted middle →", fill="#8ba8c4", font=small)
        draw.text((plot_left - 45, plot_top - 22), "actual ↑", fill="#8ba8c4", font=small)
        probability = result["by_family"][family]["endpoints"]["probability_mae"]
        switch = result["by_family"][family]["endpoints"]["dominant_switch_fraction"]
        improvement = probability["paired_naive_minus_weighted_absolute_error"]
        interval = improvement["percentile_95_interval"]
        normalized = probability["normalized_absolute_error"]
        draw.text((left + 28, 682), "PRIMARY · PROBABILITY", fill="#55e2ff", font=body)
        draw.text(
            (left + 28, 718),
            f"weighted MAE  {probability['input_weighted_absolute_error']['observed_mean']:.2e}",
            fill="#dcecff",
            font=small,
        )
        draw.text(
            (left + 28, 748),
            f"normalized median  {normalized['median']:.3f}  |  p90 {normalized['p90']:.3f}",
            fill="#dcecff",
            font=small,
        )
        draw.text(
            (left + 28, 778),
            f"baseline error minus weighted  {improvement['observed_mean']:.2e}",
            fill="#63e6a7" if interval[0] > 0 else "#ffcf66",
            font=small,
        )
        draw.text((left + 28, 808), f"95% [{interval[0]:.2e}, {interval[1]:.2e}]", fill="#8ba8c4", font=small)
        switch_norm = switch["normalized_absolute_error"]
        draw.text((left + 28, 850), "SECONDARY · SWITCH", fill="#c477ff", font=body)
        switch_text = (
            f"normalized median {switch_norm['median']:.3f} | zero-span {switch_norm['zero_span_image_count']}"
            if switch_norm["median"] is not None
            else f"normalized undefined | zero-span {switch_norm['zero_span_image_count']}"
        )
        draw.text((left + 28, 883), switch_text, fill="#dcecff", font=small)
    draw.rounded_rectangle((70, 960, 1730, 1015), radius=14, fill="#2b2030")
    draw.text(
        (96, 977),
        "GUARDRAIL  Retrospectively specified held-level analysis. Error and baseline improvement are observations, not PASS criteria.",
        fill="#ffd06b",
        font=small,
    )
    canvas.save(output)


def run(config_path: Path, *, run_id: str | None = None, update_latest: bool = True) -> Path:
    config = _load_config(config_path, run_id)
    project_source = _verify_project_source_state(bool(config.get("require_committed_source", False)))
    parent_dir = (PROJECT_ROOT / str(config["parent_run_dir"])).resolve()
    try:
        parent_dir.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError("parent_run_dir must stay inside the project repository") from error
    run_dir = PROJECT_ROOT / "artifacts" / "p2" / config["run_id"]
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty evidence directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"e3_p2.dose_holdout.{config['run_id']}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    file_handler = logging.FileHandler(run_dir / "full.log", encoding="utf-8")
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (run_dir / "command.txt").write_text("run_dose_holdout.cmd\n", encoding="utf-8")
    started = time.perf_counter()
    verification = verify_parent_evidence(parent_dir, config["expected_parent_manifest_sha256"])
    parent_summary = json.loads((parent_dir / "summary.json").read_text(encoding="utf-8"))
    if parent_summary.get("run_id") != config["expected_parent_run_id"] or parent_summary.get("status") != "PASS":
        raise RuntimeError("parent summary run identity or PASS state mismatch")
    if int(parent_summary.get("selected_image_count", -1)) != int(config["expected_image_count"]):
        raise RuntimeError("parent image count does not match holdout contract")
    parent = json.loads((parent_dir / "dose-response-records.json").read_text(encoding="utf-8"))
    records = build_holdout_records(
        parent["records"], config["families"], int(config["expected_image_count"]), config["expected_seeds"]
    )
    analysis = analyze_holdout(
        records,
        config["families"],
        config["expected_seeds"],
        int(config["bootstrap_draws"]),
        int(config["bootstrap_seed"]),
    )
    write_json(run_dir / "parent-evidence-verification.json", verification)
    write_json(run_dir / "dose-holdout-records.json", {"record_count": len(records), "records": records})
    write_json(run_dir / "dose-holdout-analysis.json", analysis)
    _save_overview(records, analysis, run_dir / "dose-holdout-overview.png")
    summary = {
        "status": "PASS",
        "scope": "integrity-bound held-middle validation of the formal MoA dose-response evidence",
        "run_id": config["run_id"],
        "tool_source": project_source,
        "parent_run_id": parent_summary["run_id"],
        "parent_evidence_verification": verification,
        "image_count": int(config["expected_image_count"]),
        "family_count": len(config["families"]),
        "holdout_record_count": len(records),
        "seed_slope_association_count": len(config["families"]) * len(ENDPOINTS) * 3,
        "headline_probability_metrics": {
            family: {
                "weighted_absolute_error": item["endpoints"]["probability_mae"][
                    "input_weighted_absolute_error"
                ]["observed_mean"],
                "normalized_absolute_error_median": item["endpoints"]["probability_mae"][
                    "normalized_absolute_error"
                ]["median"],
                "naive_minus_weighted_error": item["endpoints"]["probability_mae"][
                    "paired_naive_minus_weighted_absolute_error"
                ]["observed_mean"],
            }
            for family, item in analysis["by_family"].items()
        },
        "interpretation_boundary": "retrospectively specified fixed-subset random-initialization interpolation audit; not blind validation, trained robustness or accuracy",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "duration_seconds_observation_only": time.perf_counter() - started,
    }
    write_json(run_dir / "summary.json", summary)
    logger.info("status=PASS families=%d records=%d", len(config["families"]), len(records))
    for handler in logger.handlers:
        handler.flush()
    evidence_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    if evidence_bytes > int(config["max_evidence_bytes"]):
        raise RuntimeError(f"holdout evidence budget exceeded: {evidence_bytes} > {config['max_evidence_bytes']}")
    summary["evidence_bytes_before_manifest"] = evidence_bytes
    write_json(run_dir / "summary.json", summary)
    write_manifest(run_dir)
    if update_latest:
        (run_dir.parent / "DOSE_HOLDOUT_LATEST.txt").write_text(run_dir.name + "\n", encoding="utf-8")
    return run_dir
