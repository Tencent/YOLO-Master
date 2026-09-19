"""Integrity-bound image-level association analysis for the formal coco128 scaling run."""

from __future__ import annotations

import json
import logging
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from .io_utils import write_json, write_manifest
from .layer_drilldown import verify_parent_evidence
from .runner import PROJECT_ROOT, RUN_ID_PATTERN, _verify_project_source_state


def _load_config(path: Path, run_id_override: str | None) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"config must contain a mapping: {path}")
    config = dict(loaded)
    run_id = str(run_id_override if run_id_override is not None else config.get("run_id", "")).strip()
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must be path-safe and at most 128 characters")
    config["run_id"] = run_id
    transforms = [str(value) for value in config.get("candidate_transformations", [])]
    if not transforms or len(set(transforms)) != len(transforms) or "identity" in transforms:
        raise ValueError("candidate_transformations must be non-empty, unique and exclude identity")
    config["candidate_transformations"] = transforms
    endpoints = [str(value) for value in config.get("endpoints", [])]
    allowed = {"probability_mae", "dominant_switch_fraction"}
    if set(endpoints) != allowed or len(endpoints) != len(allowed):
        raise ValueError(f"endpoints must be exactly {sorted(allowed)}")
    config["endpoints"] = endpoints
    if int(config.get("expected_image_count", 0)) < 4:
        raise ValueError("expected_image_count must be at least four")
    if int(config.get("bootstrap_draws", 0)) < 1000:
        raise ValueError("bootstrap_draws must be at least 1000")
    digest = str(config.get("expected_parent_manifest_sha256", ""))
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("expected_parent_manifest_sha256 must be a lowercase SHA-256 digest")
    return config


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks with deterministic tie handling."""

    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size == 0 or not np.isfinite(data).all():
        raise ValueError("rank input must be a non-empty finite vector")
    order = np.argsort(data, kind="mergesort")
    ranks = np.empty(data.size, dtype=np.float64)
    position = 0
    while position < data.size:
        end = position + 1
        while end < data.size and data[order[end]] == data[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + 1 + end) / 2.0
        position = end
    return ranks


def spearman(values_x: np.ndarray, values_y: np.ndarray) -> float | None:
    """Compute tie-aware Spearman rho, returning None when either ranked vector is constant."""

    x = np.asarray(values_x, dtype=np.float64)
    y = np.asarray(values_y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 3:
        raise ValueError("Spearman inputs must be equal one-dimensional vectors of length at least three")
    ranks_x, ranks_y = average_ranks(x), average_ranks(y)
    centered_x = ranks_x - ranks_x.mean()
    centered_y = ranks_y - ranks_y.mean()
    denominator = float(np.sqrt(np.sum(centered_x**2) * np.sum(centered_y**2)))
    if denominator == 0.0:
        return None
    return float(np.sum(centered_x * centered_y) / denominator)


def bootstrap_spearman(
    values_x: np.ndarray, values_y: np.ndarray, draws: int, seed: int
) -> dict[str, Any]:
    """Resample complete images and describe the selected subset's rank association."""

    x = np.asarray(values_x, dtype=np.float64)
    y = np.asarray(values_y, dtype=np.float64)
    observed = spearman(x, y)
    if observed is None:
        raise ValueError("observed Spearman correlation is undefined")
    rng = np.random.default_rng(seed)
    coefficients: list[float] = []
    undefined = 0
    for _ in range(draws):
        indices = rng.integers(0, x.size, size=x.size)
        value = spearman(x[indices], y[indices])
        if value is None:
            undefined += 1
        else:
            coefficients.append(value)
    if len(coefficients) < int(draws * 0.95):
        raise RuntimeError("fewer than 95% of bootstrap Spearman draws are defined")
    array = np.asarray(coefficients, dtype=np.float64)
    return {
        "image_unit_count": int(x.size),
        "draws": draws,
        "seed": seed,
        "observed_spearman_rho": observed,
        "defined_draw_count": len(coefficients),
        "undefined_draw_count": undefined,
        "percentile_95_interval": [float(value) for value in np.quantile(array, [0.025, 0.975])],
        "bootstrap_median": float(np.median(array)),
    }


def leave_one_out_spearman(values_x: np.ndarray, values_y: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values_x, dtype=np.float64)
    y = np.asarray(values_y, dtype=np.float64)
    records = []
    for omitted in range(x.size):
        keep = np.arange(x.size) != omitted
        value = spearman(x[keep], y[keep])
        if value is None:
            raise RuntimeError(f"leave-one-out Spearman undefined after omitting image {omitted}")
        records.append({"omitted_position": omitted, "spearman_rho": value})
    values = [item["spearman_rho"] for item in records]
    return {"count": len(records), "minimum": min(values), "maximum": max(values), "records": records}


def build_image_records(
    effects: list[dict[str, Any]],
    target_cases: list[dict[str, Any]],
    transforms: list[str],
    expected_images: int,
    expected_seeds: list[int],
) -> list[dict[str, Any]]:
    """Join one input-effect value with seed-averaged target routing endpoints per image."""

    effect_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for item in effects:
        key = (str(item["transform"]), int(item["sample_index"]))
        if key in effect_lookup:
            raise ValueError(f"duplicate input-effect record: {key}")
        effect_lookup[key] = item
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in target_cases:
        grouped[(str(item["transform"]), int(item["sample_index"]))].append(item)
    output = []
    expected_seed_set = set(expected_seeds)
    for transform in transforms:
        for sample_index in range(expected_images):
            key = (transform, sample_index)
            if key not in effect_lookup:
                raise ValueError(f"missing input-effect record: {key}")
            cases = grouped.get(key, [])
            seeds = [int(item["seed"]) for item in cases]
            if len(cases) != len(expected_seeds) or set(seeds) != expected_seed_set or len(seeds) != len(set(seeds)):
                raise ValueError(f"incomplete or duplicate seed matrix: {key}, seeds={seeds}")
            effect = effect_lookup[key]
            if not bool(effect.get("model_input_changed")):
                raise ValueError(f"input effect is a no-op: {key}")
            output.append(
                {
                    "transform": transform,
                    "sample_index": sample_index,
                    "input_rgb_mae_0_255": float(effect["rgb_mae_0_255"]),
                    "target_probability_mae_mean_across_seeds": float(
                        np.mean([float(item["probability_mae"]) for item in cases])
                    ),
                    "target_dominant_switch_fraction_mean_across_seeds": float(
                        np.mean([float(item["dominant_switch_fraction"]) for item in cases])
                    ),
                }
            )
    return output


def analyze_associations(
    records: list[dict[str, Any]], transforms: list[str], draws: int, seed: int
) -> dict[str, Any]:
    endpoint_fields = {
        "probability_mae": "target_probability_mae_mean_across_seeds",
        "dominant_switch_fraction": "target_dominant_switch_fraction_mean_across_seeds",
    }
    by_transform: dict[str, Any] = {}
    for transform_index, transform in enumerate(transforms):
        selected = [item for item in records if item["transform"] == transform]
        x = np.asarray([item["input_rgb_mae_0_255"] for item in selected], dtype=np.float64)
        endpoint_results = {}
        for endpoint_index, (endpoint, field) in enumerate(endpoint_fields.items()):
            y = np.asarray([item[field] for item in selected], dtype=np.float64)
            endpoint_results[endpoint] = {
                "x_field": "input_rgb_mae_0_255",
                "y_field": field,
                "bootstrap": bootstrap_spearman(
                    x, y, draws, seed + transform_index * 1000 + endpoint_index
                ),
                "leave_one_image_out": leave_one_out_spearman(x, y),
            }
        by_transform[transform] = {"image_count": len(selected), "endpoints": endpoint_results}
    return {
        "method": {
            "association": "tie-aware Spearman rank correlation",
            "primary_unit": "image after equal averaging across seeds",
            "predictor": "mean absolute raw RGB canvas difference on [0,255]",
            "stratification": "each transformation analyzed separately; cross-transform pooling forbidden",
            "bootstrap": "10,000 image resamples with replacement; descriptive for fixed selected subset",
            "causal_status": "association only",
        },
        "by_transform": by_transform,
    }


def _font(size: int, *, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidate = "C:/Windows/Fonts/consola.ttf" if mono else "C:/Windows/Fonts/segoeui.ttf"
    try:
        return ImageFont.truetype(candidate, size)
    except OSError:
        return ImageFont.load_default()


def _save_overview(records: list[dict[str, Any]], analysis: dict[str, Any], output: Path) -> None:
    width, height = 1800, 1240
    canvas = Image.new("RGB", (width, height), "#071426")
    draw = ImageDraw.Draw(canvas)
    title, subtitle, section, body, small = _font(50), _font(24), _font(27), _font(20), _font(16, mono=True)
    draw.text((72, 54), "DOES MORE INPUT CHANGE MEAN MORE ROUTING CHANGE?", fill="#f0f7ff", font=title)
    draw.text(
        (74, 122),
        "32 image units · within-transform Spearman · 10,000 image bootstrap draws · association, not causation",
        fill="#88a8c7",
        font=subtitle,
    )
    draw.rounded_rectangle((72, 176, 1728, 262), radius=18, fill="#0d2540", outline="#1d5879", width=2)
    draw.text((98, 196), "WHY THIS GATE", fill="#55e2ff", font=section)
    draw.text(
        (330, 199),
        "Pooling brightness, contrast and blur would confound transformation family. Each column is separate.",
        fill="#dcecff",
        font=body,
    )

    transforms = list(analysis["by_transform"])
    endpoint_specs = [
        ("probability_mae", "TARGET PROBABILITY MAE", "#18d6c4"),
        ("dominant_switch_fraction", "TARGET EXPERT SWITCH RATE", "#bf78ff"),
    ]
    panel_width, panel_height = 520, 375
    for column, transform in enumerate(transforms):
        left = 72 + column * 552
        draw.text((left + 4, 294), transform.replace("_", " ").upper(), fill="#ffcf66", font=section)
        selected = [item for item in records if item["transform"] == transform]
        x = np.asarray([item["input_rgb_mae_0_255"] for item in selected], dtype=np.float64)
        for row, (endpoint, label, color) in enumerate(endpoint_specs):
            top = 338 + row * 420
            field = (
                "target_probability_mae_mean_across_seeds"
                if endpoint == "probability_mae"
                else "target_dominant_switch_fraction_mean_across_seeds"
            )
            y = np.asarray([item[field] for item in selected], dtype=np.float64)
            result = analysis["by_transform"][transform]["endpoints"][endpoint]
            bootstrap = result["bootstrap"]
            interval = bootstrap["percentile_95_interval"]
            draw.rounded_rectangle(
                (left, top, left + panel_width, top + panel_height),
                radius=18,
                fill="#0b2038",
                outline="#204d70",
                width=2,
            )
            draw.text((left + 24, top + 20), label, fill="#dcecff", font=body)
            draw.text(
                (left + 24, top + 54),
                f"rho {bootstrap['observed_spearman_rho']:+.3f}  |  95% [{interval[0]:+.3f}, {interval[1]:+.3f}]",
                fill=color,
                font=small,
            )
            plot_left, plot_top = left + 58, top + 106
            plot_right, plot_bottom = left + panel_width - 26, top + panel_height - 48
            draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#496783", width=2)
            draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#496783", width=2)
            x_span = float(np.ptp(x)) or 1.0
            y_span = float(np.ptp(y)) or 1.0
            for item_x, item_y in zip(x, y):
                px = plot_left + 8 + int((plot_right - plot_left - 16) * (item_x - x.min()) / x_span)
                py = plot_bottom - 8 - int((plot_bottom - plot_top - 16) * (item_y - y.min()) / y_span)
                draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color, outline="#ffffff", width=1)
            draw.text((plot_left, plot_bottom + 10), f"RGB MAE {x.min():.3g} → {x.max():.3g}", fill="#7f9fbd", font=small)
            y_label = (
                f"route MAE {y.min():.2e} → {y.max():.2e}"
                if endpoint == "probability_mae"
                else f"switch {y.min() * 100:.2f}% → {y.max() * 100:.2f}%"
            )
            draw.text((plot_left + 205, plot_bottom + 10), y_label, fill="#7f9fbd", font=small)

    draw.rounded_rectangle((72, 1185, 1728, 1225), radius=12, fill="#2b2030")
    draw.text(
        (96, 1195),
        "GUARDRAIL  Fixed deterministic subset + random initialization. Bootstrap describes subset heterogeneity only.",
        fill="#ffd06b",
        font=small,
    )
    canvas.save(output)


def run(config_path: Path, *, run_id: str | None = None, update_latest: bool = True) -> Path:
    config = _load_config(config_path, run_id)
    project_source = _verify_project_source_state(bool(config.get("require_committed_source", False)))
    source_dir = (PROJECT_ROOT / config["parent_run_dir"]).resolve()
    try:
        source_dir.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError("parent_run_dir must stay inside the project repository") from error
    run_dir = PROJECT_ROOT / "artifacts" / "p2" / config["run_id"]
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty evidence directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"e3_p2.image_driver.{config['run_id']}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console, file_handler = logging.StreamHandler(), logging.FileHandler(run_dir / "full.log", encoding="utf-8")
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (run_dir / "command.txt").write_text("run_image_driver.cmd\n", encoding="utf-8")
    started = time.perf_counter()
    verification = verify_parent_evidence(source_dir, config["expected_parent_manifest_sha256"])
    parent_summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    if parent_summary.get("run_id") != config["expected_parent_run_id"] or parent_summary.get("status") != "PASS":
        raise RuntimeError("parent summary run identity or PASS state mismatch")
    if int(parent_summary.get("selected_image_count", -1)) != int(config["expected_image_count"]):
        raise RuntimeError("parent image count does not match the predeclared contract")
    effects = json.loads((source_dir / "transformation-effect.json").read_text(encoding="utf-8"))
    cases = json.loads((source_dir / "target-layer-cases.json").read_text(encoding="utf-8"))
    records = build_image_records(
        effects["records"],
        cases,
        config["candidate_transformations"],
        int(config["expected_image_count"]),
        [int(value) for value in config["expected_seeds"]],
    )
    analysis = analyze_associations(
        records,
        config["candidate_transformations"],
        int(config["bootstrap_draws"]),
        int(config["bootstrap_seed"]),
    )
    write_json(run_dir / "parent-evidence-verification.json", verification)
    write_json(run_dir / "image-level-records.json", {"record_count": len(records), "records": records})
    write_json(run_dir / "image-driver-associations.json", analysis)
    _save_overview(records, analysis, run_dir / "image-driver-overview.png")
    coefficients = {
        transform: {
            endpoint: values["bootstrap"]["observed_spearman_rho"]
            for endpoint, values in result["endpoints"].items()
        }
        for transform, result in analysis["by_transform"].items()
    }
    summary = {
        "status": "PASS",
        "scope": "post-hoc within-transform image-level input-change and target-routing-change associations",
        "run_id": config["run_id"],
        "tool_source": project_source,
        "parent_run_id": parent_summary["run_id"],
        "parent_evidence_verification": verification,
        "image_count": int(config["expected_image_count"]),
        "image_transform_record_count": len(records),
        "candidate_transformations": config["candidate_transformations"],
        "observed_spearman_rho": coefficients,
        "interpretation_boundary": "within-transform association in a deterministic subset under random initialization; not causal or population-generalizable",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "duration_seconds_observation_only": time.perf_counter() - started,
    }
    write_json(run_dir / "summary.json", summary)
    logger.info("status=PASS records=%d transforms=%d endpoints=2", len(records), len(config["candidate_transformations"]))
    for handler in logger.handlers:
        handler.flush()
    write_manifest(run_dir)
    if update_latest:
        latest = PROJECT_ROOT / "artifacts" / "p2" / "IMAGE_DRIVER_LATEST.txt"
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(config["run_id"] + "\n", encoding="utf-8")
    return run_dir
