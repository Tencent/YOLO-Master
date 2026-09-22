"""Predeclared multi-strength MoA appearance audit with image-paired dose summaries."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from .image_driver_analysis import _font
from .io_utils import write_json, write_manifest
from .scale_runner import bootstrap_mean_interval
from .scale_runner import run as run_image_scale


def _dose_families(config: dict[str, Any]) -> list[dict[str, Any]]:
    transforms = [str(item["name"]) for item in config.get("transformations", [])]
    candidates = set(transforms) - {"identity"}
    families = config.get("dose_response_families")
    if not isinstance(families, list) or not families:
        raise ValueError("dose_response_families must be a non-empty list")
    seen_names: set[str] = set()
    seen_transforms: set[str] = set()
    normalized = []
    for family in families:
        name = str(family.get("name", "")).strip()
        if not name or name in seen_names:
            raise ValueError(f"duplicate or empty dose family: {name!r}")
        seen_names.add(name)
        conditions = family.get("conditions")
        if not isinstance(conditions, list) or len(conditions) < 3:
            raise ValueError(f"dose family {name} requires at least three conditions")
        normalized_conditions = []
        severities = []
        for item in conditions:
            transform = str(item.get("transform", ""))
            severity = float(item.get("severity"))
            if transform not in candidates or transform in seen_transforms:
                raise ValueError(f"unknown or reused dose transform: {transform}")
            seen_transforms.add(transform)
            severities.append(severity)
            normalized_conditions.append(
                {"transform": transform, "severity": severity, "label": str(item.get("label", severity))}
            )
        if severities != sorted(severities) or len(set(severities)) != len(severities):
            raise ValueError(f"dose severities must be strictly increasing for {name}")
        normalized.append({"name": name, "conditions": normalized_conditions})
    if seen_transforms != candidates:
        raise ValueError(f"dose families do not cover candidates: missing={sorted(candidates - seen_transforms)}")
    return normalized


def build_dose_records(
    effect_records: list[dict[str, Any]],
    target_cases: list[dict[str, Any]],
    families: list[dict[str, Any]],
    expected_images: int,
    expected_seeds: list[int],
) -> list[dict[str, Any]]:
    effects: dict[tuple[str, int], dict[str, Any]] = {}
    for item in effect_records:
        key = (str(item["transform"]), int(item["sample_index"]))
        if key in effects:
            raise ValueError(f"duplicate effect record: {key}")
        effects[key] = item
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in target_cases:
        grouped[(str(item["transform"]), int(item["sample_index"]))].append(item)
    expected_seed_set = set(expected_seeds)
    output = []
    for family in families:
        for condition in family["conditions"]:
            transform = condition["transform"]
            for sample_index in range(expected_images):
                key = (transform, sample_index)
                if key not in effects or not bool(effects[key].get("model_input_changed")):
                    raise ValueError(f"missing or no-op effect record: {key}")
                cases = grouped.get(key, [])
                seeds = [int(item["seed"]) for item in cases]
                if len(cases) != len(expected_seeds) or set(seeds) != expected_seed_set or len(seeds) != len(set(seeds)):
                    raise ValueError(f"incomplete or duplicate dose seed matrix: {key}, seeds={seeds}")
                ordered = sorted(cases, key=lambda item: int(item["seed"]))
                seed_values = [
                    {
                        "seed": int(item["seed"]),
                        "probability_mae": float(item["probability_mae"]),
                        "dominant_switch_fraction": float(item["dominant_switch_fraction"]),
                    }
                    for item in ordered
                ]
                output.append(
                    {
                        "family": family["name"],
                        "transform": transform,
                        "severity": float(condition["severity"]),
                        "label": condition["label"],
                        "sample_index": sample_index,
                        "input_rgb_mae_0_255": float(effects[key]["rgb_mae_0_255"]),
                        "target_probability_mae_mean_across_seeds": float(
                            np.mean([item["probability_mae"] for item in seed_values])
                        ),
                        "target_dominant_switch_fraction_mean_across_seeds": float(
                            np.mean([item["dominant_switch_fraction"] for item in seed_values])
                        ),
                        "seed_values": seed_values,
                    }
                )
    return output


def _nondecreasing(values: list[float]) -> bool:
    return all(right >= left for left, right in zip(values, values[1:]))


def summarize_dose_response(
    records: list[dict[str, Any]],
    families: list[dict[str, Any]],
    expected_images: int,
    expected_seeds: list[int],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    metric_fields = {
        "input_rgb_mae_0_255": "input_rgb_mae_0_255",
        "target_probability_mae": "target_probability_mae_mean_across_seeds",
        "target_dominant_switch_fraction": "target_dominant_switch_fraction_mean_across_seeds",
    }
    by_family = {}
    for family_index, family in enumerate(families):
        family_records = [item for item in records if item["family"] == family["name"]]
        expected_count = expected_images * len(family["conditions"])
        if len(family_records) != expected_count:
            raise ValueError(f"incomplete dose family records for {family['name']}")
        levels = []
        for level_index, condition in enumerate(family["conditions"]):
            selected = [item for item in family_records if item["transform"] == condition["transform"]]
            if len(selected) != expected_images:
                raise ValueError(f"incomplete dose level: {condition['transform']}")
            levels.append(
                {
                    **condition,
                    "metrics": {
                        metric: bootstrap_mean_interval(
                            np.asarray([item[field] for item in selected], dtype=np.float64),
                            draws,
                            seed + family_index * 10000 + level_index * 100 + metric_index,
                        )
                        for metric_index, (metric, field) in enumerate(metric_fields.items())
                    },
                }
            )
        monotonicity: dict[str, Any] = {}
        for metric, field in metric_fields.items():
            image_records = []
            for sample_index in range(expected_images):
                ordered = sorted(
                    (item for item in family_records if item["sample_index"] == sample_index),
                    key=lambda item: item["severity"],
                )
                values = [float(item[field]) for item in ordered]
                image_records.append(
                    {"sample_index": sample_index, "values": values, "nondecreasing": _nondecreasing(values)}
                )
            monotonicity[metric] = {
                "image_count": expected_images,
                "nondecreasing_image_count": sum(item["nondecreasing"] for item in image_records),
                "nondecreasing_image_fraction": float(np.mean([item["nondecreasing"] for item in image_records])),
                "image_records": image_records,
            }
        for metric, seed_field in (
            ("target_probability_mae", "probability_mae"),
            ("target_dominant_switch_fraction", "dominant_switch_fraction"),
        ):
            unit_records = []
            for sample_index in range(expected_images):
                ordered = sorted(
                    (item for item in family_records if item["sample_index"] == sample_index),
                    key=lambda item: item["severity"],
                )
                for current_seed in expected_seeds:
                    values = [
                        next(value[seed_field] for value in item["seed_values"] if value["seed"] == current_seed)
                        for item in ordered
                    ]
                    unit_records.append(
                        {
                            "sample_index": sample_index,
                            "seed": current_seed,
                            "values": values,
                            "nondecreasing": _nondecreasing(values),
                        }
                    )
            monotonicity[metric].update(
                {
                    "image_seed_unit_count": len(unit_records),
                    "nondecreasing_image_seed_count": sum(item["nondecreasing"] for item in unit_records),
                    "nondecreasing_image_seed_fraction": float(
                        np.mean([item["nondecreasing"] for item in unit_records])
                    ),
                    "image_seed_records": unit_records,
                }
            )
        if monotonicity["input_rgb_mae_0_255"]["nondecreasing_image_count"] != expected_images:
            raise RuntimeError(f"input severity ladder is not monotone for every image: {family['name']}")
        low_transform = family["conditions"][0]["transform"]
        high_transform = family["conditions"][-1]["transform"]
        paired_differences = {}
        for metric_index, (metric, field) in enumerate(metric_fields.items()):
            differences = []
            for sample_index in range(expected_images):
                low = next(
                    item for item in family_records if item["sample_index"] == sample_index and item["transform"] == low_transform
                )
                high = next(
                    item for item in family_records if item["sample_index"] == sample_index and item["transform"] == high_transform
                )
                differences.append(float(high[field]) - float(low[field]))
            paired_differences[metric] = bootstrap_mean_interval(
                np.asarray(differences, dtype=np.float64),
                draws,
                seed + family_index * 10000 + 9000 + metric_index,
            )
        by_family[family["name"]] = {
            "conditions": family["conditions"],
            "levels": levels,
            "monotonicity": monotonicity,
            "high_minus_low_paired_difference": paired_differences,
        }
    return {
        "method": {
            "primary_unit": "image after equal averaging across three seeds",
            "primary_endpoint": "target raw-grid probability MAE",
            "secondary_endpoint": "target raw-grid dominant-expert switch fraction",
            "bootstrap": "10,000 paired image resamples; descriptive for the fixed selected subset",
            "monotonic_rule": "non-decreasing at all adjacent levels; exact floating-point comparison",
            "family_pooling": "forbidden",
        },
        "by_family": by_family,
    }


def _save_dose_overview(result: dict[str, Any], output: Path) -> None:
    width, height = 1800, 1040
    canvas = Image.new("RGB", (width, height), "#061426")
    draw = ImageDraw.Draw(canvas)
    title, subtitle, section, body, small = _font(48), _font(22), _font(27), _font(18), _font(15)
    draw.text((70, 52), "APPEARANCE STRENGTH → ROUTING RESPONSE", fill="#eff7ff", font=title)
    draw.text(
        (72, 116),
        "32 image units · 3 seeds · low / medium / high · image-paired bootstrap · random-init mechanism audit",
        fill="#8faecc",
        font=subtitle,
    )
    colors = ["#22d3c5", "#ffcb66", "#c477ff"]
    for column, (family, item) in enumerate(result["by_family"].items()):
        left = 70 + column * 570
        draw.rounded_rectangle((left, 178, left + 530, 910), radius=24, fill="#0b213a", outline="#226287", width=2)
        draw.text((left + 28, 205), family.replace("_", " ").upper(), fill="#58e2ff", font=section)
        labels = [level["label"] for level in item["levels"]]
        probability = [level["metrics"]["target_probability_mae"]["observed_mean"] for level in item["levels"]]
        switches = [
            level["metrics"]["target_dominant_switch_fraction"]["observed_mean"] for level in item["levels"]
        ]
        inputs = [level["metrics"]["input_rgb_mae_0_255"]["observed_mean"] for level in item["levels"]]
        for index, label in enumerate(labels):
            x = left + 95 + index * 165
            draw.text((x - 18, 263), label, fill="#dcecff", font=body)
            if index < len(labels) - 1:
                draw.line((x + 37, 276, x + 127, 276), fill="#3d6687", width=3)
                draw.polygon([(x + 127, 270), (x + 142, 276), (x + 127, 282)], fill="#3d6687")
        blocks = [
            ("INPUT RGB MAE", inputs, "{:.3g}"),
            ("TARGET PROBABILITY MAE", probability, "{:.2e}"),
            ("TARGET SWITCH RATE", [value * 100 for value in switches], "{:.2f}%"),
        ]
        for row, (label, values, formatter) in enumerate(blocks):
            top = 320 + row * 150
            draw.text((left + 28, top), label, fill="#92aec8", font=small)
            maximum = max(values) or 1.0
            for index, value in enumerate(values):
                x = left + 52 + index * 165
                bar_height = int(72 * value / maximum)
                draw.rounded_rectangle((x, top + 38, x + 100, top + 118), radius=9, fill="#102c49")
                draw.rounded_rectangle(
                    (x, top + 118 - bar_height, x + 100, top + 118), radius=9, fill=colors[index]
                )
                draw.text((x + 7, top + 124), formatter.format(value), fill="#dcecff", font=small)
        probability_mono = item["monotonicity"]["target_probability_mae"]
        switch_mono = item["monotonicity"]["target_dominant_switch_fraction"]
        draw.rounded_rectangle((left + 28, 782, left + 502, 878), radius=14, fill="#08182b", outline="#254764", width=2)
        draw.text(
            (left + 48, 802),
            f"probability monotone  {probability_mono['nondecreasing_image_count']}/32 images",
            fill="#63e6a7",
            font=body,
        )
        draw.text(
            (left + 48, 840),
            f"switch monotone       {switch_mono['nondecreasing_image_count']}/32 images",
            fill="#d6a2ff",
            font=body,
        )
    draw.rounded_rectangle((70, 940, 1730, 996), radius=14, fill="#302233")
    draw.text(
        (96, 957),
        "GUARDRAIL  Probability MAE is primary. Switch is margin-sensitive. Families are never pooled; monotonicity is not a PASS criterion.",
        fill="#ffd16e",
        font=small,
    )
    canvas.save(output)


def run(config_path: Path, *, run_id: str | None = None, update_latest: bool = True) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"config must contain a mapping: {config_path}")
    families = _dose_families(config)
    started = time.perf_counter()
    run_dir = run_image_scale(config_path, run_id=run_id, update_latest=False)
    resolved = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    effects = json.loads((run_dir / "transformation-effect.json").read_text(encoding="utf-8"))
    cases = json.loads((run_dir / "target-layer-cases.json").read_text(encoding="utf-8"))
    records = build_dose_records(
        effects["records"],
        cases,
        families,
        int(resolved["selected_image_count"]),
        [int(value) for value in resolved["seeds"]],
    )
    result = summarize_dose_response(
        records,
        families,
        int(resolved["selected_image_count"]),
        [int(value) for value in resolved["seeds"]],
        int(resolved["bootstrap_draws"]),
        int(resolved["bootstrap_seed"]),
    )
    write_json(run_dir / "dose-response-records.json", {"record_count": len(records), "records": records})
    write_json(run_dir / "dose-response.json", result)
    _save_dose_overview(result, run_dir / "dose-response-overview.png")
    legacy_overview = run_dir / "image-level-overview.png"
    if legacy_overview.exists():
        legacy_overview.unlink()
    (run_dir / "command.txt").write_text("run_dose_response.cmd\n", encoding="utf-8")
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "scope": "CPU-only 32-image MoA appearance-strength dose-response audit",
            "dose_response_family_count": len(families),
            "dose_response_level_count": sum(len(item["conditions"]) for item in families),
            "dose_response_record_count": len(records),
            "dose_response": {
                name: {
                    "probability_nondecreasing_image_count": item["monotonicity"]["target_probability_mae"][
                        "nondecreasing_image_count"
                    ],
                    "switch_nondecreasing_image_count": item["monotonicity"][
                        "target_dominant_switch_fraction"
                    ]["nondecreasing_image_count"],
                }
                for name, item in result["by_family"].items()
            },
            "interpretation_boundary": "deterministic coco128 subset and random initialization; descriptive dose response, not learned robustness or accuracy",
            "duration_seconds_observation_only": time.perf_counter() - started,
        }
    )
    write_json(summary_path, summary)
    with (run_dir / "full.log").open("a", encoding="utf-8") as stream:
        stream.write(
            f"dose-response status=PASS families={len(families)} levels={sum(len(item['conditions']) for item in families)} records={len(records)}\n"
        )
    evidence_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file() and path.name != "manifest.sha256.json")
    if evidence_bytes > int(resolved["max_evidence_bytes"]):
        raise RuntimeError(f"dose evidence budget exceeded: {evidence_bytes} > {resolved['max_evidence_bytes']}")
    summary["evidence_bytes_before_manifest"] = evidence_bytes
    write_json(summary_path, summary)
    write_manifest(run_dir)
    if update_latest:
        latest = run_dir.parent / "DOSE_RESPONSE_LATEST.txt"
        latest.write_text(run_dir.name + "\n", encoding="utf-8")
    return run_dir
