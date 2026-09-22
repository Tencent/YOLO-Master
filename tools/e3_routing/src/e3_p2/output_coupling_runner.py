"""Run and summarize the predeclared router-to-detector output coupling study."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from .image_driver_analysis import bootstrap_spearman, leave_one_out_spearman, spearman
from .io_utils import write_json, write_manifest
from .runner import PROJECT_ROOT
from .scale_runner import run as run_scale

ROUTE_FIELDS = {
    "probability_mae": "target_probability_mae_mean_across_seeds",
    "dominant_switch_fraction": "target_dominant_switch_fraction_mean_across_seeds",
}
DETECTOR_TENSORS = (
    "one2one_scores",
    "one2one_boxes",
    "one2many_scores",
    "one2many_boxes",
    "decoded_top300",
)


def build_output_coupling_records(
    target_cases: list[dict[str, Any]],
    detector_comparisons: list[dict[str, Any]],
    transforms: list[str],
    expected_images: int,
    expected_seeds: list[int],
) -> list[dict[str, Any]]:
    """Join target-router and detector changes at image x transform after equal seed averaging."""

    route_grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    detector_grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in target_cases:
        route_grouped[(str(item["transform"]), int(item["sample_index"]))].append(item)
    for item in detector_comparisons:
        detector_grouped[(str(item["transform"]), int(item["sample_index"]))].append(item)

    expected_seed_set = set(expected_seeds)
    output: list[dict[str, Any]] = []
    for transform in transforms:
        for sample_index in range(expected_images):
            key = (transform, sample_index)
            route_items = route_grouped.get(key, [])
            detector_items = detector_grouped.get(key, [])
            for label, items in (("route", route_items), ("detector", detector_items)):
                seeds = [int(item["seed"]) for item in items]
                if (
                    len(items) != len(expected_seeds)
                    or set(seeds) != expected_seed_set
                    or len(seeds) != len(set(seeds))
                ):
                    raise ValueError(f"incomplete or duplicate {label} seed matrix: {key}, seeds={seeds}")

            tensor_values: dict[str, float] = {}
            for tensor in DETECTOR_TENSORS:
                values = []
                for item in detector_items:
                    metrics = item.get("metrics", {})
                    if metrics.get("primary_tensor") != "one2one_scores":
                        raise ValueError(f"unexpected detector primary tensor: {key}")
                    tensors = metrics.get("tensors", {})
                    if set(tensors) != set(DETECTOR_TENSORS):
                        raise ValueError(f"detector tensor contract changed: {key}")
                    value = float(tensors[tensor]["mean_absolute_change"])
                    if not np.isfinite(value):
                        raise ValueError(f"non-finite detector change: {key}, tensor={tensor}")
                    values.append(value)
                tensor_values[tensor] = float(np.mean(values))

            output.append(
                {
                    "transform": transform,
                    "sample_index": sample_index,
                    "seed_count": len(expected_seeds),
                    "target_probability_mae_mean_across_seeds": float(
                        np.mean([float(item["probability_mae"]) for item in route_items])
                    ),
                    "target_dominant_switch_fraction_mean_across_seeds": float(
                        np.mean([float(item["dominant_switch_fraction"]) for item in route_items])
                    ),
                    "detector_mean_absolute_change_mean_across_seeds": tensor_values,
                }
            )
    if len(output) != len(transforms) * expected_images:
        raise RuntimeError("output-coupling record matrix size changed")
    return output


def analyze_output_coupling(
    records: list[dict[str, Any]], transforms: list[str], draws: int, seed: int
) -> dict[str, Any]:
    """Compute within-transform image-level associations without imposing a sign gate."""

    by_transform: dict[str, Any] = {}
    for transform_index, transform in enumerate(transforms):
        selected = [item for item in records if item["transform"] == transform]
        tensor_results: dict[str, Any] = {}
        for tensor_index, tensor in enumerate(DETECTOR_TENSORS):
            y = np.asarray(
                [item["detector_mean_absolute_change_mean_across_seeds"][tensor] for item in selected],
                dtype=np.float64,
            )
            route_results: dict[str, Any] = {}
            for route_index, (endpoint, field) in enumerate(ROUTE_FIELDS.items()):
                x = np.asarray([item[field] for item in selected], dtype=np.float64)
                analysis_seed = seed + transform_index * 1000 + tensor_index * 10 + route_index
                observed = spearman(x, y)
                result = {
                    "x_field": field,
                    "y_field": f"detector.{tensor}.mean_absolute_change",
                    "x_unique_value_count": int(np.unique(x).size),
                    "y_unique_value_count": int(np.unique(y).size),
                }
                if observed is None:
                    result.update(
                        {
                            "status": "UNDEFINED_CONSTANT_VECTOR",
                            "observed_spearman_rho": None,
                            "reason": "Spearman is undefined because at least one ranked vector is constant",
                            "bootstrap": None,
                            "leave_one_image_out": None,
                        }
                    )
                else:
                    result.update(
                        {
                            "status": "DEFINED",
                            "observed_spearman_rho": observed,
                            "bootstrap": bootstrap_spearman(x, y, draws, analysis_seed),
                            "leave_one_image_out": leave_one_out_spearman(x, y),
                        }
                    )
                route_results[endpoint] = result
            tensor_results[tensor] = {"route_endpoints": route_results}
        by_transform[transform] = {"image_count": len(selected), "detector_tensors": tensor_results}
    return {
        "method": {
            "association": "tie-aware Spearman rank correlation",
            "primary_unit": "image after equal averaging across three random-initialization seeds",
            "stratification": "each transformation analyzed separately; cross-transform pooling forbidden",
            "primary_detector_endpoint": "one2one_scores mean absolute change on fixed grid/channel order",
            "secondary_detector_endpoints": [
                "one2one_boxes",
                "one2many_scores",
                "one2many_boxes",
                "decoded_top300 (row order not anchor-aligned)",
            ],
            "bootstrap": f"{draws:,} image resamples with replacement; descriptive for fixed subset",
            "pass_gate": "matrix completeness, finite metrics and reproducibility only; sign/magnitude are observations",
            "causal_status": "association only; no accuracy or causal claim",
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
    width, height = 1800, 1080
    canvas = Image.new("RGB", (width, height), "#071426")
    draw = ImageDraw.Draw(canvas)
    title, subtitle, section = _font(47), _font(23), _font(27)
    body, small = _font(19), _font(16, mono=True)
    draw.text((72, 52), "DO ROUTER CHANGES TRACK DETECTOR-SCORE CHANGES?", fill="#f2f7ff", font=title)
    draw.text(
        (74, 116),
        "32 image units · 3 seeds averaged equally · within-transform Spearman · association only",
        fill="#8eabc8",
        font=subtitle,
    )
    draw.rounded_rectangle((72, 165, 1728, 258), radius=18, fill="#0d2540", outline="#1d5879", width=2)
    draw.text((98, 185), "PRIMARY ENDPOINT", fill="#55e2ff", font=section)
    draw.text(
        (380, 190),
        "One-to-one class-score tensor MAE: fixed grid and channel order, before Top-300 row reordering.",
        fill="#dcecff",
        font=body,
    )

    colors = {"probability_mae": "#18d6c4", "dominant_switch_fraction": "#bf78ff"}
    labels = {"probability_mae": "ROUTER PROBABILITY MAE", "dominant_switch_fraction": "EXPERT SWITCH RATE"}
    transforms = list(analysis["by_transform"])
    for column, transform in enumerate(transforms):
        left, top = 72 + column * 552, 308
        draw.text((left + 4, top - 37), transform.replace("_", " ").upper(), fill="#ffcf66", font=section)
        draw.rounded_rectangle((left, top, left + 520, top + 655), radius=18, fill="#0b2038", outline="#204d70", width=2)
        selected = [item for item in records if item["transform"] == transform]
        y = np.asarray(
            [item["detector_mean_absolute_change_mean_across_seeds"]["one2one_scores"] for item in selected],
            dtype=np.float64,
        )
        for row, (endpoint, field) in enumerate(ROUTE_FIELDS.items()):
            x = np.asarray([item[field] for item in selected], dtype=np.float64)
            result = analysis["by_transform"][transform]["detector_tensors"]["one2one_scores"][
                "route_endpoints"
            ][endpoint]
            panel_top = top + 38 + row * 292
            draw.text((left + 24, panel_top), labels[endpoint], fill="#dcecff", font=body)
            if result["status"] == "DEFINED":
                bootstrap = result["bootstrap"]
                interval = bootstrap["percentile_95_interval"]
                statistic = (
                    f"rho {bootstrap['observed_spearman_rho']:+.3f} | "
                    f"95% [{interval[0]:+.3f}, {interval[1]:+.3f}]"
                )
            else:
                statistic = f"UNDEFINED | detector unique values = {result['y_unique_value_count']}"
            draw.text((left + 24, panel_top + 34), statistic, fill=colors[endpoint], font=small)
            plot_left, plot_top = left + 58, panel_top + 83
            plot_right, plot_bottom = left + 490, panel_top + 238
            draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#496783", width=2)
            draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#496783", width=2)
            x_span, y_span = float(np.ptp(x)) or 1.0, float(np.ptp(y)) or 1.0
            for item_x, item_y in zip(x, y):
                px = plot_left + 8 + int((plot_right - plot_left - 16) * (item_x - x.min()) / x_span)
                py = plot_bottom - 8 - int((plot_bottom - plot_top - 16) * (item_y - y.min()) / y_span)
                draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=colors[endpoint], outline="#ffffff")
            draw.text((plot_left, plot_bottom + 8), f"router {x.min():.2e} → {x.max():.2e}", fill="#7f9fbd", font=small)
            draw.text((plot_left + 235, plot_bottom + 8), f"score {y.min():.2e} → {y.max():.2e}", fill="#7f9fbd", font=small)

    draw.rounded_rectangle((72, 1000, 1728, 1046), radius=12, fill="#2b2030")
    draw.text(
        (96, 1012),
        "GUARDRAIL  Random initialization + fixed coco128 subset. Correlation does not establish accuracy or causality.",
        fill="#ffd06b",
        font=small,
    )
    canvas.save(output)


def run(config_path: Path, *, run_id: str | None = None, update_latest: bool = True) -> Path:
    run_dir = run_scale(config_path, run_id=run_id, update_latest=False)
    config = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    target_cases = json.loads((run_dir / "target-layer-cases.json").read_text(encoding="utf-8"))
    detector_comparisons = json.loads(
        (run_dir / "detector-output-comparisons.json").read_text(encoding="utf-8")
    )
    transforms = [str(item["name"]) for item in config["transformations"] if item["name"] != "identity"]
    records = build_output_coupling_records(
        target_cases,
        detector_comparisons,
        transforms,
        int(config["selected_image_count"]),
        [int(value) for value in config["seeds"]],
    )
    analysis = analyze_output_coupling(
        records, transforms, int(config["bootstrap_draws"]), int(config["bootstrap_seed"])
    )
    write_json(run_dir / "output-coupling-records.json", {"record_count": len(records), "records": records})
    write_json(run_dir / "output-coupling-associations.json", analysis)
    _save_overview(records, analysis, run_dir / "output-coupling-overview.png")
    (run_dir / "command.txt").write_text("run_output_coupling.cmd\n", encoding="utf-8")

    primary = {
        transform: {
            endpoint: values["observed_spearman_rho"]
            for endpoint, values in result["detector_tensors"]["one2one_scores"]["route_endpoints"].items()
        }
        for transform, result in analysis["by_transform"].items()
    }
    association_results = [
        endpoint
        for transform in analysis["by_transform"].values()
        for tensor in transform["detector_tensors"].values()
        for endpoint in tensor["route_endpoints"].values()
    ]
    summary.update(
        {
            "scope": "CPU-only MoA target-router to detector-output image-level coupling audit",
            "output_coupling_record_count": len(records),
            "output_coupling_association_count": len(transforms) * len(DETECTOR_TENSORS) * len(ROUTE_FIELDS),
            "defined_association_count": sum(item["status"] == "DEFINED" for item in association_results),
            "undefined_constant_association_count": sum(
                item["status"] == "UNDEFINED_CONSTANT_VECTOR" for item in association_results
            ),
            "primary_detector_endpoint": "one2one_scores mean absolute change",
            "primary_observed_spearman_rho": primary,
            "pass_gate": "complete finite evidence matrix and reproducibility; association sign/magnitude not gated",
            "interpretation_boundary": "within-transform association on a deterministic coco128 subset under random initialization; no causal, learned-accuracy or population claim",
        }
    )
    write_json(run_dir / "summary.json", summary)
    evidence_bytes = sum(
        path.stat().st_size
        for path in run_dir.rglob("*")
        if path.is_file() and path.name != "manifest.sha256.json"
    )
    if evidence_bytes > int(config["max_evidence_bytes"]):
        raise RuntimeError(f"final evidence budget exceeded: {evidence_bytes} > {config['max_evidence_bytes']}")
    summary["evidence_bytes_before_manifest"] = evidence_bytes
    write_json(run_dir / "summary.json", summary)
    write_manifest(run_dir)
    if update_latest:
        latest = PROJECT_ROOT / "artifacts" / "p2" / "OUTPUT_COUPLING_LATEST.txt"
        latest.write_text(config["run_id"] + "\n", encoding="utf-8")
    return run_dir
