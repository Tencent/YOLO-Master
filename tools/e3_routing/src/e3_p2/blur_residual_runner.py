"""Integrity-bound input-side diagnosis of held-middle Gaussian-blur residuals."""

from __future__ import annotations

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
from .io_utils import sha256_file, write_json, write_manifest
from .layer_drilldown import verify_parent_evidence
from .runner import PROJECT_ROOT, RUN_ID_PATTERN, _verify_project_source_state

FEATURES = (
    "letterbox_content_fraction",
    "luminance_mean_0_255",
    "edge_total_variation_0_1",
)


def _load_config(path: Path, run_id_override: str | None) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"config must contain a mapping: {path}")
    config = dict(loaded)
    run_id = str(run_id_override if run_id_override is not None else config.get("run_id", "")).strip()
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must be path-safe and at most 128 characters")
    config["run_id"] = run_id
    if tuple(config.get("features", [])) != FEATURES:
        raise ValueError(f"features must be exactly {list(FEATURES)} in predeclared order")
    if str(config.get("family")) != "gaussian_blur":
        raise ValueError("family must remain gaussian_blur")
    if str(config.get("endpoint")) != "probability_mae":
        raise ValueError("endpoint must remain probability_mae")
    if int(config.get("expected_image_count", 0)) < 16:
        raise ValueError("expected_image_count must be at least 16")
    if int(config.get("bootstrap_draws", 0)) < 1000:
        raise ValueError("bootstrap_draws must be at least 1000")
    if int(config.get("permutation_draws", 0)) < 5000:
        raise ValueError("permutation_draws must be at least 5000")
    if not 0.0 < float(config.get("alpha", 0.0)) < 0.5:
        raise ValueError("alpha must be between zero and 0.5")
    if not 1 <= int(config.get("top_k", 0)) <= int(config["expected_image_count"]):
        raise ValueError("top_k must be within the image count")
    for key in ("expected_holdout_manifest_sha256", "expected_image_source_manifest_sha256"):
        digest = str(config.get(key, ""))
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"{key} must be a lowercase SHA-256 digest")
    selected_digest = str(config.get("expected_selected_image_and_label_set_sha256", ""))
    if len(selected_digest) != 64 or any(character not in "0123456789abcdef" for character in selected_digest):
        raise ValueError("expected_selected_image_and_label_set_sha256 must be a lowercase SHA-256 digest")
    if int(config.get("max_evidence_bytes", 0)) < 100_000:
        raise ValueError("max_evidence_bytes must be at least 100 KB")
    return config


def extract_input_features(image_path: Path, *, canvas_size: int) -> dict[str, float]:
    """Compute the three predeclared model-output-independent image features."""

    if canvas_size <= 0:
        raise ValueError("canvas_size must be positive")
    with Image.open(image_path) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.float64)
    height, width = rgb.shape[:2]
    scale = min(canvas_size / width, canvas_size / height)
    content_fraction = (width * scale * height * scale) / float(canvas_size**2)
    luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    horizontal = float(np.mean(np.abs(np.diff(luminance, axis=1)))) if width > 1 else 0.0
    vertical = float(np.mean(np.abs(np.diff(luminance, axis=0)))) if height > 1 else 0.0
    return {
        "letterbox_content_fraction": float(content_fraction),
        "luminance_mean_0_255": float(np.mean(luminance)),
        "edge_total_variation_0_1": (horizontal + vertical) / (2.0 * 255.0),
    }


def build_feature_records(
    holdout_payload: dict[str, Any],
    input_payload: dict[str, Any],
    image_source_dir: Path,
    *,
    family: str,
    endpoint: str,
    expected_image_count: int,
    canvas_size: int,
) -> list[dict[str, Any]]:
    holdout_records = [item for item in holdout_payload.get("records", []) if item.get("family") == family]
    if int(holdout_payload.get("record_count", -1)) != len(holdout_payload.get("records", [])):
        raise ValueError("holdout record_count does not match payload")
    if len(holdout_records) != expected_image_count:
        raise ValueError("blur holdout record count does not match contract")
    image_items = input_payload.get("images", [])
    if int(input_payload.get("selected_image_count", -1)) != expected_image_count or len(image_items) != expected_image_count:
        raise ValueError("source image count does not match contract")
    holdout_by_index = {int(item["sample_index"]): item for item in holdout_records}
    images_by_index = {int(item["sample_index"]): item for item in image_items}
    expected_indices = set(range(expected_image_count))
    if set(holdout_by_index) != expected_indices or set(images_by_index) != expected_indices:
        raise ValueError("sample indices must exactly cover the expected range")

    output = []
    for sample_index in range(expected_image_count):
        image_item = images_by_index[sample_index]
        relative = Path(str(image_item["image_artifact"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe image artifact path: {relative}")
        image_path = image_source_dir / relative
        if not image_path.is_file() or sha256_file(image_path) != image_item["sha256"]:
            raise RuntimeError(f"source image missing or hash mismatch: sample={sample_index}")
        endpoint_record = holdout_by_index[sample_index]["endpoints"][endpoint]
        residual = endpoint_record.get("normalized_absolute_residual")
        if residual is None or not np.isfinite(float(residual)):
            raise ValueError(f"normalized residual must be finite: sample={sample_index}")
        features = extract_input_features(image_path, canvas_size=canvas_size)
        output.append(
            {
                "sample_index": sample_index,
                "image_name": image_item["name"],
                "image_sha256": image_item["sha256"],
                "image_artifact": relative.as_posix(),
                "original_width": int(image_item["original_width"]),
                "original_height": int(image_item["original_height"]),
                "label_status": image_item["label_status"],
                "ground_truth_box_count_descriptive_only": int(image_item["ground_truth_box_count"]),
                "normalized_absolute_residual": float(residual),
                "absolute_residual": float(endpoint_record["absolute_residual_input_weighted"]),
                "features": features,
            }
        )
    return output


def _bootstrap_spearman(x: np.ndarray, y: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    observed = spearman(x, y)
    if observed is None:
        return {
            "status": "UNDEFINED_CONSTANT_VECTOR",
            "observed_spearman_rho": None,
            "draws": draws,
            "defined_draw_count": 0,
            "undefined_draw_count": draws,
            "percentile_95_interval": None,
        }
    rng = np.random.default_rng(seed)
    values = []
    undefined = 0
    for _ in range(draws):
        indices = rng.integers(0, x.size, size=x.size)
        coefficient = spearman(x[indices], y[indices])
        if coefficient is None:
            undefined += 1
        else:
            values.append(coefficient)
    defined_fraction = len(values) / draws
    array = np.asarray(values, dtype=np.float64)
    return {
        "status": "DEFINED" if defined_fraction >= 0.95 else "INSUFFICIENT_DEFINED_DRAWS",
        "observed_spearman_rho": observed,
        "draws": draws,
        "seed": seed,
        "defined_draw_count": len(values),
        "undefined_draw_count": undefined,
        "percentile_95_interval": (
            [float(value) for value in np.quantile(array, [0.025, 0.975])]
            if defined_fraction >= 0.95
            else None
        ),
        "bootstrap_median": float(np.median(array)) if values else None,
    }


def _leave_one_out(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    values = []
    records = []
    for omitted in range(x.size):
        keep = np.arange(x.size) != omitted
        coefficient = spearman(x[keep], y[keep])
        records.append({"omitted_position": omitted, "spearman_rho": coefficient})
        if coefficient is not None:
            values.append(coefficient)
    return {
        "count": int(x.size),
        "defined_count": len(values),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "sign_stable": bool(values) and (min(values) > 0.0 or max(values) < 0.0),
        "records": records,
    }


def permutation_p_value(x: np.ndarray, y: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    """Return a deterministic two-sided randomization p-value for Spearman rho."""

    observed = spearman(x, y)
    if observed is None:
        return {"status": "UNDEFINED_CONSTANT_VECTOR", "draws": draws, "two_sided_p_value": None}
    rng = np.random.default_rng(seed)
    exceed = 0
    undefined = 0
    threshold = abs(observed) - 1e-15
    for _ in range(draws):
        value = spearman(x, rng.permutation(y))
        if value is None:
            undefined += 1
        elif abs(value) >= threshold:
            exceed += 1
    defined = draws - undefined
    if defined == 0:
        return {"status": "UNDEFINED_ALL_DRAWS", "draws": draws, "two_sided_p_value": None}
    return {
        "status": "DEFINED",
        "draws": draws,
        "seed": seed,
        "defined_draw_count": defined,
        "undefined_draw_count": undefined,
        "exceedance_count": exceed,
        "two_sided_p_value": (exceed + 1) / (defined + 1),
    }


def holm_adjust(raw_p_values: dict[str, float | None], alpha: float) -> dict[str, dict[str, Any]]:
    """Apply Holm's step-down adjustment across all defined predeclared tests."""

    defined = sorted((value, name) for name, value in raw_p_values.items() if value is not None)
    adjusted: dict[str, dict[str, Any]] = {
        name: {"raw_p_value": value, "holm_adjusted_p_value": None, "reject_at_alpha": False}
        for name, value in raw_p_values.items()
    }
    running = 0.0
    count = len(defined)
    for rank, (value, name) in enumerate(defined):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name]["holm_adjusted_p_value"] = running
        adjusted[name]["reject_at_alpha"] = running <= alpha
    return adjusted


def analyze_records(
    records: list[dict[str, Any]],
    *,
    features: tuple[str, ...],
    bootstrap_draws: int,
    bootstrap_seed: int,
    permutation_draws: int,
    permutation_seed: int,
    alpha: float,
    top_k: int,
) -> dict[str, Any]:
    y = np.asarray([item["normalized_absolute_residual"] for item in records], dtype=np.float64)
    associations: dict[str, Any] = {}
    raw_p_values: dict[str, float | None] = {}
    for index, feature in enumerate(features):
        x = np.asarray([item["features"][feature] for item in records], dtype=np.float64)
        bootstrap = _bootstrap_spearman(x, y, bootstrap_draws, bootstrap_seed + index)
        permutation = permutation_p_value(x, y, permutation_draws, permutation_seed + index)
        raw_p_values[feature] = permutation["two_sided_p_value"]
        associations[feature] = {
            "feature_unique_value_count": int(np.unique(x).size),
            "residual_unique_value_count": int(np.unique(y).size),
            "bootstrap": bootstrap,
            "permutation": permutation,
            "leave_one_image_out": _leave_one_out(x, y),
        }
    multiplicity = holm_adjust(raw_p_values, alpha)
    for feature in features:
        associations[feature]["multiplicity"] = multiplicity[feature]

    feature_collinearity = []
    for left_index, left in enumerate(features):
        for right in features[left_index + 1 :]:
            x = np.asarray([item["features"][left] for item in records], dtype=np.float64)
            z = np.asarray([item["features"][right] for item in records], dtype=np.float64)
            feature_collinearity.append({"left": left, "right": right, "spearman_rho": spearman(x, z)})
    top = sorted(records, key=lambda item: (-item["normalized_absolute_residual"], item["sample_index"]))[:top_k]
    return {
        "analysis_unit": "one selected image; no token or seed pseudo-replication",
        "endpoint": "Gaussian-blur held-middle span-normalized absolute probability residual",
        "image_count": len(records),
        "predeclared_features": list(features),
        "multiplicity_control": f"Holm adjustment over {len(features)} two-sided permutation tests",
        "alpha": alpha,
        "associations": associations,
        "feature_collinearity": feature_collinearity,
        "residual_distribution": {
            "minimum": float(np.min(y)),
            "median": float(np.median(y)),
            "p75": float(np.quantile(y, 0.75)),
            "p90": float(np.quantile(y, 0.90)),
            "maximum": float(np.max(y)),
        },
        "top_residual_images_descriptive_only": [
            {
                "rank": rank,
                "sample_index": item["sample_index"],
                "image_name": item["image_name"],
                "normalized_absolute_residual": item["normalized_absolute_residual"],
                "features": item["features"],
            }
            for rank, item in enumerate(top, start=1)
        ],
    }


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _save_overview(records: list[dict[str, Any]], analysis: dict[str, Any], output: Path) -> None:
    canvas = Image.new("RGB", (1680, 1080), "#07111f")
    draw = ImageDraw.Draw(canvas)
    title, body, small = _font(40, True), _font(24), _font(19)
    draw.text((55, 42), "P2 / WHAT DRIVES BLUR INTERPOLATION ERROR?", fill="#eff6ff", font=title)
    draw.text((55, 96), "32 fixed images · input-only features · image bootstrap · permutation + Holm", fill="#63e6a7", font=body)
    labels = {
        "letterbox_content_fraction": "Letterbox content fraction",
        "luminance_mean_0_255": "Mean luminance (0–255)",
        "edge_total_variation_0_1": "Edge total variation (0–1)",
    }
    colors = ["#48cae4", "#f9c74f", "#c77dff"]
    for panel, (feature, color) in enumerate(zip(FEATURES, colors)):
        left, top, width, height = 55 + panel * 540, 165, 480, 480
        draw.rounded_rectangle((left, top, left + width, top + height), 18, fill="#0c1d33", outline="#28425f", width=2)
        xs = np.asarray([item["features"][feature] for item in records], dtype=np.float64)
        ys = np.asarray([item["normalized_absolute_residual"] * 100 for item in records], dtype=np.float64)
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = 0.0, max(ys.max() * 1.08, 1e-6)
        plot_left, plot_top, plot_right, plot_bottom = left + 58, top + 70, left + width - 26, top + height - 70
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#7890aa", width=2)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#7890aa", width=2)
        top_indices = {item["sample_index"] for item in sorted(records, key=lambda value: -value["normalized_absolute_residual"])[:5]}
        for item, x, y in zip(records, xs, ys):
            px = plot_left + (float(x) - xmin) / max(xmax - xmin, 1e-12) * (plot_right - plot_left)
            py = plot_bottom - (float(y) - ymin) / max(ymax - ymin, 1e-12) * (plot_bottom - plot_top)
            radius = 7 if item["sample_index"] in top_indices else 5
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color, outline="#f8fafc")
            if item["sample_index"] in top_indices:
                draw.text((px + 7, py - 14), str(item["sample_index"]), fill="#f8fafc", font=small)
        result = analysis["associations"][feature]
        rho = result["bootstrap"]["observed_spearman_rho"]
        adjusted = result["multiplicity"]["holm_adjusted_p_value"]
        draw.text((left + 24, top + 20), labels[feature], fill="#f8fafc", font=body)
        draw.text((left + 28, top + height - 52), f"rho={rho:+.3f}   Holm p={adjusted:.4f}", fill=color, font=small)
        draw.text((plot_left - 48, plot_top - 10), f"{ymax:.1f}%", fill="#9fb2c8", font=small)
        draw.text((plot_left - 32, plot_bottom - 10), "0", fill="#9fb2c8", font=small)
    draw.text((55, 690), "PREDECLARED INFERENCE", fill="#63e6a7", font=body)
    y = 735
    for feature in FEATURES:
        item = analysis["associations"][feature]
        ci = item["bootstrap"]["percentile_95_interval"]
        loo = item["leave_one_image_out"]
        text = (
            f"{labels[feature]:31s}  bootstrap 95% [{ci[0]:+.3f}, {ci[1]:+.3f}]   "
            f"LOO [{loo['minimum']:+.3f}, {loo['maximum']:+.3f}]   reject={item['multiplicity']['reject_at_alpha']}"
        )
        draw.text((75, y), text, fill="#d7e3f1", font=small)
        y += 48
    draw.text((55, 910), "BOUNDARY", fill="#ffd166", font=body)
    draw.text((55, 952), "Associations diagnose this fixed random-initialization subset; they do not prove mechanism or trained accuracy.", fill="#f5d58a", font=small)
    draw.text((55, 1000), "PASS depends on integrity/completeness/finite statistics, never on significance.", fill="#f5d58a", font=small)
    canvas.save(output)


def _save_top_contact_sheet(records: list[dict[str, Any]], source_dir: Path, top_k: int, output: Path) -> None:
    selected = sorted(records, key=lambda item: (-item["normalized_absolute_residual"], item["sample_index"]))[:top_k]
    canvas = Image.new("RGB", (1680, 920), "#07111f")
    draw = ImageDraw.Draw(canvas)
    draw.text((55, 38), "TOP HELD-MIDDLE BLUR RESIDUALS / DESCRIPTIVE ONLY", fill="#eff6ff", font=_font(38, True))
    draw.text((55, 90), "Rank fixed after the predeclared residual analysis; labels are sample indices.", fill="#9fb2c8", font=_font(21))
    columns = 3
    for index, item in enumerate(selected):
        row, column = divmod(index, columns)
        left, top = 55 + column * 540, 145 + row * 360
        with Image.open(source_dir / item["image_artifact"]) as opened:
            image = opened.convert("RGB")
            image.thumbnail((490, 250), Image.Resampling.LANCZOS)
        frame = Image.new("RGB", (500, 260), "#132944")
        frame.paste(image, ((500 - image.width) // 2, (260 - image.height) // 2))
        canvas.paste(frame, (left, top))
        draw.text((left, top + 270), f"#{index + 1} sample {item['sample_index']} · {item['image_name']}", fill="#f8fafc", font=_font(20, True))
        draw.text((left, top + 302), f"normalized error {item['normalized_absolute_residual'] * 100:.2f}%", fill="#f9c74f", font=_font(20))
    canvas.save(output)


def _inside_project(value: str, field: str) -> Path:
    path = (PROJECT_ROOT / value).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"{field} must stay inside the project repository") from error
    return path


def run(config_path: Path, *, run_id: str | None = None, update_latest: bool = True) -> Path:
    config = _load_config(config_path, run_id)
    project_source = _verify_project_source_state(bool(config.get("require_committed_source", False)))
    holdout_dir = _inside_project(str(config["holdout_run_dir"]), "holdout_run_dir")
    image_source_dir = _inside_project(str(config["image_source_run_dir"]), "image_source_run_dir")
    run_dir = PROJECT_ROOT / "artifacts" / "p2" / config["run_id"]
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty evidence directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"e3_p2.blur_residual.{config['run_id']}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    file_handler = logging.FileHandler(run_dir / "full.log", encoding="utf-8")
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "command.txt").write_text("run_blur_residual.cmd\n", encoding="utf-8")
    started = time.perf_counter()

    holdout_verification = verify_parent_evidence(holdout_dir, config["expected_holdout_manifest_sha256"])
    image_verification = verify_parent_evidence(image_source_dir, config["expected_image_source_manifest_sha256"])
    holdout_summary = json.loads((holdout_dir / "summary.json").read_text(encoding="utf-8"))
    if holdout_summary.get("run_id") != config["expected_holdout_run_id"] or holdout_summary.get("status") != "PASS":
        raise RuntimeError("holdout parent identity or PASS state mismatch")
    input_payload = json.loads((image_source_dir / "input.json").read_text(encoding="utf-8"))
    if input_payload.get("selected_image_and_label_set_sha256") != config["expected_selected_image_and_label_set_sha256"]:
        raise RuntimeError("selected image-and-label set digest mismatch")
    holdout_payload = json.loads((holdout_dir / "dose-holdout-records.json").read_text(encoding="utf-8"))
    records = build_feature_records(
        holdout_payload,
        input_payload,
        image_source_dir,
        family=config["family"],
        endpoint=config["endpoint"],
        expected_image_count=int(config["expected_image_count"]),
        canvas_size=int(config["canvas_size"]),
    )
    analysis = analyze_records(
        records,
        features=FEATURES,
        bootstrap_draws=int(config["bootstrap_draws"]),
        bootstrap_seed=int(config["bootstrap_seed"]),
        permutation_draws=int(config["permutation_draws"]),
        permutation_seed=int(config["permutation_seed"]),
        alpha=float(config["alpha"]),
        top_k=int(config["top_k"]),
    )
    write_json(run_dir / "holdout-parent-verification.json", holdout_verification)
    write_json(run_dir / "image-source-verification.json", image_verification)
    write_json(run_dir / "blur-residual-feature-records.json", {"record_count": len(records), "records": records})
    write_json(run_dir / "blur-residual-analysis.json", analysis)
    _save_overview(records, analysis, run_dir / "blur-residual-overview.png")
    _save_top_contact_sheet(records, image_source_dir, int(config["top_k"]), run_dir / "top-residual-images.png")
    rejected = [name for name, item in analysis["associations"].items() if item["multiplicity"]["reject_at_alpha"]]
    summary = {
        "status": "PASS",
        "scope": "integrity-bound input-side diagnosis of held-middle Gaussian-blur probability residuals",
        "run_id": config["run_id"],
        "tool_source": project_source,
        "holdout_parent_run_id": holdout_summary["run_id"],
        "holdout_parent_verification": holdout_verification,
        "image_source_verification": image_verification,
        "image_count": len(records),
        "predeclared_feature_count": len(FEATURES),
        "holm_rejected_features": rejected,
        "headline_associations": {
            feature: {
                "spearman_rho": item["bootstrap"]["observed_spearman_rho"],
                "bootstrap_95_interval": item["bootstrap"]["percentile_95_interval"],
                "raw_permutation_p": item["multiplicity"]["raw_p_value"],
                "holm_adjusted_p": item["multiplicity"]["holm_adjusted_p_value"],
                "leave_one_out_sign_stable": item["leave_one_image_out"]["sign_stable"],
            }
            for feature, item in analysis["associations"].items()
        },
        "interpretation_boundary": "fixed selected images and random initialization; association is not mechanism, causality or trained accuracy",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "duration_seconds_observation_only": time.perf_counter() - started,
    }
    write_json(run_dir / "summary.json", summary)
    logger.info("status=PASS images=%d features=%d holm_rejected=%d", len(records), len(FEATURES), len(rejected))
    for handler in logger.handlers:
        handler.flush()
    evidence_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    if evidence_bytes > int(config["max_evidence_bytes"]):
        raise RuntimeError(f"blur residual evidence budget exceeded: {evidence_bytes} > {config['max_evidence_bytes']}")
    summary["evidence_bytes_before_manifest"] = evidence_bytes
    write_json(run_dir / "summary.json", summary)
    write_manifest(run_dir)
    if update_latest:
        (run_dir.parent / "BLUR_RESIDUAL_LATEST.txt").write_text(run_dir.name + "\n", encoding="utf-8")
    return run_dir
