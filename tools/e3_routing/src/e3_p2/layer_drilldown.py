"""Post-hoc, integrity-bound layer attribution for the formal appearance run."""

from __future__ import annotations

import json
import platform
import time
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from .io_utils import sha256_file, write_json, write_manifest
from .runner import PROJECT_ROOT, RUN_ID_PATTERN, _logger, _slug, _verify_project_source_state


def _load_config(path: Path, run_id_override: str | None) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"config must contain a mapping: {path}")
    config = dict(loaded)
    run_id = str(run_id_override if run_id_override is not None else config.get("run_id", "")).strip()
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must be path-safe and at most 128 characters")
    config["run_id"] = run_id
    if config.get("family") != "moa":
        raise ValueError("layer attribution is intentionally scoped to the non-constant MoA family")
    if not str(config.get("target_module", "")).strip():
        raise ValueError("target_module is required")
    transforms = [str(value) for value in config.get("candidate_transformations", [])]
    if not transforms or len(set(transforms)) != len(transforms) or "identity" in transforms:
        raise ValueError("candidate_transformations must be non-empty, unique and exclude identity")
    config["candidate_transformations"] = transforms
    if int(config.get("margin_bin_count", 0)) != 10:
        raise ValueError("the audited margin contract requires exactly 10 percentile bins")
    expected_digest = str(config.get("expected_parent_manifest_sha256", ""))
    if len(expected_digest) != 64 or any(character not in "0123456789abcdef" for character in expected_digest):
        raise ValueError("expected_parent_manifest_sha256 must be a lowercase SHA-256 digest")
    return config


def _safe_parent_path(relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe manifest path: {relative}")
    return Path(*pure.parts)


def verify_parent_evidence(source_dir: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    """Fail closed unless the parent evidence directory exactly matches its locked manifest."""

    manifest_path = source_dir / "manifest.sha256.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"parent manifest missing: {manifest_path}")
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError(
            f"parent manifest digest mismatch: {manifest_sha256} != {expected_manifest_sha256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if manifest.get("algorithm") != "sha256" or not isinstance(entries, list):
        raise ValueError("parent manifest schema is invalid")
    if int(manifest.get("file_count", -1)) != len(entries):
        raise ValueError("parent manifest file_count does not match entries")
    declared: dict[str, dict[str, Any]] = {}
    for entry in entries:
        relative = str(entry.get("path", ""))
        _safe_parent_path(relative)
        if relative in declared:
            raise ValueError(f"duplicate parent manifest path: {relative}")
        declared[relative] = entry
    actual = {
        path.relative_to(source_dir).as_posix()
        for path in source_dir.rglob("*")
        if path.is_file() and path.name != "manifest.sha256.json"
    }
    if actual != set(declared):
        raise RuntimeError(
            f"parent evidence file set mismatch: missing={sorted(set(declared) - actual)[:3]}, "
            f"extra={sorted(actual - set(declared))[:3]}"
        )
    for relative, entry in declared.items():
        path = source_dir / _safe_parent_path(relative)
        if path.stat().st_size != int(entry["bytes"]) or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"parent evidence integrity mismatch: {relative}")
    return {
        "status": "PASS",
        "manifest_sha256": manifest_sha256,
        "verified_file_count": len(entries),
        "file_set_exact": True,
        "hash_and_size_mismatch_count": 0,
        "key_artifacts": {
            name: declared[name]["sha256"]
            for name in (
                "appearance-routing-raw.npz",
                "appearance-stability-comparisons.json",
                "summary.json",
            )
            if name in declared
        },
    }


def summarize_layer_attribution(
    comparisons: list[dict[str, Any]], family: str, transforms: list[str], target_module: str
) -> dict[str, Any]:
    """Rank layers using equal-weight mean original-coordinate probability MAE."""

    selected = [
        item
        for item in comparisons
        if item["family"] == family
        and item["comparison_type"].removeprefix("appearance_") in transforms
    ]
    if not selected:
        raise ValueError("no matching appearance comparisons")
    by_transform_module: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_transform_seed_module: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for item in selected:
        transform = item["comparison_type"].removeprefix("appearance_")
        module = item["module"]
        value = float(item["metrics"]["probability_mae"])
        by_transform_module[(transform, module)].append(value)
        by_transform_seed_module[(transform, int(item["seed"]), module)].append(value)

    by_transform = {}
    for transform in transforms:
        means = {
            module: float(np.mean(values))
            for (candidate, module), values in by_transform_module.items()
            if candidate == transform
        }
        if target_module not in means or len(means) < 2:
            raise ValueError(f"target or comparison layers missing for {transform}")
        denominator = sum(means.values())
        if denominator <= 0.0:
            raise ValueError(f"non-positive layer MAE sum for {transform}")
        ranking = sorted(means, key=lambda module: (-means[module], module))
        by_transform[transform] = {
            "layer_count": len(ranking),
            "ranking": [
                {
                    "rank": rank,
                    "module": module,
                    "mean_probability_mae": means[module],
                    "share_of_layer_mean_mae_sum": means[module] / denominator,
                }
                for rank, module in enumerate(ranking, start=1)
            ],
            "target_rank": ranking.index(target_module) + 1,
            "target_share_of_layer_mean_mae_sum": means[target_module] / denominator,
        }

    seed_rankings = []
    seed_values = sorted({int(item["seed"]) for item in selected})
    for transform in transforms:
        for seed in seed_values:
            means = {
                module: float(np.mean(values))
                for (candidate, candidate_seed, module), values in by_transform_seed_module.items()
                if candidate == transform and candidate_seed == seed
            }
            ranking = sorted(means, key=lambda module: (-means[module], module))
            seed_rankings.append(
                {
                    "transform": transform,
                    "seed": seed,
                    "target_rank": ranking.index(target_module) + 1,
                    "rank_one_module": ranking[0],
                    "target_mean_probability_mae": means[target_module],
                }
            )
    rank_one_count = sum(item["target_rank"] == 1 for item in seed_rankings)
    return {
        "method": {
            "metric": "probability_mae after exact restoration to original-image pixels",
            "unit": "one seed x sample x module comparison",
            "aggregation": "equal-weight mean within transform and module",
            "share_denominator": "sum of the four module-level mean MAE values; descriptive, not causal",
        },
        "comparison_count": len(selected),
        "target_module": target_module,
        "by_transform": by_transform,
        "target_rank_one_transform_count": sum(
            item["target_rank"] == 1 for item in by_transform.values()
        ),
        "transform_count": len(transforms),
        "seed_stratification": seed_rankings,
        "target_rank_one_transform_seed_count": rank_one_count,
        "transform_seed_group_count": len(seed_rankings),
    }


def margin_bin_summary(margins: np.ndarray, switched: np.ndarray, bin_count: int = 10) -> dict[str, Any]:
    """Summarize switch rate in quantile-defined, mutually exclusive margin bins."""

    values = np.asarray(margins, dtype=np.float64).reshape(-1)
    flags = np.asarray(switched, dtype=bool).reshape(-1)
    if values.size == 0 or values.shape != flags.shape or not np.isfinite(values).all():
        raise ValueError("margins and switched must be matching, finite, non-empty vectors")
    if (values < 0.0).any() or bin_count < 2:
        raise ValueError("margins must be non-negative and bin_count >= 2")
    edges = np.quantile(values, np.linspace(0.0, 1.0, bin_count + 1))
    indices = np.searchsorted(edges[1:-1], values, side="right")
    bins = []
    for index in range(bin_count):
        selected = indices == index
        count = int(selected.sum())
        bins.append(
            {
                "bin_index": index,
                "percentile_range": [index * 100 // bin_count, (index + 1) * 100 // bin_count],
                "lower_edge": float(edges[index]),
                "upper_edge": float(edges[index + 1]),
                "upper_edge_inclusive": index == bin_count - 1,
                "token_comparison_count": count,
                "switch_count": int(flags[selected].sum()),
                "switch_fraction": float(flags[selected].mean()) if count else None,
                "margin_mean": float(values[selected].mean()) if count else None,
            }
        )
    if sum(item["token_comparison_count"] for item in bins) != values.size:
        raise RuntimeError("margin bins did not partition every token comparison exactly once")
    return {
        "token_comparison_count": int(values.size),
        "switch_count": int(flags.sum()),
        "overall_switch_fraction": float(flags.mean()),
        "quantile_edges": [float(value) for value in edges],
        "bins": bins,
    }


def _weight_key(family: str, seed: int, sample: int, transform: str, module: str) -> str:
    return f"{family}__seed-{seed}__sample-{sample}__{transform}__{_slug(module)}__weights"


def _padding_key(family: str, sample: int, module: str) -> str:
    return f"{family}__sample-{sample}__{_slug(module)}__mask-padding"


def _target_cases(
    arrays: Any,
    family: str,
    transforms: list[str],
    target_module: str,
    seeds: list[int],
    samples: list[int],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    records = []
    margin_groups: dict[str, list[np.ndarray]] = defaultdict(list)
    switch_groups: dict[str, list[np.ndarray]] = defaultdict(list)
    for transform in transforms:
        for seed in seeds:
            for sample in samples:
                reference = np.asarray(arrays[_weight_key(family, seed, sample, "identity", target_module)])[0]
                candidate = np.asarray(arrays[_weight_key(family, seed, sample, transform, target_module)])[0]
                padding = np.asarray(arrays[_padding_key(family, sample, target_module)]).astype(bool)
                if reference.shape != candidate.shape or reference.shape[1:] != padding.shape:
                    raise ValueError(f"raw array geometry mismatch for {transform}, seed={seed}, sample={sample}")
                valid = ~padding
                difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
                reference_ordered = np.sort(reference.astype(np.float64), axis=0)
                margin = reference_ordered[-1] - reference_ordered[-2]
                switched = np.argmax(reference, axis=0) != np.argmax(candidate, axis=0)
                valid_margin = margin[valid]
                valid_switched = switched[valid]
                token_tv = 0.5 * difference.sum(axis=0)
                record = {
                    "transform": transform,
                    "seed": seed,
                    "sample_index": sample,
                    "grid_shape_hw": [int(value) for value in valid.shape],
                    "valid_token_count": int(valid.sum()),
                    "padding_token_count": int(padding.sum()),
                    "probability_mae": float(difference[:, valid].mean()),
                    "mean_total_variation_distance": float(token_tv[valid].mean()),
                    "dominant_switch_count": int(valid_switched.sum()),
                    "dominant_switch_fraction": float(valid_switched.mean()),
                    "reference_margin_mean": float(valid_margin.mean()),
                    "switched_reference_margin_mean": (
                        float(valid_margin[valid_switched].mean()) if valid_switched.any() else None
                    ),
                    "unchanged_reference_margin_mean": (
                        float(valid_margin[~valid_switched].mean()) if (~valid_switched).any() else None
                    ),
                }
                records.append(record)
                margin_groups[transform].append(valid_margin)
                switch_groups[transform].append(valid_switched)
                margin_groups["overall"].append(valid_margin)
                switch_groups["overall"].append(valid_switched)
    worst = {
        transform: max(
            (item for item in records if item["transform"] == transform),
            key=lambda item: (
                item["dominant_switch_fraction"],
                item["probability_mae"],
                -item["seed"],
                -item["sample_index"],
            ),
        )
        for transform in transforms
    }
    deciles = {
        key: margin_bin_summary(np.concatenate(margin_groups[key]), np.concatenate(switch_groups[key]))
        for key in [*transforms, "overall"]
    }
    return records, worst, deciles


def _font(size: int, *, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = ["C:/Windows/Fonts/consola.ttf"] if mono else ["C:/Windows/Fonts/segoeui.ttf"]
    candidates += ["DejaVuSansMono.ttf"] if mono else ["DejaVuSans.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _dominant_rgb(weights: np.ndarray, valid: np.ndarray) -> np.ndarray:
    colors = np.asarray([[0, 229, 255], [151, 71, 255], [255, 196, 0], [44, 224, 123]], dtype=np.uint8)
    result = colors[np.argmax(weights, axis=0) % len(colors)]
    result[~valid] = (58, 68, 84)
    return result


def _margin_rank_rgb(weights: np.ndarray, valid: np.ndarray) -> np.ndarray:
    ordered = np.sort(weights.astype(np.float64), axis=0)
    margin = ordered[-1] - ordered[-2]
    valid_values = margin[valid]
    sorted_values = np.sort(valid_values)
    ranks = np.zeros_like(margin, dtype=np.float64)
    ranks[valid] = np.searchsorted(sorted_values, valid_values, side="right") / len(sorted_values)
    rgb = np.zeros((*margin.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(20 + 235 * ranks, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(30 + 185 * np.sqrt(ranks), 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(85 + 170 * (1.0 - ranks), 0, 255).astype(np.uint8)
    rgb[~valid] = (58, 68, 84)
    return rgb


def _save_case_figure(
    source_dir: Path,
    output: Path,
    case: dict[str, Any],
    reference: np.ndarray,
    candidate: np.ndarray,
    valid: np.ndarray,
) -> None:
    width, height, panel = 1130, 950, 300
    canvas = Image.new("RGB", (width, height), "#050c1b")
    draw = ImageDraw.Draw(canvas)
    title, body, small, mono = _font(29), _font(18), _font(15), _font(16, mono=True)
    draw.rounded_rectangle((24, 22, width - 24, height - 22), radius=24, fill="#09172d", outline="#21678d", width=3)
    label = case["transform"].replace("_", " ")
    draw.text((58, 52), f"MODEL.16 ROUTER  /  WORST CASE: {label.upper()}", fill="#00e5ff", font=title)
    draw.text(
        (60, 96),
        f"seed {case['seed']} · sample {case['sample_index']} · valid tokens {case['valid_token_count']}",
        fill="#9bb8d6",
        font=body,
    )
    image_paths = [
        source_dir / "model-inputs" / f"sample-{case['sample_index']}--identity--128px.png",
        source_dir / "model-inputs" / f"sample-{case['sample_index']}--{case['transform']}--128px.png",
    ]
    switched = (np.argmax(reference, axis=0) != np.argmax(candidate, axis=0)) & valid
    switch_rgb = np.full((*valid.shape, 3), (9, 22, 44), dtype=np.uint8)
    switch_rgb[valid] = (28, 64, 88)
    switch_rgb[switched] = (255, 64, 151)
    switch_rgb[~valid] = (58, 68, 84)
    panels = [
        (Image.open(image_paths[0]).convert("RGB"), "Identity model input"),
        (Image.open(image_paths[1]).convert("RGB"), "Perturbed model input"),
        (Image.fromarray(_dominant_rgb(reference, valid)), "Identity dominant expert"),
        (Image.fromarray(_dominant_rgb(candidate, valid)), "Perturbed dominant expert"),
        (Image.fromarray(switch_rgb), "Switch mask · magenta = switched"),
        (Image.fromarray(_margin_rank_rgb(reference, valid)), "Identity margin rank · dark = near tie"),
    ]
    for index, (image, caption) in enumerate(panels):
        column, row = index % 3, index // 3
        x, y = 66 + column * 344, 150 + row * 365
        resized = image.resize((panel, panel), Image.Resampling.NEAREST if index >= 2 else Image.Resampling.LANCZOS)
        canvas.paste(resized, (x, y))
        draw.text((x, y + panel + 8), caption, fill="#dcecff", font=small)
    draw.rounded_rectangle((56, 875, 1074, 915), radius=8, fill="#10233c")
    draw.text(
        (74, 887),
        f"switch {case['dominant_switch_fraction'] * 100:.2f}%  |  MAE {case['probability_mae']:.2e}  |  "
        "padding is gray; maps are raw 16x16 tokens enlarged with nearest-neighbor",
        fill="#63e6a7",
        font=mono,
    )
    canvas.save(output)


def _save_overview(
    attribution: dict[str, Any], deciles: dict[str, Any], worst: dict[str, dict[str, Any]], output: Path
) -> None:
    width, height = 1800, 1160
    canvas = Image.new("RGB", (width, height), "#050c1b")
    draw = ImageDraw.Draw(canvas)
    title, section, body, small, metric = _font(38), _font(24), _font(18), _font(15), _font(31, mono=True)
    draw.rounded_rectangle((40, 34, width - 40, height - 34), radius=30, fill="#09172d", outline="#21678d", width=3)
    draw.text((80, 68), "E3 P2  /  MODEL.16 ROUTER ATTRIBUTION", fill="#00e5ff", font=title)
    draw.text((82, 120), "Integrity-bound post-hoc analysis of the formal CPU appearance run", fill="#9bb8d6", font=body)
    shares = [
        item["target_share_of_layer_mean_mae_sum"]
        for item in attribution["by_transform"].values()
    ]
    overall_bins = deciles["overall"]["bins"]
    first_rate = overall_bins[0]["switch_fraction"]
    upper_rate = max(item["switch_fraction"] or 0.0 for item in overall_bins[3:])
    cards = [
        ("15 / 15", "rank #1 across transform × seed"),
        (f"{min(shares) * 100:.1f}–{max(shares) * 100:.1f}%", "share of four-layer mean-MAE sum"),
        (f"{first_rate * 100:.1f}%", "switch rate in lowest margin decile"),
        (f"{upper_rate * 100:.1f}%", "maximum switch rate in deciles 4–10"),
    ]
    for index, (value, label) in enumerate(cards):
        left = 82 + index * 410
        draw.rounded_rectangle((left, 170, left + 375, 286), radius=18, fill="#0d223e", outline="#204d70", width=2)
        draw.text((left + 20, 191), value, fill="#63e6a7", font=metric)
        draw.text((left + 20, 247), label, fill="#a9c1da", font=small)

    draw.text((82, 335), "Layer contribution by transformation", fill="#eaf4ff", font=section)
    draw.text((82, 372), "Share of the sum of four layer-level mean original-pixel MAEs", fill="#829fbd", font=small)
    transforms = list(attribution["by_transform"])
    palette = ["#00cfe8", "#985eff", "#ffc400", "#2cda7b"]
    for row, transform in enumerate(transforms):
        y = 420 + row * 88
        draw.text((82, y + 7), transform.replace("_", " "), fill="#b5cce2", font=body)
        x = 300
        for index, item in enumerate(attribution["by_transform"][transform]["ranking"]):
            segment = int(650 * item["share_of_layer_mean_mae_sum"])
            draw.rectangle((x, y, x + segment, y + 42), fill=palette[index % len(palette)])
            x += segment
        target = attribution["by_transform"][transform]
        draw.text((970, y + 10), f"model.16 {target['target_share_of_layer_mean_mae_sum'] * 100:5.2f}%", fill="#ffffff", font=small)

    draw.text((1130, 335), "Margin-decile switch concentration", fill="#eaf4ff", font=section)
    draw.text((1130, 372), "All valid model.16 token-comparison exposures", fill="#829fbd", font=small)
    max_rate = max(item["switch_fraction"] or 0.0 for item in overall_bins)
    for index, item in enumerate(overall_bins):
        y = 414 + index * 50
        rate = item["switch_fraction"] or 0.0
        draw.text((1130, y + 6), f"D{index + 1}", fill="#b5cce2", font=body)
        draw.rounded_rectangle((1190, y, 1640, y + 30), radius=7, fill="#10233c")
        if max_rate > 0.0:
            draw.rounded_rectangle((1190, y, 1190 + int(450 * rate / max_rate), y + 30), radius=7, fill="#d65c8a")
        draw.text((1650, y + 5), f"{rate * 100:5.2f}%", fill="#ffffff", font=small)

    draw.text((82, 905), "Worst case selected independently per perturbation", fill="#eaf4ff", font=section)
    for index, (transform, case) in enumerate(worst.items()):
        left = 82 + index * 330
        draw.rounded_rectangle((left, 952, left + 300, 1053), radius=14, fill="#0d223e", outline="#204d70", width=2)
        draw.text((left + 15, 970), transform.replace("_", " "), fill="#d5b7ff", font=small)
        draw.text(
            (left + 15, 1001),
            f"seed {case['seed']} / img {case['sample_index']} / switch {case['dominant_switch_fraction'] * 100:.2f}%",
            fill="#dcecff",
            font=small,
        )
    draw.text(
        (82, 1091),
        "Guardrail: diagnostic attribution within random initialization; contribution share is descriptive, not causal.",
        fill="#ffcc66",
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
    logger = _logger(run_dir / "full.log")
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (run_dir / "command.txt").write_text("run_layer_drilldown.cmd\n", encoding="utf-8")
    started = time.perf_counter()
    parent_verification = verify_parent_evidence(source_dir, config["expected_parent_manifest_sha256"])
    parent_summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    if parent_summary.get("run_id") != config["expected_parent_run_id"] or parent_summary.get("status") != "PASS":
        raise RuntimeError("parent summary run identity or PASS state mismatch")
    comparisons = json.loads((source_dir / "appearance-stability-comparisons.json").read_text(encoding="utf-8"))
    attribution = summarize_layer_attribution(
        comparisons, config["family"], config["candidate_transformations"], config["target_module"]
    )
    with np.load(source_dir / "appearance-routing-raw.npz", allow_pickle=False) as arrays:
        records, worst, deciles = _target_cases(
            arrays,
            config["family"],
            config["candidate_transformations"],
            config["target_module"],
            [int(value) for value in parent_summary["seeds"]],
            list(range(int(parent_summary["sample_count"]))),
        )
        case_dir = run_dir / "cases"
        case_dir.mkdir(parents=True, exist_ok=True)
        for transform, case in worst.items():
            reference = np.asarray(
                arrays[
                    _weight_key(
                        config["family"], case["seed"], case["sample_index"], "identity", config["target_module"]
                    )
                ]
            )[0]
            candidate = np.asarray(
                arrays[
                    _weight_key(
                        config["family"],
                        case["seed"],
                        case["sample_index"],
                        transform,
                        config["target_module"],
                    )
                ]
            )[0]
            padding = np.asarray(
                arrays[_padding_key(config["family"], case["sample_index"], config["target_module"])]
            ).astype(bool)
            case["figure"] = f"cases/{transform}.png"
            _save_case_figure(source_dir, run_dir / case["figure"], case, reference, candidate, ~padding)

    write_json(run_dir / "parent-evidence-verification.json", parent_verification)
    write_json(run_dir / "layer-attribution.json", attribution)
    write_json(
        run_dir / "margin-deciles.json",
        {
            "method": {
                "scope": "target module raw token grid with letterbox padding excluded",
                "unit": "one valid token-comparison exposure; identity margin repeats once per candidate transform",
                "binning": "within-scope empirical quantile edges; half-open except final inclusive bin",
            },
            "by_transform_and_overall": deciles,
        },
    )
    write_json(
        run_dir / "target-layer-cases.json",
        {
            "method": "raw target-layer tokens; equal-weight case summaries; padding excluded",
            "case_count": len(records),
            "records": records,
            "worst_case_selection": "maximum switch fraction, then MAE, then lowest seed/sample",
            "worst_by_transform": worst,
        },
    )
    _save_overview(attribution, deciles, worst, run_dir / "layer-attribution-overview.png")
    shares = [
        item["target_share_of_layer_mean_mae_sum"] for item in attribution["by_transform"].values()
    ]
    overall_bins = deciles["overall"]["bins"]
    summary = {
        "status": "PASS",
        "scope": "post-hoc MoA layer attribution and margin-conditioned switch localization",
        "run_id": config["run_id"],
        "tool_source": project_source,
        "parent_run_id": parent_summary["run_id"],
        "parent_evidence_verification": parent_verification,
        "target_module": config["target_module"],
        "candidate_transformations": config["candidate_transformations"],
        "layer_attribution_comparison_count": attribution["comparison_count"],
        "target_case_count": len(records),
        "valid_token_comparison_exposure_count": deciles["overall"]["token_comparison_count"],
        "target_rank_one_transform_seed_count": attribution["target_rank_one_transform_seed_count"],
        "transform_seed_group_count": attribution["transform_seed_group_count"],
        "target_share_of_layer_mean_mae_sum_min": min(shares),
        "target_share_of_layer_mean_mae_sum_max": max(shares),
        "lowest_margin_decile_switch_fraction": overall_bins[0]["switch_fraction"],
        "deciles_four_through_ten_max_switch_fraction": max(
            item["switch_fraction"] or 0.0 for item in overall_bins[3:]
        ),
        "interpretation_boundary": "random initialization; descriptive concentration, not causal attribution or learned robustness",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "duration_seconds_observation_only": time.perf_counter() - started,
    }
    write_json(run_dir / "summary.json", summary)
    logger.info(
        "status=PASS comparisons=%d target_cases=%d token_exposures=%d rank_one=%d/%d",
        attribution["comparison_count"],
        len(records),
        deciles["overall"]["token_comparison_count"],
        attribution["target_rank_one_transform_seed_count"],
        attribution["transform_seed_group_count"],
    )
    for handler in logger.handlers:
        handler.flush()
    write_manifest(run_dir)
    if update_latest:
        latest = PROJECT_ROOT / "artifacts" / "p2" / "ATTRIBUTION_LATEST.txt"
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(config["run_id"] + "\n", encoding="utf-8")
    return run_dir
