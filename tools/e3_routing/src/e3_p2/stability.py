"""Probability-map restoration and transformation-stability metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .geometry import LetterboxMeta, restore_heatmap, validate_probability_grid


def restore_probability_stack(weights: np.ndarray, meta: LetterboxMeta) -> tuple[np.ndarray, dict[str, float]]:
    """Restore ``[E,H,W]`` probabilities to original pixels and re-normalize experts.

    Bilinear interpolation is applied to each expert independently. Floating-point
    interpolation can move the expert sum slightly away from one, so the restored
    stack is explicitly normalized and both pre/post errors are retained as evidence.
    """

    values = np.asarray(weights, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"expected [E,H,W], got shape={list(values.shape)}")
    validate_probability_grid(values[None, ...])
    restored = np.stack([restore_heatmap(field, meta) for field in values], axis=0).astype(np.float32)
    if not np.isfinite(restored).all() or float(restored.min()) < -1e-6:
        raise ValueError("restored probability stack is not finite and non-negative")
    sums = restored.sum(axis=0, keepdims=True)
    pre_error = float(np.max(np.abs(sums - 1.0)))
    if float(sums.min()) <= 0.0:
        raise ValueError("restored expert sum contains a non-positive pixel")
    normalized = restored / sums
    post_error = float(np.max(np.abs(normalized.sum(axis=0) - 1.0)))
    return normalized.astype(np.float32), {
        "pre_normalization_max_expert_sum_error": pre_error,
        "post_normalization_max_expert_sum_error": post_error,
    }


def _validate_probability_stack(value: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=np.float32)
    if values.ndim != 3 or min(values.shape) <= 0:
        raise ValueError(f"{name} must be a non-empty [E,H,W] stack")
    if not np.isfinite(values).all() or float(values.min()) < -1e-6:
        raise ValueError(f"{name} must contain finite non-negative probabilities")
    error = float(np.max(np.abs(values.sum(axis=0) - 1.0)))
    if error > 1e-5:
        raise ValueError(f"{name} probabilities do not sum to one: max_error={error}")
    return values


def probability_map_comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    """Compare aligned expert-probability maps without inventing constant-map correlation."""

    left = _validate_probability_stack(reference, "reference")
    right = _validate_probability_stack(candidate, "candidate")
    if left.shape != right.shape:
        raise ValueError(f"probability map shapes differ: {list(left.shape)} != {list(right.shape)}")
    left64 = left.astype(np.float64)
    right64 = right.astype(np.float64)
    difference = np.abs(left64 - right64)
    midpoint = 0.5 * (left64 + right64)
    tiny = np.finfo(np.float64).tiny

    def kl(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return np.where(first > 0.0, first * np.log(np.clip(first, tiny, None) / np.clip(second, tiny, None)), 0.0)

    js_per_pixel = np.maximum(0.0, 0.5 * np.sum(kl(left64, midpoint) + kl(right64, midpoint), axis=0))
    tv_per_pixel = 0.5 * np.sum(difference, axis=0)
    ordered_left = np.sort(left64, axis=0)
    ordered_right = np.sort(right64, axis=0)
    reference_margin = ordered_left[-1] - ordered_left[-2] if left.shape[0] > 1 else np.ones(left.shape[1:])
    candidate_margin = ordered_right[-1] - ordered_right[-2] if right.shape[0] > 1 else np.ones(right.shape[1:])
    correlations = []
    for expert in range(left.shape[0]):
        first = left64[expert].reshape(-1)
        second = right64[expert].reshape(-1)
        first_std = float(first.std())
        second_std = float(second.std())
        if first_std <= 1e-12 or second_std <= 1e-12:
            correlations.append(
                {
                    "expert_index": expert,
                    "pearson": None,
                    "status": "UNDEFINED_CONSTANT_INPUT",
                    "reference_constant": first_std <= 1e-12,
                    "candidate_constant": second_std <= 1e-12,
                    "constant_maps_equal": bool(np.array_equal(first, second)),
                }
            )
        else:
            correlations.append(
                {
                    "expert_index": expert,
                    "pearson": float(np.corrcoef(first, second)[0, 1]),
                    "status": "DEFINED",
                    "reference_constant": False,
                    "candidate_constant": False,
                    "constant_maps_equal": None,
                }
            )
    defined = [item["pearson"] for item in correlations if item["pearson"] is not None]
    dominant_agreement = np.argmax(left, axis=0) == np.argmax(right, axis=0)
    margin_curve = []
    for percentile in (0, 25, 50, 75, 90):
        threshold = float(np.percentile(reference_margin, percentile))
        selected = reference_margin >= threshold
        margin_curve.append(
            {
                "reference_margin_percentile": percentile,
                "threshold": threshold,
                "pixel_count": int(selected.sum()),
                "dominant_expert_agreement_fraction": float(dominant_agreement[selected].mean()),
            }
        )
    changed = ~dominant_agreement
    return {
        "shape": [int(item) for item in left.shape],
        "probability_mae": float(difference.mean()),
        "probability_rmse": float(np.sqrt(np.mean((left64 - right64) ** 2))),
        "probability_max_abs_error": float(difference.max()),
        "mean_total_variation_distance": float(tv_per_pixel.mean()),
        "max_total_variation_distance": float(tv_per_pixel.max()),
        "mean_jensen_shannon_divergence_nats": float(js_per_pixel.mean()),
        "max_jensen_shannon_divergence_nats": float(js_per_pixel.max()),
        "dominant_expert_agreement_fraction": float(dominant_agreement.mean()),
        "reference_top1_margin_mean": float(reference_margin.mean()),
        "reference_top1_margin_max": float(reference_margin.max()),
        "candidate_top1_margin_mean": float(candidate_margin.mean()),
        "candidate_top1_margin_max": float(candidate_margin.max()),
        "changed_pixel_reference_margin_mean": float(reference_margin[changed].mean()) if changed.any() else None,
        "unchanged_pixel_reference_margin_mean": (
            float(reference_margin[dominant_agreement].mean()) if dominant_agreement.any() else None
        ),
        "agreement_at_or_above_reference_margin_percentiles": margin_curve,
        "expert_pearson": correlations,
        "defined_pearson_count": len(defined),
        "undefined_pearson_count": len(correlations) - len(defined),
        "defined_pearson_mean": float(np.mean(defined)) if defined else None,
    }


def aggregate_stability_comparisons(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate equal-weight comparison records by type, family and target size."""

    metrics = (
        "probability_mae",
        "probability_rmse",
        "probability_max_abs_error",
        "mean_total_variation_distance",
        "mean_jensen_shannon_divergence_nats",
        "dominant_expert_agreement_fraction",
        "reference_top1_margin_mean",
        "candidate_top1_margin_mean",
    )

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {"comparison_count": len(items)}
        for metric in metrics:
            values = np.asarray([item["metrics"][metric] for item in items], dtype=np.float64)
            result[metric] = {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}
        correlations = [
            correlation["pearson"]
            for item in items
            for correlation in item["metrics"]["expert_pearson"]
            if correlation["pearson"] is not None
        ]
        total_correlations = sum(len(item["metrics"]["expert_pearson"]) for item in items)
        result["expert_pearson"] = {
            "defined_count": len(correlations),
            "undefined_count": total_correlations - len(correlations),
            "defined_mean": float(np.mean(correlations)) if correlations else None,
            "defined_min": float(np.min(correlations)) if correlations else None,
            "defined_max": float(np.max(correlations)) if correlations else None,
        }
        for key in ("changed_pixel_reference_margin_mean", "unchanged_pixel_reference_margin_mean"):
            values = [item["metrics"][key] for item in items if item["metrics"][key] is not None]
            result[key] = {
                "defined_comparison_count": len(values),
                "mean": float(np.mean(values)) if values else None,
                "min": float(np.min(values)) if values else None,
                "max": float(np.max(values)) if values else None,
            }
        curve = {}
        for percentile in (0, 25, 50, 75, 90):
            entries = [
                next(
                    entry
                    for entry in item["metrics"]["agreement_at_or_above_reference_margin_percentiles"]
                    if entry["reference_margin_percentile"] == percentile
                )
                for item in items
            ]
            agreements = np.asarray(
                [entry["dominant_expert_agreement_fraction"] for entry in entries], dtype=np.float64
            )
            curve[str(percentile)] = {
                "comparison_count": len(entries),
                "mean_threshold": float(np.mean([entry["threshold"] for entry in entries])),
                "selected_pixel_count_total": sum(entry["pixel_count"] for entry in entries),
                "agreement_mean": float(agreements.mean()),
                "agreement_min": float(agreements.min()),
                "agreement_max": float(agreements.max()),
            }
        result["agreement_at_or_above_reference_margin_percentiles"] = curve
        return result

    by_resolution: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    by_module: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_seed: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in comparisons:
        by_resolution[(item["comparison_type"], item["family"], int(item["candidate_resolution"]))].append(item)
        by_module[(item["comparison_type"], item["family"], item["module"])].append(item)
        by_seed[(item["comparison_type"], item["family"], int(item["seed"]))].append(item)
    return {
        "method": {
            "unit": "one seed x sample x router-module aligned comparison",
            "weighting": "each comparison receives equal weight",
            "probability_space": "original-image pixels after exact letterbox inverse and expert re-normalization",
            "correlation_policy": "undefined when either aligned expert map is constant",
            "interpretation_boundary": "random-initialization pipeline diagnostic; no learned robustness claim",
        },
        "comparison_count": len(comparisons),
        "by_type_family_resolution": {
            f"{kind}:{family}:{resolution}": summarize(items)
            for (kind, family, resolution), items in sorted(by_resolution.items())
        },
        "by_type_family_module": {
            f"{kind}:{family}:{module}": summarize(items)
            for (kind, family, module), items in sorted(by_module.items())
        },
        "by_type_family_seed": {
            f"{kind}:{family}:seed-{seed}": summarize(items)
            for (kind, family, seed), items in sorted(by_seed.items())
        },
    }
