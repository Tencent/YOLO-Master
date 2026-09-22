import numpy as np
import pytest

from e3_p2.geometry import LetterboxMeta
from e3_p2.stability import (
    aggregate_stability_comparisons,
    probability_map_comparison,
    restore_probability_stack,
)


def test_identical_nonconstant_maps_have_exact_identity_metrics():
    first = np.asarray([[[0.2, 0.8], [0.4, 0.6]], [[0.8, 0.2], [0.6, 0.4]]], dtype=np.float32)
    result = probability_map_comparison(first, first.copy())
    assert result["probability_mae"] == 0.0
    assert result["mean_jensen_shannon_divergence_nats"] == 0.0
    assert result["dominant_expert_agreement_fraction"] == 1.0
    assert result["defined_pearson_mean"] == pytest.approx(1.0)


def test_constant_correlation_is_explicitly_undefined():
    constant = np.full((2, 3, 4), 0.5, dtype=np.float32)
    result = probability_map_comparison(constant, constant.copy())
    assert result["probability_mae"] == 0.0
    assert result["defined_pearson_count"] == 0
    assert result["undefined_pearson_count"] == 2
    assert result["defined_pearson_mean"] is None
    assert all(item["status"] == "UNDEFINED_CONSTANT_INPUT" for item in result["expert_pearson"])


def test_known_probability_change_has_expected_mae_tv_and_dominant_agreement():
    first = np.asarray([[[0.9, 0.2]], [[0.1, 0.8]]], dtype=np.float32)
    second = np.asarray([[[0.8, 0.7]], [[0.2, 0.3]]], dtype=np.float32)
    result = probability_map_comparison(first, second)
    assert result["probability_mae"] == pytest.approx(0.3)
    assert result["mean_total_variation_distance"] == pytest.approx(0.3)
    assert result["dominant_expert_agreement_fraction"] == pytest.approx(0.5)
    assert result["mean_jensen_shannon_divergence_nats"] >= 0.0
    assert result["reference_top1_margin_mean"] == pytest.approx(0.7)
    curve = result["agreement_at_or_above_reference_margin_percentiles"]
    assert curve[0]["dominant_expert_agreement_fraction"] == pytest.approx(0.5)
    assert curve[3]["reference_margin_percentile"] == 75
    assert curve[3]["dominant_expert_agreement_fraction"] == 1.0
    assert result["changed_pixel_reference_margin_mean"] == pytest.approx(0.6)
    assert result["unchanged_pixel_reference_margin_mean"] == pytest.approx(0.8)


def test_horizontal_unflip_restores_the_same_coordinate_system():
    original = np.asarray(
        [[[0.1, 0.3, 0.8], [0.2, 0.4, 0.7]], [[0.9, 0.7, 0.2], [0.8, 0.6, 0.3]]],
        dtype=np.float32,
    )
    flipped_observation = np.flip(original, axis=2)
    result = probability_map_comparison(original, np.flip(flipped_observation, axis=2))
    assert result["probability_mae"] == 0.0
    assert result["dominant_expert_agreement_fraction"] == 1.0


def test_restore_probability_stack_keeps_original_shape_and_normalization():
    meta = LetterboxMeta(8, 4, 8, 8, 4, 0, 2, 0, 2, 1.0)
    weights = np.asarray(
        [
            [[0.2, 0.4], [0.6, 0.8]],
            [[0.8, 0.6], [0.4, 0.2]],
        ],
        dtype=np.float32,
    )
    restored, validation = restore_probability_stack(weights, meta)
    assert restored.shape == (2, 4, 8)
    assert np.allclose(restored.sum(axis=0), 1.0)
    assert validation["post_normalization_max_expert_sum_error"] <= 1e-6


def test_aggregate_counts_defined_and_undefined_correlations():
    variable = np.asarray([[[0.2, 0.8]], [[0.8, 0.2]]], dtype=np.float32)
    constant = np.full((2, 1, 2), 0.5, dtype=np.float32)
    comparisons = []
    for metrics in (probability_map_comparison(variable, variable), probability_map_comparison(constant, constant)):
        comparisons.append(
            {
                "comparison_type": "horizontal_flip",
                "family": "mot",
                "candidate_resolution": 64,
                "module": "router",
                "seed": 0,
                "metrics": metrics,
            }
        )
    aggregate = aggregate_stability_comparisons(comparisons)
    summary = aggregate["by_type_family_resolution"]["horizontal_flip:mot:64"]
    assert summary["comparison_count"] == 2
    assert summary["expert_pearson"]["defined_count"] == 2
    assert summary["expert_pearson"]["undefined_count"] == 2
    assert aggregate["by_type_family_module"]["horizontal_flip:mot:router"]["comparison_count"] == 2
    assert aggregate["by_type_family_seed"]["horizontal_flip:mot:seed-0"]["comparison_count"] == 2
