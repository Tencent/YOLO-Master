from __future__ import annotations

import pytest
import torch

from scripts.evaluate_mot_utility_router import (
    adaptive_k_from_probabilities,
    blend_router_probabilities,
    blend_sweep,
    comparison_rows,
    mean_router_kl,
    parse_blend_alphas,
    parse_thresholds,
    threshold_summary,
)


def test_adaptive_k_uses_cumulative_mass_and_clamps_to_max_k():
    probabilities = torch.tensor([[0.70, 0.20, 0.10], [0.40, 0.35, 0.25]])

    selected = adaptive_k_from_probabilities(probabilities, max_k=2, threshold=0.6)

    assert selected.tolist() == [1, 2]
    assert adaptive_k_from_probabilities(probabilities, max_k=2, threshold=0.95).tolist() == [2, 2]


def test_threshold_summary_reports_compute_savings():
    probabilities = torch.tensor([[0.70, 0.20, 0.10], [0.40, 0.35, 0.25]])

    summary = threshold_summary(probabilities, max_k=2, thresholds=[0.6])[0]

    assert summary["mean_k"] == pytest.approx(1.5)
    assert summary["k_1_share"] == pytest.approx(0.5)
    assert summary["expert_sample_saving_vs_fixed_max_k"] == pytest.approx(0.25)


def test_comparison_rows_use_forced_detection_losses():
    rows = comparison_rows(
        ("a.jpg",),
        ("seq",),
        torch.tensor([[0.8, 0.1, 0.1]]),
        torch.tensor([[0.1, 0.1, 0.8]]),
        torch.tensor([[0.1, 0.1, 0.8]]),
        torch.tensor([[2.0, 3.0, 1.0]]),
        ("local", "window", "deformable"),
    )

    assert rows[0]["baseline_regret"] == pytest.approx(1.0)
    assert rows[0]["utility_regret"] == pytest.approx(0.0)
    assert rows[0]["deployment_regret"] == pytest.approx(0.0)
    assert rows[0]["regret_reduction"] == pytest.approx(1.0)


def test_threshold_parser_rejects_out_of_range_values():
    assert parse_thresholds("0.5, 0.4, 0.5") == [0.4, 0.5]
    with pytest.raises(Exception, match="threshold"):
        parse_thresholds("1.1")


def test_trust_region_blend_and_sweep_preserve_normalization():
    baseline = torch.tensor([[0.7, 0.2, 0.1]])
    utility = torch.tensor([[0.1, 0.2, 0.7]])
    targets = torch.tensor([[0.1, 0.1, 0.8]])
    losses = torch.tensor([[2.0, 3.0, 1.0]])

    blended = blend_router_probabilities(baseline, utility, alpha=0.5)
    sweep = blend_sweep(baseline, utility, targets, losses, [0.0, 1.0])

    assert blended.sum() == pytest.approx(1.0)
    assert sweep[0]["mean_regret"] == pytest.approx(1.0)
    assert sweep[1]["mean_regret"] == pytest.approx(0.0)
    assert parse_blend_alphas("1,0.4,0") == [0.0, 0.4, 1.0]
    assert mean_router_kl(baseline, baseline) == pytest.approx(0.0)
    assert mean_router_kl(baseline, utility) > 0
