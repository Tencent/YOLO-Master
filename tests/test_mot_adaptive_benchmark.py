from __future__ import annotations

import pytest

from scripts.benchmark_mot_adaptive_k import (
    aggregate_timing_rounds,
    latency_percentiles,
    scalar_metrics,
)


def test_latency_percentiles_are_ordered():
    metrics = latency_percentiles([1.0, 2.0, 3.0, 4.0, 10.0])

    assert metrics["latency_p50_ms"] <= metrics["latency_p95_ms"] <= metrics["latency_p99_ms"]
    assert metrics["latency_mean_ms"] == pytest.approx(4.0)


def test_latency_percentiles_reject_empty_values():
    with pytest.raises(ValueError, match="non-empty"):
        latency_percentiles([])


def test_scalar_metrics_filters_non_scalars_and_non_finite_values():
    metrics = scalar_metrics({"map": 0.5, "bad": float("nan"), "curve": [1, 2]})

    assert metrics == {"map": 0.5}


def test_aggregate_timing_rounds_uses_median_and_preserves_spread():
    rows = [
        {
            "latency_mean_ms": value,
            "latency_p50_ms": value,
            "latency_p95_ms": value + 1,
            "latency_p99_ms": value + 2,
            "mean_selected_k": 1.5,
            "mean_expert_sample_calls": 2.0,
            "expert_sample_saving_vs_dense": 1 / 3,
        }
        for value in (30.0, 10.0, 20.0)
    ]

    summary = aggregate_timing_rounds(rows)

    assert summary["latency_p50_ms"] == 20.0
    assert summary["latency_p50_ms_run_min"] == 10.0
    assert summary["latency_p50_ms_run_max"] == 30.0
