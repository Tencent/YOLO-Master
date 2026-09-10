from __future__ import annotations

from copy import deepcopy

import pytest

from e3_p0.aggregation import aggregate_events, compare_batch_aggregates, compare_repeated_runs


def _event(load, *, sample_indices, pass_name="primary"):
    return {
        "schema_version": "e3.routing/v1.0.0",
        "run_id": "test",
        "sequence": 0,
        "captured_at": "ignored",
        "family": "moe",
        "module": {"name": "route", "type": "Fake", "python_module": "fake"},
        "routing": {
            "num_experts": 2,
            "top_k": 1,
            "granularity": "test",
            "expert_load": load,
            "expert_load_sum": 1.0,
            "entropy_nats": 0.0,
            "entropy_normalized": 0.0,
            "load_gini": abs(load[0] - load[1]) / 2,
            "dominant_expert_share": max(load),
            "mixing_weights": {"state": "derived", "source": "expert_load", "value": load},
        },
        "aux_loss": {
            "state": "unavailable",
            "source": None,
            "value": None,
            "finite": None,
            "mode": "eval_observation",
        },
        "runtime": {
            "training": False,
            "input_shape": [len(sample_indices), 3, 2, 2],
            "output_shapes": [[len(sample_indices), 2]],
            "pass_name": pass_name,
            "batch_size": len(sample_indices),
            "batch_index": 0,
            "sample_indices": sample_indices,
            "input_ids": [str(index) for index in sample_indices],
        },
        "provenance": {"adapter_version": "test"},
        "source_snapshot": {"expert_usage": load},
    }


def test_aggregate_events_uses_batch_size_weighting():
    aggregate = aggregate_events([_event([1.0, 0.0], sample_indices=[0]), _event([0.0, 1.0], sample_indices=[1, 2, 3])])[0]

    assert aggregate["sample_observations"] == 4
    assert aggregate["expert_load_mean"] == pytest.approx([0.25, 0.75])


def test_repeated_run_ignores_time_run_sequence_and_pass_name():
    reference = _event([0.4, 0.6], sample_indices=[0])
    repeated = deepcopy(reference)
    repeated.update(run_id="other", sequence=99, captured_at="later")
    repeated["runtime"]["pass_name"] = "repeat-2"

    comparison = compare_repeated_runs([reference], [repeated])

    assert comparison["status"] == "PASS"
    assert comparison["matching_events"] == 1


def test_batch_consistency_compares_weighted_module_loads():
    reference = [_event([0.8, 0.2], sample_indices=[0]), _event([0.2, 0.8], sample_indices=[1])]
    candidate = [_event([0.5, 0.5], sample_indices=[0, 1], pass_name="batch-2")]

    comparison = compare_batch_aggregates(reference, candidate, candidate_batch_size=2, tolerance=1e-9)

    assert comparison["status"] == "PASS"
    assert comparison["max_abs_load_delta"] == pytest.approx(0.0)
