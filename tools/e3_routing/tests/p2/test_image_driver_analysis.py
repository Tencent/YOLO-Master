import numpy as np
import pytest

from e3_p2.image_driver_analysis import (
    average_ranks,
    bootstrap_spearman,
    build_image_records,
    leave_one_out_spearman,
    spearman,
)


def test_average_ranks_and_spearman_handle_ties_and_constants():
    assert average_ranks(np.array([30.0, 10.0, 10.0, 20.0])).tolist() == [4.0, 1.5, 1.5, 3.0]
    assert spearman(np.arange(5), np.arange(5)) == pytest.approx(1.0)
    assert spearman(np.arange(5), -np.arange(5)) == pytest.approx(-1.0)
    assert spearman(np.ones(5), np.arange(5)) is None


def test_bootstrap_and_leave_one_out_are_deterministic():
    x = np.arange(12, dtype=np.float64)
    y = x**2 + np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    first = bootstrap_spearman(x, y, 1000, 7)
    second = bootstrap_spearman(x, y, 1000, 7)
    assert first == second
    assert first["defined_draw_count"] == 1000
    leave_one_out = leave_one_out_spearman(x, y)
    assert leave_one_out["count"] == 12
    assert leave_one_out["minimum"] > 0.9


def _effect(transform: str, sample: int) -> dict:
    return {
        "transform": transform,
        "sample_index": sample,
        "rgb_mae_0_255": float(sample + 1),
        "model_input_changed": True,
    }


def _case(transform: str, sample: int, seed: int) -> dict:
    return {
        "transform": transform,
        "sample_index": sample,
        "seed": seed,
        "probability_mae": float(sample + seed),
        "dominant_switch_fraction": float(sample + seed) / 10.0,
    }


def test_build_image_records_uses_complete_seed_averages():
    effects = [_effect("dim", sample) for sample in range(4)]
    cases = [_case("dim", sample, seed) for sample in range(4) for seed in (0, 1, 2)]
    records = build_image_records(effects, cases, ["dim"], 4, [0, 1, 2])
    assert len(records) == 4
    assert records[2]["target_probability_mae_mean_across_seeds"] == pytest.approx(3.0)


def test_build_image_records_rejects_incomplete_seed_matrix():
    effects = [_effect("dim", sample) for sample in range(4)]
    cases = [_case("dim", sample, seed) for sample in range(4) for seed in (0, 1, 2)]
    with pytest.raises(ValueError, match="incomplete or duplicate seed matrix"):
        build_image_records(effects, cases[:-1], ["dim"], 4, [0, 1, 2])
