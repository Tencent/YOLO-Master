import numpy as np
import pytest

from e3_p2.dose_response_runner import (
    build_dose_records,
    summarize_dose_response,
)

FAMILIES = [
    {
        "name": "brightness",
        "conditions": [
            {"transform": "low", "severity": 1.0, "label": "low"},
            {"transform": "mid", "severity": 2.0, "label": "mid"},
            {"transform": "high", "severity": 3.0, "label": "high"},
        ],
    }
]


def _fixtures(nonmonotone_switch: bool = False):
    effects, cases = [], []
    for image in range(4):
        for level, transform in enumerate(("low", "mid", "high"), start=1):
            effects.append(
                {
                    "transform": transform,
                    "sample_index": image,
                    "rgb_mae_0_255": float(level * (image + 1)),
                    "model_input_changed": True,
                }
            )
            for seed in (0, 1):
                switch = float((4 - level if nonmonotone_switch else level) + image) / 20.0
                cases.append(
                    {
                        "transform": transform,
                        "sample_index": image,
                        "seed": seed,
                        "probability_mae": float(level * (image + 1) + seed),
                        "dominant_switch_fraction": switch,
                    }
                )
    return effects, cases


def test_dose_summary_preserves_primary_and_negative_secondary_results():
    effects, cases = _fixtures(nonmonotone_switch=True)
    records = build_dose_records(effects, cases, FAMILIES, 4, [0, 1])
    result = summarize_dose_response(records, FAMILIES, 4, [0, 1], 1000, 7)
    family = result["by_family"]["brightness"]
    assert family["monotonicity"]["input_rgb_mae_0_255"]["nondecreasing_image_count"] == 4
    assert family["monotonicity"]["target_probability_mae"]["nondecreasing_image_count"] == 4
    assert family["monotonicity"]["target_dominant_switch_fraction"]["nondecreasing_image_count"] == 0
    assert family["high_minus_low_paired_difference"]["target_probability_mae"]["observed_mean"] > 0


def test_dose_records_reject_incomplete_seed_matrix():
    effects, cases = _fixtures()
    with pytest.raises(ValueError, match="incomplete or duplicate dose seed matrix"):
        build_dose_records(effects, cases[:-1], FAMILIES, 4, [0, 1])


def test_dose_summary_rejects_nonmonotone_input_ladder():
    effects, cases = _fixtures()
    effects[-1]["rgb_mae_0_255"] = 0.0
    records = build_dose_records(effects, cases, FAMILIES, 4, [0, 1])
    with pytest.raises(RuntimeError, match="input severity ladder is not monotone"):
        summarize_dose_response(records, FAMILIES, 4, [0, 1], 1000, 7)


def test_high_minus_low_bootstrap_is_image_paired():
    effects, cases = _fixtures()
    records = build_dose_records(effects, cases, FAMILIES, 4, [0, 1])
    result = summarize_dose_response(records, FAMILIES, 4, [0, 1], 1000, 11)
    difference = result["by_family"]["brightness"]["high_minus_low_paired_difference"]
    assert difference["input_rgb_mae_0_255"]["unit_count"] == 4
    assert np.isfinite(difference["target_probability_mae"]["observed_mean"])
