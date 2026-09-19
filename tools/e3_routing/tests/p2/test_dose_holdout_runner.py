from pathlib import Path

import pytest
import yaml

from e3_p2.dose_holdout_runner import _load_config, _slope_association, analyze_holdout, build_holdout_records

FAMILIES = [{"name": "brightness", "transforms": ["low", "middle", "high"]}]


def _parent_records(*, zero_switch_span: bool = False):
    records = []
    for image in range(4):
        x_values = [1.0 + image, 2.0 + 2 * image, 4.0 + 4 * image]
        for level, (transform, x) in enumerate(zip(("low", "middle", "high"), x_values)):
            probability = 2.0 * x + image
            switch = 0.2 if zero_switch_span else probability / 100.0
            records.append(
                {
                    "family": "brightness",
                    "transform": transform,
                    "sample_index": image,
                    "input_rgb_mae_0_255": x,
                    "target_probability_mae_mean_across_seeds": probability,
                    "target_dominant_switch_fraction_mean_across_seeds": switch,
                    "seed_values": [
                        {
                            "seed": seed,
                            "probability_mae": probability * (seed + 1),
                            "dominant_switch_fraction": switch,
                        }
                        for seed in (0, 1, 2)
                    ],
                }
            )
    return records


def test_holdout_prediction_is_exact_for_linear_response_and_reproducible():
    records = build_holdout_records(_parent_records(), FAMILIES, 4, [0, 1, 2])
    assert len(records) == 4
    assert records[0]["endpoints"]["probability_mae"]["absolute_residual_input_weighted"] == pytest.approx(0.0)
    first = analyze_holdout(records, FAMILIES, [0, 1, 2], 1000, 7)
    second = analyze_holdout(records, FAMILIES, [0, 1, 2], 1000, 7)
    assert first == second
    metric = first["by_family"]["brightness"]["endpoints"]["probability_mae"]
    assert metric["input_weighted_absolute_error"]["observed_mean"] == pytest.approx(0.0)


def test_zero_endpoint_span_keeps_null_normalized_error():
    records = build_holdout_records(_parent_records(zero_switch_span=True), FAMILIES, 4, [0, 1, 2])
    for item in records:
        assert item["endpoints"]["dominant_switch_fraction"]["normalized_absolute_residual"] is None
    result = analyze_holdout(records, FAMILIES, [0, 1, 2], 1000, 11)
    normalized = result["by_family"]["brightness"]["endpoints"]["dominant_switch_fraction"][
        "normalized_absolute_error"
    ]
    assert normalized["defined_image_count"] == 0
    assert normalized["zero_span_image_count"] == 4


def test_sparse_slope_resampling_counts_undefined_draws_without_crashing():
    import numpy as np

    x = np.asarray([0.0, 0.0, 0.0, 1.0])
    y = np.asarray([0.0, 0.0, 0.0, 2.0])
    result = _slope_association(x, y, 1000, 23)
    assert result["status"] == "DEFINED"
    assert result["bootstrap"]["undefined_draw_count"] > 0
    assert result["leave_one_image_out"]["undefined_count"] == 1


def test_holdout_rejects_incomplete_matrix_and_unordered_input():
    parent = _parent_records()
    with pytest.raises(ValueError, match="parent dose matrix mismatch"):
        build_holdout_records(parent[:-1], FAMILIES, 4, [0, 1, 2])
    parent[1]["input_rgb_mae_0_255"] = parent[0]["input_rgb_mae_0_255"]
    with pytest.raises(ValueError, match="strictly ordered"):
        build_holdout_records(parent, FAMILIES, 4, [0, 1, 2])


def test_config_rejects_reused_transform(tmp_path: Path):
    config = {
        "run_id": "holdout-test",
        "expected_image_count": 4,
        "expected_seeds": [0, 1],
        "bootstrap_draws": 1000,
        "max_evidence_bytes": 100_000,
        "expected_parent_manifest_sha256": "0" * 64,
        "families": [
            {"name": name, "low_transform": "same", "middle_transform": f"{name}-m", "high_transform": f"{name}-h"}
            for name in ("a", "b", "c")
        ],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="globally unique"):
        _load_config(path, None)
