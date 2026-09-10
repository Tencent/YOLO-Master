from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from e3_p2.blur_residual_runner import (
    FEATURES,
    _load_config,
    analyze_records,
    extract_input_features,
    holm_adjust,
    permutation_p_value,
)


def test_input_features_distinguish_uniform_and_checkerboard(tmp_path: Path):
    uniform = np.full((10, 20, 3), 100, dtype=np.uint8)
    checker = ((np.indices((10, 20)).sum(axis=0) % 2) * 255).astype(np.uint8)
    checker = np.repeat(checker[..., None], 3, axis=2)
    uniform_path, checker_path = tmp_path / "uniform.png", tmp_path / "checker.png"
    Image.fromarray(uniform).save(uniform_path)
    Image.fromarray(checker).save(checker_path)

    plain = extract_input_features(uniform_path, canvas_size=128)
    textured = extract_input_features(checker_path, canvas_size=128)
    assert plain["letterbox_content_fraction"] == pytest.approx(0.5)
    assert plain["luminance_mean_0_255"] == pytest.approx(100.0)
    assert plain["edge_total_variation_0_1"] == pytest.approx(0.0)
    assert textured["edge_total_variation_0_1"] == pytest.approx(1.0)


def test_permutation_is_reproducible_and_detects_perfect_order():
    x = np.arange(12, dtype=np.float64)
    y = x**2
    first = permutation_p_value(x, y, 5000, 17)
    second = permutation_p_value(x, y, 5000, 17)
    assert first == second
    assert first["two_sided_p_value"] < 0.01


def test_constant_feature_is_explicitly_undefined():
    result = permutation_p_value(np.ones(8), np.arange(8, dtype=np.float64), 5000, 3)
    assert result["status"] == "UNDEFINED_CONSTANT_VECTOR"
    assert result["two_sided_p_value"] is None


def test_holm_adjustment_is_monotone_and_keeps_undefined():
    result = holm_adjust({"a": 0.01, "b": 0.03, "c": None}, 0.05)
    assert result["a"]["holm_adjusted_p_value"] == pytest.approx(0.02)
    assert result["b"]["holm_adjusted_p_value"] == pytest.approx(0.03)
    assert result["a"]["reject_at_alpha"] is True
    assert result["b"]["reject_at_alpha"] is True
    assert result["c"]["holm_adjusted_p_value"] is None


def test_analysis_is_deterministic_and_uses_all_predeclared_features():
    records = []
    for index in range(16):
        records.append(
            {
                "sample_index": index,
                "image_name": f"{index}.jpg",
                "normalized_absolute_residual": (index + 1) / 100.0,
                "features": {
                    "letterbox_content_fraction": 0.5 + index / 100.0,
                    "luminance_mean_0_255": 100.0 - index,
                    "edge_total_variation_0_1": 0.1 + (index % 5) / 100.0,
                },
            }
        )
    kwargs = {
        "features": FEATURES,
        "bootstrap_draws": 1000,
        "bootstrap_seed": 5,
        "permutation_draws": 5000,
        "permutation_seed": 9,
        "alpha": 0.05,
        "top_k": 4,
    }
    first = analyze_records(records, **kwargs)
    second = analyze_records(records, **kwargs)
    assert first == second
    assert set(first["associations"]) == set(FEATURES)
    assert len(first["top_residual_images_descriptive_only"]) == 4


def _valid_config() -> dict:
    return {
        "run_id": "p2x-test",
        "features": list(FEATURES),
        "family": "gaussian_blur",
        "endpoint": "probability_mae",
        "expected_image_count": 16,
        "bootstrap_draws": 1000,
        "permutation_draws": 5000,
        "alpha": 0.05,
        "top_k": 4,
        "expected_holdout_manifest_sha256": "a" * 64,
        "expected_image_source_manifest_sha256": "b" * 64,
        "expected_selected_image_and_label_set_sha256": "c" * 64,
        "max_evidence_bytes": 100_000,
    }


def test_config_rejects_feature_order_change(tmp_path: Path):
    config = _valid_config()
    config["features"] = list(reversed(FEATURES))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="predeclared order"):
        _load_config(path, None)


def test_config_accepts_locked_contract(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_valid_config()), encoding="utf-8")
    loaded = _load_config(path, None)
    assert loaded["run_id"] == "p2x-test"
    assert tuple(loaded["features"]) == FEATURES
