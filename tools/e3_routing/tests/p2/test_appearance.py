import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from e3_p2.appearance_runner import (
    _aggregate_region_comparisons,
    _apply_transform,
    _load_config,
    _prepare_inputs,
    _summarize_input_effects,
)
from e3_p2.stability import probability_map_comparison


def test_appearance_transforms_are_deterministic_and_preserve_geometry():
    image = Image.new("RGB", (8, 4), (100, 120, 140))
    identity = np.asarray(_apply_transform(image, {"name": "identity", "kind": "identity"}))
    brighter = np.asarray(
        _apply_transform(image, {"name": "brightness_110", "kind": "brightness", "factor": 1.1})
    )
    blurred = np.asarray(
        _apply_transform(image, {"name": "gaussian_blur_075", "kind": "gaussian_blur", "radius": 0.75})
    )
    assert identity.shape == brighter.shape == blurred.shape == (4, 8, 3)
    assert np.array_equal(identity, np.asarray(image))
    assert brighter.mean() > identity.mean()
    assert np.array_equal(blurred, identity)  # a solid-color image is unchanged by blur


def test_config_rejects_identity_or_factor_contract_drift(tmp_path: Path):
    base = {
        "run_id": "run",
        "device": "cpu",
        "resolution": 128,
        "seeds": [0, 1],
        "sample_indices": [0],
        "spatial_profiles": {"mot": {}, "moa": {}},
        "transformations": [
            {"name": "identity", "kind": "identity"},
            {"name": "brightness", "kind": "brightness", "factor": 1.1},
        ],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    assert _load_config(path, None)["resolution"] == 128
    base["transformations"][1]["factor"] = 1.0
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    with pytest.raises(ValueError, match="factor"):
        _load_config(path, None)


def test_prepared_inputs_archive_exact_model_canvas_and_hash(tmp_path: Path):
    source = tmp_path / "source.png"
    gradient = np.zeros((5, 10, 3), dtype=np.uint8)
    gradient[:, :, 0] = np.arange(10, dtype=np.uint8) * 20
    gradient[:, :, 1] = np.arange(5, dtype=np.uint8)[:, None] * 30
    gradient[:, :, 2] = 80
    Image.fromarray(gradient).save(source)
    prepared, audit = _prepare_inputs(
        [source],
        [7],
        [
            {"name": "identity", "kind": "identity"},
            {"name": "contrast_110", "kind": "contrast", "factor": 1.1},
        ],
        32,
        tmp_path,
    )
    assert len(audit) == 2
    identity = prepared[(7, "identity")]
    assert identity["canvas"].shape == (32, 32, 3)
    assert audit[0]["model_input_rgb_bytes_sha256"] == hashlib.sha256(identity["canvas"].tobytes()).hexdigest()
    assert (tmp_path / audit[0]["transformed_artifact"]).is_file()
    assert (tmp_path / audit[0]["model_input_artifact"]).is_file()
    effects = _summarize_input_effects(prepared, [7], ["identity", "contrast_110"])
    assert effects["all_candidate_samples_changed"] is True
    assert effects["by_transform"]["contrast_110"]["changed_sample_count"] == 1


def test_input_effect_gate_rejects_a_noop_candidate():
    canvas = np.zeros((4, 4, 3), dtype=np.uint8)
    prepared = {
        (0, "identity"): {"canvas": canvas},
        (0, "noop"): {"canvas": canvas.copy()},
    }
    with pytest.raises(RuntimeError, match="no-op"):
        _summarize_input_effects(prepared, [0], ["identity", "noop"])


def test_region_aggregate_keeps_foreground_and_background_separate():
    reference = np.asarray([[[0.8, 0.6]], [[0.2, 0.4]]], dtype=np.float32)
    candidate = np.asarray([[[0.7, 0.5]], [[0.3, 0.5]]], dtype=np.float32)
    metrics = probability_map_comparison(reference, candidate)
    records = [
        {
            "comparison_type": "appearance_brightness_090",
            "family": "moa",
            "seed": 0,
            "sample_index": 0,
            "module": "router",
            "region": region,
            "token_count": 2,
            "metrics": metrics,
        }
        for region in ("foreground", "background")
    ]
    aggregate = _aggregate_region_comparisons(records)
    assert aggregate["comparison_count"] == 2
    assert aggregate["by_type_family_region"]["appearance_brightness_090:moa:foreground"][
        "comparison_count"
    ] == 1
    assert aggregate["by_type_family_region"]["appearance_brightness_090:moa:background"][
        "probability_mae"
    ]["mean"] == pytest.approx(0.1)
