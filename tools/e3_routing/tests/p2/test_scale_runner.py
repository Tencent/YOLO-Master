from pathlib import Path

import numpy as np
import pytest

from e3_p2.scale_runner import (
    _load_config,
    archive_selected_inputs,
    bootstrap_layer_share_interval,
    bootstrap_mean_interval,
    deterministic_image_selection,
    summarize_image_level_attribution,
)


def test_scale_config_keeps_default_contract_but_allows_explicit_extended_modes(tmp_path: Path):
    import yaml

    base = {
        "run_id": "test-dose",
        "device": "cpu",
        "resolution": 128,
        "family": "moa",
        "target_module": "target",
        "seeds": [0, 1],
        "selected_image_count": 16,
        "expected_dataset_image_count": 16,
        "selection_salt": "fixed",
        "bootstrap_draws": 1000,
        "max_evidence_bytes": 1_000_000,
        "transformations": [{"name": "identity", "kind": "identity"}]
        + [{"name": f"condition_{index}", "kind": "brightness", "factor": 0.9} for index in range(9)],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    with pytest.raises(ValueError, match="transform contract must remain fixed"):
        _load_config(config_path, None)
    base["study_kind"] = "dose_response"
    config_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    assert _load_config(config_path, None)["study_kind"] == "dose_response"
    base["study_kind"] = "output_coupling"
    base["record_detector_outputs"] = True
    base["transformations"] = base["transformations"][:4]
    config_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    assert _load_config(config_path, None)["study_kind"] == "output_coupling"
    base["record_detector_outputs"] = False
    config_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    with pytest.raises(ValueError, match="detector-output recording"):
        _load_config(config_path, None)


def test_hash_selection_is_order_independent_and_not_content_driven():
    paths = [Path(name) for name in ("c.jpg", "a.jpg", "d.jpg", "b.jpg")]
    first, first_records = deterministic_image_selection(paths, 2, "fixed")
    second, second_records = deterministic_image_selection(list(reversed(paths)), 2, "fixed")
    assert [path.name for path in first] == [path.name for path in second]
    assert [item["selection_sha256"] for item in first_records] == [
        item["selection_sha256"] for item in second_records
    ]


def test_archive_retains_a_selected_image_without_label(tmp_path: Path):
    from PIL import Image

    image_dir = tmp_path / "images" / "train"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "background.jpg"
    Image.new("RGB", (8, 6), (20, 30, 40)).save(image_path)
    run_dir = tmp_path / "run"
    records = archive_selected_inputs([image_path], run_dir)
    assert records[0]["label_status"] == "MISSING_DATASET_LABEL"
    assert records[0]["label_sha256"] is None
    assert records[0]["label_artifact"] is None
    assert (run_dir / records[0]["image_artifact"]).is_file()


def test_bootstrap_helpers_are_deterministic_and_image_level():
    values = np.asarray([0.1, 0.2, 0.3, 0.4])
    first = bootstrap_mean_interval(values, 1000, 7)
    second = bootstrap_mean_interval(values, 1000, 7)
    assert first == second
    assert first["unit_count"] == 4
    matrix = np.asarray([[9.0, 1.0], [8.0, 2.0], [7.0, 3.0], [6.0, 4.0]])
    result = bootstrap_layer_share_interval(matrix, 0, 1000, 9)
    assert result["image_unit_count"] == 4
    assert result["observed_target_share"] == pytest.approx(0.75)
    assert result["target_rank_one_draw_fraction"] == 1.0


def _comparison(image: int, transform: str, seed: int, module: str, value: float) -> dict:
    return {
        "family": "moa",
        "sample_index": image,
        "comparison_type": f"appearance_{transform}",
        "seed": seed,
        "module": module,
        "metrics": {"probability_mae": value},
    }


def test_image_level_attribution_keeps_case_and_leave_one_out_counts():
    records = []
    for image in range(3):
        for transform in ("brightness", "blur"):
            for seed in (0, 1):
                records.append(_comparison(image, transform, seed, "target", 10.0 + image))
                records.append(_comparison(image, transform, seed, "other", 1.0))
    result = summarize_image_level_attribution(records, ["brightness", "blur"], "target", 1000, 4)
    assert result["target_rank_one_case_count"] == 12
    assert result["case_count"] == 12
    assert result["target_rank_one_image_transform_count"] == 6
    assert result["target_rank_one_image_count"] == 3
    assert result["target_rank_one_leave_one_out_count"] == 6
    assert result["leave_one_out_count"] == 6


def test_image_level_attribution_rejects_incomplete_matrix():
    records = [
        _comparison(0, "brightness", 0, "target", 2.0),
        _comparison(0, "brightness", 0, "other", 1.0),
        _comparison(1, "brightness", 0, "target", 2.0),
    ]
    with pytest.raises(ValueError, match="incomplete"):
        summarize_image_level_attribution(records, ["brightness"], "target", 1000, 4)
