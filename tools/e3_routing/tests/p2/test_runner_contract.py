from pathlib import Path

import pytest
import yaml

from e3_p2.demo import resolve_run_dir
from e3_p2.runner import _demo_html, _load_config, _slug, _verify_project_source_state


def _config() -> dict:
    return {
        "run_id": "safe-run",
        "sample_indices": [0],
        "spatial_profiles": {"mot": {}, "moa": {}},
        "non_spatial_profiles": {"moe": {}, "latent": {}, "molora": {}},
        "batch_equivalence_sizes": [2],
        "region_analysis": {
            "enabled": True,
            "label_format": "yolo_detection_normalized_xywh",
            "assignment_rule": "valid_token_center_inside_any_ground_truth_box",
            "exclude_letterbox_padding": True,
        },
    }


def test_run_id_cannot_escape_evidence_directory(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_config()), encoding="utf-8")
    assert _load_config(path)["run_id"] == "safe-run"
    with pytest.raises(ValueError, match="path-safe"):
        _load_config(path, "../escape")


def test_family_contract_is_explicit(tmp_path: Path):
    config = _config()
    config["non_spatial_profiles"].pop("molora")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        _load_config(path)


def test_slug_is_stable_for_module_paths():
    assert _slug("model.13.m.0/router") == "model.13.m.0-router"


def test_demo_requires_both_index_and_html(tmp_path: Path):
    (tmp_path / "demo.html").write_text("ok", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Not a P2 demo"):
        resolve_run_dir(tmp_path)


def test_repository_implementation_state_is_inspectable():
    state = _verify_project_source_state(False)
    assert set(state) == {"commit", "tree", "implementation_status_porcelain", "implementation_clean"}


def test_invalid_batch_equivalence_size_is_rejected(tmp_path: Path):
    config = _config()
    config["batch_equivalence_sizes"] = [1]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match=">= 2"):
        _load_config(path)


def test_region_assignment_contract_cannot_be_silently_changed(tmp_path: Path):
    config = _config()
    config["region_analysis"]["exclude_letterbox_padding"] = False
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="region_analysis"):
        _load_config(path)


def test_demo_exposes_original_entropy_margin_and_failure_state():
    html = _demo_html()
    assert 'id=\"original\"' in html
    assert 'id=\"groundTruth\"' in html
    assert "Normalized routing entropy" in html
    assert "Top-1 routing margin" in html
    assert "FG/BG TV" in html
    assert "Evidence failed to load" in html
