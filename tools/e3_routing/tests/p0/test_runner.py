from __future__ import annotations

from pathlib import Path

import pytest

from e3_p0.runner import _load_config


def test_run_id_override_is_written_to_resolved_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run_id: archived-evidence\nseed: 0\nsample_indices: [0]\n", encoding="utf-8")

    config = _load_config(config_path, "repro-local-001")

    assert config["run_id"] == "repro-local-001"
    assert config["seed"] == 0


def test_run_id_rejects_path_traversal_and_separators(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run_id: archived-evidence\nsample_indices: [0]\n", encoding="utf-8")

    for invalid in ("", ".", "../escape", "nested/run", r"nested\run"):
        with pytest.raises(ValueError, match="run_id"):
            _load_config(config_path, invalid)


def test_config_rejects_duplicate_sample_indices(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run_id: test\nsample_indices: [0, 0]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicates"):
        _load_config(config_path)


def test_legacy_single_sample_config_is_upgraded(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run_id: test\nsample_index: 2\n", encoding="utf-8")

    config = _load_config(config_path)

    assert config["sample_indices"] == [2]
    assert config["primary_batch_size"] == 1
