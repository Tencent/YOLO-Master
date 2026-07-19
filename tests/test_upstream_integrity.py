"""Tests for the v8.4.101 upstream integrity contract."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from tools.migration import check_upstream_integrity
from tools.migration.check_upstream_integrity import UPSTREAM_COMMIT, _sha256_lf, verify_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/governance/upstream-v8.4.101-manifest.json"


def test_upstream_manifest_matches_official_tag():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["upstream"]["ref"] == "v8.4.101"
    assert data["upstream"]["commit"] == UPSTREAM_COMMIT
    assert data["counts"] == {"backends": 19, "export_modules": 17, "yolo26_configs": 10}
    assert not verify_manifest(data)


def test_upstream_manifest_verification_does_not_require_local_tag(monkeypatch):
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    monkeypatch.setattr(check_upstream_integrity, "_git_commit", lambda _: pytest.fail("tag lookup is not allowed"))

    assert not verify_manifest(data)


def test_upstream_hash_normalizes_windows_checkout_line_endings(tmp_path):
    path = tmp_path / "protected.py"
    path.write_bytes(b"first\r\nsecond\r\n")

    assert _sha256_lf(path) == hashlib.sha256(b"first\nsecond\n").hexdigest()


def test_upstream_manifest_protects_yolo26_and_export_paths():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    exact = set(data["exact_files"])
    assert "ultralytics/cfg/models/26/yolo26.yaml" in exact
    assert "ultralytics/cfg/models/26/yolo26-seg.yaml" in exact
    assert "ultralytics/nn/backends/onnx.py" in exact
    assert "ultralytics/utils/export/engine.py" in exact
    assert set(data["extension_points"]) >= {
        "ultralytics/nn/tasks.py",
        "ultralytics/utils/loss.py",
        "ultralytics/engine/trainer.py",
        "ultralytics/engine/exporter.py",
    }


def test_native_baseline_preserves_yolo26_head_contract():
    baseline = json.loads((ROOT / "reports/migration/v8.4.101-native-baseline.json").read_text(encoding="utf-8"))
    names = {item["name"] for item in baseline["models"]}
    assert names == {"detect", "segment", "semantic", "pose", "obb", "classify", "yoloe", "yoloe-seg"}
    for item in baseline["models"]:
        assert item["output_count"] > 0
        assert item["outputs_finite"] is True
        if item["name"] in {"detect", "segment", "pose", "obb", "yoloe", "yoloe-seg"}:
            assert item["reg_max"] == 1
            assert item["end2end"] is True
            assert item["one2many"] is True
            assert item["one2one"] is True
