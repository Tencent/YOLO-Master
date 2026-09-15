"""Contract tests for scripts/release_baseline_preflight.py."""

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "release_baseline_preflight.py"

spec = importlib.util.spec_from_file_location("release_baseline_preflight", SCRIPT)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def test_find_corrupt_weights_detects_truncated_archive(tmp_path, monkeypatch):
    """A zero-padded non-zip .pt must be reported as corrupt."""
    weights = tmp_path / "weights"
    weights.mkdir()
    with zipfile.ZipFile(weights / "good.pt", "w"):
        pass
    (weights / "bad.pt").write_bytes(b"\x00" * 128)

    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    corrupt = preflight.find_corrupt_weights()
    names = {Path(item["path"]).name for item in corrupt}
    assert "bad.pt" in names
    assert "good.pt" not in names


def test_check_datasets_flags_missing_images(tmp_path, monkeypatch):
    """A dataset YAML whose image dirs do not exist must be flagged."""
    cfg = tmp_path / "ultralytics/cfg/datasets"
    cfg.mkdir(parents=True)
    (cfg / "coco-multitask.yaml").write_text(f"path: {tmp_path}/datasets/nope\ntrain: images/train\n", encoding="utf-8")
    (tmp_path / "datasets/coco8/images/train").mkdir(parents=True)
    (tmp_path / "datasets/coco8/images/val").mkdir(parents=True)
    (cfg / "coco8.yaml").write_text(
        f"path: {tmp_path}/datasets/coco8\ntrain: images/train\nval: images/val\n", encoding="utf-8"
    )

    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    problems = preflight.check_datasets()
    flagged = {p["yaml"] for p in problems}
    assert "ultralytics/cfg/datasets/coco-multitask.yaml" in flagged
    assert "ultralytics/cfg/datasets/coco8.yaml" not in flagged


def test_report_schema_roundtrip(tmp_path):
    """The JSON report must contain the four top-level keys consumers rely on."""
    report = {
        "corrupt_weights": [],
        "dataset_problems": [],
        "network": [{"host": "github.com", "reachable": True}],
        "network_ok": True,
        "baseline_ready": True,
    }
    out = tmp_path / "report.json"
    out.write_text(json.dumps(report), encoding="utf-8")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["baseline_ready"] is True
    assert set(loaded) >= {"corrupt_weights", "dataset_problems", "network", "network_ok", "baseline_ready"}


def test_script_runs_and_reports_current_environment():
    """Smoke: the script must exit 0 or 1 (never crash) and print the readiness line."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=ROOT,
    )
    assert result.returncode in (0, 1)
    assert "baseline_ready:" in result.stdout


def test_resolved_install_paths_treats_dot_as_cwd(tmp_path, monkeypatch):
    """Editable egg-info reports location='.'; that must resolve to the preflight cwd."""
    monkeypatch.chdir(tmp_path)
    paths = preflight._resolved_install_paths(".", "")
    assert paths[0] == tmp_path.resolve()


def test_is_this_repo_accepts_nested_module_path(tmp_path):
    module = tmp_path / "ultralytics" / "__init__.py"
    module.parent.mkdir()
    module.write_text("", encoding="utf-8")
    assert preflight._is_this_repo(module, tmp_path)
    assert not preflight._is_this_repo(tmp_path.parent / "other", tmp_path)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
