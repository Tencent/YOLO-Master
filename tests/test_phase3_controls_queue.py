"""Regression tests for the serial Phase 3 controls queue."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.issue54 import run_phase3_controls_queue as queue


def _queue_args(controls_file: Path, mot_file: Path, *extra: str) -> list[str]:
    return [
        "--controls-root-file",
        str(controls_file),
        "--mot-root-file",
        str(mot_file),
        "--dataset-manifest",
        str(controls_file.parent / "dataset-manifest.json"),
        *extra,
    ]


def test_default_queue_order() -> None:
    assert queue.parse_runs(None) == (("v10", 0), ("v10", 1), ("v10", 2), ("v10_moa", 0))


def test_duplicate_run_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        queue.parse_runs(["v10:0", "v10:0"])


def test_existing_passed_run_is_skipped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    controls = tmp_path / "controls"
    controls.mkdir()
    mot = tmp_path / "mot"
    mot.mkdir()
    (controls / "phase3_v10_seed0").mkdir()
    controls_file = tmp_path / "controls.txt"
    controls_file.write_text(str(controls))
    mot_file = tmp_path / "mot.txt"
    mot_file.write_text(str(mot))
    monkeypatch.setattr(queue, "mot_hashes", lambda _root: {"mot"})
    monkeypatch.setattr(queue, "validate_run", lambda *_args: None)
    monkeypatch.setattr(queue, "run_one", lambda *_args: pytest.fail("passed run must not relaunch"))
    assert queue.main(_queue_args(controls_file, mot_file, "--runs", "v10:0")) == 0


def test_incomplete_directory_stops_before_launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    controls = tmp_path / "controls"
    controls.mkdir()
    mot = tmp_path / "mot"
    mot.mkdir()
    (controls / "phase3_v10_seed0").mkdir()
    controls_file = tmp_path / "controls.txt"
    controls_file.write_text(str(controls))
    mot_file = tmp_path / "mot.txt"
    mot_file.write_text(str(mot))
    monkeypatch.setattr(queue, "mot_hashes", lambda _root: {"mot"})
    monkeypatch.setattr(queue, "validate_run", lambda *_args: "missing best.pt")
    monkeypatch.setattr(queue, "run_one", lambda *_args: pytest.fail("incomplete run must not relaunch"))
    assert queue.main(_queue_args(controls_file, mot_file, "--runs", "v10:0")) == 1


def test_failure_does_not_continue(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    controls = tmp_path / "controls"
    controls.mkdir()
    mot = tmp_path / "mot"
    mot.mkdir()
    controls_file = tmp_path / "controls.txt"
    controls_file.write_text(str(controls))
    mot_file = tmp_path / "mot.txt"
    mot_file.write_text(str(mot))
    launched = []
    monkeypatch.setattr(queue, "mot_hashes", lambda _root: {"mot"})
    monkeypatch.setattr(queue, "preflight", lambda *_args: None)
    monkeypatch.setattr(
        queue,
        "run_one",
        lambda _root, model, seed, _poll, **_kwargs: launched.append((model, seed)) or 2,
    )
    assert queue.main(_queue_args(controls_file, mot_file, "--runs", "v10:0", "v10:1")) == 1
    assert launched == [("v10", 0)]


def test_duplicate_checkpoint_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run = tmp_path / "phase3_v10_seed0"
    checkpoint = run / "training/v10/weights/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"same")
    (run / "training/v10/weights/last.pt").write_bytes(b"last")
    results = run / "training/v10/results.csv"
    results.write_text("epoch,metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss\n29,0.5,0.3,1.0\n")
    manifest = run / "experiment_manifest.json"
    manifest.write_text("{}")
    (run / "exitcode").write_text("0\n")
    monkeypatch.setattr(queue, "load_checkpoint", lambda _path: None)
    monkeypatch.setattr(queue, "load_json", lambda _path: {})
    monkeypatch.setattr(
        queue,
        "validate_experiment_manifest",
        lambda _payload: {
            "model_variant": "v10",
            "seed": 0,
            "status": "passed",
            "failure_reason": None,
            "requested_epochs": 30,
            "requested_batch": 8,
            "batch": 8,
            "effective_batch": 8,
            "imgsz": 640,
            "precision_mode": "amp",
            **queue.EXPECTED,
            "checkpoint_path": "training/v10/weights/best.pt",
            "checkpoint_sha256": queue.sha256_file(checkpoint),
        },
    )
    assert "duplicate checkpoint" in queue.validate_run(tmp_path, "v10", 0, {queue.sha256_file(checkpoint)})


def test_checkpoint_path_resolves_relative_to_run_and_accepts_absolute(tmp_path: Path) -> None:
    run = tmp_path / "phase3_v10_seed0"
    relative = Path("training/v10/weights/best.pt")
    assert queue.resolve_checkpoint(run, relative) == run / relative
    absolute = tmp_path / "external.pt"
    assert queue.resolve_checkpoint(run, absolute.resolve()) == absolute.resolve()
