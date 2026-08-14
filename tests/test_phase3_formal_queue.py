"""Tests for the Issue #54 serial Phase 3 formal controller."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.issue54 import build_phase3_mot_report as report
from scripts.issue54 import run_phase3_formal_queue as queue
from scripts.issue54 import run_phase3_seed as seed_runner
from scripts.issue54.run_phase3_formal_queue import DEFAULT_RUN_SEEDS, parse_seeds
from scripts.issue54.schema import (
    EXPERIMENT_MANIFEST_SCHEMA_VERSION,
    SchemaValidationError,
    with_manifest_checksum,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module",
    (
        "scripts.issue54.build_phase3_mot_report",
        "scripts.issue54.run_phase3_control_seed",
        "scripts.issue54.run_phase3_controls_queue",
        "scripts.issue54.run_phase3_formal_queue",
        "scripts.issue54.run_phase3_seed",
    ),
)
def test_phase3_script_imports_are_side_effect_free(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_completed_seed(root: Path, seed: int) -> None:
    run_root = root / f"phase3_v10_mot_seed{seed}"
    checkpoint = run_root / "training" / "v10_mot" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(f"seed-{seed}".encode())
    config = run_root / "config.yaml"
    config.write_text("nc: 1\n", encoding="utf-8")
    routes = run_root / "routing" / "routing.jsonl"
    routes.parent.mkdir()
    routes.write_text("{}\n" * 384, encoding="utf-8")
    payload = {
        "schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
        "experiment_id": f"phase3_v10_mot_seed{seed}",
        "model_variant": "v10_mot",
        "seed": seed,
        "dataset": "VisDrone2019-DET",
        "dataset_version": "2019-DET",
        "dataset_manifest_sha256": "0" * 64,
        "split": "val-fixed32",
        "requested_epochs": 30,
        "epochs": 30,
        "requested_batch": 8,
        "batch": 8,
        "effective_batch": 8,
        "imgsz": 640,
        "optimizer": "auto",
        "precision_mode": "fp32",
        "checkpoint_path": "training/v10_mot/weights/best.pt",
        "checkpoint_sha256": _sha256(checkpoint),
        "config_path": "config.yaml",
        "config_sha256": _sha256(config),
        "git_commit": "0123456789abcdef",
        "timestamp": "2026-08-01T00:00:00Z",
        "status": "passed",
        "failure_reason": None,
    }
    write_json(run_root / "experiment_manifest.json", with_manifest_checksum(payload))


def _launch_args(root_file: Path, *seeds: str) -> list[str]:
    args = [
        "--formal-root-file",
        str(root_file),
        "--image-manifest",
        "image-manifest.json",
        "--data-root",
        "dataset-view",
        "--dataset-manifest",
        "dataset-manifest.json",
    ]
    if seeds:
        args.extend(("--seeds", *seeds))
    return args


def test_validate_only_defaults_to_the_unfinished_queue(tmp_path: Path) -> None:
    """The default validation-only path checks seeds 1 through 4 without launching runners."""
    formal_root = tmp_path / "formal"
    formal_root.mkdir()
    for seed in range(5):
        _write_completed_seed(formal_root, seed)
    root_file = tmp_path / "PHASE3_FORMAL_ROOT.txt"
    root_file.write_text(str(formal_root), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/issue54/run_phase3_formal_queue.py"),
            "--formal-root-file",
            str(root_file),
            "--validate-only",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "validated seeds=(1, 2, 3, 4)" in result.stdout


def test_seed_zero_is_preserved_and_validate_only_checks_only_zero(tmp_path: Path) -> None:
    """An explicit seed zero must not fall back to the default unfinished queue."""
    formal_root = tmp_path / "formal"
    formal_root.mkdir()
    _write_completed_seed(formal_root, 0)
    root_file = tmp_path / "PHASE3_FORMAL_ROOT.txt"
    root_file.write_text(str(formal_root), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/issue54/run_phase3_formal_queue.py"),
            "--formal-root-file",
            str(root_file),
            "--validate-only",
            "--seeds",
            "0",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "validated seeds=(0,)" in result.stdout


def test_seed_parsing_defaults_and_rejects_duplicates() -> None:
    """Defaults and comma-separated explicit seeds retain the formal protocol."""
    assert parse_seeds(None) == DEFAULT_RUN_SEEDS
    assert parse_seeds(["0"]) == (0,)
    assert parse_seeds(["1", "2", "3", "4"]) == (1, 2, 3, 4)
    assert parse_seeds(["1,2,3,4"]) == (1, 2, 3, 4)
    with pytest.raises(ValueError, match="unique"):
        parse_seeds(["0,0"])


def test_checkpoint_paths_are_run_relative_or_absolute(tmp_path: Path) -> None:
    run_root = tmp_path / "phase3_v10_mot_seed0"
    relative = Path("training/v10_mot/weights/best.pt")
    assert queue.resolve_checkpoint(run_root, relative) == run_root / relative
    assert report.resolve_checkpoint(run_root, relative) == run_root / relative
    absolute = (tmp_path / "external.pt").resolve()
    assert queue.resolve_checkpoint(run_root, absolute) == absolute
    assert report.resolve_checkpoint(run_root, absolute) == absolute


def test_mot_seed_runner_dry_run_is_inert_and_rejects_non_routing_model(tmp_path: Path) -> None:
    args = [
        "--seed",
        "0",
        "--model",
        "v10_mot",
        "--data",
        "cloud-data.yaml",
        "--dataset-name",
        "dataset",
        "--dataset-version",
        "v1",
        "--dataset-manifest",
        "cloud-dataset.json",
        "--routing-split",
        "val",
        "--output-root",
        str(tmp_path / "outputs"),
        "--image-manifest",
        "cloud-images.json",
        "--data-root",
        "cloud-data-root",
        "--dry-run",
    ]
    assert seed_runner.main(args) == 0
    assert not (tmp_path / "outputs").exists()
    args[args.index("v10_mot")] = "v10"
    with pytest.raises(SystemExit):
        seed_runner.main(args)


def test_validate_seed_rejects_missing_checkpoint_and_sha_mismatch(tmp_path: Path) -> None:
    _write_completed_seed(tmp_path, 0)
    checkpoint = tmp_path / "phase3_v10_mot_seed0/training/v10_mot/weights/best.pt"
    checkpoint.unlink()
    _, error = queue.validate_seed(tmp_path, 0)
    assert error is not None and "missing checkpoint" in error

    checkpoint.write_bytes(b"changed")
    _, error = queue.validate_seed(tmp_path, 0)
    assert error is not None and "SHA256" in error


def test_formal_queue_skips_passed_seed_without_automatic_finalization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    formal_root = tmp_path / "formal"
    formal_root.mkdir()
    _write_completed_seed(formal_root, 0)
    _write_completed_seed(formal_root, 1)
    root_file = tmp_path / "root.txt"
    root_file.write_text(str(formal_root), encoding="utf-8")
    monkeypatch.setattr(queue, "run_seed", lambda *_args, **_kwargs: pytest.fail("passed seed must not relaunch"))
    monkeypatch.setattr(queue, "finalize", lambda *_args: pytest.fail("finalization requires explicit authorization"))
    assert queue.main(_launch_args(root_file, "1")) == 0


def test_formal_queue_failure_stops_before_next_seed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    formal_root = tmp_path / "formal"
    formal_root.mkdir()
    _write_completed_seed(formal_root, 0)
    root_file = tmp_path / "root.txt"
    root_file.write_text(str(formal_root), encoding="utf-8")
    launched = []
    monkeypatch.setattr(queue, "preflight", lambda _root: None)
    monkeypatch.setattr(
        queue,
        "run_seed",
        lambda _root, seed, _poll, _command: launched.append(seed) or 9,
    )
    assert queue.main(_launch_args(root_file, "1", "2")) == 1
    assert launched == [1]


def test_report_rejects_checkpoint_reuse_as_independent_seeds() -> None:
    with pytest.raises(SchemaValidationError, match="not unique"):
        report.validate_unique_checkpoints(["a" * 64] * 5)


def test_report_validates_real_checkpoint_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifests = []
    for seed in range(5):
        run = tmp_path / f"phase3_v10_mot_seed{seed}"
        checkpoint = run / "training/v10_mot/weights/best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        results = run / "training/v10_mot/results.csv"
        results.write_text(
            "epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n29,0.5,0.3\n",
            encoding="utf-8",
        )
        manifest = {
            "experiment_id": f"phase3_v10_mot_seed{seed}",
            "model_variant": "v10_mot",
            "seed": seed,
            "status": "passed",
            "checkpoint_path": "training/v10_mot/weights/best.pt",
            "checkpoint_sha256": _sha256(checkpoint),
            "manifest_sha256": f"{seed + 1:064x}",
        }
        manifests.append(manifest)
        (run / "experiment_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    registry = {"registry_sha256": "registry", "experiments": manifests}
    (tmp_path / "phase3_formal_registry.json").write_text(json.dumps(registry), encoding="utf-8")
    (tmp_path / "phase3_cross_seed_routing.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(report, "validate_registry", lambda payload: payload)
    monkeypatch.setattr(report, "validate_analysis", lambda payload, _sha: payload)
    monkeypatch.setattr(report, "validate_experiment_manifest", lambda payload: payload)

    _, _, loaded, _ = report.validate_inputs(tmp_path)
    assert len(loaded) == 5

    checkpoint = tmp_path / "phase3_v10_mot_seed4/training/v10_mot/weights/best.pt"
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match="missing checkpoint"):
        report.validate_inputs(tmp_path)
