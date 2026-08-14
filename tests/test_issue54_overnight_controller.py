"""Regression tests for the Issue #54 overnight controller and recovery path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import pytest
import yaml

from scripts.issue54 import run_phase2_2_overnight as overnight


def _write_completed_run(
    root: Path,
    *,
    epochs_completed: int = 3,
    checkpoint_name: str = "best.pt",
    amp: bool = True,
) -> Path:
    run_dir = root / "training" / "calibration_mot_amp"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    (weights / checkpoint_name).write_bytes(b"checkpoint")
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "time",
                "train/box_loss",
                "train/cls_loss",
                "train/dfl_loss",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": epochs_completed,
                "time": 900,
                "train/box_loss": 1.0,
                "train/cls_loss": 2.0,
                "train/dfl_loss": 3.0,
                "metrics/mAP50(B)": 0.4,
                "metrics/mAP50-95(B)": 0.2,
            }
        )
    (run_dir / "args.yaml").write_text(
        yaml.safe_dump(
            {
                "epochs": 3,
                "batch": 8,
                "imgsz": 640,
                "cache": False,
                "amp": amp,
                "optimizer": "auto",
                "lr0": 0.01,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_merge_training_payload_combines_parser_and_metadata() -> None:
    payload = {"pid": 123}
    parsed = {"epoch": 3.0, "map50": 0.4}
    metadata = {"status": "passed_pilot", "epochs_completed": 3}

    result = overnight.merge_training_payload(payload, parsed, metadata)

    assert result is payload
    assert result == {
        "pid": 123,
        "epoch": 3.0,
        "map50": 0.4,
        "status": "passed_pilot",
        "epochs_completed": 3,
    }


def test_recover_completed_training_reports_three_epochs_and_checkpoint(tmp_path: Path) -> None:
    run_dir = _write_completed_run(tmp_path)
    expected_sha = hashlib.sha256(b"checkpoint").hexdigest()

    result = overnight.recover_training_payload(
        run_dir,
        expected_epochs=3,
        expected_amp=True,
        expected_checkpoint_sha256=expected_sha,
        checkpoint_loader=lambda _: object(),
    )

    assert result["status"] == "passed_pilot"
    assert result["epochs_completed"] == 3
    assert result["training_time_seconds"] == 900
    assert result["loss"] == 6.0
    assert result["checkpoint_loadable"] is True
    assert result["checkpoint"]["kind"] == "best"
    assert result["checkpoint"]["sha256"] == expected_sha
    assert Path(result["checkpoint"]["path"]) == run_dir / "weights/best.pt"


@pytest.mark.parametrize(
    ("mutator", "expected_status"),
    [
        (lambda run_dir: (run_dir / "results.csv").unlink(), "interrupted_with_checkpoint"),
        (lambda run_dir: (run_dir / "weights/best.pt").unlink(), "failed"),
    ],
)
def test_recover_failed_training_is_not_passed(
    tmp_path: Path,
    mutator,
    expected_status: str,
) -> None:
    run_dir = _write_completed_run(tmp_path)
    mutator(run_dir)

    result = overnight.recover_training_payload(
        run_dir,
        expected_epochs=3,
        expected_amp=True,
        checkpoint_loader=lambda _: object(),
    )

    assert result["status"] == expected_status
    assert result["checkpoint_loadable"] is False
    assert result["failure_reason"]


def test_recover_partial_checkpoint_is_interrupted_not_passed(tmp_path: Path) -> None:
    run_dir = _write_completed_run(
        tmp_path,
        epochs_completed=2,
        checkpoint_name="epoch1.pt",
    )

    result = overnight.recover_training_payload(
        run_dir,
        expected_epochs=3,
        expected_amp=True,
        checkpoint_loader=lambda _: object(),
    )

    assert result["status"] == "interrupted_with_checkpoint"
    assert result["epochs_completed"] == 2
    assert result["checkpoint"]["kind"] == "latest_epoch"
    assert result["checkpoint_loadable"] is False


def test_recover_calibration_does_not_duplicate_existing_run() -> None:
    existing = {"experiment_id": "calibration_mot_amp", "status": "passed_pilot"}
    controller = overnight.OvernightController.__new__(overnight.OvernightController)
    controller.runs = [existing]

    result = controller.recover_calibration(
        experiment_id="calibration_mot_amp",
        precision="amp",
        config=Path("unused.yaml"),
    )

    assert result is existing
    assert controller.runs == [existing]


def test_recovered_routing_uses_output_checkpoint_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_export(run, **_):
        captured.update(run)
        return {
            "experiment_id": run["experiment_id"],
            "deterministic_repeats": True,
            "hooks_cleaned": True,
            "probabilities_valid": True,
        }

    monkeypatch.setattr(overnight, "export_routing", fake_export)
    controller = overnight.OvernightController.__new__(overnight.OvernightController)
    controller.output = tmp_path
    controller.data_root = tmp_path / "data"
    controller.dataset = {}
    controller.git_commit = "commit"
    controller.routing = []
    controller.current_task = ""
    controller.write_state = lambda *_: None
    run = {
        "experiment_id": "calibration_mot_amp",
        "seed": 0,
        "precision": "amp",
        "epochs_requested": 3,
        "requested_batch": 8,
        "actual_batch": 8,
        "optimizer": "auto",
        "config_path": "model.yaml",
        "config_sha256": "a" * 64,
        "checkpoint": {
            "path": "/old/results/weights/best.pt",
            "index_path": (tmp_path / "checkpoints/calibration_mot_amp_best.pt").as_posix(),
            "sha256": "b" * 64,
        },
    }

    result = controller.route_checkpoint(run)

    assert result["status"] == "passed_pilot"
    assert captured["checkpoint"]["path"] == run["checkpoint"]["index_path"]
    assert run["checkpoint"]["path"] == "/old/results/weights/best.pt"


def test_rendered_manifest_csv_and_markdown_are_consistent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    (repository / "reports/issue54").mkdir(parents=True)
    output = tmp_path / "output"
    (output / "reports").mkdir(parents=True)
    monkeypatch.setattr(overnight, "ROOT", repository)
    run = {
        "experiment_id": "calibration_mot_amp",
        "phase": "calibration",
        "model_variant": "mot",
        "seed": 0,
        "precision": "amp",
        "epochs_requested": 3,
        "epochs_completed": 3,
        "requested_batch": 8,
        "actual_batch": 8,
        "seconds_per_epoch": 300.0,
        "total_duration_seconds": 0.0,
        "peak_gpu_memory_bytes": 1024,
        "loss": 1.0,
        "map50": 0.1,
        "map50_95": 0.05,
        "nan_or_inf": False,
        "exit_code": 0,
        "status": "passed_pilot",
        "failure_reason": None,
        "checkpoint": {
            "path": "weights/best.pt",
            "size_bytes": 10,
            "sha256": "a" * 64,
        },
    }
    controller = overnight.OvernightController.__new__(overnight.OvernightController)
    controller.output = output
    controller.resume_from = tmp_path / "source"
    controller.report_names = overnight.PHASE2_3_REPORT_NAMES
    controller.runs = [run]
    controller.routing = []
    controller.route_comparison = {"available": False}
    controller.precision_selection = {"selected": "fp32", "reason": "test"}
    controller.admission_decisions = []
    controller.controller_failure = None
    controller.environment = {"gpu": {}, "torch": "test", "torch_cuda": "test"}
    controller.dataset = {"inventory_sha256": "dataset"}
    controller.git_commit = "commit"
    controller.started_at = "2026-01-01T00:00:00Z"
    controller.expected_end_at = "2026-01-01T01:00:00Z"
    controller.started_monotonic = time.monotonic()
    controller.args = argparse.Namespace(
        budget_seconds=3600,
        no_new_training_seconds=3000,
    )

    controller.render_reports("complete", {"available": False})

    manifest_path = output / "reports" / overnight.PHASE2_3_REPORT_NAMES["manifest"]
    csv_path = output / "reports" / overnight.PHASE2_3_REPORT_NAMES["csv"]
    markdown_path = output / "reports" / overnight.PHASE2_3_REPORT_NAMES["markdown"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert manifest["runs"][0]["status"] == "passed_pilot"
    assert csv_rows[0]["status"] == "passed_pilot"
    assert "3/3" in markdown
    assert "passed_pilot" in markdown
    for filename in overnight.PHASE2_3_REPORT_NAMES.values():
        assert (repository / "reports/issue54" / filename).is_file()
