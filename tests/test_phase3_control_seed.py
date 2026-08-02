"""Regression tests for the Phase 3 control seed runner."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.issue54 import run_phase3_control_seed as runner
from scripts.issue54.run_phase3_control_seed import (
    build_training_command,
    experiment_root,
    parse_args,
    read_final_metrics,
)


def _args(model: str, seed: int = 0) -> argparse.Namespace:
    return argparse.Namespace(model=model, seed=seed, data=Path("data.yaml"), epochs=30, device="0", batch=8, imgsz=640)


@pytest.mark.parametrize("model", ["v10", "v10_moa"])
def test_amp_command_and_isolated_directory(model: str, tmp_path: Path) -> None:
    args = _args(model, seed=2)
    command = build_training_command(args, tmp_path / "training")
    assert "--amp" in command
    assert "--no-amp" not in command
    assert command[command.index("--models") + 1] == model
    assert experiment_root(tmp_path, model, 2) == tmp_path / f"phase3_{model}_seed2"


def test_invalid_model_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--seed",
                "0",
                "--model",
                "v10_mot",
                "--data",
                "data.yaml",
                "--output-root",
                "out",
                "--dataset-name",
                "d",
                "--dataset-version",
                "v",
                "--dataset-manifest",
                "m.json",
                "--validation-split",
                "val",
            ]
        )


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_non_finite_metric_rejected(tmp_path: Path, bad: str) -> None:
    path = tmp_path / "results.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "val/box_loss"])
        writer.writeheader()
        writer.writerow({"epoch": 29, "metrics/mAP50(B)": 0.5, "metrics/mAP50-95(B)": 0.3, "val/box_loss": bad})
    with pytest.raises(ValueError, match="non-finite"):
        read_final_metrics(path, 30)


def _main_args(tmp_path: Path) -> list[str]:
    data = tmp_path / "data.yaml"
    data.write_text("path: .\n", encoding="utf-8")
    dataset_manifest = tmp_path / "dataset.json"
    dataset_manifest.write_text("{}\n", encoding="utf-8")
    return [
        "--seed",
        "0",
        "--model",
        "v10",
        "--data",
        str(data),
        "--output-root",
        str(tmp_path / "outputs"),
        "--dataset-name",
        "dataset",
        "--dataset-version",
        "v1",
        "--dataset-manifest",
        str(dataset_manifest),
        "--validation-split",
        "val",
    ]


def test_nonzero_training_exit_writes_failed_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    original_run = runner.subprocess.run

    def fake_run(command, *args, **kwargs):
        if str(command[1]).endswith("compare_mot_ablation.py"):
            return SimpleNamespace(returncode=7)
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.main(_main_args(tmp_path)) == 2
    run_root = tmp_path / "outputs/phase3_v10_seed0"
    payload = json.loads((run_root / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "exit code 7" in payload["failure_reason"]
    assert (run_root / "exitcode").read_text(encoding="utf-8").strip() == "7"


def test_existing_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    args = _main_args(tmp_path)
    (tmp_path / "outputs/phase3_v10_seed0").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.main(args)
