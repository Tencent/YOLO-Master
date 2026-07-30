"""Experiment identity and evidence-counting tests for Issue #54."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.issue54.build_experiment_registry import build_registry, validate_registry
from scripts.issue54.schema import SchemaValidationError

ROOT = Path(__file__).resolve().parents[1]


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _manifest(seed: int, *, experiment_id: str | None = None, status: str = "passed", **updates):
    evidence = status in {"passed", "diagnostic"}
    payload = {
        "schema_version": 1,
        "experiment_id": experiment_id or f"mot-seed-{seed}",
        "model_variant": "mot",
        "seed": seed,
        "dataset": "visdrone",
        "dataset_version": "2019-det-v1",
        "dataset_manifest_sha256": _hash("dataset"),
        "split": "val",
        "requested_epochs": 100,
        "epochs": 100 if evidence else None,
        "requested_batch": 16,
        "batch": 8 if evidence else None,
        "effective_batch": 16 if evidence else None,
        "imgsz": 640,
        "optimizer": "AdamW",
        "precision_mode": "amp",
        "checkpoint_path": f"runs/issue54/mot/seed-{seed}/weights/best.pt" if evidence else None,
        "checkpoint_sha256": _hash(f"checkpoint-{seed}") if evidence else None,
        "config_path": "ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-n.yaml",
        "config_sha256": _hash("config"),
        "git_commit": "a" * 40,
        "timestamp": "2026-07-30T00:00:00Z",
        "status": status,
        "failure_reason": "synthetic failure" if status == "failed" else None,
    }
    if status == "not_executed":
        payload["config_path"] = None
        payload["config_sha256"] = None
    payload.update(updates)
    return payload


def test_registry_rejects_duplicate_experiment_id():
    with pytest.raises(SchemaValidationError, match="duplicate experiment_id"):
        build_registry([_manifest(17), _manifest(73, experiment_id="mot-seed-17")])


def test_registry_rejects_same_seed_registered_as_another_run():
    with pytest.raises(SchemaValidationError, match="registered more than once"):
        build_registry([_manifest(17), _manifest(17, experiment_id="mot-seed-17-retry")])


def test_registry_rejects_same_checkpoint_hash_as_independent_seed():
    shared = _hash("same-checkpoint")
    with pytest.raises(SchemaValidationError, match="checkpoint hash reuse"):
        build_registry(
            [
                _manifest(17, checkpoint_sha256=shared),
                _manifest(73, checkpoint_sha256=shared),
            ]
        )


def test_diagnostic_failed_and_not_executed_do_not_inflate_formal_seed_count():
    registry = build_registry(
        [
            _manifest(17),
            _manifest(42, status="diagnostic"),
            _manifest(73, status="failed"),
            _manifest(101, status="not_executed"),
        ]
    )

    summary = registry["variant_summary"]["mot"]
    assert summary["registered_experiments"] == 4
    assert summary["formal_seed_count"] == 1
    assert summary["formal_seeds"] == [17]
    assert summary["inference_level"] == "insufficient_for_cross_seed_inference"


@pytest.mark.parametrize(
    ("seed_count", "expected"),
    [
        (1, "insufficient_for_cross_seed_inference"),
        (2, "insufficient_for_cross_seed_inference"),
        (3, "exploratory_only"),
        (5, "minimum_for_stronger_cross_seed_claims"),
    ],
)
def test_registry_safely_labels_small_seed_counts(seed_count, expected):
    registry = build_registry([_manifest(seed) for seed in range(seed_count)])

    assert registry["variant_summary"]["mot"]["formal_seed_count"] == seed_count
    assert registry["variant_summary"]["mot"]["inference_level"] == expected


def test_registry_output_is_deterministic_and_self_validating():
    forward = build_registry([_manifest(17), _manifest(42), _manifest(73)])
    reverse = build_registry([_manifest(73), _manifest(42), _manifest(17)])

    assert forward == reverse
    assert validate_registry(forward) == forward


def test_synthetic_cli_end_to_end_is_diagnostic_unicode_safe_and_overwrite_guarded(tmp_path):
    output_dir = tmp_path / "中文输出"
    manifests = []
    routes = []
    for seed in (17, 42, 73):
        manifest = output_dir / f"manifest-{seed}.json"
        route = output_dir / f"routes-{seed}.jsonl"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/issue54/export_mot_routing.py",
                "--synthetic",
                "--seed",
                str(seed),
                "--repeats",
                "2",
                "--manifest-output",
                str(manifest),
                "--output",
                str(route),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        manifests.append(manifest)
        routes.append(route)

    registry = output_dir / "registry.json"
    registry_command = [sys.executable, "scripts/issue54/build_experiment_registry.py"]
    for manifest in manifests:
        registry_command.extend(["--manifest", str(manifest)])
    registry_command.extend(["--output", str(registry)])
    result = subprocess.run(registry_command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

    analysis = output_dir / "analysis.json"
    analysis_command = [
        sys.executable,
        "scripts/issue54/analyze_cross_seed_routing.py",
        "--registry",
        str(registry),
    ]
    for route in routes:
        analysis_command.extend(["--routes", str(route)])
    analysis_command.extend(["--output", str(analysis)])
    result = subprocess.run(analysis_command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    first_bytes = analysis.read_bytes()
    payload = json.loads(first_bytes)
    assert payload["analysis_mode"] == "diagnostic_not_formal_evidence"
    assert payload["record_count"] > 0
    assert str(tmp_path) not in analysis.read_text(encoding="utf-8")

    blocked = subprocess.run(analysis_command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert blocked.returncode != 0
    assert "--overwrite" in blocked.stderr

    replaced = subprocess.run(
        [*analysis_command, "--overwrite"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert replaced.returncode == 0, replaced.stderr
    assert analysis.read_bytes() == first_bytes


def test_registry_cli_missing_input_returns_nonzero(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/issue54/build_experiment_registry.py",
            "--manifest",
            str(tmp_path / "missing.json"),
            "--output",
            str(tmp_path / "registry.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert str(tmp_path) not in result.stderr
    assert str(ROOT) not in result.stderr


def test_analysis_cli_empty_routes_returns_nonzero_without_output(tmp_path):
    registry = tmp_path / "registry.json"
    routes = tmp_path / "empty.jsonl"
    output = tmp_path / "analysis.json"
    registry.write_text(json.dumps(build_registry([_manifest(17)])), encoding="utf-8")
    routes.write_text("", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/issue54/analyze_cross_seed_routing.py",
            "--registry",
            str(registry),
            "--routes",
            str(routes),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "at least one routing record" in result.stderr
    assert str(tmp_path) not in result.stderr
    assert str(ROOT) not in result.stderr
    assert not output.exists()
