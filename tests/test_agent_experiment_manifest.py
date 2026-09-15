"""Contracts for profile-driven Agent requests and reproducible experiment manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.runtime.cli.contract import (
    MANIFEST_LOG_LIMIT,
    MANIFEST_SCHEMA_VERSION,
    manifest_checksum,
    write_manifest,
)
from agent.runtime.cli.normalize import normalize_request
from agent.runtime.cli.validate import run_dispatcher_case
from ultralytics.cfg.mixture_catalog import DEFAULT_MIXTURE_MODEL_ROOT


def test_profile_input_resolves_to_model_with_provenance() -> None:
    """Resolve a stable profile identifier before emitting downstream YOLO inputs."""
    request = normalize_request(
        {
            "skill": "yolo.pipeline.experiment",
            "inputs": {"profile": "26/yolo26-master-mot-n", "data": "coco8.yaml"},
            "params": {"train": {"epochs": 1}},
        }
    )

    model_path = DEFAULT_MIXTURE_MODEL_ROOT / "26/yolo26-master-mot-n.yaml"
    assert "profile" not in request["inputs"]
    assert request["inputs"]["model"] == str(model_path.resolve())
    assert request["inputs"]["task"] == "detect"
    assert request["profile"]["profile_id"] == "26/yolo26-master-mot-n"
    assert request["profile"]["mixture_kinds"] == ["mot"]
    assert request["profile"]["model_path"] == str(model_path.resolve())
    assert request["profile"]["sha256"] == hashlib.sha256(model_path.read_bytes()).hexdigest()


def test_profile_provenance_is_revalidated_on_repeated_normalization() -> None:
    """Rebuild carried provenance so an async child request cannot trust forged metadata."""
    request = normalize_request(
        {"skill": "yolo.train", "inputs": {"profile": "26/yolo26-master-mot-n", "data": "coco8.yaml"}}
    )
    request["profile"]["sha256"] = "forged"
    request["profile"]["mixture_kinds"] = ["moe"]

    normalized = normalize_request(request)

    model_path = DEFAULT_MIXTURE_MODEL_ROOT / "26/yolo26-master-mot-n.yaml"
    assert normalized["profile"]["sha256"] == hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert normalized["profile"]["mixture_kinds"] == ["mot"]


@pytest.mark.parametrize(
    "inputs, match",
    [
        (
            {"profile": "26/yolo26-master-mot-n", "model": "yolo26n.pt"},
            "mutually exclusive",
        ),
        ({"profile": "missing/profile"}, "unknown mixture profile"),
        (
            {"profile": "26/yolo26-master-mot-n", "task": "segment"},
            "task 'segment' conflicts",
        ),
    ],
)
def test_profile_input_rejects_ambiguous_or_invalid_requests(inputs: dict, match: str) -> None:
    """Fail request normalization before execution when profile selection is unsafe."""
    with pytest.raises(ValueError, match=match):
        normalize_request({"skill": "yolo.train", "inputs": inputs})


def test_manifest_preserves_pipeline_evidence_and_redacts_secrets(tmp_path: Path) -> None:
    """Write a versioned manifest without losing stages or persisting credentials."""
    secret_request = "sk-request-value"
    secret_stage = "sk-stage-value"
    long_stdout = "prefix\n" + ("x" * (MANIFEST_LOG_LIMIT + 500)) + f"\nopenai_api_key={secret_stage}"
    request = {
        "skill": "yolo.pipeline.experiment",
        "request_id": "experiment-1",
        "inputs": {"model": "/models/mot.yaml", "data": "/datasets/coco8.yaml"},
        "params": {
            "train": {"epochs": 1},
            "openai_api_key": secret_request,
            "max_output_tokens": 128,
        },
        "runtime": {"headers": {"Authorization": "Bearer hidden-token"}},
        "artifacts": {"project": str(tmp_path), "name": "run-a"},
        "policy": {"dry_run": False},
        "profile": {
            "profile_id": "26/yolo26-master-mot-n",
            "sha256": "a" * 64,
        },
    }
    payload = {
        "skill": request["skill"],
        "status": "ok",
        "summary": "pipeline finished",
        "metrics": {"map50": 0.5},
        "environment": {"python": "3.11"},
        "artifacts": [{"kind": "checkpoint", "path": "/runs/best.pt"}],
        "pipeline": {"stage_order": ["train", "val"], "failed_stage": None},
        "stages": {
            "train": {
                "skill": "yolo.train",
                "status": "ok",
                "logs": {
                    "cmd": ["yolo", f"openai_api_key={secret_stage}"],
                    "stdout": long_stdout,
                    "stderr": "Authorization: Bearer another-hidden-token",
                },
            },
            "val": {"skill": "yolo.val", "status": "ok", "metrics": {"map50": 0.5}},
        },
        "best_checkpoint": "/runs/best.pt",
        "usage": {"tokens": {"input": 0, "output": 0, "total": 0}, "requests": 0, "records": []},
        "cost_estimate": {"currency": "USD", "amount": 0.0, "basis": "no token usage"},
    }

    manifest_path = write_manifest(request, payload)
    raw = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(raw)

    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["skill"] == "yolo.pipeline.experiment"
    assert manifest["metrics"] == {"map50": 0.5}
    assert manifest["request"]["profile"]["profile_id"] == "26/yolo26-master-mot-n"
    assert manifest["request"]["params"]["max_output_tokens"] == 128
    assert manifest["result"]["pipeline"]["stage_order"] == ["train", "val"]
    assert manifest["result"]["stages"]["val"]["metrics"]["map50"] == 0.5
    assert manifest["result"]["best_checkpoint"] == "/runs/best.pt"
    assert manifest_checksum(manifest) == manifest["manifest_sha256"]
    assert secret_request not in raw
    assert secret_stage not in raw
    assert "hidden-token" not in raw
    assert raw.count("<redacted>") >= 4
    compact_stdout = manifest["result"]["stages"]["train"]["logs"]["stdout"]
    assert len(compact_stdout) <= MANIFEST_LOG_LIMIT + 100
    assert compact_stdout.startswith("[truncated ")
    assert not list(manifest_path.parent.glob(".skill_manifest.*.tmp"))

    manifest["status"] = "failed"
    assert manifest_checksum(manifest) != manifest["manifest_sha256"]


def test_agent_validator_decodes_dispatcher_output_as_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep non-ASCII dispatcher JSON independent of the Windows locale."""
    payload = {"skill": "yolo.system", "status": "ok", "summary": "检测完成"}

    def fake_run(*_args, **kwargs):
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["errors"] == "replace"
        assert kwargs["env"]["PYTHONIOENCODING"] == "utf-8"
        return SimpleNamespace(stdout=json.dumps(payload, ensure_ascii=False), stderr="", returncode=0)

    monkeypatch.setattr("agent.runtime.cli.validate.subprocess.run", fake_run)
    result = run_dispatcher_case(
        {"name": "utf8-output", "request": {"skill": "yolo.system"}, "expect": {"status": "ok"}}
    )

    assert result["passed"] is True
    assert result["payload"]["summary"] == "检测完成"
