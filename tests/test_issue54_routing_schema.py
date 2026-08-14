"""Schema and export contracts for the Issue #54 routing stability study."""

from __future__ import annotations

import hashlib

import pytest
import torch

from scripts.issue54.export_mot_routing import (
    _SyntheticMoTModel,
    _dense_probabilities_from_router_output,
    capture_mot_routing,
    synthetic_evidence,
)
from scripts.issue54.schema import (
    EXPERIMENT_MANIFEST_JSON_SCHEMA,
    ROUTING_RECORD_JSON_SCHEMA,
    SchemaValidationError,
    validate_experiment_manifest,
    validate_routing_record,
    write_json,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _manifest(**updates):
    payload = {
        "schema_version": 1,
        "experiment_id": "mot-seed-17",
        "model_variant": "mot",
        "seed": 17,
        "dataset": "visdrone",
        "dataset_version": "2019-det-v1",
        "dataset_manifest_sha256": _hash("dataset"),
        "split": "val",
        "requested_epochs": 100,
        "epochs": 98,
        "requested_batch": 16,
        "batch": 8,
        "effective_batch": 16,
        "imgsz": 640,
        "optimizer": "AdamW",
        "precision_mode": "amp",
        "checkpoint_path": "runs/issue54/mot/seed-17/weights/best.pt",
        "checkpoint_sha256": _hash("checkpoint-17"),
        "config_path": "ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-n.yaml",
        "config_sha256": _hash("config"),
        "git_commit": "a" * 40,
        "timestamp": "2026-07-30T00:00:00Z",
        "status": "passed",
        "failure_reason": None,
    }
    payload.update(updates)
    return payload


def _record(**updates):
    payload = {
        "schema_version": 1,
        "experiment_id": "mot-seed-17",
        "model_variant": "mot",
        "seed": 17,
        "dataset": "visdrone",
        "dataset_version": "2019-det-v1",
        "split": "val",
        "checkpoint_sha256": _hash("checkpoint-17"),
        "image_id": "image-001",
        "image_path": "val/images/image-001.jpg",
        "image_sha256": _hash("image-001"),
        "scene_groups": {"density": "dense"},
        "layer_name": "model.14.m.0",
        "layer_index": 0,
        "expert_names": ["Local", "Window", "Deformable"],
        "expert_probabilities": [0.5, 0.3, 0.2],
        "selected_expert": "Local",
        "top_k": 2,
        "token_top1_indices": [0, 0, 1, 2],
        "spatial_shape": [2, 2],
        "inference_repeat": 0,
        "inference_batch_actual": 1,
        "timestamp": "2026-07-30T00:00:00Z",
        "status": "passed",
        "failure_reason": None,
    }
    payload.update(updates)
    return payload


def test_manifest_keeps_requested_actual_and_effective_batch_separate():
    manifest = validate_experiment_manifest(_manifest())

    assert manifest["requested_batch"] == 16
    assert manifest["batch"] == 8
    assert manifest["effective_batch"] == 16
    assert manifest["requested_epochs"] == 100
    assert manifest["epochs"] == 98


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint_path", "C:/private/best.pt"),
        ("checkpoint_path", "../best.pt"),
        ("config_path", "/private/model.yaml"),
    ],
)
def test_manifest_rejects_absolute_or_traversal_paths(field, value):
    with pytest.raises(SchemaValidationError, match=field):
        validate_experiment_manifest(_manifest(**{field: value}))


def test_failed_and_not_executed_statuses_cannot_claim_passed_evidence():
    with pytest.raises(SchemaValidationError, match="failure_reason"):
        validate_experiment_manifest(_manifest(status="failed", failure_reason=None))
    with pytest.raises(SchemaValidationError, match="not_executed"):
        validate_experiment_manifest(
            _manifest(
                status="not_executed",
                requested_epochs=100,
                epochs=None,
                batch=None,
                checkpoint_path="planned/best.pt",
                checkpoint_sha256=None,
                failure_reason=None,
            )
        )


@pytest.mark.parametrize(
    "probabilities",
    [
        [float("nan"), 0.5, 0.5],
        [float("inf"), 0.0, 0.0],
        [0.7, 0.2, 0.2],
        [-0.1, 0.5, 0.6],
    ],
)
def test_routing_record_rejects_nonfinite_or_invalid_probabilities(probabilities):
    with pytest.raises(SchemaValidationError, match="probabilities"):
        validate_routing_record(_record(expert_probabilities=probabilities))


def test_routing_record_rejects_schema_shape_and_selected_expert_errors():
    with pytest.raises(SchemaValidationError, match="length"):
        validate_routing_record(_record(expert_probabilities=[0.5, 0.5]))
    with pytest.raises(SchemaValidationError, match="argmax"):
        validate_routing_record(_record(selected_expert="Window"))
    with pytest.raises(SchemaValidationError, match=r"product\(spatial_shape\)"):
        validate_routing_record(_record(token_top1_indices=[0, 1]))


def test_json_schema_documents_all_required_fields():
    assert set(EXPERIMENT_MANIFEST_JSON_SCHEMA["required"]) == set(_manifest())
    assert set(ROUTING_RECORD_JSON_SCHEMA["required"]) == set(_record())


def test_synthetic_export_uses_full_probabilities_and_repeat_identity():
    manifest, records = synthetic_evidence(seed=17, repeats=2)

    assert manifest["status"] == "diagnostic"
    assert manifest["epochs"] == 0
    assert len(records) == 4
    assert {record["inference_repeat"] for record in records} == {0, 1}
    assert all(sum(record["expert_probabilities"]) == pytest.approx(1.0) for record in records)
    assert all(len(record["token_top1_indices"]) == 16 for record in records)
    assert records[0]["checkpoint_sha256"] == records[-1]["checkpoint_sha256"]


def test_synthetic_checkpoints_share_fixed_validation_inputs_across_model_seeds():
    manifest_a, records_a = synthetic_evidence(seed=17, repeats=1)
    manifest_b, records_b = synthetic_evidence(seed=73, repeats=1)

    hashes_a = {record["image_id"]: record["image_sha256"] for record in records_a}
    hashes_b = {record["image_id"]: record["image_sha256"] for record in records_b}
    assert hashes_a == hashes_b
    assert manifest_a["dataset_manifest_sha256"] == manifest_b["dataset_manifest_sha256"]
    assert manifest_a["checkpoint_sha256"] != manifest_b["checkpoint_sha256"]


def test_writer_requires_explicit_overwrite(tmp_path):
    output = tmp_path / "中文目录" / "result.json"
    write_json(output, {"value": 1})

    with pytest.raises(SchemaValidationError, match="--overwrite"):
        write_json(output, {"value": 2})

    write_json(output, {"value": 2}, overwrite=True)
    assert '"value": 2' in output.read_text(encoding="utf-8")


def test_router_contract_rejects_probabilities_in_logits_position():
    model = _SyntheticMoTModel().eval()
    probabilities = torch.full((1, 3, 2, 2), 1 / 3)
    indices = torch.tensor([[[[0, 0], [0, 0]], [[1, 1], [1, 1]]]])
    sparse = torch.zeros_like(probabilities)
    sparse[:, :2] = 0.5

    with pytest.raises(SchemaValidationError, match="normalized probabilities"):
        _dense_probabilities_from_router_output(
            model.mot.router,
            (sparse, indices, probabilities),
            layer_name="mot",
            expert_count=3,
            top_k=2,
        )


def test_capture_restores_state_clears_hooks_and_keeps_parameters_and_gradients():
    manifest, seed_records = synthetic_evidence(seed=17, repeats=1)
    model = _SyntheticMoTModel().train()
    batch = torch.rand(1, 12, 4, 4)
    entry = {
        "image_id": "state-check",
        "image_path": "synthetic/state-check.tensor",
        "image_sha256": seed_records[0]["image_sha256"],
        "scene_groups": {},
    }
    flags_before = {name: module.training for name, module in model.named_modules()}
    parameters_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    hooks_before = len(model.mot.router._forward_hooks)

    records = capture_mot_routing(
        model,
        batch,
        [entry],
        manifest,
        inference_repeat=0,
        timestamp=manifest["timestamp"],
    )

    assert records
    assert {name: module.training for name, module in model.named_modules()} == flags_before
    assert len(model.mot.router._forward_hooks) == hooks_before
    assert all(torch.equal(parameter, parameters_before[name]) for name, parameter in model.named_parameters())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_capture_removes_hooks_and_restores_state_when_forward_raises(monkeypatch):
    manifest, seed_records = synthetic_evidence(seed=17, repeats=1)
    model = _SyntheticMoTModel().train()
    batch = torch.rand(1, 12, 4, 4)
    entry = {
        "image_id": "failure-check",
        "image_path": "synthetic/failure-check.tensor",
        "image_sha256": seed_records[0]["image_sha256"],
        "scene_groups": {},
    }
    flags_before = {name: module.training for name, module in model.named_modules()}
    hooks_before = len(model.mot.router._forward_hooks)
    original = model.forward

    def failing_forward(value):
        original(value)
        raise RuntimeError("synthetic forward failure")

    monkeypatch.setattr(model, "forward", failing_forward)
    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        capture_mot_routing(
            model,
            batch,
            [entry],
            manifest,
            inference_repeat=0,
            timestamp=manifest["timestamp"],
        )

    assert len(model.mot.router._forward_hooks) == hooks_before
    assert {name: module.training for name, module in model.named_modules()} == flags_before


def test_capture_removes_partial_hooks_when_later_registration_fails(monkeypatch):
    class TwoLayerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = _SyntheticMoTModel().mot
            self.second = _SyntheticMoTModel().mot

        def forward(self, value):
            value, _ = self.first(value)
            value, _ = self.second(value)
            return value

    manifest, seed_records = synthetic_evidence(seed=17, repeats=1)
    model = TwoLayerModel().train()
    entry = {
        "image_id": "registration-check",
        "image_path": "synthetic/registration-check.tensor",
        "image_sha256": seed_records[0]["image_sha256"],
        "scene_groups": {},
    }
    first_hooks = len(model.first.router._forward_hooks)

    def fail_registration(_hook):
        raise RuntimeError("registration failure")

    monkeypatch.setattr(model.second.router, "register_forward_hook", fail_registration)
    with pytest.raises(RuntimeError, match="registration failure"):
        capture_mot_routing(
            model,
            torch.rand(1, 12, 4, 4),
            [entry],
            manifest,
            inference_repeat=0,
            timestamp=manifest["timestamp"],
        )

    assert len(model.first.router._forward_hooks) == first_hooks
