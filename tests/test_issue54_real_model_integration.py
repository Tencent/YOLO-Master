"""Real YOLO-Master MoT YAML integration with random initialization and synthetic input."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.issue54.export_mot_routing import _module_state_sha256, _tensor_sha256, capture_mot_routing
from scripts.issue54.schema import (
    EXPERIMENT_MANIFEST_SCHEMA_VERSION,
    canonical_payload_sha256,
    sha256_file,
    validate_experiment_manifest,
    with_manifest_checksum,
)
from ultralytics.nn.modules.mot import MoTBlock
from ultralytics.nn.tasks import DetectionModel

ROOT = Path(__file__).resolve().parents[1]
MOT_YAML = ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-n.yaml"
PHASE0_COMMIT = "5c0db33af899b039f94bfdd6453857ff9795542c"


def _diagnostic_manifest(model: torch.nn.Module, image_sha256: str) -> dict:
    """Build explicit non-formal provenance for the random initialized diagnostic."""
    manifest = {
        "schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
        "experiment_id": "issue54-real-yaml-random-init-seed-54",
        "model_variant": "yolo-master-mot-n-random-init",
        "seed": 54,
        "dataset": "synthetic-tensor",
        "dataset_version": "phase1-real-yaml-v1",
        "dataset_manifest_sha256": canonical_payload_sha256({"image_sha256": image_sha256, "shape": [3, 64, 64]}),
        "split": "diagnostic",
        "requested_epochs": 1,
        "epochs": 0,
        "requested_batch": 1,
        "batch": 1,
        "effective_batch": 1,
        "imgsz": 64,
        "optimizer": "not_applicable",
        "precision_mode": "fp32_cpu_synthetic",
        "checkpoint_path": "diagnostic/random-initialized-state-no-checkpoint",
        "checkpoint_sha256": _module_state_sha256(model),
        "config_path": "ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-n.yaml",
        "config_sha256": sha256_file(MOT_YAML),
        "git_commit": PHASE0_COMMIT,
        "timestamp": "2026-07-30T00:00:00Z",
        "status": "diagnostic",
        "failure_reason": None,
    }
    return with_manifest_checksum(validate_experiment_manifest(manifest))


@pytest.mark.slow
def test_real_mot_yaml_random_init_synthetic_routing_integration():
    """Capture real MoT layers twice without hooks, gradients, or persistent model changes."""
    torch.manual_seed(54)
    model = DetectionModel(str(MOT_YAML), ch=3, nc=80, verbose=False).cpu()
    model.train()
    generator = torch.Generator().manual_seed(1054)
    batch = torch.rand(1, 3, 64, 64, generator=generator)
    manifest = _diagnostic_manifest(model, _tensor_sha256(batch[0]))
    entry = {
        "image_id": "synthetic-real-yaml-000",
        "image_path": "synthetic/real-yaml/image-000.tensor",
        "image_sha256": _tensor_sha256(batch[0]),
        "scene_groups": {"source": "synthetic"},
    }
    mot_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, MoTBlock)]
    assert mot_layers
    hook_counts_before = {name: len(module.router._forward_hooks) for name, module in mot_layers}
    training_before = {name: module.training for name, module in model.named_modules()}
    state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    with torch.no_grad():
        first = capture_mot_routing(
            model,
            batch,
            [entry],
            manifest,
            inference_repeat=0,
            timestamp=manifest["timestamp"],
        )
        second = capture_mot_routing(
            model,
            batch,
            [entry],
            manifest,
            inference_repeat=1,
            timestamp=manifest["timestamp"],
        )

    assert len(first) == len(mot_layers)
    assert len(second) == len(mot_layers)
    assert all(record["status"] == "diagnostic" for record in first + second)
    assert all(
        record["expert_names"]
        == [
            "LocalConvTransformer",
            "WindowTransformer",
            "DeformableTransformer",
        ]
        for record in first + second
    )
    assert all(torch.isfinite(torch.tensor(record["expert_probabilities"])).all() for record in first + second)
    assert all(min(record["expert_probabilities"]) >= 0.0 for record in first + second)
    assert all(sum(record["expert_probabilities"]) == pytest.approx(1.0, abs=1e-6) for record in first + second)

    for left, right in zip(first, second):
        assert left["layer_name"] == right["layer_name"]
        assert left["expert_names"] == right["expert_names"]
        assert left["expert_probabilities"] == pytest.approx(right["expert_probabilities"], abs=1e-7)
        assert left["token_top1_indices"] == right["token_top1_indices"]

    assert {name: module.training for name, module in model.named_modules()} == training_before
    assert {name: len(module.router._forward_hooks) for name, module in mot_layers} == hook_counts_before
    assert all(torch.equal(value, state_before[name]) for name, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
