"""Regression coverage for MoT ablation artifacts and seed layout."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.compare_mot_ablation import (
    SPECS,
    build_model,
    telemetry_summary_fields,
    validate_training_specs,
    write_run_manifest,
)
from ultralytics.nn.modules.moe.config import apply_mixture_config, resolve_mixture_config
from ultralytics.nn.modules.mot import C2fMoT, MoTBlock
from ultralytics.nn.modules.multitask import MultiTaskHead


ROOT = Path(__file__).resolve().parents[1]


def _all_finite(value):
    """Return whether every tensor in a nested model output is finite."""
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_finite(item) for item in value)
    return True


def test_telemetry_summary_fields_preserves_memory_semantics():
    fields = telemetry_summary_fields(
        {
            "aggregation": {
                "world_size": 2,
                "rank_step_counts_consistent": True,
                "rank_loss_relative_spread_max": 0.04,
                "global_samples_per_second": 10.0,
            },
            "ranks": [
                {
                    "metadata": {"rank": 0},
                    "steps": {"samples_per_second": 5.0, "p50_milliseconds": 20.0, "p95_milliseconds": 25.0},
                    "memory": {
                        "measurement": "mps_sampled_current_allocated_memory",
                        "is_true_peak": False,
                        "peak_device_memory_bytes": None,
                        "device_total_memory_bytes": None,
                        "peak_device_memory_fraction": None,
                        "sampled_current_memory_bytes_max": 123,
                    },
                }
            ],
        }
    )

    assert fields["train_world_size"] == "2"
    assert fields["train_memory_is_true_peak"] == "False"
    assert fields["train_peak_device_memory_bytes"] == ""
    assert fields["train_peak_device_memory_fraction"] == ""
    assert fields["train_rank_peak_device_memory_fraction_max"] == ""
    assert fields["train_sampled_current_memory_bytes_max"] == "123"


def test_run_manifest_records_requested_configuration(tmp_path, monkeypatch):
    class EmptyModel:
        def named_modules(self):
            return []

    args = SimpleNamespace(
        data=tmp_path / "data.yaml",
        epochs=1,
        imgsz=64,
        batch=2,
        workers=0,
        optimizer="AdamW",
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        device="cpu",
        amp=False,
        deterministic=True,
        resume=False,
        telemetry=True,
        telemetry_loss_steps=20,
        temperature_factor=0.97,
        temperature_min=0.3,
    )
    monkeypatch.setattr("scripts.compare_mot_ablation.torch.cuda.is_available", lambda: False)
    model = SimpleNamespace(model=EmptyModel())

    manifest = write_run_manifest(args, SPECS["v10"], tmp_path, 42, model)
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["seed"] == 42
    assert payload["training"]["telemetry_enabled"] is True
    assert payload["training"]["moa_mot_temperature_factor"] == pytest.approx(0.97)


@pytest.mark.parametrize("key", ("mt_off", "mt_mot_dense", "mt_mot_sparse"))
def test_multitask_mot_ablation_specs_build_and_forward(key):
    """All three treatments retain the same three-task head contract."""
    spec = SPECS[key]
    model = build_model(spec)
    head = model.model[-1]

    assert spec.task == "multitask"
    assert isinstance(head, MultiTaskHead)
    assert set(head.active_tasks) == {"detect", "segment", "pose"}
    assert sum(isinstance(module, C2fMoT) for module in model.modules()) == (0 if key == "mt_off" else 4)
    with torch.inference_mode():
        outputs = model(torch.zeros(1, 3, 64, 64))
    assert _all_finite(outputs)


def test_multitask_sparse_spec_leaves_sparse_runtime_config_injectable():
    """Sparse training is a run-time treatment, not an immutable YAML property."""
    model = build_model(SPECS["mt_mot_sparse"])
    blocks = [module for module in model.modules() if isinstance(module, MoTBlock)]

    assert blocks
    assert all(not module.sparse_train for module in blocks)
    assert all(not getattr(module, "_mixture_config_explicit", {}) for module in blocks)
    resolved = resolve_mixture_config(
        SimpleNamespace(mot_sparse_train=True, mot_local_attn_window=7),
        model,
    )
    apply_mixture_config(model, resolved)
    assert all(module.sparse_train for module in blocks)
    assert {module.experts[0].local_window_size for module in blocks} == {7}
    model.train()
    with torch.no_grad():
        model(torch.zeros(1, 3, 64, 64))
    assert {module._last_dispatch_stats["policy"] for module in blocks} == {"sparse_train"}


def test_multitask_sparse_manifest_records_requested_runtime_policy(tmp_path, monkeypatch):
    """Requested sparse mode remains visible before the trainer applies it."""
    args = SimpleNamespace(
        data=tmp_path / "data.yaml",
        epochs=1,
        imgsz=64,
        batch=2,
        workers=0,
        optimizer="AdamW",
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        device="cpu",
        amp=False,
        deterministic=True,
        resume=False,
        telemetry=True,
        telemetry_loss_steps=20,
        temperature_factor=0.97,
        temperature_min=0.3,
    )
    monkeypatch.setattr("scripts.compare_mot_ablation.torch.cuda.is_available", lambda: False)
    facade = SimpleNamespace(model=build_model(SPECS["mt_mot_sparse"]))

    manifest = write_run_manifest(args, SPECS["mt_mot_sparse"], tmp_path, 42, facade)
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["model"]["task"] == "multitask"
    assert payload["model"]["variant"] == "mot_sparse_train"
    assert payload["training"]["runtime_overrides_requested"] == {
        "mot_local_attn_window": 7,
        "mot_sparse_train": True,
    }
    assert payload["routing_requested"]
    assert all(not row["sparse_train_at_construction"] for row in payload["routing_requested"])
    assert all(row["sparse_train_requested"] for row in payload["routing_requested"])
    assert {row["local_attn_window_requested"] for row in payload["routing_requested"]} == {7}


def test_multitask_ablation_specs_share_backbone_and_head_topology():
    """The ablation changes only the neck operator, not the task or FPN wiring."""
    payloads = {
        key: (ROOT / SPECS[key].cfg.relative_to(ROOT)).read_text(encoding="utf-8")
        for key in ("mt_off", "mt_mot_dense", "mt_mot_sparse")
    }
    required = ("tasks: ['detect', 'segment', 'pose']", "- [[16, 19, 22], 1, MultiTaskHead, [nc]]")

    for text in payloads.values():
        assert all(value in text for value in required)
        assert text.count("nn.Upsample") == 2
        assert text.count("Concat") == 4


def test_multitask_train_guard_accepts_the_aligned_three_task_contract():
    validate_training_specs(
        [SPECS["mt_off"], SPECS["mt_mot_dense"], SPECS["mt_mot_sparse"]],
        ROOT / "scripts/coco2017_multitask_mps_smoke.yaml",
    )


def test_train_guard_rejects_mixed_task_families(tmp_path):
    with pytest.raises(SystemExit, match="cannot mix task families"):
        validate_training_specs([SPECS["v10"], SPECS["mt_off"]], tmp_path / "ignored.yaml")


def test_multitask_train_guard_rejects_incompatible_tasks(tmp_path):
    data = tmp_path / "data.yaml"
    data.write_text("multitask_format: coco\ntasks: [detect, segment, pose, depth]\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="declare exactly"):
        validate_training_specs([SPECS["mt_off"]], data)
