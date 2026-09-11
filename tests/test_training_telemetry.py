"""Unit coverage for opt-in training telemetry measurement contracts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ultralytics.engine.telemetry import (
    TELEMETRY_MAX_STEPS_ENV,
    TELEMETRY_RAW_STEPS_ENV,
    TELEMETRY_ROUTING_ENABLED_ENV,
    TELEMETRY_ROUTING_INTERVAL_ENV,
    TELEMETRY_WARMUP_STEPS_ENV,
    TrainingTelemetry,
    aggregate_rank_records,
    device_memory_sample,
)
from ultralytics.utils import MACOS
from ultralytics.utils.dist import ddp_launch_env, ddp_launch_prefix, find_free_network_port
from ultralytics.utils.routing_telemetry import (
    E3_ROUTING_SCHEMA_VERSION,
    RoutingSnapshotError,
    collect_routing_snapshot,
    normalize_routing_layer,
    routing_aux_status,
    routing_snapshot_scalars,
)

ROOT = Path(__file__).resolve().parents[1]


class RoutedFixture(nn.Module):
    """Small routed module that publishes a configurable snapshot."""

    def __init__(self, family="moe", usage=(0.25, 0.75), *, balance=0.0, observed=0.0, training_only=True):
        super().__init__()
        self.num_experts = len(usage)
        self.top_k = 1
        self.balance_loss_coeff = balance
        self.router_z_loss_coeff = 0.0
        self._training_only = training_only
        self.last_routing_snapshot = {
            "family": family,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "expert_usage": torch.tensor(usage),
            "mean_router_probs": torch.tensor(usage),
            "aux_loss": observed,
            "routing_axis": "image",
        }

    def export_capabilities(self):
        return {"aux_loss_training_only": self._training_only}


class RoutedModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.routed = RoutedFixture(**kwargs)


@pytest.mark.parametrize("family", ["moe", "mot", "latent"])
def test_normalize_routing_layer_supports_required_families(family):
    module = RoutedFixture(family=family)
    probabilities = torch.tensor([[[[0.2]], [[0.8]]], [[[0.4]], [[0.6]]]])

    layer = normalize_routing_layer("model.0", module, probabilities=probabilities)

    assert layer["family"] == family
    assert layer["probability_shape"] == [2, 2, 1, 1]
    assert layer["mean_router_probs"] == pytest.approx([0.3, 0.7])
    assert sum(layer["expert_usage"]) == pytest.approx(1.0)
    assert 0.0 <= layer["normalized_entropy"] <= 1.0


@pytest.mark.parametrize(
    ("usage", "match"),
    [
        ((1.0, float("nan")), "non-finite"),
        ((1.0, -0.1), "negative"),
        ((0.0, 0.0), "positive sum"),
    ],
)
def test_normalize_routing_layer_rejects_invalid_probabilities(usage, match):
    with pytest.raises(RoutingSnapshotError, match=match):
        normalize_routing_layer("bad", RoutedFixture(usage=usage))


def test_normalize_routing_layer_rejects_length_mismatch_and_unknown_family():
    mismatch = RoutedFixture()
    mismatch.num_experts = 3
    mismatch.last_routing_snapshot["num_experts"] = 3
    with pytest.raises(RoutingSnapshotError, match="does not match"):
        normalize_routing_layer("mismatch", mismatch)
    with pytest.raises(RoutingSnapshotError, match="unsupported routing family"):
        normalize_routing_layer("unknown", RoutedFixture(family="custom"))


def test_capture_probabilities_can_normalize_after_interpreter_restores_empty_snapshot():
    module = RoutedFixture()
    module._routing_aux_kind = "moe"
    module.last_routing_snapshot = {}

    layer = normalize_routing_layer("captured", module, probabilities=torch.tensor([[0.4, 0.6]]))

    assert layer["family"] == "moe"
    assert layer["mean_router_probs"] == pytest.approx([0.4, 0.6])


def test_normalize_routing_layer_falls_back_when_optional_snapshot_vectors_are_none():
    module = RoutedFixture()
    module.last_routing_snapshot["mean_router_probs"] = None
    assert normalize_routing_layer("fallback", module)["mean_router_probs"] == pytest.approx([0.25, 0.75])


@pytest.mark.parametrize(
    ("training", "balance", "observed", "training_only", "expected"),
    [
        (True, 0.0, 0.0, True, "not_configured"),
        (False, 0.1, 0.0, True, "configured_inactive_eval"),
        (True, 0.1, 0.0, True, "configured_zero_observed"),
        (True, 0.1, 0.2, True, "active_training"),
        (False, 0.1, 0.2, False, "available"),
    ],
)
def test_routing_aux_status_distinguishes_lifecycle(training, balance, observed, training_only, expected):
    module = RoutedFixture(balance=balance, observed=observed, training_only=training_only)
    module.train(training)
    result = routing_aux_status(module, module.last_routing_snapshot)
    assert result["status"] == expected
    assert isinstance(result["status_code"], float)


def test_collection_reports_unsupported_and_invalid_layers_explicitly():
    model = nn.Module()
    model.good = RoutedFixture(family="moe")
    model.unknown = RoutedFixture(family="custom")
    model.invalid = RoutedFixture(family="mot", usage=(0.0, 0.0))
    payload = collect_routing_snapshot(model)
    assert payload["schema_version"] == E3_ROUTING_SCHEMA_VERSION
    assert set(payload["layers"]) == {"good"}
    assert payload["unsupported_layers"] == payload["invalid_layers"] == 1
    assert {issue["status"] for issue in payload["issues"]} == {"invalid", "unsupported"}


def test_collection_uses_leaf_routed_modules_unless_wrappers_are_requested():
    model = nn.Module()
    model.wrapper = RoutedFixture(family="mot")
    model.wrapper.child = RoutedFixture(family="mot")
    assert set(collect_routing_snapshot(model)["layers"]) == {"wrapper.child"}
    assert set(collect_routing_snapshot(model, include_wrappers=True)["layers"]) == {"wrapper", "wrapper.child"}


def test_tensorboard_scalars_use_stable_family_and_layer_paths():
    scalars = routing_snapshot_scalars(collect_routing_snapshot(RoutedModel(family="latent")))
    prefix = "routing/latent/routed"
    assert scalars["routing/global/routed_layers"] == 1.0
    assert scalars[f"{prefix}/expert_0_usage"] == pytest.approx(0.25)
    assert scalars[f"{prefix}/expert_1_mean_probability"] == pytest.approx(0.75)
    assert scalars[f"{prefix}/aux_status_code"] == 0.0


def test_tensorboard_callback_logs_e3_scalars_without_a_second_writer(monkeypatch):
    from ultralytics.utils.callbacks import tensorboard

    writer = SimpleNamespace(calls=[])
    writer.add_scalar = lambda key, value, step: writer.calls.append((key, value, step))
    monkeypatch.setattr(tensorboard, "WRITER", writer, raising=False)
    trainer = SimpleNamespace(
        epoch=1,
        tloss=torch.tensor([1.0]),
        lr={"lr/pg0": 0.01},
        training_telemetry=SimpleNamespace(tensorboard_scalars=lambda: {"routing/global/routed_layers": 3.0}),
        label_loss_items=lambda _loss, prefix: {f"{prefix}/box_loss": 1.0},
    )
    tensorboard.on_train_epoch_end(trainer)
    assert ("routing/global/routed_layers", 3.0, 2) in writer.calls


def test_disabled_telemetry_exposes_no_tensorboard_metrics():
    assert TrainingTelemetry(enabled=False).tensorboard_scalars() == {}


def _rank_record(rank: int, *, steps: int, samples: int, seconds: float, losses: list[float]):
    return {
        "metadata": {"rank": rank},
        "steps": {"count": steps, "samples": samples, "total_seconds": seconds},
        "loss": {"first_steps": losses},
    }


def test_cpu_memory_is_explicitly_unavailable():
    sample = device_memory_sample("cpu")

    assert sample == {
        "measurement": "unavailable",
        "is_true_peak": False,
        "peak_device_memory_bytes": None,
        "device_total_memory_bytes": None,
        "peak_device_memory_fraction": None,
        "sampled_current_memory_bytes": None,
    }


def test_cuda_memory_reports_true_peak_fraction(monkeypatch):
    properties = SimpleNamespace(total_memory=10_000)
    monkeypatch.setattr("ultralytics.engine.telemetry.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("ultralytics.engine.telemetry.torch.cuda.max_memory_allocated", lambda _device: 8_500)
    monkeypatch.setattr("ultralytics.engine.telemetry.torch.cuda.get_device_properties", lambda _device: properties)

    sample = device_memory_sample("cuda:0")

    assert sample["measurement"] == "cuda_max_memory_allocated"
    assert sample["is_true_peak"] is True
    assert sample["device_total_memory_bytes"] == 10_000
    assert sample["peak_device_memory_fraction"] == pytest.approx(0.85)


def test_rank_aggregation_preserves_step_and_loss_consistency_evidence():
    summary = aggregate_rank_records(
        [
            _rank_record(1, steps=3, samples=6, seconds=3.0, losses=[1.0, 0.9]),
            _rank_record(0, steps=3, samples=6, seconds=2.0, losses=[1.2, 0.9]),
        ]
    )

    assert summary["rank_step_counts"] == {"0": 3, "1": 3}
    assert summary["rank_step_counts_consistent"] is True
    assert summary["global_samples_per_second"] == pytest.approx(4.0)
    assert summary["rank_loss_relative_spread_first_steps"][0]["relative_spread"] == pytest.approx(2 / 11)
    assert summary["rank_peak_device_memory_fraction_max"] is None


def test_rank_aggregation_keeps_the_largest_true_cuda_peak_fraction():
    records = [
        {
            **_rank_record(0, steps=1, samples=2, seconds=1.0, losses=[1.0]),
            "memory": {"peak_device_memory_fraction": 0.70},
        },
        {
            **_rank_record(1, steps=1, samples=2, seconds=1.0, losses=[1.0]),
            "memory": {"peak_device_memory_fraction": 0.85},
        },
    ]

    summary = aggregate_rank_records(records)

    assert summary["rank_peak_device_memory_fraction_max"] == pytest.approx(0.85)


def test_training_telemetry_records_cpu_step_contract(tmp_path, monkeypatch):
    monkeypatch.setattr("ultralytics.engine.telemetry.routing_runtime_metrics", lambda _model: {"routed_layers": 0})
    telemetry = TrainingTelemetry(enabled=True, loss_steps=2)
    trainer = SimpleNamespace(
        device=torch.device("cpu"),
        batch_size=2,
        args=SimpleNamespace(device="cpu", deterministic=True),
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=0.1),
        model=torch.nn.Identity(),
        wdir=tmp_path / "weights",
        save_dir=tmp_path,
    )
    trainer.wdir.mkdir()

    telemetry.on_pretrain_routine_end(trainer)
    trainer.batch = {"img": torch.ones(2, 3, 8, 8)}
    trainer.loss_items = torch.tensor([1.0, 2.0])
    telemetry.on_train_batch_start(trainer)
    telemetry.on_train_batch_end(trainer)
    record = telemetry._record(trainer)

    assert record["steps"]["count"] == 1
    assert record["steps"]["samples"] == 2
    assert record["loss"]["first_steps"] == [3.0]
    assert record["memory"]["measurement"] == "unavailable"
    assert record["memory"]["peak_device_memory_bytes"] is None
    assert record["memory"]["peak_device_memory_fraction"] is None


def test_training_telemetry_samples_first_batch_then_interval(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "ultralytics.engine.telemetry.collect_routing_snapshot",
        lambda _model: calls.append(len(calls)) or {"routed_layers": 0, "layers": {}, "issues": []},
    )
    monkeypatch.setattr("ultralytics.engine.telemetry.routing_runtime_metrics", lambda _model: {})
    telemetry = TrainingTelemetry(enabled=True, routing_interval=3)
    trainer = SimpleNamespace(
        device=torch.device("cpu"),
        batch_size=1,
        args=SimpleNamespace(device="cpu", deterministic=True),
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=0.1),
        model=torch.nn.Identity(),
        wdir=tmp_path / "weights",
        save_dir=tmp_path,
        batch={"img": torch.ones(1, 3, 8, 8)},
        loss_items=torch.tensor([1.0]),
    )
    trainer.wdir.mkdir()
    telemetry.on_pretrain_routine_end(trainer)

    for _ in range(7):
        telemetry.on_train_batch_start(trainer)
        telemetry.on_train_batch_end(trainer)

    assert len(calls) == 3
    assert telemetry._record(trainer)["routing"]["sampling_interval_steps"] == 3


def test_training_telemetry_reads_routing_interval_from_environment(monkeypatch):
    monkeypatch.setenv(TELEMETRY_ROUTING_INTERVAL_ENV, "17")
    monkeypatch.setenv(TELEMETRY_ROUTING_ENABLED_ENV, "false")
    monkeypatch.setenv(TELEMETRY_WARMUP_STEPS_ENV, "50")
    monkeypatch.setenv(TELEMETRY_MAX_STEPS_ENV, "200")
    monkeypatch.setenv(TELEMETRY_RAW_STEPS_ENV, "yes")

    telemetry = TrainingTelemetry.from_environment()

    assert telemetry.routing_interval == 17
    assert telemetry.routing_enabled is False
    assert telemetry.warmup_steps == 50
    assert telemetry.max_steps == 200
    assert telemetry.raw_steps is True


def test_routing_telemetry_does_not_change_output_gradient_or_rng(tmp_path, monkeypatch):
    """Sampling may request diagnostics but must leave training semantics unchanged."""

    class RoutedToy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.last_routing_snapshot = {}

        def forward(self, value):
            return value * self.weight + torch.rand_like(value)

    def run(routing_enabled):
        torch.manual_seed(11)
        model = RoutedToy()
        trainer = SimpleNamespace(
            device=torch.device("cpu"),
            batch_size=1,
            args=SimpleNamespace(device="cpu", deterministic=True),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            model=model,
            wdir=tmp_path / f"weights-{routing_enabled}",
            save_dir=tmp_path,
            batch={"img": torch.ones(1)},
            loss_items=torch.tensor([1.0]),
        )
        trainer.wdir.mkdir()
        telemetry = TrainingTelemetry(enabled=True, routing_enabled=routing_enabled)
        telemetry.on_pretrain_routine_end(trainer)
        telemetry.on_train_batch_start(trainer)
        output = model(torch.ones(1))
        output.sum().backward()
        telemetry.on_train_batch_end(trainer)
        return output.detach(), model.weight.grad.detach(), torch.rand(1)

    monkeypatch.setattr(
        "ultralytics.engine.telemetry.collect_routing_snapshot",
        lambda _model: {"routed_layers": 0, "layers": {}, "issues": []},
    )
    monkeypatch.setattr("ultralytics.engine.telemetry.routing_runtime_metrics", lambda _model: {})

    enabled = run(True)
    disabled = run(False)

    for enabled_value, disabled_value in zip(enabled, disabled):
        assert torch.equal(enabled_value, disabled_value)


def test_training_telemetry_window_routing_disable_and_raw_steps(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "ultralytics.engine.telemetry.collect_routing_snapshot",
        lambda _model: calls.append(True) or {"routed_layers": 0, "layers": {}, "issues": []},
    )
    monkeypatch.setattr("ultralytics.engine.telemetry.routing_runtime_metrics", lambda _model: {})
    telemetry = TrainingTelemetry(
        enabled=True,
        routing_enabled=False,
        warmup_steps=2,
        max_steps=3,
        raw_steps=True,
    )
    trainer = SimpleNamespace(
        device=torch.device("cpu"),
        batch_size=1,
        args=SimpleNamespace(device="cpu", deterministic=True),
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=0.1),
        model=torch.nn.Identity(),
        wdir=tmp_path / "weights",
        save_dir=tmp_path,
        batch={"img": torch.ones(1, 3, 8, 8)},
        loss_items=torch.tensor([1.0]),
    )
    trainer.wdir.mkdir()
    telemetry.on_pretrain_routine_end(trainer)

    for _ in range(7):
        telemetry.on_train_batch_start(trainer)
        telemetry.on_train_batch_end(trainer)

    record = telemetry._record(trainer)
    assert record["steps"]["count"] == 3
    assert len(record["steps"]["raw_milliseconds"]) == 3
    assert len(record["instrumented_steps"]["raw_milliseconds"]) == 3
    assert record["instrumented_steps"]["warmup_steps"] == 2
    assert record["instrumented_steps"]["max_steps"] == 3
    assert record["routing"]["enabled"] is False
    assert record["routing"]["observations"] == 0
    assert calls == []


def test_training_telemetry_invalid_environment_values_keep_compatible_defaults(monkeypatch):
    monkeypatch.setenv(TELEMETRY_ROUTING_INTERVAL_ENV, "invalid")
    monkeypatch.setenv(TELEMETRY_WARMUP_STEPS_ENV, "invalid")
    monkeypatch.setenv(TELEMETRY_MAX_STEPS_ENV, "invalid")

    telemetry = TrainingTelemetry.from_environment()

    assert telemetry.routing_interval == 1
    assert telemetry.routing_enabled is True
    assert telemetry.warmup_steps == 0
    assert telemetry.max_steps == 0
    assert telemetry.raw_steps is False


def test_training_telemetry_forces_fresh_snapshot_only_during_sample_step(tmp_path, monkeypatch):
    class SnapshotProducer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.last_routing_snapshot = {}

    producer = SnapshotProducer()
    telemetry = TrainingTelemetry(enabled=True, routing_interval=2)
    monkeypatch.setattr(
        "ultralytics.engine.telemetry.collect_routing_snapshot",
        lambda _model: {"routed_layers": 0, "layers": {}, "issues": []},
    )
    monkeypatch.setattr("ultralytics.engine.telemetry.routing_runtime_metrics", lambda _model: {})
    trainer = SimpleNamespace(
        device=torch.device("cpu"),
        batch_size=1,
        args=SimpleNamespace(device="cpu", deterministic=True),
        optimizer=torch.optim.SGD(producer.parameters(), lr=0.1),
        model=producer,
        wdir=tmp_path / "weights",
        save_dir=tmp_path,
        batch={"img": torch.ones(1, 3, 8, 8)},
        loss_items=torch.tensor([1.0]),
    )
    trainer.wdir.mkdir()
    telemetry.on_pretrain_routine_end(trainer)

    telemetry.on_train_batch_start(trainer)
    assert producer._moe_force_snapshot is True
    telemetry.on_train_batch_end(trainer)
    assert not hasattr(producer, "_moe_force_snapshot")

    telemetry.on_train_batch_start(trainer)
    assert not hasattr(producer, "_moe_force_snapshot")


def test_training_telemetry_teardown_restores_force_flag_after_interrupted_batch(tmp_path):
    class SnapshotProducer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.last_routing_snapshot = {}

    producer = SnapshotProducer()
    telemetry = TrainingTelemetry(enabled=True)
    trainer = SimpleNamespace(
        device=torch.device("cpu"),
        batch_size=1,
        args=SimpleNamespace(device="cpu", deterministic=True),
        optimizer=torch.optim.SGD(producer.parameters(), lr=0.1),
        model=producer,
        wdir=tmp_path / "weights",
        save_dir=tmp_path,
        batch={"img": torch.ones(1, 3, 8, 8)},
        loss_items=torch.tensor([1.0]),
    )
    trainer.wdir.mkdir()
    telemetry.on_pretrain_routine_end(trainer)
    telemetry.on_train_batch_start(trainer)
    assert producer._moe_force_snapshot is True

    telemetry.on_teardown(trainer)

    assert not hasattr(producer, "_moe_force_snapshot")


def test_cpu_gloo_two_rank_telemetry_artifact_gate(tmp_path):
    if MACOS and os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.skip("Nested torchrun under macOS xdist is unreliable; run this gate serially")

    command = [
        *ddp_launch_prefix(),
        "--master_addr=127.0.0.1",
        f"--master_port={find_free_network_port()}",
        "--nproc_per_node=2",
        str(ROOT / "tests/ddp_telemetry_smoke.py"),
    ]
    env = {
        **ddp_launch_env(),
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": os.pathsep.join(filter(None, (str(ROOT), os.environ.get("PYTHONPATH")))),
        "TELEMETRY_SMOKE_DIR": str(tmp_path),
    }
    completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=180, check=False)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "P1 telemetry DDP gate passed" in completed.stdout
