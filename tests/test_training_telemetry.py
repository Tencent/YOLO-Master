"""Unit coverage for opt-in training telemetry measurement contracts."""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ultralytics.engine.telemetry import TrainingTelemetry, aggregate_rank_records, device_memory_sample
from ultralytics.utils.dist import ddp_launch_env, ddp_launch_prefix, find_free_network_port


ROOT = Path(__file__).resolve().parents[1]


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


def test_cpu_gloo_two_rank_telemetry_artifact_gate(tmp_path):
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
    completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=90)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "P1 telemetry DDP gate passed" in completed.stdout
