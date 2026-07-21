from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ultralytics.engine.extensions.recovery import TrainingRecoveryController
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.mixture_loss import initialize_mixture_loss_ema_buffer


def _gloo_coordination_worker(rank, init_file, output_dir):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        trainer = object.__new__(BaseTrainer)
        with patch("ultralytics.engine.trainer.RANK", rank):
            validated = trainer._sync_validation_gate(rank == 0)
            error = "OSError: injected save failure" if rank == 0 else None
            try:
                trainer._sync_rank0_epoch_end_result(error)
            except RuntimeError as exc:
                propagated = str(exc)
            else:
                propagated = ""
        Path(output_dir, f"rank{rank}.txt").write_text(f"{validated}|{propagated}", encoding="utf-8")
    finally:
        dist.destroy_process_group()


def test_validation_gate_uses_rank0_decision():
    trainer = object.__new__(BaseTrainer)
    with patch("ultralytics.engine.trainer.RANK", 1), patch(
        "torch.distributed.is_initialized", return_value=True
    ), patch("torch.distributed.broadcast_object_list", side_effect=lambda value, src: value.__setitem__(0, True)):
        assert trainer._sync_validation_gate(False) is True


def test_rank0_save_error_is_raised_on_nonzero_rank():
    with patch("ultralytics.engine.trainer.RANK", 1), patch(
        "torch.distributed.is_initialized", return_value=True
    ), patch(
        "torch.distributed.broadcast_object_list",
        side_effect=lambda value, src: value.__setitem__(0, "OSError: disk full"),
    ):
        with pytest.raises(RuntimeError, match="disk full"):
            BaseTrainer._sync_rank0_epoch_end_result()


def test_healthy_save_does_not_block_last_and_best(tmp_path):
    trainer = object.__new__(BaseTrainer)
    trainer.healthy = tmp_path / "last_healthy.pt"
    trainer.wdir = tmp_path
    trainer.last = tmp_path / "last.pt"
    trainer.best = tmp_path / "best.pt"
    trainer.best_fitness = trainer.fitness = 0.5
    trainer.save_period = -1
    trainer.epoch = 0
    trainer._serialize_checkpoint = MagicMock(return_value=b"checkpoint")

    assert trainer._recovery_controller().save_healthy(b"checkpoint", state_verified=True) is True
    assert trainer.save_model() is True
    assert trainer.healthy.read_bytes() == b"checkpoint"
    assert trainer.last.read_bytes() == b"checkpoint"
    assert trainer.best.read_bytes() == b"checkpoint"


def _schema_worker(rank, init_file, output_dir, mismatch):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        model = torch.nn.Linear(2, 2)
        model.register_buffer("shared", torch.zeros(1 if not mismatch or rank == 0 else 2))
        trainer = SimpleNamespace(ema=SimpleNamespace(ema=model), world_size=2, device=torch.device("cpu"))
        try:
            TrainingRecoveryController(trainer).sync_ema_buffers()
        except RuntimeError as exc:
            result = f"error:{exc}"
        else:
            result = "ok"
        Path(output_dir, f"schema-rank{rank}.txt").write_text(result, encoding="utf-8")
    finally:
        dist.destroy_process_group()


def test_mixture_buffer_initializer_is_idempotent_and_shape_three():
    model = torch.nn.Linear(2, 2)
    first = initialize_mixture_loss_ema_buffer(model)
    second = initialize_mixture_loss_ema_buffer(model)
    assert first is second
    assert first.shape == (3,)
    assert first.dtype == torch.float32


@pytest.mark.parametrize("mismatch", [False, True])
def test_two_process_gloo_ema_schema_guard(tmp_path, mismatch):
    init_file = tmp_path / f"schema-init-{mismatch}"
    mp.spawn(_schema_worker, args=(str(init_file), str(tmp_path), mismatch), nprocs=2, join=True)
    for rank in range(2):
        result = (tmp_path / f"schema-rank{rank}.txt").read_text(encoding="utf-8")
        assert ("EMA buffer schema mismatch" in result) if mismatch else result == "ok"


def test_two_process_gloo_coordinates_gate_and_save_failure(tmp_path):
    init_file = tmp_path / "gloo-init"
    mp.spawn(_gloo_coordination_worker, args=(str(init_file), str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        result = (tmp_path / f"rank{rank}.txt").read_text(encoding="utf-8")
        assert result.startswith("True|")
        assert "injected save failure" in result
