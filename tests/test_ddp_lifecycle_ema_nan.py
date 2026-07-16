from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.errors import MoERouterError
from ultralytics.utils.torch_utils import ModelEMA


class E(nn.Module):
    def __init__(self, p=False):
        super().__init__()
        self.register_buffer("diagnostic", torch.tensor(1.0), persistent=p)


def tr(m):
    t = object.__new__(BaseTrainer)
    t.ema = SimpleNamespace(ema=m)
    t.world_size = 2
    return t


def recovery_trainer(tmp_path, loss=1.0, fitness=0.0, best_fitness=0.4):
    t = object.__new__(BaseTrainer)
    t.loss = torch.tensor(loss)
    t.fitness = fitness
    t.best_fitness = best_fitness
    t.start_epoch = 0
    t.device = torch.device("cpu")
    t.healthy = tmp_path / "last_healthy.pt"
    t.last = tmp_path / "last.pt"
    t.wdir = tmp_path
    t._gradient_nonfinite = False
    t.nan_recovery_attempts = 0
    t.model = nn.Linear(1, 1)
    t.scheduler = SimpleNamespace(last_epoch=0)
    t._model_train = MagicMock()
    t._load_checkpoint_state = MagicMock()
    return t


def write_healthy(path):
    torch.save({"ema": nn.Linear(1, 1), "optimizer": None, "scaler": None, "best_fitness": 0.4, "updates": 0}, path)


def test_nccl_skips_nonpersistent_cpu():
    t = tr(E(False))
    with patch("torch.distributed.is_initialized", return_value=True), patch(
        "torch.distributed.get_backend", return_value="nccl"
    ), patch("torch.distributed.broadcast") as broadcast:
        t._sync_ema_buffers_for_validation()
    broadcast.assert_not_called()


def test_nccl_moves_persistent_cpu_buffer_before_broadcast():
    t = tr(E(True))
    t.device = torch.device("cpu")
    with patch("torch.distributed.is_initialized", return_value=True), patch(
        "torch.distributed.get_backend", return_value="nccl"
    ), patch("torch.distributed.broadcast") as broadcast:
        t._sync_ema_buffers_for_validation()
    broadcast.assert_called_once()


def test_train_destroys_group_on_error():
    t = object.__new__(BaseTrainer)
    t.ddp = False
    t._do_train = MagicMock(side_effect=RuntimeError("boom"))
    with patch("torch.distributed.is_available", return_value=True), patch(
        "torch.distributed.is_initialized", return_value=True
    ), patch("torch.distributed.destroy_process_group") as destroy, pytest.raises(RuntimeError):
        t.train()
    destroy.assert_called_once_with()


def test_nonfinite_without_checkpoint_fails(tmp_path):
    t = recovery_trainer(tmp_path, loss=float("nan"))
    with pytest.raises(RuntimeError, match="without a healthy recovery checkpoint"):
        t._handle_nan_recovery(0)


def test_zero_fitness_is_not_nonfinite_or_recovery_trigger(tmp_path):
    t = recovery_trainer(tmp_path, loss=1.0, fitness=0.0, best_fitness=0.4)
    assert t._handle_nan_recovery(3) is False


def test_bootstrap_checkpoint_serializes_before_training_epoch_is_set(tmp_path):
    trainer = bootstrap_trainer(tmp_path)
    trainer.start_epoch = 7
    del trainer.epoch

    checkpoint = torch.load(
        __import__("io").BytesIO(trainer._serialize_checkpoint()), map_location="cpu", weights_only=False
    )

    assert checkpoint["epoch"] == 6


def test_nonfinite_loss_recovers_from_healthy_checkpoint(tmp_path):
    t = recovery_trainer(tmp_path, loss=float("nan"))
    write_healthy(t.healthy)
    assert t._handle_nan_recovery(3) is True
    assert t.nan_recovery_attempts == 1
    t._load_checkpoint_state.assert_called_once()


def test_gradient_recovery_preserves_reduced_scaler_state(tmp_path):
    t = recovery_trainer(tmp_path, loss=1.0)
    t._gradient_nonfinite = True
    t.scaler = MagicMock()
    t.scaler.state_dict.return_value = {"scale": 32768.0}
    write_healthy(t.healthy)

    assert t._handle_nan_recovery(0) is True
    t.scaler.load_state_dict.assert_called_once_with({"scale": 32768.0})


def test_validate_skips_and_disables_val_on_router_nonfinite():
    t = object.__new__(BaseTrainer)
    t.loss = torch.tensor(1.0)
    t.best_fitness = 0.0
    t._sync_ema_buffers_for_validation = MagicMock()
    t.validator = MagicMock(side_effect=MoERouterError("Router input contains NaN/Inf values [EfficientSpatialRouter]"))
    t.args = SimpleNamespace(val=True)

    with patch("ultralytics.engine.trainer.RANK", -1):
        metrics, fitness = t.validate()

    assert metrics is None
    assert fitness is None
    assert t.args.val is False


def test_checkpoint_restore_tolerates_missing_lazy_ema_buffer():
    t = object.__new__(BaseTrainer)
    t.model = nn.Linear(1, 1)
    t.ema = ModelEMA(t.model)
    t.optimizer = torch.optim.SGD(t.model.parameters(), lr=0.01)
    t.scaler = torch.amp.GradScaler("cuda", enabled=False)
    old_ema = nn.Linear(1, 1)

    t.model.register_buffer("_mixture_loss_ema_buf", torch.tensor([1.0, 0.1, 0.1]))
    t._load_checkpoint_state(
        {"ema": old_ema, "optimizer": None, "scaler": None, "best_fitness": 0.0, "updates": 0}
    )

    assert torch.equal(t.ema.ema._mixture_loss_ema_buf, torch.tensor([1.0, 0.1, 0.1]))


def test_healthy_checkpoint_rejects_nonfinite_state_and_preserves_prior(tmp_path):
    t = object.__new__(BaseTrainer)
    t.healthy = tmp_path / "last_healthy.pt"
    t.healthy.write_bytes(b"known-good")
    buffer = __import__("io").BytesIO()
    torch.save({"tensor": torch.tensor(float("nan"))}, buffer)
    assert t._save_healthy_checkpoint(buffer.getvalue()) is False
    assert t.healthy.read_bytes() == b"known-good"


def bootstrap_trainer(tmp_path):
    t = object.__new__(BaseTrainer)
    t.model = nn.Linear(1, 1)
    t.ema = ModelEMA(t.model)
    t.optimizer = torch.optim.AdamW(t.model.parameters(), lr=0.01)
    t.scaler = torch.amp.GradScaler("cuda", enabled=False)
    t.scheduler = SimpleNamespace(last_epoch=0)
    t.args = SimpleNamespace()
    t.epoch = 0
    t.start_epoch = 0
    t.best_fitness = 0.0
    t.fitness = 0.0
    t.metrics = {}
    t.wdir = tmp_path
    t.healthy = tmp_path / "last_healthy.pt"
    t.last = tmp_path / "last.pt"
    t.device = torch.device("cpu")
    t.loss = torch.tensor(float("nan"))
    t._gradient_nonfinite = True
    t.nan_recovery_attempts = 0
    t._model_train = MagicMock()
    t.read_results_csv = MagicMock(return_value={})
    return t


def test_bootstrap_checkpoint_precedes_first_nonfinite_recovery(tmp_path):
    t = bootstrap_trainer(tmp_path)
    with patch("ultralytics.engine.trainer.RANK", -1):
        t._bootstrap_healthy_checkpoint()
    payload = torch.load(t.healthy, map_location="cpu", weights_only=False)
    assert BaseTrainer._state_is_finite(payload)
    assert payload["optimizer"] is not None
    assert payload["scaler"] == t.scaler.state_dict()
    assert payload["updates"] == t.ema.updates

    with torch.no_grad():
        t.model.weight.fill_(99.0)
    assert t._handle_nan_recovery(0) is True
    assert torch.allclose(t.model.weight, payload["ema"].float().weight)
    assert t.optimizer.state_dict()["state"] == {}
    assert t.scaler.state_dict() == payload["scaler"]
    assert t.ema.updates == payload["updates"]


def test_bootstrap_failure_never_creates_unverified_checkpoint(tmp_path):
    t = bootstrap_trainer(tmp_path)
    with torch.no_grad():
        t.model.weight.fill_(float("nan"))
    with patch("ultralytics.engine.trainer.RANK", -1), pytest.raises(RuntimeError, match="Initial training state is nonfinite"):
        t._bootstrap_healthy_checkpoint()
    assert not t.healthy.exists()


def test_bootstrap_broadcasts_rank0_health_to_all_ddp_ranks(tmp_path):
    t = bootstrap_trainer(tmp_path)

    def copy_rank0_status(status, src):
        assert src == 0
        status.fill_(1)

    with patch("ultralytics.engine.trainer.RANK", 1), patch(
        "torch.distributed.get_backend", return_value="gloo"
    ), patch("torch.distributed.broadcast", side_effect=copy_rank0_status) as broadcast:
        t._bootstrap_healthy_checkpoint()
    broadcast.assert_called_once()
    assert not t.healthy.exists()
