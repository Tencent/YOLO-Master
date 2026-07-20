from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch
from torch import nn

from ultralytics.engine.extensions import AdapterRuntimeController
from ultralytics.engine.trainer import BaseTrainer, validate_adapter_configuration
from ultralytics.nn.peft.molora import MoLoRAConfig, MoLoRALayer, get_peft_molora_model
from ultralytics.utils.errors import MoERouterError
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import ModelEMA


class E(nn.Module):
    def __init__(self, p=False):
        super().__init__()
        self.register_buffer("diagnostic", torch.tensor(1.0), persistent=p)


class ImageSmokeModel(nn.Module):
    def __init__(self, nonfinite=False):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.yaml = {"channels": 3}
        self.stride = torch.tensor([32.0])
        self.nonfinite = nonfinite

    def forward(self, x):
        output = self.conv(x)
        return output / 0.0 if self.nonfinite else output


class FuseSensitiveSmokeModel(ImageSmokeModel):
    def fuse(self, verbose=False):
        self.nonfinite = True
        return self


class RTDETRDecoder(nn.Module):
    def forward(self, x):
        if min(x.shape[-2:]) < 128:
            raise RuntimeError("selected index k out of range")
        return x


class RTDETRSmokeModel(ImageSmokeModel):
    def __init__(self):
        super().__init__()
        self.decoder = RTDETRDecoder()

    def forward(self, x):
        return self.decoder(super().forward(x))


def test_adapter_configuration_rejects_lora_and_molora_together():
    args = SimpleNamespace(lora_r=8, lora_auto_r_ratio=0.0, lora_type="lora", molora_num_experts=4)

    with pytest.raises(ValueError, match="cannot be enabled in the same training run"):
        validate_adapter_configuration(args)


@pytest.mark.parametrize("peft_type", ["oft", "boft", "ia3", "hra"])
def test_adapter_configuration_rejects_rankless_peft_and_molora_together(peft_type):
    args = SimpleNamespace(lora_r=0, lora_auto_r_ratio=0.0, lora_type=peft_type, molora_num_experts=4)

    with pytest.raises(ValueError, match="rankless"):
        validate_adapter_configuration(args)


def test_adapter_configuration_accepts_single_adapter_family():
    validate_adapter_configuration(
        SimpleNamespace(lora_r=0, lora_auto_r_ratio=0.0, lora_type="lora", molora_num_experts=4)
    )
    validate_adapter_configuration(
        SimpleNamespace(lora_r=8, lora_auto_r_ratio=0.0, lora_type="lora", molora_num_experts=0)
    )


def test_trainer_freeze_pass_preserves_molora_base_parameters():
    trainer = object.__new__(BaseTrainer)
    trainer.model = get_peft_molora_model(
        nn.Sequential(nn.Linear(8, 8)),
        MoLoRAConfig(r=2, alpha=4, num_experts=2, top_k=1, target_modules=["0"]),
    )
    trainer.args = SimpleNamespace(freeze=None)
    trainer.adapter_controller = AdapterRuntimeController(trainer)

    trainer._freeze_model_parameters()

    layer = next(module for module in trainer.model.modules() if isinstance(module, MoLoRALayer))
    assert not any(parameter.requires_grad for parameter in layer.base_layer.parameters())
    assert any(parameter.requires_grad for parameter in layer.experts.parameters())


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
    model = nn.Linear(1, 1)
    torch.save({"model": model, "ema": nn.Linear(1, 1), "optimizer": None, "scaler": None, "best_fitness": 0.4, "updates": 0}, path)


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

    checkpoint = torch_load(
        __import__("io").BytesIO(trainer._serialize_checkpoint()), map_location="cpu", weights_only=False
    )

    assert checkpoint["epoch"] == 6


def test_checkpoint_serialization_clamps_fp16_overflow_without_mutating_live_ema(tmp_path):
    trainer = bootstrap_trainer(tmp_path)
    parameter = next(trainer.ema.ema.parameters())
    parameter.data.flatten()[0] = 1.0e5

    checkpoint = torch_load(
        __import__("io").BytesIO(trainer._serialize_checkpoint()), map_location="cpu", weights_only=False
    )

    assert all(
        torch.isfinite(value).all()
        for value in checkpoint["ema"].state_dict().values()
        if isinstance(value, torch.Tensor)
    )
    assert parameter.dtype == torch.float32
    assert parameter.data.flatten()[0] == 1.0e5


def test_recovery_controller_resyncs_nonfinite_ema_from_online_model(tmp_path):
    trainer = bootstrap_trainer(tmp_path)
    online = next(trainer.model.parameters())
    ema = next(trainer.ema.ema.parameters())
    ema.data.flatten()[0] = float("inf")

    assert trainer._recovery_controller().resync_nonfinite_ema() is True
    assert torch.equal(ema, online)


def test_recovery_rejects_legacy_checkpoint_without_online_model(tmp_path):
    t = recovery_trainer(tmp_path, loss=float("nan"))
    torch.save({"ema": nn.Linear(1, 1), "optimizer": None, "scaler": None, "best_fitness": 0.4, "updates": 0}, t.healthy)
    with pytest.raises(RuntimeError, match="lacks online model state"):
        t._handle_nan_recovery(0)


def test_nonfinite_loss_recovers_from_healthy_checkpoint(tmp_path):
    t = recovery_trainer(tmp_path, loss=float("nan"))
    write_healthy(t.healthy)
    assert t._handle_nan_recovery(3) is True
    assert t.nan_recovery_attempts == 1
    t._load_checkpoint_state.assert_called_once()


def test_recovery_clears_non_checkpoint_moe_registry(tmp_path):
    from ultralytics.nn.modules.moe._common import MOE_LOSS_REGISTRY

    t = recovery_trainer(tmp_path, loss=float("nan"))
    module = nn.Linear(1, 1)
    MOE_LOSS_REGISTRY[module] = torch.tensor(float("nan"))
    write_healthy(t.healthy)
    try:
        assert t._handle_nan_recovery(0) is True
        assert not list(MOE_LOSS_REGISTRY.items())
    finally:
        MOE_LOSS_REGISTRY.clear()


def test_validate_skips_nonfinite_ema_and_marks_recovery():
    t = object.__new__(BaseTrainer)
    t.device = torch.device("cpu")
    t.ema = SimpleNamespace(ema=nn.Linear(1, 1))
    with torch.no_grad():
        t.ema.ema.weight.fill_(float("nan"))
    t._sync_ema_buffers_for_validation = MagicMock()
    t.validator = MagicMock()

    metrics, fitness = t.validate()

    assert metrics == {}
    assert fitness != fitness
    assert t._ema_nonfinite is True
    t.validator.assert_not_called()


def test_loss_recovery_halves_scaler_and_clears_gradients(tmp_path):
    t = bootstrap_trainer(tmp_path)
    t.loss = torch.tensor(float("nan"))
    t._gradient_nonfinite = False
    t._bootstrap_healthy_checkpoint()
    t.model.weight.grad = torch.full_like(t.model.weight, float("nan"))
    initial_scale = t.scaler.get_scale()

    assert t._handle_nan_recovery(0) is True
    assert t.scaler.get_scale() == max(initial_scale * 0.5, 1.0)
    assert t.model.weight.grad is None


def test_gradient_recovery_preserves_reduced_scaler_state(tmp_path):
    t = recovery_trainer(tmp_path, loss=1.0)
    t._gradient_nonfinite = True
    t.scaler = MagicMock()
    t.scaler.state_dict.return_value = {"scale": 32768.0}
    write_healthy(t.healthy)

    assert t._handle_nan_recovery(0) is True
    t.scaler.load_state_dict.assert_called_once_with({"scale": 32768.0})


def test_nonfinite_amp_recovery_switches_to_fp32(tmp_path):
    t = bootstrap_trainer(tmp_path)
    t.amp = True
    t._gradient_nonfinite = True
    write_healthy(t.healthy)

    assert t._handle_nan_recovery(0) is True
    assert t.amp is False
    assert t.scaler.is_enabled() is False


def test_nonfinite_gradient_skips_optimizer_on_all_ranks():
    class RecordingScaler:
        def __init__(self):
            self.events = []

        def unscale_(self, optimizer):
            self.events.append("unscale")

        def get_scale(self):
            return 65536.0

        def is_enabled(self):
            return True

        def step(self, optimizer):
            self.events.append("step")

        def update(self):
            self.events.append("update")

    trainer = object.__new__(BaseTrainer)
    trainer.model = nn.Linear(1, 1)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    trainer.scaler = RecordingScaler()
    trainer.ema = None
    trainer._gradient_nonfinite = False
    trainer._nonfinite_diagnostic = None
    trainer.epoch = 0
    trainer.ni = 0
    trainer.loss = torch.tensor(1.0)
    trainer.fitness = 0.0

    trainer.model.weight.grad = torch.full_like(trainer.model.weight, float("nan"))
    assert trainer.optimizer_step() is False

    assert trainer.scaler.events == ["unscale", "update"]
    assert trainer.model.weight.grad is None


def test_remote_nonfinite_gradient_prevents_local_optimizer_commit():
    trainer = object.__new__(BaseTrainer)
    trainer.model = nn.Linear(1, 1)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
    trainer.ema = None
    trainer._gradient_nonfinite = False
    trainer._nonfinite_diagnostic = None
    trainer.device = torch.device("cpu")
    trainer.model(torch.ones(1, 1)).sum().backward()
    initial_weight = trainer.model.weight.detach().clone()

    def report_remote_nonfinite(flag, op):
        assert op == torch.distributed.ReduceOp.MAX
        flag.fill_(1)

    with patch("ultralytics.engine.trainer.RANK", 1), patch(
        "torch.distributed.is_initialized", return_value=True
    ), patch("torch.distributed.get_backend", return_value="gloo"), patch(
        "torch.distributed.all_reduce", side_effect=report_remote_nonfinite
    ) as all_reduce:
        assert trainer.optimizer_step() is False

    all_reduce.assert_called_once()
    assert trainer._gradient_nonfinite is True
    assert torch.equal(trainer.model.weight, initial_weight)
    assert trainer.model.weight.grad is None


def test_remote_nonfinite_loss_is_shared_with_local_rank():
    trainer = object.__new__(BaseTrainer)
    trainer.device = torch.device("cpu")

    def report_remote_nonfinite(flag, op):
        assert op == torch.distributed.ReduceOp.MAX
        flag.fill_(1)

    with patch("ultralytics.engine.trainer.RANK", 1), patch(
        "torch.distributed.is_initialized", return_value=True
    ), patch("torch.distributed.get_backend", return_value="gloo"), patch(
        "torch.distributed.all_reduce", side_effect=report_remote_nonfinite
    ):
        assert trainer._sync_nonfinite_flag(False) is True


def test_finite_optimizer_step_reports_commit():
    trainer = object.__new__(BaseTrainer)
    trainer.model = nn.Linear(1, 1)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
    trainer.ema = None
    trainer._gradient_nonfinite = False
    trainer.args = SimpleNamespace(lora_few_shot_mode=False)
    trainer.model(torch.ones(1, 1)).sum().backward()

    assert trainer.optimizer_step() is True


def test_validate_converts_router_nan_to_recovery_signal():
    t = object.__new__(BaseTrainer)
    t.loss = torch.tensor(1.0)
    t.best_fitness = 0.0
    t._sync_ema_buffers_for_validation = MagicMock()
    t.validator = MagicMock(side_effect=MoERouterError("Router input contains NaN/Inf values [EfficientSpatialRouter]"))

    with patch("ultralytics.engine.trainer.RANK", -1):
        metrics, fitness = t.validate()

    assert metrics == {}
    assert fitness != fitness


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


def test_checkpoint_forward_smoke_rejects_nonfinite_activation():
    t = object.__new__(BaseTrainer)
    t.args = SimpleNamespace(imgsz=640)
    checkpoint = {"model": ImageSmokeModel(nonfinite=True), "ema": None}

    healthy, reason = t._checkpoint_forward_smoke(checkpoint)

    assert healthy is False
    assert "non-finite output" in reason


def test_checkpoint_forward_smoke_covers_fused_fp32_path():
    t = object.__new__(BaseTrainer)
    t.args = SimpleNamespace(imgsz=640)
    checkpoint = {"model": FuseSensitiveSmokeModel(), "ema": None}

    healthy, reason = t._checkpoint_forward_smoke(checkpoint)

    assert healthy is False
    assert "non-finite output" in reason


def test_checkpoint_forward_smoke_uses_rtdetr_safe_minimum_shape():
    t = object.__new__(BaseTrainer)
    t.args = SimpleNamespace(imgsz=160)

    healthy, reason = t._checkpoint_forward_smoke({"model": RTDETRSmokeModel(), "ema": None})

    assert healthy is True
    assert reason == ""


def test_save_model_writes_last_and_best_without_recovery_gate(tmp_path):
    t = object.__new__(BaseTrainer)
    t.wdir = tmp_path
    t.last = tmp_path / "last.pt"
    t.best = tmp_path / "best.pt"
    t.last.write_bytes(b"prior-last")
    t.best.write_bytes(b"prior-best")
    t.best_fitness = t.fitness = 0.5
    t.save_period = -1
    t.epoch = 0
    t._serialize_checkpoint = MagicMock(return_value=b"bad-checkpoint")
    assert t.save_model() is True
    assert t.last.read_bytes() == b"bad-checkpoint"
    assert t.best.read_bytes() == b"bad-checkpoint"


def final_eval_trainer(tmp_path):
    t = object.__new__(BaseTrainer)
    t.best = tmp_path / "best.pt"
    t.last = tmp_path / "last.pt"
    t.healthy = tmp_path / "last_healthy.pt"
    for path in (t.best, t.last, t.healthy):
        path.write_bytes(b"checkpoint")
    t.args = SimpleNamespace(plots=False)
    t.validator = MagicMock(return_value={"fitness": 0.5, "metrics/mAP50": 0.4})
    t.run_callbacks = MagicMock()
    return t


def test_final_eval_falls_back_from_bad_best_to_healthy_checkpoint(tmp_path):
    t = final_eval_trainer(tmp_path)
    t._validate_checkpoint_artifact = MagicMock(
        side_effect=lambda path: (False, "bad best") if path == t.best else (True, "")
    )

    with patch("ultralytics.engine.trainer.strip_optimizer", return_value={}):
        t.final_eval()

    t.validator.assert_called_once_with(model=t.healthy)
    assert t.metrics == {"metrics/mAP50": 0.4}


def test_final_eval_catches_router_error_and_retries_healthy(tmp_path):
    t = final_eval_trainer(tmp_path)
    t._validate_checkpoint_artifact = MagicMock(return_value=(True, ""))
    t.validator.side_effect = [
        MoERouterError("Router input contains NaN/Inf values [EfficientSpatialRouter]"),
        {"fitness": 0.5, "metrics/mAP50": 0.4},
    ]

    with patch("ultralytics.engine.trainer.strip_optimizer", return_value={}):
        t.final_eval()

    assert t.validator.call_args_list == [
        call(model=t.best),
        call(model=t.healthy),
    ]


def test_final_eval_forces_fp32_for_reloaded_fused_checkpoints(tmp_path):
    t = final_eval_trainer(tmp_path)
    t.validator.args.half = True
    t._validate_checkpoint_artifact = MagicMock(return_value=(True, ""))

    def reject_fused_fp16(*, model):
        if t.validator.args.half:
            raise MoERouterError("Router input contains NaN/Inf values [EfficientSpatialRouter]")
        return {"fitness": 0.5, "metrics/mAP50": 0.4}

    t.validator.side_effect = reject_fused_fp16
    with patch("ultralytics.engine.trainer.strip_optimizer", return_value={}):
        t.final_eval()

    assert t.validator.args.half is False
    t.validator.assert_called_once_with(model=t.best)


def test_final_eval_resets_router_runtime_before_each_candidate(tmp_path):
    t = final_eval_trainer(tmp_path)
    t._validate_checkpoint_artifact = MagicMock(return_value=(True, ""))
    t.validator.side_effect = [
        MoERouterError("Router input contains NaN/Inf values [EfficientSpatialRouter]"),
        {"fitness": 0.5, "metrics/mAP50": 0.4},
    ]
    t._reset_non_checkpoint_moe_runtime_state = MagicMock()

    with patch("ultralytics.engine.trainer.strip_optimizer", return_value={}):
        t.final_eval()

    assert t._reset_non_checkpoint_moe_runtime_state.call_count == 2


def test_final_eval_raises_clear_error_when_best_and_healthy_are_bad(tmp_path):
    t = final_eval_trainer(tmp_path)
    t._validate_checkpoint_artifact = MagicMock(return_value=(False, "non-finite output"))

    with patch("ultralytics.engine.trainer.strip_optimizer", return_value={}), pytest.raises(
        RuntimeError, match="No healthy checkpoint is available for final evaluation"
    ):
        t.final_eval()

    t.validator.assert_not_called()


def test_final_eval_checkpoint_decision_is_broadcast_to_nonzero_rank(tmp_path):
    t = final_eval_trainer(tmp_path)

    def share_rank0_decision(container, src):
        assert src == 0
        container[0] = ([str(t.healthy)], ["best.pt: failed smoke"])

    with patch("ultralytics.engine.trainer.RANK", 1), patch(
        "torch.distributed.is_initialized", return_value=True
    ), patch("torch.distributed.broadcast_object_list", side_effect=share_rank0_decision) as broadcast:
        candidates, rejected = t._select_final_eval_checkpoints()

    assert candidates == [t.healthy]
    assert rejected == ["best.pt: failed smoke"]
    broadcast.assert_called_once()


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
    payload = torch_load(t.healthy, map_location="cpu", weights_only=False)
    assert BaseTrainer._state_is_finite(payload)
    assert payload["optimizer"] is not None
    assert payload["scaler"] == t.scaler.state_dict()
    assert payload["updates"] == t.ema.updates

    with torch.no_grad():
        t.model.weight.fill_(99.0)
    assert t._handle_nan_recovery(0) is True
    assert torch.allclose(t.model.weight, payload["model"].float().weight)
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
