import io
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from ultralytics.engine.extensions.adapters import AdapterRuntimeController
from ultralytics.engine.extensions.recovery import TrainingRecoveryController
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.lora.fallback import FewShotLoRAConv, ManualLoRAConv
from ultralytics.utils.torch_utils import ModelEMA


class TinyFallbackGraph(nn.Module):
    def __init__(
        self,
        *,
        wrapper_type: type[nn.Module] = ManualLoRAConv,
        use_rslora: bool = True,
        backend: str = "fallback",
    ):
        super().__init__()
        self.layer = wrapper_type(
            nn.Conv2d(4, 4, 1),
            r=8,
            alpha=16,
            dropout=0.05,
            use_rslora=use_rslora,
        )
        self.lora_enabled = True
        self.lora_backend = backend
        self.lora_config = SimpleNamespace(r=8, alpha=16)
        self.lora_runtime_metadata = {
            "effective_backend": backend,
            "requested_use_rslora": use_rslora,
            "effective_use_rslora": use_rslora,
        }
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.endswith((".lora_A", ".lora_B"))

    def forward(self, x):
        return self.layer(x)


class TinyPeftLayer(nn.Module):
    """Minimal PEFT-like layer whose scaling dictionary is absent from state_dict."""

    def __init__(self):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(2, 2, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 2, bias=False)})
        self.scaling = {"default": 0.0}

    def forward(self, inputs):
        return self.lora_B["default"](self.lora_A["default"](inputs)) * self.scaling["default"]


class TinyPeftGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = TinyPeftLayer()
        self.lora_enabled = True
        self.lora_backend = "peft"
        self.lora_runtime_metadata = {"effective_backend": "peft"}

    def forward(self, inputs):
        return self.layer(inputs)


def _trainer(model: nn.Module, *, warmup: int) -> SimpleNamespace:
    optimizer = torch.optim.SGD((parameter for parameter in model.parameters() if parameter.requires_grad), lr=0.1)
    trainer = SimpleNamespace(
        model=model,
        optimizer=optimizer,
        epochs=7,
        resume=True,
        ema=None,
        lora_strategy=None,
        args=SimpleNamespace(
            model="tiny.pt",
            close_mosaic=0,
            lora_type="lora",
            lora_alpha_warmup=warmup,
            lora_layer_decay=0.0,
            lora_ortho_weight=0.0,
            lora_ortho_frequency=10,
            lora_dropout=0.05,
            lora_dropout_end=0.05,
            lora_dropout_start_ratio=1.0,
        ),
    )
    trainer._restore_lora_resume_model = lambda _checkpoint: None
    trainer._load_checkpoint_state = lambda _checkpoint: None
    return trainer


def _controller(model: nn.Module, *, warmup: int):
    trainer = _trainer(model, warmup=warmup)
    controller = AdapterRuntimeController(trainer)
    trainer.adapter_controller = controller
    controller.configure_optimizer(trainer.optimizer)
    trainer.ema = ModelEMA(model)
    return trainer, controller


@pytest.mark.parametrize("wrapper_type", (ManualLoRAConv, FewShotLoRAConv))
@pytest.mark.parametrize("use_rslora", (False, True))
def test_fallback_alpha_warmup_updates_online_and_ema_scaling(wrapper_type, use_rslora: bool):
    model = TinyFallbackGraph(wrapper_type=wrapper_type, use_rslora=use_rslora)
    trainer, controller = _controller(model, warmup=5)
    full_scale = 16 / math.sqrt(8) if use_rslora else 16 / 8

    assert model.layer.scaling == 0.0
    assert trainer.ema.ema.layer.scaling == 0.0

    for epoch in range(7):
        controller.begin_epoch(epoch)
        expected_factor = 0.5 * (1 - math.cos(math.pi * min(epoch / 5, 1.0)))
        assert model.layer.scaling == pytest.approx(full_scale * expected_factor)
        assert trainer.ema.ema.layer.scaling == pytest.approx(model.layer.scaling)


@pytest.mark.parametrize("start_epoch", (2, 6))
def test_resume_restores_scheduled_scaling_online_and_ema(start_epoch: int):
    model = TinyFallbackGraph(use_rslora=True)
    trainer, _ = _controller(model, warmup=5)

    BaseTrainer.resume_training(trainer, {"epoch": start_epoch - 1})

    expected_factor = 0.5 * (1 - math.cos(math.pi * min(start_epoch / 5, 1.0)))
    expected = 16 / math.sqrt(8) * expected_factor
    assert model.layer.scaling == pytest.approx(expected)
    assert trainer.ema.ema.layer.scaling == pytest.approx(expected)


def test_peft_backend_syncs_scaling_dictionary_to_ema():
    model = TinyPeftGraph()
    trainer = _trainer(model, warmup=0)
    controller = AdapterRuntimeController(trainer)
    trainer.adapter_controller = controller
    trainer.ema = ModelEMA(model)
    model.layer.scaling["default"] = 3.0

    assert controller.sync_ema_treatment() == 1
    assert trainer.ema.ema.layer.scaling["default"] == 3.0


def test_validation_syncs_peft_treatment_before_selecting_ema_model():
    model = TinyPeftGraph()
    trainer = _trainer(model, warmup=0)
    controller = AdapterRuntimeController(trainer)
    trainer.adapter_controller = controller
    trainer.ema = ModelEMA(model)
    model.layer.scaling["default"] = 3.0
    trainer._sync_ema_buffers_for_validation = lambda: None
    trainer._state_is_finite = lambda _value: True
    trainer.best_fitness = None
    trainer.loss = torch.tensor(1.0)

    def validator(runtime_trainer):
        assert runtime_trainer.ema.ema.layer.scaling["default"] == 3.0
        return {"fitness": 1.0}

    trainer.validator = validator
    metrics, fitness = BaseTrainer.validate(trainer)

    assert metrics == {}
    assert fitness == 1.0


def test_validation_syncs_fallback_treatment_before_selecting_ema_model():
    model = TinyFallbackGraph()
    trainer, controller = _controller(model, warmup=0)
    trainer.ema.ema.layer.scaling = 0.0
    trainer._sync_ema_buffers_for_validation = lambda: None
    trainer._state_is_finite = lambda _value: True
    trainer.best_fitness = None
    trainer.loss = torch.tensor(1.0)

    def validator(runtime_trainer):
        assert runtime_trainer.ema.ema.layer.scaling == model.layer.scaling
        return {"fitness": 1.0}

    trainer.validator = validator
    metrics, fitness = BaseTrainer.validate(trainer)

    assert metrics == {}
    assert fitness == 1.0
    assert controller.sync_ema_treatment() == 1


def test_checkpoint_serialization_syncs_fallback_treatment():
    model = TinyFallbackGraph()
    trainer, _ = _controller(model, warmup=0)
    trainer.ema.ema.layer.scaling = 0.0
    trainer.scaler = SimpleNamespace(state_dict=lambda: {})
    trainer.epoch = 0
    trainer.start_epoch = 0
    trainer.best_fitness = 0.0
    trainer.metrics = {}
    trainer.fitness = 0.0
    trainer.read_results_csv = lambda: {}

    serialized = TrainingRecoveryController(trainer).serialize_checkpoint(include_online_model=True)
    checkpoint = torch.load(io.BytesIO(serialized), map_location="cpu", weights_only=False)

    assert checkpoint["ema"].layer.scaling == model.layer.scaling
    assert checkpoint["ema"].layer.use_rslora is True


def test_checkpoint_serialization_syncs_peft_treatment():
    model = TinyPeftGraph()
    trainer = _trainer(model, warmup=0)
    trainer.adapter_controller = AdapterRuntimeController(trainer)
    trainer.ema = ModelEMA(model)
    model.layer.scaling["default"] = 3.0
    trainer.scaler = SimpleNamespace(state_dict=lambda: {})
    trainer.epoch = trainer.start_epoch = 0
    trainer.best_fitness = trainer.fitness = 0.0
    trainer.metrics = {}
    trainer.read_results_csv = lambda: {}

    serialized = TrainingRecoveryController(trainer).serialize_checkpoint(include_online_model=True)
    checkpoint = torch.load(io.BytesIO(serialized), map_location="cpu", weights_only=False)

    assert checkpoint["ema"].layer.scaling["default"] == 3.0
