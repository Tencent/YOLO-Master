from types import SimpleNamespace
from unittest import mock

import pytest
import torch.nn as nn

from ultralytics.engine.model import Model
from ultralytics.nn.peft.molora import MoLoRAConfig, MoLoRAModel
from ultralytics.utils.lora import LoRAConfig, adapter_metadata
from ultralytics.utils.lora.fallback import ManualLoRAConv, apply_manual_lora
from ultralytics.utils.patches import torch_load


def _facade(model: nn.Module, trainer_model: nn.Module | None = None) -> Model:
    facade = Model.__new__(Model)
    nn.Module.__init__(facade)
    facade.model = model
    facade.trainer = SimpleNamespace(model=trainer_model) if trainer_model is not None else None
    return facade


def _fallback_model() -> nn.Module:
    model = nn.Sequential(nn.Conv2d(4, 4, 1))
    return apply_manual_lora(
        model,
        LoRAConfig(r=2, alpha=4, backend="fallback", target_modules=["0"], skip_stem=False),
    )


def _molora_model() -> MoLoRAModel:
    return MoLoRAModel(
        nn.Sequential(nn.Linear(8, 8)),
        MoLoRAConfig(r=2, alpha=4, num_experts=2, top_k=1, target_modules=["0"]),
    )


def test_save_lora_only_prefers_live_trainer_model(tmp_path):
    live_model = _fallback_model()
    facade = _facade(nn.Sequential(nn.Conv2d(4, 4, 1)), live_model)

    assert facade.save_lora_only(tmp_path / "adapter")
    assert (tmp_path / "adapter" / "fallback_adapter.pt").exists()


def test_fallback_lora_facade_round_trip_and_merge(tmp_path):
    source = _facade(_fallback_model())
    adapter_path = tmp_path / "adapter"
    assert source.save_lora_only(adapter_path)

    trainer_model = nn.Sequential(nn.Conv2d(4, 4, 1))
    restored = _facade(nn.Sequential(nn.Conv2d(4, 4, 1)), trainer_model)
    assert restored.load_lora(adapter_path, trainable=True)
    assert restored.trainer.model is restored.model
    assert isinstance(restored.model[0], ManualLoRAConv)
    assert restored.merge_lora()
    assert restored.trainer.model is restored.model
    assert isinstance(restored.model[0], nn.Conv2d)
    assert not getattr(restored.model, "lora_enabled", False)


def test_molora_facade_uses_backend_metadata_and_live_model(tmp_path):
    live_model = _molora_model()
    facade = _facade(nn.Sequential(nn.Linear(8, 8)), live_model)
    adapter_path = tmp_path / "molora"

    assert facade.save_adapters(adapter_path)
    assert (adapter_path / "runtime_metadata.json").exists()
    assert facade.merge_adapters(mode="uniform")
    assert facade.model is live_model
    assert facade.trainer.model is live_model

    restored = _facade(_molora_model())
    assert restored.load_adapters(adapter_path)


def test_train_preserves_preloaded_lora_model(monkeypatch):
    facade = _facade(_fallback_model())
    facade.model.yaml = {}
    facade.overrides = {"model": "toy.yaml"}
    facade.ckpt = {}
    facade.ckpt_path = None
    facade.cfg = "toy.yaml"
    facade.task = "detect"
    facade.session = None
    facade.callbacks = {}

    trainer = mock.Mock()
    trainer.model = None
    trainer.train.side_effect = RuntimeError("stop after trainer setup")
    monkeypatch.setattr(facade, "_smart_load", mock.Mock(return_value=lambda overrides, _callbacks: trainer))
    monkeypatch.setattr("ultralytics.utils.checks.check_pip_update_available", lambda: None)

    with pytest.raises(RuntimeError, match="stop after trainer setup"):
        facade.train(data="coco8.yaml", epochs=1)

    trainer.get_model.assert_not_called()
    assert trainer.model is facade.model


def test_facade_save_records_mixture_checkpoint_metadata(tmp_path):
    model = nn.Sequential(nn.Linear(8, 8))
    model.task = "classify"
    facade = _facade(model)
    facade.ckpt = {}
    checkpoint = tmp_path / "model.pt"

    facade.save(checkpoint)
    payload = torch_load(checkpoint, map_location="cpu", weights_only=False)

    assert payload["mixture_checkpoint"]["schema_version"] == 1
    assert payload["mixture_checkpoint"]["graph"]["model_class"].endswith("Sequential")


def test_molora_metadata_accepts_none_target_modules():
    model = nn.Sequential(nn.Linear(8, 8))
    model.molora_enabled = True
    model.molora_config = MoLoRAConfig(r=2, alpha=4, num_experts=2, top_k=1, target_modules=None)

    metadata = adapter_metadata(model)

    assert metadata["backend"] == "molora"
    assert metadata["target_modules"] == []
