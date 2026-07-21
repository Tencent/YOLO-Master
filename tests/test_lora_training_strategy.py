from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ultralytics.engine.extensions.adapters import AdapterRuntimeController
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.lora import LoraTrainingStrategy


class _AdapterLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.ones(2, 2))
        self.lora_B = nn.Parameter(torch.zeros(2, 2))


class _LayeredAdapterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList([_AdapterLayer(), _AdapterLayer()])


def test_orthogonal_regularization_is_disabled_by_default():
    assert DEFAULT_CFG_DICT["lora_ortho_weight"] == 0.0


def test_adapter_regularizer_is_added_once_for_component_loss(monkeypatch):
    trainer = SimpleNamespace(model=nn.Linear(2, 2))
    controller = AdapterRuntimeController(trainer)
    controller.strategy = object()
    controller.ortho_weight = 0.5
    controller.ortho_frequency = 1
    monkeypatch.setattr(
        LoraTrainingStrategy,
        "compute_orthogonal_loss",
        staticmethod(lambda model, weight: torch.tensor(weight)),
    )

    augmented = controller.augment_loss(torch.tensor([1.0, 2.0, 3.0]))

    assert augmented.tolist() == pytest.approx([1.5, 2.0, 3.0])
    assert augmented.sum().item() == pytest.approx(6.5)


def test_layer_decay_rebuilds_all_adapter_groups_without_duplicates():
    model = _LayeredAdapterModel()
    named = dict(model.named_parameters())
    first = [named["model.0.lora_A"], named["model.0.lora_B"]]
    second = [named["model.1.lora_A"], named["model.1.lora_B"]]
    optimizer = torch.optim.AdamW(
        [
            {"params": first, "lr": 0.01, "param_group": "adapter"},
            {"params": second, "lr": 0.02, "param_group": "adapter"},
        ]
    )

    adjusted = LoraTrainingStrategy(model).apply_layer_decay_to_optimizer(optimizer, decay_rate=0.85)
    parameter_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]

    assert adjusted == 4
    assert len(parameter_ids) == len(set(parameter_ids)) == 4
    assert set(parameter_ids) == {id(parameter) for parameter in model.parameters()}
    assert sorted(group["lr"] for group in optimizer.param_groups) == pytest.approx([0.01, 0.018])
