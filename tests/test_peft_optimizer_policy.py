from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.optim.muon import MuSGD


class _AdapterFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.lora_A = nn.Parameter(torch.ones(4, 2))
        self.lora_B = nn.Parameter(torch.zeros(2, 4))
        self.lora_enabled = True


class _MoLoRAResumeFixture(nn.Module):
    def __init__(self, value=0.0):
        super().__init__()
        self.router = nn.Parameter(torch.full((2, 2), value))
        self.register_buffer("_step_count", torch.tensor(0, dtype=torch.long))
        self.molora_enabled = True


def _trainer(*, adapter_active: bool, nc: int = 20) -> BaseTrainer:
    trainer = object.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(
        moe_router_lr_scale=0.5,
        lora_lr_mult=1.0,
        warmup_bias_lr=0.1,
        lr0=0.01,
        momentum=0.937,
    )
    trainer.data = {"nc": nc}
    trainer.adapter_controller = SimpleNamespace(active=adapter_active)
    return trainer


@pytest.mark.parametrize("iterations", [100, 78_000])
def test_auto_optimizer_uses_stable_adamw_policy_for_active_peft(iterations):
    trainer = _trainer(adapter_active=True)
    model = _AdapterFixture()

    optimizer = trainer.build_optimizer(
        model, name="auto", lr=0.01, momentum=0.937, decay=0.001, iterations=iterations
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    assert trainer.args.warmup_bias_lr == 0.0
    active_groups = [group for group in optimizer.param_groups if group["params"]]
    assert all(group["lr"] == pytest.approx(0.000417) for group in active_groups)
    parameter_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert set(parameter_ids) == {id(parameter) for parameter in model.parameters()}


def test_auto_optimizer_preserves_fitted_lr_for_single_class_peft():
    trainer = _trainer(adapter_active=True, nc=1)

    optimizer = trainer.build_optimizer(
        _AdapterFixture(), name="auto", lr=0.01, momentum=0.937, decay=0.001, iterations=78_000
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    active_groups = [group for group in optimizer.param_groups if group["params"]]
    assert all(group["lr"] == pytest.approx(0.002) for group in active_groups)


def test_auto_optimizer_preserves_iteration_based_musgd_for_full_finetuning():
    trainer = _trainer(adapter_active=False)
    model = nn.Linear(4, 4)

    optimizer = trainer.build_optimizer(
        model, name="auto", lr=0.01, momentum=0.937, decay=0.001, iterations=78_000
    )

    assert isinstance(optimizer, MuSGD)


def test_explicit_musgd_request_is_respected_for_adapter_training():
    trainer = _trainer(adapter_active=True)

    optimizer = trainer.build_optimizer(
        _AdapterFixture(), name="MuSGD", lr=0.01, momentum=0.9, decay=0.001, iterations=78_000
    )

    assert isinstance(optimizer, MuSGD)


def test_resume_skips_musgd_state_when_auto_policy_now_builds_adamw():
    model = _AdapterFixture()
    model.lora_enabled = True
    source = MuSGD(model.parameters(), lr=0.01, momentum=0.9)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    source.step()

    trainer = object.__new__(BaseTrainer)
    trainer.model = model
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=0.000417)
    trainer.scaler = SimpleNamespace(load_state_dict=lambda _: None)
    trainer.ema = None
    trainer.best_fitness = None
    trainer._load_checkpoint_state(
        {
            "optimizer": source.state_dict(),
            "scaler": None,
            "ema": None,
            "best_fitness": 0.0,
        }
    )

    assert trainer.optimizer.state == {}


def test_resume_skips_incompatible_state_for_molora_adapter():
    model = _AdapterFixture()
    model.lora_enabled = False
    model.molora_enabled = True
    source = MuSGD(model.parameters(), lr=0.01, momentum=0.9)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    source.step()

    trainer = object.__new__(BaseTrainer)
    trainer.model = model
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=0.000417)
    trainer.scaler = SimpleNamespace(load_state_dict=lambda _: None)
    trainer.ema = None
    trainer.best_fitness = None
    trainer._load_checkpoint_state(
        {
            "optimizer": source.state_dict(),
            "scaler": None,
            "ema": None,
            "best_fitness": 0.0,
        }
    )

    assert trainer.optimizer.state == {}


def test_resume_restores_molora_state_into_online_model():
    online = _MoLoRAResumeFixture(0.0)
    checkpoint_ema = _MoLoRAResumeFixture(3.0)
    checkpoint_ema._step_count.fill_(7)
    trainer = object.__new__(BaseTrainer)
    trainer.model = online

    trainer._restore_lora_resume_model({"ema": checkpoint_ema})

    assert torch.allclose(online.router, checkpoint_ema.router)
    assert online._step_count.item() == 7


def test_resume_rejects_incompatible_full_sft_optimizer_state():
    model = nn.Linear(4, 4)
    source = MuSGD(model.parameters(), lr=0.01, momentum=0.9)
    trainer = object.__new__(BaseTrainer)
    trainer.model = model
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=0.000417)
    trainer.scaler = SimpleNamespace(load_state_dict=lambda _: None)
    trainer.ema = None
    trainer.best_fitness = None

    with pytest.raises(ValueError, match="incompatible full-SFT"):
        trainer._load_checkpoint_state({"optimizer": source.state_dict(), "scaler": None, "ema": None})
