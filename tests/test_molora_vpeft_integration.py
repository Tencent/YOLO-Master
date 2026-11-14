"""Integration contracts between V-PEFT placement plans and MoLoRA injection."""

import pytest
import torch
import torch.nn as nn

from ultralytics.nn.peft.molora import MoLoRAConfig, MoLoRALayer, MoLoRAModel, get_peft_molora_model
from ultralytics.utils.lora.api import _vpeft_model_fingerprint
from ultralytics.vpeft import PlacementPlan, PlacementTarget


def _model() -> nn.Sequential:
    return nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))


def _plan(model: nn.Module, *, status: str = "ACCEPT") -> PlacementPlan:
    return PlacementPlan(
        model_fingerprint=_vpeft_model_fingerprint(model),
        planner_backend="vpeft",
        solver="ao",
        budget={"max_adapter_params": 1_024},
        targets=(PlacementTarget("0", "lora", 2), PlacementTarget("2", "molora", 3)),
        status=status,
    )


def _config() -> MoLoRAConfig:
    return MoLoRAConfig(r=1, alpha=8, num_experts=2, top_k=1, target_modules=["0"])


def test_molora_wrapper_consumes_per_target_plan_ranks_and_metadata():
    model = _model()
    plan = _plan(model)

    wrapper = MoLoRAModel(model, _config(), placement_plan=plan)

    assert isinstance(wrapper.model[0], MoLoRALayer)
    assert isinstance(wrapper.model[2], MoLoRALayer)
    assert wrapper.model[0].r == 2
    assert wrapper.model[2].r == 3
    assert wrapper.model.molora_rank_map == {"0": 2, "2": 3}
    assert wrapper.model.molora_placement_fingerprint == plan.fingerprint
    assert wrapper.model.molora_placement_plan == plan.to_dict()
    assert wrapper.model[0].molora_placement == {
        "target": "0",
        "variant": "lora",
        "rank": 2,
        "plan_fingerprint": plan.fingerprint,
    }


def test_molora_entry_point_accepts_serialized_placement_plan():
    model = _model()
    plan = _plan(model)

    wrapped = get_peft_molora_model(model, _config(), placement_plan=plan.to_dict())

    assert wrapped.molora_rank_map == {"0": 2, "2": 3}
    assert [wrapped[index].r for index in (0, 2)] == [2, 3]


@pytest.mark.parametrize("failure", ["fingerprint", "target", "variant", "status"])
def test_invalid_molora_placement_plan_is_rejected_before_any_replacement(failure):
    model = _model()
    fingerprint = _vpeft_model_fingerprint(model)
    target = PlacementTarget("0", "lora", 2)
    status = "ACCEPT"
    match = ""
    if failure == "fingerprint":
        fingerprint = "stale-model"
        match = "fingerprint"
    elif failure == "target":
        target = PlacementTarget("1", "lora", 1)
        match = "unsupported"
    elif failure == "variant":
        target = PlacementTarget("0", "dora", 2)
        match = "variant"
    else:
        status = "REFUSE"
        match = "status"
    plan = PlacementPlan(
        model_fingerprint=fingerprint,
        planner_backend="vpeft",
        solver="ao",
        budget={"max_adapter_params": 1_024},
        targets=(target,),
        status=status,
    )
    original_children = tuple(model.children())

    with pytest.raises(ValueError, match=match):
        get_peft_molora_model(model, _config(), placement_plan=plan)

    assert tuple(model.children()) == original_children
    assert not any(isinstance(module, MoLoRALayer) for module in model.modules())
    assert not hasattr(model, "molora_enabled")


def _snapshot(model):
    return (
        tuple(model.named_modules()),
        {name: parameter.detach().clone() for name, parameter in model.named_parameters()},
        {name: parameter.requires_grad for name, parameter in model.named_parameters()},
    )


def _assert_unchanged(model, snapshot):
    modules, values, trainable = snapshot
    assert tuple(model.named_modules()) == modules
    assert {name: parameter.requires_grad for name, parameter in model.named_parameters()} == trainable
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, values[name])
    assert not hasattr(model, "molora_enabled")
    assert not hasattr(model, "molora_config")
    assert not hasattr(model, "molora_placement_plan")


@pytest.mark.parametrize("targets", [["typo"], ["0", "typo"], ["0", "1"], ["0", "0"], [""]])
def test_invalid_explicit_targets_leave_model_and_trainability_unchanged(targets):
    model = _model()
    model[2].bias.requires_grad_(False)
    snapshot = _snapshot(model)

    with pytest.raises(ValueError, match="target"):
        get_peft_molora_model(model, MoLoRAConfig(r=2, target_modules=targets))

    _assert_unchanged(model, snapshot)


def test_empty_auto_detection_refuses_without_freezing_model(monkeypatch):
    from ultralytics.nn.peft.molora.config import MoLoRAConfigBuilder

    model = _model()
    snapshot = _snapshot(model)
    monkeypatch.setattr(MoLoRAConfigBuilder, "auto_detect_targets", lambda *args, **kwargs: [])

    with pytest.raises(ValueError, match="target"):
        get_peft_molora_model(model, MoLoRAConfig(r=2))

    _assert_unchanged(model, snapshot)


@pytest.mark.parametrize("budget", [0, 32, 128, 157])
@pytest.mark.parametrize("frozen_experts", [None, [0, 1]])
def test_molora_budget_includes_all_experts_and_router_and_rolls_back(budget, frozen_experts):
    model = nn.Sequential(nn.Linear(8, 8))
    model[0].bias.requires_grad_(False)
    snapshot = _snapshot(model)
    plan = PlacementPlan(
        model_fingerprint=_vpeft_model_fingerprint(model),
        planner_backend="vpeft",
        solver="ao",
        budget={"max_adapter_params": budget},
        targets=(PlacementTarget("0", "molora", 2),),
        status="ACCEPT",
    )

    cfg = MoLoRAConfig(r=2, num_experts=4, top_k=1, freeze_experts=frozen_experts)
    with pytest.raises(ValueError, match="budget"):
        get_peft_molora_model(model, cfg, placement_plan=plan)

    _assert_unchanged(model, snapshot)
    assert cfg.target_modules is None


def test_molora_exact_budget_reports_real_adapter_parameter_count():
    model = nn.Sequential(nn.Linear(8, 8))
    plan = PlacementPlan(
        model_fingerprint=_vpeft_model_fingerprint(model),
        planner_backend="vpeft",
        solver="ao",
        budget={"max_adapter_params": 158},
        targets=(PlacementTarget("0", "molora", 2),),
        status="ACCEPT",
    )

    get_peft_molora_model(model, MoLoRAConfig(r=2, num_experts=4, top_k=1), placement_plan=plan)

    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 158
    assert model.molora_parameter_budget == {"max_adapter_params": 158, "used_adapter_params": 158}
    assert model.molora_placement_plan == plan.to_dict()


def test_budget_rejection_restores_shared_parameter_trainability():
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    model[1].weight = model[0].weight
    model[1].bias.requires_grad_(False)
    snapshot = _snapshot(model)
    plan = PlacementPlan(
        model_fingerprint=_vpeft_model_fingerprint(model),
        planner_backend="vpeft",
        solver="ao",
        budget={"max_adapter_params": 1},
        targets=(PlacementTarget("0", "molora", 2), PlacementTarget("1", "molora", 2)),
        status="ACCEPT",
    )

    with pytest.raises(ValueError, match="budget"):
        get_peft_molora_model(model, MoLoRAConfig(r=2, num_experts=2, top_k=1), placement_plan=plan)

    _assert_unchanged(model, snapshot)
    assert model[0].weight is model[1].weight
