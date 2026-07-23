"""End-to-end contract tests for the opt-in V-PEFT LoRA backend."""

import torch.nn as nn

from ultralytics.utils.lora.api import apply_lora
from ultralytics.utils.lora.config import LoRAConfig


def _model():
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.Conv2d(8, 8, 3, padding=1),
    )


def test_vpeft_backend_compiles_plan_and_injects_selected_targets():
    model = apply_lora(
        _model(),
        LoRAConfig(
            r=4,
            alpha=8,
            backend="fallback",
            planner_backend="vpeft",
            adapter_budget=100_000,
        ),
    )

    plan = model.lora_placement_plan
    assert plan["planner_backend"] == "vpeft"
    assert plan["status"] == "ACCEPT"
    assert plan["targets"]
    assert model.lora_target_modules == [item["name"] for item in plan["targets"]]
    assert all(item["rank"] > 0 for item in plan["targets"])


def test_vpeft_refusal_falls_back_to_legacy_targets():
    model = apply_lora(
        _model(),
        LoRAConfig(
            r=4,
            alpha=8,
            backend="fallback",
            planner_backend="vpeft",
            adapter_budget=1,
        ),
    )

    assert model.lora_placement_plan["planner_backend"] == "vpeft"
    assert model.lora_placement_plan["status"] == "REFUSE"
    # The legacy path remains usable even if the budget is infeasible.
    assert getattr(model, "lora_enabled", False)


def test_legacy_backend_does_not_compile_vpeft_plan(monkeypatch):
    import ultralytics.utils.lora.api as api

    monkeypatch.setattr(
        api,
        "_build_vpeft_placement_plan",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    model = apply_lora(
        _model(),
        LoRAConfig(r=4, alpha=8, backend="fallback", planner_backend="legacy"),
    )
    assert not hasattr(model, "lora_placement_plan")

