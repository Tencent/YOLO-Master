"""End-to-end contract tests for the opt-in V-PEFT LoRA backend."""

import pytest
import torch.nn as nn

from ultralytics.utils.lora.api import _vpeft_model_fingerprint, apply_lora
from ultralytics.utils.lora.config import LoRAConfig, LoRAConfigBuilder
from ultralytics.utils.lora.fallback import ManualLoRAConv
from ultralytics.vpeft import PlacementPlan, PlacementTarget


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


def test_lora_config_from_args_preserves_rank_pattern():
    rank_pattern = {"model.0": 2, "model.1": 4}

    config = LoRAConfig.from_args(lora_rank_pattern=rank_pattern)

    assert config.rank_pattern == rank_pattern


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


def test_vpeft_plan_preserves_per_target_ranks_in_fallback(monkeypatch):
    import ultralytics.utils.lora.api as api

    model = _model()
    plan = PlacementPlan(
        model_fingerprint=_vpeft_model_fingerprint(model),
        planner_backend="vpeft",
        solver="ao",
        budget={"max_adapter_params": 100_000},
        targets=(PlacementTarget("0", "lora", 1), PlacementTarget("1", "lora", 3)),
        status="ACCEPT",
    )
    monkeypatch.setattr(api, "_build_vpeft_placement_plan", lambda *args, **kwargs: plan)

    wrapped = apply_lora(
        model,
        LoRAConfig(r=4, alpha=8, backend="fallback", planner_backend="vpeft", adapter_budget=100_000),
    )

    assert isinstance(wrapped[0], ManualLoRAConv)
    assert isinstance(wrapped[1], ManualLoRAConv)
    assert wrapped[0].r == 1
    assert wrapped[1].r == 3
    assert wrapped.lora_config.r == 4
    assert wrapped.lora_config.rank_pattern == {"0": 1, "1": 3}
    assert wrapped.lora_runtime_metadata["rank_pattern"] == {"0": 1, "1": 3}
    assert wrapped.lora_runtime_metadata["placement_plan"] == plan.to_dict()


def test_standard_peft_config_receives_per_target_rank_pattern():
    pytest.importorskip("peft")
    rank_pattern = {"0": 1, "1": 3}

    peft_config = LoRAConfigBuilder.create_config(
        _model(),
        r=4,
        alpha=8,
        target_modules=list(rank_pattern),
        rank_pattern=rank_pattern,
        skip_stem=False,
    )

    assert peft_config.r == 4
    assert peft_config.rank_pattern == rank_pattern


def test_active_vpeft_plan_does_not_run_legacy_planner(monkeypatch):
    import ultralytics.utils.lora.api as api
    import ultralytics.utils.lora.planner as planner

    model = _model()
    plan = PlacementPlan(
        model_fingerprint=_vpeft_model_fingerprint(model),
        planner_backend="vpeft",
        solver="ao",
        budget={"max_adapter_params": 100_000},
        targets=(PlacementTarget("0", "lora", 2),),
        status="ACCEPT",
    )
    monkeypatch.setattr(api, "_build_vpeft_placement_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(
        planner.PEFTPlanner,
        "plan",
        lambda *args, **kwargs: pytest.fail("legacy planner must not overwrite an active V-PEFT plan"),
    )

    wrapped = apply_lora(
        model,
        LoRAConfig(
            r=4,
            alpha=8,
            backend="fallback",
            planner_backend="vpeft",
            planner_enabled=True,
            adapter_budget=100_000,
        ),
    )

    assert wrapped.lora_target_modules == ["0"]
    assert wrapped[0].r == 2


def test_vpeft_internal_failure_falls_back_with_structured_metadata(monkeypatch):
    import ultralytics.utils.lora.api as api

    def fail(*args, **kwargs):
        raise RuntimeError("planner bug")

    monkeypatch.setattr(api, "_build_vpeft_placement_plan", fail)
    model = apply_lora(
        _model(),
        LoRAConfig(r=4, alpha=8, backend="fallback", planner_backend="vpeft"),
    )

    assert model.lora_runtime_metadata["vpeft_fallback"] == {
        "category": "internal",
        "reason": "planner bug",
        "message": "planner bug",
        "exception_type": "RuntimeError",
    }
    assert model.lora_runtime_metadata["planner_result"]["status"] == "FALLBACK"
    assert model.lora_runtime_metadata["planner_result"]["reason"]["category"] == "internal"


def test_vpeft_strict_reraises_internal_failure(monkeypatch):
    import ultralytics.utils.lora.api as api

    def fail(*args, **kwargs):
        raise RuntimeError("planner bug")

    monkeypatch.setattr(api, "_build_vpeft_placement_plan", fail)
    with pytest.raises(RuntimeError, match="planner bug"):
        apply_lora(
            _model(),
            LoRAConfig(r=4, alpha=8, backend="fallback", planner_backend="vpeft", vpeft_strict=True),
        )


def test_vpeft_strict_reraises_configuration_failure(monkeypatch):
    import ultralytics.utils.lora.api as api

    def fail(*args, **kwargs):
        raise ValueError("unsupported solver")

    monkeypatch.setattr(api, "_build_vpeft_placement_plan", fail)
    with pytest.raises(ValueError, match="unsupported solver"):
        apply_lora(
            _model(),
            LoRAConfig(r=4, alpha=8, backend="fallback", planner_backend="vpeft", vpeft_strict=True),
        )
