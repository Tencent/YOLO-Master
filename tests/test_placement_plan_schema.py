"""PlacementPlan serialization and fingerprint contracts."""

import pytest
import torch.nn as nn

from ultralytics.vpeft import PlacementPlan, PlacementTarget, PlannerResult


def test_placement_plan_roundtrip_is_stable():
    plan = PlacementPlan(
        model_fingerprint="model-1",
        planner_backend="vpeft",
        solver="ao",
        budget={"max_adapter_params": 1024},
        targets=(PlacementTarget("model.1", "lora", 8),),
        status="ADAPT",
    )
    restored = PlacementPlan.from_dict(plan.to_dict())
    assert restored == plan
    assert restored.fingerprint == plan.fingerprint


def test_placement_plan_rejects_tampering():
    plan = PlacementPlan(
        model_fingerprint="model-1",
        planner_backend="legacy",
        solver="none",
        budget={"max_adapter_params": 0},
    )
    payload = plan.to_dict()
    payload["targets"] = [{"name": "tampered", "rank": 4, "variant": "lora"}]
    with pytest.raises(ValueError, match="fingerprint"):
        PlacementPlan.from_dict(payload)


def test_placement_plan_validates_bound_model_and_target_capacity():
    from ultralytics.utils.lora.api import _vpeft_model_fingerprint

    model = nn.Sequential(nn.Linear(4, 4))
    plan = PlacementPlan(
        model_fingerprint=_vpeft_model_fingerprint(model),
        planner_backend="vpeft",
        solver="ao",
        budget={"max_adapter_params": 128},
        targets=(PlacementTarget("0", "lora", 2),),
        status="ACCEPT",
    )
    plan.validate_model(model)

    with pytest.raises(ValueError, match="fingerprint"):
        plan.validate_model(nn.Sequential(nn.Linear(4, 8)))


def test_placement_plan_rejects_stale_or_unsupported_target():
    model = nn.Sequential(nn.ReLU())
    plan = PlacementPlan(
        model_fingerprint="",
        planner_backend="vpeft",
        solver="ao",
        budget={"max_adapter_params": 128},
        targets=(PlacementTarget("0", "lora", 1),),
        status="ACCEPT",
    )
    with pytest.raises(ValueError, match="unsupported"):
        plan.validate_model(model)
