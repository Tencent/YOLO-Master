"""PlacementPlan serialization and fingerprint contracts."""

import pytest

from ultralytics.vpeft import PlacementPlan, PlacementTarget


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
