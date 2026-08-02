"""MoE facade tiers, compatibility exports, and snapshot protocol regression tests."""

import pytest
import torch

import ultralytics.nn.modules.moe as facade
from ultralytics.nn.modules.moe.protocol import normalize_routing_snapshot, routing_metrics
from ultralytics.nn.modules.moe.schedule import usage_gini as schedule_gini
from ultralytics.nn.modules.moe.scheduler import compute_gini


def test_public_api_tiers_are_disjoint_and_resolvable():
    tiers = (facade.STABLE_MOE_CLASSES, facade.EXPERIMENTAL_MOE_CLASSES, facade.LEGACY_MOE_CLASSES)
    assert not (tiers[0] & tiers[1] or tiers[0] & tiers[2] or tiers[1] & tiers[2])
    for name in set().union(*tiers):
        assert name in facade.__all__
        assert getattr(facade, name) is not None


def test_legacy_checkpoint_aliases_remain_importable():
    assert facade.A2C2fMoE.__name__ == "A2C2fMoE"
    assert facade.ABlockMoE.__name__ == "ABlockMoE"
    assert facade.is_legacy_moe("A2C2fMoE")


def test_snapshot_adapter_accepts_legacy_keys_without_mutation():
    source = {"usage": torch.tensor([0.75, 0.25]), "counts": [3, 1], "top_k": 1}
    normalized = normalize_routing_snapshot(source, num_experts=2)
    assert normalized["expert_usage"] == [0.75, 0.25]
    assert normalized["topk_counts"] == [3.0, 1.0]
    assert "expert_usage" not in source


def test_scheduler_and_diagnostics_share_gini_definition():
    usage = torch.tensor([1.0, 0.0, 0.0, 0.0])
    metrics = routing_metrics({"expert_usage": usage}, num_experts=4)
    assert metrics.gini == pytest.approx(0.75)
    assert compute_gini(usage) == pytest.approx(metrics.gini)
    assert schedule_gini(usage) == pytest.approx(metrics.gini)
