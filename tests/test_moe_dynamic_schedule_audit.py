"""Focused tests for epoch-level MoE routing and recovery-safe scheduling."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.modules.moe._common import _record_moe_snapshot, _should_record_snapshot
from ultralytics.nn.modules.moe.schedule import usage_ginis_with_observation_counts_from_model
from ultralytics.nn.modules.moe.utils import reset_moe_usage_accumulator


def test_forced_snapshot_bypasses_sampling_interval_once():
    module = nn.Identity()
    module._force_moe_snapshot = True
    assert _should_record_snapshot(module)
    assert module._force_moe_snapshot is False


def test_dynamic_epoch_usage_accumulates_every_forward():
    module = nn.Identity()
    reset_moe_usage_accumulator(module)
    module._force_moe_snapshot = True
    _record_moe_snapshot(module, expert_usage=torch.tensor([1.0, 0.0, 0.0]))
    _record_moe_snapshot(module, expert_usage=torch.tensor([0.0, 1.0, 0.0]))
    gini, observations = usage_ginis_with_observation_counts_from_model(module)[0]
    assert gini == pytest.approx(1 / 3)
    assert observations == 2


def test_trainer_emits_nontrivial_event_trace(tmp_path):
    class SnapshotModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.balance_loss_coeff = 1.0
            self.last_routing_snapshot = {"expert_usage": torch.tensor([1.0, 0.0, 0.0, 0.0])}

    layer = SnapshotModule()
    trainer = SimpleNamespace(
        args=SimpleNamespace(
            moe_dynamic_schedule="gini_balance",
            moe_dynamic_gini_target=0.25,
            moe_dynamic_gini_alpha=1.0,
            moe_dynamic_gini_beta=0.8,
            moe_dynamic_balance_min=0.5,
            moe_dynamic_balance_max=2.0,
        ),
        world_size=1,
        model=nn.Sequential(layer),
        save_dir=tmp_path,
        moe_dynamic_scheduler=None,
        moe_dynamic_metrics={},
        moe_dynamic_counts={"opportunity_count": 0, "event_count": 0, "nontrivial_action_count": 0},
        moe_dynamic_previous_coeff=None,
        _arm_moe_dynamic_snapshots=lambda: 0,
    )
    BaseTrainer._setup_moe_dynamic_scheduler(trainer, base_balance_loss=1.0)
    BaseTrainer._update_moe_dynamic_scheduler(trainer, epoch=0)
    record = json.loads((tmp_path / "moe_dynamic_trace.jsonl").read_text().strip())
    assert record["routing_observation_count"] == 1
    assert record["opportunity_count"] == 1
    assert record["event_count"] == 1
    assert record["nontrivial_action_count"] == 1


@pytest.mark.parametrize("recovered, expected", [(True, {"arm": 1, "update": 0}), (False, {"arm": 0, "update": 1})])
def test_scheduler_advances_only_for_accepted_epoch(recovered, expected):
    calls = {"arm": 0, "update": 0}
    trainer = SimpleNamespace(
        moe_dynamic_scheduler=object(),
        moe_dynamic_metrics={"stale": 1},
        _arm_moe_dynamic_snapshots=lambda: calls.__setitem__("arm", calls["arm"] + 1),
        _update_moe_dynamic_scheduler=lambda epoch: calls.__setitem__("update", calls["update"] + 1),
    )
    accepted = BaseTrainer._finalize_moe_dynamic_epoch(trainer, epoch=8, recovered=recovered)
    assert accepted is not recovered
    assert calls == expected


def test_dynamic_schedule_rejects_unaggregated_ddp_usage():
    trainer = SimpleNamespace(
        args=SimpleNamespace(moe_dynamic_schedule="gini_balance"),
        world_size=2,
        moe_dynamic_scheduler=None,
        moe_dynamic_metrics={},
    )
    with pytest.raises(RuntimeError, match="not aggregated across DDP ranks"):
        BaseTrainer._setup_moe_dynamic_scheduler(trainer, base_balance_loss=1.0)
