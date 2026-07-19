# ruff: noqa: E402
import sys
import types

import pytest
import torch

_seaborn_stub = types.ModuleType("seaborn")
_seaborn_stub.heatmap = lambda *_args, **_kw: None  # type: ignore[attr-defined]
sys.modules.setdefault("seaborn", _seaborn_stub)

from ultralytics.nn.modules.moe.loss import MoELoss
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.modules.moe.modules import AdaptiveBalanceController, UltraOptimizedMoE
from ultralytics.nn.modules.moe.scheduler import (
    MoEDynamicScheduler,
    MoEDynamicSchedulerConfig,
    MapSaturationScheduler,
    MapSaturationSchedulerConfig,
    MapSaturationScheduleState,
    compute_gini,
)


def test_compute_gini_uniform_and_collapsed():
    assert compute_gini(torch.ones(4)) == pytest.approx(0.0, abs=1e-6)
    assert compute_gini(torch.tensor([1.0, 0.0, 0.0, 0.0])) > 0.70


def test_dynamic_scheduler_raises_coeff_for_imbalanced_usage():
    scheduler = MoEDynamicScheduler(MoEDynamicSchedulerConfig(target_gini=0.2, gain=2.0, ema_momentum=0.0))
    balanced = scheduler.step(torch.ones(4), base_balance_coeff=1.0)
    collapsed = scheduler.step(torch.tensor([1.0, 0.0, 0.0, 0.0]), base_balance_coeff=1.0)

    assert balanced.balance_loss_coeff < 1.0
    assert collapsed.balance_loss_coeff > 1.0
    assert collapsed.gini > balanced.gini


def test_moeloss_dynamic_schedule_changes_balance_component():
    torch.manual_seed(0)
    logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]]).repeat(8, 1).requires_grad_(True)
    probs = torch.softmax(logits, dim=1)
    indices = torch.zeros(8, 2, dtype=torch.long)
    loss_fn = MoELoss(
        num_experts=4,
        top_k=2,
        z_loss_coeff=0.0,
        dynamic_scheduler_config=MoEDynamicSchedulerConfig(target_gini=0.2, gain=2.0, ema_momentum=0.0),
    )

    out = loss_fn(probs, logits, indices, return_dict=True)

    assert out["dynamic_schedule"]["balance_loss_coeff"] > 1.0
    assert out["loss"].requires_grad


def test_adaptive_balance_controller_accepts_dynamic_scheduler():
    ctrl = AdaptiveBalanceController(
        4,
        initial_coeff=1.0,
        final_coeff=1.0,
        dynamic_scheduler_config=MoEDynamicSchedulerConfig(target_gini=0.2, gain=2.0, ema_momentum=0.0),
    )
    usage = torch.tensor([1.0, 0.0, 0.0, 0.0])
    router_probs = torch.softmax(torch.randn(8, 4, requires_grad=True), dim=1)

    loss = ctrl({"expert_usage": usage, "router_probs": router_probs}, torch.tensor(0.0))

    assert torch.isfinite(loss)
    assert ctrl.last_dynamic_schedule is not None
    assert ctrl.last_dynamic_schedule["balance_loss_coeff"] > 1.0



def test_compute_gini_edge_cases():
    assert compute_gini(torch.tensor([])) == 0.0
    assert compute_gini(torch.zeros(4)) == 0.0
    assert compute_gini(torch.tensor([1.0])) == pytest.approx(0.0, abs=1e-6)


def test_dynamic_scheduler_disabled_passthrough():
    scheduler = MoEDynamicScheduler(MoEDynamicSchedulerConfig(enabled=False))
    state = scheduler.step(torch.tensor([1.0, 0.0, 0.0, 0.0]), base_balance_coeff=1.23)
    assert state.balance_loss_coeff == pytest.approx(1.23)


def test_dynamic_scheduler_state_dict_round_trip():
    scheduler = MoEDynamicScheduler(MoEDynamicSchedulerConfig(target_gini=0.3, gain=1.0))
    scheduler.step(torch.tensor([1.0, 0.0, 0.0, 0.0]), base_balance_coeff=1.0)
    sd = scheduler.state_dict()

    restored = MoEDynamicScheduler()
    restored.load_state_dict(sd)

    assert restored.ema_gini == pytest.approx(scheduler.ema_gini)
    assert restored.config.target_gini == pytest.approx(0.3)
    assert restored.last_state is not None




def test_map_saturation_no_plateau_keeps_scale_at_one():
    scheduler = MapSaturationScheduler(
        MapSaturationSchedulerConfig(window_size=3, saturation_threshold=0.005, decay_factor=0.8)
    )
    # Feed strictly increasing mAP — no plateau should be detected.
    maps = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
    state = None
    for m in maps:
        state = scheduler.update(m)
    assert state is not None
    assert state.saturation_scale == pytest.approx(1.0)
    assert not state.plateau_detected


def test_map_saturation_plateau_decays_scale():
    scheduler = MapSaturationScheduler(
        MapSaturationSchedulerConfig(window_size=3, saturation_threshold=0.01, decay_factor=0.7)
    )
    # Feed flat mAP — plateau should trigger after 2*window_size epochs.
    state = None
    for _ in range(6):
        state = scheduler.update(0.20)
    assert state is not None
    assert state.plateau_detected
    assert state.saturation_scale < 1.0


def test_map_saturation_scale_floored_at_min():
    scheduler = MapSaturationScheduler(
        MapSaturationSchedulerConfig(window_size=2, saturation_threshold=1.0, decay_factor=0.1, min_scale=0.05)
    )
    # Force repeated plateaus to hit the floor.
    for _ in range(20):
        scheduler.update(0.20)
    assert scheduler.saturation_scale == pytest.approx(0.05)


def test_map_saturation_apply_scales_coefficient():
    scheduler = MapSaturationScheduler(MapSaturationSchedulerConfig(enabled=True))
    scheduler.saturation_scale = 0.5
    assert scheduler.apply(2.0) == pytest.approx(1.0)


def test_map_saturation_disabled_passthrough():
    scheduler = MapSaturationScheduler(MapSaturationSchedulerConfig(enabled=False))
    for _ in range(10):
        scheduler.update(0.20)
    assert scheduler.apply(1.5) == pytest.approx(1.5)


def test_map_saturation_state_dict_round_trip():
    scheduler = MapSaturationScheduler(MapSaturationSchedulerConfig(window_size=3, decay_factor=0.9))
    for m in [0.10, 0.11, 0.12, 0.13, 0.14, 0.15]:
        scheduler.update(m)
    sd = scheduler.state_dict()

    restored = MapSaturationScheduler()
    restored.load_state_dict(sd)

    assert restored.saturation_scale == pytest.approx(scheduler.saturation_scale)
    assert restored.map_history == scheduler.map_history
    assert restored.config.window_size == 3


@pytest.mark.parametrize(
    ("recovered", "validated", "fitness", "expected_updates"),
    [
        (True, True, 0.5, 0),
        (False, False, 0.5, 0),
        (False, True, None, 0),
        (False, True, float("nan"), 0),
        (False, True, 0.5, 1),
    ],
)
def test_map_saturation_updates_only_for_accepted_epoch(recovered, validated, fitness, expected_updates):
    block = UltraOptimizedMoE(8, 8, num_experts=2, top_k=1, router_reduction=2, router_pool_scale=1, num_groups=1)
    block.map_saturation_scheduler = MapSaturationScheduler(MapSaturationSchedulerConfig(window_size=1))
    trainer = types.SimpleNamespace(
        _has_moe=True,
        args=types.SimpleNamespace(moe_map_saturation_enabled=True),
        model=torch.nn.Sequential(block),
        fitness=fitness,
    )

    BaseTrainer._finalize_moe_map_saturation_epoch(trainer, recovered=recovered, validated=validated)

    assert len(block.map_saturation_scheduler.map_history) == expected_updates


def test_map_saturation_config_validation():
    with pytest.raises(ValueError, match="window_size must be > 0"):
        MapSaturationSchedulerConfig(window_size=0)
    with pytest.raises(ValueError, match="decay_factor must be in \\(0, 1\\)"):
        MapSaturationSchedulerConfig(decay_factor=1.0)
    with pytest.raises(ValueError, match="decay_factor must be in \\(0, 1\\)"):
        MapSaturationSchedulerConfig(decay_factor=0.0)
    with pytest.raises(ValueError, match="min_scale must be >= 0"):
        MapSaturationSchedulerConfig(min_scale=-0.1)
    with pytest.raises(ValueError, match="saturation_threshold must be >= 0"):
        MapSaturationSchedulerConfig(saturation_threshold=-0.001)


def test_moeloss_map_saturation_scheduler_apply():
    torch.manual_seed(0)
    logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]]).repeat(8, 1).requires_grad_(True)
    probs = torch.softmax(logits, dim=1)
    indices = torch.zeros(8, 2, dtype=torch.long)
    loss_fn = MoELoss(
        num_experts=4,
        top_k=2,
        z_loss_coeff=0.0,
        balance_loss_coeff=1.0,
    )
    # Inject a fake scheduler with scale=0.5
    from ultralytics.nn.modules.moe.scheduler import MapSaturationScheduler, MapSaturationSchedulerConfig
    loss_fn.map_saturation_scheduler = MapSaturationScheduler(MapSaturationSchedulerConfig(enabled=True))
    loss_fn.map_saturation_scheduler.saturation_scale = 0.5
    loss_fn.map_saturation_scheduler.last_state = MapSaturationScheduleState(
        val_map=0.5, saturation_scale=0.5, plateau_detected=False
    )

    out = loss_fn(probs, logits, indices, return_dict=True)

    # With base coeff=1.0 and scale=0.5, effective balance coeff should be 0.5
    assert out["map_saturation_schedule"] is not None
    assert out["map_saturation_schedule"]["saturation_scale"] == pytest.approx(0.5)
    assert out["loss"].requires_grad


def test_map_saturation_scheduler_disabled_passthrough_in_loss():
    torch.manual_seed(0)
    logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]]).repeat(8, 1).requires_grad_(True)
    probs = torch.softmax(logits, dim=1)
    indices = torch.zeros(8, 2, dtype=torch.long)
    loss_fn = MoELoss(
        num_experts=4,
        top_k=2,
        z_loss_coeff=0.0,
        balance_loss_coeff=2.0,
    )
    from ultralytics.nn.modules.moe.scheduler import MapSaturationScheduler, MapSaturationSchedulerConfig
    loss_fn.map_saturation_scheduler = MapSaturationScheduler(MapSaturationSchedulerConfig(enabled=False))
    loss_fn.map_saturation_scheduler.last_state = MapSaturationScheduleState(
        val_map=0.5, saturation_scale=1.0, plateau_detected=False
    )

    out = loss_fn(probs, logits, indices, return_dict=True)
    # disabled -> passthrough, balance coeff should remain 2.0
    assert out["map_saturation_schedule"] is None or out["map_saturation_schedule"]["saturation_scale"] == 1.0
