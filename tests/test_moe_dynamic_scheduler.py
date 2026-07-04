import sys
import types

import pytest
import torch

sys.modules.setdefault("seaborn", types.SimpleNamespace(heatmap=lambda *args, **kwargs: None))

from ultralytics.nn.modules.moe.loss import MoELoss
from ultralytics.nn.modules.moe.modules import AdaptiveBalanceController
from ultralytics.nn.modules.moe.scheduler import (
    MoEDynamicScheduler,
    MoEDynamicSchedulerConfig,
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
