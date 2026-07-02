"""Issue #52 regression tests for MoE dynamic scheduling and pruning metrics."""

import pytest

from ultralytics.nn.modules.moe.modules import ES_MOE, OptimizedMOE
from ultralytics.nn.modules.moe.schedule import GiniBalanceScheduler, apply_balance_loss_coeff, usage_gini
from ultralytics.utils import DEFAULT_CFG_DICT


def test_usage_gini_uniform_and_collapsed():
    """Gini is zero for uniform usage and high for collapsed routing."""
    assert usage_gini([0.25, 0.25, 0.25, 0.25]) == pytest.approx(0.0)
    assert usage_gini([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.75)


def test_gini_balance_scheduler_clamps_and_updates_modules():
    """High Gini increases balance loss while clamp bounds are respected."""
    scheduler = GiniBalanceScheduler(base=1.0, target=0.25, alpha=2.0, beta=0.0, min_coeff=0.5, max_coeff=2.0)
    high = scheduler.update(0.75)
    low = scheduler.update(0.0)
    assert high == pytest.approx(2.0)
    assert 0.5 <= low < 1.0

    module = OptimizedMOE(32, 32, num_experts=4, top_k=2)
    updated = apply_balance_loss_coeff(module, 1.5)
    assert updated >= 1
    assert module.balance_loss_coeff == pytest.approx(1.5)
    assert module.moe_loss_fn.balance_loss_coeff == pytest.approx(1.5)


def test_dynamic_schedule_is_disabled_by_default():
    """The dynamic schedule is opt-in for backward compatibility."""
    assert DEFAULT_CFG_DICT["moe_dynamic_schedule"] == "none"


def test_es_moe_get_gflops_reports_nonzero_total():
    """The pruning script can collect ES_MOE FLOPs via get_gflops()."""
    module = ES_MOE(32, 32, num_experts=3, top_k=2)
    gflops = module.get_gflops((1, 32, 16, 16))
    assert isinstance(gflops, dict)
    assert gflops["total_gflops"] > 0
    assert apply_balance_loss_coeff(module, 1.5) >= 1
    assert module.balance_loss_coeff == pytest.approx(1.5)
