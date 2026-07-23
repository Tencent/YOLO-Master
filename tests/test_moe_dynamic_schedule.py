"""Issue #52 regression tests for MoE dynamic scheduling and pruning metrics."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from ultralytics.nn.modules.moe.gated import HybridAdaptiveGateMoE
from ultralytics.nn.modules.moe.modules import ES_MOE, OptimizedMOE
from ultralytics.nn.modules.moe.pruning import MoEPruner
from ultralytics.nn.modules.moe.schedule import GiniBalanceScheduler, apply_balance_loss_coeff, usage_gini
from ultralytics.utils import DEFAULT_CFG_DICT
from scripts.moe_pruning_sweep import compare_expert_signatures
from ultralytics.engine.extensions.mixture import MixtureRuntimeController


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


def _gini_controller(tmp_path, *, mode="gini"):
    model = nn.Sequential(ES_MOE(8, 8, num_experts=3, top_k=2))
    trainer = SimpleNamespace(
        model=model,
        ema=SimpleNamespace(ema=copy.deepcopy(model)),
        args=SimpleNamespace(
            moe_dynamic_schedule=mode,
            moe_balance_loss=1.0,
            moe_dynamic_gini_target=0.25,
            moe_dynamic_gini_alpha=1.0,
            moe_dynamic_gini_beta=0.8,
            moe_dynamic_balance_min=0.5,
            moe_dynamic_balance_max=2.0,
        ),
        _has_moe=True,
        save_dir=tmp_path,
        epoch=0,
        start_epoch=0,
    )
    controller = MixtureRuntimeController(trainer)
    controller._configure_gini_schedule()
    return controller, trainer


def test_gini_runtime_accumulates_epoch_updates_model_ema_and_trace(tmp_path):
    controller, trainer = _gini_controller(tmp_path)
    block = trainer.model[0]
    assert block._moe_force_snapshot is True

    block.last_routing_snapshot = {"expert_usage": torch.tensor([1.0, 0.0, 0.0])}
    assert controller.collect_routing_usage(batch_weight=4) == 1
    block.last_routing_snapshot = {"expert_usage": torch.tensor([0.5, 0.5, 0.0])}
    assert controller.collect_routing_usage(batch_weight=2) == 1

    assert controller.finalize_epoch(recovered=False, validated=True) >= 1
    state = trainer.model._moe_gini_schedule_state
    assert state["mean_gini"] > 0.25
    assert state["balance_loss_coeff"] > 1.0
    assert trainer.model[0].balance_loss_coeff == pytest.approx(state["balance_loss_coeff"])
    assert trainer.ema.ema[0].balance_loss_coeff == pytest.approx(state["balance_loss_coeff"])
    assert trainer.ema.ema._moe_gini_schedule_state == state

    trace = tmp_path / "moe_dynamic_schedule.csv"
    lines = trace.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "routing_observations" in lines[0]


def test_gini_runtime_recovery_rejects_epoch_and_resume_restores_ema(tmp_path):
    controller, trainer = _gini_controller(tmp_path)
    trainer.model[0].last_routing_snapshot = {"expert_usage": torch.tensor([1.0, 0.0, 0.0])}
    controller.collect_routing_usage(batch_weight=4)
    assert controller.finalize_epoch(recovered=True, validated=True) == 0
    assert not hasattr(trainer.model, "_moe_gini_schedule_state")
    assert not (tmp_path / "moe_dynamic_schedule.csv").exists()

    controller.begin_epoch(1)
    trainer.model[0].last_routing_snapshot = {"expert_usage": torch.tensor([1.0, 0.0, 0.0])}
    controller.collect_routing_usage(batch_weight=4)
    controller.finalize_epoch(recovered=False, validated=True)
    saved_ema = copy.deepcopy(trainer.ema.ema)

    resumed = SimpleNamespace(model=saved_ema, args=trainer.args, _has_moe=True)
    restored = MixtureRuntimeController(resumed)
    restored._configure_gini_schedule()
    assert restored._gini_scheduler.ema == pytest.approx(saved_ema._moe_gini_schedule_state["ema_gini"])
    assert saved_ema[0].balance_loss_coeff == pytest.approx(saved_ema._moe_gini_schedule_state["balance_loss_coeff"])


def test_gini_runtime_rejects_unknown_schedule(tmp_path):
    controller, trainer = _gini_controller(tmp_path)
    trainer.args.moe_dynamic_schedule = "unknown"
    with pytest.raises(ValueError, match="Unsupported moe_dynamic_schedule"):
        controller._configure_gini_schedule()


def test_es_moe_get_gflops_reports_nonzero_total():
    """The pruning script can collect ES_MOE FLOPs via get_gflops()."""
    module = ES_MOE(32, 32, num_experts=3, top_k=2)
    gflops = module.get_gflops((1, 32, 16, 16))
    assert isinstance(gflops, dict)
    assert gflops["total_gflops"] > 0
    assert apply_balance_loss_coeff(module, 1.5) >= 1
    assert module.balance_loss_coeff == pytest.approx(1.5)


def test_moe_pruner_usage_weight_score_preserves_usage_default():
    """The optional weighted score distinguishes equally selected experts."""
    stats = [
        SimpleNamespace(hits=10.0, avg_weight=0.2),
        SimpleNamespace(hits=10.0, avg_weight=0.4),
    ]

    usage_pruner = MoEPruner("dummy.pt")
    weighted_pruner = MoEPruner("dummy.pt", importance_mode="usage_weight", keep_top_m=1)

    assert usage_pruner._expert_score(stats[0], 20.0) == pytest.approx(0.5)
    assert usage_pruner._expert_score(stats[1], 20.0) == pytest.approx(0.5)
    assert weighted_pruner._expert_score(stats[0], 20.0) == pytest.approx(0.1)
    assert weighted_pruner._expert_score(stats[1], 20.0) == pytest.approx(0.2)


def test_compare_expert_signatures_detects_lora_structure_mismatch():
    """LoRA recovery should report when a pruned structure is rebuilt."""
    pruned = [("model.3", 2, 2), ("model.6", 2, 2)]
    preserved = [("model.3", 2, 2), ("model.6", 2, 2)]
    rebuilt = [("model.3", 3, 3), ("model.6", 3, 3)]

    assert compare_expert_signatures(pruned, preserved) == ("preserved", "")

    status, note = compare_expert_signatures(pruned, rebuilt)
    assert status == "structure_mismatch"
    assert "reference=2:2/2:2" in note
    assert "candidate=3:3/3:3" in note


def test_gated_router_pruning_updates_all_projection_branches_and_forward():
    pruner = MoEPruner("dummy.pt")
    module = HybridAdaptiveGateMoE(16, 16, num_experts=4, top_k=2, fused_expert_threshold=2)
    pruner.model = SimpleNamespace(model=nn.Sequential(module))
    pruner.pruning_plan = {"0.routing": [0, 2]}

    pruned = pruner._perform_surgery()
    result = pruned(torch.randn(2, 16, 8, 8))
    assert result.shape == (2, 16, 8, 8)
    assert len(pruned[0].fused_experts.expert_projections) == 2
    assert pruned[0].routing.num_experts == 2
    assert pruned[0].routing.global_fc.out_features == 2
    assert pruned[0].routing.local_conv[-1].out_channels == 2


def test_es_moe_pruning_resizes_usage_buffer_and_forward():
    pruner = MoEPruner("dummy.pt")
    module = ES_MOE(8, 8, num_experts=3, top_k=None)
    pruner.model = SimpleNamespace(model=nn.Sequential(module))
    pruner.pruning_plan = {"0.routing": [0, 2]}

    pruned = pruner._perform_surgery()
    result = pruned(torch.randn(2, 8, 8, 8))

    assert result.shape == (2, 8, 8, 8)
    assert len(pruned[0].experts) == 2
    assert pruned[0].routing.num_experts == 2
    assert pruned[0].expert_usage_counts.shape == (2,)


def test_pruning_fails_fast_when_router_projection_is_missing():
    class InvalidMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = 2
            self.top_k = 1
            self.experts = nn.ModuleList([nn.Identity(), nn.Identity()])
            self.routing = nn.Identity()

        def forward(self, x):
            return x

    pruner = MoEPruner("dummy.pt")
    pruner.model = SimpleNamespace(model=nn.Sequential(InvalidMoE()))
    pruner.pruning_plan = {"0.routing": [0]}
    with pytest.raises(RuntimeError, match="router has no expert-output projection"):
        pruner._perform_surgery()
