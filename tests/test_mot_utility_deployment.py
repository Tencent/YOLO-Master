from __future__ import annotations

import pytest
import torch

from ultralytics.nn.modules.mot import UtilityRouterDeployment
from ultralytics.nn.modules.mot.router import _MoTRouter


def _router_with_bias(bias: list[float]) -> _MoTRouter:
    router = _MoTRouter(8, num_experts=3, top_k=2, use_spatial=False).eval()
    with torch.no_grad():
        router.router[-1].weight.zero_()
        router.router[-1].bias.copy_(torch.tensor(bias))
    return router


def test_utility_deployment_blends_probabilities_and_restores_baseline():
    baseline = _router_with_bias([2.0, 1.0, 0.0])
    utility = _router_with_bias([0.0, 1.0, 2.0])
    x = torch.randn(2, 8, 4, 4)
    natural_weights, _ = baseline(x)

    with UtilityRouterDeployment(baseline, utility.state_dict(), alpha=1.0):
        utility_weights, utility_indices = baseline(x)
        assert utility_indices[:, 0].unique().item() == 2
        assert not torch.allclose(utility_weights, natural_weights)

    restored_weights, restored_indices = baseline(x)
    assert restored_indices[:, 0].unique().item() == 0
    assert torch.allclose(restored_weights, natural_weights)


def test_utility_deployment_drives_adaptive_k_and_restores_configuration():
    baseline = _router_with_bias([2.0, 1.0, 0.0])
    utility = _router_with_bias([0.0, 1.0, 2.0])
    x = torch.randn(2, 8, 4, 4)

    with UtilityRouterDeployment(
        baseline,
        utility.state_dict(),
        alpha=1.0,
        adaptive_k=True,
        adaptive_k_threshold=0.6,
    ):
        weights, indices = baseline(x)
        assert baseline.last_selected_k.tolist() == [1, 1]
        assert (indices[:, 1] == -1).all()
        assert torch.equal((weights > 0).sum(dim=1), torch.ones_like(weights[:, 0], dtype=torch.long))

    assert baseline.adaptive_k is False
    assert baseline.adaptive_k_threshold == pytest.approx(0.5)


def test_utility_deployment_restores_hook_after_exception_and_rejects_training():
    baseline = _router_with_bias([2.0, 1.0, 0.0])
    utility = _router_with_bias([0.0, 1.0, 2.0])

    with pytest.raises(RuntimeError, match="stop"), UtilityRouterDeployment(
        baseline,
        utility.state_dict(),
        alpha=0.5,
    ):
        raise RuntimeError("stop")
    assert not baseline._forward_hooks

    baseline.train()
    with pytest.raises(RuntimeError, match="evaluation-only"), UtilityRouterDeployment(
        baseline,
        utility.state_dict(),
        alpha=0.5,
    ):
        pass


def test_utility_deployment_restores_scene_head_architecture_from_state():
    baseline = _router_with_bias([2.0, 1.0, 0.0])
    utility = _router_with_bias([0.0, 1.0, 2.0])
    utility.enable_scene_aware(hidden_dim=7)

    deployment = UtilityRouterDeployment(baseline, utility.state_dict(), alpha=0.5)

    assert baseline.scene_projector is None
    assert deployment.utility_router.scene_aware is True
    assert deployment.utility_router.scene_hidden_dim == 7


def test_utility_deployment_restores_global_head_architecture_from_state():
    baseline = _router_with_bias([2.0, 1.0, 0.0])
    utility = _router_with_bias([0.0, 1.0, 2.0])
    utility.enable_global_utility_head(hidden_dim=9)

    deployment = UtilityRouterDeployment(baseline, utility.state_dict(), alpha=0.5)

    assert getattr(baseline, "utility_projector", None) is None
    assert deployment.utility_router.utility_projector[0].out_features == 9


def test_utility_deployment_drift_guard_falls_back_to_baseline_distribution():
    baseline = _router_with_bias([4.0, 0.0, -4.0])
    utility = _router_with_bias([-4.0, 0.0, 4.0])
    features = torch.randn(2, 8, 4, 4)
    baseline_weights, baseline_indices = baseline(features)

    with UtilityRouterDeployment(
        baseline,
        utility.state_dict(),
        alpha=1.0,
        max_batch_router_kl=0.01,
    ) as deployment:
        guarded_weights, guarded_indices = baseline(features)

    assert deployment.drift_guard_triggered is True
    assert deployment.last_mean_router_kl > 0.01
    assert torch.allclose(guarded_weights, baseline_weights)
    assert torch.equal(guarded_indices, baseline_indices)
