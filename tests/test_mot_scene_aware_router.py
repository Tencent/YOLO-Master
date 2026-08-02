"""Tests for the opt-in scene-aware MoT router."""

from pathlib import Path

import pytest
import torch

from ultralytics.nn.modules.mot import C2fMoT, MoTBlock
from ultralytics.nn.modules.moe.config import apply_mixture_config, resolve_mixture_config
from ultralytics.nn.modules.mot.router import _MoTRouter
from ultralytics.nn.tasks import DetectionModel


ROOT = Path(__file__).resolve().parents[1]


def test_scene_statistics_are_finite_and_differentiable():
    router = _MoTRouter(8, scene_aware=True)
    x = torch.randn(2, 8, 5, 7, requires_grad=True)

    stats = router.compute_scene_stats(x)
    stats.sum().backward()

    assert stats.shape == (2, 3)
    assert torch.isfinite(stats).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_scene_aware_router_zero_init_matches_legacy_path():
    torch.manual_seed(0)
    legacy = _MoTRouter(8, top_k=3, scene_aware=False).eval()
    enhanced = _MoTRouter(8, top_k=3, scene_aware=True).eval()
    enhanced.router.load_state_dict(legacy.router.state_dict())
    x = torch.randn(2, 8, 6, 6)

    legacy_weights, _ = legacy(x)
    enhanced_weights, _ = enhanced(x)

    assert torch.equal(enhanced_weights, legacy_weights)


def test_scene_aware_eval_bypass_matches_base_router_and_reports_policy():
    torch.manual_seed(0)
    base = _MoTRouter(8, top_k=3, scene_aware=False).eval()
    bypass = _MoTRouter(8, top_k=3, scene_aware=True, scene_inference_mode="bypass").eval()
    bypass.router.load_state_dict(base.router.state_dict())
    with torch.no_grad():
        bypass.scene_projector[-1].weight.normal_()
        bypass.scene_projector[-1].bias.fill_(2.0)
    x = torch.randn(2, 8, 6, 6)

    base_weights, _ = base(x)
    bypass_weights, _ = bypass(x)

    assert torch.equal(bypass_weights, base_weights)
    assert bypass.last_scene_applied is False
    assert bypass.last_scene_bypass_reason == "inference_policy_bypass"
    assert bypass.last_scene_stats is None
    assert bypass.last_scene_bias is None


def test_scene_aware_eval_bypass_still_applies_scene_branch_during_training():
    router = _MoTRouter(8, top_k=3, scene_aware=True, scene_inference_mode="bypass").train()
    with torch.no_grad():
        router.scene_projector[-1].weight.normal_()

    weights, _ = router(torch.randn(2, 8, 6, 6))
    weights[:, 0].mean().backward()

    assert router.last_scene_applied is True
    assert router.last_scene_bypass_reason is None
    assert any(parameter.grad is not None for parameter in router.scene_projector.parameters())


def test_scene_inference_mode_rejects_unknown_policy():
    with pytest.raises(ValueError, match="scene_inference_mode"):
        _MoTRouter(8, scene_aware=True, scene_inference_mode="cached")


def test_learned_scene_residual_distinguishes_smooth_and_high_frequency_inputs():
    router = _MoTRouter(4, top_k=3, scene_aware=True).eval()
    with torch.no_grad():
        router.router[-1].weight.zero_()
        router.router[-1].bias.zero_()
        router.scene_projector[0].weight.copy_(torch.eye(3))
        router.scene_projector[0].bias.zero_()
        router.scene_projector[-1].weight.zero_()
        router.scene_projector[-1].weight[0, 0] = 4.0
        router.scene_projector[-1].bias.zero_()

    smooth = torch.ones(1, 4, 8, 8)
    checker = torch.tensor([[0.0, 1.0] * 4, [1.0, 0.0] * 4] * 4).view(1, 1, 8, 8).expand(1, 4, -1, -1)
    smooth_weights, _ = router(smooth)
    checker_weights, _ = router(checker)

    assert checker_weights[:, 0].mean() > smooth_weights[:, 0].mean()
    assert router.last_scene_applied is True


def test_scene_consistency_loss_prefers_matching_expert_distribution():
    router = _MoTRouter(8, top_k=3, scene_aware=True)
    stats = torch.tensor([[4.0, 0.2, 0.1]])
    matching = torch.tensor([[[[0.9]], [[0.05]], [[0.05]]]], requires_grad=True)
    mismatching = torch.tensor([[[[0.05]], [[0.05]], [[0.9]]]], requires_grad=True)

    match_loss = router.scene_consistency_loss(matching, stats)
    mismatch_loss = router.scene_consistency_loss(mismatching, stats)

    assert match_loss < mismatch_loss
    mismatch_loss.backward()
    assert mismatching.grad is not None


def test_c2fmot_plumbs_scene_aware_options_to_children():
    module = C2fMoT(
        32,
        32,
        n=2,
        num_heads=4,
        scene_aware_router=True,
        scene_hidden_dim=6,
        scene_consistency_coeff=0.02,
        scene_inference_mode="bypass",
    )

    assert all(block.router.scene_aware for block in module.m)
    assert all(block.scene_consistency_coeff == 0.02 for block in module.m)
    assert all(block.router.scene_inference_mode == "bypass" for block in module.m)

    module.eval()
    with torch.no_grad():
        _ = module(torch.randn(1, 32, 8, 8))
    assert module.last_routing_snapshot["scene_inference_mode"] == "bypass"
    assert not any(module.last_routing_snapshot["scene_aware_applied"])
    assert set(module.last_routing_snapshot["scene_bypass_reason"]) == {"inference_policy_bypass"}


def test_scene_consistency_component_reaches_scene_projector():
    block = MoTBlock(
        24,
        num_heads=3,
        top_k=3,
        window_size=4,
        n_points=2,
        balance_loss_coeff=0.0,
        router_z_loss_coeff=0.0,
        scene_aware_router=True,
        scene_consistency_coeff=1.0,
    ).train()
    with torch.no_grad():
        block.router.scene_projector[-1].weight.normal_(std=0.1)

    _, aux = block(torch.randn(2, 24, 5, 7))
    aux.backward()

    assert aux.requires_grad and torch.isfinite(aux)
    assert any(parameter.grad is not None for parameter in block.router.scene_projector.parameters())


def test_mot_snapshot_reports_exact_runtime_sparse_metrics():
    block = MoTBlock(24, num_heads=3, top_k=1).eval()
    with torch.no_grad():
        block.router.router[-1].bias.copy_(torch.tensor([3.0, 0.0, -3.0]))
        block(torch.randn(2, 24, 4, 4))
    dispatch = block.routing_snapshot()["dispatch"]

    assert dispatch["token_mask_sparsity"] == pytest.approx(2.0 / 3.0)
    assert dispatch["experts_per_sample"].tolist() == [1, 1]
    assert dispatch["batch_expert_union"] == 1
    assert dispatch["actual_expert_calls"] == 1


def test_scene_aware_master_config_parses_and_runs():
    config = ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-scene-n.yaml"
    model = DetectionModel(str(config), ch=3, nc=80, verbose=False).eval()

    with torch.no_grad():
        output = model(torch.zeros(1, 3, 64, 64))

    assert output is not None
    assert all(block.router.scene_aware for block in model.modules() if isinstance(block, MoTBlock))


def test_scene_aware_yaml_survives_runtime_default_resolution():
    config = ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-scene-n.yaml"
    model = DetectionModel(str(config), ch=3, nc=80, verbose=False)

    resolved = resolve_mixture_config(model=model)
    apply_mixture_config(model, resolved)

    audit = [item for item in resolved.audit if item["kind"] == "mot"]
    assert audit and all(item["sources"]["scene_aware_router"] == "yaml" for item in audit)
    assert all(block.router.scene_aware for block in model.modules() if isinstance(block, MoTBlock))


def test_scene_aware_enable_after_router_move_preserves_device_and_fp32_contract():
    router = _MoTRouter(8, scene_aware=False).to(dtype=torch.float64)

    router.enable_scene_aware()

    reference = next(router.router.parameters())
    assert {parameter.device for parameter in router.scene_projector.parameters()} == {reference.device}
    assert {parameter.dtype for parameter in router.scene_projector.parameters()} == {torch.float32}
    weights, _ = router(torch.randn(1, 8, 4, 4, dtype=torch.float64))
    assert torch.isfinite(weights).all()
