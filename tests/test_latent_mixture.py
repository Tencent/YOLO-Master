from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from ultralytics.nn.mixture_loss import build_composite_criterion
from ultralytics.nn.modules import LatentMixture, LatentRouter, MultiScaleLatentMixture
from ultralytics.nn.modules.routing_protocol import (
    anneal_mixture_temperatures,
    clear_aux_records,
    collect_aux_loss,
    iter_aux_records,
)
from ultralytics.nn.tasks import DetectionModel


ROOT = Path(__file__).resolve().parents[1]


def test_latent_router_noise_is_train_only_and_persistent():
    router = LatentRouter(16, 4, noise_std=0.5)
    x = torch.randn(2, 3, 16)

    router.train()
    torch.manual_seed(1)
    _, train_probs_a = router(x)
    torch.manual_seed(2)
    _, train_probs_b = router(x)
    assert not torch.allclose(train_probs_a, train_probs_b)

    router.eval()
    torch.manual_seed(1)
    _, eval_probs_a = router(x)
    torch.manual_seed(2)
    _, eval_probs_b = router(x)
    assert torch.allclose(eval_probs_a, eval_probs_b)
    assert "_noise_std" in router.state_dict()
    assert float(router.state_dict()["_noise_std"]) == 0.5


def test_latent_router_stays_fp32_after_half_conversion():
    router = LatentRouter(16, 4)
    router.half()
    assert {p.dtype for p in router.parameters()} == {torch.float32}
    assert {b.dtype for b in router.buffers()} == {torch.float32}


def test_latent_router_init_std_breaks_uniform_symmetry():
    torch.manual_seed(0)
    router = LatentRouter(16, 4, router_init_std=0.05).eval()
    x = torch.randn(2, 3, 16)

    _, probs = router(x)
    assert not torch.allclose(probs, torch.full_like(probs, 0.25))
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)), atol=1e-6)


def test_latent_mixture_preserves_first_input_when_residual_zero():
    module = LatentMixture([16, 16], 16, residual_init=0.0).eval()
    xs = [torch.randn(2, 16, 8, 8), torch.randn(2, 16, 8, 8)]
    with torch.no_grad():
        y = module(xs)
    assert y.shape == xs[0].shape
    assert torch.allclose(y, xs[0])


def test_latent_mixture_optional_sparse_inference_dispatches_top_k(monkeypatch):
    module = LatentMixture([8, 8], 8, num_experts=4, inference_top_k=2, residual_init=0.1).eval()
    calls = [0] * module.num_experts
    for index, expert in enumerate(module.experts):
        original = expert.forward

        def wrapped(value, *, _index=index, _original=original):
            calls[_index] += 1
            return _original(value)

        monkeypatch.setattr(expert, "forward", wrapped)

    with torch.no_grad():
        module([torch.randn(2, 8, 4, 4), torch.randn(2, 8, 4, 4)])

    assert sum(call > 0 for call in calls) <= 2
    assert module.export_capabilities()["eager_sparse_dispatch"] is True


def test_latent_mixture_publishes_single_train_aux_and_snapshot():
    module = LatentMixture([16, 8], 16, residual_init=0.01, balance_loss_coeff=0.01, router_z_loss_coeff=0.001)
    module.train()
    xs = [torch.randn(2, 16, 8, 8), torch.randn(2, 8, 8, 8)]

    clear_aux_records()
    y = module(xs)
    aux = collect_aux_loss(module, include_kinds=("latent",), device=y.device)
    records = iter_aux_records(module)

    assert y.shape == xs[0].shape
    assert aux.requires_grad
    assert len(records) == 1
    assert records[0][0] is module
    snapshot = module.routing_snapshot()
    assert snapshot["family"] == "latent"
    assert snapshot["top_k"] == snapshot["training_top_k"] == 4
    assert snapshot["inference_top_k"] == 4
    assert snapshot["executed_experts"] == 4
    assert snapshot["configured_top_k"] == 4
    assert snapshot["active_experts_per_sample"].tolist() == [4, 4]
    assert snapshot["batch_expert_union"] == 4
    assert snapshot["kernel_calls"] == 4
    assert snapshot["noise_std"] == 0.0
    assert snapshot["router_init_std"] == 0.0
    assert snapshot["identity_cold_start"] is False
    assert snapshot["residual_gain_magnitude"] == pytest.approx(0.01)
    assert snapshot["router_aux_gradient_enabled"] is True
    assert snapshot["ddp_balance_synced"] is False
    assert snapshot["mean_router_probs"].requires_grad is False
    assert torch.allclose(snapshot["mean_router_probs"].sum(), torch.tensor(1.0), atol=1e-5)


def test_latent_balance_uses_ddp_global_value_with_local_gradient():
    module = LatentMixture([8], 8, num_experts=2, balance_loss_coeff=1.0, router_z_loss_coeff=0.0).train()
    logits = torch.tensor([[1.3862944, 0.0]], requires_grad=True)
    probs = logits.softmax(dim=-1)

    def add_remote_importance(value, op=None):
        value.add_(torch.tensor([0.4, 0.6]))

    with patch("torch.distributed.is_available", return_value=True), patch(
        "torch.distributed.is_initialized", return_value=True
    ), patch("torch.distributed.get_world_size", return_value=2), patch(
        "torch.distributed.get_backend", return_value="gloo"
    ), patch("torch.distributed.all_reduce", side_effect=add_remote_importance):
        aux, balance, z_loss = module._compute_aux(logits, probs)
        aux.backward()
    module._record_routing(logits, probs, aux, balance, z_loss)

    assert balance.item() == pytest.approx(0.04, abs=1e-6)
    assert module._last_ddp_balance_synced is True
    assert module.routing_snapshot()["ddp_balance_synced"] is True
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_latent_identity_cold_start_keeps_router_and_residual_gain_gradients():
    class NativeCriterion:
        def __call__(self, preds, batch):
            return preds.square().mean(), torch.zeros(1, device=preds.device)

    torch.manual_seed(0)
    clear_aux_records(step=1)
    block = LatentMixture([8, 8], 8).train()
    model = nn.ModuleList([block])
    xs = [torch.randn(2, 8, 4, 4), torch.randn(2, 8, 4, 4)]

    output = block(xs)
    loss, _ = build_composite_criterion(model, NativeCriterion())(output, {})
    loss.backward()

    snapshot = block.routing_snapshot()
    assert snapshot["identity_cold_start"] is True
    assert snapshot["residual_gain_magnitude"] == 0.0
    assert snapshot["router_aux_gradient_enabled"] is True
    assert block.residual_gain.grad is not None
    assert block.residual_gain.grad.abs().sum() > 0
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in block.router.parameters())


def test_multiscale_latent_mixture_shapes_and_scale_snapshot():
    module = MultiScaleLatentMixture([8, 16, 32], latent_dim=12, residual_init=0.01).train()
    xs = [
        torch.randn(2, 8, 16, 16),
        torch.randn(2, 16, 8, 8),
        torch.randn(2, 32, 4, 4),
    ]
    clear_aux_records()
    ys = module(xs)
    assert [y.shape for y in ys] == [x.shape for x in xs]
    snapshot = module.routing_snapshot()
    assert snapshot["num_scales"] == 3
    assert tuple(snapshot["scale_mean_probs"].shape) == (3, 4)


def test_yolo26_latent_yaml_builds_and_runs():
    model = DetectionModel(
        ROOT / "ultralytics/cfg/models/26/yolo26-master-latent-n-resinit010.yaml", ch=3, nc=80, verbose=False
    ).eval()
    latent_layers = [m for m in model.modules() if isinstance(m, LatentMixture)]
    assert len(latent_layers) == 3
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 64, 64))
    assert output is not None


def test_yolo26_latent_initperturb_yaml_builds_and_runs():
    model = DetectionModel(
        ROOT / "ultralytics/cfg/models/26/yolo26-master-latent-n-initperturb020.yaml",
        ch=3,
        nc=80,
        verbose=False,
    ).eval()
    latent_layers = [m for m in model.modules() if isinstance(m, LatentMixture)]
    assert len(latent_layers) == 3
    assert all(layer.router.router_init_std == 0.02 for layer in latent_layers)
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 64, 64))
    assert output is not None


def test_yolo26_latent_initperturb_lowtemp_yaml_builds_and_runs():
    model = DetectionModel(
        ROOT / "ultralytics/cfg/models/26/yolo26-master-latent-n-initperturb020-temp025.yaml",
        ch=3,
        nc=80,
        verbose=False,
    ).eval()
    latent_layers = [m for m in model.modules() if isinstance(m, LatentMixture)]
    assert len(latent_layers) == 3
    assert all(layer.router.router_init_std == 0.02 for layer in latent_layers)
    assert all(float(layer.temperature) == pytest.approx(0.25) for layer in latent_layers)
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 64, 64))
    assert output is not None


def test_yolo26_latent_initperturb_anneal_yaml_builds_and_runs():
    model = DetectionModel(
        ROOT / "ultralytics/cfg/models/26/yolo26-master-latent-n-initperturb020-temp05.yaml",
        ch=3,
        nc=80,
        verbose=False,
    ).eval()
    latent_layers = [m for m in model.modules() if isinstance(m, LatentMixture)]
    assert len(latent_layers) == 3
    assert all(layer.router.router_init_std == 0.02 for layer in latent_layers)
    assert all(float(layer.temperature) == pytest.approx(0.5) for layer in latent_layers)
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 64, 64))
    assert output is not None


def test_latent_temperature_anneal_updates_latent_modules():
    module = LatentMixture([16, 16], 16, temperature=0.5)
    updated = anneal_mixture_temperatures(module, factor=0.95, min_temp=0.2)

    assert updated == 1
    assert float(module.temperature) == pytest.approx(0.475)


def test_latent_value_fusion_has_explicit_causal_semantics():
    torch.manual_seed(7)
    first = torch.randn(2, 8, 4, 4)
    auxiliary = torch.randn(2, 8, 4, 4)
    changed = auxiliary.flip(0)

    legacy = LatentMixture([8, 8], 8, residual_init=0.0).eval()
    fused = LatentMixture(
        [8, 8], 8, residual_init=0.0, value_fusion_mode="weighted_sum", value_fusion_weights=[1.0, 1.0]
    ).eval()
    fused.load_state_dict(legacy.state_dict(), strict=False)

    with patch.object(legacy.router, "forward", return_value=(torch.zeros(2, 4), torch.full((2, 4), 0.25))):
        legacy_a = legacy([first, auxiliary])
        legacy_b = legacy([first, changed])
    with patch.object(fused.router, "forward", return_value=(torch.zeros(2, 4), torch.full((2, 4), 0.25))):
        fused_a = fused([first, auxiliary])
        fused_b = fused([first, changed])

    assert torch.allclose(legacy_a, legacy_b)
    assert not torch.allclose(fused_a, fused_b)
    assert fused.routing_snapshot()["value_fusion_mode"] == "weighted_sum"


def test_latent_value_fusion_ablations_are_reproducible():
    torch.manual_seed(8)
    xs = [torch.randn(2, 8, 4, 4), torch.randn(2, 8, 4, 4)]
    shuffled = [xs[0], xs[1].flip(0)]
    first_only = LatentMixture(
        [8, 8], 8, residual_init=0.0, value_fusion_mode="weighted_sum", value_fusion_weights=[1.0, 0.0]
    ).eval()
    equal = LatentMixture(
        [8, 8], 8, residual_init=0.0, value_fusion_mode="weighted_sum", value_fusion_weights=[1.0, 1.0]
    ).eval()
    equal.load_state_dict(first_only.state_dict(), strict=False)

    assert torch.allclose(first_only(xs), first_only(shuffled))
    assert not torch.allclose(equal(xs), equal(shuffled))
    assert torch.allclose(first_only.value_fusion_weights, torch.tensor([1.0, 0.0]))


def test_latent_value_fusion_is_opt_in_and_checkpoint_compatible():
    default = LatentMixture([8, 8], 8)
    restored = LatentMixture([8, 8], 8)
    restored.load_state_dict(default.state_dict(), strict=True)

    assert default.value_fusion_mode == "router_only"
    assert "value_fusion_weights" not in default.state_dict()
    with pytest.raises(ValueError, match="value_fusion_mode"):
        LatentMixture([8, 8], 8, value_fusion_mode="unknown")
