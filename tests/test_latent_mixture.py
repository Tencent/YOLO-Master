from pathlib import Path

import pytest
import torch

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
    assert snapshot["noise_std"] == 0.0
    assert snapshot["router_init_std"] == 0.0
    assert snapshot["mean_router_probs"].requires_grad is False
    assert torch.allclose(snapshot["mean_router_probs"].sum(), torch.tensor(1.0), atol=1e-5)


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
