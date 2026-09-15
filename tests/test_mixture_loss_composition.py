"""Tests for composing native task criteria with routed auxiliary losses."""

import pytest
import torch
from torch import nn

from ultralytics.nn.mixture_loss import CompositeCriterion, build_composite_criterion, compose_native_result
from ultralytics.nn.modules.latent_mixture import LatentMixture
from ultralytics.nn.modules.moa import C2fMoA
from ultralytics.nn.modules.routing_protocol import clear_aux_records


class NativeCriterion:
    def __init__(self):
        self.calls = 0
        self.updates = 0

    def __call__(self, preds, batch):
        self.calls += 1
        return preds.square().mean(), torch.tensor([1.0, 2.0])

    def update(self):
        self.updates += 1


@pytest.mark.parametrize("entrypoint", ["wrapper", "direct"])
@pytest.mark.parametrize("shape", [(), (1,), (3,), (5,)])
def test_vector_native_loss_counts_aux_once_with_correct_gradients(monkeypatch, entrypoint, shape):
    """Trainer reduction must not multiply auxiliary regularization by the task's loss-item count."""
    model = nn.Sequential(C2fMoA(16, 16, n=1, num_heads=3)).train()
    native_loss = torch.ones(shape, requires_grad=True)
    native_items = native_loss.detach().reshape(-1).clone()
    aux = torch.tensor(2.0, requires_grad=True)
    monkeypatch.setattr("ultralytics.nn.mixture_loss._collect_mixture_aux_loss", lambda *args, **kwargs: aux)

    if entrypoint == "wrapper":
        loss, items = CompositeCriterion(model, lambda *args: (native_loss, native_items))(None, {})
    else:
        loss, items = compose_native_result(model, native_loss, native_items)

    torch.testing.assert_close(loss.sum(), native_loss.sum() + aux)
    loss.sum().backward()
    torch.testing.assert_close(native_loss.grad, torch.ones_like(native_loss))
    torch.testing.assert_close(aux.grad, torch.ones_like(aux))
    torch.testing.assert_close(items[:-1], native_items)
    torch.testing.assert_close(items[-1], aux.detach())
    assert not items.requires_grad


def test_dense_direct_composition_preserves_native_vector():
    """Dense models keep the native loss shape, tensors, and gradient path unchanged."""
    model = nn.Linear(2, 2)
    native_loss = torch.ones(3, requires_grad=True)
    native_items = native_loss.detach()
    loss, items = compose_native_result(model, native_loss, native_items)
    assert loss is native_loss
    assert items is native_items


def test_dense_model_keeps_exact_native_criterion():
    model = nn.Sequential(nn.Linear(4, 4))
    native = NativeCriterion()
    assert build_composite_criterion(model, native) is native


def test_routed_model_adds_aux_once_and_appends_log_item():
    clear_aux_records(step=1)
    model = nn.Sequential(C2fMoA(16, 16, n=1, num_heads=3)).train()
    output = model(torch.randn(2, 16, 4, 4))
    native = NativeCriterion()
    criterion = build_composite_criterion(model, native)

    loss, items = criterion(output, {})
    expected_native = output.square().mean()
    aux = model._last_mixture_aux_loss

    assert isinstance(criterion, CompositeCriterion)
    assert native.calls == 1
    assert torch.allclose(loss.detach(), expected_native.detach() + aux)
    assert items.shape == (3,)
    assert torch.allclose(items[-1], aux)
    criterion.update()
    assert native.updates == 1


def test_composite_aux_keeps_router_gradient_connection():
    clear_aux_records(step=2)
    block = C2fMoA(16, 16, n=1, num_heads=3).train()
    model = nn.Sequential(block)
    output = model(torch.randn(2, 16, 4, 4))
    loss, _ = build_composite_criterion(model, NativeCriterion())(output, {})
    loss.backward()

    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in block.m[0].router.parameters()
    )


def test_latent_aux_uses_conservative_default_gain():
    clear_aux_records(step=3)
    block = LatentMixture([8, 8], 8, residual_init=0.01, balance_loss_coeff=0.1, router_z_loss_coeff=0.01).train()
    model = nn.Sequential(block)
    output = block([torch.randn(2, 8, 4, 4), torch.randn(2, 8, 4, 4)])
    native = NativeCriterion()
    criterion = CompositeCriterion(model, native)
    loss, _ = criterion(output, {})

    assert loss.requires_grad
    assert model._last_mixture_aux_loss.detach().abs().item() > 0
    assert loss.detach().abs().item() > 0


@pytest.mark.parametrize("shape", [(), (1,), (3,), (2, 3)])
@pytest.mark.parametrize("entry", ["wrapper", "custom"])
def test_vector_aux_is_added_once_in_value_and_gradient(monkeypatch, shape, entry):
    from ultralytics.nn import mixture_loss

    model = nn.Linear(1, 1)
    native = torch.full(shape, 2.0, requires_grad=True)
    aux = torch.tensor(0.3, requires_grad=True)
    metrics = torch.tensor([1.0, 2.0, 3.0])
    monkeypatch.setattr(mixture_loss, "has_routed_modules", lambda _: True)
    monkeypatch.setattr(mixture_loss, "_collect_mixture_aux_loss", lambda *a, **kw: aux)
    if entry == "wrapper":
        result, logged = CompositeCriterion(model, lambda *args: (native, metrics))(None, {})
    else:
        result, logged = mixture_loss.compose_native_result(model, native, metrics)
    assert result.ndim == 0
    torch.testing.assert_close(result.sum(), native.sum() + aux)
    result.sum().backward()
    torch.testing.assert_close(native.grad, torch.ones_like(native))
    torch.testing.assert_close(aux.grad, torch.ones_like(aux))
    torch.testing.assert_close(logged[:-1], metrics)
    torch.testing.assert_close(logged[-1], aux.detach())
    assert not logged.requires_grad


@pytest.mark.parametrize("batch_size", [1, 8, 64])
@pytest.mark.parametrize("world_size", [1, 2, 6])
def test_aux_preserves_explicit_native_batch_and_ddp_scaling(batch_size, world_size):
    from ultralytics.nn.mixture_loss import _add_aux_once

    # Simulate Trainer's world-size multiplier followed by DDP gradient averaging.
    parameter = torch.tensor(2.0, requires_grad=True)
    ranks, expected = [], []
    for rank in range(world_size):
        native = torch.stack([parameter.square(), parameter * 2, parameter * 3]) * batch_size
        aux = parameter * (rank + 1) * 0.1
        ranks.append(world_size * _add_aux_once(native, aux).sum())
        expected.append(world_size * (native.sum() + aux))
    actual_gradient = torch.autograd.grad(sum(ranks) / world_size, parameter, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(sum(expected) / world_size, parameter)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_empty_native_and_nonscalar_aux_are_rejected():
    from ultralytics.nn.mixture_loss import _add_aux_once

    with pytest.raises(ValueError, match="empty"):
        _add_aux_once(torch.empty(0), torch.tensor(1.0))
    with pytest.raises(ValueError, match="scalar"):
        _add_aux_once(torch.ones(3), torch.ones(2))


def test_custom_dense_path_returns_original_objects():
    from ultralytics.nn.mixture_loss import compose_native_result

    native, items = torch.ones(3), torch.ones(3)
    result, logged = compose_native_result(nn.Linear(1, 1), native, items)
    assert result is native
    assert logged is items
