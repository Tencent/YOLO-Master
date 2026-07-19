"""Tests for composing native task criteria with routed auxiliary losses."""

import torch
import torch.nn as nn

from ultralytics.nn.mixture_loss import CompositeCriterion, build_composite_criterion
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
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in block.m[0].router.parameters()
    )
