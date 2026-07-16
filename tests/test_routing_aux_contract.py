"""Regression tests for the canonical routing auxiliary-loss contract."""

import copy

import torch
import torch.nn as nn

from ultralytics.nn.modules.moa import C2fMoA, MoABlock
from ultralytics.nn.modules.moe.modules import MOE_LOSS_REGISTRY
from ultralytics.nn.modules.mot import C2fMoT
from ultralytics.nn.modules.routing_protocol import (
    clear_aux_records,
    collect_aux_loss,
    get_aux_record,
    publish_aux_loss,
    RoutingAuxPublisher,
)
from ultralytics.nn.peft.molora.layer import MoLoRALayer
from ultralytics.utils.loss import _collect_moe_aux_loss


def test_training_publication_is_graph_connected_and_replaces_record():
    clear_aux_records(step=11)
    module = MoABlock(48, num_heads=6).train()
    assert isinstance(module, RoutingAuxPublisher)
    module(torch.randn(1, 48, 4, 4))
    record = get_aux_record(module)
    assert record is not None
    assert record.step == 11
    assert record.training is True
    assert record.value.requires_grad
    first = collect_aux_loss(module, step=11)
    module(torch.randn(1, 48, 4, 4))
    second = collect_aux_loss(module, step=11)
    assert first.requires_grad and second.requires_grad
    assert len(list(module.parameters())) > 0


def test_eval_publication_is_detached_and_not_collected():
    clear_aux_records(step=12)
    module = MoABlock(48, num_heads=6).eval()
    module(torch.randn(1, 48, 4, 4))
    record = get_aux_record(module)
    assert record is not None
    assert record.training is False
    assert not record.value.requires_grad
    total, diagnostics = collect_aux_loss(module, step=12, return_diagnostics=True)
    assert float(total) == 0.0
    assert diagnostics["eval_skipped"] == 1


def test_wrapper_record_covers_children_without_double_counting():
    clear_aux_records(step=13)
    module = C2fMoA(32, 32, n=2, num_heads=6).train()
    module(torch.randn(1, 32, 5, 5))
    total, diagnostics = collect_aux_loss(module, step=13, return_diagnostics=True)
    expected = module.last_aux_loss
    assert torch.allclose(total, expected)
    assert diagnostics["counts_by_kind"]["moa"] == 1
    assert diagnostics["duplicate_skipped"] == 2


def test_stale_step_is_rejected():
    clear_aux_records(step=14)
    module = MoABlock(48, num_heads=6).train()
    value = torch.ones((), requires_grad=True)
    publish_aux_loss(module, value, step=14, kind="moa", training=True)
    total, diagnostics = collect_aux_loss(module, step=15, return_diagnostics=True)
    assert float(total) == 0.0
    assert diagnostics["stale_skipped"] == 1


def test_molora_without_legacy_registry_is_canonical():
    clear_aux_records(step=15)
    MOE_LOSS_REGISTRY.clear()
    base = nn.Linear(8, 8)
    layer = MoLoRALayer(base, r=2, num_experts=2, top_k=1, share_moe_registry=False).train()
    layer(torch.randn(2, 8))
    total = collect_aux_loss(layer, step=15, include_kinds=("molora",))
    assert total.requires_grad
    assert torch.allclose(total, layer.aux_loss)


def test_molora_is_included_once_by_mixture_collector():
    clear_aux_records(step=18)
    base = nn.Linear(8, 8)
    layer = MoLoRALayer(base, r=2, num_experts=2, top_k=1, share_moe_registry=True).train()
    layer(torch.randn(2, 8))
    from ultralytics.utils.loss import _collect_moe_aux_loss

    collected = _collect_moe_aux_loss(layer, torch.device("cpu"))
    assert torch.allclose(collected, layer.aux_loss)


def test_canonical_state_does_not_break_deepcopy():
    clear_aux_records(step=16)
    module = C2fMoT(24, 24, n=1, num_heads=3).train()
    module(torch.randn(1, 24, 4, 4))
    clone = copy.deepcopy(module)
    assert isinstance(clone, C2fMoT)
    assert get_aux_record(clone) is None


def test_legacy_registry_is_only_a_fallback():
    clear_aux_records(step=17)
    module = nn.Linear(2, 2).train()
    value = torch.ones((), requires_grad=True)
    MOE_LOSS_REGISTRY[module] = value
    total = _collect_moe_aux_loss(module, torch.device("cpu"))
    assert torch.allclose(total, value)
