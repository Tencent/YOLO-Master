"""Routing diagnostics, capability declarations, and wrapper propagation tests."""

import pytest
import torch
import torch.nn as nn

from ultralytics.nn.modules.moa import C2fMoA, MoABlock
from ultralytics.nn.modules.moa.router import _moa_router_aux_loss
from ultralytics.nn.modules.moe.modules import ES_MOE
from ultralytics.nn.modules.mot import MoTBlock
from ultralytics.nn.modules.routing_protocol import global_routing_metrics
from ultralytics.utils.errors import MoERouterError


def test_nonfinite_moa_aux_preserves_finite_graph_and_reports_boundary():
    weights = torch.full((1, 3, 2, 2), 1.0 / 3.0, requires_grad=True)
    logits = torch.full((1, 3, 2, 2), float("nan"))

    loss, diagnostics = _moa_router_aux_loss(weights, logits, 0.01, return_diagnostics=True)

    assert torch.isfinite(loss)
    assert loss.requires_grad
    assert diagnostics["first_nonfinite_boundary"] == "router_logits"
    assert diagnostics["logits_nonfinite_count"] == logits.numel()
    loss.backward()
    assert weights.grad is not None
    assert torch.isfinite(weights.grad).all()


def test_moa_aux_ddp_local_jacobian_is_world_size_invariant(monkeypatch):
    """Averaged mock-DDP gradients must match the single-process global objective."""
    rank_inputs = [torch.tensor([0.2, -0.1]), torch.tensor([0.5, 0.3])]
    world_size = len(rank_inputs)
    global_weights = torch.cat([torch.softmax(values, dim=0).view(1, 2, 1, 1) for values in rank_inputs], dim=0)
    global_logits = torch.zeros_like(global_weights)
    reference_values = torch.cat(rank_inputs).requires_grad_()
    reference_weights = torch.stack(
        [torch.softmax(reference_values[i : i + 2], dim=0) for i in range(0, reference_values.numel(), 2)]
    ).view(world_size, 2, 1, 1)
    reference = _moa_router_aux_loss(reference_weights, global_logits, 1.0)
    reference.backward()
    expected = sum(reference_values.grad.view(world_size, 2), start=torch.zeros(2))

    global_sum_mean = global_weights.sum(dim=(0, 2, 3)) / world_size
    global_count_mean = torch.tensor(1.0)
    rank_gradients = []
    for values in rank_inputs:
        local_values = values.clone().requires_grad_()
        local_weights = torch.softmax(local_values, dim=0).view(1, 2, 1, 1)
        reductions = iter((global_sum_mean, global_count_mean))
        monkeypatch.setattr("ultralytics.nn.modules.moa.router._all_reduce_mean", lambda _: next(reductions).clone())
        local_loss = _moa_router_aux_loss(local_weights, torch.zeros_like(local_weights), 1.0, reduce_ddp=True)
        local_loss.backward()
        rank_gradients.append(local_values.grad)

    ddp_averaged = torch.stack(rank_gradients).mean(dim=0)
    assert torch.allclose(ddp_averaged, expected, atol=1e-6, rtol=1e-6)


def test_moa_block_snapshot_keeps_pre_fallback_nonfinite_diagnostics(monkeypatch):
    block = MoABlock(24, num_heads=3).train()

    def bad_router(x, return_logits=False):
        weights = x[:, :1].repeat(1, 3, 1, 1) * 0.0 + (1.0 / 3.0)
        logits = torch.full_like(weights, float("nan"))
        return (weights, logits) if return_logits else weights

    monkeypatch.setattr(block.router, "forward", bad_router)
    _ = block(torch.randn(1, 24, 4, 4, requires_grad=True))

    diagnostics = block.last_routing_snapshot["finite_diagnostics"]
    assert diagnostics["first_nonfinite_boundary"] == "router_logits"
    assert diagnostics["logits_finite"] is False
    assert torch.isfinite(block.aux_loss)


def test_es_moe_preserves_router_failure_diagnostics():
    module = ES_MOE(16, 16, num_experts=3, top_k=1).train()
    with torch.no_grad():
        module.routing.routing_network[-1].bias.fill_(float("nan"))

    with pytest.raises(MoERouterError, match="internal output"):
        module(torch.randn(1, 16, 4, 4))

    diagnostics = module.last_routing_diagnostics
    assert diagnostics["first_nonfinite_boundary"] == "router_logits"
    assert diagnostics["logits_finite"] is False


def test_routed_modules_declare_sparse_export_boundary():
    moa = MoABlock(24, num_heads=3).export_capabilities()
    mot = MoTBlock(24, num_heads=3, top_k=2).export_capabilities()
    mot_dense = MoTBlock(24, num_heads=3, top_k=2, export_masked=False).export_capabilities()
    moe = ES_MOE(16, 16, num_experts=3, top_k=1).export_capabilities()

    assert moa["routing_kind"] == "moa"
    assert moa["eager_sparse_dispatch"] is False
    assert mot["eager_sparse_dispatch"] is True
    assert mot["onnx_sparse_dispatch"] is False
    assert mot["torchscript_trace_sparse_dispatch"] is False
    assert mot["exact_sparse_export"] is False
    assert mot["export_router_weights"] == "masked_topk"
    assert "masked" in mot["sparse_export_limitation"].lower()
    assert mot_dense["export_router_weights"] == "dense_softmax"
    assert "dense" in mot_dense["sparse_export_limitation"].lower()
    assert moe["eager_sparse_dispatch"] is True
    assert moe["onnx_sparse_dispatch"] is False
    assert moe["torchscript_trace_sparse_dispatch"] is False
    assert moe["exact_sparse_export"] is False
    assert "dense" in moe["sparse_export_limitation"].lower()


def test_global_routing_metrics_marks_local_usage_without_process_group():
    result = global_routing_metrics({"expert_usage": torch.tensor([0.75, 0.25])})
    assert result["usage_scope"] == "rank_local"
    assert result["global_usage_available"] is False
    assert torch.allclose(result["global_expert_usage"], torch.tensor([0.75, 0.25]))
    assert result["global_gini"] == pytest.approx(0.25)


def test_global_routing_metrics_reduces_detached_usage(monkeypatch):
    calls = []

    monkeypatch.setattr("torch.distributed.is_available", lambda: True)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 2)
    monkeypatch.setattr("torch.distributed.get_backend", lambda: "gloo")

    def reduce(value, op=None):
        calls.append(value)
        value.add_(torch.tensor([0.25, 0.75]))

    monkeypatch.setattr("torch.distributed.all_reduce", reduce)
    result = global_routing_metrics({"expert_usage": torch.tensor([0.75, 0.25])})
    assert len(calls) == 1
    assert result["global_usage_available"] is True
    assert torch.allclose(result["global_expert_usage"], torch.tensor([0.5, 0.5]))
    assert result["global_gini"] == pytest.approx(0.0)


def test_moa_sparse_inference_keeps_one_group_and_renormalizes(monkeypatch):
    block = MoABlock(24, num_heads=3, inference_sparse_threshold=0.4, shortcut=False).eval()
    weights = torch.tensor([0.8, 0.1, 0.1]).view(1, 3, 1, 1).expand(2, 3, 4, 4)
    monkeypatch.setattr(block.router, "forward", lambda x, return_logits=False: (weights, weights.log()))
    block.fusion = nn.Identity()
    block.ffn = nn.Identity()
    calls = [0, 0, 0]
    for idx, name in enumerate(("local_head", "region_head", "global_head")):
        head = getattr(block, name)
        original = head.forward

        def counted(x, *, _idx=idx, _original=original):
            calls[_idx] += 1
            return torch.ones_like(x)

        monkeypatch.setattr(head, "forward", counted)

    output = block(torch.randn(2, 24, 4, 4))
    snapshot = block.routing_snapshot()

    assert output.shape == (2, 24, 4, 4)
    assert torch.allclose(output, torch.ones_like(output))
    assert calls == [1, 0, 0]
    assert snapshot["executed_groups"] == 1
    assert snapshot["approximation_error"] > 0


def test_c2f_moa_forwards_sparse_inference_configuration():
    module = C2fMoA(48, 48, n=2, num_heads=3, sparse_inference=True, sparse_inference_threshold=0.3)

    assert all(block.sparse_inference for block in module.m)
    assert all(block.sparse_inference_threshold == 0.3 for block in module.m)


def test_moa_sparse_inference_retains_a_group_when_threshold_excludes_all(monkeypatch):
    block = MoABlock(24, num_heads=3, sparse_inference=True, sparse_inference_threshold=0.5).eval()
    weights = torch.full((1, 3, 4, 4), 1.0 / 3.0)
    monkeypatch.setattr(block.router, "forward", lambda x, return_logits=False: (weights, weights.log()))
    calls = [0, 0, 0]
    for index, name in enumerate(("local_head", "region_head", "global_head")):
        head = getattr(block, name)
        original = head.forward

        def counted(x, *, _index=index, _original=original):
            calls[_index] += 1
            return _original(x)

        monkeypatch.setattr(head, "forward", counted)

    output = block(torch.randn(1, 24, 4, 4))

    assert torch.isfinite(output).all()
    assert sum(calls) == 1
    assert block.routing_snapshot()["executed_groups"] == 1


def test_c2f_moa_propagates_sequential_head_configuration():
    module = C2fMoA(48, 48, n=3, num_heads=3, sequential_heads=True)

    assert len(module.m) == 3
    assert all(block.sequential_heads for block in module.m)
