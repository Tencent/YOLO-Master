"""Regressions for capacity accounting and cross-runtime expert selection."""

import inspect
import io

import numpy as np
import pytest
import torch
from torch import nn

from ultralytics.nn.modules._numeric import deterministic_topk_indices
from ultralytics.nn.modules.moa import MoABlock
from ultralytics.nn.modules.moe.routers import BaseRouter
from ultralytics.nn.modules.mot import MoTBlock
from ultralytics.nn.modules.topk_contract import DEFAULT_DETERMINISTIC_TOPK, LEGACY_PRIORITY_BIAS_TOPK
from ultralytics.utils.torch_utils import TORCH_1_13


def _onnx_export_kwargs(opset):
    """Use the legacy tracer without passing new keywords to older PyTorch."""
    kwargs = {"opset_version": opset}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    return kwargs


def test_capacity_counts_experts_across_topk_positions():
    """An expert changing rank must not receive a fresh capacity allocation."""
    router = BaseRouter(num_experts=3, top_k=2, capacity_factor=0.5)
    logits = torch.tensor([[9.0, 8.0, 0.0], [8.0, 9.0, 0.0], [9.0, 8.0, 0.0]])
    weights, indices, info = router._process_logits(logits, noise_std=0.0, training=True)

    assert info["capacity_limit"] == 1
    for expert in range(router.num_experts):
        accepted = (indices == expert) & ~info["overflow_mask"]
        assert int(accepted.sum()) <= info["capacity_limit"]
    assert info["overflow_count"] == 4
    assert info["fallback_count"] == 2
    assert torch.equal(info["token_overflow_mask"], torch.tensor([False, True, True]))
    # The emergency default-expert fallback is allowed beyond normal capacity.
    assert torch.equal(indices[1:, 0], torch.zeros(2, dtype=torch.long))
    assert torch.equal(weights[1:], torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
    assert torch.allclose(weights.sum(1), torch.ones(3))


def test_capacity_without_overflow_keeps_original_topk_weights():
    """Enabling a nonbinding capacity limit must leave routing unchanged."""
    router = BaseRouter(num_experts=3, top_k=2, capacity_factor=3.0)
    logits = torch.tensor([[9.0, 8.0, 0.0], [8.0, 9.0, 0.0]])
    expected_weights, expected_indices, _ = router._process_logits(logits, noise_std=0.0, training=False)
    weights, indices, info = router._process_logits(logits, noise_std=0.0, training=True)

    assert torch.allclose(weights, expected_weights)
    assert torch.equal(indices, expected_indices)
    assert info["overflow_count"] == 0
    assert info["fallback_count"] == 0
    assert not info["overflow_mask"].any()
    assert not info["token_overflow_mask"].any()


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_mot_image_routing_all_topk_shapes(training, top_k):
    """Image-level routes retain singleton spatial axes, including K=E."""
    block = MoTBlock(24, num_heads=3, top_k=top_k, use_spatial_router=False).train(training)
    x = torch.randn(2, 24, 3, 5)
    weights, indices = block.router(x)
    assert weights.shape == (2, block.num_experts, 1, 1)
    assert indices.shape == (2, top_k, 1, 1)
    output, aux = block(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert torch.isfinite(aux)


class _MoTExportOutput(nn.Module):
    """Expose block output and routing weights in the same export artifact."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return self.block(x)[0], self.block.router(x)[0]


@pytest.mark.parametrize("backend", ["torchscript", "onnx"])
@pytest.mark.parametrize("use_spatial", [False, True])
@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("tie_break", [DEFAULT_DETERMINISTIC_TOPK, LEGACY_PRIORITY_BIAS_TOPK])
def test_mot_near_tie_exports_preserve_expert_selection(backend, use_spatial, top_k, tie_break):
    """Near-ties inside the deadband must select the same experts after export."""
    if backend == "onnx" and not TORCH_1_13:
        pytest.skip("Full MoT ONNX parity requires opset 17 (PyTorch >= 1.13), including GridSample support")
    torch.manual_seed(0)
    block = MoTBlock(24, num_heads=3, top_k=top_k, use_spatial_router=use_spatial).eval()
    block.router.route_tie_break = tie_break
    with torch.no_grad():
        bias = (
            torch.tensor([0.3333325, 0.3333333, 0.3333341]).log()
            if tie_break == LEGACY_PRIORITY_BIAS_TOPK
            else torch.tensor([0.0, 1e-6, 2e-6])
        )
        block.router.router[-1].bias.copy_(bias)
    model = _MoTExportOutput(block).eval()
    x = torch.randn(2, 24, 4, 4)
    with torch.no_grad():
        expected_output, expected_weights = model(x)
        # Both fixtures choose the lowest IDs under their configured tie policy.
        # The legacy fixture would instead select experts 1/2 under the default deadband.
        assert torch.all(expected_weights[:, :top_k] > 0)
        assert torch.count_nonzero(expected_weights[:, top_k:]) == 0
        if backend == "torchscript":
            traced = torch.jit.trace(model, x, check_trace=False)
            actual_output, actual_weights = traced(x)
        else:
            pytest.importorskip("onnx")
            ort = pytest.importorskip("onnxruntime")
            buffer = io.BytesIO()
            torch.onnx.export(
                model,
                x,
                buffer,
                input_names=["input"],
                output_names=["output", "weights"],
                **_onnx_export_kwargs(17),
            )
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(buffer.getvalue(), sess_options=options, providers=["CPUExecutionProvider"])
            output, weights = session.run(None, {"input": x.numpy()})
            actual_output, actual_weights = torch.from_numpy(output), torch.from_numpy(weights)
    assert torch.equal(actual_weights > 0, expected_weights > 0)
    np.testing.assert_allclose(actual_weights.numpy(), expected_weights.numpy(), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(actual_output.numpy(), expected_output.numpy(), atol=1e-4, rtol=1e-4)


class _LegacyTopKIndices(nn.Module):
    """Export the legacy policy independently of attention/GridSample operators."""

    def __init__(self, top_k, tolerance):
        super().__init__()
        self.top_k = top_k
        self.tolerance = tolerance

    def forward(self, scores):
        return deterministic_topk_indices(
            scores, self.top_k, tie_tolerance=self.tolerance, tie_break=LEGACY_PRIORITY_BIAS_TOPK
        )


@pytest.mark.parametrize("backend", ["torchscript", "onnx"])
@pytest.mark.parametrize("top_k", [1, 2, 4])
@pytest.mark.parametrize("tolerance", [0.0, 1e-6])
def test_legacy_topk_export_matches_stable_sort_for_negative_scores_and_ties(backend, top_k, tolerance):
    """Traceable repeated maxima must preserve stable sort, including exact ties."""
    model = _LegacyTopKIndices(top_k, tolerance)
    scores = torch.tensor([[-1.0, -1.0, -2.0, -3.0], [-3.0, -2.0, -1.0, -1.0], [-2.0, 0.0, 0.0, -2.0]])
    ranking = scores - torch.arange(scores.shape[1], dtype=scores.dtype) * tolerance
    expected = np.argsort(-ranking.numpy(), axis=1, kind="stable")[:, :top_k]
    np.testing.assert_array_equal(model(scores).numpy(), expected)
    if backend == "torchscript":
        artifact = torch.jit.trace(model, scores, check_trace=False)
        actual = artifact(scores).numpy()
    else:
        pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")
        buffer = io.BytesIO()
        torch.onnx.export(
            model, scores, buffer, input_names=["scores"], output_names=["indices"], **_onnx_export_kwargs(12)
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(buffer.getvalue(), sess_options=options, providers=["CPUExecutionProvider"])
        actual = session.run(None, {"scores": scores.numpy()})[0]
    np.testing.assert_array_equal(actual, expected)


def test_moa_default_constructor_accepts_default_head_count():
    block = MoABlock(24).eval()
    x = torch.randn(1, 24, 3, 5)
    with torch.no_grad():
        output = block(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
