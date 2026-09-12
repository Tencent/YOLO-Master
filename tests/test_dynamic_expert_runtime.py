"""Regression tests for true conditional expert export and host dispatch."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ultralytics.nn.modules.dynamic_runtime import (
    DynamicDispatchContractError,
    ORTDynamicExpertRuntime,
    ORTRouterTorchExpertAdapter,
    dispatch_numpy_experts,
    export_dynamic_expert_bundle,
    sparsify_topk_probabilities,
)
from ultralytics.nn.modules.moe.modules import ES_MOE
from ultralytics.nn.modules.mot import MoTBlock


class _SpyExpert:
    def __init__(self, scale: float):
        self.scale = scale
        self.batch_sizes: list[int] = []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.batch_sizes.append(int(x.shape[0]))
        return x * self.scale


def test_dispatch_invokes_only_selected_sample_experts():
    inputs = np.ones((3, 1, 2, 2), dtype=np.float32)
    weights = np.zeros((3, 4, 1, 1), dtype=np.float32)
    weights[0, 0] = 1.0
    weights[1, 1] = 1.0
    weights[2, 0] = 0.25
    weights[2, 2] = 0.75
    experts = [_SpyExpert(scale) for scale in (1.0, 10.0, 100.0, 1000.0)]

    output, audit = dispatch_numpy_experts(
        inputs,
        weights,
        experts,
        top_k=2,
        routing_granularity="sample",
        require_reduction=True,
    )

    np.testing.assert_allclose(output[:, 0, 0, 0], np.array([1.0, 10.0, 75.25], dtype=np.float32))
    assert experts[0].batch_sizes == [2]
    assert experts[1].batch_sizes == [1]
    assert experts[2].batch_sizes == [1]
    assert experts[3].batch_sizes == []
    assert audit.executed_expert_ids == (0, 1, 2)
    assert audit.selected_sample_expert_pairs == 4
    assert audit.dense_sample_expert_pairs == 12
    assert audit.conditional_execution_observed is True
    assert audit.sample_pair_reduction_ratio == pytest.approx(2 / 3)


def test_dispatch_rejects_dense_router_weights():
    inputs = np.ones((1, 1, 1, 1), dtype=np.float32)
    dense_weights = np.full((1, 3, 1, 1), 1 / 3, dtype=np.float32)

    with pytest.raises(DynamicDispatchContractError, match="not sparse Top-K"):
        dispatch_numpy_experts(
            inputs,
            dense_weights,
            [_SpyExpert(1.0) for _ in range(3)],
            top_k=1,
            routing_granularity="sample",
        )


def test_host_topk_is_sparse_normalized_and_uses_lowest_id_for_ties():
    probabilities = np.array([[[[0.4]], [[0.4000005]], [[0.1999995]]]], dtype=np.float32)
    sparse = sparsify_topk_probabilities(probabilities, top_k=1, tie_tolerance=1e-6)

    np.testing.assert_allclose(sparse[:, 0], 1.0)
    np.testing.assert_allclose(sparse[:, 1:], 0.0)
    np.testing.assert_allclose(sparse.sum(axis=1), 1.0)


def test_spatial_union_refuses_false_reduction_claim():
    inputs = np.ones((1, 1, 1, 3), dtype=np.float32)
    weights = np.zeros((1, 3, 1, 3), dtype=np.float32)
    weights[0, 0, 0, 0] = 1.0
    weights[0, 1, 0, 1] = 1.0
    weights[0, 2, 0, 2] = 1.0

    output, audit = dispatch_numpy_experts(
        inputs,
        weights,
        [_SpyExpert(scale) for scale in (1.0, 2.0, 3.0)],
        top_k=1,
        routing_granularity="spatial_union",
    )
    np.testing.assert_allclose(output, np.array([[[[1.0, 2.0, 3.0]]]], dtype=np.float32))
    assert audit.executed_expert_ids == (0, 1, 2)
    assert audit.conditional_execution_observed is False
    assert audit.dense_fallback_detected is True

    with pytest.raises(DynamicDispatchContractError, match="true conditional execution was not observed"):
        dispatch_numpy_experts(
            inputs,
            weights,
            [_SpyExpert(scale) for scale in (1.0, 2.0, 3.0)],
            top_k=1,
            routing_granularity="spatial_union",
            require_reduction=True,
        )


def _force_router_to_expert_zero(module: ES_MOE | MoTBlock) -> None:
    with torch.no_grad():
        if isinstance(module, ES_MOE):
            final_layer = module.routing.routing_network[-1]
        else:
            final_layer = module.router.router[-1]
        final_layer.weight.zero_()
        final_layer.bias.copy_(torch.tensor([4.0, 0.0, -4.0], dtype=final_layer.bias.dtype))


def test_esmoe_split_onnx_runtime_matches_eager_and_loads_one_expert(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    module = ES_MOE(4, 4, num_experts=3, top_k=1, dynamic_threshold=0.0).eval()
    _force_router_to_expert_zero(module)
    sample = torch.randn(2, 4, 6, 6)
    with torch.no_grad():
        expected = module(sample).cpu().numpy()

    manifest_path = export_dynamic_expert_bundle(module, sample, tmp_path / "esmoe")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["execution_semantics"] == "host_conditional_expert_dispatch"
    assert manifest["masked_dense_allowed"] is False
    assert manifest["routing_granularity"] == "sample"
    assert manifest["router_output_semantics"] == "dense_probabilities_host_topk"
    assert manifest["host_topk_tie_break"] == "probability_minus_expert_id_times_tolerance"
    assert manifest["host_topk_tie_tolerance"] == pytest.approx(1e-6)

    runtime = ORTDynamicExpertRuntime(manifest_path)
    actual, audit = runtime.run(sample.cpu().numpy(), require_reduction=True)

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    assert audit.executed_expert_ids == (0,)
    assert runtime.loaded_expert_ids == (0,)
    assert audit.sample_pair_reduction_ratio == pytest.approx(2 / 3)


def test_mot_image_router_split_onnx_runtime_matches_eager(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch.manual_seed(1)
    module = MoTBlock(
        8,
        num_heads=1,
        top_k=1,
        use_spatial_router=False,
        export_masked=True,
    ).eval()
    _force_router_to_expert_zero(module)
    sample = torch.randn(2, 8, 4, 4)
    with torch.no_grad():
        expected = module(sample)[0].cpu().numpy()

    manifest_path = export_dynamic_expert_bundle(module, sample, tmp_path / "mot")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["routing_granularity"] == "sample"
    assert manifest["compute_reduction_guaranteed_per_sample"] is True

    runtime = ORTDynamicExpertRuntime(manifest_path)
    actual, audit = runtime.run(sample.cpu().numpy(), require_reduction=True)

    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-5)
    assert audit.executed_expert_ids == (0,)
    assert runtime.loaded_expert_ids == (0,)


def test_mot_spatial_router_manifest_does_not_guarantee_reduction(tmp_path):
    pytest.importorskip("onnx")
    module = MoTBlock(8, num_heads=1, top_k=1, use_spatial_router=True, export_masked=True).eval()
    sample = torch.randn(1, 8, 4, 4)

    manifest_path = export_dynamic_expert_bundle(module, sample, tmp_path / "mot_spatial")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["routing_granularity"] == "spatial_union"
    assert manifest["compute_reduction_guaranteed_per_sample"] is False
    assert "may select the union of all experts" in manifest["claim_boundary"]


def test_ort_router_torch_experts_adapter_uses_checkpoint_experts_without_loading_onnx_experts(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch.manual_seed(2)
    module = MoTBlock(8, num_heads=1, top_k=1, use_spatial_router=False).eval()
    _force_router_to_expert_zero(module)
    sample = torch.randn(2, 8, 4, 4)
    with torch.no_grad():
        expected = module(sample)[0]

    manifest_path = export_dynamic_expert_bundle(module, sample, tmp_path / "mot_hybrid")
    adapter = ORTRouterTorchExpertAdapter(module, manifest_path, require_reduction=True).eval()
    with torch.no_grad():
        actual = adapter(sample)[0]

    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
    assert adapter.loaded_onnx_expert_ids == ()
    summary = adapter.execution_summary()
    assert summary["checkpoint_pytorch_experts_executed"] is True
    assert summary["sample_pair_reduction_ratio"] == pytest.approx(2 / 3)
    assert summary["route_location_mismatch_count"] == 0
    assert adapter.runtime.last_dense_routing_probabilities is not None
    assert summary["route_margin_audit"]["available"] is True
    assert summary["route_margin_audit"]["locations"] == sample.shape[0]
    assert summary["route_margin_audit"]["mismatch_locations"] == 0
