"""Validate split-ONNX conditional expert execution without pytest."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def _force_router_to_expert_zero(module: ES_MOE | MoTBlock) -> None:
    with torch.no_grad():
        final_layer = (
            module.routing.routing_network[-1] if isinstance(module, ES_MOE) else module.router.router[-1]
        )
        final_layer.weight.zero_()
        final_layer.bias.copy_(torch.tensor([4.0, 0.0, -4.0], dtype=final_layer.bias.dtype))


def _validate_pure_dispatch() -> dict:
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
    assert [expert.batch_sizes for expert in experts] == [[2], [1], [1], []]

    dense_weights = np.full((1, 3, 1, 1), 1 / 3, dtype=np.float32)
    try:
        dispatch_numpy_experts(
            np.ones((1, 1, 1, 1), dtype=np.float32),
            dense_weights,
            [_SpyExpert(1.0) for _ in range(3)],
            top_k=1,
            routing_granularity="sample",
        )
    except DynamicDispatchContractError:
        dense_rejected = True
    else:
        raise AssertionError("dense routing weights were not rejected")
    return {"audit": audit.to_dict(), "dense_router_rejected": dense_rejected}


def _validate_spatial_union_gate() -> dict:
    inputs = np.ones((1, 1, 1, 3), dtype=np.float32)
    weights = np.zeros((1, 3, 1, 3), dtype=np.float32)
    weights[0, 0, 0, 0] = 1.0
    weights[0, 1, 0, 1] = 1.0
    weights[0, 2, 0, 2] = 1.0
    _, audit = dispatch_numpy_experts(
        inputs,
        weights,
        [_SpyExpert(scale) for scale in (1.0, 2.0, 3.0)],
        top_k=1,
        routing_granularity="spatial_union",
    )
    assert audit.dense_fallback_detected is True
    try:
        dispatch_numpy_experts(
            inputs,
            weights,
            [_SpyExpert(scale) for scale in (1.0, 2.0, 3.0)],
            top_k=1,
            routing_granularity="spatial_union",
            require_reduction=True,
        )
    except DynamicDispatchContractError:
        false_claim_rejected = True
    else:
        raise AssertionError("spatial union selected every expert but the reduction gate passed")
    return {"audit": audit.to_dict(), "false_reduction_claim_rejected": false_claim_rejected}


def _validate_portable_topk_tie_policy() -> dict:
    probabilities = np.array([[[[0.4]], [[0.4000005]], [[0.1999995]]]], dtype=np.float32)
    sparse = sparsify_topk_probabilities(probabilities, top_k=1, tie_tolerance=1e-6)
    np.testing.assert_allclose(sparse[:, 0], 1.0)
    np.testing.assert_allclose(sparse[:, 1:], 0.0)
    return {
        "status": "PASS",
        "selected_expert": int(sparse.reshape(3).argmax()),
        "policy": "probability_minus_expert_id_times_tolerance",
        "tie_tolerance": 1e-6,
    }


def _validate_onnx_module(module: ES_MOE | MoTBlock, sample: torch.Tensor, output_dir: Path) -> dict:
    _force_router_to_expert_zero(module)
    with torch.no_grad():
        eager_output = module(sample)
        if isinstance(eager_output, tuple):
            eager_output = eager_output[0]
        expected = eager_output.cpu().numpy()

    manifest_path = export_dynamic_expert_bundle(module, sample, output_dir)
    runtime = ORTDynamicExpertRuntime(manifest_path)
    actual, audit = runtime.run(sample.cpu().numpy(), require_reduction=True)
    max_abs_error = float(np.max(np.abs(actual - expected)))
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-5)
    assert audit.executed_expert_ids == (0,)
    assert runtime.loaded_expert_ids == (0,)
    result = {
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "max_abs_error": max_abs_error,
        "audit": audit.to_dict(),
        "loaded_expert_ids": list(runtime.loaded_expert_ids),
    }
    if isinstance(module, MoTBlock):
        adapter = ORTRouterTorchExpertAdapter(module, manifest_path).eval()
        with torch.no_grad():
            hybrid = adapter(sample)[0].cpu().numpy()
        hybrid_error = float(np.max(np.abs(hybrid - expected)))
        np.testing.assert_allclose(hybrid, expected, rtol=3e-4, atol=3e-5)
        assert adapter.loaded_onnx_expert_ids == ()
        result["ort_router_torch_experts"] = {
            "max_abs_error": hybrid_error,
            "onnx_expert_sessions_loaded": list(adapter.loaded_onnx_expert_ids),
            "execution_summary": adapter.execution_summary(),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory(prefix="yolo_dynamic_expert_") as temporary_directory:
        root = Path(temporary_directory)
        result = {
            "schema_version": 1,
            "status": "PASS",
            "pure_dispatch": _validate_pure_dispatch(),
            "spatial_union_gate": _validate_spatial_union_gate(),
            "portable_topk_tie_policy": _validate_portable_topk_tie_policy(),
            "esmoe": _validate_onnx_module(
                ES_MOE(4, 4, num_experts=3, top_k=1, dynamic_threshold=0.0).eval(),
                torch.randn(2, 4, 6, 6),
                root / "esmoe",
            ),
            "mot_image_router": _validate_onnx_module(
                MoTBlock(
                    8,
                    num_heads=1,
                    top_k=1,
                    use_spatial_router=False,
                    export_masked=True,
                ).eval(),
                torch.randn(2, 8, 4, 4),
                root / "mot",
            ),
        }

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
