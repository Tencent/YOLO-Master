"""Unit tests for the A3 five-family precision harness."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from scripts.a3_precision.manifest import FIVE_FAMILIES, build_lock, load_config, verify_lock, write_lock
from scripts.a3_precision.routing import (
    RouteTensor,
    compare_route_tensors,
    normalize_router_output,
    quantize_dequantize,
    weighted_layer_summary,
)
from scripts.a3_precision.sensitivity import write_validation_subset


def _route(probabilities: torch.Tensor, top_k: int = 2) -> RouteTensor:
    return RouteTensor(
        probabilities=probabilities,
        topk_indices=probabilities.topk(top_k, dim=1).indices,
        top_k=top_k,
        num_experts=probabilities.shape[1],
    )


def test_runner_primes_cudnn_before_importing_ultralytics_dependencies():
    runner = Path("scripts/run_a3_five_family_precision.py").read_text(encoding="utf-8")
    prime_call = runner.index("_prime_torch_cuda_runtime()")
    first_pipeline_import = runner.index("from scripts.a3_mot_trained_routing_drift")
    assert prime_call < first_pipeline_import


def test_quantize_dequantize_int8_is_per_output_channel_and_bounded():
    weights = torch.tensor([[1.0, -0.5], [100.0, -50.0]])
    quantized = quantize_dequantize(weights, "int8", per_channel=True)
    assert quantized.shape == weights.shape
    assert torch.allclose(quantized[:, 0], weights[:, 0])
    assert (weights - quantized).abs().max() <= 100.0 / 127.0
    assert torch.equal(quantize_dequantize(weights, "fp32"), weights)


def test_normalize_router_output_accepts_logits_and_moves_expert_axis():
    logits = torch.tensor([[[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]]])  # [B,T,E]
    route = normalize_router_output({"logits": logits}, num_experts=3, top_k=2)
    assert route.probabilities.shape == (1, 3, 2)
    assert torch.allclose(route.probabilities.sum(dim=1), torch.ones(1, 2))
    assert route.topk_indices.shape == (1, 2, 2)


def test_normalize_router_output_reconstructs_sparse_topk_tuple():
    weights = torch.tensor([[[[0.7]], [[0.3]]]])
    indices = torch.tensor([[[[3]], [[1]]]])
    route = normalize_router_output((weights, indices, None), num_experts=4, top_k=2)
    assert route.probabilities.shape == (1, 4, 1, 1)
    assert route.topk_indices.flatten().tolist() == [3, 1]
    assert torch.allclose(route.probabilities.sum(dim=1), torch.ones(1, 1, 1))


def test_route_metrics_distinguish_exact_set_order_and_top1_flip():
    reference = _route(torch.tensor([[[0.6, 0.5], [0.3, 0.4], [0.1, 0.1]]]))
    candidate = _route(torch.tensor([[[0.3, 0.5], [0.6, 0.1], [0.1, 0.4]]]))
    metrics = compare_route_tensors(reference, candidate)
    assert metrics["tokens"] == 2
    assert metrics["topk_exact_rate"] == pytest.approx(0.5)
    assert metrics["topk_ordered_exact_rate"] == pytest.approx(0.0)
    assert metrics["jaccard_rate"] == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)
    assert metrics["token_flip_rate"] == pytest.approx(0.5)
    assert metrics["top1_flip_rate"] == pytest.approx(0.5)
    assert metrics["sample_flipped"] is True


def test_weighted_layer_summary_uses_token_counts():
    base = {
        "topk_ordered_exact_rate": 1.0,
        "topk_exact_rate": 1.0,
        "jaccard_rate": 1.0,
        "token_flip_rate": 0.0,
        "top1_flip_rate": 0.0,
        "sample_flipped": False,
        "sample_top1_flipped": False,
        "reference_margin_mean": 0.5,
        "reference_margin_p01": 0.1,
        "reference_margin_p05": 0.2,
        "candidate_margin_mean": 0.5,
        "candidate_margin_p01": 0.1,
        "candidate_margin_p05": 0.2,
        "margin_delta_mean": 0.0,
        "probability_mae": 0.0,
        "total_variation": 0.0,
        "jensen_shannon": 0.0,
    }
    rows = [{**base, "tokens": 1}, {**base, "tokens": 3, "topk_exact_rate": 0.0, "sample_flipped": True}]
    summary = weighted_layer_summary(rows)
    assert summary["topk_exact_rate"] == pytest.approx(0.25)
    assert summary["sample_flip_rate"] == pytest.approx(0.5)


def test_five_family_lock_detects_checkpoint_drift(tmp_path: Path):
    calibration = tmp_path / "calibration"
    validation = tmp_path / "validation"
    labels = tmp_path / "labels"
    for directory in (calibration, validation, labels):
        directory.mkdir()
    (calibration / "calib.jpg").write_bytes(b"calibration")
    (validation / "val.jpg").write_bytes(b"validation")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("path: .\n", encoding="utf-8")
    model_rows = []
    for family in FIVE_FAMILIES:
        model_config = tmp_path / f"{family}.yaml"
        checkpoint = tmp_path / f"{family}.pt"
        model_config.write_text("nc: 10\n", encoding="utf-8")
        checkpoint.write_bytes(family.encode())
        model_rows.append(f"  - family: {family}\n    config: {model_config}\n    checkpoint: {checkpoint}")
    config_path = tmp_path / "harness.yaml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                f"experiment:\n  output_dir: {tmp_path / 'out'}",
                "dataset:",
                f"  data_yaml: {data_yaml}",
                f"  calibration_images: {calibration}",
                f"  validation_images: {validation}",
                f"  validation_labels: {labels}",
                "  calibration_samples: 1",
                "  validation_samples: 1",
                "models:",
                *model_rows,
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    lock = build_lock(config)
    assert {row["family"] for row in lock["models"]} == set(FIVE_FAMILIES)
    lock_path = write_lock(config)
    assert verify_lock(config, lock_path)["schema_version"] == 1
    config.models[0].checkpoint.write_bytes(b"drift")
    with pytest.raises(RuntimeError, match="sha256_mismatch"):
        verify_lock(config, lock_path)


def test_discover_sensitivity_groups_finds_structural_roles(tmp_path: Path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    nodes = [
        helper.make_node("Conv", ["x", "w"], ["router_logits"], name="/model.1/router/Conv"),
        helper.make_node("TopK", ["router_logits", "k"], ["values", "indices"], name="/model.1/router/TopK"),
        helper.make_node("MatMul", ["x", "a"], ["attn"], name="/model.2/attn/MatMul"),
        helper.make_node("MatMul", ["x", "b"], ["out"], name="/model.3/proj/MatMul"),
    ]
    graph = helper.make_graph(
        nodes,
        "sensitivity",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 1, 1])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1])],
    )
    path = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph), path)
    from scripts.a3_precision.onnx_backend import discover_sensitivity_groups

    groups = discover_sensitivity_groups(path)
    roles = {row["role"] for row in groups.values()}
    assert {"router", "topk", "attention", "matmul"} <= roles


def test_instrument_topk_outputs_exposes_route_values_and_indices(tmp_path: Path):
    onnx = pytest.importorskip("onnx")
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    node = helper.make_node(
        "TopK",
        ["router_probs", "k"],
        ["route_values", "route_indices"],
        name="/model.1/router/TopK",
        axis=-1,
    )
    graph = helper.make_graph(
        [node],
        "routing_outputs",
        [helper.make_tensor_value_info("router_probs", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("route_values", TensorProto.FLOAT, [1, 2])],
        initializer=[numpy_helper.from_array(np.asarray([2], dtype=np.int64), name="k")],
    )
    source = tmp_path / "source.onnx"
    target = tmp_path / "instrumented.onnx"
    onnx.save(helper.make_model(graph), source)

    from scripts.a3_precision.onnx_backend import instrument_topk_outputs

    specs = instrument_topk_outputs(source, target)
    spec = specs["/model.1/router/TopK"]
    assert spec["num_experts"] == 4
    assert spec["top_k"] == 2
    outputs = {item.name for item in onnx.load(target).graph.output}
    assert {"route_values", "route_indices"} <= outputs


def test_instrument_topk_outputs_falls_back_to_dense_router_softmax(tmp_path: Path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    node = helper.make_node("Softmax", ["router_logits"], ["router_probs"], name="/model.1/router/Softmax", axis=1)
    graph = helper.make_graph(
        [node],
        "dense_routing_outputs",
        [helper.make_tensor_value_info("router_logits", TensorProto.FLOAT, [1, 4, 8, 8])],
        [helper.make_tensor_value_info("router_probs", TensorProto.FLOAT, [1, 4, 8, 8])],
    )
    source = tmp_path / "source.onnx"
    target = tmp_path / "instrumented.onnx"
    onnx.save(helper.make_model(graph), source)

    from scripts.a3_precision.onnx_backend import instrument_topk_outputs

    spec = instrument_topk_outputs(source, target)["/model.1/router/Softmax"]
    assert spec["representation"] == "dense_probabilities"
    assert spec["num_experts"] == 4
    assert spec["top_k"] == 2
    assert spec["indices_output"] is None


def test_locked_validation_yaml_uses_exact_image_list(tmp_path: Path):
    image = tmp_path / "images" / "val" / "sample.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("path: .\ntrain: images/val\nval: images/val\nnames: [object]\n", encoding="utf-8")
    derived = write_validation_subset(data_yaml, [image.resolve()], tmp_path / "locked")
    payload = yaml.safe_load(derived.read_text(encoding="utf-8"))
    image_list = Path(payload["val"])
    assert image_list.read_text(encoding="utf-8").strip() == str(image.resolve())
