"""Foundation student latency/export benchmark contracts."""

import json

import pytest

from scripts.foundation_student_latency_benchmark import (
    BENCHMARK,
    FORBIDDEN_TOKENS,
    _summarize_times,
    benchmark_checkpoints,
    gate_c_compare,
    parse_args,
    scan_forbidden_keys,
    scan_onnx_graph,
)


def test_scan_forbidden_keys_matches_teacher_tokens_case_insensitive():
    keys = [
        "model.0.conv.weight",
        "teacher_manager.dinov3.layer.weight",
        "projector.student_proj.weight",
        "Student.backbone.bias",
    ]
    found = scan_forbidden_keys(keys)
    assert "model.0.conv.weight" not in found
    assert "teacher_manager.dinov3.layer.weight" in found
    assert "projector.student_proj.weight" in found


def test_scan_onnx_graph_missing_file_is_explicit(tmp_path):
    result = scan_onnx_graph(tmp_path / "missing.onnx")
    assert result["available"] is False
    assert result["reason"]


def test_scan_onnx_graph_detects_forbidden_and_clean(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper

    def make_graph(node_name: str) -> "onnx.ModelProto":
        node = helper.make_node("Relu", ["x"], ["y"], name=node_name)
        return helper.make_model(
            helper.make_graph(
                [node],
                "g",
                [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 1])],
                [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 1])],
            )
        )

    clean = tmp_path / "clean.onnx"
    onnx.save(make_graph("student_conv"), str(clean))
    clean_result = scan_onnx_graph(clean)
    assert clean_result["available"] is True
    assert clean_result["forbidden"] == []
    assert clean_result["nodes"] == 1

    dirty = tmp_path / "dirty.onnx"
    onnx.save(make_graph("dinov3_teacher_encoder"), str(dirty))
    dirty_result = scan_onnx_graph(dirty)
    assert dirty_result["available"] is True
    assert dirty_result["forbidden"] == ["dinov3_teacher_encoder"]


def test_summarize_times_reports_ms_stats():
    stats = _summarize_times([0.001, 0.002, 0.003, 0.004])
    assert stats["iters"] == 4
    assert stats["mean_ms"] == pytest.approx(2.5)
    assert stats["min_ms"] == pytest.approx(1.0)
    assert stats["max_ms"] == pytest.approx(4.0)
    assert stats["p50_ms"] <= stats["p95_ms"] <= stats["max_ms"]


def _record(params, gflops, name):
    return {"checkpoint": name, "pytorch": {"params": params, "gflops": gflops}}


def test_gate_c_compare_parity_and_mismatch():
    gate = gate_c_compare([_record(100, 2.5, "base.pt"), _record(100, 2.5, "kd.pt")])
    assert gate["checked"] is True
    assert gate["params_match"] is True
    assert gate["gflops_match"] is True
    assert gate["details"][0]["params_delta"] == 0

    gate = gate_c_compare([_record(100, 2.5, "base.pt"), _record(200, 2.5, "fat.pt")])
    assert gate["params_match"] is False
    assert gate["details"][0]["params_delta"] == 100

    gate = gate_c_compare([_record(100, 2.5, "base.pt"), _record(100, 3.5, "slow.pt")])
    assert gate["gflops_match"] is False


def test_gate_c_compare_requires_baseline_pair_and_metrics():
    assert gate_c_compare([_record(100, 2.5, "base.pt")])["checked"] is False
    gate = gate_c_compare([{"checkpoint": "base.pt", "pytorch": {}}, _record(100, 2.5, "kd.pt")])
    assert gate["checked"] is False
    assert "reason" in gate


def test_benchmark_checkpoints_missing_checkpoint_is_explicit(tmp_path):
    report = benchmark_checkpoints(
        [tmp_path / "missing.pt"], imgsz=64, device="cpu", warmup=0, iters=1, export=False, workdir=tmp_path
    )
    assert report["benchmark"] == BENCHMARK
    assert report["accuracy_claim"] is False
    assert report["records"][0]["available"] is False
    assert report["records"][0]["reason"] == "checkpoint not found"
    assert report["forbidden_tokens"] == list(FORBIDDEN_TOKENS)


def test_parse_args_boundaries(tmp_path):
    args = parse_args(["--checkpoints", f"{tmp_path}/a.pt,{tmp_path}/b.pt", "--imgsz", "128"])
    assert len(args.checkpoints) == 2
    assert args.export is True
    with pytest.raises(SystemExit):
        parse_args(["--checkpoints", "a.pt", "--imgsz", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--checkpoints", "a.pt", "--iters", "0"])
    with pytest.raises(SystemExit):
        parse_args([])


def test_report_json_roundtrip_serializable(tmp_path):
    report = benchmark_checkpoints(
        [tmp_path / "missing.pt"], imgsz=64, device="cpu", warmup=0, iters=1, export=False, workdir=tmp_path
    )
    encoded = json.dumps(report, ensure_ascii=False)
    assert json.loads(encoded)["benchmark"] == BENCHMARK
