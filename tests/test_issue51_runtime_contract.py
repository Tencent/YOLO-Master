"""Contract tests for the issue #51 export and validation protocol.

The tests exercise dependency-light helpers only. Optional runtime bindings
such as MNN and NCNN are imported by the command-line entry points after input
validation, so the contract suite remains runnable on a standard CI worker.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EDGE = ROOT / "examples" / "YOLO-Master-Cross-Platform-Edge-Deployment"


def load_module(name: str, path: Path):
    """Load a script module without importing optional runtime packages."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_prediction_diff():
    """Load the dataclass-based prediction diagnostic as an importable script."""
    name = "issue51_prediction_diff"
    spec = importlib.util.spec_from_file_location(name, EDGE / "scripts" / "prediction_diff.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_environment_collector():
    """Load the dependency-free host evidence collector."""
    name = "issue51_collect_environment"
    spec = importlib.util.spec_from_file_location(
        name, EDGE / "scripts" / "collect_environment.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_environment_collector_emits_auditable_schema(tmp_path):
    """Environment evidence is machine-readable and records missing tools explicitly."""
    collector = load_environment_collector()
    args = collector._parser().parse_args(
        [
            "--repo-root", str(ROOT),
            "--backend", "onnx",
            "--execution-provider", "cpu",
            "--threads", "4",
            "--warmup", "2",
            "--runs", "3",
        ]
    )
    payload = collector.collect_environment(args)
    assert payload["schema_version"] == "issue51-environment/v1"
    assert payload["host"]["logical_cpus"] is None or payload["host"]["logical_cpus"] >= 1
    assert payload["runtime_protocol"] == {
        "backend": "onnx",
        "execution_provider": "cpu",
        "threads": 4,
        "warmup": 2,
        "runs": 3,
    }
    assert set(payload["sdk_roots"]) == {"onnxruntime", "ncnn", "mnn", "tensorrt"}
    assert "available" in payload["gpu"]
    json.dumps(payload, ensure_ascii=False)

    output = tmp_path / "environment.json"
    assert collector.main(["--repo-root", str(ROOT), "--output", str(output)]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["schema_version"] == "issue51-environment/v1"


def test_environment_collector_rejects_invalid_benchmark_protocol():
    collector = load_environment_collector()
    assert collector.main(["--threads", "0"]) == 2
    assert collector.main(["--runs", "-1"]) == 2
    assert collector.main(["--warmup", "-1"]) == 2


def test_cpp_runner_exposes_auditable_benchmark_sidecar():
    """The documented host/protocol sidecar must be wired into the C++ CLI."""
    source = (EDGE / "cpp" / "src" / "main.cpp").read_text(encoding="utf-8")
    assert "--benchmark-json" in source
    assert "write_benchmark_json" in source
    for field in ("schema_version", "status", "protocol", "execution_provider", "host",
                  "architecture", "compiler", "cpu", "logical_cpus", "build_date",
                  "summary", "timed_images", "timing_csv", "letterbox", "nms_mode",
                  "cw_sigma", "class_count", "timing_ms"):
        assert f'\\"{field}\\"' in source
    assert "benchmark_requested" in source


def test_environment_collector_is_dependency_free_and_schema_versioned():
    """Environment evidence must be machine-readable without edge SDKs."""
    collector = EDGE / "scripts" / "collect_environment.py"
    schema = json.loads((EDGE / "environment.schema.json").read_text(encoding="utf-8"))
    source = collector.read_text(encoding="utf-8")
    assert "subprocess" in source and "SCHEMA_VERSION = \"issue51-environment/v1\"" in source
    assert schema["properties"]["schema_version"]["const"] == "issue51-environment/v1"
    assert set(schema["required"]) >= {
        "captured_at_utc", "host", "tools", "sdk_roots", "runtime_protocol", "repository"
    }


def test_calibration_selection_enforces_issue_floor(tmp_path):
    """Calibration selection is deterministic and requires at least 300 images."""
    quant = load_module("issue51_quant", EDGE / "scripts" / "quantize_int8.py")
    with pytest.raises(ValueError, match="300"):
        quant.select_calibration_images(tmp_path, 299)

    for index in range(300):
        (tmp_path / f"{index:04d}.jpg").touch()
    selected = quant.select_calibration_images(tmp_path, 300)
    assert len(selected) == 300
    assert selected[0].name == "0000.jpg"
    assert selected[-1].name == "0299.jpg"


def test_export_layout_normalizers_accept_common_shapes():
    """MNN and ONNX exporters may transpose the feature/anchor dimensions."""
    parity = load_module("issue51_parity", EDGE / "scripts" / "mnn_parity.py")
    values = np.arange(14 * 5, dtype=np.float32).reshape(14, 5)
    assert parity.normalize_output(values, 14).shape == (14, 5)
    assert parity.normalize_output(values.T[None], 14).shape == (14, 5)
    assert np.array_equal(parity.normalize_output(values.T, 14), values)
    assert parity.select_detection_output(
        [np.zeros((1, 2, 2)), values[None]], 14
    ).shape == (1, 14, 5)

    mnn_val = load_module("issue51_mnn_val", EDGE / "scripts" / "mnn_val.py")
    assert mnn_val.normalize_output(values[None, :, :], 14).shape == (14, 5)
    with pytest.raises(ValueError, match="batch"):
        parity.normalize_output(np.zeros((2, 5, 14), dtype=np.float32), 14)


def test_mnn_directory_order_matches_case_folded_protocol(tmp_path):
    """MNN helpers must enumerate a directory in the canonical order."""
    for name in ("Z.jpg", "a.jpg", "B.jpg"):
        (tmp_path / name).write_bytes(b"image")
    mnn_val = load_module("issue51_mnn_val_order", EDGE / "scripts" / "mnn_val.py")
    mnn_parity = load_module("issue51_mnn_parity_order", EDGE / "scripts" / "mnn_parity.py")
    expected = ["a.jpg", "B.jpg", "Z.jpg"]
    assert [path.name for path in mnn_val.image_list(tmp_path, 0)] == expected
    assert [path.name for path in mnn_parity.image_list(tmp_path, 0)] == expected


def test_nms_is_class_offset_friendly_and_handles_empty():
    """The MNN decoder handles empty candidates and suppresses overlaps."""
    mnn_val = load_module("issue51_mnn_val_nms", EDGE / "scripts" / "mnn_val.py")
    assert mnn_val.class_nms_offset(1920, 1080) == 2.0 * 1920 + 8192.0
    assert mnn_val.nms(np.empty((0, 4)), np.empty((0,)), 0.5) == []
    boxes = np.array(
        [[0, 0, 10, 10], [1, 1, 9, 9], [30, 30, 40, 40]], dtype=np.float32
    )
    keep = mnn_val.nms(boxes, np.array([0.9, 0.8, 0.7], dtype=np.float32), 0.5)
    assert keep == [0, 2]


def test_parity_image_list_rejects_duplicate_stems(tmp_path):
    """A repeated stem would overwrite per-image debug snapshots."""
    parity = load_module("issue51_parity_images", EDGE / "scripts" / "mnn_parity.py")
    (tmp_path / "a.jpg").touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.png").touch()
    with pytest.raises(RuntimeError, match="stems are not unique"):
        parity.image_list(tmp_path, 0)


@pytest.mark.parametrize("script", ["mnn_val.py", "mnn_parity.py"])
def test_mnn_tools_preserve_frozen_image_list_order(tmp_path, script):
    """MNN validation must consume the same ordered list as the mAP tools."""
    module = load_module("issue51_mnn_list_" + script, EDGE / "scripts" / script)
    image_dir = tmp_path / "images with spaces"
    image_dir.mkdir()
    first = image_dir / "B frame.png"
    second = image_dir / "a frame.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    image_list = tmp_path / "validation.list"
    image_list.write_text(
        '\ufeff# frozen validation order\n"images with spaces/B frame.png"\n'
        "'images with spaces/a frame.jpg'\n",
        encoding="utf-8",
    )

    assert module.image_list(image_list, 0) == [first.resolve(), second.resolve()]
    assert module.image_list(image_list, 1) == [first.resolve()]


@pytest.mark.parametrize("script", ["mnn_val.py", "mnn_parity.py"])
def test_mnn_tools_reject_casefolded_duplicate_list_stems(tmp_path, script):
    module = load_module("issue51_mnn_duplicate_" + script, EDGE / "scripts" / script)
    first = tmp_path / "Frame.jpg"
    second = tmp_path / "frame.png"
    first.touch()
    second.touch()
    image_list = tmp_path / "validation.txt"
    image_list.write_text(f"{first}\n{second}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="stems are not unique"):
        module.image_list(image_list, 0)


def test_map_delta_gate_is_optional_and_inclusive():
    """A declared mAP budget is inclusive and requires a reference value."""
    evaluator = load_module("issue51_eval_map_gate", EDGE / "scripts" / "eval_map.py")
    assert evaluator.delta_gate_passes(25.0, None) is True
    assert evaluator.delta_gate_passes(0.5, 0.5) is True
    assert evaluator.delta_gate_passes(0.500001, 0.5) is False

    passing = {"abs_delta_mAP50-95_pct": 0.5}
    assert evaluator.apply_delta_gate(passing, 0.5) == 0
    assert passing["mAP50-95_delta_gate_passed"] is True
    failing = {"abs_delta_mAP50-95_pct": 0.500001}
    assert evaluator.apply_delta_gate(failing, 0.5) != 0
    assert failing["mAP50-95_delta_gate_passed"] is False

    assert evaluator.nonnegative_finite_float("0.5") == pytest.approx(0.5)
    with pytest.raises(argparse.ArgumentTypeError):
        evaluator.nonnegative_finite_float("nan")
    with pytest.raises(argparse.ArgumentTypeError):
        evaluator.nonnegative_finite_float("-0.1")
    assert evaluator.extract_reference_map({"metrics/mAP50-95(B)": 0.2036}) == pytest.approx(0.2036)
    assert evaluator.extract_reference_map({"results_dict": {"mAP50-95": 0.2036}}) == pytest.approx(0.2036)


def test_map_delta_reports_and_gates_absolute_percentage_points():
    evaluator = load_module("issue51_eval_pp_gate", EDGE / "scripts" / "eval_map.py")
    assert evaluator.delta_gate_passes_pp(0.5, 0.5) is True
    assert evaluator.delta_gate_passes_pp(0.500001, 0.5) is False
    result = {
        "mAP50-95": 0.199,
        "reference_mAP50-95": 0.2,
        "delta_mAP50-95_abs": -0.001,
        "delta_mAP50-95_pp": -0.1,
        "abs_delta_mAP50-95_pp": 0.1,
        "delta_mAP50-95_pct": -0.5,
        "abs_delta_mAP50-95_pct": 0.5,
    }
    assert evaluator.apply_delta_gate(result, None, 0.5) == 0
    assert result["mAP50-95_absolute_delta_gate_passed"] is True
    assert result["mAP50-95_delta_gate_passed"] is True
    with pytest.raises(ValueError, match="either"):
        evaluator.validate_delta_budget(0.5, Path("reference.json"), 0.5)


def test_prediction_class_range_is_strict_only_for_formal_runs(tmp_path):
    """Diagnostic parsing skips stale class IDs; formal parsing rejects them."""
    evaluator = load_module("issue51_eval_prediction_class", EDGE / "scripts" / "eval_map.py")
    class TorchStub:
        float32 = object()
        int64 = object()

        @staticmethod
        def tensor(values, dtype=None):
            return np.asarray(values)

    prediction = tmp_path / "frame.txt"
    prediction.write_text("99 0.8 0 0 10 10\n0 0.7 1 1 8 8\n", encoding="utf-8")

    boxes, scores, classes = evaluator.load_predictions(prediction, TorchStub, 10, strict=False)
    assert boxes.shape == (1, 4)
    assert scores.tolist() == pytest.approx([0.7])
    assert classes.tolist() == [0]
    with pytest.raises(ValueError, match="class 99 outside"):
        evaluator.load_predictions(prediction, TorchStub, 10, strict=True)


def test_map_gate_rejects_ambiguous_cli_combinations(monkeypatch):
    """Smoke subsets cannot claim the full image-count or delta gates."""
    evaluator = load_module("issue51_eval_map_args", EDGE / "scripts" / "eval_map.py")
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_map.py", "--preds", "preds", "--max-abs-delta-pct", "0.5"],
    )
    args = evaluator.parse_args()
    assert args.max_abs_delta_pct == pytest.approx(0.5)
    with pytest.raises(ValueError, match="reference-json"):
        evaluator.validate_delta_budget(args.max_abs_delta_pct, args.reference_json)

    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_map.py", "--preds", "preds", "--min-images", "1"],
    )
    args = evaluator.parse_args()
    assert args.smoke is False and args.min_images == 1
    with pytest.raises(ValueError, match="500"):
        evaluator.validate_acceptance_image_floor(args)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_map.py",
            "--preds",
            "preds",
            "--smoke",
            "--reference-json",
            "reference.json",
            "--max-abs-delta-pct",
            "0.5",
        ],
    )
    args = evaluator.parse_args()
    with pytest.raises(ValueError, match="smoke"):
        evaluator.validate_smoke_gate(args.smoke, args.max_abs_delta_pct)


def test_mnn_cli_rejects_invalid_numeric_options_before_import(tmp_path, monkeypatch):
    """Invalid MNN options fail before optional bindings are imported."""
    mnn_val = load_module("issue51_mnn_val_args", EDGE / "scripts" / "mnn_val.py")
    model = tmp_path / "model.mnn"
    image = tmp_path / "image.jpg"
    model.touch()
    image.touch()

    monkeypatch.setattr(
        sys,
        "argv",
        ["mnn_val.py", "--mnn", str(model), "--images", str(image), "--conf", "nan"],
    )
    with pytest.raises(ValueError, match="finite"):
        mnn_val.main()

    monkeypatch.setattr(
        sys,
        "argv",
        ["mnn_val.py", "--mnn", str(model), "--images", str(image), "--limit", "-1"],
    )
    with pytest.raises(ValueError, match="non-negative"):
        mnn_val.main()


def test_exporter_reports_mnn_as_pending_parity():
    """MNN serialization alone is not reported as runtime acceptance evidence."""
    export = (EDGE / "scripts" / "export_models.py").read_text(encoding="utf-8")
    assert '"checked_scope": "converter_output"' in export
    assert '"runtime_smoke_checked": False' in export
    assert '"acceptance_ready": False' in export
    assert '"parity_required": True' in export
    assert '"routing_semantics"' in export


def test_ncnn_status_normalization_handles_boolean_bindings():
    """Python ncnn bindings may expose success as ``True`` instead of zero."""
    export = load_module("issue51_export_ncnn_status", EDGE / "scripts" / "export_models.py")
    assert export._ncnn_code(True) == 0
    assert export._ncnn_code(False) != 0
    assert export._ncnn_code(0) == 0
    assert export._ncnn_code(1) != 0


def test_exporter_dense_routing_state_is_reversible():
    """Export preflight records and restores the model's routing flags."""
    export = load_module("issue51_export_routing", EDGE / "scripts" / "export_models.py")

    class RoutingLayer:
        use_top_k = True

    class MoELayer:
        use_sparse_inference = True

    class ModuleList:
        def modules(self):
            return [self, RoutingLayer(), MoELayer()]

    class Model:
        model = ModuleList()

    state, router_count, esmoe_count = export._force_ncnn_dense(Model())
    assert (router_count, esmoe_count, export._routing_overlap(state)) == (1, 1, 0)
    assert state[0][0].use_top_k is False
    assert state[1][0].use_sparse_inference is False
    for module, attribute, value in reversed(state):
        setattr(module, attribute, value)
    assert state[0][0].use_top_k is True
    assert state[1][0].use_sparse_inference is True
    assert export._routing_record(1, 1)["routing_semantics"] == "dense_fallback"
    assert export._routing_record(0, 0)["routing_semantics"] == "not_applicable"

    class CombinedMoERoutingLayer:
        use_top_k = True
        use_sparse_inference = True

    class CombinedModel:
        class Modules:
            def modules(self):
                return [self, CombinedMoERoutingLayer()]

        model = Modules()

    combined_state, routers, esmoe = export._force_ncnn_dense(CombinedModel())
    overlap = export._routing_overlap(combined_state)
    assert (routers, esmoe, overlap) == (1, 1, 1)
    assert export._routing_record(routers, esmoe, overlap)["routing_layers"]["total"] == 1
    for module, attribute, value in reversed(combined_state):
        setattr(module, attribute, value)
    assert combined_state[0][0].use_top_k is True
    assert combined_state[1][0].use_sparse_inference is True


def test_repository_contains_no_legacy_issue51_path():
    """The final branch uses the current cross-platform directory only."""
    for path in (EDGE / "scripts", EDGE / "cpp"):
        assert "YOLO-Master-EsMoE-N-ONNX-NCNN-MNN-CPP" not in str(path)
    validation = (ROOT / "examples" / "YOLO-Master-Edge-Deployment" / "VALIDATION.md").read_text(
        encoding="utf-8"
    )
    assert "YOLO-Master-EsMoE-N-ONNX-NCNN-MNN-CPP" not in validation


def test_runtime_sources_expose_profile_and_portability_guards():
    """The C++ runner records the canonical profile and validates SDK/runtime assumptions."""
    main = (EDGE / "cpp" / "src" / "main.cpp").read_text(encoding="utf-8")
    assert '"--profile"' in main
    assert 'profile == "visdrone"' in main
    assert "imgsz = 640" in main and "conf = 0.001f" in main and "iou = 0.70f" in main
    assert "multi_label = true" in main and "profile=" in main


def test_ncnn_graph_endpoint_resolution_is_metadata_first_and_fail_closed():
    """NCNN exports must not silently decode an arbitrary terminal tensor."""
    source = (EDGE / "cpp" / "src" / "ncnn_backend.cpp").read_text(encoding="utf-8")
    header = (EDGE / "cpp" / "include" / "ncnn_backend.hpp").read_text(encoding="utf-8")
    assert "inspect_param_graph" in source
    assert "multiple input blobs; provide metadata.yaml input_blob" in source
    assert "multiple terminal blobs; provide metadata.yaml output_blob" in source
    assert "metadata output_blob '" in source
    assert "proto_required_" in source and "required prototype blob" in source
    assert "std::filesystem::u8path(path)" in source
    assert "per_model_metadata.replace_extension(\".metadata.yaml\")" in source
    assert "metadata input_blob and output_blob must differ" in source
    assert "std::string out_proto_" in header


def test_cpp_sku_profile_records_multi_label_protocol():
    """The SKU-110K profile must match the evaluator's explicit decode metadata."""
    main = (EDGE / "cpp" / "src" / "main.cpp").read_text(encoding="utf-8")
    sku_block = main.split('} else if (profile == "sku110k") {', 1)[1].split(
        '} else if (profile != "default")', 1
    )[0]
    assert "multilabel_opt->count() == 0" in sku_block
    assert "multilabel = true" in sku_block
    assert "validate_unique_stems" in main

    common_source = (EDGE / "cpp" / "src" / "common.cpp").read_text(encoding="utf-8")
    assert 'ext == ".txt" || ext == ".list"' in common_source
    assert ".txt or .list image list" in main
    assert "yaml_scalar" in common_source
    assert 'extension == ".txt" || extension == ".list"' in common_source
    assert "return resolve_image_list(c.string())" in common_source

    cmake = (EDGE / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "REQUIRE_ORT" in cmake and "REQUIRE_NCNN" in cmake and "REQUIRE_MNN" in cmake
    assert "ALLOW_NO_BACKENDS" in cmake

    package = (EDGE / "scripts" / "package_linux.sh").read_text(encoding="utf-8")
    assert "NCNN_AVAILABLE" in package and "MNN_AVAILABLE" in package
    assert "neither NCNN nor MNN SDK is available" in package
    assert 'MNN SDK not found at "$MNN_ROOT"' not in package

    ncnn = (EDGE / "cpp" / "src" / "ncnn_backend.cpp").read_text(encoding="utf-8")
    assert "metadata_input" in ncnn and "failed to set input blob" in ncnn
    ort = (EDGE / "cpp" / "src" / "ort_backend.cpp").read_text(encoding="utf-8")
    assert "MultiByteToWideChar" in ort and "no FP32 rank-3 detection output" in ort
    assert "detection output is ambiguous" in ort
    assert "round-to-nearest" in ort and "remainder == halfway" in ort
    mnn = (EDGE / "cpp" / "src" / "mnn_backend.cpp").read_text(encoding="utf-8")
    assert "checked_mnn_call" in mnn and "detection output is ambiguous" in mnn
    assert "#include <iostream>" in mnn

    common = (EDGE / "cpp" / "src" / "common.cpp").read_text(encoding="utf-8")
    assert "std::string* input_blob" in common and "input_blob:" in common
    assert "small_conf_thresh" in common and "candidate_conf_threshold" in common
    assert "--small-conf" in main and "--small-area" in main
    assert "small_conf=" in main and "small_area=" in main

    quant = (EDGE / "scripts" / "quantize_int8.py").read_text(encoding="utf-8")
    assert "--validation-images" in quant
    assert "calibration and validation sets overlap by SHA256" in quant
    assert '"acceptance_ready": False' in quant

    standalone = (EDGE / "scripts" / "eval_map_standalone.py").read_text(encoding="utf-8")
    evaluator = (EDGE / "scripts" / "eval_map.py").read_text(encoding="utf-8")
    assert "PROFILE_PROTOCOLS" in evaluator and '"sku110k"' in evaluator
    assert "--max-abs-delta-pp" in standalone
    assert "--profile" in standalone and "--nc" in standalone
    assert "PROFILE_PROTOCOLS" in standalone and '"sku110k"' in standalone
    assert "--imgsz" in standalone and "--max-det" in standalone
    assert "--max-abs-delta-pct" in standalone
    assert "validation image stems are not unique" in standalone
    assert '"image_manifest_sha256"' in standalone


def test_model_name_metadata_parser_accepts_python_and_json_forms():
    """Metadata serialization must not shift class indices across exporters."""
    # The parser is implemented in C++; keep the source-level contract explicit
    # here because this test suite does not require an OpenCV/NCNN toolchain.
    source = (EDGE / "cpp" / "src" / "common.cpp").read_text(encoding="utf-8")
    assert "JSON object" in source and "JSON list" in source
    assert "std::map<int, std::string> keyed" in source


def test_standalone_map_uses_fixed_class_profile():
    """Absent classes contribute zero instead of being dropped from mAP."""
    standalone = load_module(
        "issue51_standalone_metrics", EDGE / "scripts" / "eval_map_standalone.py"
    )
    tp = np.ones((1, 10), dtype=bool)
    ap = standalone.ap_per_class(
        tp, np.array([0.9]), np.array([0]), np.array([0]), num_classes=2
    )
    assert ap.shape == (2, 10)
    # The 101-point trapezoidal integration used by Ultralytics assigns a
    # 0.995 area to a single perfect detection (the final endpoint is zero).
    assert ap[0, 0] == pytest.approx(0.995)
    assert np.all(ap[1] == 0.0)
    assert ap.mean() == pytest.approx(0.4975)


def test_standalone_profile_protocol_defaults_match_runner_profiles():
    standalone = load_module(
        "issue51_standalone_profile_protocol", EDGE / "scripts" / "eval_map_standalone.py"
    )
    assert standalone.PROFILE_PROTOCOLS["visdrone"]["imgsz"] == 640
    assert standalone.PROFILE_PROTOCOLS["visdrone"]["conf"] == pytest.approx(0.001)
    assert standalone.PROFILE_PROTOCOLS["visdrone"]["iou"] == pytest.approx(0.70)
    assert standalone.PROFILE_PROTOCOLS["sku110k"]["imgsz"] == 1280
    assert standalone.PROFILE_PROTOCOLS["sku110k"]["conf"] == pytest.approx(0.25)
    assert standalone.PROFILE_PROTOCOLS["sku110k"]["iou"] == pytest.approx(0.60)


def test_ultralytics_evaluator_profile_protocol_defaults(monkeypatch):
    evaluator = load_module(
        "issue51_eval_profile_protocol", EDGE / "scripts" / "eval_map.py"
    )
    monkeypatch.setattr(
        sys, "argv", ["eval_map.py", "--preds", "preds", "--classes", "sku110k", "--smoke"]
    )
    args = evaluator.parse_args()
    assert args.imgsz == 1280
    assert args.conf == pytest.approx(0.25)
    assert args.iou == pytest.approx(0.60)
    assert args.max_det == 300


@pytest.mark.parametrize(
    ("kind", "content", "pattern"),
    [
        ("gt", "0 0.5 0.5 0.2\n", "exactly 5 columns"),
        ("gt", "0 0.5 nan 0.2 0.2\n", "NaN or Inf"),
        ("gt", "0 0.5 0.5 -0.2 0.2\n", "width and height"),
        ("pred", "0 0.9 0 0 1\n", "exactly 6 columns"),
        ("pred", "0 0.9 2 2 1 3\n", "x2>x1 and y2>y1"),
    ],
)
def test_standalone_parsers_report_malformed_rows(tmp_path, kind, content, pattern):
    """Malformed evidence must fail with a path/line-specific diagnostic."""
    standalone = load_module(
        "issue51_standalone_parser_" + kind + pattern[:2].replace(" ", "_"),
        EDGE / "scripts" / "eval_map_standalone.py",
    )
    path = tmp_path / (kind + ".txt")
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=pattern):
        if kind == "gt":
            standalone.load_gt(path, 100, 100, num_classes=10)
        else:
            standalone.load_pred(path, num_classes=10)


def test_standalone_visdrone_parser_keeps_valid_classes_and_ignores_reserved_rows(tmp_path):
    standalone = load_module(
        "issue51_standalone_visdrone", EDGE / "scripts" / "eval_map_standalone.py"
    )
    path = tmp_path / "annotations.txt"
    path.write_text(
        "0,0,10,10,1,1,0,0\n"
        "10,10,10,10,0,2,0,0\n"
        "20,20,10,10,1,11,0,0\n",
        encoding="utf-8",
    )
    boxes, classes = standalone.load_gt(path, 100, 100, "visdrone")
    assert boxes.shape == (1, 4)
    assert classes.tolist() == [0]


def test_standalone_visdrone_parser_accepts_whitespace_rows_in_auto_mode(tmp_path):
    """Native VisDrone exports may use spaces instead of commas."""
    standalone = load_module(
        "issue51_standalone_visdrone_whitespace", EDGE / "scripts" / "eval_map_standalone.py"
    )
    path = tmp_path / "annotations.txt"
    path.write_text("10 20 30 40 1 2 0 0\n", encoding="utf-8")
    boxes, classes = standalone.load_gt(path, 100, 100, "auto")
    assert boxes.tolist() == [[10.0, 20.0, 40.0, 60.0]]
    assert classes.tolist() == [1]


def test_evidence_manifest_cli_template_and_validation(tmp_path):
    manifest_module = load_module(
        "issue51_evidence_manifest",
        EDGE / "scripts" / "evidence_manifest.py",
    )
    output = tmp_path / "template.json"
    assert manifest_module.main(["create", "--template", "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "template"
    assert payload["artifacts"]["labels"] is None
    assert payload["artifacts"]["predictions"] is None
    assert manifest_module.validate_manifest(payload) == []
    assert manifest_module.validate_manifest(payload, acceptance=True)


def test_evidence_manifest_records_single_label_protocol(tmp_path):
    """The manifest CLI must preserve an explicitly requested decoder mode."""
    manifest_module = load_module(
        "issue51_evidence_manifest_label_mode",
        EDGE / "scripts" / "evidence_manifest.py",
    )
    output = tmp_path / "single-label.json"
    assert manifest_module.main(
        ["create", "--template", "--single-label", "--output", str(output)]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["protocol"]["multi_label"] is False

    output_default = tmp_path / "multi-label.json"
    assert manifest_module.main(
        ["create", "--template", "--multi-label", "--output", str(output_default)]
    ) == 0
    default_payload = json.loads(output_default.read_text(encoding="utf-8"))
    assert default_payload["protocol"]["multi_label"] is True


def test_evidence_manifest_acceptance_requires_auditable_digests_and_format_keys():
    manifest_module = load_module(
        "issue51_evidence_manifest_audit", EDGE / "scripts" / "evidence_manifest.py"
    )
    record = lambda path, digest: {"path": path, "bytes": 1, "sha256": digest}
    images = [record(f"{index:04d}.jpg", f"{index:064x}") for index in range(500)]
    calibration = [record(f"cal-{index:04d}.jpg", f"{1000 + index:064x}") for index in range(300)]
    models = {
        "onnx_fp32": {"files": [record("model.onnx", "d" * 64)]},
        "mnn_int8": {"files": [record("model.mnn", "e" * 64)]},
    }
    for model in models.values():
        model["sha256"] = manifest_module._list_digest(model["files"])
    report_files = [record("metrics.json", "f" * 64)]
    payload = {
        "schema_version": manifest_module.SCHEMA_VERSION,
        "status": "acceptance-candidate",
        "dataset": {
            "image_count": len(images), "images": images,
            "image_list_sha256": manifest_module._list_digest(images),
        },
        "protocol": {"imgsz": 640, "conf": 0.001, "iou": 0.7, "max_det": 300,
                     "multi_label": True, "letterbox": True,
                     "small_conf": -1.0, "small_area": 1024.0,
                     "routing_semantics": "dense_fallback"},
        "training": {
            "base_model": "esmoe-n.yaml", "dataset_version": "VisDrone2019-DET",
            "epochs": 120, "seed": 0, "command": "yolo train ...",
        },
        "artifacts": {
            "checkpoint": record("best.pt", "c" * 64), "models": models,
            "reports": {"metrics": {
                "files": report_files, "sha256": manifest_module._list_digest(report_files),
            }},
            "labels": {"count": len(images), "files": images},
            "predictions": {"count": len(images), "files": images},
        },
        "calibration": {
            "enabled": True, "image_count": len(calibration), "images": calibration,
            "image_list_sha256": manifest_module._list_digest(calibration),
            "disjoint_from_validation": True,
        },
        "environment": {
            "python": "3.10.12", "platform": "Ubuntu-22.04", "machine": "x86_64",
            "git_commit": "a" * 40,
        },
        "run": {"command": "./yolomaster_edge --profile visdrone ..."},
    }
    assert manifest_module.validate_manifest(payload, acceptance=True) == []
    training = payload.pop("training")
    assert any("training provenance" in error
               for error in manifest_module.validate_manifest(payload, acceptance=True))
    payload["training"] = training
    payload["artifacts"]["models"]["onnx_fp32"].pop("sha256")
    errors = manifest_module.validate_manifest(payload, acceptance=True)
    assert any("onnx_fp32.sha256" in error for error in errors)
    payload["artifacts"]["models"]["onnx_fp32"]["sha256"] = manifest_module._list_digest(
        payload["artifacts"]["models"]["onnx_fp32"]["files"]
    )
    payload["calibration"]["disjoint_from_validation"] = None
    assert any("disjoint_from_validation=true" in error
               for error in manifest_module.validate_manifest(payload, acceptance=True))
    assert manifest_module._model_format("onnx_fp16") == "onnx"
    assert manifest_module._model_format("artifact", {"files": [{"path": "artifact.mnn"}]}) == "mnn"


def test_evidence_schema_keeps_acceptance_floors_and_portable_paths():
    """The standalone JSON Schema must enforce the Python validator's core floors."""
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            assert key not in result, f"duplicate JSON key: {key}"
            result[key] = value
        return result

    schema = json.loads(
        (EDGE / "evidence-manifest.schema.json").read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
    acceptance = schema["allOf"][0]["then"]
    dataset = acceptance["properties"]["dataset"]["properties"]
    artifacts = acceptance["properties"]["artifacts"]["properties"]
    calibration = acceptance["allOf"][0]["then"]["properties"]["calibration"]["properties"]

    assert dataset["image_count"]["minimum"] == 500
    assert dataset["images"]["minItems"] == 500
    assert artifacts["models"]["minProperties"] == 1
    assert artifacts["reports"]["minProperties"] == 1
    assert calibration["image_count"]["minimum"] == 300
    assert calibration["images"]["minItems"] == 300
    assert "routing_semantics" in acceptance["properties"]["protocol"]["required"]
    assert "small_conf" in acceptance["properties"]["protocol"]["required"]
    assert "small_area" in acceptance["properties"]["protocol"]["required"]
    assert acceptance["properties"]["protocol"]["properties"]["small_conf"]["minimum"] == -1
    assert acceptance["properties"]["protocol"]["properties"]["small_area"]["minimum"] == 0

    pattern = re.compile(schema["$defs"]["file_record"]["properties"]["path"]["pattern"])
    assert pattern.fullmatch("images/0001.jpg")
    for unsafe in ("../outside.jpg", "images/../outside.jpg", "/absolute.jpg", "C:/absolute.jpg", "a\\b.jpg"):
        assert pattern.fullmatch(unsafe) is None


def test_evidence_manifest_records_and_validates_small_object_protocol(tmp_path):
    manifest_module = load_module(
        "issue51_evidence_manifest_small_protocol", EDGE / "scripts" / "evidence_manifest.py"
    )
    output = tmp_path / "small-protocol-template.json"
    assert manifest_module.main([
        "create", "--template", "--small-conf", "0.05", "--small-area", "1024",
        "--output", str(output),
    ]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["protocol"]["small_conf"] == pytest.approx(0.05)
    assert payload["protocol"]["small_area"] == pytest.approx(1024.0)
    payload["protocol"]["small_conf"] = 1.5
    assert any("small_conf" in error for error in manifest_module.validate_manifest(payload))
    payload["protocol"]["small_conf"] = -1.0
    payload["protocol"]["small_area"] = -0.1
    assert any("small_area" in error for error in manifest_module.validate_manifest(payload))


def test_evidence_manifest_enforces_image_and_calibration_gates():
    manifest_module = load_module(
        "issue51_evidence_manifest_gates",
        EDGE / "scripts" / "evidence_manifest.py",
    )

    def record(path, digest):
        return {"path": path, "bytes": 1, "sha256": digest}

    images = [record("{:04d}.jpg".format(i), "{:064x}".format(i)) for i in range(500)]
    calibration = [
        record("cal-{:04d}.jpg".format(i), "{:064x}".format(1000 + i))
        for i in range(300)
    ]
    model_onnx = [{"path": "model.onnx", "bytes": 1, "sha256": "d" * 64}]
    model_mnn = [{"path": "model.mnn", "bytes": 1, "sha256": "e" * 64}]
    report_files = [{"path": "metrics.json", "bytes": 1, "sha256": "f" * 64}]
    payload = {
        "schema_version": manifest_module.SCHEMA_VERSION,
        "status": "acceptance-candidate",
        "dataset": {
            "image_count": len(images),
            "images": images,
            "image_list_sha256": manifest_module._list_digest(images),
        },
        "protocol": {
            "imgsz": 640,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "multi_label": True,
            "letterbox": True,
            "small_conf": -1.0,
            "small_area": 1024.0,
            "routing_semantics": "dense_fallback",
        },
        "training": {
            "base_model": "esmoe-n.yaml",
            "dataset_version": "VisDrone2019-DET",
            "epochs": 120,
            "seed": 0,
            "command": "yolo train ...",
        },
        "artifacts": {
            "checkpoint": {"path": "best.pt", "bytes": 1, "sha256": "c" * 64},
            "models": {
                "onnx": {"files": model_onnx, "sha256": manifest_module._list_digest(model_onnx)},
                "mnn": {"files": model_mnn, "sha256": manifest_module._list_digest(model_mnn)},
            },
            "reports": {
                "metrics": {"files": report_files, "sha256": manifest_module._list_digest(report_files)},
            },
            "labels": {"count": len(images), "files": images},
            "predictions": {"count": len(images), "files": images},
        },
        "calibration": {
            "enabled": True,
            "image_count": len(calibration),
            "images": calibration,
            "image_list_sha256": manifest_module._list_digest(calibration),
            "disjoint_from_validation": True,
        },
        "environment": {
            "python": "3.10.12",
            "platform": "Ubuntu-22.04",
            "machine": "x86_64",
            "git_commit": "a" * 40,
        },
        "run": {"command": "./yolomaster_edge --profile visdrone ..."},
    }
    assert manifest_module.validate_manifest(payload, acceptance=True) == []
    payload["calibration"]["images"] = calibration[:299]
    payload["calibration"]["image_count"] = 299
    assert any("300" in error for error in manifest_module.validate_manifest(payload, acceptance=True))


def test_evidence_manifest_requires_explicit_routing_semantics_for_acceptance():
    manifest_module = load_module(
        "issue51_evidence_manifest_routing", EDGE / "scripts" / "evidence_manifest.py"
    )
    # Use a minimal payload that already satisfies all other acceptance floors;
    # the route field should be the only newly introduced violation.
    record = lambda path, digest: {"path": path, "bytes": 1, "sha256": digest}
    images = [record("{:04d}.jpg".format(i), "{:064x}".format(i)) for i in range(500)]
    model_onnx = [record("model.onnx", "d" * 64)]
    model_mnn = [record("model.mnn", "e" * 64)]
    reports = [record("metrics.json", "f" * 64)]
    payload = {
        "schema_version": manifest_module.SCHEMA_VERSION,
        "status": "acceptance-candidate",
        "dataset": {"image_count": 500, "images": images,
                    "image_list_sha256": manifest_module._list_digest(images)},
        "protocol": {"imgsz": 640, "conf": 0.001, "iou": 0.7, "max_det": 300,
                     "multi_label": True, "letterbox": True,
                     "small_conf": -1.0, "small_area": 1024.0},
        "training": {"base_model": "esmoe", "dataset_version": "v1", "epochs": 1,
                      "seed": 0, "command": "train"},
        "artifacts": {
            "checkpoint": record("best.pt", "c" * 64),
            "models": {
                "onnx": {"files": model_onnx, "sha256": manifest_module._list_digest(model_onnx)},
                "mnn": {"files": model_mnn, "sha256": manifest_module._list_digest(model_mnn)},
            },
            "reports": {"metrics": {"files": reports, "sha256": manifest_module._list_digest(reports)}},
            "labels": {"count": 500, "files": images},
            "predictions": {"count": 500, "files": images},
        },
        "calibration": {"enabled": False, "image_count": 0, "images": [],
                        "image_list_sha256": None, "disjoint_from_validation": None},
        "environment": {"python": "3.10", "platform": "linux", "machine": "x86_64",
                         "git_commit": "a" * 40},
        "run": {"command": "run"},
    }
    errors = manifest_module.validate_manifest(payload, acceptance=True)
    assert any("routing_semantics" in error for error in errors)
    payload["protocol"]["routing_semantics"] = "dense_fallback"
    assert not any("routing_semantics" in error for error in manifest_module.validate_manifest(
        payload, acceptance=True
    ))


def test_evidence_manifest_rejects_duplicate_image_stems(tmp_path):
    manifest_module = load_module(
        "issue51_evidence_manifest_stems",
        EDGE / "scripts" / "evidence_manifest.py",
    )
    (tmp_path / "a.jpg").write_bytes(b"a")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "a.png").write_bytes(b"b")
    base, paths = manifest_module.resolve_image_list(tmp_path)
    records = manifest_module._image_records(paths, base)
    assert any("duplicate image stems" in error for error in manifest_module.validate_manifest(
        {"schema_version": manifest_module.SCHEMA_VERSION, "dataset": {"image_count": 2, "images": records}},
        acceptance=False,
    ))


def test_evidence_manifest_verify_detects_hash_changes(tmp_path):
    manifest_module = load_module(
        "issue51_evidence_manifest_verify",
        EDGE / "scripts" / "evidence_manifest.py",
    )
    root = tmp_path / "images"
    root.mkdir()
    image = root / "a.jpg"
    image.write_bytes(b"original")
    _, paths = manifest_module.resolve_image_list(root)
    records = manifest_module._image_records(paths, root)
    payload = {
        "schema_version": manifest_module.SCHEMA_VERSION,
        "status": "diagnostic",
        "dataset": {"image_count": 1, "images": records},
        "protocol": {},
        "artifacts": {"checkpoint": None, "models": {}, "labels": None, "predictions": None},
        "calibration": {"enabled": False, "images": []},
    }
    assert manifest_module._verify_records(records, root, "images") == []
    image.write_bytes(b"changed")
    assert any("mismatch" in error for error in manifest_module._verify_records(records, root, "images"))


def test_evidence_manifest_records_and_verifies_report_collections(tmp_path):
    manifest_module = load_module(
        "issue51_evidence_manifest_reports",
        EDGE / "scripts" / "evidence_manifest.py",
    )
    report_root = tmp_path / "reports"
    report_root.mkdir()
    report = report_root / "map.json"
    report.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "manifest.json"
    assert manifest_module.main([
        "create", "--template", "--output", str(output),
    ]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    records = manifest_module.collect_records(report)
    payload["artifacts"]["reports"] = {
        "map": {"files": records, "sha256": manifest_module._list_digest(records)},
    }
    assert manifest_module.validate_manifest(payload) == []
    assert manifest_module._verify_records(records, report_root, "report.map") == []
    report.write_text("changed\n", encoding="utf-8")
    assert any("SHA256 mismatch" in error for error in manifest_module._verify_records(
        records, report_root, "report.map"
    ))


def test_evidence_manifest_create_accepts_external_image_list_and_report(tmp_path):
    manifest_module = load_module(
        "issue51_evidence_manifest_cli_roots",
        EDGE / "scripts" / "evidence_manifest.py",
    )
    image_root = tmp_path / "images"
    image_root.mkdir()
    image = image_root / "frame.jpg"
    image.write_bytes(b"image")
    list_file = tmp_path / "ordered.list"
    list_file.write_text(str(image) + "\n", encoding="utf-8")
    report = tmp_path / "timing.csv"
    report.write_text("tag,total_ms\nframe.jpg,1\n", encoding="utf-8")
    output = tmp_path / "manifest.json"
    assert manifest_module.main([
        "create", "--images", str(list_file), "--image-root", str(tmp_path),
        "--report", "timing=" + str(report), "--output", str(output),
    ]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["dataset"]["images"][0]["path"] == "images/frame.jpg"
    assert payload["artifacts"]["reports"]["timing"]["files"][0]["path"] == "timing.csv"


def test_all_image_list_consumers_accept_bom_and_quoted_paths(tmp_path):
    """A frozen list must have identical semantics in every evaluator."""
    image_root = tmp_path / "images with spaces"
    image_root.mkdir()
    image = image_root / "frame 01.jpg"
    image.write_bytes(b"image")
    list_file = tmp_path / "ordered.list"
    list_file.write_text("\ufeff# generated on Windows\n\"images with spaces/frame 01.jpg\"\n", encoding="utf-8")

    manifest = load_module("issue51_manifest_list_syntax", EDGE / "scripts" / "evidence_manifest.py")
    base, paths = manifest.resolve_image_list(list_file)
    assert paths == [image.resolve()]
    assert base == tmp_path.resolve()

    evaluator = load_module("issue51_eval_list_syntax", EDGE / "scripts" / "eval_map.py")
    eval_base, eval_paths = evaluator._resolve_images(list_file)
    assert eval_paths == [image.resolve()]
    assert eval_base == tmp_path.resolve()

    standalone = load_module(
        "issue51_standalone_list_syntax", EDGE / "scripts" / "eval_map_standalone.py"
    )
    standalone_base, standalone_paths = standalone._resolve_images(list_file)
    assert standalone_paths == [str(image.resolve())]
    assert standalone_base == str(tmp_path.resolve())


def test_eval_map_formal_loaders_reject_malformed_rows(tmp_path):
    """The Ultralytics-backed evaluator must not silently alter formal labels."""
    evaluator = load_module("issue51_eval_map_strict", EDGE / "scripts" / "eval_map.py")

    class TorchStub:
        float32 = object()
        int64 = object()

        @staticmethod
        def tensor(values, dtype=None):
            return np.asarray(values)

    labels = tmp_path / "bad-labels.txt"
    labels.write_text("0 0.5 0.5 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 5 columns"):
        evaluator.load_gt(labels, 100, 100, TorchStub, "yolo", 10, strict=True)
    preds = tmp_path / "bad-preds.txt"
    preds.write_text("0 0.9 0 0 1 1 stale\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 6 columns"):
        evaluator.load_predictions(preds, TorchStub, 10, strict=True)


def test_formal_evaluators_require_reference_protocol_metadata():
    """A delta gate cannot compare a scalar-only reference report."""
    evaluator = load_module("issue51_eval_map_reference", EDGE / "scripts" / "eval_map.py")
    current = {
        "images": 500,
        "classes": 10,
        "class_profile": "visdrone",
        "label_format": "yolo",
        "image_manifest_sha256": "a" * 64,
        "image_content_manifest_sha256": "b" * 64,
        "protocol": {
            "imgsz": 640, "conf": 0.001, "iou": 0.70, "max_det": 300,
            "multi_label": True, "letterbox": True, "color": "RGB", "layout": "NCHW",
        },
    }
    with pytest.raises(ValueError, match="image_manifest_sha256"):
        evaluator.validate_reference_metadata({"mAP50-95": 0.2}, current, strict=True)


def test_formal_evaluators_reject_mismatched_routing_semantics():
    evaluator = load_module("issue51_eval_map_routing", EDGE / "scripts" / "eval_map.py")
    current = {
        "images": 500,
        "classes": 10,
        "class_profile": "visdrone",
        "label_format": "yolo",
        "image_manifest_sha256": "a" * 64,
        "image_list_sha256": "b" * 64,
        "protocol": {
            "imgsz": 640, "conf": 0.001, "iou": 0.70, "max_det": 300,
            "multi_label": True, "letterbox": True, "color": "RGB", "layout": "NCHW",
            "routing_semantics": "dense_fallback",
        },
    }
    reference = dict(current)
    reference["protocol"] = dict(current["protocol"], routing_semantics="native_sparse")
    with pytest.raises(ValueError, match="routing_semantics"):
        evaluator.validate_reference_metadata(reference, current, strict=True)


def test_metric_reports_hash_ordered_image_contents(tmp_path):
    evaluator = load_module("issue51_eval_content_hash", EDGE / "scripts" / "eval_map.py")
    root = tmp_path / "images"
    root.mkdir()
    image = root / "a.jpg"
    image.write_bytes(b"first")
    first = evaluator.image_content_manifest([image], root)
    image.write_bytes(b"second")
    second = evaluator.image_content_manifest([image], root)
    assert first != second

    standalone = load_module(
        "issue51_standalone_content_hash", EDGE / "scripts" / "eval_map_standalone.py"
    )
    assert standalone._image_content_manifest([str(image)], str(root)) == second


def test_manifest_verifier_rejects_paths_that_escape_root():
    """Evidence verification must never follow an absolute or parent path."""
    manifest = load_module(
        "issue51_manifest_path_safety", EDGE / "scripts" / "evidence_manifest.py"
    )
    record = {"path": "../outside.bin", "bytes": 1, "sha256": "0" * 64}
    assert any("relative POSIX" in error for error in manifest._validate_records([record], "artifact"))


def test_calibration_and_validation_stems_are_unambiguous(tmp_path):
    """Calibration manifests must be deterministic on case-sensitive hosts."""
    quant = load_module("issue51_quant_stems", EDGE / "scripts" / "quantize_int8.py")
    first = tmp_path / "A.jpg"
    second = tmp_path / "a.png"
    first.touch()
    second.touch()
    with pytest.raises(ValueError, match="stems are not unique"):
        quant._validate_unique_stems([first, second], "calibration")


def test_evaluators_match_evidence_manifest_content_digest(tmp_path):
    """All evidence tools must identify the same ordered image bytes."""
    image_root = tmp_path / "dataset" / "images"
    image_root.mkdir(parents=True)
    first = image_root / "A.jpg"
    second = image_root / "b.png"
    first.write_bytes(b"first image")
    second.write_bytes(b"second image")
    list_file = tmp_path / "artifacts" / "val.list"
    list_file.parent.mkdir()
    list_file.write_text(f"{second}\n{first}\n", encoding="utf-8")

    evaluator = load_module("issue51_eval_digest", EDGE / "scripts" / "eval_map.py")
    standalone = load_module("issue51_standalone_digest", EDGE / "scripts" / "eval_map_standalone.py")
    manifest = load_module("issue51_manifest_digest", EDGE / "scripts" / "evidence_manifest.py")

    eval_root, eval_images = evaluator._resolve_images(list_file, image_root)
    standalone_root, standalone_images = standalone._resolve_images(list_file, image_root)
    manifest_root, manifest_images = manifest.resolve_image_list(list_file, image_root)

    expected = manifest._list_digest(manifest._image_records(manifest_images, manifest_root))
    assert evaluator.image_content_manifest(eval_images, eval_root) == expected
    assert standalone._image_content_manifest(standalone_images, standalone_root) == expected
    assert [path.name for path in eval_images] == ["b.png", "A.jpg"]


@pytest.mark.parametrize("standalone", [False, True])
def test_evaluator_rejects_list_entry_outside_image_root(tmp_path, standalone):
    image_root = tmp_path / "images"
    image_root.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    list_file = tmp_path / "val.list"
    list_file.write_text(str(outside) + "\n", encoding="utf-8")

    script = "eval_map_standalone.py" if standalone else "eval_map.py"
    module = load_module("issue51_root_" + str(standalone), EDGE / "scripts" / script)
    with pytest.raises(ValueError, match="outside evaluation root"):
        module._resolve_images(list_file, image_root)


def test_content_digest_changes_when_image_bytes_change(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"version one")
    evaluator = load_module("issue51_eval_content_change", EDGE / "scripts" / "eval_map.py")

    first = evaluator.image_content_manifest([image], tmp_path)
    image.write_bytes(b"version two")
    second = evaluator.image_content_manifest([image], tmp_path)

    assert first != second


def test_prediction_matching_is_class_aware_and_reports_coordinate_delta():
    diff = load_prediction_diff()
    reference = [
        diff.Prediction(0, 0.80, 0, 0, 10, 10),
        diff.Prediction(1, 0.90, 20, 20, 30, 30),
    ]
    candidate = [
        diff.Prediction(0, 0.75, 1, 0, 11, 10),
        diff.Prediction(2, 0.90, 20, 20, 30, 30),
    ]
    matches = diff.match_predictions(reference, candidate, 0.5)
    assert len(matches) == 1
    assert matches[0].reference_index == 0 and matches[0].candidate_index == 0
    assert matches[0].iou > 0.7
    assert matches[0].confidence_abs_delta == pytest.approx(0.05)
    report = diff.compare_image(reference, candidate, 0.5)
    assert report["matched"] == 1
    assert report["unmatched_reference"] == 1
    assert report["unmatched_candidate"] == 1


def test_prediction_parser_rejects_invalid_geometry(tmp_path):
    diff = load_prediction_diff()
    path = tmp_path / "bad.txt"
    path.write_text("0 0.5 4 4 3 8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="positive width"):
        diff.read_predictions(path)


def test_prediction_directory_compare_requires_matching_sets(tmp_path):
    diff = load_prediction_diff()
    reference, candidate = tmp_path / "ref", tmp_path / "cand"
    reference.mkdir()
    candidate.mkdir()
    (reference / "a.txt").write_text("0 0.9 0 0 10 10\n", encoding="utf-8")
    (candidate / "b.txt").write_text("0 0.9 0 0 10 10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file sets differ"):
        diff.compare_directories(reference, candidate)


def test_prediction_directory_compare_reports_machine_readable_deltas(tmp_path):
    diff = load_prediction_diff()
    reference, candidate = tmp_path / "ref", tmp_path / "cand"
    reference.mkdir()
    candidate.mkdir()
    (reference / "a.txt").write_text("0 0.9 0 0 10 10\n", encoding="utf-8")
    (candidate / "a.txt").write_text("0 0.8 0 0 10 10\n", encoding="utf-8")
    report = diff.compare_directories(reference, candidate)
    assert report["summary"]["matched_detections"] == 1
    assert report["images"][0]["max_confidence_abs_delta"] == pytest.approx(0.1)


def test_prediction_diff_image_lists_match_evaluator_syntax_and_root_guard(tmp_path):
    """Prediction diagnostics must consume the same frozen list as mAP tools."""
    diff = load_prediction_diff()
    image_root = tmp_path / "images with spaces"
    image_root.mkdir()
    image = image_root / "frame 01.jpg"
    image.write_bytes(b"image")
    list_file = tmp_path / "ordered.list"
    list_file.write_text('\ufeff# exported by Windows\n"images with spaces/frame 01.jpg"\n', encoding="utf-8")

    indexed = diff.image_files(list_file, image_root)
    assert indexed == {"frame 01": image.resolve()}

    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    list_file.write_text(str(outside) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside evaluation root"):
        diff.image_files(list_file, image_root)


def test_prediction_diff_reports_image_and_summary_iou_statistics(tmp_path):
    diff = load_prediction_diff()
    reference, candidate = tmp_path / "ref", tmp_path / "cand"
    reference.mkdir()
    candidate.mkdir()
    (reference / "a.txt").write_text(
        "0 0.9 0 0 10 10\n0 0.8 20 20 30 30\n", encoding="utf-8"
    )
    (candidate / "a.txt").write_text(
        "0 0.9 0 0 9 10\n0 0.8 20 20 30 30\n", encoding="utf-8"
    )
    report = diff.compare_directories(reference, candidate, iou_threshold=0.1)
    row = report["images"][0]
    summary = report["summary"]
    assert row["matched_iou_count"] == 2
    assert 0.0 < row["p05_iou"] <= row["p50_iou"] <= row["p95_iou"] <= 1.0
    assert summary["matched_iou_count"] == 2
    assert summary["min_iou"] == pytest.approx(row["min_iou"])
    assert summary["p99_iou"] >= summary["p05_iou"]


def test_prediction_diff_min_iou_gate_fails_closed(tmp_path, monkeypatch):
    diff = load_prediction_diff()
    reference, candidate = tmp_path / "ref", tmp_path / "cand"
    reference.mkdir()
    candidate.mkdir()
    (reference / "a.txt").write_text("0 0.9 0 0 10 10\n", encoding="utf-8")
    (candidate / "a.txt").write_text("0 0.9 2 0 12 10\n", encoding="utf-8")
    monkeypatch.setattr(
        diff.sys,
        "argv",
        [
            "prediction_diff.py",
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--iou",
            "0.1",
            "--min-iou",
            "0.9",
        ],
    )
    assert diff.main() == 1


def test_prediction_diff_rejects_invalid_matching_iou(tmp_path):
    diff = load_prediction_diff()
    reference, candidate = tmp_path / "ref", tmp_path / "cand"
    reference.mkdir()
    candidate.mkdir()
    (reference / "a.txt").write_text("0 0.9 0 0 10 10\n", encoding="utf-8")
    (candidate / "a.txt").write_text("0 0.9 0 0 10 10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="matching IoU threshold"):
        diff.compare_directories(reference, candidate, iou_threshold=1.1)
