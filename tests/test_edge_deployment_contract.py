"""Contract tests for the dependency-light edge deployment scaffold.

These tests intentionally avoid downloading a checkpoint or an edge runtime SDK.
They exercise the preprocessing/validation API and the CMake CLI contract so a
backend implementation can be swapped in without changing benchmark automation.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EDGE_DIR = ROOT / "examples" / "YOLO-Master-Edge-Deployment"
EDGE_UTILS = EDGE_DIR / "edge_utils.py"

spec = importlib.util.spec_from_file_location("edge_utils_contract", EDGE_UTILS)
edge_utils = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = edge_utils
spec.loader.exec_module(edge_utils)


def test_profiles_are_explicit_and_case_insensitive():
    """Both vertical domains expose bounded, non-zero preprocessing defaults."""
    assert set(edge_utils.PROFILES) == {"visdrone", "sku110k"}
    assert edge_utils.get_profile("VISDRONE") is edge_utils.PROFILES["visdrone"]
    for profile in edge_utils.PROFILES.values():
        assert all(size > 0 for size in profile.image_size)
        assert 0.0 < profile.conf_threshold < 1.0
        assert 0.0 < profile.iou_threshold < 1.0
        assert profile.keep_aspect_ratio is True
        assert profile.max_det == 300
    vis = edge_utils.get_profile("visdrone")
    assert vis.image_size == (640, 640)
    assert vis.conf_threshold == pytest.approx(0.001)
    assert vis.iou_threshold == pytest.approx(0.70)
    assert vis.multi_label is True
    sku = edge_utils.get_profile("sku110k")
    assert sku.image_size == (1280, 1280)
    assert sku.conf_threshold == pytest.approx(0.25)
    assert sku.iou_threshold == pytest.approx(0.60)
    assert sku.multi_label is True


def test_unknown_profile_error_lists_supported_profiles():
    with pytest.raises(ValueError, match="visdrone.*sku110k|sku110k.*visdrone"):
        edge_utils.get_profile("unknown")


def test_letterbox_shape_preserves_ratio_and_stride_aligned_auto_padding():
    ratio, unpadded, pad = edge_utils.letterbox_shape((1080, 1920), (544, 960))
    assert ratio == pytest.approx(0.5)
    assert unpadded == (960, 540)
    assert pad == (0, 2)

    auto_ratio, auto_unpadded, auto_pad = edge_utils.letterbox_shape(
        (100, 200), (640, 640), stride=32, auto=True
    )
    assert auto_ratio == pytest.approx(3.2)
    assert auto_unpadded == (640, 320)
    assert (640 - auto_unpadded[0] - 2 * auto_pad[0]) % 32 == 0
    assert (640 - auto_unpadded[1] - 2 * auto_pad[1]) % 32 == 0


def test_letterbox_shape_matches_ultralytics_odd_padding_tiebreak():
    # A 3-pixel residual pad must place the extra pixel on the right/bottom,
    # exactly as LetterBox(round(dw - 0.1), round(dh - 0.1)) does.
    _, unpadded, pad = edge_utils.letterbox_shape((7, 10), (10, 10))
    assert unpadded == (10, 7)
    assert pad == (0, 1)


def test_scale_boxes_handles_empty_input_without_mutation():
    boxes = np.array([[1.0, 2.0, 10.0, 20.0]], dtype=np.float32)
    original = boxes.copy()
    scaled = edge_utils.scale_xyxy_boxes(boxes, (100, 200), (640, 640), (4, 8), 2.0)
    assert np.array_equal(boxes, original)
    assert scaled.dtype == np.float32

    empty = edge_utils.scale_xyxy_boxes(np.empty((0, 4)), (100, 200), (640, 640), (0, 0), 1.0)
    assert empty.shape == (0, 4)

    with pytest.raises(ValueError, match="shape"):
        edge_utils.scale_xyxy_boxes(np.zeros((2, 3)), (100, 200), (640, 640), (0, 0), 1.0)


def test_compare_arrays_covers_empty_and_shape_mismatch():
    empty = edge_utils.compare_arrays(np.empty((0,)), np.empty((0,)), tolerance=0.0)
    assert empty == {"max_abs_error": 0.0, "mean_abs_error": 0.0, "rmse": 0.0, "passed": True}

    with pytest.raises(ValueError, match="shape mismatch"):
        edge_utils.compare_arrays(np.zeros((1,)), np.zeros((2,)), tolerance=0.1)

    nan_report = edge_utils.compare_arrays(np.array([np.nan]), np.array([0.0]), tolerance=0.1)
    assert nan_report["passed"] is False


def test_latency_csv_and_empty_summary(tmp_path):
    csv_path = tmp_path / "latency.csv"
    csv_path.write_text("image,latency_ms\na.jpg,10\nb.jpg,20.5\n", encoding="utf-8")
    assert edge_utils.read_latency_csv(csv_path) == [10.0, 20.5]
    total_path = tmp_path / "total.csv"
    total_path.write_text("tag,total_ms\na.jpg,4.5\n", encoding="utf-8")
    assert edge_utils.read_latency_csv(total_path) == [4.5]

    runtime_csv = tmp_path / "runtime.csv"
    runtime_csv.write_text(
        "tag,pre_ms,infer_ms,post_ms,total_ms,detections,mean_ms,p50_ms,p95_ms,p99_ms,fps\n"
        "a.jpg,1,5,1,7.5,2,,,,,\n"
        "#summary,,,,,,7.5,7.5,7.5,7.5,133.3\n",
        encoding="utf-8",
    )
    assert edge_utils.read_latency_csv(runtime_csv) == [7.5]

    invalid_csv = tmp_path / "invalid.csv"
    invalid_csv.write_text("image,inference_ms\na.jpg,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="latency_ms.*total_ms"):
        edge_utils.read_latency_csv(invalid_csv)

    assert edge_utils.summarize_latency_ms([]) == {
        "count": 0,
        "mean_ms": 0.0,
        "p50_ms": 0.0,
        "p95_ms": 0.0,
        "p99_ms": 0.0,
        "fps": 0.0,
    }
    with pytest.raises(ValueError, match="non-negative"):
        edge_utils.summarize_latency_ms([-1.0])


def test_profile_arg_parser_applies_explicit_threshold_overrides():
    parser = argparse.ArgumentParser()
    edge_utils.add_profile_args(parser)
    args = parser.parse_args(["--profile", "sku110k", "--conf", "0.31", "--iou", "0.71"])
    assert args.profile.name == "sku110k"
    assert args.conf == pytest.approx(0.31)
    assert args.iou == pytest.approx(0.71)


def test_profile_resolution_produces_canonical_visdrone_protocol():
    profile = edge_utils.get_profile("visdrone")
    resolved = edge_utils.resolve_profile_options(profile)
    assert resolved == {
        "imgsz": (640, 640),
        "conf": pytest.approx(0.001),
        "iou": pytest.approx(0.70),
        "max_det": 300,
        "multi_label": True,
        "letterbox": True,
    }
    overridden = edge_utils.resolve_profile_options(
        profile, imgsz=512, conf=0.2, iou=0.5, max_det=50, multi_label=False
    )
    assert overridden == {
        "imgsz": 512,
        "conf": pytest.approx(0.2),
        "iou": pytest.approx(0.5),
        "max_det": 50,
        "multi_label": False,
        "letterbox": True,
    }
    with pytest.raises(ValueError, match="finite"):
        edge_utils.resolve_profile_options(profile, conf=float("nan"))


def test_cpp_benchmark_declares_canonical_profile_and_evidence_sidecar():
    source = (EDGE_DIR / "cpp" / "edge_benchmark.cpp").read_text(encoding="utf-8")
    assert 'args.profile == "visdrone" ? 640 : 1280' in source
    assert 'args.profile == "visdrone" ? 0.001f : 0.25f' in source
    assert 'args.profile == "visdrone" ? 0.70f : 0.60f' in source
    assert 'args.profile == "sku110k"' in source
    assert 'args.profile == "visdrone" || args.profile == "sku110k"' in source
    assert "--multi-label" in source and "--single-label" in source
    assert "--max-det" in source and "--min-images" in source
    assert "write_json(args.json_output" in source
    assert "validate_unique_stems" in source
    assert "static_cast<unsigned char>(line[0]) == 0xef" in source
    assert "line.front() == line.back()" in source
    assert "line.front() == '\"' || line.front() == '\\''" in source


@pytest.mark.skipif(shutil.which("cmake") is None, reason="cmake is required for the C++ smoke test")
def test_cmake_stub_cli_contract(tmp_path):
    """The C++ target keeps a stable usage contract in both repository variants.

    The original scaffold builds ``edge_benchmark_stub.cpp`` and emits a
    dependency-free CSV row.  The fork this test is ported into already ships
    the full OpenCV/backend runner, whose CLI intentionally requires real model
    and image inputs.  Exercise the appropriate contract without pretending
    that a full runner is a stub.
    """
    build_dir = tmp_path / "build"
    cmake_text = (EDGE_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
    is_stub_target = (
        "cpp/edge_benchmark_stub.cpp" in cmake_text
        and "cpp/edge_benchmark.cpp" not in cmake_text
    )
    generator = []
    # On Windows, CMake may default to an unavailable Visual Studio/NMake
    # generator even when the bundled MinGW toolchain is present.
    if sys.platform == "win32" and shutil.which("mingw32-make"):
        generator = ["-G", "MinGW Makefiles"]
    elif sys.platform == "win32" and shutil.which("ninja"):
        generator = ["-G", "Ninja"]
    elif sys.platform == "win32":
        pytest.skip("no usable CMake generator found (MinGW Makefiles or Ninja)")
    configure = subprocess.run(
        ["cmake", *generator, "-S", str(EDGE_DIR), "-B", str(build_dir)],
        text=True,
        capture_output=True,
        check=False,
    )
    if configure.returncode != 0 and not is_stub_target:
        configure_output = configure.stdout + configure.stderr
        if "OpenCV" in configure_output and (
            "Could not find" in configure_output
            or "OpenCVConfig.cmake" in configure_output
            or "opencv-config.cmake" in configure_output
        ):
            pytest.skip("full runner requires the optional OpenCV C++ development package")
    assert configure.returncode == 0, configure.stdout + configure.stderr

    build = subprocess.run(
        ["cmake", "--build", str(build_dir), "--config", "Release"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert build.returncode == 0, build.stdout + build.stderr

    candidates = [
        build_dir / "yolo_master_edge_benchmark",
        build_dir / "yolo_master_edge_benchmark.exe",
        build_dir / "Release" / "yolo_master_edge_benchmark.exe",
    ]
    executable = next((path for path in candidates if path.exists()), None)
    assert executable is not None, f"compiled executable not found under {build_dir}"

    missing = subprocess.run([str(executable)], text=True, capture_output=True, check=False)
    assert missing.returncode != 0
    assert "Usage:" in missing.stderr or "Missing required" in missing.stderr

    if not is_stub_target:
        # The full runner must reject missing model/image inputs before trying
        # to initialize a backend.  Real inference is covered by the Ubuntu
        # smoke evidence and the production runtime contract tests.
        return

    run = subprocess.run(
        [
            str(executable),
            "--backend",
            "onnx",
            "--model",
            "model.onnx",
            "--images",
            "images.txt",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    rows = list(csv.DictReader(run.stdout.splitlines()))
    assert len(rows) == 1
    assert rows[0]["backend"] == "onnx"
    assert rows[0]["model"] == "model.onnx"
    assert rows[0]["images"] == "images.txt"
    assert float(rows[0]["latency_ms"]) >= 0.0
