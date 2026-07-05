import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EDGE_UTILS = ROOT / "examples" / "YOLO-Master-Edge-Deployment" / "edge_utils.py"

spec = importlib.util.spec_from_file_location("edge_utils", EDGE_UTILS)
edge_utils = importlib.util.module_from_spec(spec)
sys.modules["edge_utils"] = edge_utils
spec.loader.exec_module(edge_utils)


def test_letterbox_profile_keeps_aspect_ratio():
    ratio, new_unpad, pad = edge_utils.letterbox_shape((540, 960), edge_utils.get_profile("visdrone").image_size)
    assert ratio > 0
    assert new_unpad[0] <= 960
    assert new_unpad[1] <= 544
    assert pad[0] >= 0 and pad[1] >= 0


def test_scale_xyxy_boxes_clips_to_original_shape():
    boxes = np.array([[10, 20, 2000, 1200]], dtype=np.float32)
    scaled = edge_utils.scale_xyxy_boxes(boxes, original_shape=(720, 1280), input_shape=(960, 960), pad=(0, 0), ratio=0.75)
    assert scaled.shape == (1, 4)
    assert scaled[0, 2] <= 1280
    assert scaled[0, 3] <= 720


def test_compare_arrays_reports_pass_fail():
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    cand = np.array([1.0, 2.004, 3.0], dtype=np.float32)
    report = edge_utils.compare_arrays(ref, cand, tolerance=0.005)
    assert report["passed"] is True
    assert report["max_abs_error"] > 0


def test_latency_summary_percentiles_and_fps():
    report = edge_utils.summarize_latency_ms([10, 20, 30, 40, 50])
    assert report["count"] == 5
    assert report["p50_ms"] == 30
    assert report["p95_ms"] == 50
    assert report["fps"] > 0
