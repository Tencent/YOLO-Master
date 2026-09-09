"""Tests for the A2OR frozen-checkpoint resolution probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_probe_module():
    path = Path(__file__).resolve().parents[1] / "A2OR" / "probe_resolution_bottleneck.py"
    spec = importlib.util.spec_from_file_location("probe_resolution_bottleneck", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _result(ap: float, aps: float) -> dict:
    return {
        "coco_max_dets_100": {
            "AP_all": ap,
            "AP_50": ap + 0.1,
            "AP_75": ap - 0.02,
            "AP_small": aps,
            "AP_medium": ap + 0.05,
            "AP_large": ap + 0.1,
        }
    }


def test_delta_points_are_absolute_percentage_points():
    probe = _load_probe_module()

    delta = probe.delta_points(_result(ap=0.21, aps=0.13), _result(ap=0.20, aps=0.12))

    assert abs(delta["AP_all"] - 1.0) < 1e-9
    assert abs(delta["AP_small"] - 1.0) < 1e-9


def test_resolution_signal_uses_aps_gate():
    probe = _load_probe_module()
    delta = probe.delta_points(_result(ap=0.201, aps=0.126), _result(ap=0.20, aps=0.12))

    signal = probe.resolution_signal(delta, aps_gate=0.5)

    assert signal["passed"] is True
    assert abs(signal["aps_delta_points"] - 0.6) < 1e-9
