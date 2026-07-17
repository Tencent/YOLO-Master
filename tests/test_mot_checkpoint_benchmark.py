"""Tests for trained-checkpoint, real-image MoT benchmarking."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/benchmark_mot_checkpoints.py"


def load_script():
    spec = importlib.util.spec_from_file_location("_mot_checkpoint_benchmark_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_checkpoint_benchmark_uses_all_real_images_and_preserves_grad_mode(tmp_path, monkeypatch):
    module = load_script()
    paths = []
    for index, shape in enumerate(((20, 40), (40, 20), (30, 30))):
        path = tmp_path / f"image_{index}.jpg"
        cv2.imwrite(str(path), np.full((*shape, 3), 127, dtype=np.uint8))
        paths.append(path)
    model = torch.nn.Conv2d(3, 4, 1).eval()
    monkeypatch.setattr(module, "profile_image_flops", lambda _model, _tensor: 0.001)
    torch.set_grad_enabled(True)

    latency, flops = module.benchmark_model(model, paths, "cpu", imgsz=32, warmup=1, flops_images=2)

    assert torch.is_grad_enabled()
    assert [Path(row["image"]).name for row in latency] == [path.name for path in paths]
    assert all(float(row["latency_ms"]) > 0 for row in latency)
    assert len(flops) == 2
    summary = module.summarize_benchmark("v10", tmp_path / "best.pt", model, latency, flops, "cpu", 32, 1)
    assert summary["images"] == 3
    assert summary["flops_images"] == 2
    assert summary["latency_ms_p50"] > 0


@pytest.mark.parametrize("warmup,flops", [(-1, 0), (0, -1)])
def test_checkpoint_benchmark_rejects_invalid_sampling(tmp_path, warmup, flops):
    module = load_script()
    path = tmp_path / "image.jpg"
    cv2.imwrite(str(path), np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="must be non-negative"):
        module.benchmark_model(torch.nn.Identity(), [path], "cpu", 8, warmup, flops)
