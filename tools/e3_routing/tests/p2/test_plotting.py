from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from e3_p2.geometry import letterbox
from e3_p2.plotting import save_dominant_overlay, save_metric_overlay, save_probability_overlay


def test_overlay_outputs_original_size_and_reports_raw_probability(tmp_path: Path):
    original = Image.new("RGB", (80, 40), (120, 130, 140))
    _, meta = letterbox(original, 64)
    probability = np.linspace(0, 1, 20, dtype=np.float32).reshape(4, 5)
    output = tmp_path / "probability.png"
    stats = save_probability_overlay(original, probability, meta, str(output), expert_index=1, alpha=0.5)
    assert Image.open(output).size == original.size
    assert stats["feature_min"] == 0.0
    assert stats["feature_max"] == 1.0


def test_dominant_overlay_counts_match_feature_tokens(tmp_path: Path):
    original = Image.new("RGB", (50, 50), "black")
    _, meta = letterbox(original, 32)
    weights = np.zeros((3, 2, 2), dtype=np.float32)
    weights[0, 0, :] = 1
    weights[1, 1, 0] = 1
    weights[2, 1, 1] = 1
    output = tmp_path / "dominant.png"
    counts = save_dominant_overlay(original, weights, meta, str(output), alpha=0.5)
    assert counts == [2, 1, 1]
    assert Image.open(output).size == original.size


def test_metric_overlay_preserves_fixed_scale_and_original_size(tmp_path: Path):
    original = Image.new("RGB", (63, 40), "black")
    _, meta = letterbox(original, 64)
    output = tmp_path / "entropy.png"
    stats = save_metric_overlay(
        original,
        np.full((4, 5), 0.75, dtype=np.float32),
        meta,
        str(output),
        color=(166, 108, 255),
        alpha=0.5,
    )
    assert Image.open(output).size == original.size
    assert stats["feature_mean"] == pytest.approx(0.75)
    with pytest.raises(ValueError, match="within"):
        save_metric_overlay(
            original,
            np.full((4, 5), 1.01, dtype=np.float32),
            meta,
            str(tmp_path / "bad.png"),
            color=(0, 0, 0),
            alpha=0.5,
        )
