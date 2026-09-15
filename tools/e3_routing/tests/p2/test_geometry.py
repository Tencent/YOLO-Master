import numpy as np
import pytest
from PIL import Image

from e3_p2.geometry import LetterboxMeta, letterbox, restore_heatmap, validate_probability_grid


def test_letterbox_records_exact_padding_and_restores_original_shape():
    image = Image.new("RGB", (80, 40), "white")
    canvas, meta = letterbox(image, 64)
    assert canvas.shape == (64, 64, 3)
    assert (meta.resized_width, meta.resized_height) == (64, 32)
    assert (meta.pad_left, meta.pad_top, meta.pad_right, meta.pad_bottom) == (0, 16, 0, 16)
    restored = restore_heatmap(np.arange(32, dtype=np.float32).reshape(4, 8), meta)
    assert restored.shape == (40, 80)
    assert np.isfinite(restored).all()


def test_odd_padding_round_trip_shape_is_unambiguous():
    image = Image.new("RGB", (63, 40), "white")
    _, meta = letterbox(image, 64)
    assert meta.pad_left + meta.resized_width + meta.pad_right == 64
    assert meta.pad_top + meta.resized_height + meta.pad_bottom == 64
    restored = restore_heatmap(np.ones((5, 7), dtype=np.float32), meta)
    assert restored.shape == (40, 63)
    assert np.allclose(restored, 1.0)


def test_restore_heatmap_removes_letterbox_padding_before_resizing():
    image = Image.new("RGB", (80, 40), "white")
    _, meta = letterbox(image, 64)
    field = np.zeros((64, 64), dtype=np.float32)
    field[meta.pad_top : 64 - meta.pad_bottom, meta.pad_left : 64 - meta.pad_right] = 1.0
    restored = restore_heatmap(field, meta)
    assert np.allclose(restored, 1.0)


def test_probability_grid_requires_real_spatial_axes_and_normalization():
    valid = np.full((1, 4, 3, 5), 0.25, dtype=np.float32)
    result = validate_probability_grid(valid)
    assert result["shape"] == [1, 4, 3, 5]
    assert result["max_expert_sum_error"] == 0.0
    with pytest.raises(ValueError, match="not a token/spatial grid"):
        validate_probability_grid(np.full((1, 4, 1, 1), 0.25, dtype=np.float32))
    with pytest.raises(ValueError, match="sum to one"):
        validate_probability_grid(np.full((1, 4, 3, 5), 0.20, dtype=np.float32))


def test_invalid_crop_metadata_fails_closed():
    meta = LetterboxMeta(10, 10, 8, 0, 0, 4, 4, 4, 4, 0.8)
    with pytest.raises(ValueError, match="empty crop"):
        restore_heatmap(np.ones((2, 2), dtype=np.float32), meta)
