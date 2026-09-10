"""Letterbox geometry and truthful feature-grid to original-image mapping."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class LetterboxMeta:
    original_width: int
    original_height: int
    input_size: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    scale: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def letterbox(image: Image.Image, input_size: int, fill: int = 114) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize without distortion and record the exact integer padding transform."""

    if input_size <= 0:
        raise ValueError("input_size must be positive")
    rgb = image.convert("RGB")
    original_width, original_height = rgb.size
    if original_width <= 0 or original_height <= 0:
        raise ValueError("image dimensions must be positive")
    scale = min(input_size / original_width, input_size / original_height)
    resized_width = max(1, min(input_size, round(original_width * scale)))
    resized_height = max(1, min(input_size, round(original_height * scale)))
    pad_width = input_size - resized_width
    pad_height = input_size - resized_height
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    resized = rgb.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (input_size, input_size), color=(fill, fill, fill))
    canvas.paste(resized, (pad_left, pad_top))
    meta = LetterboxMeta(
        original_width=original_width,
        original_height=original_height,
        input_size=input_size,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        scale=scale,
    )
    return np.asarray(canvas, dtype=np.uint8), meta


def restore_heatmap(heatmap: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Map a real feature-grid scalar field through letterbox space back to original pixels."""

    field = np.asarray(heatmap, dtype=np.float32)
    if field.ndim != 2 or not field.size or not np.isfinite(field).all():
        raise ValueError("heatmap must be a finite non-empty 2D array")
    upsampled = Image.fromarray(field).resize((meta.input_size, meta.input_size), Image.Resampling.BILINEAR)
    right = meta.input_size - meta.pad_right
    bottom = meta.input_size - meta.pad_bottom
    if right <= meta.pad_left or bottom <= meta.pad_top:
        raise ValueError("letterbox metadata produces an empty crop")
    unpadded = upsampled.crop((meta.pad_left, meta.pad_top, right, bottom))
    restored = unpadded.resize((meta.original_width, meta.original_height), Image.Resampling.BILINEAR)
    return np.asarray(restored, dtype=np.float32)


def validate_probability_grid(weights: np.ndarray, tolerance: float = 1e-5) -> dict[str, float | list[int]]:
    """Validate `[B,E,H,W]` routing semantics; singleton spatial axes are rejected."""

    values = np.asarray(weights, dtype=np.float32)
    if values.ndim != 4:
        raise ValueError(f"expected [B,E,H,W], got shape={list(values.shape)}")
    if min(values.shape) <= 0 or values.shape[2] <= 1 or values.shape[3] <= 1:
        raise ValueError(f"not a token/spatial grid: shape={list(values.shape)}")
    if not np.isfinite(values).all():
        raise ValueError("routing grid contains non-finite values")
    if float(values.min()) < -tolerance:
        raise ValueError("routing grid contains negative probabilities")
    sums = values.sum(axis=1)
    max_sum_error = float(np.max(np.abs(sums - 1.0)))
    if max_sum_error > tolerance:
        raise ValueError(f"probabilities do not sum to one across experts: max_error={max_sum_error}")
    return {
        "shape": [int(item) for item in values.shape],
        "min": float(values.min()),
        "max": float(values.max()),
        "max_expert_sum_error": max_sum_error,
    }
