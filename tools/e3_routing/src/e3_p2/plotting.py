"""Dependency-light routing overlays and categorical expert maps."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import LetterboxMeta, restore_heatmap
from .regions import YoloBox, box_xyxy_original

COLORS = np.asarray(
    [
        [0, 229, 255],
        [151, 71, 255],
        [255, 196, 0],
        [44, 224, 123],
        [255, 82, 119],
        [79, 121, 255],
    ],
    dtype=np.float32,
)


def _probability_color(field: np.ndarray, color: np.ndarray) -> np.ndarray:
    clipped = np.clip(field, 0.0, 1.0)[..., None]
    dark = np.asarray([7, 14, 32], dtype=np.float32)
    return dark * (1.0 - clipped) + color * clipped


def save_probability_overlay(
    original: Image.Image,
    feature_probability: np.ndarray,
    meta: LetterboxMeta,
    destination: str,
    *,
    expert_index: int,
    alpha: float,
) -> dict[str, float]:
    """Blend a raw `[0,1]` expert probability with the original image using fixed scaling."""

    restored = restore_heatmap(feature_probability, meta)
    base = np.asarray(original.convert("RGB"), dtype=np.float32)
    heat = _probability_color(restored, COLORS[expert_index % len(COLORS)])
    strength = np.clip(restored, 0.0, 1.0)[..., None] * float(alpha)
    blended = np.clip(base * (1.0 - strength) + heat * strength, 0, 255).astype(np.uint8)
    Image.fromarray(blended).save(destination)
    return {
        "feature_min": float(np.min(feature_probability)),
        "feature_max": float(np.max(feature_probability)),
        "feature_mean": float(np.mean(feature_probability)),
        "restored_min": float(restored.min()),
        "restored_max": float(restored.max()),
    }


def save_metric_overlay(
    original: Image.Image,
    feature_field: np.ndarray,
    meta: LetterboxMeta,
    destination: str,
    *,
    color: tuple[int, int, int],
    alpha: float,
) -> dict[str, float]:
    """Blend a bounded routing diagnostic while preserving its absolute ``[0,1]`` scale."""

    values = np.asarray(feature_field, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("metric field must be a finite 2D array")
    if float(values.min()) < 0.0 or float(values.max()) > 1.0:
        raise ValueError("metric field must stay within [0,1]")
    restored = restore_heatmap(values, meta)
    base = np.asarray(original.convert("RGB"), dtype=np.float32)
    heat = _probability_color(restored, np.asarray(color, dtype=np.float32))
    strength = restored[..., None] * float(alpha)
    blended = np.clip(base * (1.0 - strength) + heat * strength, 0, 255).astype(np.uint8)
    Image.fromarray(blended).save(destination)
    return {
        "feature_min": float(values.min()),
        "feature_max": float(values.max()),
        "feature_mean": float(values.mean()),
        "restored_min": float(restored.min()),
        "restored_max": float(restored.max()),
    }


def save_dominant_overlay(
    original: Image.Image,
    weights: np.ndarray,
    meta: LetterboxMeta,
    destination: str,
    *,
    alpha: float,
) -> list[int]:
    """Show the argmax expert at each token without inventing intermediate values."""

    dominant = np.argmax(weights, axis=0).astype(np.int32)
    restored_planes = np.stack(
        [restore_heatmap((dominant == expert).astype(np.float32), meta) for expert in range(weights.shape[0])], axis=0
    )
    restored_dominant = np.argmax(restored_planes, axis=0)
    palette = COLORS[restored_dominant % len(COLORS)]
    base = np.asarray(original.convert("RGB"), dtype=np.float32)
    blended = np.clip(base * (1.0 - alpha) + palette * alpha, 0, 255).astype(np.uint8)
    Image.fromarray(blended).save(destination)
    return [int(np.sum(dominant == expert)) for expert in range(weights.shape[0])]


def save_overview(cards: list[dict[str, str]], destination: str) -> None:
    """Create a compact evidence contact sheet from representative overlays."""

    if not cards:
        raise ValueError("at least one card is required")
    thumb_width, thumb_height, caption_height = 360, 240, 54
    columns = 2
    rows = (len(cards) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + caption_height)), (7, 14, 32))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, card in enumerate(cards):
        image = Image.open(card["path"]).convert("RGB")
        image.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_width
        y = (index // columns) * (thumb_height + caption_height)
        sheet.paste(image, (x + (thumb_width - image.width) // 2, y))
        draw.text((x + 12, y + thumb_height + 8), card["caption"], fill=(230, 238, 255), font=font)
    sheet.save(destination)


def save_ground_truth_overlay(
    original: Image.Image, boxes: list[YoloBox], meta: LetterboxMeta, destination: str
) -> None:
    """Draw archived dataset boxes on an original-size copy for demo inspection."""

    rendered = original.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)
    line_width = max(2, round(min(rendered.size) / 220))
    font = ImageFont.load_default()
    for index, box in enumerate(boxes):
        coordinates = box_xyxy_original(box, meta)
        color = tuple(int(value) for value in COLORS[index % len(COLORS)])
        draw.rectangle(coordinates, outline=color, width=line_width)
        label = f"GT {box.class_id}"
        x1, y1, _, _ = coordinates
        text_box = draw.textbbox((x1, y1), label, font=font)
        top = max(0.0, y1 - (text_box[3] - text_box[1]) - 4)
        right = x1 + (text_box[2] - text_box[0]) + 6
        draw.rectangle((x1, top, right, y1), fill=color)
        draw.text((x1 + 3, top + 1), label, fill=(7, 14, 32), font=font)
    rendered.save(destination)
