"""Ground-truth region masks and region-conditioned routing diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .capture import routing_metric_fields
from .geometry import LetterboxMeta, validate_probability_grid


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "class_id": self.class_id,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "width": self.width,
            "height": self.height,
        }


def label_path_for_image(image_path: Path) -> Path:
    """Resolve the conventional YOLO ``images/...`` to ``labels/...`` path."""

    resolved = image_path.resolve()
    parts = list(resolved.parts)
    image_positions = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not image_positions:
        raise ValueError(f"image path has no images directory component: {resolved}")
    parts[image_positions[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def parse_yolo_labels(path: Path) -> list[YoloBox]:
    """Parse strict normalized detection labels without silently clipping invalid data."""

    if not path.is_file():
        raise FileNotFoundError(f"YOLO label file is missing: {path}")
    boxes: list[YoloBox] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields, got {len(fields)}")
        try:
            class_value, center_x, center_y, width, height = (float(value) for value in fields)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: non-numeric label field") from error
        class_id = int(class_value)
        if class_value != class_id or class_id < 0:
            raise ValueError(f"{path}:{line_number}: class id must be a non-negative integer")
        if not all(np.isfinite(value) for value in (center_x, center_y, width, height)):
            raise ValueError(f"{path}:{line_number}: label values must be finite")
        if not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0):
            raise ValueError(f"{path}:{line_number}: box center must be within [0,1]")
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise ValueError(f"{path}:{line_number}: box size must be within (0,1]")
        boxes.append(YoloBox(class_id, center_x, center_y, width, height))
    return boxes


def box_xyxy_original(box: YoloBox, meta: LetterboxMeta) -> tuple[float, float, float, float]:
    """Return a normalized YOLO box as clipped original-image pixel coordinates."""

    x1 = (box.center_x - box.width / 2.0) * meta.original_width
    y1 = (box.center_y - box.height / 2.0) * meta.original_height
    x2 = (box.center_x + box.width / 2.0) * meta.original_width
    y2 = (box.center_y + box.height / 2.0) * meta.original_height
    return (
        max(0.0, min(float(meta.original_width), x1)),
        max(0.0, min(float(meta.original_height), y1)),
        max(0.0, min(float(meta.original_width), x2)),
        max(0.0, min(float(meta.original_height), y2)),
    )


def token_region_masks(
    boxes: list[YoloBox], meta: LetterboxMeta, height: int, width: int
) -> dict[str, np.ndarray]:
    """Classify feature-cell centers as foreground, background, or padding.

    Foreground means the center of a valid (non-padding) token falls inside at
    least one ground-truth box. Background includes only valid image tokens;
    letterbox padding is intentionally excluded from both groups.
    """

    if height <= 0 or width <= 0:
        raise ValueError("feature-grid dimensions must be positive")
    x = (np.arange(width, dtype=np.float64) + 0.5) * meta.input_size / width
    y = (np.arange(height, dtype=np.float64) + 0.5) * meta.input_size / height
    xx, yy = np.meshgrid(x, y)
    image_right = meta.pad_left + meta.resized_width
    image_bottom = meta.pad_top + meta.resized_height
    valid = (
        (xx >= meta.pad_left)
        & (xx < image_right)
        & (yy >= meta.pad_top)
        & (yy < image_bottom)
    )
    foreground = np.zeros((height, width), dtype=bool)
    scale_x = meta.resized_width / meta.original_width
    scale_y = meta.resized_height / meta.original_height
    for box in boxes:
        x1, y1, x2, y2 = box_xyxy_original(box, meta)
        x1 = meta.pad_left + x1 * scale_x
        x2 = meta.pad_left + x2 * scale_x
        y1 = meta.pad_top + y1 * scale_y
        y2 = meta.pad_top + y2 * scale_y
        foreground |= valid & (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)
    background = valid & ~foreground
    padding = ~valid
    if np.any(foreground & background) or np.any((foreground | background) & padding):
        raise RuntimeError("region masks overlap")
    if not np.array_equal(foreground | background | padding, np.ones_like(valid)):
        raise RuntimeError("region masks do not partition the feature grid")
    return {"foreground": foreground, "background": background, "padding": padding}


def _masked_summary(weights: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    token_count = int(mask.sum())
    expert_count = int(weights.shape[0])
    if token_count == 0:
        return {
            "token_count": 0,
            "mean_expert_probability": None,
            "dominant_token_count": [0] * expert_count,
            "dominant_token_fraction": None,
            "normalized_entropy_mean": None,
            "top1_margin_mean": None,
        }
    fields = routing_metric_fields(weights)
    dominant = np.argmax(weights, axis=0)
    counts = [int(np.sum((dominant == expert) & mask)) for expert in range(expert_count)]
    return {
        "token_count": token_count,
        "mean_expert_probability": [float(weights[expert][mask].mean()) for expert in range(expert_count)],
        "dominant_token_count": counts,
        "dominant_token_fraction": [count / token_count for count in counts],
        "normalized_entropy_mean": float(fields["normalized_entropy"][mask].mean()),
        "top1_margin_mean": float(fields["top1_margin"][mask].mean()),
    }


def region_routing_diagnostics(weights: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, Any]:
    """Compare routing on labelled-object tokens with valid background tokens."""

    values = np.asarray(weights, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"expected [E,H,W], got shape={list(values.shape)}")
    validate_probability_grid(values[None, ...])
    expected_shape = values.shape[1:]
    for name in ("foreground", "background", "padding"):
        if name not in masks or masks[name].shape != expected_shape or masks[name].dtype != np.bool_:
            raise ValueError(f"{name} mask must be boolean with shape={list(expected_shape)}")
    foreground = _masked_summary(values, masks["foreground"])
    background = _masked_summary(values, masks["background"])
    valid_count = foreground["token_count"] + background["token_count"]
    status = "SUPPORTED" if foreground["token_count"] and background["token_count"] else "INSUFFICIENT_TOKENS"
    contrast: dict[str, Any] | None = None
    if status == "SUPPORTED":
        fg_prob = np.asarray(foreground["mean_expert_probability"], dtype=np.float64)
        bg_prob = np.asarray(background["mean_expert_probability"], dtype=np.float64)
        mean_prob = 0.5 * (fg_prob + bg_prob)

        def kl(left: np.ndarray, right: np.ndarray) -> float:
            keep = left > 0
            return float(np.sum(left[keep] * np.log(left[keep] / right[keep])))

        contrast = {
            "foreground_minus_background_expert_probability": [float(item) for item in fg_prob - bg_prob],
            "total_variation_distance": float(0.5 * np.abs(fg_prob - bg_prob).sum()),
            "jensen_shannon_divergence_nats": float(0.5 * kl(fg_prob, mean_prob) + 0.5 * kl(bg_prob, mean_prob)),
            "foreground_minus_background_entropy": float(
                foreground["normalized_entropy_mean"] - background["normalized_entropy_mean"]
            ),
            "foreground_minus_background_top1_margin": float(
                foreground["top1_margin_mean"] - background["top1_margin_mean"]
            ),
        }
    return {
        "status": status,
        "assignment_rule": "valid token center inside any ground-truth box",
        "padding_policy": "excluded from both foreground and background",
        "foreground": foreground,
        "background": background,
        "padding_token_count": int(masks["padding"].sum()),
        "valid_token_count": valid_count,
        "contrast": contrast,
    }


def aggregate_region_diagnostics(captures: list[dict[str, Any]]) -> dict[str, Any]:
    """Build token-weighted region summaries by family and module."""

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        supported = [item for item in items if item["region_diagnostics"]["status"] == "SUPPORTED"]
        expert_count = len(items[0]["region_diagnostics"]["foreground"]["dominant_token_count"])

        def region(name: str) -> dict[str, Any]:
            entries = [item["region_diagnostics"][name] for item in items]
            count = sum(entry["token_count"] for entry in entries)
            if not count:
                return {"token_count": 0, "mean_expert_probability": None, "dominant_token_count": [0] * expert_count,
                        "dominant_token_fraction": None, "normalized_entropy_mean": None, "top1_margin_mean": None}
            probabilities = [
                sum(entry["mean_expert_probability"][expert] * entry["token_count"] for entry in entries if entry["token_count"])
                / count
                for expert in range(expert_count)
            ]
            dominant = [sum(entry["dominant_token_count"][expert] for entry in entries) for expert in range(expert_count)]
            return {
                "token_count": count,
                "mean_expert_probability": probabilities,
                "dominant_token_count": dominant,
                "dominant_token_fraction": [value / count for value in dominant],
                "normalized_entropy_mean": sum(
                    entry["normalized_entropy_mean"] * entry["token_count"] for entry in entries if entry["token_count"]
                ) / count,
                "top1_margin_mean": sum(
                    entry["top1_margin_mean"] * entry["token_count"] for entry in entries if entry["token_count"]
                ) / count,
            }

        foreground = region("foreground")
        background = region("background")
        pooled_contrast = None
        if foreground["token_count"] and background["token_count"]:
            fg_prob = np.asarray(foreground["mean_expert_probability"], dtype=np.float64)
            bg_prob = np.asarray(background["mean_expert_probability"], dtype=np.float64)
            mean_prob = 0.5 * (fg_prob + bg_prob)

            def kl(left: np.ndarray, right: np.ndarray) -> float:
                keep = left > 0
                return float(np.sum(left[keep] * np.log(left[keep] / right[keep])))

            pooled_contrast = {
                "foreground_minus_background_expert_probability": [float(item) for item in fg_prob - bg_prob],
                "total_variation_distance": float(0.5 * np.abs(fg_prob - bg_prob).sum()),
                "jensen_shannon_divergence_nats": float(0.5 * kl(fg_prob, mean_prob) + 0.5 * kl(bg_prob, mean_prob)),
                "foreground_minus_background_entropy": float(
                    foreground["normalized_entropy_mean"] - background["normalized_entropy_mean"]
                ),
                "foreground_minus_background_top1_margin": float(
                    foreground["top1_margin_mean"] - background["top1_margin_mean"]
                ),
            }
        paired_contrast = None
        if supported:
            contrasts = [item["region_diagnostics"]["contrast"] for item in supported]

            def distribution(key: str) -> dict[str, float]:
                values = np.asarray([contrast[key] for contrast in contrasts], dtype=np.float64)
                return {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}

            paired_contrast = {
                "capture_count": len(supported),
                "weighting": "each supported capture receives equal weight",
                "foreground_minus_background_expert_probability_mean": [
                    float(
                        np.mean(
                            [contrast["foreground_minus_background_expert_probability"][expert] for contrast in contrasts]
                        )
                    )
                    for expert in range(expert_count)
                ],
                "total_variation_distance": distribution("total_variation_distance"),
                "jensen_shannon_divergence_nats": distribution("jensen_shannon_divergence_nats"),
                "foreground_minus_background_entropy": distribution("foreground_minus_background_entropy"),
                "foreground_minus_background_top1_margin": distribution(
                    "foreground_minus_background_top1_margin"
                ),
            }
        return {
            "capture_count": len(items),
            "supported_capture_count": len(supported),
            "insufficient_capture_count": len(items) - len(supported),
            "foreground": foreground,
            "background": background,
            "padding_token_count": sum(item["region_diagnostics"]["padding_token_count"] for item in items),
            "pooled_contrast": pooled_contrast,
            "paired_capture_contrast": paired_contrast,
        }

    families = sorted({item["family"] for item in captures})
    modules = sorted({(item["family"], item["module"]) for item in captures})
    return {
        "method": {
            "assignment_rule": "valid token center inside any ground-truth box",
            "padding_policy": "excluded from both foreground and background",
            "aggregation": "token-weighted within each family/module",
            "contrast_reporting": "pooled token contrast plus equal-weight paired-capture descriptive distribution",
            "primary_conclusion_basis": "paired_capture_contrast; pooled_contrast is diagnostic only",
            "inference_boundary": "descriptive random-initialization baseline; no learned specialization claim",
        },
        "capture_count": len(captures),
        "by_family": {
            family: summarize([item for item in captures if item["family"] == family]) for family in families
        },
        "by_module": {
            f"{family}:{module}": summarize(
                [item for item in captures if item["family"] == family and item["module"] == module]
            )
            for family, module in modules
        },
    }
