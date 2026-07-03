"""Utilities for YOLO-Master vertical edge deployment examples."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class EdgeProfile:
    name: str
    image_size: tuple[int, int]
    conf_threshold: float
    iou_threshold: float
    keep_aspect_ratio: bool = True


PROFILES = {
    "visdrone": EdgeProfile("visdrone", (960, 544), 0.20, 0.55),
    "sku110k": EdgeProfile("sku110k", (1280, 768), 0.25, 0.60),
}


def get_profile(name: str) -> EdgeProfile:
    try:
        return PROFILES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown profile {name!r}; choose one of {sorted(PROFILES)}") from exc


def letterbox_shape(
    shape: tuple[int, int],
    new_shape: tuple[int, int],
    stride: int = 32,
    auto: bool = False,
) -> tuple[float, tuple[int, int], tuple[int, int]]:
    """Return resize ratio, unpadded shape, and half-padding for letterbox preprocessing."""
    height, width = shape
    target_h, target_w = new_shape
    ratio = min(target_h / height, target_w / width)
    new_unpad = (int(round(width * ratio)), int(round(height * ratio)))
    pad_w = target_w - new_unpad[0]
    pad_h = target_h - new_unpad[1]
    if auto:
        pad_w %= stride
        pad_h %= stride
    return ratio, new_unpad, (pad_w // 2, pad_h // 2)


def scale_xyxy_boxes(
    boxes: np.ndarray,
    original_shape: tuple[int, int],
    input_shape: tuple[int, int],
    pad: tuple[int, int],
    ratio: float,
) -> np.ndarray:
    """Map xyxy boxes from letterboxed network input back to original image coordinates."""
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    out = boxes.astype(np.float32).copy()
    out[:, [0, 2]] -= pad[0]
    out[:, [1, 3]] -= pad[1]
    out[:, :4] /= ratio
    h, w = original_shape
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
    return out


def compare_arrays(reference: np.ndarray, candidate: np.ndarray, tolerance: float) -> dict[str, float | bool]:
    """Compare two backend output tensors."""
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: reference {reference.shape}, candidate {candidate.shape}")
    diff = np.abs(reference.astype(np.float32) - candidate.astype(np.float32))
    return {
        "max_abs_error": float(diff.max(initial=0.0)),
        "mean_abs_error": float(diff.mean() if diff.size else 0.0),
        "rmse": float(math.sqrt(float((diff ** 2).mean())) if diff.size else 0.0),
        "passed": bool(diff.max(initial=0.0) <= tolerance),
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))
    return float(ordered[idx])


def summarize_latency_ms(values: Iterable[float]) -> dict[str, float]:
    data = [float(v) for v in values]
    if not data:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "fps": 0.0}
    avg = mean(data)
    return {
        "count": len(data),
        "mean_ms": float(avg),
        "p50_ms": float(median(data)),
        "p95_ms": percentile(data, 95),
        "p99_ms": percentile(data, 99),
        "fps": float(1000.0 / avg) if avg > 0 else 0.0,
    }


def read_latency_csv(path: Path) -> list[float]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [float(row["latency_ms"]) for row in rows if row.get("latency_ms")]


def profile_arg(value: str) -> EdgeProfile:
    return get_profile(value)


def add_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", type=profile_arg, default=get_profile("visdrone"), help="Vertical profile")
    parser.add_argument("--conf", type=float, default=None, help="Override profile confidence threshold")
    parser.add_argument("--iou", type=float, default=None, help="Override profile NMS IoU threshold")
