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
    max_det: int = 300
    multi_label: bool = False


PROFILES = {
    # The acceptance recipe is deliberately identical to the production
    # runner: square 640 input, aspect-preserving letterbox, low confidence
    # floor, class-aware NMS at IoU 0.70, and Ultralytics-style multi-label
    # decoding.  Callers can override values explicitly, but the resolved
    # values must be recorded with the resulting metrics.
    "visdrone": EdgeProfile("visdrone", (640, 640), 0.001, 0.70, True, 300, True),
    # Keep SKU-110K identical to the full runner and both evaluators.  A square
    # static input is required for cross-backend parity; callers that need a
    # rectangular deployment must make that protocol override explicit.
    "sku110k": EdgeProfile("sku110k", (1280, 1280), 0.25, 0.60, True, 300, True),
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
    if height <= 0 or width <= 0 or target_h <= 0 or target_w <= 0:
        raise ValueError("image and target dimensions must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    ratio = min(target_h / height, target_w / width)
    new_unpad = (max(1, int(round(width * ratio))), max(1, int(round(height * ratio))))
    pad_w = target_w - new_unpad[0]
    pad_h = target_h - new_unpad[1]
    if auto:
        pad_w %= stride
        pad_h %= stride
    # Ultralytics places the odd remainder on the right/bottom side with
    # ``round(dw / 2 - 0.1)`` (and ``+ 0.1`` for the opposite edge).  Keep the
    # same left/top convention as the native runner and calibration scripts.
    return ratio, new_unpad, (round(pad_w / 2 - 0.1), round(pad_h / 2 - 0.1))


def scale_xyxy_boxes(
    boxes: np.ndarray,
    original_shape: tuple[int, int],
    input_shape: tuple[int, int],
    pad: tuple[int, int],
    ratio: float,
) -> np.ndarray:
    """Map xyxy boxes from letterboxed network input back to original image coordinates."""
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    if len(original_shape) != 2 or any(dim <= 0 for dim in original_shape):
        raise ValueError("original_shape must contain positive height and width")
    if len(input_shape) != 2 or any(dim <= 0 for dim in input_shape):
        raise ValueError("input_shape must contain positive height and width")
    if len(pad) != 2 or any(dim < 0 for dim in pad):
        raise ValueError("pad must contain non-negative x/y values")
    if boxes.size == 0:
        return boxes.astype(np.float32, copy=True).reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes must have shape (N,4), got {boxes.shape}")
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
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: reference {reference.shape}, candidate {candidate.shape}")
    reference = reference.astype(np.float32, copy=False)
    candidate = candidate.astype(np.float32, copy=False)
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        return {"max_abs_error": float("inf"), "mean_abs_error": float("inf"), "rmse": float("inf"), "passed": False}
    diff = np.abs(reference - candidate)
    max_error = float(diff.max()) if diff.size else 0.0
    return {
        "max_abs_error": max_error,
        "mean_abs_error": float(diff.mean() if diff.size else 0.0),
        "rmse": float(math.sqrt(float((diff ** 2).mean())) if diff.size else 0.0),
        "passed": bool(max_error <= tolerance),
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))
    return float(ordered[idx])


def summarize_latency_ms(values: Iterable[float]) -> dict[str, float]:
    data = [float(v) for v in values]
    if any(not math.isfinite(value) or value < 0 for value in data):
        raise ValueError("latency values must be finite and non-negative")
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
    """Read per-image latency values from either scaffold or runtime CSV output.

    The dependency-free C++ scaffold uses ``latency_ms`` while the full edge
    runner records end-to-end latency as ``total_ms``. Accepting both keeps the
    reporting helper independent of which backend runner produced the file.
    """
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        field = next((name for name in ("latency_ms", "total_ms") if name in (reader.fieldnames or [])), None)
        if field is None:
            raise ValueError("latency CSV must contain a 'latency_ms' or 'total_ms' column")
        values = []
        for row in reader:
            raw = (row.get(field) or "").strip()
            if not raw:
                continue
            value = float(raw)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid latency value: {raw!r}")
            values.append(value)
        return values


def profile_arg(value: str) -> EdgeProfile:
    return get_profile(value)


def add_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", type=profile_arg, default=get_profile("visdrone"), help="Vertical profile")
    parser.add_argument("--conf", type=float, default=None, help="Override profile confidence threshold")
    parser.add_argument("--iou", type=float, default=None, help="Override profile NMS IoU threshold")


def resolve_profile_options(
    profile: EdgeProfile,
    *,
    imgsz: int | tuple[int, int] | None = None,
    conf: float | None = None,
    iou: float | None = None,
    max_det: int | None = None,
    multi_label: bool | None = None,
) -> dict[str, object]:
    """Resolve optional overrides into one serialisable protocol dictionary.

    Keeping this resolution in the dependency-light module prevents exporters,
    benchmark wrappers, and validation scripts from silently drifting apart.
    ``imgsz`` may be a scalar (square input) or an explicit ``(height,width)``.
    """
    if imgsz is None:
        image_size: int | tuple[int, int] = profile.image_size
    elif isinstance(imgsz, tuple):
        if len(imgsz) != 2 or any(int(value) <= 0 for value in imgsz):
            raise ValueError("imgsz tuple must contain two positive dimensions")
        image_size = (int(imgsz[0]), int(imgsz[1]))
    else:
        image_size = int(imgsz)
        if image_size <= 0:
            raise ValueError("imgsz must be positive")
    resolved_conf = profile.conf_threshold if conf is None else float(conf)
    resolved_iou = profile.iou_threshold if iou is None else float(iou)
    resolved_max_det = profile.max_det if max_det is None else int(max_det)
    resolved_multi = profile.multi_label if multi_label is None else bool(multi_label)
    if not math.isfinite(resolved_conf) or not 0.0 <= resolved_conf <= 1.0:
        raise ValueError("conf must be finite and in [0,1]")
    if not math.isfinite(resolved_iou) or not 0.0 <= resolved_iou <= 1.0:
        raise ValueError("iou must be finite and in [0,1]")
    if resolved_max_det <= 0:
        raise ValueError("max_det must be positive")
    return {
        "imgsz": image_size,
        "conf": resolved_conf,
        "iou": resolved_iou,
        "max_det": resolved_max_det,
        "multi_label": resolved_multi,
        "letterbox": bool(profile.keep_aspect_ratio),
    }
