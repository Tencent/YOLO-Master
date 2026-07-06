"""Domain-specific preprocessing for VisDrone edge inference.

VisDrone is a UAV dataset: large source images (often >1500px) dominated by *tiny*
objects. Two domain concerns drive this module:

  1. Aspect-ratio preservation — naive square resize crushes horizontal/vertical
     structure; we use classic letterbox (min-side scale + pad) so the network sees
     the true aspect ratio.
  2. Determinism across backends — the EXACT same letterbox params (ratio + pad) are
     used for PyTorch / ONNX / NCNN / MNN, so any mAP delta is purely numerical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


LETTERBOX_FILL = 114  # standard YOLO pad grey


@dataclass(frozen=True)
class LetterboxResult:
    image: np.ndarray  # (imgsz, imgsz, 3) RGB float32, NCHW-ready
    ratio: float
    pad_w: float
    pad_h: float
    orig_h: int
    orig_w: int


def letterbox(image_bgr: np.ndarray, imgsz: int = 640) -> LetterboxResult:
    """Aspect-ratio-preserving resize + pad to (imgsz, imgsz).

    Returns the padded RGB image plus the geometric params needed to map detected
    boxes back into original image coordinates.
    """
    h, w = image_bgr.shape[:2]
    ratio = min(imgsz / h, imgsz / w)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # center-pad the resized image up to (imgsz, imgsz); store the INTEGER pad actually
    # applied so scale_boxes subtracts exactly what was added (no 0.5px drift).
    pad_w = (imgsz - new_w) / 2.0
    pad_h = (imgsz - new_h) / 2.0
    left = int(round(pad_w))
    top = int(round(pad_h))
    right = imgsz - new_w - left
    bottom = imgsz - new_h - top
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(LETTERBOX_FILL,) * 3)
    # BGR -> RGB, HWC -> CHW, 0..255 -> 0..1 (matches ultralytics export normalization)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return LetterboxResult(chw, ratio, float(left), float(top), h, w)


def to_nchw(chw: np.ndarray) -> np.ndarray:
    """Add batch dim -> (1, 3, H, W)."""
    return np.ascontiguousarray(chw[np.newaxis, ...])


def scale_boxes(boxes_xyxy: np.ndarray, lb: LetterboxResult) -> np.ndarray:
    """Map boxes from letterboxed image space back to original image space."""
    if len(boxes_xyxy) == 0:
        return boxes_xyxy
    out = boxes_xyxy.copy().astype(np.float32)
    out -= np.array([lb.pad_w, lb.pad_h, lb.pad_w, lb.pad_h], dtype=np.float32)
    out /= lb.ratio
    out[:, 0::2] = np.clip(out[:, 0::2], 0, lb.orig_w)
    out[:, 1::2] = np.clip(out[:, 1::2], 0, lb.orig_h)
    return out
