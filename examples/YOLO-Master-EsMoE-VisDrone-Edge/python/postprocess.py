"""Post-processing for YOLO-Master-EsMoE edge inference, with VisDrone domain tuning.

The exported Detect head emits a tensor of shape (1, 4 + nc, N) where the first 4
channels are [cx, cy, w, h] in letterboxed-image pixels and the remaining nc channels
are class confidence scores. This module turns that into final boxes.

Domain tuning (issue #51 — "VisDrone 小目标可能需要更低 conf 阈值"):
  * An *area-adaptive* confidence threshold: boxes whose area is below `small_area`
    are admitted with a lower confidence (`small_conf`) than the default (`conf`).
    This recovers tiny pedestrians/cars that a uniform 0.25 threshold would drop.
  * Standard per-class greedy NMS at `iou`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class NmsConfig:
    conf: float = 0.15          # VisDrone-tuned: lower than COCO's 0.25 to keep small objects
    small_conf: float = 0.05    # even lower threshold for tiny boxes (< small_area px^2)
    small_area: float = 32 * 32 # boxes smaller than 32x32 (in letterboxed px) count as "small"
    iou: float = 0.45           # NMS IoU threshold
    max_det: int = 300
    num_classes: int = 10


def _xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    x1 = b[..., 0] - b[..., 2] / 2
    y1 = b[..., 1] - b[..., 3] / 2
    x2 = b[..., 0] + b[..., 2] / 2
    y2 = b[..., 1] + b[..., 3] / 2
    return np.stack([x1, y1, x2, y2], axis=-1)


def _nms_per_class(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, max_det: int) -> np.ndarray:
    """Greedy per-class NMS. boxes:(M,4) xyxy, scores:(M,) -> kept indices."""
    if len(boxes) == 0:
        return np.empty(0, dtype=int)
    order = scores.argsort()[::-1]
    keep = []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip((x2 - x1), 0, None) * np.clip((y2 - y1), 0, None)
    while order.size > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = order[1:][iou < iou_thr]
    return np.array(keep, dtype=int)


def decode_and_nms(raw: np.ndarray, cfg: NmsConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """raw: (1, 4+nc, N) or (4+nc, N) -> (boxes_xyxy[M,4], scores[M], cls[M]).

    Applies area-adaptive confidence filtering then per-class NMS, all in
    letterboxed-image pixel space (scale back to original coords separately).
    """
    arr = raw.reshape(-1, raw.shape[-1]) if raw.ndim == 3 else raw
    # arr: (4+nc, N) -> (N, 4+nc)
    arr = arr.T
    boxes_xywh = arr[:, :4]
    class_scores = arr[:, 4:4 + cfg.num_classes]
    cls_idx = class_scores.argmax(axis=1)
    top_score = class_scores[np.arange(len(cls_idx)), cls_idx]

    boxes_xyxy = _xywh_to_xyxy(boxes_xywh)
    areas = np.clip(boxes_xyxy[:, 2] - boxes_xyxy[:, 0], 0, None) * np.clip(
        boxes_xyxy[:, 3] - boxes_xyxy[:, 1], 0, None
    )
    # Area-adaptive confidence gate: small boxes get the lower small_conf threshold.
    thr = np.where(areas < cfg.small_area, cfg.small_conf, cfg.conf)
    mask = top_score >= thr
    if not mask.any():
        return (np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=int))

    boxes_xyxy = boxes_xyxy[mask]
    top_score = top_score[mask]
    cls_idx = cls_idx[mask]

    final_boxes = []
    final_scores = []
    final_cls = []
    for c in range(cfg.num_classes):
        cmask = cls_idx == c
        if not cmask.any():
            continue
        keep = _nms_per_class(boxes_xyxy[cmask], top_score[cmask], cfg.iou, cfg.max_det)
        if len(keep) == 0:
            continue
        final_boxes.append(boxes_xyxy[cmask][keep])
        final_scores.append(top_score[cmask][keep])
        final_cls.append(cls_idx[cmask][keep])

    if not final_boxes:
        return (np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=int))
    boxes = np.concatenate(final_boxes).astype(np.float32)
    scores = np.concatenate(final_scores).astype(np.float32)
    clsarr = np.concatenate(final_cls).astype(int)
    # global max_det cap (per-class NMS above can yield up to max_det*num_classes)
    if len(boxes) > cfg.max_det:
        order = np.argsort(-scores)[: cfg.max_det]
        boxes, scores, clsarr = boxes[order], scores[order], clsarr[order]
    return boxes, scores, clsarr
