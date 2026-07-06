"""Self-contained mAP50 / mAP50-95 evaluator (COCO 101-point interpolation).

Every backend — including the PyTorch baseline — is scored by THIS module, so the
ΔmAP reported by the consistency check reflects pure numerical divergence between
the original and exported graphs, never a difference in metric implementation.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

DetList = List[Tuple[int, float, np.ndarray]]  # (class_id, confidence, xyxy box)
ImgDets = Dict[str, DetList]
ImgGts = Dict[str, List[Tuple[int, np.ndarray]]]  # (class_id, xyxy box)

IOU_THRS = np.round(np.linspace(0.5, 0.95, 10), 2)


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (Na,4) and (Nb,4) xyxy boxes -> (Na, Nb)."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]
    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h
    area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def _ap_from_pr(rec: np.ndarray, prec: np.ndarray) -> float:
    """COCO-style 101-point interpolated AP."""
    rec_idx = np.linspace(0.0, 1.0, 101)
    # precision envelope (max to the right)
    prec = np.concatenate([[0.0], prec, [0.0]])
    rec = np.concatenate([[0.0], rec, [1.0]])
    for i in range(len(prec) - 1, 0, -1):
        prec[i - 1] = max(prec[i - 1], prec[i])
    ap = np.trapz(np.interp(rec_idx, rec, prec), rec_idx)
    return float(ap)


def _class_ap_at_iou(
    dets: Sequence[Tuple[str, float, np.ndarray]],
    gts_by_img: Dict[str, np.ndarray],
    iou_thr: float,
    image_ids: Sequence[str],
) -> float:
    """AP for one class at one IoU threshold.

    dets: list of (image_id, confidence, box) for this class.
    gts_by_img: {image_id: (M,4)} GT boxes of this class.
    """
    nd = len(dets)
    ng = sum(len(g) for g in gts_by_img.values())
    if ng == 0:
        return 0.0  # no GT for class -> undefined, counted as 0 contribution (excluded later)
    if nd == 0:
        return 0.0
    order = np.argsort(-np.array([d[1] for d in dets]))
    tp = np.zeros(nd, dtype=np.float32)
    fp = np.zeros(nd, dtype=np.float32)
    used = {iid: np.zeros(len(gts_by_img.get(iid, [])), dtype=bool) for iid in image_ids}
    for rank in order:
        iid, _, box = dets[rank]
        gts = gts_by_img.get(iid)
        if gts is None or len(gts) == 0:
            fp[rank] = 1
            continue
        ious = _iou_matrix(box[None], gts)[0]
        best = int(ious.argmax())
        if ious[best] >= iou_thr and not used[iid][best]:
            tp[rank] = 1
            used[iid][best] = True
        else:
            fp[rank] = 1
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / ng
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    return _ap_from_pr(rec, prec)


def evaluate_map(detections: ImgDets, gts: ImgGts, num_classes: int) -> Dict[str, float]:
    """Compute mAP50, mAP50-95 and per-class AP50.

    Images missing from `detections` are treated as producing no detections (correct).
    Classes with zero GT across the whole val set are excluded from the mean.
    """
    image_ids = sorted(set(list(gts.keys()) + list(detections.keys())))

    # Pre-bucket GT boxes per (class, image)
    gts_by_cls_img: Dict[int, Dict[str, np.ndarray]] = defaultdict(lambda: defaultdict(list))
    cls_with_gt = set()
    for iid, items in gts.items():
        for cls_id, box in items:
            gts_by_cls_img[cls_id][iid].append(box)
            cls_with_gt.add(cls_id)
    for c in gts_by_cls_img:
        for iid in gts_by_cls_img[c]:
            gts_by_cls_img[c][iid] = np.asarray(gts_by_cls_img[c][iid], dtype=np.float32)

    # Pre-bucket detections per class
    dets_by_cls: Dict[int, List[Tuple[str, float, np.ndarray]]] = defaultdict(list)
    for iid, items in detections.items():
        for cls_id, conf, box in items:
            dets_by_cls[cls_id].append((iid, float(conf), np.asarray(box, dtype=np.float32)))

    eval_classes = sorted(cls_with_gt)
    per_iou_maps = []
    per_class_ap50 = {}
    for iou_thr in IOU_THRS:
        aps = []
        for c in eval_classes:
            ap = _class_ap_at_iou(dets_by_cls.get(c, []), gts_by_cls_img[c], iou_thr, image_ids)
            aps.append(ap)
            if iou_thr == 0.5:
                per_class_ap50[c] = ap
        per_iou_maps.append(float(np.mean(aps)) if aps else 0.0)

    map50 = per_iou_maps[0]
    map_5095 = float(np.mean(per_iou_maps))
    return {"mAP50": map50, "mAP50-95": map_5095, "per_class_AP50": per_class_ap50}
