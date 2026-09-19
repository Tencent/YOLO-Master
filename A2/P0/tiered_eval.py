#!/usr/bin/env python3
"""Standalone COCO-style area-tiered AP/AR evaluation for VisDrone.

Runs batched inference over the val split with a trained checkpoint, then computes small/medium/large
AP and AR from GT box area. A detection is bucketed by the area of the GT it matches; detections that
match nothing are false positives in every tier, and detections matched to a GT outside a tier are
ignored for that tier (COCO semantics). Overall and tiered metrics come from this single code path, so
the table is self-consistent (and reused for the P1 on/off comparison).

Tiering uses this project's COCO-style 32²/96² thresholds (pixel area of GT box in the original val
image). This is NOT the VisDrone official definition — VisDrone DET has no area tiers and only reports
AP / AP50 / AP75 / AR@1/10/100/500:
    small   < 32^2
    medium  32^2 .. 96^2
    large   >= 96^2

The main STAL metric is APs = small-tier mAP@[.50:.95] at maxDets=500. This script also reports the
VisDrone-official-style overall AP / AP50 / AP75 / AR@500 plus per-tier APm / APl and ARs@500.

Usage:
    python A2/P0/tiered_eval.py --weights runs/a2/p0/visdrone-baseline-v01n/weights/best.pt \
                             --imgsz 800 --device 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Ensure this repo's ``ultralytics`` is imported (a stale editable install elsewhere would shadow it).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO
from ultralytics.utils import YAML

DATA_YAML = REPO_ROOT / "datasets" / "VisDrone.yaml"

IOU_THRS = np.round(np.linspace(0.5, 0.95, 10), 2)
AP75_IDX = int(np.where(IOU_THRS == 0.75)[0][0])  # index of IoU=0.75 within IOU_THRS
# (label, area_low, area_high); area_high=None means open-ended upper bound. COCO-style 32²/96².
DEFAULT_TIERS = [
    ("small", 0, 32**2),
    ("medium", 32**2, 96**2),
    ("large", 96**2, None),
]

# Inference chunk size (images per model.predict call); see _predict for why a single big call OOMs.
_PRED_CHUNK = 16


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (Na,4) and (Nb,4) xyxy boxes -> (Na, Nb)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a, b = a[:, None, :], b[None, :, :]
    inter = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None) * np.clip(
        np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None
    )
    area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    return np.where(area_a + area_b - inter > 0, inter / (area_a + area_b - inter), 0.0)


def _ap_from_pr(rec: np.ndarray, prec: np.ndarray) -> float:
    """COCO-style 101-point interpolated AP."""
    idx = np.linspace(0.0, 1.0, 101)
    prec = np.concatenate([[0.0], prec, [0.0]])
    rec = np.concatenate([[0.0], rec, [1.0]])
    for i in range(len(prec) - 1, 0, -1):
        prec[i - 1] = max(prec[i - 1], prec[i])
    return float(np.trapezoid(np.interp(idx, rec, prec), idx))


def _img_sizes(images_dir: Path) -> dict[str, tuple[int, int]]:
    """Map image stem -> (W, H) pixel size (needed to denormalize GT boxes)."""
    sizes = {}
    for p in images_dir.glob("*.jpg"):
        sizes[p.stem] = Image.open(p).size  # (W, H)
    return sizes


def _load_gt(labels_dir: Path, sizes: dict[str, tuple[int, int]]) -> dict[str, list[tuple[int, np.ndarray, float]]]:
    """Return {stem: [(cls, xyxy, area_px)]} from YOLO-format labels."""
    gts: dict[str, list[tuple[int, np.ndarray, float]]] = {}
    for p in labels_dir.glob("*.txt"):
        stem = p.stem
        if stem not in sizes:
            continue
        w, h = sizes[stem]
        items = []
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xc, yc, bw, bh = (float(x) for x in parts[1:5])
            x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
            x2, y2 = (xc + bw / 2) * w, (yc + bh / 2) * h
            items.append((cls, np.array([x1, y1, x2, y2], dtype=np.float32), (x2 - x1) * (y2 - y1)))
        gts[stem] = items
    return gts


def _predict(
    model, images: list[Path], device: str, imgsz: int, max_det: int
) -> dict[str, list[tuple[int, float, np.ndarray]]]:
    """Return {stem: [(cls, conf, xyxy)]} via chunked inference.

    A single ``model.predict`` over all 548 images OOMs: the predict harness retains GPU tensors across
    the inference generator until the whole call returns, so the full list accumulates. Chunking (and
    extracting each result to CPU within its chunk) bounds peak memory. ``rect=False`` keeps every image
    letterboxed to the same square ``imgsz`` (matching the square-input training val), which also avoids
    per-image rect-shape fragmentation.
    """
    dets: dict[str, list[tuple[int, float, np.ndarray]]] = {}
    for start in range(0, len(images), _PRED_CHUNK):
        chunk = images[start : start + _PRED_CHUNK]
        results = model.predict(
            source=[str(p) for p in chunk],
            conf=0.001,
            iou=0.7,
            imgsz=imgsz,
            device=device,
            max_det=max_det,
            batch=1,
            rect=False,
            verbose=False,
        )
        for img, r in zip(chunk, results):
            items = []
            if r.boxes is not None and len(r.boxes) > 0:
                for cls, conf, xyxy in zip(
                    r.boxes.cls.cpu().numpy(),
                    r.boxes.conf.cpu().numpy(),
                    r.boxes.xyxy.cpu().numpy(),
                ):
                    items.append((int(cls), float(conf), np.asarray(xyxy, dtype=np.float32)))
            dets[img.stem] = items
    return dets


def _match_class(dets_cls, gts_cls_img, image_ids, iou_thr):
    """Greedy per-image match (confidence desc). Returns detections sorted desc as (conf, matched_area|None)."""
    order = sorted(range(len(dets_cls)), key=lambda i: -dets_cls[i][1])
    used = {iid: np.zeros(len(gts_cls_img.get(iid, [])), dtype=bool) for iid in image_ids}
    out = []
    for rank in order:
        iid, conf, box = dets_cls[rank]
        gts = gts_cls_img.get(iid)
        matched_area = None
        if gts:
            boxes = np.stack([g[0] for g in gts])
            ious = _iou(box[None], boxes)[0]
            best = int(ious.argmax())
            if ious[best] >= iou_thr and not used[iid][best]:
                used[iid][best] = True
                matched_area = gts[best][1]
        out.append((conf, matched_area))
    return out


def _ap_for_area(out, gt_areas, lo, hi):
    """AP from pre-matched detections for the area range [lo, hi)."""
    npos = sum(1 for a in gt_areas if lo <= a < (hi if hi is not None else float("inf")))
    if npos == 0:
        return 0.0
    tp, fp = [], []
    for _, area in out:  # out already sorted desc by conf
        if area is None:
            tp.append(0)
            fp.append(1)
        elif lo <= area < (hi if hi is not None else float("inf")):
            tp.append(1)
            fp.append(0)
        # else: matched GT outside tier -> ignored
    tp = np.cumsum(np.asarray(tp, dtype=np.float32))
    fp = np.cumsum(np.asarray(fp, dtype=np.float32))
    rec = tp / npos
    prec = tp / np.maximum(tp + fp, 1e-9)
    return _ap_from_pr(rec, prec)


def _recall_for_area(out, gt_areas, lo, hi):
    """Recall (matched GT / total GT) for the area range [lo, hi), from pre-matched detections."""
    hi = hi if hi is not None else float("inf")
    npos = sum(1 for a in gt_areas if lo <= a < hi)
    if npos == 0:
        return 0.0
    matched = sum(1 for _, area in out if area is not None and lo <= area < hi)
    return matched / npos


def main() -> None:
    parser = argparse.ArgumentParser(description="VisDrone COCO-style tiered AP/AR evaluation")
    parser.add_argument("--weights", required=True, help="trained checkpoint, e.g. .../weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-det", type=int, default=500)
    args = parser.parse_args()

    data = YAML.load(DATA_YAML)
    root = Path(data["path"])
    images = sorted((root / "images" / "val").glob("*.jpg"))
    sizes = _img_sizes(root / "images" / "val")
    gts = _load_gt(root / "labels" / "val", sizes)

    print(f"Loading {args.weights} and inferring {len(images)} val images...")
    model = YOLO(args.weights)
    dets = _predict(model, images, args.device, args.imgsz, args.max_det)
    image_ids = sorted(set(gts) | set(dets))

    # Group GT per (class, image) and collect per-class GT areas.
    gts_cls_img: dict[int, dict[str, list[tuple[np.ndarray, float]]]] = {}
    gt_areas_cls: dict[int, list[float]] = {}
    for iid, items in gts.items():
        for cls, box, area in items:
            gts_cls_img.setdefault(cls, {}).setdefault(iid, []).append((box, area))
            gt_areas_cls.setdefault(cls, []).append(area)

    dets_cls: dict[int, list[tuple[str, float, np.ndarray]]] = {}
    for iid, items in dets.items():
        for cls, conf, box in items:
            dets_cls.setdefault(cls, []).append((iid, conf, box))

    classes = sorted(set(gt_areas_cls))
    tiers = DEFAULT_TIERS + [("overall", 0, None)]

    # Accumulate AP and AR per tier across IoU thresholds.
    tier_aps: dict[str, list[float]] = {label: [] for label, _, _ in tiers}
    tier_ars: dict[str, list[float]] = {label: [] for label, _, _ in tiers}
    for iou_thr in IOU_THRS:
        per_cls_ap = {label: [] for label, _, _ in tiers}
        per_cls_ar = {label: [] for label, _, _ in tiers}
        for c in classes:
            out = _match_class(dets_cls.get(c, []), gts_cls_img[c], image_ids, iou_thr)
            for label, lo, hi in tiers:
                per_cls_ap[label].append(_ap_for_area(out, gt_areas_cls[c], lo, hi))
                per_cls_ar[label].append(_recall_for_area(out, gt_areas_cls[c], lo, hi))
        for label, _, _ in tiers:
            tier_aps[label].append(float(np.mean(per_cls_ap[label])))
            tier_ars[label].append(float(np.mean(per_cls_ar[label])))

    tier_counts: dict[str, int] = {label: 0 for label, _, _ in tiers}
    for label, lo, hi in tiers:
        upper = hi if hi is not None else float("inf")
        tier_counts[label] = sum(1 for areas in gt_areas_cls.values() for a in areas if lo <= a < upper)

    # Report. APs (small mAP@[.50:.95]) is the main STAL metric.
    print("\n=== VisDrone AP/AR (maxDets=500) ===")
    print("overall row = VisDrone-official AP / AP50 / AP75 / AR500.")
    print(
        "small/medium/large = this project's COCO-style 32²/96² tiers (supplementary, NOT a VisDrone-official definition)."
    )
    print(f"{'tier':<8}{'instances':>10}{'AP50':>10}{'AP75':>10}{'AP50-95':>12}{'AR@500':>10}")
    for label, _, _ in tiers:
        map50 = tier_aps[label][0]
        map75 = tier_aps[label][AP75_IDX]
        map5095 = float(np.mean(tier_aps[label]))
        ar500 = float(np.mean(tier_ars[label]))
        print(f"{label:<8}{tier_counts[label]:>10}{map50:>10.4f}{map75:>10.4f}{map5095:>12.4f}{ar500:>10.4f}")

    print("\nmain STAL metric: APs = small-tier AP50-95 (report absolute-percentage-point deltas vs baseline).")
    print("AP50s is diagnostic only and is NOT used for P1 acceptance.")


if __name__ == "__main__":
    main()
