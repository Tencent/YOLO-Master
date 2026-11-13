#!/usr/bin/env python3
"""OV-Distill: coco128 verification -- teacher annotation quality vs. GT.

Metrics (IoU >= 0.5, greedy matching):
  - positive queries (present in GT): box-level precision / recall / F1
  - negative queries (absent from GT): rejection rate = fraction of queries
    for which the teacher produced no boxes
  - end-to-end: positive-query recall also captures "an entire class missed"

Usage:
  python scripts/ov_distill/eval_coco128.py \
      --manifest runs/ov_distill_coco128/manifest.jsonl \
      --raw runs/ov_distill_coco128/raw.jsonl \
      --labels datasets/coco128/labels/train2017
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ov_distill.clean import box_iou  # noqa: E402
from scripts.ov_distill.make_coco128_manifest import COCO_NAMES  # noqa: E402
from scripts.ov_distill.teachers import image_size  # noqa: E402


def load_gt(labels_dir: Path, image_path: Path) -> Dict[str, List[List[float]]]:
    """Load the GT of a single image, returning {class_name: [[x1,y1,x2,y2] absolute pixels]}."""
    lab = labels_dir / f"{image_path.stem}.txt"
    gt: Dict[str, List[List[float]]] = defaultdict(list)
    if not lab.exists():
        return gt
    w, h = image_size(image_path)
    for line in lab.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        xc, yc, bw, bh = (float(v) for v in parts[1:5])
        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        gt[COCO_NAMES[cid]].append([x1, y1, (xc + bw / 2) * w, (yc + bh / 2) * h])
    return gt


def greedy_match(preds: List[List[float]], gts: List[List[float]], iou_thr: float) -> Tuple[int, int, int]:
    """Greedy IoU matching; returns (TP, FP, FN)."""
    matched = set()
    tp = 0
    for p in preds:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j in matched:
                continue
            iou = box_iou(p, g)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            matched.add(best_j)
            tp += 1
    return tp, len(preds) - tp, len(gts) - tp


def main() -> None:
    """CLI entry point: prints overall and per-category metrics."""
    parser = argparse.ArgumentParser(description="coco128 teacher annotation quality evaluation")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--raw", required=True, help="raw annotate JSONL or cleaned fused JSONL (format auto-detected)")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--teacher", default="", help="with raw format, evaluate only the given teacher (e.g. rexomni-master)")
    args = parser.parse_args()

    manifest = [json.loads(line) for line in Path(args.manifest).open(encoding="utf-8") if line.strip()]
    raw = [json.loads(line) for line in Path(args.raw).open(encoding="utf-8") if line.strip()]
    pred_map: Dict[Tuple[str, str], List[List[float]]] = {}
    is_cleaned = any("box" in r for r in raw)
    for rec in raw:
        if not rec.get("ok", True):
            continue
        if args.teacher and rec.get("teacher", "") != args.teacher:
            continue
        key = (rec["image"], rec["query"])
        if is_cleaned:
            pred_map.setdefault(key, []).append(rec["box"])
        else:
            pred_map[key] = rec["boxes"]

    tp = fp = fn = 0
    neg_total = neg_correct = 0
    per_cat: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])
    missing_records = 0
    for item in manifest:
        image = item["image"]
        image_path = Path(image)
        gt = load_gt(Path(args.labels), image_path)
        for query in item["queries"]:
            boxes = pred_map.get((image, query))
            if boxes is None:
                # No record in the prediction source (teacher skipped the entry or the whole class
                # was filtered by voting): treat as empty prediction, counted into FN / rejection
                missing_records += 1
                boxes = []
            if query in gt:  # positive query
                t, f, n = greedy_match(boxes, gt[query], args.iou)
                tp += t
                fp += f
                fn += n
                per_cat[query][0] += t
                per_cat[query][1] += f
                per_cat[query][2] += n
            else:  # negative query: absent from GT, the teacher should abstain
                neg_total += 1
                if not boxes:
                    neg_correct += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    report = {
        "iou": args.iou,
        "positive": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "negative": {
            "total": neg_total,
            "correctly_rejected": neg_correct,
            "rejection_rate": round(neg_correct / neg_total, 4) if neg_total else None,
        },
        "missing_records": missing_records,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # Per-category F1 (only categories with >= 3 GT instances; sorted ascending to expose weaknesses)
    rows = []
    for cat, (t, f, n) in per_cat.items():
        p = t / (t + f) if t + f else 0.0
        r = t / (t + n) if t + n else 0.0
        ff = 2 * p * r / (p + r) if p + r else 0.0
        if t + n >= 3:
            rows.append((cat, t + n, round(p, 3), round(r, 3), round(ff, 3)))
    rows.sort(key=lambda x: x[4])
    print(f"\n{'category':<18}{'#gt':>5}{'P':>8}{'R':>8}{'F1':>8}")
    for cat, n, p, r, ff in rows:
        print(f"{cat:<18}{n:>5}{p:>8}{r:>8}{ff:>8}")


if __name__ == "__main__":
    main()
