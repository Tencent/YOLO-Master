#!/usr/bin/env python3
"""OV-Distill: multi-teacher cross-validation cleaning.

Rules (ADR-0007, data pipeline Stage 3):
  1. Box-level voting: for each (image, query), boxes from all teachers are
     greedily clustered at IoU >= iou_thr; clusters with at least min_votes
     member teachers are kept, and boxes inside a cluster are mean-fused;
  2. Abstention negatives: an (image, query) with no surviving box and at
     least neg_min_absent abstaining teachers (default n_teachers - 1, at
     least 2) is recorded as a negative -- this tolerates one teacher that
     never abstains (e.g. GroundingPI hallucinated boxes are voted out);
  3. Confidence bucketing: full-vote clusters with mean intra-cluster
     IoU >= 0.7 are tagged ``high``, all other kept boxes ``medium``; the
     bucket is written to the record so training can treat hard vs. soft
     labels differently.

Input: raw prediction JSONL produced by ``annotate`` (only ok=true records).
Output: cleaned JSONL + negatives JSONL.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

Box = List[float]  # [x1, y1, x2, y2] absolute pixels


def box_iou(a: Box, b: Box) -> float:
    """Compute the IoU of two absolute-pixel boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def cluster_boxes(members: List[Tuple[str, Box]], iou_thr: float) -> List[List[Tuple[str, Box]]]:
    """Greedy IoU clustering: every box in a cluster has IoU >= iou_thr with the seed box, and each teacher contributes at most one box per cluster."""
    clusters: List[List[Tuple[str, Box]]] = []
    for teacher, box in members:
        placed = False
        for cluster in clusters:
            seed_teacher, seed_box = cluster[0]
            if teacher == seed_teacher:
                continue  # a teacher never joins the same cluster twice
            if box_iou(seed_box, box) >= iou_thr and all(t != teacher for t, _ in cluster):
                cluster.append((teacher, box))
                placed = True
                break
        if not placed:
            clusters.append([(teacher, box)])
    return clusters


def fuse_cluster(cluster: List[Tuple[str, Box]]) -> Tuple[Box, float]:
    """Fuse the boxes of a cluster into their mean box; returns (box, mean intra-cluster IoU)."""
    n = len(cluster)
    fused = [sum(b[i] for _, b in cluster) / n for i in range(4)]
    if n == 1:
        return fused, 1.0
    ious = [box_iou(cluster[i][1], cluster[j][1]) for i in range(n) for j in range(i + 1, n)]
    return fused, sum(ious) / len(ious)


def run_clean(
    raw: Path,
    out: Path,
    neg_out: Optional[Path] = None,
    iou_thr: float = 0.5,
    min_votes: int = 2,
    neg_min_absent: Optional[int] = None,
) -> Dict[str, int]:
    """Run cross-validation cleaning and return summary statistics.

    Args:
        neg_min_absent: minimum number of abstaining teachers required to
            record a negative; None means n_teachers - 1 (floor of 2),
            tolerating one teacher that never abstains.
    """
    groups: Dict[Tuple[str, str], dict] = defaultdict(
        lambda: {"members": [], "teachers": set(), "absent": set(), "image_path": ""}
    )
    with raw.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("ok"):
                continue
            key = (rec["image"], rec["query"])
            group = groups[key]
            group["teachers"].add(rec["teacher"])
            if not group["image_path"]:
                group["image_path"] = rec.get("image_path", rec["image"])
            if rec.get("absent") or not rec.get("boxes"):
                group["absent"].add(rec["teacher"])
            for box in rec.get("boxes", []):
                group["members"].append((rec["teacher"], [float(v) for v in box[:4]]))

    if neg_out is None:
        neg_out = out.parent / "negatives.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    stats = {"pairs": 0, "kept_boxes": 0, "high": 0, "medium": 0, "negatives": 0, "dropped": 0}
    with out.open("w", encoding="utf-8") as fo, neg_out.open("w", encoding="utf-8") as fn:
        for (image, query), group in sorted(groups.items()):
            stats["pairs"] += 1
            image_path = group["image_path"] or image
            teachers = group["teachers"]
            n_absent_needed = neg_min_absent if neg_min_absent is not None else max(2, len(teachers) - 1)
            clusters = cluster_boxes(group["members"], iou_thr)
            kept = False
            for cluster in clusters:
                votes = len({t for t, _ in cluster})
                if votes < min_votes:
                    stats["dropped"] += 1
                    continue
                box, mean_iou = fuse_cluster(cluster)
                bucket = "high" if votes == len(teachers) and mean_iou >= 0.7 else "medium"
                fo.write(
                    json.dumps(
                        {
                            "image": image,
                            "image_path": image_path,
                            "query": query,
                            "box": box,
                            "votes": votes,
                            "teachers": len(teachers),
                            "cluster_iou": round(mean_iou, 4),
                            "bucket": bucket,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stats["kept_boxes"] += 1
                stats[bucket] += 1
                kept = True
            if not kept and len(group["absent"]) >= n_absent_needed:
                fn.write(
                    json.dumps(
                        {
                            "image": image,
                            "image_path": image_path,
                            "query": query,
                            "reason": "teachers_absent",
                            "absent": len(group["absent"]),
                            "teachers": len(teachers),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stats["negatives"] += 1
    return stats
