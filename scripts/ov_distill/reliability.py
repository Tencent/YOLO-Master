#!/usr/bin/env python3
"""Calibrate teacher-agreement scores against independently labeled images."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from scripts.ov_distill.class_split import load_class_split
from scripts.ov_distill.clean import box_iou
from scripts.ov_distill.make_coco128_manifest import COCO_NAMES
from scripts.ov_distill.teachers import image_size


def image_key(value: str, base: Path | None = None) -> str:
    """Normalize an image path for split membership checks."""
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return str(path.resolve())


def load_image_manifest(path: Path) -> set:
    """Read image paths from a JSONL manifest (or a plain one-path-per-line file)."""
    images = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = line
            if isinstance(row, dict):
                image = row.get("image") or row.get("image_path") or row.get("file_name")
            elif isinstance(row, str):
                image = row
            else:
                image = None
            if not image:
                raise ValueError(f"manifest {path}:{line_no} has no image path")
            images.add(image_key(str(image), Path(path).parent))
    if not images:
        raise ValueError(f"manifest contains no images: {path}")
    return images


def assert_disjoint_manifests(
    first_manifest: Path,
    second_manifest: Path,
    first_name: str = "first",
    second_name: str = "second",
) -> tuple[set, set]:
    """Load two image manifests and reject canonical-path overlap."""
    first_images = load_image_manifest(first_manifest)
    second_images = load_image_manifest(second_manifest)
    overlap = first_images & second_images
    if overlap:
        examples = sorted(overlap)[:5]
        raise ValueError(
            f"{first_name} and {second_name} manifests overlap on {len(overlap)} images; "
            f"the splits must be image-disjoint (examples: {examples})"
        )
    return first_images, second_images


def record_score(record: dict) -> float:
    """Get the unfloored agreement score from a cleaned pseudo-label record."""
    if record.get("reliability_score") is not None:
        value = float(record["reliability_score"])
    elif record.get("vote_ratio") is not None:
        value = float(record["vote_ratio"]) * float(record.get("cluster_iou", 1.0))
    elif record.get("votes") and record.get("teachers"):
        value = float(record["votes"]) / float(record["teachers"]) * float(record.get("cluster_iou", 1.0))
    elif record.get("supervision_weight") is not None:
        value = float(record["supervision_weight"])
    else:
        raise ValueError("cleaned record is missing reliability_score/vote_ratio/votes metadata")
    if not np.isfinite(value):
        raise ValueError(f"non-finite reliability score: {value}")
    return min(1.0, max(0.0, value))


def _load_gt(image_path: Path, gt_labels: Path) -> dict[str, list[list[float]]]:
    """Read a YOLO-format COCO label file and convert boxes to absolute pixels."""
    width, height = image_size(image_path)
    result: dict[str, list[list[float]]] = defaultdict(list)
    label_path = gt_labels / f"{image_path.stem}.txt"
    if not label_path.exists():
        return result
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(parts[0])
        if not 0 <= class_id < len(COCO_NAMES):
            raise ValueError(f"invalid COCO class id {class_id} in {label_path}:{line_no}")
        xc, yc, box_w, box_h = (float(v) for v in parts[1:5])
        result[COCO_NAMES[class_id].lower()].append(
            [
                (xc - box_w / 2) * width,
                (yc - box_h / 2) * height,
                (xc + box_w / 2) * width,
                (yc + box_h / 2) * height,
            ]
        )
    return result


def _match_predictions(records: Sequence[dict], gt_labels: Path, iou_thr: float) -> tuple[list[float], list[int]]:
    """Greedily match score-ranked pseudo boxes to GT, with each GT used at most once."""
    groups: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    gt_cache: dict[str, dict[str, list[list[float]]]] = {}
    score_by_index: dict[int, float] = {}
    for index, record in enumerate(records):
        if len(record.get("box", [])) < 4:
            raise ValueError(f"cleaned record for {record.get('image')} is missing a four-coordinate box")
        image_value = str(record.get("image_path") or record["image"])
        image_path = Path(image_value).expanduser()
        if not image_path.is_absolute():
            image_path = (Path(record.get("_image_base", ".")) / image_path).resolve()
        image_id = str(image_path.resolve())
        query = str(record["query"]).strip().lower()
        groups[(image_id, query)].append((index, record))
        score_by_index[index] = record_score(record)
        if image_id not in gt_cache:
            gt_cache[image_id] = _load_gt(image_path, gt_labels)

    labels_by_index: dict[int, int] = {}
    for (image_id, query), candidates in groups.items():
        unmatched = list(gt_cache[image_id].get(query, []))
        candidates.sort(key=lambda item: (-score_by_index[item[0]], item[0]))
        for index, record in candidates:
            box = [float(v) for v in record["box"][:4]]
            if not unmatched:
                labels_by_index[index] = 0
                continue
            overlaps = [box_iou(box, gt) for gt in unmatched]
            best = int(np.argmax(overlaps))
            if overlaps[best] >= iou_thr:
                labels_by_index[index] = 1
                unmatched.pop(best)
            else:
                labels_by_index[index] = 0
    ordered_indices = sorted(labels_by_index)
    return [score_by_index[i] for i in ordered_indices], [labels_by_index[i] for i in ordered_indices]


def fit_isotonic(scores: Sequence[float], labels: Sequence[int]) -> dict:
    """Fit a dependency-free pool-adjacent-violators isotonic calibrator."""
    if len(scores) != len(labels) or not scores:
        raise ValueError("isotonic calibration needs equally sized, non-empty score and label arrays")
    grouped: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    for score, label in zip(scores, labels):
        score = min(1.0, max(0.0, float(score)))
        label = int(label)
        if label not in (0, 1):
            raise ValueError(f"calibration label must be binary, got {label}")
        grouped[score][0] += 1
        grouped[score][1] += label

    blocks: list[dict] = []
    for score in sorted(grouped):
        count, positives = grouped[score]
        blocks.append({"upper": score, "count": count, "positives": positives})
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left["positives"] / left["count"] <= right["positives"] / right["count"]:
                break
            blocks[-2:] = [
                {
                    "upper": right["upper"],
                    "count": left["count"] + right["count"],
                    "positives": left["positives"] + right["positives"],
                }
            ]
    return {
        "method": "isotonic_pav",
        "upper_bounds": [round(block["upper"], 8) for block in blocks],
        "probabilities": [round(block["positives"] / block["count"], 8) for block in blocks],
        "block_counts": [block["count"] for block in blocks],
    }


def predict_isotonic(scores: Iterable[float], calibrator: dict) -> np.ndarray:
    """Apply the stepwise isotonic mapping to raw agreement scores."""
    upper = np.asarray(calibrator["upper_bounds"], dtype=np.float64)
    probabilities = np.asarray(calibrator["probabilities"], dtype=np.float64)
    if upper.size == 0 or upper.size != probabilities.size:
        raise ValueError("invalid isotonic calibrator: upper_bounds and probabilities must be non-empty and aligned")
    values = np.clip(np.asarray(list(scores), dtype=np.float64), 0.0, 1.0)
    indices = np.searchsorted(upper, values, side="left").clip(0, len(upper) - 1)
    return probabilities[indices]


def calibration_metrics(scores: Sequence[float], labels: Sequence[int], bins: int = 10) -> dict:
    """Compute Brier score and equal-width expected calibration error."""
    if bins < 1:
        raise ValueError("calibration metric bins must be at least one")
    values = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    targets = np.asarray(labels, dtype=np.float64)
    if values.size == 0 or values.size != targets.size:
        raise ValueError("metrics need equally sized, non-empty scores and labels")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    details = []
    for i in range(bins):
        mask = (values >= edges[i]) & (values <= edges[i + 1] if i == bins - 1 else values < edges[i + 1])
        if not np.any(mask):
            continue
        confidence = float(values[mask].mean())
        accuracy = float(targets[mask].mean())
        count = int(mask.sum())
        ece += count / len(values) * abs(confidence - accuracy)
        details.append(
            {
                "lower": round(float(edges[i]), 3),
                "upper": round(float(edges[i + 1]), 3),
                "count": count,
                "mean_score": round(confidence, 6),
                "empirical_precision": round(accuracy, 6),
            }
        )
    return {
        "brier": round(float(np.mean((values - targets) ** 2)), 8),
        "ece": round(float(ece), 8),
        "bins": details,
    }


def calibrate_reliability(
    cleaned: Path,
    gt_labels: Path,
    calibrate_manifest: Path,
    train_manifest: Path,
    out: Path,
    iou_thr: float = 0.5,
    bins: int = 10,
    min_samples: int = 50,
    min_positives: int = 5,
    min_negatives: int = 5,
    class_split: Path | None = None,
) -> dict:
    """Fit and save a GT-based reliability mapping, enforcing image-disjoint splits."""
    if not 0.0 <= iou_thr <= 1.0:
        raise ValueError(f"IoU threshold must be in [0, 1], got {iou_thr}")
    if min_samples < 1 or min_positives < 1 or min_negatives < 1:
        raise ValueError("minimum calibration sample counts must all be positive")
    _training_images, calibration_images = assert_disjoint_manifests(
        train_manifest, calibrate_manifest, first_name="training", second_name="calibration"
    )
    class_split_payload = load_class_split(class_split) if class_split is not None else None
    seen_classes = (
        {" ".join(name.lower().split()) for name in class_split_payload["seen"]}
        if class_split_payload is not None
        else None
    )

    rows = []
    with Path(cleaned).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            image_value = str(record.get("image_path") or record["image"])
            key = image_key(image_value, Path(cleaned).parent)
            if key not in calibration_images:
                continue
            if seen_classes is not None:
                query = " ".join(str(record.get("query", "")).strip().lower().split())
                if query not in set(COCO_NAMES):
                    raise ValueError(
                        f"calibration query is outside COCO-80 class split: {query!r}; "
                        "canonicalize queries before class-disjoint calibration"
                    )
                if query not in seen_classes:
                    continue
            record["_image_base"] = str(Path(cleaned).parent)
            rows.append(record)
    if not rows:
        raise ValueError("no cleaned pseudo boxes matched --calibrate-manifest")

    scores, labels = _match_predictions(rows, Path(gt_labels), iou_thr)
    positives = int(sum(labels))
    negatives = len(labels) - positives
    if len(scores) < min_samples:
        raise ValueError(f"calibration has {len(scores)} matched boxes; need at least {min_samples}")
    if positives < min_positives or negatives < min_negatives:
        raise ValueError(
            f"calibration needs at least {min_positives} GT matches and {min_negatives} false boxes; "
            f"found {positives} and {negatives}"
        )

    calibrator = fit_isotonic(scores, labels)
    calibrated_scores = predict_isotonic(scores, calibrator)
    stats = {
        "images": len(calibration_images),
        "images_with_pseudo_boxes": len(
            {image_key(str(r.get("image_path") or r["image"]), Path(r["_image_base"])) for r in rows}
        ),
        "boxes": len(scores),
        "true_positives": positives,
        "false_positives": negatives,
        "iou_threshold": iou_thr,
        "raw": calibration_metrics(scores, labels, bins=bins),
        "calibrated": calibration_metrics(calibrated_scores.tolist(), labels, bins=bins),
    }
    payload = {
        "schema_version": 1,
        "target": "pseudo_box_precision_at_iou",
        "class_vocabulary": "COCO-80 names from YOLO class ids",
        "calibration": calibrator,
        "statistics": stats,
        "calibration_manifest": str(Path(calibrate_manifest).resolve()),
        "training_manifest": str(Path(train_manifest).resolve()),
    }
    if class_split_payload is not None:
        payload["class_split"] = str(Path(class_split).resolve())
        payload["seen_classes"] = sorted(seen_classes)
        payload["unseen_classes"] = sorted(class_split_payload["unseen"])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def load_calibration_artifact(path: Path) -> dict:
    """Load and validate an exported calibration artifact."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    calibrator = payload.get("calibration", payload)
    if calibrator.get("method") != "isotonic_pav":
        raise ValueError(f"unsupported reliability calibrator in {path}")
    # Validate array alignment before converter applies it to potentially large datasets.
    predict_isotonic([0.0], calibrator)
    payload["calibration"] = calibrator
    return payload


def load_calibrator(path: Path) -> dict:
    """Load only the isotonic mapping from an exported calibration artifact."""
    return load_calibration_artifact(path)["calibration"]
