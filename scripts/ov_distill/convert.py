#!/usr/bin/env python3
"""OV-Distill: convert cleaned results into YOLO grounding training formats.

Output layout (``--format yolo``, default):

    <out>/images/<name>            # copied when --copy-images is set
    <out>/labels/<name>.txt        # one line per box: class_id x_center y_center w h (normalized)
    <out>/categories.json          # [{"id": 0, "name": "person"}, ...] dynamic vocabulary
    <out>/grounding.jsonl          # one line per image: {image, phrases: [{text, box_xywh_norm}]}

``--format grounding`` produces a single-file JSON in the GoldG style that
the ultralytics ``GroundingDataset`` loads directly:

    <out>/grounding_train.json     # {"images": [...], "annotations": [...]}
        images:      {id, file_name, height, width, caption, negative_captions}
        annotations: {image_id, bbox(xywh absolute pixels), iscrowd, tokens_positive, weight, vote metadata}

The caption joins the image's distinct category names with ". ", and
tokens_positive holds the character span of each category name inside the
caption (Python slice semantics, end exclusive), matching how
``ultralytics.data.dataset.GroundingDataset.cache_labels`` parses them.
Confirmed negatives are stored as per-image candidate captions without boxes. The training-side ``RandomLoadText``
prioritizes these explicit negatives within the prompt budget before adding its usual random negatives.
Negative-only images remain in the dataset and provide image-level absence supervision.

class_id is mapped dynamically from the query text into categories.json, so
records with the same semantic category naturally share an id.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from scripts.ov_distill.class_split import load_class_split
from scripts.ov_distill.reliability import (
    image_key,
    load_calibration_artifact,
    load_image_manifest,
    predict_isotonic,
    record_score,
)
from scripts.ov_distill.teachers import image_size


def _iter_records(
    cleaned: Path, buckets: Optional[Iterable[str]], include_images: Optional[set] = None
) -> Tuple[List[Tuple[Path, dict, int, int]], int]:
    """Read cleaned JSONL, filtering by bucket/image and skipping missing images."""
    allow = set(buckets) if buckets else None
    rows: List[Tuple[Path, dict, int, int]] = []
    skipped = 0
    with cleaned.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if allow is not None and rec.get("bucket", "medium") not in allow:
                continue
            image_value = str(rec.get("image_path") or rec["image"])
            if include_images is not None and image_key(image_value, cleaned.parent) not in include_images:
                continue
            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = (cleaned.parent / image_path).resolve()
            if not image_path.exists():
                skipped += 1
                continue
            w, h = image_size(image_path)
            rows.append((image_path, rec, w, h))
    return rows, skipped


def _convert_yolo(rows: List[Tuple[Path, dict, int, int]], out: Path, copy_images: bool) -> Dict[str, int]:
    """Classic YOLO txt + categories.json + grounding.jsonl output (original behavior)."""
    labels_dir = out / "labels"
    images_dir = out / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    if copy_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    cat_ids: Dict[str, int] = {}
    per_image: Dict[str, List[dict]] = defaultdict(list)
    for image_path, rec, w, h in rows:
        x1, y1, x2, y2 = rec["box"]
        xc = max(0.0, min(1.0, (x1 + x2) / 2 / w))
        yc = max(0.0, min(1.0, (y1 + y2) / 2 / h))
        bw = max(0.0, min(1.0, (x2 - x1) / w))
        bh = max(0.0, min(1.0, (y2 - y1) / h))
        query = str(rec["query"])
        if query not in cat_ids:
            cat_ids[query] = len(cat_ids)
        per_image[str(image_path)].append(
            {
                "class_id": cat_ids[query],
                "text": query,
                "xywh": [round(xc, 6), round(yc, 6), round(bw, 6), round(bh, 6)],
                "bucket": rec.get("bucket", "medium"),
            }
        )
        if copy_images:
            dst = images_dir / image_path.name
            if not dst.exists():
                shutil.copy2(image_path, dst)

    n_boxes = 0
    with (out / "grounding.jsonl").open("w", encoding="utf-8") as fg:
        for image, items in sorted(per_image.items()):
            stem = Path(image).stem
            with (labels_dir / f"{stem}.txt").open("w", encoding="utf-8") as fl:
                for it in items:
                    x, y, bw, bh = it["xywh"]
                    fl.write(f"{it['class_id']} {x} {y} {bw} {bh}\n")
                    n_boxes += 1
            fg.write(
                json.dumps(
                    {
                        "image": image,
                        "phrases": [
                            {"text": it["text"], "box_xywh_norm": it["xywh"], "bucket": it["bucket"]} for it in items
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    categories = [{"id": cid, "name": name} for name, cid in sorted(cat_ids.items(), key=lambda kv: kv[1])]
    with (out / "categories.json").open("w", encoding="utf-8") as fc:
        json.dump(categories, fc, ensure_ascii=False, indent=2)
    return {"images": len(per_image), "boxes": n_boxes, "categories": len(cat_ids)}


def _build_caption(categories: List[str]) -> Tuple[str, Dict[str, List[int]]]:
    """Build the image-level caption and return each category's tokens_positive span (end exclusive)."""
    caption = ". ".join(categories) + ("." if categories else "")
    spans: Dict[str, List[int]] = {}
    cursor = 0
    for name in categories:
        spans[name] = [cursor, cursor + len(name)]
        cursor += len(name) + 2  # skip the ". " separator
    return caption, spans


def _convert_grounding(
    rows: List[Tuple[Path, dict, int, int]],
    out: Path,
    image_root: Optional[Path],
    negative_rows: Optional[List[Tuple[Path, dict, int, int]]] = None,
    weight_mode: str = "agreement",
    calibrator: Optional[dict] = None,
) -> Dict[str, int]:
    """GoldG-style single-JSON output, directly trainable with the ultralytics GroundingDataset."""
    per_image: Dict[str, dict] = {}
    for image_path, rec, w, h in rows:
        key = str(image_path)
        entry = per_image.setdefault(key, {"path": image_path, "w": w, "h": h, "boxes": [], "negatives": set()})
        entry["boxes"].append((str(rec["query"]), rec["box"], rec))
    for image_path, rec, w, h in negative_rows or []:
        key = str(image_path)
        entry = per_image.setdefault(key, {"path": image_path, "w": w, "h": h, "boxes": [], "negatives": set()})
        entry["negatives"].add(str(rec["query"]).strip().lower())

    images: List[dict] = []
    annotations: List[dict] = []
    categories: set = set()
    num_negatives = 0
    for img_id, key in enumerate(sorted(per_image)):
        entry = per_image[key]
        image_path: Path = entry["path"]
        positive_names = {q.strip().lower() for q, _, _ in entry["boxes"]}
        negative_names = entry["negatives"] - positive_names
        cat_names = sorted(positive_names | negative_names)
        num_negatives += len(negative_names)
        caption, spans = _build_caption(cat_names)
        categories.update(cat_names)
        if image_root is not None:
            try:
                file_name = str(image_path.relative_to(image_root))
            except ValueError:
                file_name = image_path.name
        else:
            file_name = image_path.name
        images.append(
            {
                "id": img_id,
                "file_name": file_name,
                "height": entry["h"],
                "width": entry["w"],
                "caption": caption,
                "negative_captions": sorted(negative_names),
            }
        )
        for query, (x1, y1, x2, y2), rec in entry["boxes"]:
            votes, teachers = int(rec.get("votes", 0)), int(rec.get("teachers", 0))
            fallback_weight = (votes / teachers) * float(rec.get("cluster_iou", 1.0)) if votes and teachers else 1.0
            if weight_mode == "uniform":
                weight = 1.0
            elif weight_mode == "calibrated":
                if calibrator is None:
                    raise ValueError("weight_mode='calibrated' requires a reliability calibrator")
                weight = float(predict_isotonic([record_score(rec)], calibrator)[0])
            else:
                weight = float(rec.get("supervision_weight", max(0.25, min(1.0, fallback_weight))))
            annotations.append(
                {
                    "image_id": img_id,
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    "iscrowd": 0,
                    "tokens_positive": [spans[query.strip().lower()]],
                    "weight": max(0.0, min(1.0, weight)),
                    "bucket": rec.get("bucket", "medium"),
                    "votes": votes or None,
                    "teachers": teachers or None,
                    "reliability_score": rec.get("reliability_score"),
                    "clip_score": rec.get("clip_score"),
                }
            )

    out.mkdir(parents=True, exist_ok=True)
    payload = {"images": images, "annotations": annotations}
    with (out / "grounding_train.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    annotated_images = {ann["image_id"] for ann in annotations}
    return {
        "images": len(images),
        "boxes": len(annotations),
        "categories": len(categories),
        "confirmed_negatives": num_negatives,
        "negative_only_images": sum(im["id"] not in annotated_images for im in images),
        "weight_mode": weight_mode,
    }


def _iter_negative_records(
    negatives: Path, include_images: Optional[set] = None
) -> Tuple[List[Tuple[Path, dict, int, int]], int]:
    """Read confirmed-absence records and retain those whose source images are available."""
    rows = []
    skipped = 0
    if not negatives.exists():
        return rows, skipped
    with negatives.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            image_value = str(rec.get("image_path") or rec["image"])
            if include_images is not None and image_key(image_value, negatives.parent) not in include_images:
                continue
            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = (negatives.parent / image_path).resolve()
            if not image_path.exists():
                skipped += 1
                continue
            w, h = image_size(image_path)
            rows.append((image_path, rec, w, h))
    return rows, skipped


def _filter_class_rows(
    rows: List[Tuple[Path, dict, int, int]], allowed_classes: set, kind: str, known_classes: set
) -> tuple[list, int]:
    """Keep rows in the supplied class vocabulary and reject non-canonical labels."""
    kept = []
    removed = 0
    unknown = set()
    for row in rows:
        name = " ".join(str(row[1].get("query", "")).strip().lower().split())
        if name not in known_classes:
            unknown.add(name or "<empty>")
        elif name not in allowed_classes:
            removed += 1
        else:
            kept.append(row)
    if unknown:
        raise ValueError(
            f"{kind} contain labels outside the COCO-80 class split: {sorted(unknown)[:10]}; "
            "canonicalize queries before building a class-disjoint training set"
        )
    return kept, removed


def run_convert(
    cleaned: Path,
    out: Path,
    copy_images: bool = False,
    fmt: str = "yolo",
    buckets: Optional[Iterable[str]] = None,
    image_root: Optional[Path] = None,
    negatives: Optional[Path] = None,
    weight_mode: str = "agreement",
    calibration_map: Optional[Path] = None,
    include_manifest: Optional[Path] = None,
    class_split: Optional[Path] = None,
) -> Dict[str, int]:
    """Convert the absolute-pixel box JSONL produced by ``clean`` into a training dataset; returns statistics.

    Args:
        cleaned: JSONL produced by the ``clean`` subcommand.
        out: output dataset directory.
        copy_images: only effective with ``fmt="yolo"``; copies images into the output directory.
        fmt: ``"yolo"`` (txt + categories.json) or ``"grounding"`` (GoldG single JSON).
        buckets: keep only the given buckets (e.g. ``{"high"}``); None keeps everything.
        image_root: only effective with ``fmt="grounding"``; file_name is written
            relative to this root and should match the ``img_path`` used at training time.
        negatives: confirmed-absence JSONL produced by ``clean``; defaults to ``negatives.jsonl`` beside ``cleaned``.
        weight_mode: ``agreement`` keeps the current heuristic, ``calibrated`` maps it to GT precision, and ``uniform``
            disables weighting.
        calibration_map: isotonic calibration JSON; required for ``weight_mode="calibrated"``.
        include_manifest: optional JSONL/path manifest restricting training data to listed images.
        class_split: optional COCO-80 split JSON; when set, only seen-class positives and negatives are retained.
    """
    out.mkdir(parents=True, exist_ok=True)
    if weight_mode not in {"agreement", "calibrated", "uniform"}:
        raise ValueError(f"unknown weight mode: {weight_mode}")
    if fmt != "grounding" and weight_mode != "agreement":
        raise ValueError("reliability weight modes are supported only with --format grounding")
    if class_split is not None and fmt != "grounding":
        raise ValueError("--class-split is supported only with --format grounding")
    if weight_mode == "calibrated" and calibration_map is None:
        raise ValueError("--calibration-map is required with --weight-mode calibrated")
    include_images = load_image_manifest(include_manifest) if include_manifest is not None else None
    split = load_class_split(class_split) if class_split is not None else None
    calibrator = None
    calibration_artifact = None
    calibration_manifest_path = None
    if weight_mode == "calibrated":
        if include_images is None:
            raise ValueError(
                "--include-manifest is required with --weight-mode calibrated to exclude GT calibration images"
            )
        calibration_artifact = load_calibration_artifact(calibration_map)
        if split is not None:
            if set(calibration_artifact.get("seen_classes", [])) != set(split["seen"]):
                raise ValueError(
                    "calibration artifact seen_classes do not match --class-split; "
                    "recalibrate with the same split before converting"
                )
            if set(calibration_artifact.get("unseen_classes", [])) != set(split["unseen"]):
                raise ValueError(
                    "calibration artifact unseen_classes do not match --class-split; "
                    "recalibrate with the same split before converting"
                )
        calibration_manifest = calibration_artifact.get("calibration_manifest")
        if not calibration_manifest:
            raise ValueError(
                "calibration artifact is missing its calibration_manifest; regenerate it with calibrate_reliability.py"
            )
        calibration_manifest_path = Path(calibration_manifest)
        if not calibration_manifest_path.is_absolute():
            calibration_manifest_path = Path(calibration_map).parent / calibration_manifest_path
        calibration_images = load_image_manifest(calibration_manifest_path)
        calibration_manifest_path = calibration_manifest_path.resolve()
        overlap = include_images & calibration_images
        if overlap:
            raise ValueError(
                f"training include-manifest overlaps the GT calibration split on {len(overlap)} images; "
                "exclude calibration images before exporting training data"
            )
        calibrator = calibration_artifact["calibration"]
    rows, skipped = _iter_records(cleaned, buckets, include_images=include_images)
    negative_path = negatives or cleaned.parent / "negatives.jsonl"
    negative_rows, skipped_negatives = (
        _iter_negative_records(negative_path, include_images=include_images) if fmt == "grounding" else ([], 0)
    )
    class_stats = {}
    if class_split is not None:
        seen = {" ".join(name.lower().split()) for name in split["seen"]}
        known = {" ".join(name.lower().split()) for name in split["class_names"]}
        rows, removed_boxes = _filter_class_rows(rows, seen, "positive annotations", known)
        negative_rows, removed_negatives = _filter_class_rows(negative_rows, seen, "confirmed negatives", known)
        class_stats = {
            "class_split": str(Path(class_split).resolve()),
            "training_seen_classes": len(seen),
            "filtered_unseen_boxes": removed_boxes,
            "filtered_unseen_negatives": removed_negatives,
        }
    if fmt == "yolo":
        stats = _convert_yolo(rows, out, copy_images)
    elif fmt == "grounding":
        stats = _convert_grounding(rows, out, image_root, negative_rows, weight_mode, calibrator)
        metadata = {}
        metadata["weight_mode"] = weight_mode
        if include_manifest is not None:
            metadata["training_manifest"] = str(Path(include_manifest).resolve())
        if split is not None:
            metadata.update(
                {
                    "provenance_version": 1,
                    "class_split": str(Path(class_split).resolve()),
                    "seen_classes": sorted(split["seen"]),
                    "unseen_classes": sorted(split["unseen"]),
                }
            )
        if calibration_manifest_path is not None:
            metadata["calibration_manifest"] = str(calibration_manifest_path)
        if metadata:
            grounding_path = out / "grounding_train.json"
            payload = json.loads(grounding_path.read_text(encoding="utf-8"))
            payload["metadata"] = metadata
            grounding_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    else:
        raise ValueError(f"unknown output format: {fmt} (expected yolo / grounding)")
    stats["skipped_missing_image"] = skipped
    stats["skipped_missing_negative_image"] = skipped_negatives
    stats.update(class_stats)
    if include_images is not None:
        stats["manifest_images"] = len(include_images)
    return stats
