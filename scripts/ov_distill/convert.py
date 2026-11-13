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
        images:      {id, file_name, height, width, caption}
        annotations: {image_id, bbox(xywh absolute pixels), iscrowd, tokens_positive}

The caption joins the image's distinct category names with ". ", and
tokens_positive holds the character span of each category name inside the
caption (Python slice semantics, end exclusive), matching how
``ultralytics.data.dataset.GroundingDataset.cache_labels`` parses them.
Negatives need no annotation entries: the training-side ``RandomLoadText``
augments negatives automatically from the dataset vocabulary.

class_id is mapped dynamically from the query text into categories.json, so
records with the same semantic category naturally share an id.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from scripts.ov_distill.teachers import image_size


def _iter_records(cleaned: Path, buckets: Optional[Iterable[str]]) -> Tuple[List[Tuple[Path, dict, int, int]], int]:
    """Read a cleaned JSONL, returning (rows, skipped_missing_image); filters by bucket and skips missing images."""
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
            image_path = Path(rec.get("image_path") or rec["image"])
            if not image_path.is_absolute():
                image_path = (cleaned.parent / rec["image"]).resolve()
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
    rows: List[Tuple[Path, dict, int, int]], out: Path, image_root: Optional[Path]
) -> Dict[str, int]:
    """GoldG-style single-JSON output, directly trainable with the ultralytics GroundingDataset."""
    per_image: Dict[str, dict] = {}
    for image_path, rec, w, h in rows:
        key = str(image_path)
        entry = per_image.setdefault(key, {"path": image_path, "w": w, "h": h, "boxes": []})
        entry["boxes"].append((str(rec["query"]), rec["box"]))

    images: List[dict] = []
    annotations: List[dict] = []
    categories: set = set()
    for img_id, key in enumerate(sorted(per_image)):
        entry = per_image[key]
        image_path: Path = entry["path"]
        cat_names = sorted({q for q, _ in entry["boxes"]})
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
            }
        )
        for query, (x1, y1, x2, y2) in entry["boxes"]:
            annotations.append(
                {
                    "image_id": img_id,
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    "iscrowd": 0,
                    "tokens_positive": [spans[query]],
                }
            )

    out.mkdir(parents=True, exist_ok=True)
    payload = {"images": images, "annotations": annotations}
    with (out / "grounding_train.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return {"images": len(images), "boxes": len(annotations), "categories": len(categories)}


def run_convert(
    cleaned: Path,
    out: Path,
    copy_images: bool = False,
    fmt: str = "yolo",
    buckets: Optional[Iterable[str]] = None,
    image_root: Optional[Path] = None,
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
    """
    out.mkdir(parents=True, exist_ok=True)
    rows, skipped = _iter_records(cleaned, buckets)
    if fmt == "yolo":
        stats = _convert_yolo(rows, out, copy_images)
    elif fmt == "grounding":
        stats = _convert_grounding(rows, out, image_root)
    else:
        raise ValueError(f"unknown output format: {fmt} (expected yolo / grounding)")
    stats["skipped_missing_image"] = skipped
    return stats
