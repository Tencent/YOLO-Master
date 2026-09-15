#!/usr/bin/env python3
"""Prepare immutable VisDrone labels and provenance without changing originals."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from scripts.d1.artifacts import digest, encoded, file_sha, immutable

COUNTS = {"train": 6471, "val": 548, "test-dev": 1610}
NAMES = ("pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor")
SCHEMA = "d1-visdrone-data-v1"


def convert_annotation(text, width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions")
    labels, sidecar = [], []
    stats = Counter()
    for number, row in enumerate(csv.reader(io.StringIO(text)), 1):
        if not row:
            continue
        if row[-1:] == [""]:
            row = row[:-1]
        if len(row) != 8:
            raise ValueError(f"Annotation row {number} must have eight fields")
        values = [float(v) for v in row]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"Nonfinite annotation at row {number}")
        x, y, w, h, score, category, _truncation, _occlusion = values
        if w < 0 or h < 0 or any(v != int(v) for v in values[4:]):
            raise ValueError(f"Invalid box or flags at row {number}")
        if score not in (0, 1) or not 0 <= category <= 11:
            raise ValueError(f"Unknown score/category at row {number}")
        stats["original_boxes"] += 1
        valid = score == 1 and 1 <= category <= 10
        x1, y1 = min(width, max(0, x)), min(height, max(0, y))
        x2, y2 = min(width, max(0, x + w)), min(height, max(0, y + h))
        clipped = [x1, y1, x2 - x1, y2 - y1]
        changed = clipped != values[:4]
        if changed:
            stats["clipped_boxes"] += 1
        reason = "train_target" if valid else "ignored_score_or_category"
        if valid and (x2 <= x1 or y2 <= y1):
            reason = "degenerate_after_clip"
            stats["dropped_degenerate"] += 1
        elif valid:
            label = [
                int(category) - 1,
                (x1 + x2) / (2 * width),
                (y1 + y2) / (2 * height),
                (x2 - x1) / width,
                (y2 - y1) / height,
            ]
            labels.append(str(label[0]) + " " + " ".join(f"{v:.10f}" for v in label[1:]))
            stats["training_boxes"] += 1
            stats[f"class_{int(category) - 1}"] += 1
        else:
            stats["ignored_boxes"] += 1
        # Preserve every original row, including flags, without rewriting official GT.
        sidecar.append(
            {"line": number, "original": values, "clipped_xywh": clipped, "clipped": changed, "disposition": reason}
        )
    return ("\n".join(labels) + ("\n" if labels else "")).encode(), sidecar, dict(stats)


def prepare(source, output, *, verify=False, counts=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    counts = {key: COUNTS[key] for key in ("train", "val")} if counts is None else counts
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Prepared data must be separate from original data")
    if source.stat().st_dev != output.parent.stat().st_dev:
        raise ValueError("Original and prepared images must be on the same local filesystem")
    previous_ids = set()
    previous_hashes = set()
    inventory, reports = [], {}
    for split, count in counts.items():
        raw = source / f"VisDrone2019-DET-{split}"
        images = sorted((raw / "images").glob("*.jpg"))
        if len(images) != count or len(list((raw / "annotations").glob("*.txt"))) != count:
            raise ValueError(f"Wrong official split size: {split}")
        ids = {p.stem for p in images}
        if ids & previous_ids or len(ids) != count:
            raise ValueError("Split image IDs overlap")
        previous_ids.update(ids)
        totals, paths = Counter(), []
        split_hashes = set()
        hash_members = {}
        for image in images:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", image.stem):
                raise ValueError("Unsafe image ID")
            ann = raw / "annotations" / f"{image.stem}.txt"
            annotation = ann.read_bytes()
            with Image.open(image) as obj:
                width, height = obj.size
                obj.verify()
            label, sidecar, stats = convert_annotation(annotation.decode(), width, height)
            totals.update(stats)
            totals["empty_training_images"] += int(not label)
            sid = f"visdrone-{split}/{image.stem}"
            rel = f"images/{sid}.jpg"
            image_sha = file_sha(image)
            if image_sha in previous_hashes:
                raise ValueError("Duplicate image bytes across official splits")
            split_hashes.add(image_sha)
            hash_members.setdefault(image_sha, []).append(image.stem)
            target = output / rel
            if target.is_symlink():
                raise ValueError("Image target may not be a symlink")
            if target.exists():
                if file_sha(target) != image_sha:
                    raise ValueError("Prepared image differs from original")
            elif verify:
                raise FileNotFoundError(target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(image, target)
            label_rel, side_rel = f"labels/{sid}.txt", f"sidecars/{sid}.json"
            side_bytes = encoded({"source_annotation_sha256": digest(annotation), "rows": sidecar})
            immutable(output / label_rel, label, verify)
            immutable(output / side_rel, side_bytes, verify)
            inventory.append(
                {
                    "sample_id": sid,
                    "split": f"visdrone-{split}",
                    "image_path": rel,
                    "original_image": image.relative_to(source).as_posix(),
                    "original_annotation": ann.relative_to(source).as_posix(),
                    "image_sha256": image_sha,
                    "annotation_sha256": digest(annotation),
                    "label_path": label_rel,
                    "label_sha256": digest(label),
                    "sidecar_path": side_rel,
                    "sidecar_sha256": digest(side_bytes),
                    "width": width,
                    "height": height,
                }
            )
            paths.append(rel)
        previous_hashes.update(split_hashes)
        listing = ("\n".join(paths) + "\n").encode()
        immutable(output / f"{split}.txt", listing, verify)
        reports[split] = {
            "images": count,
            "paths_sha256": digest(listing),
            "counts": dict(totals),
            "same_split_duplicate_image_bytes": [v for v in hash_members.values() if len(v) > 1],
        }
    inventory.sort(key=lambda x: x["sample_id"])
    inventory_bytes = b"".join(encoded(row).replace(b"\n", b" ").rstrip() + b"\n" for row in inventory)
    immutable(output / "samples.jsonl", inventory_bytes, verify)
    # Relative paths are made absolute only in this external runtime YAML.
    import yaml

    cfg = {
        "path": str(output),
        "train": "images/visdrone-train",
        "val": "images/visdrone-val",
        "nc": 10,
        "names": list(NAMES),
    }
    if "test-dev" in counts:
        cfg["test"] = "images/visdrone-test-dev"
    immutable(output / "dataset.yaml", yaml.safe_dump(cfg, sort_keys=False).encode(), verify)
    original_manifest = source / "manifest.json"
    source_receipt = json.loads(original_manifest.read_text()) if original_manifest.exists() else None
    result = {
        "schema_version": SCHEMA,
        "dataset": "VisDrone2019-DET",
        "splits": reports,
        "samples_sha256": digest(inventory_bytes),
        "sample_count": len(inventory),
        "class_names": list(NAMES),
        "originals_preserved": True,
        "split_ids_disjoint": True,
        "split_image_hashes_disjoint": True,
        "source_download_receipt": source_receipt,
        "source_download_receipt_sha256": file_sha(original_manifest) if source_receipt else None,
        "training_ignore_background_mask": False,
        "test_dev_use": "final evaluation only",
    }
    immutable(output / "manifest.json", encoded(result), verify)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--include-test-dev", action="store_true", help="Also prepare the optional held-out split")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.source, args.output, verify=args.verify_only, counts=COUNTS if args.include_test_dev else None
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
