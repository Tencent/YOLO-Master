#!/usr/bin/env python3
"""OV-Distill: build a scaled verification/distillation manifest (with negative queries) from COCO val2017 GT.

Same structure as make_coco128_manifest.py, but reads instances_val2017.json
directly and supports subsampling via --max-images (get 1000 images working
first, then scale up to 5000).

Usage:
  python scripts/ov_distill/make_val2017_manifest.py \
      --instances datasets/coco/annotations/instances_val2017.json \
      --images datasets/coco/val2017 \
      --max-images 1000 --seed 0 \
      --out runs/ov_s2_val1k/manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

# Kept consistent with make_coco128_manifest.COCO_NAMES (the order is the name table after
# sorting by COCO category_id), but the id->name mapping here is built from the categories
# inside the instances json to avoid hard-coded drift.
from make_coco128_manifest import COCO_NAMES


def build_manifest(
    instances: Path,
    images: Path,
    max_images: int = 0,
    neg_per_image: int = 2,
    seed: int = 0,
) -> List[dict]:
    """Build the list of manifest entries from val2017 GT."""
    rng = random.Random(seed)
    data = json.loads(instances.read_text(encoding="utf-8"))
    id2name = {c["id"]: c["name"] for c in data["categories"]}

    # image_id -> set of positive class names
    pos_by_image: Dict[int, Dict[str, None]] = defaultdict(dict)
    for ann in data["annotations"]:
        if ann.get("iscrowd"):
            continue
        pos_by_image[ann["image_id"]][id2name[ann["category_id"]]] = None

    entries = []
    for img in data["images"]:
        img_path = images / img["file_name"]
        if not img_path.exists():
            continue
        positives = sorted(pos_by_image.get(img["id"], {}).keys())
        if not positives:  # skip box-less images (essentially none in val2017)
            continue
        pool = [n for n in COCO_NAMES if n not in set(positives)]
        negatives = rng.sample(pool, min(neg_per_image, len(pool)))
        entries.append({"image": str(img_path.resolve()), "queries": positives + negatives})

    rng.shuffle(entries)
    if max_images > 0:
        entries = entries[:max_images]
    entries.sort(key=lambda e: e["image"])
    return entries


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="COCO val2017 GT -> scaled manifest")
    parser.add_argument("--instances", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-images", type=int, default=0, help="0 means the full set")
    parser.add_argument("--neg-per-image", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    items = build_manifest(
        Path(args.instances), Path(args.images), args.max_images, args.neg_per_image, args.seed
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    n_pos = sum(len(it["queries"]) - args.neg_per_image for it in items)
    print(
        json.dumps(
            {"images": len(items), "positive_queries": n_pos, "negative_queries": len(items) * args.neg_per_image},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
