#!/usr/bin/env python3
"""OV-Distill: build a verification manifest (with negative queries) from coco128 GT labels.

Each image's queries = the classes actually present in the image + 2 classes
sampled from the COCO vocabulary that are absent (negatives, used to verify
teacher abstention and the pipeline's negative-extraction chain).

Usage:
  python scripts/ov_distill/make_coco128_manifest.py \
      --labels datasets/coco128/labels/train2017 \
      --images datasets/coco128/images/train2017 \
      --out runs/ov_distill_coco128/manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

COCO_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def build_manifest(labels: Path, images: Path, neg_per_image: int = 2, seed: int = 0) -> List[dict]:
    """Build the list of manifest entries from GT labels."""
    rng = random.Random(seed)
    items = []
    for lab_file in sorted(labels.glob("*.txt")):
        img = images / f"{lab_file.stem}.jpg"
        if not img.exists():
            continue
        class_ids: Dict[int, None] = {}
        for line in lab_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                class_ids[int(parts[0])] = None
        positives = [COCO_NAMES[cid] for cid in sorted(class_ids)]
        pool = [n for n in COCO_NAMES if n not in set(positives)]
        negatives = rng.sample(pool, min(neg_per_image, len(pool)))
        items.append({"image": str(img.resolve()), "queries": positives + negatives})
    return items


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="coco128 GT -> verification manifest")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--neg-per-image", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    items = build_manifest(Path(args.labels), Path(args.images), args.neg_per_image, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    n_pos = sum(len([q for q in it["queries"]]) - args.neg_per_image for it in items)
    print(
        json.dumps(
            {"images": len(items), "positive_queries": n_pos, "negative_queries": len(items) * args.neg_per_image},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
