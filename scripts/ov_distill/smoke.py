#!/usr/bin/env python3
"""OV-Distill · End-to-end smoke test: synthetic images + two MockTeachers through annotate → clean → convert.

Validates the pipeline skeleton in CI and on GPU-less machines. MockTeacher
generates deterministic pseudo-boxes from hashes with a slight systematic
jitter between teachers of different seeds (IoU > 0.8), so voting succeeds;
queries with the ``zz_absent`` prefix simulate teacher abstention, exercising
the negative-sample extraction path.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict

from PIL import Image, ImageDraw

from scripts.ov_distill.annotate import run_annotate
from scripts.ov_distill.clean import run_clean
from scripts.ov_distill.convert import run_convert

SMOKE_QUERIES = ["person", "car", "dog", "zz_absent_ufo"]


def _make_images(img_dir: Path, num_images: int) -> None:
    """Generate synthetic images with random rectangles."""
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_images):
        im = Image.new("RGB", (320, 240), color=(30 + i * 10 % 200, 60, 90))
        draw = ImageDraw.Draw(im)
        draw.rectangle([40 + i * 5, 40, 120 + i * 5, 140], fill=(200, 80, 80))
        draw.rectangle([180, 60 + i * 3, 280, 160 + i * 3], fill=(80, 200, 120))
        im.save(img_dir / f"smoke_{i:03d}.jpg", quality=90)


def run_smoke(out: Path, num_images: int = 4) -> Dict[str, int]:
    """Run the end-to-end smoke test and return per-stage statistics."""
    out = out.resolve()
    if out.exists():
        shutil.rmtree(out)
    img_dir = out / "images_src"
    _make_images(img_dir, num_images)

    manifest = out / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for img in sorted(img_dir.glob("*.jpg")):
            f.write(json.dumps({"image": str(img), "queries": SMOKE_QUERIES}, ensure_ascii=False) + "\n")

    raw = out / "raw.jsonl"
    stats_ann = run_annotate(manifest=manifest, teacher_specs=["mock:0", "mock:1"], out=raw)

    cleaned = out / "cleaned.jsonl"
    stats_clean = run_clean(raw=raw, out=cleaned, iou_thr=0.5, min_votes=2)

    dataset = out / "dataset"
    stats_conv = run_convert(cleaned=cleaned, out=dataset)

    neg_path = out / "negatives.jsonl"
    n_neg = sum(1 for line in neg_path.open("r", encoding="utf-8") if line.strip()) if neg_path.exists() else 0

    # Smoke assertions: negatives == images (zz_absent_ufo is rejected by all teachers on every image)
    assert n_neg == num_images, f"negative count {n_neg} != image count {num_images}"
    assert stats_clean["kept_boxes"] > 0, "no boxes survived cleaning; voting logic is broken"
    assert stats_conv["boxes"] == stats_clean["kept_boxes"], "converted box count differs from cleaned box count"

    return {"annotate": stats_ann, "clean": stats_clean, "convert": stats_conv, "negatives": n_neg}
