#!/usr/bin/env python3
"""Download COCO 2017 and prepare 8k-subset for multi-task training.

Downloads:
  - train2017.zip (19G, 118k images)
  - val2017.zip (1G, 5k images)  
  - coco2017labels-segments.zip (~250M, detection + segmentation labels)

Creates:
  - datasets/coco/train8k.txt  (first 8000 train2017 images)
  - datasets/coco/val2017.txt  (all 5000 val2017 images)
"""

import os, sys, zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"

# ── add project root so we can import ultralytics ──────────────
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import ASSETS_URL
from ultralytics.utils.downloads import download, safe_download


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    coco_root = DATASETS_DIR / "coco"
    images_dir = coco_root / "images"
    labels_dir = coco_root / "labels"

    # ── Step 1: download COCO 2017 labels (with segments) ──────
    print("[1/3] Downloading COCO labels (with segmentation)...")
    if not (coco_root / "annotations").exists():
        url = ASSETS_URL + "/coco2017labels-segments.zip"
        safe_download(url, dir=coco_root.parent, unzip=True, delete=True)
        print("  Labels downloaded and extracted.")
    else:
        print("  Labels already present, skipping.")

    # ── Step 2: download train2017 images ───────────────────────
    train_dir = images_dir / "train2017"
    if not train_dir.exists() or len(list(train_dir.glob("*.jpg"))) < 10000:
        print("[2/3] Downloading train2017 images (19G, 118k images)...")
        zip_path = coco_root / "train2017.zip"
        url = "http://images.cocodataset.org/zips/train2017.zip"
        safe_download(url, file=zip_path)
        print("  Extracting train2017.zip...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(images_dir)
        zip_path.unlink()
        print(f"  Extracted {len(list(train_dir.glob('*.jpg')))} images")
    else:
        n = len(list(train_dir.glob("*.jpg")))
        print(f"  train2017 already present ({n} images), skipping.")

    # ── Step 3: download val2017 images ─────────────────────────
    val_dir = images_dir / "val2017"
    if not val_dir.exists() or len(list(val_dir.glob("*.jpg"))) < 1000:
        print("[3/3] Downloading val2017 images (1G, 5k images)...")
        zip_path = coco_root / "val2017.zip"
        url = "http://images.cocodataset.org/zips/val2017.zip"
        safe_download(url, file=zip_path)
        print("  Extracting val2017.zip...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(images_dir)
        zip_path.unlink()
        print(f"  Extracted {len(list(val_dir.glob('*.jpg')))} images")
    else:
        n = len(list(val_dir.glob("*.jpg")))
        print(f"  val2017 already present ({n} images), skipping.")

    # ── Step 4: create image-list txt files ─────────────────────
    # train8k.txt: first 8000 images from train2017
    train_images = sorted(train_dir.glob("*.jpg"))
    train8k = train_images[:8000]
    train8k_path = coco_root / "train8k.txt"
    with open(train8k_path, "w") as f:
        for img in train8k:
            f.write(f"./images/train2017/{img.name}\n")
    print(f"\nCreated {train8k_path} ({len(train8k)} images)")

    # val2017.txt: all val2017 images
    val_images = sorted(val_dir.glob("*.jpg"))
    val_path = coco_root / "val2017.txt"
    with open(val_path, "w") as f:
        for img in val_images:
            f.write(f"./images/val2017/{img.name}\n")
    print(f"Created {val_path} ({len(val_images)} images)")

    # ── Step 5: verify labels exist ─────────────────────────────
    label_dir = labels_dir / "train2017"
    if label_dir.exists():
        n_labels = len(list(label_dir.glob("*.txt")))
        print(f"Labels: {n_labels} .txt files in {label_dir}")
    else:
        print("WARNING: labels/train2017/ not found! Check download.")

    print("\nDone! You can now train with: data=coco8k-multitask.yaml")


if __name__ == "__main__":
    main()
