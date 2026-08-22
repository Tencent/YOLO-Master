#!/usr/bin/env python3
"""Download VisDrone2019-DET to local path and convert to YOLO format."""

from __future__ import annotations

import shutil
import zipfile
import sys
from pathlib import Path

# Add YOLO-Master to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics.utils.downloads import download
from ultralytics.utils import TQDM

URLS = [
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
]

# Target: F:\研究\VisDrone
ROOT = Path(r"F:\研究\VisDrone")
ROOT.mkdir(parents=True, exist_ok=True)

print(f"Downloading VisDrone to {ROOT}...")
download(URLS, dir=ROOT, threads=4)

# Convert to YOLO format
from PIL import Image

for split, folder in [("train", "VisDrone2019-DET-train"), ("val", "VisDrone2019-DET-val")]:
    source_dir = ROOT / folder
    images_dir = ROOT / "images" / split
    labels_dir = ROOT / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Move images
    if (source_dir / "images").exists():
        for img in (source_dir / "images").glob("*.jpg"):
            target = images_dir / img.name
            if not target.exists():
                shutil.copy2(img, target)
    elif (source_dir / "images").exists():
        for img in (source_dir / "images").glob("*.jpg"):
            target = images_dir / img.name
            if not target.exists():
                shutil.copy2(img, target)

    # Convert annotations
    if (source_dir / "annotations").exists():
        for ann in TQDM(list((source_dir / "annotations").glob("*.txt")),
                        desc=f"Converting {split}"):
            img_path = images_dir / ann.with_suffix(".jpg").name
            if not img_path.exists():
                continue
            img_size = Image.open(img_path).size
            dw, dh = 1.0 / img_size[0], 1.0 / img_size[1]
            lines = []
            for row in [x.split(",") for x in ann.read_text(encoding="utf-8").strip().splitlines()]:
                if row[4] != "0":  # Skip ignored regions
                    x, y, w, h = map(int, row[:4])
                    cls = int(row[5]) - 1
                    x_center, y_center = (x + w / 2) * dw, (y + h / 2) * dh
                    w_norm, h_norm = w * dw, h * dh
                    lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            (labels_dir / ann.name).write_text("".join(lines), encoding="utf-8")

# Clean up original directories
for folder in ["VisDrone2019-DET-train", "VisDrone2019-DET-val"]:
    if (p := ROOT / folder).exists():
        shutil.rmtree(p)

print(f"Done! VisDrone prepared at {ROOT}")
print(f"  Train images: {len(list((ROOT / 'images/train').glob('*.jpg')))}")
print(f"  Val images: {len(list((ROOT / 'images/val').glob('*.jpg')))}")

# Write dataset YAML
import yaml
NAMES = {
    0: "pedestrian", 1: "people", 2: "bicycle", 3: "car",
    4: "van", 5: "truck", 6: "tricycle", 7: "awning-tricycle",
    8: "bus", 9: "motor",
}
yaml_path = ROOT / "VisDrone.yaml"
yaml_path.write_text(
    yaml.safe_dump({
        "path": str(ROOT),
        "train": "images/train",
        "val": "images/val",
        "names": NAMES,
    }, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
print(f"Wrote {yaml_path}")
