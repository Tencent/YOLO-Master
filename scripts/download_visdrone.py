#!/usr/bin/env python3
"""Download and prepare VisDrone2019-DET train/val in YOLO format."""

from __future__ import annotations

import argparse
import shutil
import time
import zipfile
from pathlib import Path

import requests
import yaml
from PIL import Image
from tqdm import tqdm


URLS = {
    "train": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "val": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
}

NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/jpfs/huangyidan3/datasets/VisDrone"))
    parser.add_argument("--download-dir", type=Path, default=Path("/jpfs/huangyidan3/datasets/downloads/VisDrone"))
    parser.add_argument("--yaml-out", type=Path, default=Path("/jpfs/huangyidan3/datasets/VisDrone.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--skip-download", action="store_true", help="Use existing split directories or zip files only.")
    return parser.parse_args()


def valid_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False


def download(url: str, dst: Path, retries: int) -> None:
    if valid_zip(dst):
        print(f"Using existing valid zip {dst}")
        return
    if dst.exists():
        print(f"Removing incomplete or corrupt zip {dst}")
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            if part.exists():
                part.unlink()
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                with part.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dst.name) as bar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            part.replace(dst)
            if valid_zip(dst):
                return
            raise zipfile.BadZipFile(f"downloaded file failed zip validation: {dst}")
        except Exception as exc:
            last_error = exc
            print(f"Download failed attempt={attempt}/{retries}: {exc}")
            if attempt < retries:
                time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"failed to download {url}") from last_error


def extract(zip_path: Path, root: Path, split: str) -> Path:
    source = root / f"VisDrone2019-DET-{split}"
    if source.exists():
        return source
    print(f"Extracting {zip_path} -> {root}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    if not source.exists():
        raise SystemExit(f"missing extracted directory: {source}")
    return source


def convert_split(root: Path, split: str) -> None:
    source = root / f"VisDrone2019-DET-{split}"
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    source_images = source / "images"
    if source_images.exists():
        for image in tqdm(list(source_images.glob("*.jpg")), desc=f"Linking {split} images"):
            target = images_dir / image.name
            if not target.exists():
                try:
                    target.symlink_to(image)
                except OSError:
                    shutil.copy2(image, target)

    source_labels = source / "labels"
    if source_labels.exists():
        for label in tqdm(list(source_labels.glob("*.txt")), desc=f"Linking {split} labels"):
            target = labels_dir / label.name
            if not target.exists():
                try:
                    target.symlink_to(label)
                except OSError:
                    shutil.copy2(label, target)
        return

    for ann in tqdm(list((source / "annotations").glob("*.txt")), desc=f"Converting {split} labels"):
        image_path = images_dir / ann.with_suffix(".jpg").name
        if not image_path.exists():
            continue
        width, height = Image.open(image_path).size
        dw, dh = 1.0 / width, 1.0 / height
        lines = []
        for raw in ann.read_text(encoding="utf-8").strip().splitlines():
            row = raw.split(",")
            if len(row) < 6 or row[4] == "0":
                continue
            x, y, w, h = map(int, row[:4])
            cls = int(row[5]) - 1
            if cls < 0 or cls > 9:
                continue
            xc = (x + w / 2) * dw
            yc = (y + h / 2) * dh
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {w * dw:.6f} {h * dh:.6f}\n")
        (labels_dir / ann.name).write_text("".join(lines), encoding="utf-8")


def write_yaml(root: Path, yaml_out: Path) -> None:
    yaml_out.parent.mkdir(parents=True, exist_ok=True)
    yaml_out.write_text(
        yaml.safe_dump(
            {"path": str(root), "train": "images/train", "val": "images/val", "names": NAMES},
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {yaml_out}")


def main() -> int:
    args = parse_args()
    if args.overwrite and args.root.exists():
        shutil.rmtree(args.root)
    args.root.mkdir(parents=True, exist_ok=True)

    for split, url in URLS.items():
        zip_path = args.download_dir / Path(url).name
        if not args.skip_download:
            download(url, zip_path, args.retries)
        if zip_path.exists():
            extract(zip_path, args.root, split)
        elif not (args.root / f"VisDrone2019-DET-{split}").exists():
            raise SystemExit(f"missing {zip_path} or extracted split directory for {split}")
        convert_split(args.root, split)
    write_yaml(args.root, args.yaml_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
