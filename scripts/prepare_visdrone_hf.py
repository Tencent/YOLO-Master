#!/usr/bin/env python3
"""Prepare a Hugging Face VisDrone snapshot for Ultralytics training."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import yaml


VISDRONE_NAMES = {
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
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("/jpfs/huangyidan3/datasets/VisDrone"))
    parser.add_argument("--yaml-out", type=Path, default=Path("/jpfs/huangyidan3/datasets/VisDrone-hf.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def extract_archives(snapshot_dir: Path, out_dir: Path, overwrite: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    archives = sorted(snapshot_dir.rglob("*.zip"))
    if not archives:
        return
    marker = out_dir / ".extract_complete"
    if marker.exists() and not overwrite:
        return
    if overwrite and out_dir.exists():
        for child in out_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    for archive in archives:
        print(f"Extracting {archive} -> {out_dir}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
    marker.write_text("ok\n", encoding="utf-8")


def find_dataset_root(out_dir: Path) -> tuple[Path, str]:
    split_train = out_dir / "VisDrone2019-DET-train"
    split_val = out_dir / "VisDrone2019-DET-val"
    if (split_train / "images").exists() and (split_train / "labels").exists() and (split_val / "images").exists():
        return out_dir, "split_dirs"

    candidates = []
    for image_dir in out_dir.rglob("images/train"):
        root = image_dir.parents[1]
        if (root / "labels/train").exists() and (root / "images/val").exists() and (root / "labels/val").exists():
            candidates.append(root)
    if (out_dir / "images/train").exists() and (out_dir / "labels/train").exists():
        return out_dir, "images_labels"
    if candidates:
        return sorted(candidates, key=lambda p: len(p.parts))[0], "images_labels"
    raise SystemExit(f"could not find YOLO VisDrone layout under {out_dir}")


def write_yaml(dataset_root: Path, layout: str, yaml_out: Path) -> None:
    if layout == "split_dirs":
        test = "VisDrone2019-DET-test-dev/images" if (dataset_root / "VisDrone2019-DET-test-dev/images").exists() else ""
        data = {
            "path": str(dataset_root),
            "train": "VisDrone2019-DET-train/images",
            "val": "VisDrone2019-DET-val/images",
            "test": test,
            "names": VISDRONE_NAMES,
        }
    else:
        data = {
            "path": str(dataset_root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test" if (dataset_root / "images/test").exists() else "",
            "names": VISDRONE_NAMES,
        }
    yaml_out.parent.mkdir(parents=True, exist_ok=True)
    yaml_out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Wrote {yaml_out}")


def main() -> int:
    args = parse_args()
    extract_archives(args.snapshot_dir, args.out_dir, args.overwrite)
    dataset_root, layout = find_dataset_root(args.out_dir)
    print(f"VisDrone dataset root: {dataset_root} ({layout})")
    write_yaml(dataset_root, layout, args.yaml_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
