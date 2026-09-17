#!/usr/bin/env python3
"""Diagnose native TAL and Area-Threshold STAL candidate coverage on a dataset split."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
AREA_BUCKETS = (
    "small",
    "medium",
    "large",
    "all",
)
MIN_CANDIDATES = 4


def build_parser() -> argparse.ArgumentParser:
    """Build the static assignment diagnostic CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Dataset YAML, for example dataset/VisDrone.local.yaml.")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--mode", choices=("pure-tal", "baseline", "area-stal"), default="area-stal")
    parser.add_argument("--area-threshold", type=float, default=64.0)
    parser.add_argument("--stride", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--max-images", type=int, default=0, help="Inspect at most N images (0 means all).")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def _image_files(split_path: str | list[str]) -> list[Path]:
    """Expand dataset split directories and image-list files."""
    paths = split_path if isinstance(split_path, list) else [split_path]
    extensions = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp"}
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file() and item.suffix[1:].lower() in extensions)
        elif path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                value = line.strip()
                if value:
                    item = Path(value)
                    files.append(item if item.is_absolute() else path.parent / item)
        else:
            raise FileNotFoundError(f"split path does not exist: {path}")
    return sorted(set(files))


def _letterbox_xyxy(xywh: np.ndarray, width: int, height: int, imgsz: int) -> np.ndarray:
    """Convert normalized YOLO boxes to square-letterboxed pixel coordinates."""
    x_center, y_center, box_width, box_height = xywh.T
    scale = min(imgsz / height, imgsz / width)
    resized_width, resized_height = round(width * scale), round(height * scale)
    pad_x, pad_y = (imgsz - resized_width) / 2.0, (imgsz - resized_height) / 2.0
    return np.stack(
        (
            (x_center - box_width / 2.0) * width * scale + pad_x,
            (y_center - box_height / 2.0) * height * scale + pad_y,
            (x_center + box_width / 2.0) * width * scale + pad_x,
            (y_center + box_height / 2.0) * height * scale + pad_y,
        ),
        axis=1,
    )


def _anchors(imgsz: int, strides: list[int]) -> np.ndarray:
    """Build the half-cell anchor centers used by the detector."""
    levels = []
    for stride in strides:
        height = imgsz // stride
        width = imgsz // stride
        y, x = np.meshgrid(
            np.arange(height, dtype=np.float32) + 0.5,
            np.arange(width, dtype=np.float32) + 0.5,
            indexing="ij",
        )
        levels.append(np.stack((x.ravel(), y.ravel()), axis=1) * stride)
    return np.concatenate(levels, axis=0)


def _read_boxes(image: Path, label: Path, imgsz: int) -> np.ndarray:
    """Load one YOLO label file and return valid letterboxed boxes."""
    from PIL import Image

    with Image.open(image) as handle:
        width, height = handle.size
    if not label.exists():
        return np.empty((0, 4), dtype=np.float32)
    rows = []
    for line in label.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            box = np.asarray([float(value) for value in fields[1:5]], dtype=np.float32)
        except ValueError:
            continue
        if np.isfinite(box).all() and (box[2:] > 0).all():
            rows.append(box)
    return _letterbox_xyxy(np.asarray(rows, dtype=np.float32).reshape(-1, 4), width, height, imgsz)


def _area_bucket(area: float) -> str:
    """Return the QA COCO-style scale interval for one ground truth."""
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def _native_candidate_mask(box: np.ndarray, anchors: np.ndarray, strides: list[int]) -> np.ndarray:
    """Reproduce the pinned TaskAlignedAssigner candidate geometry."""
    center = (box[:2] + box[2:]) / 2.0
    width_height = np.maximum(box[2:] - box[:2], 0.0)
    expanded_wh = np.where(width_height < strides[0], strides[1], width_height)
    expanded = np.concatenate((center - expanded_wh / 2.0, center + expanded_wh / 2.0))
    return ((anchors > expanded[:2]) & (anchors < expanded[2:])).all(axis=1)


def _pure_candidate_mask(box: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Reproduce pure TAL geometry without the fixed-stride candidate floor."""
    return ((anchors > box[:2]) & (anchors < box[2:])).all(axis=1)


def _area_floor(base: np.ndarray, box: np.ndarray, anchors: np.ndarray, threshold: float) -> np.ndarray:
    """Add the nearest missing anchors until an eligible target has four candidates."""
    width_height = np.maximum(box[2:] - box[:2], 0.0)
    if float(width_height.prod()) > threshold or int(base.sum()) >= MIN_CANDIDATES:
        return base
    needed = MIN_CANDIDATES - int(base.sum())
    center = (box[:2] + box[2:]) / 2.0
    distances = ((anchors - center) ** 2).sum(axis=1)
    distances = np.where(base, np.inf, distances)
    additions = np.argsort(distances)[:needed]
    result = base.copy()
    result[additions] = True
    return result


def diagnose(args: argparse.Namespace) -> dict:
    """Run the geometry-only diagnostic."""
    from ultralytics.data.utils import check_det_dataset, img2label_paths

    if args.area_threshold <= 0:
        raise ValueError("--area-threshold must be positive")
    data = check_det_dataset(args.data, autodownload=False, split=args.split)
    images = _image_files(data[args.split])
    if args.max_images > 0:
        images = images[: args.max_images]
    labels = [Path(path) for path in img2label_paths([str(image) for image in images])]
    anchors = _anchors(args.imgsz, args.stride)
    stats = {bucket: defaultdict(int) for bucket in AREA_BUCKETS}
    print(
        f"[A2 diagnose] mode={args.mode} area_threshold={args.area_threshold:g} "
        f"split={args.split} images={len(images)} anchors={len(anchors)}",
        flush=True,
    )

    total_gt = 0
    for index, (image, label) in enumerate(zip(images, labels), start=1):
        for box in _read_boxes(image, label, args.imgsz):
            total_gt += 1
            wh = np.maximum(box[2:] - box[:2], 0.0)
            area = float(wh.prod())
            base = (
                _pure_candidate_mask(box, anchors)
                if args.mode == "pure-tal"
                else _native_candidate_mask(box, anchors, args.stride)
            )
            candidates = _area_floor(base, box, anchors, args.area_threshold) if args.mode == "area-stal" else base
            values = {
                "gt": 1,
                "eligible_gt": int(args.mode == "area-stal" and area <= args.area_threshold),
                "base_candidates": int(base.sum()),
                "final_candidates": int(candidates.sum()),
                "floor_added": int((candidates & ~base).sum()),
                "zero_base": int(not base.any()),
                "zero_final": int(not candidates.any()),
            }
            for bucket in (_area_bucket(area), "all"):
                for key, value in values.items():
                    stats[bucket][key] += value
        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(images)):
            print(f"[A2 diagnose] {index}/{len(images)} images, {total_gt} GT", flush=True)

    return {
        "data": str(args.data),
        "split": args.split,
        "mode": args.mode,
        "images": len(images),
        "imgsz": args.imgsz,
        "strides": args.stride,
        "area_threshold": args.area_threshold,
        "min_candidates": MIN_CANDIDATES,
        "buckets": {bucket: dict(values) for bucket, values in stats.items()},
        "note": (
            "Scale buckets use transformed training bbox area: small < 32^2, "
            "medium 32^2 <= area < 96^2, large >= 96^2. Conflict, post-assignment, "
            "and zero-weight statistics are recorded during training."
        ),
    }


def main() -> int:
    """Run the diagnostic and optionally persist its JSON output."""
    args = build_parser().parse_args()
    result = diagnose(args)
    output = json.dumps(result, indent=2, ensure_ascii=False)
    print(output)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output + "\n", encoding="utf-8")
        print(f"[A2 diagnose] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
