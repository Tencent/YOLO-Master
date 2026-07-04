#!/usr/bin/env python3
"""Prepare scene folders for MoT routing diagnosis from a YOLO-format VisDrone dataset.

The routing diagnosis script expects scene-specific image folders, for example:

    datasets/VisDrone/routing_scenes/
      dense/
      sparse/
      small_objects/
      large_objects/
      dense_small/
      sparse_large/
      irregular_occluded/

This helper builds those folders from existing ``images`` and ``labels`` trees
using YOLO label statistics and quantile thresholds. The independent
``dense/sparse`` and ``small_objects/large_objects`` groups are better for
axis-wise comparisons; ``dense_small`` and ``sparse_large`` are corner-case
subsets. The ``irregular_occluded`` group is still a proxy when only YOLO labels
are available: it prioritizes dense images with high box scale/aspect-ratio
variation.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCENES = (
    "dense",
    "sparse",
    "small_objects",
    "large_objects",
    "dense_small",
    "sparse_large",
    "irregular_occluded",
)


@dataclass
class ImageStats:
    image: Path
    label: Path
    objects: int
    mean_area: float
    median_area: float
    area_cv: float
    aspect_cv: float

    @property
    def irregular_score(self) -> float:
        return self.objects * (self.area_cv + self.aspect_cv)


@dataclass(frozen=True)
class SceneThresholds:
    q_low: float
    q_high: float
    density_low: float
    density_high: float
    median_area_low: float
    median_area_high: float
    irregular_high: float


def parse_label(path: Path) -> list[tuple[float, float]]:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            continue
        if w > 0 and h > 0:
            boxes.append((w, h))
    return boxes


def coeff_var(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 0:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var) / mean


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * q
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def label_for_image(image: Path, dataset: Path) -> Path:
    images_root = dataset / "images"
    labels_root = dataset / "labels"
    rel = image.relative_to(images_root)
    return (labels_root / rel).with_suffix(".txt")


def collect_stats(dataset: Path, split: str) -> list[ImageStats]:
    images_root = dataset / "images"
    if not images_root.exists():
        raise FileNotFoundError(f"missing image root: {images_root}")

    split_root = images_root / split
    search_root = split_root if split_root.exists() else images_root
    stats = []
    for image in sorted(search_root.rglob("*")):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = label_for_image(image, dataset)
        boxes = parse_label(label)
        if not boxes:
            continue
        areas = [w * h for w, h in boxes]
        aspects = [w / max(h, 1e-9) for w, h in boxes]
        stats.append(
            ImageStats(
                image=image,
                label=label,
                objects=len(boxes),
                mean_area=sum(areas) / len(areas),
                median_area=median(areas),
                area_cv=coeff_var(areas),
                aspect_cv=coeff_var(aspects),
            )
        )
    if not stats:
        raise FileNotFoundError(f"no labeled images found under {search_root}")
    return stats


def compute_thresholds(stats: list[ImageStats], q_low: float, q_high: float) -> SceneThresholds:
    objects = [float(s.objects) for s in stats]
    median_areas = [s.median_area for s in stats]
    irregular_scores = [s.irregular_score for s in stats]
    return SceneThresholds(
        q_low=q_low,
        q_high=q_high,
        density_low=quantile(objects, q_low),
        density_high=quantile(objects, q_high),
        median_area_low=quantile(median_areas, q_low),
        median_area_high=quantile(median_areas, q_high),
        irregular_high=quantile(irregular_scores, q_high),
    )


def scene_flags(item: ImageStats, thresholds: SceneThresholds) -> dict[str, bool]:
    return {
        "dense": item.objects >= thresholds.density_high,
        "sparse": item.objects <= thresholds.density_low,
        "small": item.median_area <= thresholds.median_area_low,
        "large": item.median_area >= thresholds.median_area_high,
        "irregular_proxy": item.irregular_score >= thresholds.irregular_high,
    }


def select_scene(
    stats: list[ImageStats],
    scene: str,
    limit: int,
    thresholds: SceneThresholds,
) -> list[ImageStats]:
    def flags(item: ImageStats) -> dict[str, bool]:
        return scene_flags(item, thresholds)

    if scene == "dense_small":
        candidates = [s for s in stats if flags(s)["dense"] and flags(s)["small"]]
        ranked = sorted(candidates, key=lambda s: (s.objects, -s.median_area), reverse=True)
    elif scene == "sparse_large":
        candidates = [s for s in stats if flags(s)["sparse"] and flags(s)["large"]]
        ranked = sorted(candidates, key=lambda s: (s.median_area, -s.objects), reverse=True)
    elif scene == "dense":
        candidates = [s for s in stats if flags(s)["dense"]]
        ranked = sorted(candidates, key=lambda s: (s.objects, -s.median_area), reverse=True)
    elif scene == "sparse":
        candidates = [s for s in stats if flags(s)["sparse"]]
        ranked = sorted(candidates, key=lambda s: (s.objects, s.median_area))
    elif scene == "small_objects":
        candidates = [s for s in stats if flags(s)["small"]]
        ranked = sorted(candidates, key=lambda s: (s.median_area, -s.objects))
    elif scene == "large_objects":
        candidates = [s for s in stats if flags(s)["large"]]
        ranked = sorted(candidates, key=lambda s: (s.median_area, s.objects), reverse=True)
    elif scene == "irregular_occluded":
        candidates = [s for s in stats if flags(s)["irregular_proxy"]]
        ranked = sorted(candidates, key=lambda s: (s.irregular_score, s.objects), reverse=True)
    else:
        raise ValueError(f"unknown scene: {scene}")
    return ranked[:limit]


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if copy:
        shutil.copy2(src, dst)
        return
    rel_src = os.path.relpath(src, start=dst.parent)
    os.symlink(rel_src, dst)


def safe_image_name(path: Path) -> str:
    """Keep symlink names unique across split subfolders while preserving extension."""
    parts = path.parts[-3:] if len(path.parts) >= 3 else path.parts
    return "__".join(parts)


def write_scene(out_dir: Path, scene: str, selected: list[ImageStats], copy: bool, thresholds: SceneThresholds) -> None:
    scene_dir = out_dir / scene
    reset_dir(scene_dir)
    summary = scene_dir / "_selection_summary.csv"
    lines = [
        "image,label,objects,mean_area,median_area,area_cv,aspect_cv,irregular_score,"
        "is_dense,is_sparse,is_small,is_large,is_irregular_proxy"
    ]
    for item in selected:
        dst = scene_dir / safe_image_name(item.image)
        link_or_copy(item.image, dst, copy=copy)
        flags = scene_flags(item, thresholds)
        lines.append(
            f"{item.image},{item.label},{item.objects},{item.mean_area:.8f},"
            f"{item.median_area:.8f},{item.area_cv:.6f},{item.aspect_cv:.6f},"
            f"{item.irregular_score:.6f},{int(flags['dense'])},{int(flags['sparse'])},"
            f"{int(flags['small'])},{int(flags['large'])},{int(flags['irregular_proxy'])}"
        )
    summary.write_text("\n".join(lines) + "\n")


def write_thresholds(out_dir: Path, thresholds: SceneThresholds) -> None:
    lines = [
        "name,value",
        f"q_low,{thresholds.q_low}",
        f"q_high,{thresholds.q_high}",
        f"density_low,{thresholds.density_low}",
        f"density_high,{thresholds.density_high}",
        f"median_area_low,{thresholds.median_area_low:.8f}",
        f"median_area_high,{thresholds.median_area_high:.8f}",
        f"irregular_high,{thresholds.irregular_high:.6f}",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_selection_thresholds.csv").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/VisDrone"))
    parser.add_argument("--split", default="val", help="Prefer images/<split>; falls back to all images.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-images-per-scene", type=int, default=128)
    parser.add_argument("--low-quantile", type=float, default=0.30)
    parser.add_argument("--high-quantile", type=float, default=0.70)
    parser.add_argument("--copy", action="store_true", help="Copy images instead of creating symlinks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.low_quantile < args.high_quantile < 1.0:
        raise SystemExit("--low-quantile and --high-quantile must satisfy 0 < low < high < 1")
    dataset = args.dataset.resolve()
    out_dir = args.output.resolve() if args.output else dataset / "routing_scenes"
    stats = collect_stats(dataset, split=args.split)
    thresholds = compute_thresholds(stats, q_low=args.low_quantile, q_high=args.high_quantile)

    print(f"[scenes] found {len(stats)} labeled images")
    print(f"[scenes] writing {out_dir}")
    write_thresholds(out_dir, thresholds)
    for scene in SCENES:
        selected = select_scene(stats, scene=scene, limit=args.max_images_per_scene, thresholds=thresholds)
        write_scene(out_dir, scene=scene, selected=selected, copy=args.copy, thresholds=thresholds)
        print(f"[scenes] {scene}: {len(selected)} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
