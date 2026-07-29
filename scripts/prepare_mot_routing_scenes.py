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
import json
import math
import os
import shutil
import zipfile
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


@dataclass(frozen=True)
class OcclusionStats:
    """Image-level occlusion metadata from the original VisDrone annotations."""

    valid_objects: int
    occluded_fraction: float
    heavy_occluded_fraction: float
    mean_occlusion_level: float


@dataclass(frozen=True)
class OcclusionPair:
    """One lower/higher-occlusion image pair from the same video sequence."""

    sequence_id: str
    lower: ImageStats
    higher: ImageStats
    lower_occlusion: OcclusionStats
    higher_occlusion: OcclusionStats
    match_distance: float


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


def parse_visdrone_occlusion(text: str) -> OcclusionStats | None:
    """Parse VisDrone's score/category/truncation/occlusion annotation columns."""
    levels = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 8:
            continue
        try:
            score = int(parts[4])
            category = int(parts[5])
            occlusion = int(parts[7])
        except ValueError:
            continue
        if score > 0 and 1 <= category <= 10 and occlusion >= 0:
            levels.append(occlusion)
    if not levels:
        return None
    return OcclusionStats(
        valid_objects=len(levels),
        occluded_fraction=sum(level > 0 for level in levels) / len(levels),
        heavy_occluded_fraction=sum(level >= 2 for level in levels) / len(levels),
        mean_occlusion_level=sum(levels) / len(levels),
    )


def load_visdrone_occlusion(source: Path) -> dict[str, OcclusionStats]:
    """Load original VisDrone annotations from a directory or official ZIP."""
    output: dict[str, OcclusionStats] = {}

    def add(name: str, text: str) -> None:
        parsed = parse_visdrone_occlusion(text)
        if parsed is not None:
            output[Path(name).stem] = parsed

    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            for name in sorted(archive.namelist()):
                if "/annotations/" in name and name.lower().endswith(".txt"):
                    add(name, archive.read(name).decode("utf-8"))
    elif source.is_dir():
        for path in sorted(source.rglob("*.txt")):
            add(path.name, path.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError(f"VisDrone annotation source must be a directory or ZIP: {source}")
    if not output:
        raise ValueError(f"no original eight-column VisDrone annotations found in {source}")
    return output


def resolve_visdrone_annotations(dataset: Path, split: str, explicit: Path | None) -> Path:
    """Resolve an annotation source while supporting Ultralytics' downloaded ZIP layout."""
    if explicit is not None:
        source = explicit.expanduser()
        if not source.is_absolute():
            source = dataset / source
        if source.exists():
            return source.resolve()
        raise FileNotFoundError(f"VisDrone annotation source not found: {source}")

    aliases = {"train": "train", "val": "val", "test": "test-dev"}
    archive_split = aliases.get(split, split)
    candidates = (
        dataset / f"VisDrone2019-DET-{archive_split}.zip",
        dataset / f"VisDrone2019-DET-{archive_split}" / "annotations",
        dataset / "annotations" / split,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "original VisDrone annotations were not found; pass --visdrone-annotations with the official ZIP or directory"
    )


def image_sequence_id(item: ImageStats) -> str:
    """Return VisDrone's video sequence prefix from an image filename."""
    return item.image.stem.split("_", 1)[0]


def standardized_mean_difference(values_a: list[float], values_b: list[float]) -> float:
    """Return a pooled-standard-deviation balance diagnostic."""
    if len(values_a) < 2 or len(values_b) < 2:
        return float("nan")
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    variance_a = sum((value - mean_a) ** 2 for value in values_a) / (len(values_a) - 1)
    variance_b = sum((value - mean_b) ** 2 for value in values_b) / (len(values_b) - 1)
    pooled = math.sqrt((variance_a + variance_b) / 2.0)
    if pooled <= 0:
        return 0.0 if math.isclose(mean_a, mean_b) else float("inf")
    return (mean_b - mean_a) / pooled


def match_paired_occlusion_scenes(
    stats: list[ImageStats],
    occlusion: dict[str, OcclusionStats],
    limit: int,
    q_low: float,
    q_high: float,
) -> tuple[list[OcclusionPair], dict[str, float | int]]:
    """Match one lower/higher-occlusion image pair within each video sequence."""
    aligned = [(item, occlusion[item.image.stem]) for item in stats if item.image.stem in occlusion]
    if not aligned:
        raise ValueError("no dataset images match the supplied VisDrone occlusion annotations")
    fractions = [item_occlusion.occluded_fraction for _, item_occlusion in aligned]
    low_threshold = quantile(fractions, q_low)
    high_threshold = quantile(fractions, q_high)
    if high_threshold <= low_threshold:
        raise ValueError("occlusion quantiles do not produce two distinct groups")

    lower_by_sequence: dict[str, list[tuple[ImageStats, OcclusionStats]]] = {}
    higher_by_sequence: dict[str, list[tuple[ImageStats, OcclusionStats]]] = {}
    for item, item_occlusion in aligned:
        sequence = image_sequence_id(item)
        if item_occlusion.occluded_fraction <= low_threshold:
            lower_by_sequence.setdefault(sequence, []).append((item, item_occlusion))
        if item_occlusion.occluded_fraction >= high_threshold:
            higher_by_sequence.setdefault(sequence, []).append((item, item_occlusion))

    feature_rows = [
        (math.log1p(item.objects), math.log(max(item.median_area, 1e-12)))
        for item, _ in aligned
    ]
    feature_means = tuple(sum(row[index] for row in feature_rows) / len(feature_rows) for index in range(2))
    feature_scales = []
    for index in range(2):
        variance = sum((row[index] - feature_means[index]) ** 2 for row in feature_rows) / max(
            len(feature_rows) - 1, 1
        )
        feature_scales.append(max(math.sqrt(variance), 1e-12))

    def normalized_features(item: ImageStats) -> tuple[float, float]:
        features = (math.log1p(item.objects), math.log(max(item.median_area, 1e-12)))
        return tuple(
            (features[index] - feature_means[index]) / feature_scales[index]
            for index in range(2)
        )

    def distance(lower: ImageStats, higher: ImageStats) -> float:
        lower_features = normalized_features(lower)
        higher_features = normalized_features(higher)
        return sum(
            (higher_features[index] - lower_features[index]) ** 2
            for index in range(2)
        )

    candidates_by_sequence: dict[str, list[OcclusionPair]] = {}
    for sequence in sorted(set(lower_by_sequence).intersection(higher_by_sequence)):
        candidates = [
            (
                distance(lower_item, higher_item),
                lower_item.image.name,
                higher_item.image.name,
                lower_item,
                higher_item,
                lower_occlusion,
                higher_occlusion,
            )
            for lower_item, lower_occlusion in lower_by_sequence[sequence]
            for higher_item, higher_occlusion in higher_by_sequence[sequence]
            if lower_item.image != higher_item.image
        ]
        if not candidates:
            continue
        candidates_by_sequence[sequence] = [
            OcclusionPair(
                sequence_id=sequence,
                lower=row[3],
                higher=row[4],
                lower_occlusion=row[5],
                higher_occlusion=row[6],
                match_distance=row[0],
            )
            for row in sorted(candidates, key=lambda row: row[:3])
        ]

    selected_sequences = sorted(
        candidates_by_sequence,
        key=lambda sequence: (candidates_by_sequence[sequence][0].match_distance, sequence),
    )[:limit]
    selected = {sequence: candidates_by_sequence[sequence][0] for sequence in selected_sequences}
    balance_weight = 10.0

    def selection_objective(pairs: list[OcclusionPair]) -> float:
        mean_distance = sum(pair.match_distance for pair in pairs) / len(pairs)
        mean_deltas = []
        for feature_index in range(2):
            deltas = [
                normalized_features(pair.higher)[feature_index] - normalized_features(pair.lower)[feature_index]
                for pair in pairs
            ]
            mean_deltas.append(sum(deltas) / len(deltas))
        return mean_distance + balance_weight * sum(delta**2 for delta in mean_deltas)

    # Local nearest-neighbor choices can accumulate bias in one direction.
    # Coordinate descent keeps one pair per sequence while balancing both covariates globally.
    for _ in range(20):
        changed = False
        for sequence in selected_sequences:
            current = selected[sequence]
            best = current
            best_objective = selection_objective(list(selected.values()))
            for candidate in candidates_by_sequence[sequence]:
                selected[sequence] = candidate
                objective = selection_objective(list(selected.values()))
                if objective + 1e-12 < best_objective:
                    best = candidate
                    best_objective = objective
            selected[sequence] = best
            changed |= best != current
        if not changed:
            break

    matches = sorted(selected.values(), key=lambda pair: pair.sequence_id)
    if len(matches) < 2:
        raise ValueError("fewer than two video sequences contain both lower- and higher-occlusion candidates")

    lower_objects = [math.log1p(pair.lower.objects) for pair in matches]
    higher_objects = [math.log1p(pair.higher.objects) for pair in matches]
    lower_areas = [math.log(max(pair.lower.median_area, 1e-12)) for pair in matches]
    higher_areas = [math.log(max(pair.higher.median_area, 1e-12)) for pair in matches]
    metadata: dict[str, float | int] = {
        "aligned_images": len(aligned),
        "lower_candidate_images": sum(len(items) for items in lower_by_sequence.values()),
        "higher_candidate_images": sum(len(items) for items in higher_by_sequence.values()),
        "paired_sequences": len(matches),
        "low_quantile": q_low,
        "high_quantile": q_high,
        "lower_occlusion_threshold": low_threshold,
        "higher_occlusion_threshold": high_threshold,
        "matched_lower_occlusion_mean": sum(pair.lower_occlusion.occluded_fraction for pair in matches) / len(matches),
        "matched_higher_occlusion_mean": sum(pair.higher_occlusion.occluded_fraction for pair in matches)
        / len(matches),
        "matched_lower_heavy_occlusion_mean": sum(
            pair.lower_occlusion.heavy_occluded_fraction for pair in matches
        )
        / len(matches),
        "matched_higher_heavy_occlusion_mean": sum(
            pair.higher_occlusion.heavy_occluded_fraction for pair in matches
        )
        / len(matches),
        "matched_log_object_count_smd": standardized_mean_difference(lower_objects, higher_objects),
        "matched_log_median_area_smd": standardized_mean_difference(lower_areas, higher_areas),
        "mean_pair_match_distance": sum(pair.match_distance for pair in matches) / len(matches),
        "global_balance_weight": balance_weight,
    }
    return matches, metadata


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
        (
            "image,label,objects,mean_area,median_area,area_cv,aspect_cv,irregular_score,"
            "is_dense,is_sparse,is_small,is_large,is_irregular_proxy"
        )
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


def write_paired_occlusion_scenes(
    out_dir: Path,
    pairs: list[OcclusionPair],
    metadata: dict[str, float | int],
    annotation_source: Path,
    copy: bool,
) -> None:
    """Write disjoint paired scene folders and an auditable covariate-balance manifest."""
    lower_dir = out_dir / "lower_occlusion"
    higher_dir = out_dir / "higher_occlusion"
    reset_dir(lower_dir)
    reset_dir(higher_dir)
    lines = [
        (
            "pair_id,sequence_id,scene,image,objects,median_area,occluded_fraction,"
            "heavy_occluded_fraction,mean_occlusion_level,match_distance"
        )
    ]
    for pair_index, pair in enumerate(pairs):
        for scene, item, item_occlusion, directory in (
            ("lower_occlusion", pair.lower, pair.lower_occlusion, lower_dir),
            ("higher_occlusion", pair.higher, pair.higher_occlusion, higher_dir),
        ):
            link_or_copy(item.image, directory / safe_image_name(item.image), copy=copy)
            lines.append(
                f"{pair_index},{pair.sequence_id},{scene},{item.image.name},{item.objects},"
                f"{item.median_area:.8f},{item_occlusion.occluded_fraction:.8f},"
                f"{item_occlusion.heavy_occluded_fraction:.8f},{item_occlusion.mean_occlusion_level:.8f},"
                f"{pair.match_distance:.8f}"
            )
    (out_dir / "_occlusion_pairs.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        **metadata,
        "annotation_source_name": annotation_source.name,
        "pairing": "one covariate-matched lower/higher image pair per shared VisDrone video sequence",
        "matching_covariates": ["log1p(object_count)", "log(median_box_area)"],
    }
    (out_dir / "_occlusion_balance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/VisDrone"))
    parser.add_argument("--split", default="val", help="Prefer images/<split>; falls back to all images.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-images-per-scene", type=int, default=128)
    parser.add_argument("--low-quantile", type=float, default=0.30)
    parser.add_argument("--high-quantile", type=float, default=0.70)
    parser.add_argument("--copy", action="store_true", help="Copy images instead of creating symlinks.")
    parser.add_argument(
        "--occlusion-pairs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also create lower/higher-occlusion pairs from original VisDrone annotation metadata.",
    )
    parser.add_argument("--visdrone-annotations", type=Path, default=None)
    parser.add_argument("--occlusion-low-quantile", type=float, default=0.25)
    parser.add_argument("--occlusion-high-quantile", type=float, default=0.75)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.low_quantile < args.high_quantile < 1.0:
        raise SystemExit("--low-quantile and --high-quantile must satisfy 0 < low < high < 1")
    if not 0.0 < args.occlusion_low_quantile < args.occlusion_high_quantile < 1.0:
        raise SystemExit("--occlusion quantiles must satisfy 0 < low < high < 1")
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
    if args.occlusion_pairs:
        annotation_source = resolve_visdrone_annotations(dataset, args.split, args.visdrone_annotations)
        occlusion = load_visdrone_occlusion(annotation_source)
        pairs, metadata = match_paired_occlusion_scenes(
            stats,
            occlusion,
            limit=args.max_images_per_scene,
            q_low=args.occlusion_low_quantile,
            q_high=args.occlusion_high_quantile,
        )
        write_paired_occlusion_scenes(out_dir, pairs, metadata, annotation_source, copy=args.copy)
        print(
            f"[scenes] paired occlusion: {len(pairs)} sequences "
            f"({metadata['lower_occlusion_threshold']:.3f} -> {metadata['higher_occlusion_threshold']:.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
