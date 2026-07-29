#!/usr/bin/env python3
"""Audit MoT routing inside matched VisDrone objects instead of whole images.

The script maps original VisDrone boxes onto every spatial MoT router, extracts
inside-object and local-background routing statistics, then matches occluded and
unoccluded objects by video sequence, category, truncation, and box area.
Statistical tests use the video sequence as the repeated-measures unit. This is
a matched observational audit, not proof that occlusion caused a routing change.

Example:
    python scripts/analyze_mot_object_causal.py \
      --model runs/mot_cross_domain/training/v10_mot/weights/best.pt \
      --dataset /path/to/VisDrone \
      --max-images 128 \
      --output runs/mot_object_causal
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_mot_cross_domain import (
    EXPERT_NAMES,
    benjamini_hochberg,
    bootstrap_paired_mean_diff_ci,
    load_model,
    normalize_image_array,
    normalize_torch_device,
    paired_hedges_g,
    paired_permutation_p_value_two_sided,
    stable_seed,
    write_csv,
)
from scripts.prepare_mot_routing_scenes import resolve_visdrone_annotations

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
OBJECT_METRICS = ("inside_probability", "inside_top1_share", "inside_minus_ring_probability")


@dataclass(frozen=True)
class LetterboxGeometry:
    """Geometry required to project original boxes into a square model input."""

    original_width: int
    original_height: int
    input_size: int
    scale: float
    left: int
    top: int
    resized_width: int
    resized_height: int


@dataclass(frozen=True)
class VisDroneObject:
    """One valid object from an original eight-column VisDrone annotation."""

    object_id: str
    image_id: str
    sequence_id: str
    frame_position: int
    category_id: int
    x: float
    y: float
    width: float
    height: float
    normalized_area: float
    truncation: int
    occlusion: int


@dataclass(frozen=True)
class ObjectPair:
    """Matched unoccluded/occluded objects used in paired inference."""

    pair_id: int
    sequence_id: str
    category_id: int
    low_object_id: str
    high_object_id: str
    same_image: bool
    log_area_distance: float
    frame_distance: int


def frame_position(image_id: str) -> int:
    """Extract a deterministic frame-position proxy from a VisDrone image name."""
    parts = Path(image_id).stem.split("_")
    for index in (-1, 1):
        try:
            return int(parts[index])
        except (IndexError, ValueError):
            continue
    return 0


def sequence_id(image_id: str) -> str:
    """Return the video-sequence prefix encoded by VisDrone filenames."""
    stem = Path(image_id).stem
    return stem.split("_", 1)[0] if "_" in stem else stem


def load_annotation_texts(source: Path) -> dict[str, str]:
    """Load original VisDrone annotation text from a directory or official ZIP."""
    output: dict[str, str] = {}
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            for name in sorted(archive.namelist()):
                if "/annotations/" in name and name.lower().endswith(".txt"):
                    output[Path(name).stem] = archive.read(name).decode("utf-8")
    elif source.is_dir():
        for path in sorted(source.rglob("*.txt")):
            output[path.stem] = path.read_text(encoding="utf-8")
    else:
        raise FileNotFoundError(f"VisDrone annotations must be a directory or ZIP: {source}")
    if not output:
        raise ValueError(f"no original VisDrone annotation files found in {source}")
    return output


def parse_visdrone_objects(
    text: str,
    image_id: str,
    image_width: int,
    image_height: int,
) -> list[VisDroneObject]:
    """Parse valid detection objects while preserving truncation and occlusion."""
    objects = []
    image_area = max(image_width * image_height, 1)
    for line_index, line in enumerate(text.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 8:
            continue
        try:
            x, y, width, height = (float(value) for value in parts[:4])
            score, category, truncation, occlusion = (int(value) for value in parts[4:8])
        except ValueError:
            continue
        if score <= 0 or not 1 <= category <= 10 or width <= 0 or height <= 0 or occlusion < 0:
            continue
        clipped_x = min(max(x, 0.0), float(image_width))
        clipped_y = min(max(y, 0.0), float(image_height))
        clipped_x1 = min(max(x + width, 0.0), float(image_width))
        clipped_y1 = min(max(y + height, 0.0), float(image_height))
        clipped_width = max(clipped_x1 - clipped_x, 0.0)
        clipped_height = max(clipped_y1 - clipped_y, 0.0)
        objects.append(
            VisDroneObject(
                object_id=f"{Path(image_id).stem}:{line_index}",
                image_id=Path(image_id).name,
                sequence_id=sequence_id(image_id),
                frame_position=frame_position(image_id),
                category_id=category - 1,
                x=clipped_x,
                y=clipped_y,
                width=clipped_width,
                height=clipped_height,
                normalized_area=float(clipped_width * clipped_height / image_area),
                truncation=truncation,
                occlusion=occlusion,
            )
        )
    return [item for item in objects if item.width > 0 and item.height > 0]


def load_image_with_geometry(path: Path, imgsz: int) -> tuple[torch.Tensor, LetterboxGeometry]:
    """Load an image using the audit normalization and return letterbox geometry."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        array = normalize_image_array(np.asarray(image))
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    _, height, width = tensor.shape
    scale = min(imgsz / max(width, 1), imgsz / max(height, 1))
    new_height = max(1, min(imgsz, round(height * scale)))
    new_width = max(1, min(imgsz, round(width * scale)))
    resized = F.interpolate(
        tensor.unsqueeze(0),
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    canvas = torch.full((3, imgsz, imgsz), 114.0 / 255.0, dtype=torch.float32)
    top = (imgsz - new_height) // 2
    left = (imgsz - new_width) // 2
    canvas[:, top : top + new_height, left : left + new_width] = resized
    geometry = LetterboxGeometry(
        original_width=width,
        original_height=height,
        input_size=imgsz,
        scale=scale,
        left=left,
        top=top,
        resized_width=new_width,
        resized_height=new_height,
    )
    return canvas, geometry


def feature_roi(
    annotation: VisDroneObject,
    geometry: LetterboxGeometry,
    feature_height: int,
    feature_width: int,
) -> tuple[int, int, int, int]:
    """Project one original-image box into integer feature-map coordinates."""
    input_x0 = annotation.x * geometry.scale + geometry.left
    input_y0 = annotation.y * geometry.scale + geometry.top
    input_x1 = (annotation.x + annotation.width) * geometry.scale + geometry.left
    input_y1 = (annotation.y + annotation.height) * geometry.scale + geometry.top
    x0 = max(0, min(feature_width - 1, math.floor(input_x0 / geometry.input_size * feature_width)))
    y0 = max(0, min(feature_height - 1, math.floor(input_y0 / geometry.input_size * feature_height)))
    x1 = max(x0 + 1, min(feature_width, math.ceil(input_x1 / geometry.input_size * feature_width)))
    y1 = max(y0 + 1, min(feature_height, math.ceil(input_y1 / geometry.input_size * feature_height)))
    return x0, y0, x1, y1


def roi_routing_metrics(
    probabilities: torch.Tensor,
    roi: tuple[int, int, int, int],
) -> list[dict[str, float | int]]:
    """Summarize expert probabilities inside a box and its one-cell outer ring."""
    if probabilities.ndim != 3:
        raise ValueError(f"probabilities must have shape [E,H,W], got {tuple(probabilities.shape)}")
    experts, height, width = probabilities.shape
    x0, y0, x1, y1 = roi
    inside = probabilities[:, y0:y1, x0:x1]
    if inside.numel() == 0:
        raise ValueError(f"empty feature ROI: {roi} for map {(height, width)}")

    ring_x0, ring_y0 = max(0, x0 - 1), max(0, y0 - 1)
    ring_x1, ring_y1 = min(width, x1 + 1), min(height, y1 + 1)
    ring_region = probabilities[:, ring_y0:ring_y1, ring_x0:ring_x1]
    ring_mask = torch.ones(
        (ring_y1 - ring_y0, ring_x1 - ring_x0),
        dtype=torch.bool,
        device=probabilities.device,
    )
    ring_mask[y0 - ring_y0 : y1 - ring_y0, x0 - ring_x0 : x1 - ring_x0] = False
    top1 = probabilities.argmax(dim=0)

    output = []
    for expert_id in range(experts):
        inside_probability = float(inside[expert_id].mean())
        ring_values = ring_region[expert_id][ring_mask]
        ring_probability = float(ring_values.mean()) if ring_values.numel() else float("nan")
        output.append(
            {
                "expert_id": expert_id,
                "inside_probability": inside_probability,
                "ring_probability": ring_probability,
                "inside_minus_ring_probability": inside_probability - ring_probability,
                "inside_top1_share": float((top1[y0:y1, x0:x1] == expert_id).float().mean()),
                "feature_cells": int((y1 - y0) * (x1 - x0)),
            }
        )
    return output


class SpatialRoutingCollector:
    """Capture dense pre-Top-K probabilities from every MoTBlock."""

    def __init__(self, model: torch.nn.Module):
        from ultralytics.nn.modules.mot import MoTBlock

        self.maps: dict[str, torch.Tensor] = {}
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, MoTBlock):
                self.handles.append(module.router.register_forward_hook(self._hook(name)))
        if not self.handles:
            raise ValueError("the supplied model contains no MoTBlock")

    def _hook(self, layer_name: str):
        def capture(module, _inputs, output):
            if not isinstance(output, tuple) or len(output) < 3:
                raise RuntimeError("MoT router must return weights, indices, and logits")
            logits = output[2].detach().float()
            temperature = max(float(module.temperature), torch.finfo(torch.float32).tiny)
            self.maps[layer_name] = torch.softmax(logits / temperature, dim=1).cpu()

        return capture

    def clear(self) -> None:
        self.maps.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def discover_images(dataset: Path, split: str) -> dict[str, Path]:
    """Return public image ids mapped to files for one dataset split."""
    root = dataset / "images" / split
    if not root.is_dir():
        raise FileNotFoundError(f"image split not found: {root}")
    return {
        path.stem: path for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def select_image_ids(
    image_paths: dict[str, Path],
    annotations: dict[str, str],
    max_images: int,
    seed: int,
) -> list[str]:
    """Choose a deterministic image subset containing at least one valid object."""
    eligible = sorted(set(image_paths) & set(annotations))
    if max_images <= 0 or max_images >= len(eligible):
        return eligible
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(eligible), size=max_images, replace=False))
    return [eligible[int(index)] for index in indices]


def collect_object_routing(
    model: torch.nn.Module,
    image_paths: dict[str, Path],
    annotation_texts: dict[str, str],
    image_ids: list[str],
    *,
    device: str,
    imgsz: int,
) -> tuple[list[dict[str, Any]], list[VisDroneObject]]:
    """Run selected images and extract per-object, per-layer routing metrics."""
    collector = SpatialRoutingCollector(model)
    records: list[dict[str, Any]] = []
    all_objects: list[VisDroneObject] = []
    try:
        with torch.inference_mode():
            for image_id in image_ids:
                image, geometry = load_image_with_geometry(image_paths[image_id], imgsz)
                objects = parse_visdrone_objects(
                    annotation_texts[image_id],
                    image_paths[image_id].name,
                    geometry.original_width,
                    geometry.original_height,
                )
                if not objects:
                    continue
                collector.clear()
                _ = model(image.unsqueeze(0).to(device))
                for annotation in objects:
                    all_objects.append(annotation)
                    for layer_name, batch_probabilities in sorted(collector.maps.items()):
                        probabilities = batch_probabilities[0]
                        roi = feature_roi(
                            annotation,
                            geometry,
                            feature_height=probabilities.shape[1],
                            feature_width=probabilities.shape[2],
                        )
                        for metric in roi_routing_metrics(probabilities, roi):
                            expert_id = int(metric.pop("expert_id"))
                            records.append(
                                {
                                    **asdict(annotation),
                                    "layer": layer_name,
                                    "expert_id": expert_id,
                                    "expert": (
                                        EXPERT_NAMES[expert_id]
                                        if expert_id < len(EXPERT_NAMES)
                                        else f"Expert{expert_id}"
                                    ),
                                    **metric,
                                }
                            )
    finally:
        collector.close()
    return records, all_objects


def match_occlusion_objects(
    objects: list[VisDroneObject],
    *,
    min_high_occlusion: int,
    max_pairs: int,
    max_log_area_distance: float = math.log(2.0),
) -> list[ObjectPair]:
    """Match low/high-occlusion objects with exact covariates and an area caliper."""
    if min_high_occlusion < 1:
        raise ValueError("min_high_occlusion must be at least 1")
    if not math.isfinite(max_log_area_distance) or max_log_area_distance <= 0:
        raise ValueError("max_log_area_distance must be positive and finite")
    grouped: dict[tuple[str, int, int], list[VisDroneObject]] = {}
    for item in objects:
        grouped.setdefault((item.sequence_id, item.category_id, item.truncation), []).append(item)

    matched_candidates: list[tuple[tuple[Any, ...], VisDroneObject, VisDroneObject]] = []
    for (current_sequence, category, _truncation), items in sorted(grouped.items()):
        candidates: list[tuple[tuple[Any, ...], VisDroneObject, VisDroneObject]] = []
        lows = [item for item in items if item.occlusion == 0]
        highs = [item for item in items if item.occlusion >= min_high_occlusion]
        for low in lows:
            for high in highs:
                area_distance = abs(
                    math.log(max(low.normalized_area, 1e-12)) - math.log(max(high.normalized_area, 1e-12))
                )
                if area_distance > max_log_area_distance:
                    continue
                current_frame_distance = abs(low.frame_position - high.frame_position)
                priority = (
                    low.image_id != high.image_id,
                    area_distance,
                    current_frame_distance,
                    current_sequence,
                    category,
                    low.object_id,
                    high.object_id,
                )
                candidates.append((priority, low, high))

        # Groups are object-disjoint, so independent greedy matching is
        # equivalent to one global sort while keeping far fewer candidates.
        used_low: set[str] = set()
        used_high: set[str] = set()
        for candidate in sorted(candidates, key=lambda item: item[0]):
            priority, low, high = candidate
            if low.object_id in used_low or high.object_id in used_high:
                continue
            used_low.add(low.object_id)
            used_high.add(high.object_id)
            matched_candidates.append((priority, low, high))

    pairs = []
    for priority, low, high in sorted(matched_candidates, key=lambda item: item[0]):
        pairs.append(
            ObjectPair(
                pair_id=len(pairs),
                sequence_id=low.sequence_id,
                category_id=low.category_id,
                low_object_id=low.object_id,
                high_object_id=high.object_id,
                same_image=low.image_id == high.image_id,
                log_area_distance=float(priority[1]),
                frame_distance=int(priority[2]),
            )
        )
        if max_pairs > 0 and len(pairs) >= max_pairs:
            break
    return pairs


def object_pair_statistics(
    records: list[dict[str, Any]],
    pairs: list[ObjectPair],
    *,
    bootstrap_samples: int,
    permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Compute paired sequence-cluster inference for matched object routes."""
    by_key = {(str(row["object_id"]), str(row["layer"]), str(row["expert"])): row for row in records}
    layers = sorted({str(row["layer"]) for row in records})
    experts = sorted({str(row["expert"]) for row in records})
    output = []
    p_values = []
    high_occlusion = min_high_occlusion_from_pairs(pairs, records)
    for layer in layers:
        for expert in experts:
            for metric in OBJECT_METRICS:
                sequence_values: dict[str, list[tuple[float, float]]] = {}
                pair_count = 0
                for pair in pairs:
                    low = by_key.get((pair.low_object_id, layer, expert))
                    high = by_key.get((pair.high_object_id, layer, expert))
                    if low is None or high is None:
                        continue
                    low_value, high_value = float(low[metric]), float(high[metric])
                    if not np.isfinite(low_value) or not np.isfinite(high_value):
                        continue
                    sequence_values.setdefault(pair.sequence_id, []).append((low_value, high_value))
                    pair_count += 1
                sequence_ids = sorted(sequence_values)
                values_low = np.asarray(
                    [np.mean([item[0] for item in sequence_values[item_id]]) for item_id in sequence_ids],
                    dtype=np.float64,
                )
                values_high = np.asarray(
                    [np.mean([item[1] for item in sequence_values[item_id]]) for item_id in sequence_ids],
                    dtype=np.float64,
                )
                statistic_seed = stable_seed(seed, layer, expert, metric)
                ci_low, ci_high = bootstrap_paired_mean_diff_ci(
                    values_low,
                    values_high,
                    bootstrap_samples,
                    statistic_seed,
                )
                p_value = paired_permutation_p_value_two_sided(
                    values_low,
                    values_high,
                    permutations,
                    statistic_seed + 1,
                )
                row = {
                    "analysis_unit": "video_sequence",
                    "comparison": f"occlusion_0_to_{high_occlusion}+",
                    "layer": layer,
                    "expert": expert,
                    "metric": metric,
                    "n_pairs": pair_count,
                    "n_sequences": len(sequence_ids),
                    "mean_low": float(values_low.mean()) if values_low.size else float("nan"),
                    "mean_high": float(values_high.mean()) if values_high.size else float("nan"),
                    "mean_diff_high_minus_low": (
                        float((values_high - values_low).mean()) if values_low.size else float("nan")
                    ),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "ci_excludes_zero": bool(np.isfinite(ci_low) and (ci_low > 0 or ci_high < 0)),
                    "paired_hedges_g": paired_hedges_g(values_low, values_high),
                    "permutation_p_value_two_sided": p_value,
                }
                output.append(row)
                p_values.append(p_value)

    q_values = benjamini_hochberg(p_values)
    for row, q_value in zip(output, q_values):
        row["fdr_q_value"] = q_value
        row["significant_after_fdr"] = bool(np.isfinite(q_value) and q_value <= 0.05 and row["ci_excludes_zero"])
    return output


def min_high_occlusion_from_pairs(pairs: list[ObjectPair], records: list[dict[str, Any]]) -> int:
    """Recover the minimum high-object occlusion represented by a pair list."""
    object_occlusion = {str(row["object_id"]): int(row["occlusion"]) for row in records}
    values = [object_occlusion[pair.high_object_id] for pair in pairs if pair.high_object_id in object_occlusion]
    return min(values) if values else 1


def write_pair_csv(path: Path, pairs: list[ObjectPair], objects: list[VisDroneObject]) -> None:
    """Write matched object identities and covariates without local paths."""
    by_id = {item.object_id: item for item in objects}
    rows = []
    for pair in pairs:
        low, high = by_id[pair.low_object_id], by_id[pair.high_object_id]
        rows.append(
            {
                **asdict(pair),
                "low_image_id": low.image_id,
                "high_image_id": high.image_id,
                "low_occlusion": low.occlusion,
                "high_occlusion": high.occlusion,
                "low_truncation": low.truncation,
                "high_truncation": high.truncation,
                "low_normalized_area": low.normalized_area,
                "high_normalized_area": high.normalized_area,
            }
        )
    write_csv(path, rows)


def plot_object_delta_heatmap(statistics: list[dict[str, Any]], output: Path) -> None:
    """Plot inside-object probability changes for high versus low occlusion."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    selected = [row for row in statistics if row["metric"] == "inside_probability"]
    layers = sorted({str(row["layer"]) for row in selected})
    experts = list(EXPERT_NAMES)
    matrix = np.full((len(layers), len(experts)), np.nan, dtype=np.float64)
    for row in selected:
        if row["expert"] in experts:
            matrix[layers.index(str(row["layer"])), experts.index(str(row["expert"]))] = row["mean_diff_high_minus_low"]
    figure, axis = plt.subplots(figsize=(8.5, max(3.5, 0.55 * len(layers) + 1.5)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="+.5f",
        center=0.0,
        cmap="vlag",
        xticklabels=experts,
        yticklabels=layers,
        ax=axis,
    )
    axis.set_title("Object-level routing probability: high occlusion minus unoccluded")
    axis.set_xlabel("Transformer expert")
    axis.set_ylabel("MoT layer")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def file_sha256(path: Path) -> str:
    """Return a reproducible checkpoint identity."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--visdrone-annotations", type=Path, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-images", type=int, default=128)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--min-high-occlusion", type=int, choices=(1, 2), default=1)
    parser.add_argument("--max-log-area-distance", type=float, default=math.log(2.0))
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.imgsz <= 0 or args.bootstrap_samples <= 0 or args.permutations <= 0:
        raise SystemExit("--imgsz, --bootstrap-samples and --permutations must be positive")
    if not math.isfinite(args.max_log_area_distance) or args.max_log_area_distance <= 0:
        raise SystemExit("--max-log-area-distance must be positive and finite")
    dataset = args.dataset.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    annotation_source = resolve_visdrone_annotations(dataset, args.split, args.visdrone_annotations)
    annotation_texts = load_annotation_texts(annotation_source)
    image_paths = discover_images(dataset, args.split)
    image_ids = select_image_ids(image_paths, annotation_texts, args.max_images, args.seed)
    device = normalize_torch_device(args.device)
    model = load_model(model_path, device, nc=10)

    records, objects = collect_object_routing(
        model,
        image_paths,
        annotation_texts,
        image_ids,
        device=device,
        imgsz=args.imgsz,
    )
    if not records or not objects:
        raise SystemExit("no valid object-level routing records were collected")
    pairs = match_occlusion_objects(
        objects,
        min_high_occlusion=args.min_high_occlusion,
        max_pairs=args.max_pairs,
        max_log_area_distance=args.max_log_area_distance,
    )
    if not pairs:
        raise SystemExit(
            "no occlusion-matched object pairs were found; increase --max-images or lower --min-high-occlusion"
        )
    statistics = object_pair_statistics(
        records,
        pairs,
        bootstrap_samples=args.bootstrap_samples,
        permutations=args.permutations,
        seed=args.seed,
    )
    write_csv(output / "object_routing_detailed.csv", records)
    write_pair_csv(output / "object_pairs.csv", pairs, objects)
    write_csv(output / "object_pair_statistics.csv", statistics)
    plot_object_delta_heatmap(statistics, output / "object_routing_delta_heatmap.png")
    manifest = {
        "model_sha256": file_sha256(model_path),
        "same_checkpoint_for_all_objects": True,
        "dataset": "VisDrone",
        "split": args.split,
        "annotation_source_name": annotation_source.name,
        "image_size": args.imgsz,
        "selected_images": len(image_ids),
        "valid_objects": len(objects),
        "matched_pairs": len(pairs),
        "paired_sequences": len({pair.sequence_id for pair in pairs}),
        "same_image_pairs": sum(pair.same_image for pair in pairs),
        "min_high_occlusion": args.min_high_occlusion,
        "exact_match_covariates": ["video_sequence", "category", "truncation"],
        "max_log_area_distance": args.max_log_area_distance,
        "bootstrap_samples": args.bootstrap_samples,
        "permutations": args.permutations,
        "seed": args.seed,
        "significance_rule": "BH-FDR q <= 0.05 and paired bootstrap CI excludes zero",
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    significant = sum(bool(row["significant_after_fdr"]) for row in statistics)
    print(
        f"[object-audit] images={len(image_ids)} objects={len(objects)} "
        f"pairs={len(pairs)} sequences={manifest['paired_sequences']} significant={significant}"
    )
    print(f"[object-audit] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
