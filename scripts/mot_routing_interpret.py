#!/usr/bin/env python3
"""Pure analysis helpers for MoT (Mixture-of-Transformers) routing interpretability.

This module holds the side-effect-free statistics used by
``scripts/run_mot_routing_interpret.py``. Keeping them separate makes every
claim in the generated report reproducible and unit-testable.

Three design choices matter for correctness:

1. **Expert similarity is measured on the residual delta, not the output.**
   Every MoT expert is ``x + ls1 * attn(x) + ls2 * ffn(x)`` with ``ls`` initialised
   to 0.1, and :class:`~ultralytics.nn.modules.mot.block.MoTBlock` adds a further
   block-level residual. Cosine similarity between raw expert *outputs* is
   therefore dominated by the shared identity path and sits near 1.0 even when the
   experts are completely different functions. :func:`delta_similarity_report`
   strips the identity path first.
2. **Occlusion is confounded with density and object scale in VisDrone.** Crowded
   frames contain more occluded objects and smaller boxes, so a naive
   occluded-vs-clear image split cannot attribute a routing shift to occlusion.
   :func:`stratified_permutation_test` controls image-level covariates and
   :func:`token_object_masks` exposes ``*_solo`` masks for a within-image,
   single-object token comparison.
3. **Many hypotheses are tested at once**, so :func:`benjamini_hochberg` controls
   the false discovery rate across the whole family rather than per test.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

EXPERT_NAMES: tuple[str, ...] = ("LocalConvTransformer", "WindowTransformer", "DeformableTransformer")
"""MoT expert branch names, in ``MoTBlock.experts`` order."""

TOKEN_GROUPS: tuple[str, ...] = ("occluded", "clear", "background", "occluded_solo", "clear_solo")
"""Token mask names produced by :func:`token_object_masks`."""

ASPECT_IRREGULAR_THRESHOLD: float = 3.0
"""Pixel aspect ratio beyond which a box counts as irregularly shaped.

A box is irregular when ``w/h`` exceeds this value or falls below its reciprocal.
The ratio is taken in *pixels*, not in YOLO-normalized units: normalized width and
height are each divided by a different image dimension, so a normalized ratio
rescales every box by the image aspect and would misclassify shapes on non-square
images.
"""


@dataclass(frozen=True)
class VisDroneObject:
    """One annotated VisDrone object in absolute pixel coordinates.

    Args:
        left (int): Box left edge in pixels.
        top (int): Box top edge in pixels.
        width (int): Box width in pixels.
        height (int): Box height in pixels.
        category (int): VisDrone object category (1-10 are the trained classes).
        truncation (int): 0 = none, 1 = truncated.
        occlusion (int): 0 = none, 1 = partial (1-50%), 2 = heavy (>50%).
    """

    left: int
    top: int
    width: int
    height: int
    category: int
    truncation: int
    occlusion: int

    def normalized_box(self, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        """Return ``(x0, y0, x1, y1)`` normalized to ``[0, 1]``.

        Examples:
            >>> obj = VisDroneObject(10, 20, 30, 40, 1, 0, 0)
            >>> tuple(round(value, 4) for value in obj.normalized_box(100, 200))
            (0.1, 0.1, 0.4, 0.3)
        """
        x0 = self.left / image_width
        y0 = self.top / image_height
        return x0, y0, x0 + self.width / image_width, y0 + self.height / image_height


def parse_visdrone_annotation(text: str) -> list[VisDroneObject]:
    """Parse a VisDrone ``annotations/*.txt`` file body into objects.

    Rows with ``score == 0`` are VisDrone *ignored regions* and rows with
    ``category`` 0 (ignored) or 11 (others) are not trained classes; all are
    dropped, matching ``ultralytics``' own VisDrone converter.

    Args:
        text (str): Raw annotation file contents.

    Returns:
        (list[VisDroneObject]): Valid annotated objects.

    Examples:
        >>> body = "10,20,30,40,1,4,0,1\\n0,0,5,5,0,4,0,0\\n50,60,10,10,1,11,0,0\\n"
        >>> objects = parse_visdrone_annotation(body)
        >>> len(objects)
        1
        >>> objects[0].occlusion, objects[0].category
        (1, 4)
    """
    objects: list[VisDroneObject] = []
    for line in text.strip().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 8:
            continue
        try:
            values = [int(float(part)) for part in parts[:8]]
        except ValueError:
            continue
        left, top, width, height, score, category, truncation, occlusion = values
        if score == 0 or category in (0, 11) or width <= 0 or height <= 0:
            continue
        objects.append(VisDroneObject(left, top, width, height, category, truncation, occlusion))
    return objects


def covered_fraction(
    target: tuple[float, float, float, float], others: Sequence[tuple[float, float, float, float]]
) -> float:
    """Fraction of ``target``'s area covered by the union of ``others``.

    The union is approximated on a coarse raster rather than by inclusion-exclusion,
    which keeps the cost linear in the number of neighbours and is accurate enough
    for the three-level occlusion bucketing it feeds.

    Args:
        target (tuple[float, float, float, float]): ``(x0, y0, x1, y1)`` box.
        others (Sequence[tuple]): Candidate occluder boxes in the same units.

    Returns:
        (float): Covered area fraction in ``[0, 1]``; 0.0 for a degenerate target.

    Examples:
        >>> round(covered_fraction((0, 0, 1, 1), [(0.5, 0, 1.5, 1)]), 2)
        0.5
        >>> covered_fraction((0, 0, 1, 1), [(2, 2, 3, 3)])
        0.0
        >>> round(covered_fraction((0, 0, 1, 1), [(0, 0, 1, 1)]), 2)
        1.0
    """
    x0, y0, x1, y1 = target
    if x1 <= x0 or y1 <= y0:
        return 0.0
    resolution = 32
    grid_x = x0 + (np.arange(resolution) + 0.5) * (x1 - x0) / resolution
    grid_y = y0 + (np.arange(resolution) + 0.5) * (y1 - y0) / resolution
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    covered = np.zeros((resolution, resolution), dtype=bool)
    for ox0, oy0, ox1, oy1 in others:
        covered |= (mesh_x >= ox0) & (mesh_x < ox1) & (mesh_y >= oy0) & (mesh_y < oy1)
    return float(covered.mean())


def occlusion_code_from_overlap(fraction: float, *, partial: float = 0.1, heavy: float = 0.5) -> int:
    """Bucket a covered-area fraction into the VisDrone occlusion scale.

    Lets a dataset without occlusion annotations (COCO) reuse the same analysis
    path. The result is a *proxy*: neighbour boxes overlapping in 2D need not
    occlude in 3D, so a COCO result is weaker evidence than a VisDrone one.

    Args:
        fraction (float): Covered area fraction from :func:`covered_fraction`.
        partial (float): Threshold above which an object counts as partly occluded.
        heavy (float): Threshold above which an object counts as heavily occluded.

    Returns:
        (int): 0 = none, 1 = partial, 2 = heavy — matching ``VisDroneObject.occlusion``.

    Examples:
        >>> [occlusion_code_from_overlap(f) for f in (0.0, 0.05, 0.2, 0.8)]
        [0, 0, 1, 2]
    """
    if fraction >= heavy:
        return 2
    if fraction >= partial:
        return 1
    return 0


def coco_objects_by_image(
    instances: dict,
    *,
    partial: float = 0.1,
    heavy: float = 0.5,
) -> dict[str, tuple[list[VisDroneObject], int, int]]:
    """Convert COCO ``instances_*.json`` into per-image :class:`VisDroneObject` lists.

    Occlusion is unavailable in COCO, so it is derived: an object's occlusion code
    comes from how much of its box other boxes cover, and any ``iscrowd`` object is
    forced to heavy. ``truncation`` carries whether the box touches the image edge.

    Args:
        instances (dict): Parsed COCO detection annotation file.
        partial (float): Covered-fraction threshold for partial occlusion.
        heavy (float): Covered-fraction threshold for heavy occlusion.

    Returns:
        (dict): Maps image file stem to ``(objects, width, height)``.

    Examples:
        A box overlapped across a fifth of its width is partial; a heavily
        overlapped box and an ``iscrowd`` region are both heavy; an isolated box is
        clear. Edge-touching boxes are flagged as truncated:

        >>> instances = {
        ...     "images": [{"id": 1, "file_name": "000001.jpg", "width": 200, "height": 100}],
        ...     "annotations": [
        ...         {"image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 50], "iscrowd": 0},
        ...         {"image_id": 1, "category_id": 3, "bbox": [120, 60, 20, 20], "iscrowd": 0},
        ...         {"image_id": 1, "category_id": 1, "bbox": [150, 60, 30, 30], "iscrowd": 1},
        ...         {"image_id": 1, "category_id": 2, "bbox": [105, 5, 20, 20], "iscrowd": 0},
        ...         {"image_id": 1, "category_id": 2, "bbox": [100, 0, 40, 40], "iscrowd": 0},
        ...     ],
        ... }
        >>> objects, width, height = coco_objects_by_image(instances)["000001"]
        >>> (width, height), len(objects)
        ((200, 100), 5)
        >>> [obj.occlusion for obj in objects]  # isolated, isolated, crowd, engulfed, partial
        [0, 0, 2, 2, 1]
        >>> [obj.truncation for obj in objects]  # only the first touches an image edge
        [1, 0, 0, 0, 1]
    """
    meta = {image["id"]: image for image in instances["images"]}
    grouped: dict[int, list[dict]] = {}
    for annotation in instances["annotations"]:
        if annotation.get("bbox") is None:
            continue
        _, _, box_w, box_h = annotation["bbox"]
        if box_w <= 0 or box_h <= 0:
            continue
        grouped.setdefault(annotation["image_id"], []).append(annotation)

    result: dict[str, tuple[list[VisDroneObject], int, int]] = {}
    for image_id, records in grouped.items():
        image = meta.get(image_id)
        if image is None:
            continue
        width, height = int(image["width"]), int(image["height"])
        boxes = [
            (
                float(a["bbox"][0]),
                float(a["bbox"][1]),
                float(a["bbox"][0]) + float(a["bbox"][2]),
                float(a["bbox"][1]) + float(a["bbox"][3]),
            )
            for a in records
        ]
        objects: list[VisDroneObject] = []
        for index, annotation in enumerate(records):
            others = [box for position, box in enumerate(boxes) if position != index]
            fraction = covered_fraction(boxes[index], others)
            occlusion = (
                2 if annotation.get("iscrowd") else occlusion_code_from_overlap(fraction, partial=partial, heavy=heavy)
            )
            x0, y0, x1, y1 = boxes[index]
            truncation = int(x0 <= 1 or y0 <= 1 or x1 >= width - 1 or y1 >= height - 1)
            objects.append(
                VisDroneObject(
                    left=int(x0),
                    top=int(y0),
                    width=int(x1 - x0),
                    height=int(y1 - y0),
                    category=int(annotation["category_id"]),
                    truncation=truncation,
                    occlusion=occlusion,
                )
            )
        result[Path(image["file_name"]).stem] = (objects, width, height)
    return result


@dataclass(frozen=True)
class ImageSceneFeatures:
    """Scalar scene descriptors for one image, used for grouping and matching."""

    n_objects: int
    median_area: float
    mean_area: float
    area_cv: float
    aspect_cv: float
    occlusion_rate: float
    heavy_occlusion_rate: float
    truncation_rate: float
    irregular_rate: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable mapping of the descriptors."""
        return asdict(self)


def box_area(obj: VisDroneObject, width: int, height: int) -> float:
    """Normalized area of one box in ``[0, 1]``.

    Args:
        obj (VisDroneObject): Annotated object.
        width (int): Image width in pixels.
        height (int): Image height in pixels.

    Returns:
        (float): ``(w / W) * (h / H)``.

    Examples:
        >>> round(box_area(VisDroneObject(0, 0, 10, 20, 1, 0, 0), 100, 100), 4)
        0.02
    """
    return (obj.width / width) * (obj.height / height)


def is_irregular_box(obj: VisDroneObject, *, threshold: float = ASPECT_IRREGULAR_THRESHOLD) -> bool:
    """Whether a box is elongated enough to count as irregularly shaped.

    Args:
        obj (VisDroneObject): Annotated object.
        threshold (float): Aspect-ratio cutoff; see :data:`ASPECT_IRREGULAR_THRESHOLD`.

    Returns:
        (bool): True for a very wide or very tall box.

    Examples:
        >>> is_irregular_box(VisDroneObject(0, 0, 40, 10, 1, 0, 0))
        True
        >>> is_irregular_box(VisDroneObject(0, 0, 10, 40, 1, 0, 0))
        True
        >>> is_irregular_box(VisDroneObject(0, 0, 12, 10, 1, 0, 0))
        False
    """
    ratio = obj.width / max(obj.height, 1)
    return ratio >= threshold or ratio <= 1.0 / threshold


def _coefficient_of_variation(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if mean <= 0:
        return 0.0
    return float(array.std(ddof=0) / mean)


def image_scene_features(objects: Sequence[VisDroneObject], width: int, height: int) -> ImageSceneFeatures:
    """Summarize one image's object statistics.

    Areas and aspect ratios are computed in normalized units so images of
    different resolutions stay comparable.

    Args:
        objects (Sequence[VisDroneObject]): Valid objects for the image.
        width (int): Image width in pixels.
        height (int): Image height in pixels.

    Returns:
        (ImageSceneFeatures): Scene descriptors; all zeros when there are no objects.

    Examples:
        >>> objects = [VisDroneObject(0, 0, 10, 10, 1, 0, 0), VisDroneObject(20, 20, 10, 10, 1, 0, 2)]
        >>> features = image_scene_features(objects, 100, 100)
        >>> features.n_objects, features.occlusion_rate, features.heavy_occlusion_rate
        (2, 0.5, 0.5)
        >>> round(features.median_area, 4)
        0.01
        >>> features.irregular_rate
        0.0
        >>> image_scene_features([*objects, VisDroneObject(0, 0, 40, 5, 1, 0, 0)], 100, 100).irregular_rate
        0.3333333333333333
    """
    if not objects:
        return ImageSceneFeatures(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    areas = [box_area(obj, width, height) for obj in objects]
    aspects = [(obj.width / max(obj.height, 1)) for obj in objects]
    occluded = sum(1 for obj in objects if obj.occlusion >= 1)
    heavy = sum(1 for obj in objects if obj.occlusion >= 2)
    truncated = sum(1 for obj in objects if obj.truncation >= 1)
    irregular = sum(1 for obj in objects if is_irregular_box(obj))
    count = len(objects)
    return ImageSceneFeatures(
        n_objects=count,
        median_area=float(np.median(areas)),
        mean_area=float(np.mean(areas)),
        area_cv=_coefficient_of_variation(areas),
        aspect_cv=_coefficient_of_variation(aspects),
        occlusion_rate=occluded / count,
        heavy_occlusion_rate=heavy / count,
        truncation_rate=truncated / count,
        irregular_rate=irregular / count,
    )


def _token_grid(grid_h: int, grid_w: int) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized ``(x, y)`` token-centre coordinate meshes."""
    centre_y = (np.arange(grid_h, dtype=np.float64) + 0.5) / grid_h
    centre_x = (np.arange(grid_w, dtype=np.float64) + 0.5) / grid_w
    return np.meshgrid(centre_x, centre_y)


def _box_token_mask(obj: VisDroneObject, width: int, height: int, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """Tokens whose centre falls in ``obj``, or its single nearest token if none do."""
    x0, y0, x1, y1 = obj.normalized_box(width, height)
    inside = (grid_x >= x0) & (grid_x < x1) & (grid_y >= y0) & (grid_y < y1)
    if inside.any():
        return inside
    distance = (grid_x - 0.5 * (x0 + x1)) ** 2 + (grid_y - 0.5 * (y0 + y1)) ** 2
    nearest = np.zeros_like(inside)
    nearest.flat[int(np.argmin(distance))] = True
    return nearest


def token_group_masks(
    objects: Sequence[VisDroneObject],
    width: int,
    height: int,
    grid_h: int,
    grid_w: int,
    predicate,
    *,
    area_range: tuple[float, float] | None = None,
    others: Sequence[VisDroneObject] = (),
) -> dict[str, np.ndarray]:
    """Split tokens into two arms of an object-level contrast.

    A token belongs to a box when the token *centre* falls inside it. Boxes that
    cover no token centre (common for small objects on coarse grids) claim their
    single nearest token so small objects are never silently dropped.

    The ``*_solo`` masks select tokens covered by exactly one box. Comparing those
    two arms *within the same image* removes the density and box-overlap confounds
    that make an image-level scene split hard to attribute. ``others`` holds boxes
    that belong to neither arm but still occupy tokens, so a three-way split (e.g.
    small / medium / large) keeps an honest occupancy count.

    Args:
        objects (Sequence[VisDroneObject]): Objects taking part in the contrast.
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        grid_h (int): Feature-map height (token rows).
        grid_w (int): Feature-map width (token columns).
        predicate (Callable[[VisDroneObject], bool]): True for the positive arm.
        area_range (tuple[float, float] | None): Optional inclusive normalized-area
            band; boxes outside it move to the occupancy-only ``others`` role, which
            is how the size-matched variants of the token tests are built.
        others (Sequence[VisDroneObject]): Extra boxes counted for occupancy only.

    Returns:
        (dict[str, np.ndarray]): ``positive``, ``negative``, ``background``,
            ``positive_solo``, ``negative_solo`` boolean masks and the integer
            ``box_count`` map, each shaped ``(grid_h, grid_w)``.

    Examples:
        >>> wide = VisDroneObject(0, 0, 50, 10, 1, 0, 0)
        >>> square = VisDroneObject(50, 0, 50, 100, 1, 0, 0)
        >>> masks = token_group_masks([wide, square], 100, 100, 1, 2, is_irregular_box)
        >>> masks["positive_solo"].tolist(), masks["negative_solo"].tolist()
        ([[True, False]], [[False, True]])

        A box excluded by ``area_range`` still occupies its tokens, so the token it
        shares with the wide box is no longer solo:

        >>> masks = token_group_masks([wide, square], 100, 100, 1, 2, is_irregular_box, area_range=(0.0, 0.2))
        >>> masks["positive"].tolist(), masks["positive_solo"].tolist()
        ([[True, False]], [[True, False]])
        >>> masks["box_count"].tolist()
        [[1, 1]]
    """
    grid_x, grid_y = _token_grid(grid_h, grid_w)
    positive_count = np.zeros((grid_h, grid_w), dtype=np.int32)
    negative_count = np.zeros((grid_h, grid_w), dtype=np.int32)
    other_count = np.zeros((grid_h, grid_w), dtype=np.int32)

    for obj in objects:
        inside = _box_token_mask(obj, width, height, grid_x, grid_y)
        if area_range is not None and not area_range[0] <= box_area(obj, width, height) <= area_range[1]:
            other_count += inside
        elif predicate(obj):
            positive_count += inside
        else:
            negative_count += inside
    for obj in others:
        other_count += _box_token_mask(obj, width, height, grid_x, grid_y)

    total = positive_count + negative_count + other_count
    positive = positive_count > 0
    negative = (negative_count > 0) & ~positive
    return {
        "positive": positive,
        "negative": negative,
        "background": total == 0,
        "positive_solo": (total == 1) & (positive_count == 1),
        "negative_solo": (total == 1) & (negative_count == 1),
        "box_count": total,
    }


def token_object_masks(
    objects: Sequence[VisDroneObject],
    width: int,
    height: int,
    grid_h: int,
    grid_w: int,
    *,
    occluded_levels: tuple[int, ...] = (1, 2),
    area_range: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Build occluded/clear per-token masks aligned to a MoT feature-map grid.

    A thin naming wrapper over :func:`token_group_masks` with the occlusion
    predicate; see there for the token-assignment and ``*_solo`` semantics.

    Args:
        objects (Sequence[VisDroneObject]): Valid objects for the image.
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        grid_h (int): Feature-map height (token rows).
        grid_w (int): Feature-map width (token columns).
        occluded_levels (tuple[int, ...]): Occlusion codes treated as occluded.
        area_range (tuple[float, float] | None): Optional inclusive normalized-area
            band for the size-matched variant of the token test.

    Returns:
        (dict[str, np.ndarray]): Masks named by :data:`TOKEN_GROUPS` plus the integer
            ``box_count`` map, each shaped ``(grid_h, grid_w)``.

    Examples:
        >>> objects = [VisDroneObject(0, 0, 50, 100, 1, 0, 2), VisDroneObject(50, 0, 50, 100, 1, 0, 0)]
        >>> masks = token_object_masks(objects, 100, 100, 1, 2)
        >>> masks["occluded"].tolist(), masks["clear"].tolist()
        ([[True, False]], [[False, True]])
        >>> masks["occluded_solo"].tolist(), masks["background"].tolist()
        ([[True, False]], [[False, False]])
        >>> masks["box_count"].tolist()
        [[1, 1]]
    """
    masks = token_group_masks(
        objects,
        width,
        height,
        grid_h,
        grid_w,
        lambda obj: obj.occlusion in occluded_levels,
        area_range=area_range,
    )
    return {
        "occluded": masks["positive"],
        "clear": masks["negative"],
        "background": masks["background"],
        "occluded_solo": masks["positive_solo"],
        "clear_solo": masks["negative_solo"],
        "box_count": masks["box_count"],
    }


def token_local_density(
    objects: Sequence[VisDroneObject],
    width: int,
    height: int,
    grid_h: int,
    grid_w: int,
    *,
    radius: int = 1,
) -> np.ndarray:
    """Count object centres in each token's ``(2 * radius + 1)`` neighbourhood.

    This turns "dense vs sparse" into a *within-image* quantity, so the contrast can
    be run on token pairs drawn from the same frame instead of only across frames.

    Args:
        objects (Sequence[VisDroneObject]): Valid objects for the image.
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        grid_h (int): Feature-map height (token rows).
        grid_w (int): Feature-map width (token columns).
        radius (int): Neighbourhood half-width in tokens.

    Returns:
        (np.ndarray): Integer ``(grid_h, grid_w)`` neighbour-count map.

    Examples:
        >>> left = [VisDroneObject(x, 0, 4, 4, 1, 0, 0) for x in (0, 10, 20)]
        >>> density = token_local_density(left, 100, 10, 1, 5, radius=0)
        >>> density.tolist()
        [[2, 1, 0, 0, 0]]
        >>> token_local_density(left, 100, 10, 1, 5, radius=1).tolist()
        [[3, 3, 1, 0, 0]]
    """
    counts = np.zeros((grid_h, grid_w), dtype=np.int32)
    for obj in objects:
        x0, y0, x1, y1 = obj.normalized_box(width, height)
        column = min(int(0.5 * (x0 + x1) * grid_w), grid_w - 1)
        row = min(int(0.5 * (y0 + y1) * grid_h), grid_h - 1)
        counts[max(row, 0), max(column, 0)] += 1
    if radius <= 0:
        return counts
    padded = np.pad(counts, radius, mode="constant")
    window = 2 * radius + 1
    cumulative = padded.cumsum(axis=0).cumsum(axis=1)
    cumulative = np.pad(cumulative, ((1, 0), (1, 0)), mode="constant")
    return (
        cumulative[window:, window:]
        - cumulative[:-window, window:]
        - cumulative[window:, :-window]
        + cumulative[:-window, :-window]
    ).astype(np.int32)


def local_density_masks(
    density: np.ndarray, object_mask: np.ndarray, *, low: float = 0.3, high: float = 0.7
) -> dict[str, np.ndarray]:
    """Split object tokens into locally crowded and locally isolated arms.

    Cut points are per-image quantiles of the density over object tokens, so the
    split adapts to frames that are globally dense or globally empty. Heavy ties
    (many tokens sharing one neighbour count) can leave an arm empty; the caller is
    expected to drop such images from the pairing.

    Args:
        density (np.ndarray): Neighbour counts from :func:`token_local_density`.
        object_mask (np.ndarray): Tokens covered by at least one box.
        low (float): Quantile at or below which a token counts as isolated.
        high (float): Quantile at or above which a token counts as crowded.

    Returns:
        (dict[str, np.ndarray]): ``crowded`` and ``isolated`` boolean masks.

    Examples:
        >>> density = np.array([[1, 1, 4, 4]])
        >>> masks = local_density_masks(density, np.array([[True, True, True, True]]))
        >>> masks["crowded"].tolist(), masks["isolated"].tolist()
        ([[False, False, True, True]], [[True, True, False, False]])

        Tokens outside ``object_mask`` never join either arm, and a frame whose object
        tokens all share one density leaves both arms empty:

        >>> masks = local_density_masks(density, np.array([[True, True, True, False]]))
        >>> masks["crowded"].tolist()
        [[False, False, True, False]]
        >>> masks = local_density_masks(np.array([[3, 3]]), np.array([[True, True]]))
        >>> masks["crowded"].any(), masks["isolated"].any()
        (np.False_, np.False_)
    """
    empty = np.zeros_like(object_mask, dtype=bool)
    if not object_mask.any():
        return {"crowded": empty, "isolated": empty.copy()}
    values = np.asarray(density, dtype=np.float64)[object_mask]
    low_cut, high_cut = float(np.quantile(values, low)), float(np.quantile(values, high))
    if low_cut >= high_cut:
        return {"crowded": empty, "isolated": empty.copy()}
    return {
        "crowded": object_mask & (density >= high_cut),
        "isolated": object_mask & (density <= low_cut),
    }


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """Mean of ``values`` over ``mask``, or NaN when the mask selects nothing.

    Args:
        values (np.ndarray): Value map.
        mask (np.ndarray): Boolean mask broadcastable to ``values``.

    Returns:
        (float): Masked mean, or ``float('nan')`` for an empty mask.

    Examples:
        >>> masked_mean(np.array([[1.0, 3.0]]), np.array([[True, False]]))
        1.0
        >>> math.isnan(masked_mean(np.array([[1.0]]), np.array([[False]])))
        True
    """
    if not mask.any():
        return float("nan")
    return float(np.asarray(values, dtype=np.float64)[mask].mean())


def usage_concentration(usage: Sequence[float]) -> dict[str, float]:
    """Return normalized Gini and entropy for an expert-usage vector.

    Args:
        usage (Sequence[float]): Non-negative per-expert usage shares.

    Returns:
        (dict[str, float]): ``gini`` (0 = uniform, 1 = one expert) and ``entropy``
            (1 = uniform, 0 = collapsed).

    Examples:
        >>> concentration = usage_concentration([1 / 3, 1 / 3, 1 / 3])
        >>> round(concentration["gini"], 6), round(concentration["entropy"], 6)
        (0.0, 1.0)
        >>> collapsed = usage_concentration([1.0, 0.0, 0.0])
        >>> round(collapsed["gini"], 6), round(collapsed["entropy"], 6)
        (1.0, 0.0)
    """
    values = np.asarray(usage, dtype=np.float64).clip(min=0.0)
    count = values.size
    total = float(values.sum())
    if count <= 1 or total <= 0:
        return {"gini": 0.0, "entropy": 0.0}
    shares = np.sort(values / total)
    index = np.arange(1, count + 1, dtype=np.float64)
    gini = (2.0 * float((index * shares).sum()) / count) - (count + 1.0) / count
    entropy = -float((shares * np.log(np.clip(shares, 1e-12, None))).sum()) / math.log(count)
    return {
        # ``+ 0.0`` normalizes the -0.0 that a fully collapsed vector produces.
        "gini": float(np.clip(gini * count / (count - 1), 0.0, 1.0) + 0.0),
        "entropy": float(np.clip(entropy, 0.0, 1.0) + 0.0),
    }


@dataclass(frozen=True)
class PairedDifferenceReport:
    """Result of a paired within-image comparison of two token groups."""

    n_pairs: int
    mean_a: float
    mean_b: float
    mean_difference: float
    hodges_lehmann: float
    superiority: float
    wilcoxon_statistic: float
    p_value: float
    relative_difference: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable mapping of the report fields."""
        return asdict(self)


def hodges_lehmann(differences: Sequence[float]) -> float:
    """Hodges-Lehmann estimator: median of all pairwise Walsh averages.

    This is the location estimate that matches the Wilcoxon signed-rank test, so
    the reported effect size and p-value describe the same quantity.

    Args:
        differences (Sequence[float]): Paired differences.

    Returns:
        (float): Estimated median shift, or NaN when empty.

    Examples:
        >>> hodges_lehmann([1.0, 2.0, 3.0])
        2.0
        >>> math.isnan(hodges_lehmann([]))
        True
    """
    values = np.asarray([value for value in differences if np.isfinite(value)], dtype=np.float64)
    if values.size == 0:
        return float("nan")
    walsh = (values[:, None] + values[None, :]) / 2.0
    return float(np.median(walsh[np.triu_indices(values.size)]))


def paired_difference_report(
    group_a: Sequence[float],
    group_b: Sequence[float],
    *,
    alternative: str = "greater",
) -> PairedDifferenceReport:
    """Compare two paired per-image measurements.

    Pairs where either side is NaN (an empty token mask for that image) are
    dropped, so ``n_pairs`` reports the usable sample size.

    Args:
        group_a (Sequence[float]): Per-image values for the hypothesised-higher group.
        group_b (Sequence[float]): Per-image values for the baseline group.
        alternative (str): Wilcoxon alternative, ``'greater'``, ``'less'`` or ``'two-sided'``.

    Returns:
        (PairedDifferenceReport): Effect sizes and the signed-rank p-value.

    Examples:
        >>> report = paired_difference_report([0.5, 0.6, 0.7, 0.8], [0.4, 0.5, 0.55, 0.7])
        >>> report.n_pairs, round(report.mean_difference, 4)
        (4, 0.1125)
        >>> report.p_value < 0.1
        True
        >>> report.superiority
        1.0
    """
    from scipy import stats

    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    if a.size != b.size:
        raise ValueError(f"paired groups must have equal length, got {a.size} and {b.size}")
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    differences = a - b
    if differences.size == 0:
        nan = float("nan")
        return PairedDifferenceReport(0, nan, nan, nan, nan, nan, nan, nan, nan)

    non_zero = differences[differences != 0]
    if non_zero.size == 0:
        statistic, p_value = 0.0, 1.0
    else:
        result = stats.wilcoxon(differences, alternative=alternative, zero_method="wilcox")
        statistic, p_value = float(result.statistic), float(result.pvalue)

    mean_b = float(b.mean())
    return PairedDifferenceReport(
        n_pairs=int(differences.size),
        mean_a=float(a.mean()),
        mean_b=mean_b,
        mean_difference=float(differences.mean()),
        hodges_lehmann=hodges_lehmann(differences),
        superiority=float((differences > 0).mean()),
        wilcoxon_statistic=statistic,
        p_value=p_value,
        relative_difference=float(differences.mean() / mean_b) if mean_b != 0 else float("nan"),
    )


@dataclass(frozen=True)
class StratifiedTestReport:
    """Covariate-controlled group comparison across matched strata."""

    n_strata: int
    n_group: int
    n_baseline: int
    raw_difference: float
    stratified_difference: float
    p_value: float
    relative_difference: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable mapping of the report fields."""
        return asdict(self)


def quantile_contrast_split(
    values: Sequence[float], *, low: float = 0.3, high: float = 0.7
) -> tuple[np.ndarray, np.ndarray]:
    """Split a sample into disjoint high and low arms by quantile.

    A plain ``>= q_high`` / ``<= q_low`` rule silently breaks on tie-heavy features:
    VisDrone's ``irregular_rate`` is 0 for most images, so both cut points land on 0
    and every image ends up in *both* arms. When that happens the cut falls back to
    the smallest distinct value above the low quantile, which always yields two
    non-empty, disjoint arms as long as the sample has two distinct values.

    Args:
        values (Sequence[float]): Feature to split on.
        low (float): Lower quantile.
        high (float): Upper quantile.

    Returns:
        (tuple[np.ndarray, np.ndarray]): ``(high_mask, low_mask)`` boolean arrays.

    Examples:
        >>> high, low = quantile_contrast_split([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> high.tolist(), low.tolist()
        ([False, False, False, True, True], [True, True, False, False, False])

        Tie-heavy features do not collapse into overlapping arms:

        >>> high, low = quantile_contrast_split([0.0, 0.0, 0.0, 0.0, 0.5, 0.9])
        >>> high.tolist(), low.tolist()
        ([False, False, False, False, True, True], [True, True, True, True, False, False])
        >>> high, low = quantile_contrast_split([2.0, 2.0, 2.0])
        >>> bool(high.any()), bool(low.any())
        (False, False)
    """
    array = np.asarray(values, dtype=np.float64)
    low_cut = float(np.quantile(array, low))
    high_cut = float(np.quantile(array, high))
    if low_cut >= high_cut:
        distinct = np.unique(array[np.isfinite(array)])
        above = distinct[distinct > low_cut]
        if distinct.size < 2:
            empty = np.zeros(array.size, dtype=bool)
            return empty, empty.copy()
        high_cut = float(above[0]) if above.size else float(distinct[-1])
        low_cut = float(distinct[distinct < high_cut][-1])
    return array >= high_cut, array <= low_cut


def quantile_strata(values: Sequence[float], n_bins: int) -> np.ndarray:
    """Assign values to ``n_bins`` quantile bins.

    Args:
        values (Sequence[float]): Covariate values.
        n_bins (int): Number of quantile bins.

    Returns:
        (np.ndarray): Integer bin index per value.

    Examples:
        >>> quantile_strata([1.0, 2.0, 3.0, 4.0], 2).tolist()
        [0, 0, 1, 1]
    """
    array = np.asarray(values, dtype=np.float64)
    if n_bins <= 1 or array.size == 0:
        return np.zeros(array.size, dtype=np.int64)
    edges = np.quantile(array, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return np.searchsorted(edges, array, side="left").astype(np.int64)


def _as_extreme(candidate: float, observed: float, alternative: str) -> bool:
    """Return whether a permuted statistic is at least as extreme as the observed one.

    Args:
        candidate (float): Statistic from one permutation.
        observed (float): Statistic from the true labelling.
        alternative (str): ``'greater'``, ``'less'`` or ``'two-sided'``.

    Returns:
        (bool): True when the permutation counts toward the p-value numerator.

    Examples:
        >>> _as_extreme(0.3, 0.2, "greater"), _as_extreme(0.1, 0.2, "greater")
        (True, False)
        >>> _as_extreme(-0.5, 0.2, "two-sided")
        True
    """
    if alternative == "greater":
        return candidate >= observed
    if alternative == "less":
        return candidate <= observed
    if alternative == "two-sided":
        return abs(candidate) >= abs(observed)
    raise ValueError(f"unsupported alternative {alternative!r}")


def stratified_permutation_test(
    values: Sequence[float],
    in_group: Sequence[bool],
    strata: Sequence[int],
    *,
    n_permutations: int = 10000,
    seed: int = 0,
    alternative: str = "greater",
) -> StratifiedTestReport:
    """Compare two groups while holding stratifying covariates fixed.

    The statistic is a precision-weighted mean of within-stratum differences
    (weight ``n_a * n_b / (n_a + n_b)``, the usual Cochran-Mantel-Haenszel form),
    and the null distribution comes from shuffling group labels *within* each
    stratum. Strata missing either group contribute nothing.

    Args:
        values (Sequence[float]): Per-image measurement.
        in_group (Sequence[bool]): True for the hypothesised-higher group.
        strata (Sequence[int]): Stratum index per image, e.g. from :func:`quantile_strata`.
        n_permutations (int): Permutation count for the p-value.
        seed (int): RNG seed.
        alternative (str): ``'greater'``, ``'less'`` or ``'two-sided'``.

    Returns:
        (StratifiedTestReport): Raw and stratified differences with a permutation p-value.

    Examples:
        >>> values = [0.2, 0.3, 0.6, 0.7, 0.25, 0.35, 0.65, 0.75]
        >>> group = [False, False, False, False, True, True, True, True]
        >>> strata = [0, 0, 1, 1, 0, 0, 1, 1]
        >>> report = stratified_permutation_test(values, group, strata, n_permutations=200, seed=0)
        >>> report.n_strata, round(report.stratified_difference, 4)
        (2, 0.05)
        >>> report.p_value < 0.2
        True
    """
    array = np.asarray(values, dtype=np.float64)
    group = np.asarray(in_group, dtype=bool)
    stratum = np.asarray(strata, dtype=np.int64)
    if not (array.size == group.size == stratum.size):
        raise ValueError("values, in_group and strata must have equal length")
    valid = np.isfinite(array)
    array, group, stratum = array[valid], group[valid], stratum[valid]

    usable = [key for key in np.unique(stratum) if group[stratum == key].any() and (~group[stratum == key]).any()]
    nan = float("nan")
    if not usable:
        return StratifiedTestReport(0, int(group.sum()), int((~group).sum()), nan, nan, nan, nan)

    def statistic(labels: np.ndarray) -> float:
        total_weight = 0.0
        weighted = 0.0
        for key in usable:
            selection = stratum == key
            a = array[selection & labels]
            b = array[selection & ~labels]
            weight = a.size * b.size / (a.size + b.size)
            weighted += weight * (a.mean() - b.mean())
            total_weight += weight
        return weighted / total_weight

    observed = statistic(group)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_permutations):
        shuffled = group.copy()
        for key in usable:
            selection = np.flatnonzero(stratum == key)
            shuffled[selection] = rng.permutation(group[selection])
        candidate = statistic(shuffled)
        hits += int(_as_extreme(candidate, observed, alternative))

    baseline_mean = float(array[~group].mean())
    return StratifiedTestReport(
        n_strata=len(usable),
        n_group=int(group.sum()),
        n_baseline=int((~group).sum()),
        raw_difference=float(array[group].mean() - baseline_mean),
        stratified_difference=float(observed),
        p_value=float((hits + 1) / (n_permutations + 1)),
        relative_difference=float(observed / baseline_mean) if baseline_mean != 0 else nan,
    )


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> dict[str, list]:
    """Benjamini-Hochberg FDR control over a family of p-values.

    Args:
        p_values (Sequence[float]): Raw p-values; NaN entries never reject.
        alpha (float): Target false discovery rate.

    Returns:
        (dict[str, list]): ``q_values`` (adjusted p-values) and ``rejected`` flags.

    Examples:
        >>> outcome = benjamini_hochberg([0.001, 0.02, 0.4, 0.9])
        >>> outcome["rejected"]
        [True, True, False, False]
        >>> round(outcome["q_values"][1], 4)
        0.04
    """
    raw = np.asarray(p_values, dtype=np.float64)
    finite = np.isfinite(raw)
    q_values = np.full(raw.size, float("nan"))
    rejected = np.zeros(raw.size, dtype=bool)
    if not finite.any():
        return {"q_values": q_values.tolist(), "rejected": rejected.tolist()}

    index = np.flatnonzero(finite)
    order = index[np.argsort(raw[index], kind="stable")]
    count = order.size
    adjusted = raw[order] * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(max=1.0)
    q_values[order] = adjusted
    rejected[order] = adjusted <= alpha
    return {"q_values": q_values.tolist(), "rejected": rejected.tolist()}


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment between two representations.

    CKA is invariant to isotropic scaling and orthogonal transforms, so it
    measures whether two experts span the same feature subspace rather than
    whether they happen to have the same magnitude.

    Args:
        x (np.ndarray): Matrix of shape ``(n_samples, n_features_x)``.
        y (np.ndarray): Matrix of shape ``(n_samples, n_features_y)``.

    Returns:
        (float): Similarity in ``[0, 1]``; 1 means identical subspaces.

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> a = rng.normal(size=(64, 8))
        >>> round(linear_cka(a, a), 6)
        1.0
        >>> round(linear_cka(a, -3.0 * a), 6)
        1.0
        >>> linear_cka(a, rng.normal(size=(64, 8))) < 0.5
        True
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"CKA needs matching sample counts, got {a.shape[0]} and {b.shape[0]}")
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    cross = float(np.linalg.norm(b.T @ a, ord="fro") ** 2)
    scale = float(np.linalg.norm(a.T @ a, ord="fro") * np.linalg.norm(b.T @ b, ord="fro"))
    if scale <= 0:
        return float("nan")
    return float(np.clip(cross / scale, 0.0, 1.0))


def pairwise_cosine(vectors: np.ndarray) -> dict[tuple[int, int], float]:
    """Cosine similarity for every ordered pair ``i < j`` of row vectors.

    Args:
        vectors (np.ndarray): Matrix of shape ``(n_vectors, n_features)``.

    Returns:
        (dict[tuple[int, int], float]): Similarity keyed by index pair.

    Examples:
        >>> pairwise_cosine(np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]]))
        {(0, 1): 0.0, (0, 2): 1.0, (1, 2): 0.0}
    """
    matrix = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    result: dict[tuple[int, int], float] = {}
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[0]):
            scale = norms[i] * norms[j]
            result[(i, j)] = float(matrix[i] @ matrix[j] / scale) if scale > 0 else float("nan")
    return result


@dataclass(frozen=True)
class DeltaSimilarityReport:
    """How much each expert changes its input, and how alike those changes are."""

    layer_name: str
    relative_magnitude: tuple[float, ...]
    output_cosine: dict[str, float]
    delta_cosine: dict[str, float]
    delta_cka: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping of the report fields."""
        return asdict(self)


def delta_similarity_report(
    layer_name: str,
    inputs: np.ndarray,
    outputs: Sequence[np.ndarray],
    *,
    names: Sequence[str] = EXPERT_NAMES,
) -> DeltaSimilarityReport:
    """Measure expert diversity after removing the shared residual path.

    ``output_cosine`` is kept only to show why it is the wrong metric: every MoT
    expert returns ``x + small_delta``, so output cosine is near 1.0 by
    construction. ``delta_cosine`` and ``delta_cka`` compare the learned
    ``expert(x) - x`` contributions, which is what "did the experts specialise?"
    actually asks.

    Args:
        layer_name (str): MoT block name, for labelling.
        inputs (np.ndarray): Block input of shape ``(tokens, channels)``.
        outputs (Sequence[np.ndarray]): One ``(tokens, channels)`` output per expert.
        names (Sequence[str]): Expert names used to key the pair dictionaries.

    Returns:
        (DeltaSimilarityReport): Per-expert magnitudes and pairwise similarities.

    Examples:
        Two experts whose learned contributions are orthogonal, each scaled to 1% of
        the input, look nearly identical on raw outputs but are correctly separated
        once the shared residual is removed:

        >>> rng = np.random.default_rng(0)
        >>> x = 10.0 + rng.normal(size=(64, 8))
        >>> first, second = rng.normal(size=(64, 8)), rng.normal(size=(64, 8))
        >>> outs = [x + 0.01 * first, x + 0.01 * second]
        >>> report = delta_similarity_report("mot.0", x, outs, names=("a", "b"))
        >>> round(report.output_cosine["a|b"], 3)  # residual path dominates
        1.0
        >>> abs(report.delta_cosine["a|b"]) < 0.1  # the contributions are unrelated
        True
        >>> report.delta_cka["a|b"] < 0.2
        True
        >>> all(0.0001 < value < 0.01 for value in report.relative_magnitude)
        True
    """
    base = np.asarray(inputs, dtype=np.float64)
    expert_outputs = [np.asarray(item, dtype=np.float64) for item in outputs]
    deltas = [item - base for item in expert_outputs]
    base_norm = float(np.linalg.norm(base))

    output_cosine = pairwise_cosine(np.stack([item.reshape(-1) for item in expert_outputs]))
    delta_cosine = pairwise_cosine(np.stack([item.reshape(-1) for item in deltas]))
    keyed_output = {f"{names[i]}|{names[j]}": value for (i, j), value in output_cosine.items()}
    keyed_delta = {f"{names[i]}|{names[j]}": value for (i, j), value in delta_cosine.items()}
    keyed_cka = {
        f"{names[i]}|{names[j]}": linear_cka(deltas[i], deltas[j])
        for i in range(len(deltas))
        for j in range(i + 1, len(deltas))
    }
    return DeltaSimilarityReport(
        layer_name=layer_name,
        relative_magnitude=tuple(
            float(np.linalg.norm(item) / base_norm) if base_norm > 0 else float("nan") for item in deltas
        ),
        output_cosine=keyed_output,
        delta_cosine=keyed_delta,
        delta_cka=keyed_cka,
    )


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    """Spearman rank correlation with its two-sided p-value.

    Args:
        x (Sequence[float]): First variable.
        y (Sequence[float]): Second variable.

    A constant input (a fully collapsed router layer routes every image identically)
    has no defined correlation, so NaN is returned rather than letting SciPy warn.

    Returns:
        (dict[str, float]): ``rho`` and ``p_value``.

    Examples:
        >>> outcome = spearman_correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        >>> round(outcome["rho"], 6)
        1.0
        >>> import math
        >>> math.isnan(spearman_correlation([1, 2, 3], [1.0, 1.0, 1.0])["rho"])
        True
    """
    from scipy import stats

    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return {"rho": float("nan"), "p_value": float("nan")}
    result = stats.spearmanr(a, b)
    return {"rho": float(result.statistic), "p_value": float(result.pvalue)}


def format_expert_share_table(shares: dict[str, Sequence[float]], names: Sequence[str] = EXPERT_NAMES) -> str:
    """Render a scene-by-expert share table as GitHub-flavoured Markdown.

    Args:
        shares (dict[str, Sequence[float]]): Per-scene share vectors.
        names (Sequence[str]): Expert column names.

    Returns:
        (str): Markdown table.

    Examples:
        >>> print(format_expert_share_table({"dense": [0.5, 0.2, 0.3]}, names=("A", "B", "C")))
        | Scene | A | B | C |
        | --- | ---: | ---: | ---: |
        | dense | 0.500 | 0.200 | 0.300 |
    """
    header = "| Scene | " + " | ".join(names) + " |"
    divider = "| --- | " + " | ".join("---:" for _ in names) + " |"
    rows = [
        "| " + scene + " | " + " | ".join(f"{value:.3f}" for value in values) + " |" for scene, values in shares.items()
    ]
    return "\n".join([header, divider, *rows])


def iter_finite(values: Iterable[float]) -> list[float]:
    """Drop non-finite entries from an iterable of floats.

    Args:
        values (Iterable[float]): Input values.

    Returns:
        (list[float]): Only the finite values.

    Examples:
        >>> iter_finite([1.0, float("nan"), 2.0, float("inf")])
        [1.0, 2.0]
    """
    return [float(value) for value in values if np.isfinite(value)]


__all__ = [
    "ASPECT_IRREGULAR_THRESHOLD",
    "EXPERT_NAMES",
    "TOKEN_GROUPS",
    "DeltaSimilarityReport",
    "ImageSceneFeatures",
    "PairedDifferenceReport",
    "StratifiedTestReport",
    "VisDroneObject",
    "benjamini_hochberg",
    "box_area",
    "delta_similarity_report",
    "format_expert_share_table",
    "hodges_lehmann",
    "image_scene_features",
    "is_irregular_box",
    "iter_finite",
    "linear_cka",
    "local_density_masks",
    "masked_mean",
    "paired_difference_report",
    "pairwise_cosine",
    "parse_visdrone_annotation",
    "quantile_contrast_split",
    "quantile_strata",
    "spearman_correlation",
    "stratified_permutation_test",
    "token_group_masks",
    "token_local_density",
    "token_object_masks",
    "usage_concentration",
]
