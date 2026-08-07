#!/usr/bin/env python3
"""Run the MoT routing-interpretability suite on a trained YOLO-Master checkpoint.

The suite answers two questions and refuses to conflate them:

1. **Does the router behave differently across scenes?** Four scene axes are tested
   at equal rigour — occlusion, density (dense vs sparse), object scale (small vs
   large) and box shape (irregular vs regular). On VisDrone these four co-vary, so a
   raw group split proves nothing; every claim therefore gets a covariate-controlled
   image-level test (stratified permutation, the other axes held fixed) *and* a
   within-image paired token test, which is the only comparison that isolates one
   axis from the rest. The occlusion test is one-sided because its hypothesis was
   directional; the other three are two-sided.
2. **Do the experts compute different things?** Measured on the residual delta
   ``expert(x) - x`` (cosine + linear CKA), and causally by re-validating mAP with
   the router forced to a single expert or with its content-dependence destroyed.
   Cosine on raw expert *outputs* is reported too, purely to show it is
   uninformative: every expert is ``x + small_delta``, so that number is ~1.0 by
   construction.

Example:
    python scripts/run_mot_routing_interpret.py \\
        --checkpoint /root/autodl-tmp/runs/visdrone_mot_ablation/v10_mot/weights/best.pt \\
        --data /root/autodl-tmp/datasets/VisDrone/VisDrone.yaml \\
        --output /root/autodl-tmp/mot_routing_interpret/visdrone
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from typing_extensions import Self

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yolo_master_matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import yaml

from scripts.mot_routing_interpret import (
    EXPERT_NAMES,
    VisDroneObject,
    benjamini_hochberg,
    box_area,
    coco_objects_by_image,
    delta_similarity_report,
    image_scene_features,
    is_irregular_box,
    local_density_masks,
    paired_difference_report,
    parse_visdrone_annotation,
    quantile_contrast_split,
    quantile_strata,
    spearman_correlation,
    stratified_permutation_test,
    token_group_masks,
    token_local_density,
    token_object_masks,
    usage_concentration,
)

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
POOLED = "__pooled__"


@dataclass(frozen=True)
class TokenPair:
    """One within-image paired comparison between two token groups."""

    contrast: str
    group_a: str
    group_b: str
    description: str


TOKEN_PAIR_TESTS = (
    TokenPair(
        "occluded_vs_clear",
        "occluded_solo",
        "clear_solo",
        "within-image, single-box tokens (density-controlled)",
    ),
    TokenPair("occluded_vs_clear", "occluded_solo_sized", "clear_solo_sized", "single-box tokens, size-matched boxes"),
    TokenPair("occluded_vs_clear", "occluded", "clear", "all object tokens (density NOT controlled)"),
    TokenPair("occluded_vs_clear", "occluded", "background", "occluded objects vs empty background"),
)

SCENE_TOKEN_PAIRS = (
    TokenPair(
        "dense_vs_sparse",
        "crowded_solo",
        "isolated_solo",
        "within-image, single-box tokens in locally crowded vs locally isolated neighbourhoods",
    ),
    TokenPair(
        "small_vs_large",
        "small_solo",
        "large_solo",
        "within-image, single-box tokens on bottom-tercile vs top-tercile boxes",
    ),
    TokenPair(
        "irregular_vs_regular",
        "irregular_solo",
        "regular_solo",
        "within-image, single-box tokens on elongated vs compact boxes",
    ),
    TokenPair(
        "irregular_vs_regular",
        "irregular_solo_sized",
        "regular_solo_sized",
        "elongated vs compact boxes, size-matched",
    ),
)


@dataclass(frozen=True)
class SceneContrast:
    """An image-level scene split and the covariates that must be held fixed.

    Args:
        name (str): Contrast identifier used in the artifacts.
        feature (str): :class:`~scripts.mot_routing_interpret.ImageSceneFeatures` field
            the split is taken on.
        group_is_high (bool): Whether the named group is the *upper* tail of the
            feature (``dense`` is, ``small_objects`` is not).
        covariates (tuple[str, str]): Features stratified so the contrast cannot be
            explained by them.
    """

    name: str
    feature: str
    group_is_high: bool
    covariates: tuple[str, str]


SCENE_CONTRASTS = (
    SceneContrast("dense_vs_sparse", "n_objects", True, ("median_area", "occlusion_rate")),
    SceneContrast("small_vs_large", "median_area", False, ("n_objects", "occlusion_rate")),
    SceneContrast("irregular_vs_regular", "irregular_rate", True, ("n_objects", "median_area")),
)
"""Scene contrasts run at the same rigour as the occlusion test.

Each one stratifies away the *other two* scene axes, because in VisDrone crowding,
object scale, occlusion and box elongation all co-vary: dense frames are filmed from
higher altitude and therefore hold smaller, more occluded objects, and elongated
boxes are disproportionately vehicles seen along the road axis.

Unlike the occlusion test, these are two-sided. Only the Deformable-under-occlusion
claim was a directional pre-registered hypothesis; "does routing differ between
dense and sparse scenes" has no a-priori sign, and testing it one-sided would double
the apparent significance of whichever direction the data happened to take.
"""


def resolve_device(spec: str) -> torch.device:
    """Map a CLI device string onto an available torch device."""
    text = str(spec).strip().lower()
    if text in {"cpu", ""}:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        print(f"[interpret] CUDA unavailable; falling back to CPU (requested {spec!r})")
        return torch.device("cpu")
    return torch.device(f"cuda:{text}" if text.isdigit() else text)


def nan_mean(values: Iterable[float]) -> float:
    """Mean of the finite entries, or NaN when none are finite.

    Used instead of :func:`numpy.nanmean` so an all-NaN layer stack (a token group
    that is empty in every layer of one image) yields NaN quietly rather than
    emitting a RuntimeWarning per call.

    Args:
        values (Iterable[float]): Values that may contain NaN.

    Returns:
        (float): Mean over finite entries, or NaN.

    Examples:
        >>> nan_mean([1.0, float("nan"), 3.0])
        2.0
        >>> import math
        >>> math.isnan(nan_mean([float("nan")]))
        True
    """
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


@dataclass
class ImageRecord:
    """One analysed image: its scene descriptors and per-layer routing means."""

    stem: str
    image_path: Path
    features: dict[str, float]
    layer_expert_weight: dict[str, np.ndarray]
    layer_expert_top1: dict[str, np.ndarray]
    token_group_weight: dict[str, dict[str, np.ndarray]]
    token_group_sizes: dict[str, int]


def find_visdrone_root(data_yaml: Path, split: str) -> tuple[Path, Path]:
    """Return the ``(images, annotations)`` directories for a VisDrone split."""
    with data_yaml.open() as handle:
        config = yaml.safe_load(handle)
    base = Path(config.get("path", data_yaml.parent))
    images = base / config[split]
    annotations = images.parent / "annotations"
    if not images.is_dir():
        raise FileNotFoundError(f"images directory not found: {images}")
    if not annotations.is_dir():
        raise FileNotFoundError(
            f"VisDrone annotations directory not found: {annotations}. "
            "The occlusion analysis needs the raw VisDrone annotations, not YOLO labels."
        )
    return images, annotations


def find_coco_split(data_yaml: Path, split: str) -> tuple[Path, Path]:
    """Return the ``(images, instances_json)`` paths for a COCO split."""
    with data_yaml.open() as handle:
        config = yaml.safe_load(handle)
    base = Path(config.get("path", data_yaml.parent))
    images = base / config[split]
    if not images.is_dir():
        raise FileNotFoundError(f"images directory not found: {images}")
    instances = base / "annotations" / f"instances_{images.name}.json"
    if not instances.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {instances}")
    return images, instances


def load_visdrone_objects(annotation_path: Path) -> list[VisDroneObject]:
    """Read one VisDrone annotation file, returning an empty list when missing."""
    if not annotation_path.exists():
        return []
    return parse_visdrone_annotation(annotation_path.read_text())


def matched_area_band(
    per_image_objects: Sequence[tuple[Sequence[VisDroneObject], int, int]],
    predicate: Callable[[VisDroneObject], bool],
    *,
    low: float = 0.25,
    high: float = 0.75,
) -> tuple[float, float]:
    """Find the normalized-area band shared by the two arms of a box-level contrast.

    Restricting a token test to this band removes the object-size confound: any
    remaining routing difference cannot be explained by one arm's boxes simply being
    smaller. Used for the occluded-vs-clear and irregular-vs-regular token tests.

    Args:
        per_image_objects (Sequence[tuple]): ``(objects, width, height)`` per image.
        predicate (Callable[[VisDroneObject], bool]): True for the positive arm.
        low (float): Lower quantile of each distribution.
        high (float): Upper quantile of each distribution.

    Returns:
        (tuple[float, float]): Inclusive ``(min_area, max_area)`` overlap band, or the
            full ``(0.0, 1.0)`` range when either arm is empty.
    """
    positive: list[float] = []
    negative: list[float] = []
    for objects, width, height in per_image_objects:
        for obj in objects:
            (positive if predicate(obj) else negative).append(box_area(obj, width, height))
    if not positive or not negative:
        return (0.0, 1.0)
    return (
        max(float(np.quantile(positive, low)), float(np.quantile(negative, low))),
        min(float(np.quantile(positive, high)), float(np.quantile(negative, high))),
    )


def area_terciles(
    per_image_objects: Sequence[tuple[Sequence[VisDroneObject], int, int]],
) -> tuple[float, float]:
    """Return the pooled normalized-area terciles that define small and large boxes.

    Splitting on dataset-wide terciles rather than per-image ones keeps "small" the
    same physical size in every frame, and dropping the middle tercile from the
    contrast keeps the two arms clearly separated instead of straddling one cut point.

    Args:
        per_image_objects (Sequence[tuple]): ``(objects, width, height)`` per image.

    Returns:
        (tuple[float, float]): ``(small_max, large_min)`` normalized areas.
    """
    areas = [box_area(obj, width, height) for objects, width, height in per_image_objects for obj in objects]
    if not areas:
        return (0.0, 1.0)
    return float(np.quantile(areas, 1 / 3)), float(np.quantile(areas, 2 / 3))


class MoTRoutingProbe:
    """Capture router weights (and optionally block inputs) from every MoTBlock."""

    def __init__(self, model: torch.nn.Module) -> None:
        from ultralytics.nn.modules.mot import MoTBlock

        self.blocks: dict[str, MoTBlock] = {
            name: module for name, module in model.named_modules() if isinstance(module, MoTBlock)
        }
        if not self.blocks:
            raise SystemExit("no MoTBlock modules found in the checkpoint")
        self.weights: dict[str, torch.Tensor] = {}
        self.inputs: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []
        self._capture_inputs = False

    @property
    def layer_names(self) -> list[str]:
        """MoT block names in model order."""
        return list(self.blocks)

    def __enter__(self) -> Self:
        for name, block in self.blocks.items():
            self._handles.append(block.router.register_forward_hook(self._router_hook(name)))
            self._handles.append(block.register_forward_pre_hook(self._input_hook(name)))
        return self

    def __exit__(self, *exception: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def capture_inputs(self, enabled: bool) -> None:
        """Toggle retention of block inputs (needed only for the redundancy probe)."""
        self._capture_inputs = enabled

    def _router_hook(self, name: str) -> Callable:
        def hook(_module: torch.nn.Module, _inputs: tuple, output: tuple) -> None:
            self.weights[name] = output[0].detach().float().cpu()

        return hook

    def _input_hook(self, name: str) -> Callable:
        def hook(_module: torch.nn.Module, inputs: tuple) -> None:
            if self._capture_inputs:
                self.inputs[name] = inputs[0].detach()

        return hook

    def clear(self) -> None:
        """Drop captured tensors between images."""
        self.weights.clear()
        self.inputs.clear()


def preprocess(image_path: Path, imgsz: int, device: torch.device) -> tuple[torch.Tensor, int, int]:
    """Load an image as a stretched ``imgsz x imgsz`` tensor.

    A plain resize (not letterbox) is deliberate: it keeps normalized annotation
    coordinates valid in the network's own frame, so a box at relative position
    ``(x, y)`` maps to token ``(x * grid_w, y * grid_h)`` with no padding offsets.

    Returns:
        (tuple): ``(tensor, original_width, original_height)``.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    height, width = image.shape[:2]
    resized = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    return tensor.to(device), width, height


def scene_token_groups(
    objects: Sequence[VisDroneObject],
    width: int,
    height: int,
    grid_h: int,
    grid_w: int,
    *,
    size_split: tuple[float, float],
    shape_band: tuple[float, float],
    density_radius: int,
    box_count: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the token masks for the dense / small / irregular contrasts.

    Every arm is restricted to tokens covered by exactly one box, so the comparison
    is between object tokens of the same occupancy inside the same image. That
    removes the density and box-overlap confounds in the same way the occlusion
    token test does.

    Args:
        objects (Sequence[VisDroneObject]): Valid objects for the image.
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        grid_h (int): Feature-map height (token rows).
        grid_w (int): Feature-map width (token columns).
        size_split (tuple[float, float]): ``(small_max, large_min)`` from :func:`area_terciles`.
        shape_band (tuple[float, float]): Area band shared by irregular and regular boxes.
        density_radius (int): Neighbourhood half-width for the local-density split.
        box_count (np.ndarray): Per-token occupancy over *all* boxes.

    Returns:
        (dict[str, np.ndarray]): Boolean masks keyed by token-group name.
    """
    small_max, large_min = size_split
    extremes = [obj for obj in objects if not small_max < box_area(obj, width, height) < large_min]
    middle = [obj for obj in objects if small_max < box_area(obj, width, height) < large_min]
    size = token_group_masks(
        extremes,
        width,
        height,
        grid_h,
        grid_w,
        lambda obj: box_area(obj, width, height) <= small_max,
        others=middle,
    )
    shape = token_group_masks(objects, width, height, grid_h, grid_w, is_irregular_box)
    shape_sized = token_group_masks(objects, width, height, grid_h, grid_w, is_irregular_box, area_range=shape_band)
    density = local_density_masks(
        token_local_density(objects, width, height, grid_h, grid_w, radius=density_radius),
        box_count == 1,
    )
    return {
        "small_solo": size["positive_solo"],
        "large_solo": size["negative_solo"],
        "irregular_solo": shape["positive_solo"],
        "regular_solo": shape["negative_solo"],
        "irregular_solo_sized": shape_sized["positive_solo"],
        "regular_solo_sized": shape_sized["negative_solo"],
        "crowded_solo": density["crowded"],
        "isolated_solo": density["isolated"],
    }


def analyse_images(
    model: torch.nn.Module,
    probe: MoTRoutingProbe,
    samples: Sequence[tuple[Path, list[VisDroneObject], int, int]],
    *,
    imgsz: int,
    device: torch.device,
    area_band: tuple[float, float],
    size_split: tuple[float, float],
    shape_band: tuple[float, float],
    density_radius: int = 1,
) -> list[ImageRecord]:
    """Forward every image and reduce routing weights to per-image statistics."""
    records: list[ImageRecord] = []
    for index, (image_path, objects, width, height) in enumerate(samples):
        probe.clear()
        tensor, _, _ = preprocess(image_path, imgsz, device)
        with torch.inference_mode():
            model(tensor)

        layer_weight: dict[str, np.ndarray] = {}
        layer_top1: dict[str, np.ndarray] = {}
        token_group_weight: dict[str, dict[str, np.ndarray]] = {}
        token_group_sizes: dict[str, int] = {}

        for layer, weights in probe.weights.items():
            maps = weights[0].numpy()  # [E, H, W]
            layer_weight[layer] = maps.reshape(maps.shape[0], -1).mean(axis=1)
            winner = maps.argmax(axis=0)
            layer_top1[layer] = np.asarray(
                [float((winner == expert).mean()) for expert in range(maps.shape[0])], dtype=np.float64
            )

            grid_h, grid_w = maps.shape[1], maps.shape[2]
            masks = token_object_masks(objects, width, height, grid_h, grid_w)
            sized = token_object_masks(objects, width, height, grid_h, grid_w, area_range=area_band)
            groups = {
                "occluded": masks["occluded"],
                "clear": masks["clear"],
                "background": masks["background"],
                "occluded_solo": masks["occluded_solo"],
                "clear_solo": masks["clear_solo"],
                "occluded_solo_sized": sized["occluded_solo"],
                "clear_solo_sized": sized["clear_solo"],
                **scene_token_groups(
                    objects,
                    width,
                    height,
                    grid_h,
                    grid_w,
                    size_split=size_split,
                    shape_band=shape_band,
                    density_radius=density_radius,
                    box_count=masks["box_count"],
                ),
            }
            token_group_weight[layer] = {
                group: np.asarray(
                    [
                        float(maps[expert][mask].mean()) if mask.any() else float("nan")
                        for expert in range(maps.shape[0])
                    ],
                    dtype=np.float64,
                )
                for group, mask in groups.items()
            }
            for group, mask in groups.items():
                token_group_sizes[group] = token_group_sizes.get(group, 0) + int(mask.sum())

        records.append(
            ImageRecord(
                stem=image_path.stem,
                image_path=image_path,
                features=image_scene_features(objects, width, height).to_dict(),
                layer_expert_weight=layer_weight,
                layer_expert_top1=layer_top1,
                token_group_weight=token_group_weight,
                token_group_sizes=token_group_sizes,
            )
        )
        if (index + 1) % 50 == 0:
            print(f"[interpret] routed {index + 1}/{len(samples)} images")
    return records


def scene_assignments(records: Sequence[ImageRecord], *, low: float = 0.3, high: float = 0.7) -> dict[str, np.ndarray]:
    """Split images into overlapping scene groups by quantiles of scene features.

    ``irregular_rate`` is 0 for most VisDrone frames, so it goes through
    :func:`~scripts.mot_routing_interpret.quantile_contrast_split`, which falls back
    to a tie-safe cut instead of putting every image in both arms.
    """

    def feature(name: str) -> np.ndarray:
        return np.asarray([record.features[name] for record in records], dtype=np.float64)

    counts, areas = feature("n_objects"), feature("median_area")
    occlusion, heavy = feature("occlusion_rate"), feature("heavy_occlusion_rate")
    irregular_high, irregular_low = quantile_contrast_split(feature("irregular_rate"), low=low, high=high)
    return {
        "dense": counts >= np.quantile(counts, high),
        "sparse": counts <= np.quantile(counts, low),
        "small_objects": areas <= np.quantile(areas, low),
        "large_objects": areas >= np.quantile(areas, high),
        "high_occlusion": occlusion >= np.quantile(occlusion, high),
        "low_occlusion": occlusion <= np.quantile(occlusion, low),
        "heavy_occlusion": heavy >= np.quantile(heavy, high),
        "irregular_objects": irregular_high,
        "regular_objects": irregular_low,
    }


def aggregate_scene_shares(
    records: Sequence[ImageRecord],
    scenes: dict[str, np.ndarray],
    layer_names: Sequence[str],
    *,
    metric: str = "weight",
) -> dict[str, dict[str, list[float]]]:
    """Average per-expert routing shares per scene, per layer and pooled."""
    attribute = "layer_expert_weight" if metric == "weight" else "layer_expert_top1"
    result: dict[str, dict[str, list[float]]] = {}
    for scene, mask in scenes.items():
        selected = [record for record, flag in zip(records, mask) if flag]
        if not selected:
            continue
        per_layer: dict[str, list[float]] = {}
        for layer in layer_names:
            stack = np.stack([getattr(record, attribute)[layer] for record in selected])
            per_layer[layer] = stack.mean(axis=0).tolist()
        pooled = np.stack(
            [np.stack([getattr(record, attribute)[layer] for layer in layer_names]).mean(axis=0) for record in selected]
        )
        per_layer[POOLED] = pooled.mean(axis=0).tolist()
        per_layer["__n_images__"] = [float(len(selected))]
        result[scene] = per_layer
    return result


def expert_series(
    records: Sequence[ImageRecord], layer_names: Sequence[str], layer: str, expert_index: int
) -> np.ndarray:
    """Per-image routing weight for one expert, at one layer or pooled over layers."""
    if layer != POOLED:
        return np.asarray([record.layer_expert_weight[layer][expert_index] for record in records], dtype=np.float64)
    return np.asarray(
        [np.mean([record.layer_expert_weight[name][expert_index] for name in layer_names]) for record in records],
        dtype=np.float64,
    )


def stratified_scene_tests(
    records: Sequence[ImageRecord],
    layer_names: Sequence[str],
    contrast: SceneContrast,
    *,
    n_permutations: int,
    n_strata: int,
    seed: int,
    alternative: str,
) -> list[dict[str, Any]]:
    """Compare image-level expert shares across one scene split, covariates held fixed.

    The two covariates are binned into quantile strata and crossed, and group labels
    are permuted only *within* a stratum. Without that control none of these splits
    is interpretable on VisDrone, where crowding, object scale, occlusion and box
    elongation all move together.

    Args:
        records (Sequence[ImageRecord]): Analysed images.
        layer_names (Sequence[str]): MoT block names.
        contrast (SceneContrast): The split and the covariates to hold fixed.
        n_permutations (int): Permutations per test.
        n_strata (int): Quantile bins per covariate.
        seed (int): RNG seed.
        alternative (str): ``'greater'``, ``'less'`` or ``'two-sided'``.

    Returns:
        (list[dict[str, Any]]): One row per expert x layer, plus a pooled-layer row
            that is excluded from the FDR family.
    """

    def feature(name: str) -> np.ndarray:
        return np.asarray([record.features[name] for record in records], dtype=np.float64)

    split = feature(contrast.feature)
    high, low = quantile_contrast_split(split)
    in_group = high if contrast.group_is_high else low
    keep = high | low
    first, second = (quantile_strata(feature(name), n_strata) for name in contrast.covariates)
    joint = first * n_strata + second
    covariate_label = " x ".join(contrast.covariates) + " quantile bins"

    rows: list[dict[str, Any]] = []
    for expert_index, expert in enumerate(EXPERT_NAMES):
        for layer in [*layer_names, POOLED]:
            values = expert_series(records, layer_names, layer, expert_index)
            report = stratified_permutation_test(
                values[keep],
                in_group[keep],
                joint[keep],
                n_permutations=n_permutations,
                seed=seed,
                alternative=alternative,
            )
            correlation = spearman_correlation(split, values)
            rows.append(
                {
                    "contrast": contrast.name,
                    "expert": expert,
                    "layer": layer,
                    "test": f"{contrast.name}_stratified",
                    "split_feature": contrast.feature,
                    "group_is_high": contrast.group_is_high,
                    "alternative": alternative,
                    "covariates": covariate_label,
                    "in_fdr_family": layer != POOLED,
                    **report.to_dict(),
                    "spearman_rho_vs_split_feature": correlation["rho"],
                    "spearman_p_value": correlation["p_value"],
                }
            )
    return rows


def image_level_occlusion_tests(
    records: Sequence[ImageRecord],
    layer_names: Sequence[str],
    *,
    n_permutations: int,
    n_strata: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Test image-level expert share against occlusion, controlling density and scale.

    One-sided, because the pre-registered claim is directional: Deformable routing is
    expected to *rise* under occlusion.
    """
    return stratified_scene_tests(
        records,
        layer_names,
        SceneContrast("high_vs_low_occlusion", "occlusion_rate", True, ("n_objects", "median_area")),
        n_permutations=n_permutations,
        n_strata=n_strata,
        seed=seed,
        alternative="greater",
    )


def scene_contrast_image_tests(
    records: Sequence[ImageRecord],
    layer_names: Sequence[str],
    *,
    n_permutations: int,
    n_strata: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Run the dense / small / irregular image-level contrasts as one FDR family.

    The three contrasts share a family so the correction reflects how many scene
    hypotheses were actually asked, rather than flattering each by testing it alone.
    """
    return [
        row
        for contrast in SCENE_CONTRASTS
        for row in stratified_scene_tests(
            records,
            layer_names,
            contrast,
            n_permutations=n_permutations,
            n_strata=n_strata,
            seed=seed,
            alternative="two-sided",
        )
    ]


def token_level_tests(
    records: Sequence[ImageRecord],
    layer_names: Sequence[str],
    pairs: Sequence[TokenPair] = TOKEN_PAIR_TESTS,
    *,
    alternative: str = "greater",
) -> list[dict[str, Any]]:
    """Run within-image paired comparisons of expert weight between token groups.

    Args:
        records (Sequence[ImageRecord]): Analysed images.
        layer_names (Sequence[str]): MoT block names.
        pairs (Sequence[TokenPair]): Token-group pairs to compare.
        alternative (str): Wilcoxon alternative shared by the whole family.

    Returns:
        (list[dict[str, Any]]): One row per expert x pair x layer.
    """
    rows: list[dict[str, Any]] = []
    for expert_index, expert in enumerate(EXPERT_NAMES):
        for pair in pairs:
            for layer in [*layer_names, POOLED]:
                if layer == POOLED:
                    values_a = [
                        nan_mean(record.token_group_weight[name][pair.group_a][expert_index] for name in layer_names)
                        for record in records
                    ]
                    values_b = [
                        nan_mean(record.token_group_weight[name][pair.group_b][expert_index] for name in layer_names)
                        for record in records
                    ]
                else:
                    values_a = [record.token_group_weight[layer][pair.group_a][expert_index] for record in records]
                    values_b = [record.token_group_weight[layer][pair.group_b][expert_index] for record in records]
                report = paired_difference_report(values_a, values_b, alternative=alternative)
                rows.append(
                    {
                        "contrast": pair.contrast,
                        "expert": expert,
                        "layer": layer,
                        "group_a": pair.group_a,
                        "group_b": pair.group_b,
                        "description": pair.description,
                        "alternative": alternative,
                        "in_fdr_family": layer != POOLED,
                        **report.to_dict(),
                    }
                )
    return rows


def expert_redundancy(
    model: torch.nn.Module,
    probe: MoTRoutingProbe,
    samples: Sequence[tuple[Path, list[VisDroneObject], int, int]],
    *,
    imgsz: int,
    device: torch.device,
    max_images: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Measure how similar the experts' residual contributions are.

    Each MoT block input is replayed through all three experts, and similarity is
    computed on ``expert(x) - x``. That subtraction is the whole point: the shared
    identity path makes raw-output cosine ~1.0 no matter what the experts learned.
    """
    probe.capture_inputs(True)
    accumulated: dict[str, list[np.ndarray]] = defaultdict(list)
    try:
        for image_path, _, _, _ in samples[:max_images]:
            probe.clear()
            tensor, _, _ = preprocess(image_path, imgsz, device)
            with torch.inference_mode():
                model(tensor)
            for layer, block_input in probe.inputs.items():
                accumulated[layer].append(block_input.clone())
    finally:
        probe.capture_inputs(False)

    rows: list[dict[str, Any]] = []
    for layer, inputs in accumulated.items():
        block = probe.blocks[layer]
        base_chunks: list[np.ndarray] = []
        expert_chunks: list[list[np.ndarray]] = [[] for _ in block.experts]
        for block_input in inputs:
            with torch.inference_mode():
                outputs = [expert(block_input) for expert in block.experts]
            flat_input = block_input[0].flatten(1).transpose(0, 1).float().cpu().numpy()
            base_chunks.append(flat_input)
            for index, output in enumerate(outputs):
                expert_chunks[index].append(output[0].flatten(1).transpose(0, 1).float().cpu().numpy())

        base = np.concatenate(base_chunks, axis=0)
        experts = [np.concatenate(chunks, axis=0) for chunks in expert_chunks]
        if base.shape[0] > max_tokens:
            selection = np.random.default_rng(0).choice(base.shape[0], size=max_tokens, replace=False)
            base = base[selection]
            experts = [item[selection] for item in experts]

        report = delta_similarity_report(layer, base, experts)
        rows.append(
            {
                "layer": layer,
                "n_tokens": int(base.shape[0]),
                "relative_delta_magnitude": {
                    name: value for name, value in zip(EXPERT_NAMES, report.relative_magnitude)
                },
                "output_cosine_uninformative": report.output_cosine,
                "delta_cosine": report.delta_cosine,
                "delta_cka": report.delta_cka,
            }
        )
    return rows


def force_expert_hooks(probe: MoTRoutingProbe, expert_index: int) -> list[Any]:
    """Force every MoT router to route all tokens to one expert."""

    def hook(module: torch.nn.Module, _inputs: tuple, output: tuple) -> tuple:
        weights, indices = output[0], output[1]
        forced_weights = torch.zeros_like(weights)
        forced_weights[:, expert_index] = 1.0
        forced_indices = torch.full_like(indices, expert_index)
        return (forced_weights, forced_indices, *output[2:])

    return [block.router.register_forward_hook(hook) for block in probe.blocks.values()]


def shuffle_routing_hooks(probe: MoTRoutingProbe, seed: int) -> list[Any]:
    """Randomly permute each token's expert weights, preserving their distribution.

    This isolates *content-dependent* routing: the same weight values are dealt
    out, only the expert each one lands on is randomised. If mAP is unchanged, the
    router's learned input-dependence carries no measurable value.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def hook(module: torch.nn.Module, _inputs: tuple, output: tuple) -> tuple:
        weights, indices = output[0], output[1]
        noise = torch.rand(weights.shape, generator=generator).to(weights.device)
        permutation = noise.argsort(dim=1)
        shuffled = torch.gather(weights, 1, permutation)
        top_k = indices.shape[1]
        new_indices = shuffled.topk(top_k, dim=1).indices.to(indices.dtype)
        return (shuffled, new_indices, *output[2:])

    return [block.router.register_forward_hook(hook) for block in probe.blocks.values()]


def causal_map_tests(
    checkpoint: Path,
    data_yaml: Path,
    probe_factory: Callable[[torch.nn.Module], MoTRoutingProbe],
    *,
    device: torch.device,
    imgsz: int,
    batch: int,
    split: str,
) -> list[dict[str, Any]]:
    """Re-validate mAP under natural, forced-single-expert, and shuffled routing."""
    from ultralytics import YOLO

    interventions: list[tuple[str, Callable[[MoTRoutingProbe], list[Any]] | None]] = [
        ("natural", None),
        *[
            (f"forced_{name}", (lambda probe, index=index: force_expert_hooks(probe, index)))
            for index, name in enumerate(EXPERT_NAMES)
        ],
        ("shuffled_routing", lambda probe: shuffle_routing_hooks(probe, seed=0)),
    ]

    rows: list[dict[str, Any]] = []
    for label, install in interventions:
        model = YOLO(str(checkpoint))
        probe = probe_factory(model.model)
        handles = install(probe) if install is not None else []
        try:
            metrics = model.val(
                data=str(data_yaml),
                split=split,
                imgsz=imgsz,
                batch=batch,
                device=str(device),
                verbose=False,
                plots=False,
                save_json=False,
                project=None,
            )
            rows.append(
                {
                    "intervention": label,
                    "mAP50_95": float(metrics.box.map),
                    "mAP50": float(metrics.box.map50),
                    "precision": float(metrics.box.mp),
                    "recall": float(metrics.box.mr),
                }
            )
            print(f"[interpret] {label}: mAP50-95={metrics.box.map:.5f} mAP50={metrics.box.map50:.5f}")
        finally:
            for handle in handles:
                handle.remove()
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    baseline = next(row for row in rows if row["intervention"] == "natural")
    for row in rows:
        row["delta_mAP50_95_vs_natural"] = row["mAP50_95"] - baseline["mAP50_95"]
        row["relative_mAP50_95_vs_natural"] = (
            row["delta_mAP50_95_vs_natural"] / baseline["mAP50_95"] if baseline["mAP50_95"] else float("nan")
        )
    return rows


def collapse_report(records: Sequence[ImageRecord], layer_names: Sequence[str]) -> list[dict[str, Any]]:
    """Summarise per-layer expert usage concentration over the analysed images."""
    rows: list[dict[str, Any]] = []
    for layer in layer_names:
        usage = np.stack([record.layer_expert_weight[layer] for record in records]).mean(axis=0)
        top1 = np.stack([record.layer_expert_top1[layer] for record in records]).mean(axis=0)
        rows.append(
            {
                "layer": layer,
                "mean_weight": usage.tolist(),
                "top1_share": top1.tolist(),
                **usage_concentration(usage.tolist()),
                "dead_experts": [index for index, value in enumerate(usage.tolist()) if value < 0.01],
            }
        )
    return rows


def apply_fdr(rows: list[dict[str, Any]], key: str = "p_value") -> list[dict[str, Any]]:
    """Attach Benjamini-Hochberg q-values to the primary tests in a family.

    Rows flagged ``in_fdr_family=False`` (the cross-layer pooled summaries) are
    excluded from the correction: they are aggregates of rows already in the family,
    so counting them would inflate the family size and weaken every q-value.
    """
    family = [row for row in rows if row.get("in_fdr_family", True)]
    outcome = benjamini_hochberg([row.get(key, float("nan")) for row in family])
    for row, q_value, rejected in zip(family, outcome["q_values"], outcome["rejected"]):
        row["q_value_bh"] = q_value
        row["significant_after_fdr"] = bool(rejected)
    for row in rows:
        row.setdefault("q_value_bh", float("nan"))
        row.setdefault("significant_after_fdr", False)
    return rows


def write_json(path: Path, payload: object) -> None:
    """Write a JSON artifact, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write dict rows to CSV with a stable union-of-keys header."""
    import csv

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def collect_visdrone_samples(
    images_dir: Path,
    annotations_dir: Path,
    limit: int,
) -> list[tuple[Path, list[VisDroneObject], int, int]]:
    """Pair VisDrone images with parsed annotations and their pixel dimensions."""
    paths = [path for path in sorted(images_dir.rglob("*")) if path.suffix.lower() in IMAGE_SUFFIXES]
    if limit > 0:
        paths = paths[:limit]
    samples: list[tuple[Path, list[VisDroneObject], int, int]] = []
    for path in paths:
        objects = load_visdrone_objects(annotations_dir / f"{path.stem}.txt")
        if not objects:
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        samples.append((path, objects, image.shape[1], image.shape[0]))
    if not samples:
        raise SystemExit(f"no annotated images found under {images_dir}")
    return samples


def collect_coco_samples(
    images_dir: Path,
    instances_json: Path,
    limit: int,
) -> list[tuple[Path, list[VisDroneObject], int, int]]:
    """Pair COCO images with objects whose occlusion codes are derived from overlap.

    COCO has no occlusion annotation, so the codes come from box-overlap coverage
    plus ``iscrowd``. This makes the COCO run a *replication under a proxy label*,
    weaker evidence than the VisDrone run's ground-truth occlusion.
    """
    with instances_json.open() as handle:
        instances = json.load(handle)
    by_stem = coco_objects_by_image(instances)
    paths = [path for path in sorted(images_dir.rglob("*")) if path.suffix.lower() in IMAGE_SUFFIXES]
    if limit > 0:
        paths = paths[:limit]
    samples = [(path, *by_stem[path.stem]) for path in paths if path.stem in by_stem and by_stem[path.stem][0]]
    if not samples:
        raise SystemExit(f"no annotated images matched between {images_dir} and {instances_json}")
    return samples


def exemplar_images(records: Sequence[ImageRecord], count: int) -> list[ImageRecord]:
    """Pick the most- and least-occluded images for spatial heatmap figures.

    Candidates are first restricted to images with a typical object count (the
    middle of the ``n_objects`` distribution). Without that filter the extremes of
    ``occlusion_rate`` are all near-empty images — a 1-of-1 occluded object scores
    1.00 and would be shown as the exemplar "occluded scene", which is misleading.

    Selection takes ``count // 2`` from each end of the occlusion ordering and
    de-duplicates, so a candidate pool smaller than ``count`` yields every distinct
    candidate rather than repeating one. Call this **once** per run and pass the
    result around: re-applying it to its own output would narrow the quantile band
    again and silently drop exemplars.

    Examples:
        >>> def stub(stem, occlusion, objects):
        ...     features = {"occlusion_rate": occlusion, "n_objects": objects}
        ...     return ImageRecord(stem, Path(stem), features, {}, {}, {}, {})
        >>> pool = [stub(f"i{n}", n / 10, 40 + n) for n in range(10)]
        >>> [record.stem for record in exemplar_images(pool, 4)]
        ['i4', 'i5', 'i7', 'i8']

        A near-empty image never becomes the "occluded scene" exemplar, even when its
        occlusion rate is a perfect 1.0:

        >>> pool = [stub(f"busy{n}", n / 10, 50 + n) for n in range(8)] + [stub("one-object", 1.0, 1)]
        >>> [record.stem for record in exemplar_images(pool, 2)]
        ['busy3', 'busy6']

        A pool smaller than ``count`` yields distinct records, not repeats:

        >>> [record.stem for record in exemplar_images(pool[:1], 4)]
        ['busy0']
        >>> exemplar_images(pool, 0)
        []
    """
    if count <= 0 or not records:
        return []
    counts = [record.features["n_objects"] for record in records]
    low, high = float(np.quantile(counts, 0.4)), float(np.quantile(counts, 0.9))
    candidates = [record for record in records if low <= record.features["n_objects"] <= high] or list(records)
    ordered = sorted(candidates, key=lambda record: record.features["occlusion_rate"])
    half = max(1, count // 2)
    picked: list[ImageRecord] = []
    seen: set[str] = set()
    for record in [*ordered[:half], *reversed(ordered[-half:]), *ordered]:
        if len(picked) >= count:
            break
        if record.stem not in seen:
            seen.add(record.stem)
            picked.append(record)
    # Sorted so figure rows read clear → occluded rather than in pick order.
    return sorted(picked, key=lambda record: record.features["occlusion_rate"])


def spatial_maps(
    model: torch.nn.Module,
    probe: MoTRoutingProbe,
    exemplars: Sequence[ImageRecord],
    *,
    layer: str,
    imgsz: int,
    device: torch.device,
    thumbnail: int = 240,
) -> dict[str, Any]:
    """Re-run the chosen exemplar images and keep their per-token routing maps.

    Takes an already-selected exemplar list (see :func:`exemplar_images`) rather than
    selecting again, so the same images appear in ``spatial_maps.json`` and
    ``exemplars.json``.

    Only the exemplars are kept: the full per-token tensor for every image would be
    gigabytes, and the figure needs a visual spot-check, not the whole split.
    """
    examples: list[dict[str, Any]] = []
    for record in exemplars:
        probe.clear()
        tensor, _, _ = preprocess(record.image_path, imgsz, device)
        with torch.inference_mode():
            model(tensor)
        weights = probe.weights.get(layer)
        if weights is None:
            continue
        maps = weights[0].numpy()
        image = cv2.cvtColor(cv2.imread(str(record.image_path)), cv2.COLOR_BGR2RGB)
        scale = thumbnail / max(image.shape[:2])
        preview = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))))
        examples.append(
            {
                "image": record.image_path.as_posix(),
                "label": record.stem,
                "layer": layer,
                "occlusion_rate": record.features["occlusion_rate"],
                "n_objects": record.features["n_objects"],
                "thumbnail": preview.tolist(),
                "maps": {name: maps[index].tolist() for index, name in enumerate(EXPERT_NAMES)},
            }
        )
    return {"layer": layer, "examples": examples}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained MoT .pt checkpoint.")
    parser.add_argument("--data", type=Path, required=True, help="Dataset YAML (for splits and mAP validation).")
    parser.add_argument(
        "--dataset-kind",
        choices=("visdrone", "coco"),
        default="visdrone",
        help="visdrone uses ground-truth occlusion labels; coco derives them from box overlap.",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", type=Path, required=True, help="Directory for artifacts.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8, help="Batch size for the mAP validation runs.")
    parser.add_argument("--limit", type=int, default=0, help="Max images for routing analysis (0 = all).")
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--strata", type=int, default=3, help="Quantile bins per covariate.")
    parser.add_argument(
        "--density-radius",
        type=int,
        default=1,
        help="Neighbourhood half-width, in tokens, for the within-image dense-vs-sparse split.",
    )
    parser.add_argument("--redundancy-images", type=int, default=24)
    parser.add_argument("--redundancy-tokens", type=int, default=20000)
    parser.add_argument("--exemplars", type=int, default=4)
    parser.add_argument(
        "--spatial-layer",
        default="model.14.m.0",
        help="MoT block whose per-token maps are saved for the spatial figure.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-map", action="store_true", help="Skip the forced-routing mAP validations.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the interpretability suite."""
    args = parse_args(argv)
    from ultralytics import YOLO

    device = resolve_device(args.device)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    if args.dataset_kind == "visdrone":
        images_dir, annotations_dir = find_visdrone_root(args.data, args.split)
        samples = collect_visdrone_samples(images_dir, annotations_dir, args.limit)
        occlusion_source = "visdrone_ground_truth_occlusion_label"
    else:
        images_dir, annotations_dir = find_coco_split(args.data, args.split)
        samples = collect_coco_samples(images_dir, annotations_dir, args.limit)
        occlusion_source = "derived_from_box_overlap_and_iscrowd_proxy"
    print(f"[interpret] analysing {len(samples)} annotated images from {images_dir}")

    per_image_objects = [(objects, width, height) for _, objects, width, height in samples]
    area_band = matched_area_band(per_image_objects, lambda obj: obj.occlusion >= 1)
    shape_band = matched_area_band(per_image_objects, is_irregular_box)
    size_split = area_terciles(per_image_objects)
    print(f"[interpret] size-matched normalized-area band: [{area_band[0]:.6f}, {area_band[1]:.6f}]")
    print(f"[interpret] shape-matched normalized-area band: [{shape_band[0]:.6f}, {shape_band[1]:.6f}]")
    print(f"[interpret] small/large normalized-area terciles: {size_split[0]:.6f} / {size_split[1]:.6f}")

    model = YOLO(str(args.checkpoint)).model.eval().to(device)
    with MoTRoutingProbe(model) as probe:
        layer_names = probe.layer_names
        print(f"[interpret] MoT blocks: {layer_names}")
        records = analyse_images(
            model,
            probe,
            samples,
            imgsz=args.imgsz,
            device=device,
            area_band=area_band,
            size_split=size_split,
            shape_band=shape_band,
            density_radius=args.density_radius,
        )
        redundancy = expert_redundancy(
            model,
            probe,
            samples,
            imgsz=args.imgsz,
            device=device,
            max_images=args.redundancy_images,
            max_tokens=args.redundancy_tokens,
        )
        exemplars = exemplar_images(records, args.exemplars)
        spatial = spatial_maps(
            model,
            probe,
            exemplars,
            layer=args.spatial_layer if args.spatial_layer in layer_names else layer_names[0],
            imgsz=args.imgsz,
            device=device,
        )

    scenes = scene_assignments(records)
    payload: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "dataset_kind": args.dataset_kind,
        "occlusion_source": occlusion_source,
        "split": args.split,
        "n_images": len(records),
        "imgsz": args.imgsz,
        "expert_names": list(EXPERT_NAMES),
        "layers": layer_names,
        "size_matched_area_band": list(area_band),
        "shape_matched_area_band": list(shape_band),
        "size_contrast_area_terciles": list(size_split),
        "density_radius_tokens": args.density_radius,
        "scene_counts": {scene: int(mask.sum()) for scene, mask in scenes.items()},
        "token_group_totals": {
            group: int(sum(record.token_group_sizes.get(group, 0) for record in records))
            for group in sorted({key for record in records for key in record.token_group_sizes})
        },
    }

    payload["scene_mean_weight"] = aggregate_scene_shares(records, scenes, layer_names, metric="weight")
    payload["scene_top1_share"] = aggregate_scene_shares(records, scenes, layer_names, metric="top1")
    payload["collapse"] = collapse_report(records, layer_names)
    payload["redundancy"] = redundancy

    image_tests = apply_fdr(
        image_level_occlusion_tests(
            records,
            layer_names,
            n_permutations=args.permutations,
            n_strata=args.strata,
            seed=args.seed,
        )
    )
    token_tests = apply_fdr(token_level_tests(records, layer_names))
    scene_image_tests = apply_fdr(
        scene_contrast_image_tests(
            records,
            layer_names,
            n_permutations=args.permutations,
            n_strata=args.strata,
            seed=args.seed,
        )
    )
    scene_token_tests = apply_fdr(token_level_tests(records, layer_names, SCENE_TOKEN_PAIRS, alternative="two-sided"))
    payload["image_level_occlusion_tests"] = image_tests
    payload["token_level_tests"] = token_tests
    payload["scene_contrast_image_tests"] = scene_image_tests
    payload["scene_contrast_token_tests"] = scene_token_tests

    if not args.skip_map:
        payload["causal_map"] = causal_map_tests(
            args.checkpoint,
            args.data,
            MoTRoutingProbe,
            device=device,
            imgsz=args.imgsz,
            batch=args.batch,
            split=args.split,
        )

    per_image_rows = [
        {
            "image": record.stem,
            **record.features,
            **{
                f"{layer}|{expert}": record.layer_expert_weight[layer][index]
                for layer in layer_names
                for index, expert in enumerate(EXPERT_NAMES)
            },
        }
        for record in records
    ]

    write_json(output / "routing_analysis.json", payload)
    write_json(output / "spatial_maps.json", spatial)
    write_csv(output / "per_image_routing.csv", per_image_rows)
    write_csv(output / "image_level_occlusion_tests.csv", image_tests)
    write_csv(output / "token_level_tests.csv", token_tests)
    write_csv(output / "scene_contrast_image_tests.csv", scene_image_tests)
    write_csv(output / "scene_contrast_token_tests.csv", scene_token_tests)
    if "causal_map" in payload:
        write_csv(output / "causal_map.csv", payload["causal_map"])
    write_json(
        output / "exemplars.json",
        [
            {
                "image": record.image_path.as_posix(),
                "occlusion_rate": record.features["occlusion_rate"],
                "n_objects": record.features["n_objects"],
            }
            for record in exemplars
        ],
    )
    print(f"[interpret] wrote artifacts to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
