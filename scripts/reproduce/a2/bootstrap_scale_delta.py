#!/usr/bin/env python3
"""Compare baseline and AT-STAL small AP with paired image-level bootstrap."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

try:
    from .evaluate_scale_aware import _qa_area_ranges
except ImportError:  # pragma: no cover - supports direct script execution
    from evaluate_scale_aware import _qa_area_ranges

Evaluator = Callable[[dict, list[dict]], float]
BOOTSTRAP_MAX_DET = 500


def _bootstrap_area_ranges() -> list[list[float]]:
    """Return the same A2 half-open area ranges used by scale-aware eval."""
    return _qa_area_ranges()


def _mean_valid(values: np.ndarray) -> float:
    """Average COCO values while ignoring the -1 sentinel."""
    values = np.asarray(values, dtype=np.float64)
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else float("nan")


def _evaluate_small_ap(dataset: dict, predictions: list[dict]) -> float:
    """Evaluate COCO AP_small for an in-memory dataset and prediction list."""
    try:
        from faster_coco_eval import COCO, COCOeval_faster
    except ImportError as exc:
        raise RuntimeError(
            "faster-coco-eval is required; install it with `pip install faster-coco-eval>=1.6.7`"
        ) from exc

    annotation = COCO()
    annotation.dataset = dataset
    annotation.createIndex()
    if predictions:
        detections = annotation.loadRes(predictions)
    else:
        detections = COCO()
        detections.dataset = {
            "images": copy.deepcopy(dataset["images"]),
            "categories": copy.deepcopy(dataset["categories"]),
            "annotations": [],
        }
        detections.createIndex()
    evaluator = COCOeval_faster(
        annotation,
        detections,
        iouType="bbox",
        lvis_style=False,
        print_function=lambda *_args, **_kwargs: None,
    )
    evaluator.params.imgIds = [int(image["id"]) for image in dataset["images"]]
    evaluator.params.maxDets = [1, 10, 100, BOOTSTRAP_MAX_DET]
    evaluator.params.areaRng = _bootstrap_area_ranges()
    evaluator.params.areaRngLbl = ["all", "small", "medium", "large"]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    precision = np.asarray(evaluator.eval["precision"])
    area_labels = list(getattr(evaluator.params, "areaRngLbl", ("all", "small", "medium", "large")))
    small_index = area_labels.index("small")
    max_index = list(evaluator.params.maxDets).index(BOOTSTRAP_MAX_DET)
    return _mean_valid(precision[:, :, :, small_index, max_index])


def _group_by_image(records: list[dict]) -> dict[int, list[dict]]:
    """Group COCO annotations or predictions by integer image ID."""
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        grouped[int(record["image_id"])].append(record)
    return grouped


def _resample_pair(
    ground_truth: dict,
    baseline_predictions: list[dict],
    candidate_predictions: list[dict],
    sampled_ids: np.ndarray,
) -> tuple[dict, list[dict], list[dict]]:
    """Duplicate a paired image sample under fresh COCO image and annotation IDs."""
    images_by_id = {int(image["id"]): image for image in ground_truth["images"]}
    annotations_by_id = _group_by_image(ground_truth.get("annotations", []))
    baseline_by_id = _group_by_image(baseline_predictions)
    candidate_by_id = _group_by_image(candidate_predictions)

    sampled_gt = {
        key: copy.deepcopy(value) for key, value in ground_truth.items() if key not in {"images", "annotations"}
    }
    sampled_gt["images"] = []
    sampled_gt["annotations"] = []
    sampled_baseline: list[dict] = []
    sampled_candidate: list[dict] = []
    annotation_id = 1

    for new_image_id, original_image_id in enumerate(sampled_ids.tolist(), start=1):
        original_image_id = int(original_image_id)
        image = copy.deepcopy(images_by_id[original_image_id])
        image["id"] = new_image_id
        sampled_gt["images"].append(image)

        for annotation in annotations_by_id.get(original_image_id, []):
            item = copy.deepcopy(annotation)
            item["id"] = annotation_id
            item["image_id"] = new_image_id
            sampled_gt["annotations"].append(item)
            annotation_id += 1

        for source, destination in (
            (baseline_by_id.get(original_image_id, []), sampled_baseline),
            (candidate_by_id.get(original_image_id, []), sampled_candidate),
        ):
            for prediction in source:
                item = copy.deepcopy(prediction)
                item["image_id"] = new_image_id
                destination.append(item)

    return sampled_gt, sampled_baseline, sampled_candidate


def paired_bootstrap(
    ground_truth: dict,
    baseline_predictions: list[dict],
    candidate_predictions: list[dict],
    *,
    replicates: int = 2000,
    seed: int = 42,
    min_improvement: float = 0.01,
    evaluator: Evaluator | None = None,
    progress_every: int = 100,
) -> dict:
    """Return point estimates and a percentile CI for paired AP_small improvement."""
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if min_improvement < 0:
        raise ValueError("min_improvement must be non-negative")
    evaluator = evaluator or _evaluate_small_ap
    image_ids = np.asarray([int(image["id"]) for image in ground_truth.get("images", [])], dtype=np.int64)
    if image_ids.size == 0:
        raise ValueError("ground truth contains no images")

    known_ids = set(image_ids.tolist())
    for label, predictions in (
        ("baseline", baseline_predictions),
        ("candidate", candidate_predictions),
    ):
        unknown = {int(item["image_id"]) for item in predictions} - known_ids
        if unknown:
            raise ValueError(f"{label} predictions contain unknown image IDs: {sorted(unknown)[:5]}")

    baseline_point = evaluator(ground_truth, baseline_predictions)
    candidate_point = evaluator(ground_truth, candidate_predictions)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled_ids = rng.choice(image_ids, size=image_ids.size, replace=True)
        sampled_gt, sampled_baseline, sampled_candidate = _resample_pair(
            ground_truth,
            baseline_predictions,
            candidate_predictions,
            sampled_ids,
        )
        deltas[index] = evaluator(sampled_gt, sampled_candidate) - evaluator(sampled_gt, sampled_baseline)
        if progress_every > 0 and ((index + 1) % progress_every == 0 or index + 1 == replicates):
            print(f"[A2 bootstrap] {index + 1}/{replicates}", flush=True)

    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])
    point_delta = candidate_point - baseline_point
    return {
        "metric": "small_ap",
        "baseline": float(baseline_point),
        "candidate": float(candidate_point),
        "point_delta": float(point_delta),
        "bootstrap_mean_delta": float(deltas.mean()),
        "bootstrap_std": float(deltas.std(ddof=1 if replicates > 1 else 0)),
        "ci_level": 0.95,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "replicates": int(replicates),
        "seed": int(seed),
        "min_improvement": float(min_improvement),
        "passes_point_threshold": bool(point_delta >= min_improvement),
        "ci_excludes_zero": bool(ci_low > 0),
        "p1_passed": bool(point_delta >= min_improvement and ci_low > 0),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the paired-bootstrap CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    """Run paired bootstrap from COCO JSON files."""
    args = build_parser().parse_args()
    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    baseline_predictions = json.loads(args.baseline_predictions.read_text(encoding="utf-8"))
    candidate_predictions = json.loads(args.candidate_predictions.read_text(encoding="utf-8"))
    result = paired_bootstrap(
        ground_truth,
        baseline_predictions,
        candidate_predictions,
        replicates=args.replicates,
        seed=args.seed,
        min_improvement=args.min_improvement,
        progress_every=args.progress_every,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"[A2 bootstrap] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
