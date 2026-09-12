#!/usr/bin/env python3
"""Build COCO ground truth from YOLO labels and report size-binned AP for a detection split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths


def resolve_split_images(split_source: str | list[str]) -> list[Path]:
    """Resolve a dataset split expressed as image directories, files, or image-list text files."""
    sources = [split_source] if isinstance(split_source, (str, Path)) else split_source
    images: list[Path] = []
    for source in sources:
        path = Path(source)
        if path.is_dir():
            images.extend(p for p in path.rglob("*") if p.is_file() and p.suffix[1:].lower() in IMG_FORMATS)
        elif path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                item = line.strip()
                if item:
                    images.append((path.parent / item).resolve() if item.startswith("./") else Path(item))
        elif path.is_file() and path.suffix[1:].lower() in IMG_FORMATS:
            images.append(path)
        else:
            raise FileNotFoundError(f"Unable to resolve dataset split source: {path}")
    return sorted(images)


def build_coco_ground_truth(image_files: list[Path], names: dict[int, str] | list[str]) -> dict[str, Any]:
    """Convert YOLO normalized detection labels into an in-memory COCO ground-truth dictionary."""
    if not image_files:
        raise ValueError("Evaluation split contains no images")
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    label_files = img2label_paths([str(path) for path in image_files])
    images, annotations = [], []
    annotation_id = 1
    for image_path, label_path in zip(image_files, label_files):
        with Image.open(image_path) as image:
            width, height = image.size
        stem = image_path.stem
        image_id: int | str = int(stem) if stem.isnumeric() else stem
        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        label_path = Path(label_path)
        if not label_path.exists():
            raise FileNotFoundError(f"Missing evaluation label: {label_path}; use an empty file for a background image")
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError(f"Expected 5 YOLO fields in {label_path}:{line_number}, got {len(fields)}")
            class_id, x_center, y_center, box_width, box_height = map(float, fields)
            if not np.isfinite([class_id, x_center, y_center, box_width, box_height]).all():
                raise ValueError(f"Non-finite YOLO label in {label_path}:{line_number}")
            if not class_id.is_integer() or not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                raise ValueError(f"Invalid class or normalized center in {label_path}:{line_number}")
            if not (0 < box_width <= 1 and 0 < box_height <= 1):
                raise ValueError(f"Invalid normalized box size in {label_path}:{line_number}")
            class_id = int(class_id)
            if class_id not in names:
                raise ValueError(f"Class {class_id} in {label_path}:{line_number} is absent from dataset names")
            box_width *= width
            box_height *= height
            x = x_center * width - box_width / 2
            y = y_center * height - box_height / 2
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": [x, y, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    categories = [{"id": class_id + 1, "name": name} for class_id, name in sorted(names.items())]
    return {
        "info": {"description": "YOLO labels converted for STAL scale-aware evaluation"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def _mean_valid(values: np.ndarray) -> float:
    """Average valid COCO precision/recall entries, preserving -1 for unavailable slices."""
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else -1.0


def _coco_metric(evaluator, *, area: str = "all", iou: float | None = None, recall: bool = False) -> float:
    """Read a COCO metric slice at maxDets=500 from accumulated precision or recall tensors."""
    params = evaluator.params
    area_index = params.areaRngLbl.index(area)
    max_det_index = params.maxDets.index(500)
    if recall:
        return _mean_valid(evaluator.eval["recall"][:, :, area_index, max_det_index])
    precision = evaluator.eval["precision"][:, :, :, area_index, max_det_index]
    if iou is not None:
        matches = np.flatnonzero(np.isclose(params.iouThrs, iou))
        if not matches.size:
            raise ValueError(f"IoU threshold {iou} is absent from evaluator parameters")
        precision = precision[matches]
    return _mean_valid(precision)


def evaluate_scale_ap(annotation_path: Path, prediction_path: Path) -> dict[str, float]:
    """Evaluate the project's COCO-style size metrics with the VisDrone maxDets=500 convention."""
    try:
        from faster_coco_eval import COCO, COCOeval_faster
    except ImportError as exc:
        raise ImportError("Install faster-coco-eval>=1.6.7 before running scale-aware evaluation") from exc

    ground_truth = json.loads(annotation_path.read_text(encoding="utf-8"))
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    category_ids = {category["id"] for category in ground_truth["categories"]}
    if len(category_ids) != len(ground_truth["categories"]):
        raise ValueError("Duplicate category IDs in ground truth")
    annotation_ids = set()
    for record in ground_truth["annotations"]:
        if record["id"] in annotation_ids:
            raise ValueError(f"Duplicate annotation ID: {record['id']}")
        annotation_ids.add(record["id"])
        if record["category_id"] not in category_ids:
            raise ValueError(f"Ground truth refers to unknown category ID: {record['category_id']}")
        bbox = record["bbox"]
        if len(bbox) != 4 or not np.isfinite([*bbox, record["area"]]).all() or min(bbox[2:]) <= 0:
            raise ValueError("Ground truth must have a finite area and a finite positive-size bbox")
        if not np.isclose(record["area"], bbox[2] * bbox[3], rtol=1e-6, atol=1e-8):
            raise ValueError("Project size evaluation requires GT area equal to bbox width * height")
    for record in predictions:
        if record["category_id"] not in category_ids:
            raise ValueError(f"Prediction refers to unknown category ID: {record['category_id']}")
        bbox = record["bbox"]
        score = record["score"]
        if len(bbox) != 4 or not np.isfinite([*bbox, score]).all() or min(bbox[2:]) <= 0 or not 0 <= score <= 1:
            raise ValueError("Prediction must have a score in [0, 1] and a finite positive-size bbox")
    # VisDrone stems contain underscores; the C++ evaluator requires integer image IDs.
    # Remap both sides together, preserving the source JSON files for traceability.
    image_ids = {image["id"]: index for index, image in enumerate(ground_truth["images"], start=1)}
    if len(image_ids) != len(ground_truth["images"]):
        raise ValueError("Duplicate image IDs in ground truth")
    for record in ground_truth["images"]:
        record["id"] = image_ids[record["id"]]
    for record in ground_truth["annotations"]:
        if record["image_id"] not in image_ids:
            raise ValueError(f"Ground truth refers to unknown image ID: {record['image_id']}")
        record["image_id"] = image_ids[record["image_id"]]
    for record in predictions:
        if record["image_id"] not in image_ids:
            raise ValueError(f"Prediction refers to unknown image ID: {record['image_id']}")
        record["image_id"] = image_ids[record["image_id"]]
    annotation_api = COCO()
    annotation_api.dataset = ground_truth
    annotation_api.createIndex()
    if predictions:
        prediction_api = annotation_api.loadRes(predictions)
    else:
        # loadRes implementations may index the first detection; an empty run is valid evaluation input.
        prediction_api = COCO()
        prediction_api.dataset = {**ground_truth, "annotations": []}
        prediction_api.createIndex()
    evaluator = COCOeval_faster(annotation_api, prediction_api, iouType="bbox")
    evaluator.params.imgIds = [image["id"] for image in annotation_api.dataset["images"]]
    evaluator.params.maxDets = [1, 10, 500]
    # COCO uses inclusive upper bounds; enforce the project's disjoint half-open bins.
    evaluator.params.areaRng = [
        [0, 1e10],
        [0, np.nextafter(float(32**2), -np.inf)],
        [32**2, np.nextafter(float(96**2), -np.inf)],
        [96**2, 1e10],
    ]
    evaluator.evaluate()
    evaluator.accumulate()
    return {
        "AP": _coco_metric(evaluator),
        "AP50": _coco_metric(evaluator, iou=0.50),
        "AP75": _coco_metric(evaluator, iou=0.75),
        "APs": _coco_metric(evaluator, area="small"),
        "APm": _coco_metric(evaluator, area="medium"),
        "APl": _coco_metric(evaluator, area="large"),
        "AP50s": _coco_metric(evaluator, area="small", iou=0.50),
        "AR500": _coco_metric(evaluator, recall=True),
        "ARs500": _coco_metric(evaluator, area="small", recall=True),
        "ARm500": _coco_metric(evaluator, area="medium", recall=True),
        "ARl500": _coco_metric(evaluator, area="large", recall=True),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Dataset YAML path")
    parser.add_argument("--predictions", type=Path, required=True, help="Ultralytics predictions.json path")
    parser.add_argument("--split", default="val", help="Dataset split key, usually val")
    parser.add_argument("--annotations-out", type=Path, required=True, help="Generated COCO annotation JSON path")
    parser.add_argument("--metrics-out", type=Path, required=True, help="Scale AP metrics JSON path")
    return parser.parse_args()


def main() -> None:
    """Build annotations, evaluate predictions, and save reproducible metric artifacts."""
    args = parse_args()
    dataset = check_det_dataset(args.data, autodownload=False, split=args.split)
    if args.split not in dataset:
        raise KeyError(f"Split '{args.split}' is not defined in {args.data}")
    image_files = resolve_split_images(dataset[args.split])
    ground_truth = build_coco_ground_truth(image_files, dataset["names"])
    args.annotations_out.parent.mkdir(parents=True, exist_ok=True)
    args.annotations_out.write_text(json.dumps(ground_truth, ensure_ascii=False), encoding="utf-8")
    metrics = evaluate_scale_ap(args.annotations_out, args.predictions)
    metrics.update(
        {
            "images": len(ground_truth["images"]),
            "annotations": len(ground_truth["annotations"]),
            "max_dets": 500,
            "metric_units": "fraction [0,1]; -1 means no evaluable GT; 0.01 equals 1 absolute AP point",
            "iou_thresholds": "0.50:0.05:0.95",
            "official_visdrone_metrics": False,
            "protocol": "Project COCO-style supplemental analysis; these area bins are not VisDrone official bins",
            "area_definition": "Original-image GT bbox area: small < 32^2, medium [32^2, 96^2), large >= 96^2 pixels",
        }
    )
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
