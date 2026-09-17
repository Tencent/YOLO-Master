#!/usr/bin/env python3
"""Evaluate a VisDrone YOLO checkpoint with COCO-style scale AP metrics.

The regular Ultralytics validator reports aggregate mAP for a custom YOLO
dataset. This entry point keeps that validation path intact, asks it to save
predictions, converts the VisDrone YOLO labels to COCO JSON, and then runs
``faster-coco-eval`` to obtain AP_small/AP_medium/AP_large.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SMALL_MAX_AREA = 32**2
LARGE_MIN_AREA = 96**2
# faster-coco-eval follows COCO's inclusive upper-area comparison.  The A2
# protocol uses half-open scale buckets, so the upper bounds must be the next
# representable float below the next bucket boundary.
_AREA_MAX = 1e10


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_dataset(data_yaml: Path, split: str) -> tuple[dict[str, Any], list[Path], Path]:
    """Load a YOLO dataset YAML and enumerate images for one split."""
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    dataset_root = _resolve_path(data.get("path", "."), data_yaml.parent).resolve()
    split_value = data.get(split)
    if split_value is None:
        raise ValueError(f"dataset YAML has no '{split}' entry: {data_yaml}")
    if isinstance(split_value, list):
        sources = [_resolve_path(item, dataset_root) for item in split_value]
    else:
        sources = [_resolve_path(split_value, dataset_root)]

    images: list[Path] = []
    for source in sources:
        if source.is_dir():
            images.extend(path for path in sorted(source.rglob("*")) if path.suffix.lower() in IMAGE_SUFFIXES)
        elif source.is_file():
            for raw in source.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if raw and not raw.startswith("#"):
                    image = Path(raw)
                    images.append(image if image.is_absolute() else source.parent / image)
        else:
            raise FileNotFoundError(f"validation source does not exist: {source}")
    if not images:
        raise RuntimeError(f"no validation images found under {sources}")
    return data, images, dataset_root


def _label_path(image: Path) -> Path:
    """Map ``.../images/<split>/<name>.jpg`` to ``.../labels/<split>/<name>.txt``."""
    parts = list(image.parts)
    image_index = next((i for i, part in enumerate(parts) if part.lower() == "images"), None)
    if image_index is not None:
        parts[image_index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def _class_names(data: dict[str, Any]) -> list[str]:
    names = data.get("names", {})
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    return [str(name) for name in names]


def _build_coco_ground_truth(data: dict[str, Any], images: list[Path], output: Path) -> dict[str, int]:
    """Convert YOLO labels into a COCO detection annotation file."""
    names = _class_names(data)
    categories = [{"id": index + 1, "name": name, "supercategory": "object"} for index, name in enumerate(names)]
    coco_images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    file_to_id: dict[str, int] = {}
    annotation_id = 1

    for image_id, image in enumerate(images, start=1):
        with Image.open(image) as image_file:
            width, height = image_file.size
        file_name = image.name
        if file_name in file_to_id:
            raise ValueError(f"duplicate validation image filename: {file_name}")
        file_to_id[file_name] = image_id
        coco_images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})
        label_file = _label_path(image)
        if not label_file.exists():
            continue
        for line_number, raw in enumerate(label_file.read_text(encoding="utf-8").splitlines(), start=1):
            fields = raw.split()
            if len(fields) < 5:
                continue
            cls, xc, yc, norm_w, norm_h = fields[:5]
            class_id = int(float(cls))
            if class_id < 0 or class_id >= len(names):
                raise ValueError(f"invalid class {class_id} in {label_file}:{line_number}")
            box_w = float(norm_w) * width
            box_h = float(norm_h) * height
            box_x = (float(xc) * width) - box_w / 2
            box_y = (float(yc) * height) - box_h / 2
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": [box_x, box_y, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    payload = {
        "info": {"description": "VisDrone YOLO validation split"},
        "licenses": [],
        "images": coco_images,
        "annotations": annotations,
        "categories": categories,
    }
    output.write_text(json.dumps(payload), encoding="utf-8")
    return file_to_id


def _normalise_predictions(predictions: Path, file_to_id: dict[str, int], output: Path) -> None:
    """Replace string/image-stem IDs emitted by Ultralytics with COCO IDs."""
    rows = json.loads(predictions.read_text(encoding="utf-8"))
    normalised = []
    for row in rows:
        file_name = row.get("file_name")
        image_id = file_to_id.get(file_name)
        if image_id is None:
            raise KeyError(f"prediction references an unknown validation image: {file_name}")
        normalised.append(
            {
                "image_id": image_id,
                "category_id": int(row["category_id"]),
                "bbox": [float(value) for value in row["bbox"]],
                "score": float(row["score"]),
            }
        )
    output.write_text(json.dumps(normalised), encoding="utf-8")


def _set_dense_esmoe(model: Any) -> int:
    count = 0
    for module in model.model.modules():
        if module.__class__.__name__ == "ES_MOE":
            module.use_sparse_inference = False
            count += 1
    return count


def _mean_valid(values: np.ndarray) -> float:
    """Average COCO values while ignoring its -1 sentinel."""
    values = np.asarray(values, dtype=np.float64)
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else float("nan")


def _find_iou_index(iou_thrs: np.ndarray, target: float) -> int:
    """Find an IoU threshold index, tolerating floating-point representation."""
    matches = np.flatnonzero(np.isclose(iou_thrs, target))
    if not len(matches):
        raise ValueError(f"evaluator does not contain IoU={target:g}")
    return int(matches[0])


def _qa_area_ranges() -> list[list[float]]:
    """Return COCO area ranges matching the A2 half-open scale definitions."""
    small_upper = float(np.nextafter(float(SMALL_MAX_AREA), -np.inf))
    medium_upper = float(np.nextafter(float(LARGE_MIN_AREA), -np.inf))
    return [
        [0.0, _AREA_MAX],
        [0.0, small_upper],
        [float(SMALL_MAX_AREA), medium_upper],
        [float(LARGE_MIN_AREA), _AREA_MAX],
    ]


def _evaluate_coco(gt_json: Path, pred_json: Path, image_ids: list[int], *, max_det: int = 500) -> dict[str, float]:
    try:
        from faster_coco_eval import COCO, COCOeval_faster
    except ImportError as exc:
        raise RuntimeError(
            "faster-coco-eval is required; install it with `pip install faster-coco-eval>=1.6.7`"
        ) from exc

    anno = COCO(gt_json)
    pred = anno.loadRes(pred_json)
    evaluator = COCOeval_faster(anno, pred, iouType="bbox", lvis_style=False, print_function=print)
    evaluator.params.imgIds = image_ids
    evaluator.params.maxDets = [1, 10, 100, max_det]
    evaluator.params.areaRng = _qa_area_ranges()
    evaluator.params.areaRngLbl = ["all", "small", "medium", "large"]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    # faster-coco-eval's stats_as_dict is tied to the default three maxDets
    # slots. Read the accumulated precision/recall tensors directly so the
    # fourth 500-detection slot is measured rather than silently ignored.
    evaluation = evaluator.eval
    precision = np.asarray(evaluation["precision"])
    recall = np.asarray(evaluation["recall"])
    iou_thrs = np.asarray(evaluator.params.iouThrs)
    max_dets = list(evaluator.params.maxDets)
    max_index = {int(value): index for index, value in enumerate(max_dets)}
    if max_det not in max_index:
        raise ValueError(f"max_det={max_det} was not evaluated; maxDets={max_dets}")
    max_index_500 = max_index[max_det]
    index_100 = max_index.get(100, max_index_500)
    area_labels = list(getattr(evaluator.params, "areaRngLbl", ("all", "small", "medium", "large")))
    area_index = {label: index for index, label in enumerate(area_labels)}

    def ap(area: str, max_index_value: int, iou_index: int | None = None) -> float:
        values = precision[:, :, :, area_index[area], max_index_value]
        if iou_index is not None:
            values = values[iou_index : iou_index + 1]
        return _mean_valid(values)

    def ar(area: str, max_index_value: int) -> float:
        return _mean_valid(recall[:, :, area_index[area], max_index_value])

    ap50_index = _find_iou_index(iou_thrs, 0.50)
    ap75_index = _find_iou_index(iou_thrs, 0.75)
    metrics = {
        "mAP50": ap("all", max_index_500, ap50_index),
        "mAP50-95": ap("all", max_index_500),
        "small_ap": ap("small", max_index_500),
        "medium_ap": ap("medium", max_index_500),
        "large_ap": ap("large", max_index_500),
        "AP": ap("all", max_index_500),
        "AP50": ap("all", max_index_500, ap50_index),
        "AP75": ap("all", max_index_500, ap75_index),
        "APs": ap("small", max_index_500),
        "APm": ap("medium", max_index_500),
        "APl": ap("large", max_index_500),
        "AR@1": ar("all", max_index[1]),
        "AR@10": ar("all", max_index[10]),
        "AR@100": ar("all", index_100),
        "AR@500": ar("all", max_index_500),
        "ARs@500": ar("small", max_index_500),
        "ARm@500": ar("medium", max_index_500),
        "ARl@500": ar("large", max_index_500),
    }
    print(
        "[scale-aware] QA metrics "
        + " ".join(f"{key}={value:.6f}" for key, value in metrics.items())
        + f" maxDets={max_dets}"
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Checkpoint to evaluate, normally weights/best.pt.")
    parser.add_argument("--data", type=Path, required=True, help="VisDrone YOLO dataset YAML.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-det", type=int, default=500, help="Maximum detections per image for QA metrics.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=Path, default=Path("runs/a2/scale_eval"))
    parser.add_argument("--name", default="EsMoE-N_STAL_best")
    parser.add_argument(
        "--sparse-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep sparse ES-MoE inference (diagnostic only); dense evaluation is the default.",
    )
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", default="yolo-master-a2-0831")
    parser.add_argument("--wandb-name", default="A2_scale_eval")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_yaml = args.data.resolve()
    weights = args.weights.resolve()
    output_dir = (args.project if args.project.is_absolute() else ROOT / args.project) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    data, images, _ = _load_dataset(data_yaml, args.split)
    gt_json = output_dir / "visdrone_scale_gt.json"
    pred_json = output_dir / "predictions.json"
    normalised_pred_json = output_dir / "visdrone_scale_predictions.json"
    file_to_id = _build_coco_ground_truth(data, images, gt_json)
    # Never accept a prediction file left by an earlier run with the same name.
    pred_json.unlink(missing_ok=True)

    from ultralytics import YOLO

    model = YOLO(str(weights))
    dense_modules = 0 if args.sparse_eval else _set_dense_esmoe(model)
    print(f"[scale-aware] weights={weights} data={data_yaml} images={len(images)} dense_esmoe={dense_modules}")
    model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        max_det=args.max_det,
        save_json=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    if not pred_json.exists():
        raise FileNotFoundError(f"Ultralytics did not write predictions.json: {pred_json}")
    _normalise_predictions(pred_json, file_to_id, normalised_pred_json)
    metrics = _evaluate_coco(gt_json, normalised_pred_json, list(file_to_id.values()), max_det=args.max_det)
    result = {
        "weights": str(weights),
        "data": str(data_yaml),
        "split": args.split,
        "images": len(images),
        "gt_instances": sum(1 for _ in json.loads(gt_json.read_text(encoding="utf-8")).get("annotations", [])),
        "dense_esmoe_modules": dense_modules,
        "max_det": args.max_det,
        "area_thresholds": {"small_max": SMALL_MAX_AREA, "large_min": LARGE_MIN_AREA},
        **metrics,
    }
    result_json = output_dir / "scale_metrics.json"
    result_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"[scale-aware] wrote {result_json}")

    if args.wandb:
        import wandb

        with wandb.init(project=args.wandb_project, name=args.wandb_name, job_type="scale-eval") as run:
            run.config.update(
                {
                    "weights": str(weights),
                    "data": str(data_yaml),
                    "split": args.split,
                    "imgsz": args.imgsz,
                    "batch": args.batch,
                    "max_det": args.max_det,
                    "small_definition": "area < 32^2 on original validation image bbox",
                    "medium_definition": "32^2 <= area < 96^2 on original validation image bbox",
                    "large_definition": "area >= 96^2 on original validation image bbox",
                }
            )
            run.log({f"scale/{key}": value for key, value in metrics.items()})
            run.summary.update(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
