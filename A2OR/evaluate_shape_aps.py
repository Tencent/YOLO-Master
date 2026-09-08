"""Evaluate small-object APs and hit rate by ground-truth box shape.

This is an independent entry point for shape-wise diagnosis.  It runs one
Ultralytics validation pass for a user-supplied checkpoint, then evaluates
the same predictions against one COCO ground-truth subset per width x height
bin.  The primary metric is COCO APs (area range <= 32^2 pixels); hit_rate@0.5
is a simpler, actionable diagnostic based on one-to-one class-aware matches.

Example:
    conda run -n yolo_master python A2OR/evaluate_shape_aps.py \
        --model A2OR/runs/baseline_b4_nbs64_120e/weights/best.pt \
        --data A2OR/visdrone_full.yaml --imgsz 800 --batch 4 --device 0

For a bucket boundary that exposes 8 x 24 separately, use for example:
    --width-bins 0,8,16,32,64,inf --height-bins 0,8,16,24,32,64,inf
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path
from typing import Iterable

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VISDRONE_NAMES = (
    "pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"
)
SMALL_MAX_AREA = 32 * 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VisDrone small-object APs by width x height shape bins.")
    parser.add_argument("--model", type=Path, required=True, help="Checkpoint .pt to evaluate.")
    parser.add_argument("--data", type=Path, required=True, help="Dataset YAML used by Ultralytics validation.")
    parser.add_argument("--images", type=Path, default=None, help="Validation image directory; inferred from --data when omitted.")
    parser.add_argument("--labels", type=Path, default=None, help="Validation YOLO label directory; inferred from --images when omitted.")
    parser.add_argument("--output", type=Path, default=Path("A2OR/reports/shape_aps.json"), help="JSON report path.")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--conf", type=float, default=0.0, help="Confidence cutoff for hit_rate only; APs keeps all saved predictions.")
    parser.add_argument("--hit-iou", type=float, default=0.5, help="IoU threshold for hit_rate, default 0.5.")
    parser.add_argument("--width-bins", default="0,8,16,32,64,inf", help="Pixel boundaries, e.g. 0,8,16,24,32,64,inf.")
    parser.add_argument("--height-bins", default="0,8,16,32,64,inf", help="Pixel boundaries, e.g. 0,8,16,24,32,64,inf.")
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if args.imgsz < 1 or args.batch < 1 or args.workers < 0 or args.max_det < 1:
        raise SystemExit("imgsz, batch, and max-det must be positive; workers must be non-negative")
    if not 0 <= args.conf <= 1 or not 0 <= args.hit_iou <= 1:
        raise SystemExit("conf and hit-iou must be in [0, 1]")
    parse_bins(args.width_bins)
    parse_bins(args.height_bins)
    return args


def parse_bins(value: str) -> list[float]:
    try:
        bins = [math.inf if token.strip().lower() in {"inf", "infinity", "+inf"} else float(token) for token in value.split(",")]
    except ValueError as exc:
        raise SystemExit(f"Invalid bin boundaries: {value}") from exc
    if len(bins) < 2 or bins[0] != 0 or any(a >= b for a, b in zip(bins, bins[1:])) or bins[-1] != math.inf:
        raise SystemExit("Bins must start at 0, end at inf, and be strictly increasing")
    return bins


def infer_paths(data_path: Path, images: Path | None, labels: Path | None) -> tuple[Path, Path]:
    if images is not None and labels is not None:
        return images, labels
    try:
        import yaml

        data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
        root = Path(str(data.get("path", "")))
        if not root.is_absolute():
            root = (data_path.parent / root).resolve()
        val = Path(str(data["val"]))
        images = images or (root / val)
    except (KeyError, TypeError, OSError, ValueError) as exc:
        raise SystemExit("Could not infer validation images; provide --images and --labels explicitly") from exc
    labels = labels or images.parent.parent / "labels" / images.name
    return images, labels


def load_ground_truth(images_dir: Path, labels_dir: Path) -> tuple[dict, dict[str, int], list[dict]]:
    from PIL import Image

    images_dir, labels_dir = images_dir.resolve(), labels_dir.resolve()
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")
    images, annotations, rows = [], [], []
    filename_to_id: dict[str, int] = {}
    annotation_id = 1
    for image_id, image_path in enumerate(image_paths, 1):
        with Image.open(image_path) as image:
            width, height = image.size
        filename_to_id[image_path.name] = image_id
        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"Expected 5 fields in {label_path}:{line_no}")
            cls, xc, yc, bw, bh = map(float, values)
            class_id = int(cls)
            if cls != class_id or not 0 <= class_id < len(VISDRONE_NAMES):
                raise ValueError(f"Invalid class ID {cls} in {label_path}:{line_no}")
            x1, y1 = max(0.0, (xc - bw / 2) * width), max(0.0, (yc - bh / 2) * height)
            x2, y2 = min(float(width), (xc + bw / 2) * width), min(float(height), (yc + bh / 2) * height)
            box_w, box_h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if box_w == 0 or box_h == 0:
                continue
            area = box_w * box_h
            ann = {"id": annotation_id, "image_id": image_id, "category_id": class_id + 1,
                   "bbox": [x1, y1, box_w, box_h], "area": area, "iscrowd": 0, "segmentation": []}
            annotations.append(ann)
            rows.append({"annotation_id": annotation_id, "image_id": image_id, "file_name": image_path.name,
                         "category_id": class_id + 1, "class_name": VISDRONE_NAMES[class_id], "width_px": box_w,
                         "height_px": box_h, "area_px": area, "small": area <= SMALL_MAX_AREA})
            annotation_id += 1
    dataset = {"info": {"description": "VisDrone validation labels converted from YOLO format"}, "images": images,
               "annotations": annotations, "categories": [{"id": i + 1, "name": n} for i, n in enumerate(VISDRONE_NAMES)]}
    return dataset, filename_to_id, rows


def remap_predictions(path: Path, filename_to_id: dict[str, int]) -> list[dict]:
    values = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for pred in values:
        image_id = filename_to_id.get(Path(pred.get("file_name", "")).name)
        if image_id is None:
            continue
        result.append({"image_id": image_id, "category_id": int(pred["category_id"]),
                       "bbox": [float(v) for v in pred["bbox"]], "score": float(pred["score"])})
    if not result:
        raise ValueError(f"No predictions matched validation filenames in {path}")
    return result


def subset_coco(dataset: dict, annotations: Iterable[dict]) -> dict:
    """Build a shape subset while ignoring other GT objects in the same images.

    COCOeval does not honor a custom ``ignore`` field in this faster backend;
    ``iscrowd=1`` is its supported ignore-region mechanism.  Keeping other GT
    objects as ignored regions prevents their valid predictions from becoming
    false positives for the selected shape.
    """
    selected = list(annotations)
    selected_ids = {ann["id"] for ann in selected}
    image_ids = sorted({ann["image_id"] for ann in selected})
    subset_annotations = []
    for ann in dataset["annotations"]:
        if ann["image_id"] not in image_ids:
            continue
        copied = dict(ann)
        copied["bbox"] = list(ann["bbox"])
        copied["iscrowd"] = 0 if ann["id"] in selected_ids else 1
        subset_annotations.append(copied)
    return {"info": dataset["info"], "images": [im for im in dataset["images"] if im["id"] in image_ids],
            "annotations": subset_annotations, "categories": dataset["categories"]}


def coco_aps(dataset: dict, predictions: list[dict], max_det: int) -> dict[str, float | None]:
    from faster_coco_eval import COCO, COCOeval_faster

    if not dataset["annotations"]:
        return {"APs": None, "AP50s": None, "ARs": None, "AP": None}
    coco_gt = COCO(dataset, print_function=lambda *_: None)
    # A shape subset contains only images with at least one selected GT.  The
    # result loader requires every prediction image_id to belong to that set.
    image_ids = set(coco_gt.getImgIds())
    subset_predictions = [prediction for prediction in predictions if prediction["image_id"] in image_ids]
    coco_dt = coco_gt.loadRes(subset_predictions)
    evaluator = COCOeval_faster(coco_gt, coco_dt, iouType="bbox", print_function=lambda *_: None)
    evaluator.params.imgIds = sorted(coco_gt.getImgIds())
    evaluator.params.maxDets = [1, 10, max_det]
    # Keep the four standard COCO labels because faster_coco_eval.summarize()
    # expects all/ small/ medium/ large to exist.  The selected GT itself is
    # already restricted to small objects, so AP_small is the desired value.
    evaluator.params.areaRng = [[0, 1e10], [0, SMALL_MAX_AREA], [SMALL_MAX_AREA, 96 * 96], [96 * 96, 1e10]]
    evaluator.params.areaRngLbl = ["all", "small", "medium", "large"]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    values = {key: float(value) for key, value in evaluator.stats_as_dict.items()}
    return {"APs": values.get("AP_small", values.get("AP_all")),
            "AP50s": values.get("AP50_small", values.get("AP_50", values.get("AP50"))),
            "ARs": values.get("AR_small", values.get("AR_all")), "AP": values.get("AP_all")}


def xywh_iou(left: list[float], right: list[float]) -> float:
    lx1, ly1, lw, lh = left
    rx1, ry1, rw, rh = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx1 + lw, rx1 + rw), min(ly1 + lh, ry1 + rh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = lw * lh + rw * rh - inter
    return inter / union if union > 0 else 0.0


def hit_rate(annotations: list[dict], predictions: list[dict], threshold: float, conf: float) -> float | None:
    if not annotations:
        return None
    by_image: dict[int, list[dict]] = {}
    for pred in predictions:
        if pred["score"] >= conf:
            by_image.setdefault(pred["image_id"], []).append(pred)
    hits = 0
    groups: dict[tuple[int, int], list[dict]] = {}
    for ann in annotations:
        groups.setdefault((ann["image_id"], ann["category_id"]), []).append(ann)
    for (image_id, category_id), group in groups.items():
        candidates = sorted((p for p in by_image.get(image_id, []) if p["category_id"] == category_id),
                            key=lambda p: p["score"], reverse=True)
        used: set[int] = set()
        for ann in sorted(group, key=lambda item: item["area"], reverse=True):
            matches = [(xywh_iou(ann["bbox"], pred["bbox"]), index) for index, pred in enumerate(candidates)
                       if index not in used and xywh_iou(ann["bbox"], pred["bbox"]) >= threshold]
            if matches:
                _, index = max(matches)
                used.add(index)
                hits += 1
    return hits / len(annotations)


def bucket_index(value: float, bins: list[float]) -> int:
    return next(i for i, (lo, hi) in enumerate(zip(bins, bins[1:])) if lo <= value < hi)


def bin_name(lo: float, hi: float) -> str:
    def fmt(value: float) -> str:
        return "inf" if math.isinf(value) else str(int(value))
    return f"{fmt(lo)}_{fmt(hi)}px"


def main() -> None:
    args = parse_args()
    images_dir, labels_dir = infer_paths(args.data, args.images, args.labels)
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Validation paths not found: images={images_dir}, labels={labels_dir}")
    width_bins, height_bins = parse_bins(args.width_bins), parse_bins(args.height_bins)
    dataset, filename_to_id, rows = load_ground_truth(images_dir, labels_dir)
    small_rows = [row for row in rows if row["small"]]
    from ultralytics import YOLO

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="shape_aps_", dir=args.output.parent) as temp_dir:
        print(f"Validating checkpoint: {args.model.resolve()}", flush=True)
        validation = YOLO(str(args.model)).val(data=str(args.data), split="val", imgsz=args.imgsz, batch=args.batch,
            device=args.device, workers=args.workers, plots=False, save_json=True, project=temp_dir,
            name="predictions", exist_ok=True, verbose=False, max_det=args.max_det)
        predictions = remap_predictions(Path(validation.save_dir) / "predictions.json", filename_to_id)

    gt_by_id = {ann["id"]: ann for ann in dataset["annotations"]}
    small_annotations = [gt_by_id[row["annotation_id"]] for row in small_rows]
    results = []
    for wi, (wlo, whi) in enumerate(zip(width_bins, width_bins[1:])):
        for hi, (hlo, hhi) in enumerate(zip(height_bins, height_bins[1:])):
            selected_rows = [row for row in small_rows if bucket_index(row["width_px"], width_bins) == wi and bucket_index(row["height_px"], height_bins) == hi]
            selected_annotations = [gt_by_id[row["annotation_id"]] for row in selected_rows]
            metrics = coco_aps(subset_coco(dataset, selected_annotations), predictions, args.max_det)
            results.append({"shape": f"{bin_name(wlo, whi)}_x_{bin_name(hlo, hhi)}", "width_px": [wlo, whi],
                           "height_px": [hlo, hhi], "gt_boxes": len(selected_annotations), **metrics,
                           "hit_rate@0.5": hit_rate(selected_annotations, predictions, args.hit_iou, args.conf)})

    for result in results:
        for key in ("APs", "AP50s", "ARs", "hit_rate@0.5"):
            result[f"{key}_points"] = None if result[key] is None else result[key] * 100

    json_width_bins = [None if math.isinf(value) else value for value in width_bins]
    json_height_bins = [None if math.isinf(value) else value for value in height_bins]
    report = {"protocol": {"model": str(args.model.resolve()), "data": str(args.data.resolve()),
                            "images": str(images_dir.resolve()), "labels": str(labels_dir.resolve()),
                            "area": "small: width*height <= 32^2 original-image pixels", "primary_metric": "COCO APs",
                            "coco_max_dets": args.max_det, "imgsz": args.imgsz, "batch": args.batch,
                            "device": args.device, "workers": args.workers, "hit_iou": args.hit_iou,
                            "hit_conf": args.conf, "width_bins": json_width_bins, "height_bins": json_height_bins},
             "summary": {"validation_images": len(dataset["images"]), "all_gt_boxes": len(rows),
                         "small_gt_boxes": len(small_annotations)}, "results": results}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["shape", "gt_boxes", "APs", "APs_points", "AP50s", "ARs", "hit_rate@0.5", "hit_rate@0.5_points"])
        writer.writeheader()
        writer.writerows({key: result[key] for key in writer.fieldnames} for result in results)
    print("\nshape                                  GT    APs(pt)  AP50s(pt)  ARs(pt)  hit@0.5(%)")
    for result in sorted(results, key=lambda item: (item["APs"] is None, item["APs"] or 0)):
        print(f"{result['shape']:<38} {result['gt_boxes']:>4} {result['APs_points'] if result['APs_points'] is not None else float('nan'):>9.4f}"
              f" {result['AP50s_points'] if result['AP50s_points'] is not None else float('nan'):>10.4f}"
              f" {result['ARs_points'] if result['ARs_points'] is not None else float('nan'):>9.4f}"
              f" {result['hit_rate@0.5_points'] if result['hit_rate@0.5_points'] is not None else float('nan'):>11.4f}")
    print(f"\nSaved JSON: {args.output.resolve()}")
    print(f"Saved CSV:  {csv_path.resolve()}")


if __name__ == "__main__":
    main()
