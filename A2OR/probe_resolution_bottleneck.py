"""Probe whether input resolution is limiting small-object detection quality.

The primary probe evaluates one frozen checkpoint at multiple image sizes with the
same validation set and COCO area bins.  An optional integrated Sparse SAHI stage
is reported separately because it changes the inference pipeline.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

try:
    from A2OR.compare_aps import IMAGE_SUFFIXES, coco_metrics, validate_checkpoint, yolo_ground_truth
except ModuleNotFoundError:  # Direct execution adds A2OR/, rather than the repository root, to sys.path.
    from compare_aps import IMAGE_SUFFIXES, coco_metrics, validate_checkpoint, yolo_ground_truth

METRICS = (
    ("AP", "AP_all"),
    ("AP50", "AP_50"),
    ("AP75", "AP_75"),
    ("APs", "AP_small"),
    ("APm", "AP_medium"),
    ("APl", "AP_large"),
)


def parse_args() -> argparse.Namespace:
    """Parse resolution-probe arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate one checkpoint at 800/1280 (or custom sizes) using identical COCO area metrics."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Frozen checkpoint, normally baseline best.pt.")
    parser.add_argument("--data", type=Path, required=True, help="Resolved Ultralytics dataset YAML.")
    parser.add_argument("--images", type=Path, required=True, help="Validation images directory.")
    parser.add_argument("--labels", type=Path, required=True, help="Validation YOLO labels directory.")
    parser.add_argument("--imgsz", type=int, nargs="+", default=(800, 1280), help="Image sizes to evaluate.")
    parser.add_argument("--reference-imgsz", type=int, default=800, help="Size used as the delta reference.")
    parser.add_argument("--batch", type=int, default=4, help="Evaluation batch size; lower this if 1280 OOMs.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--aps-gate", type=float, default=0.5, help="APs-point increase treated as a useful signal.")
    parser.add_argument("--output", type=Path, default=None, help="JSON report path.")
    parser.add_argument("--sparse-sahi", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sahi-imgsz", type=int, default=800)
    parser.add_argument("--slice-size", type=int, default=640)
    parser.add_argument("--overlap-ratio", type=float, default=0.2)
    parser.add_argument("--objectness-threshold", type=float, default=0.15)
    parser.add_argument("--conf", type=float, default=0.001, help="Low confidence floor for AP-safe SAHI predictions.")
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Reject incomplete or unsafe probe configurations before loading a model."""
    for name in ("checkpoint", "data", "images", "labels"):
        path = Path(getattr(args, name))
        if not path.exists():
            raise FileNotFoundError(path)
    if args.checkpoint.suffix.lower() != ".pt":
        raise ValueError("--checkpoint must be a .pt file")
    if not args.imgsz or len(set(args.imgsz)) != len(args.imgsz) or any(size < 1 for size in args.imgsz):
        raise ValueError("--imgsz must contain unique positive image sizes")
    if args.reference_imgsz not in args.imgsz:
        raise ValueError("--reference-imgsz must be included in --imgsz")
    if args.batch < 1 or args.workers < 0 or args.max_det < 1:
        raise ValueError("batch and max-det must be positive; workers must be non-negative")
    if args.slice_size < 1 or not 0.0 <= args.overlap_ratio < 1.0:
        raise ValueError("slice-size must be positive and overlap-ratio must be in [0, 1)")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0:
        raise ValueError("conf and iou must be in [0, 1]")


def delta_points(result: dict, reference: dict) -> dict[str, float]:
    """Return COCO maxDets=100 metric deltas in percentage points."""
    current = result["coco_max_dets_100"]
    base = reference["coco_max_dets_100"]
    return {key: (current[key] - base[key]) * 100 for _, key in METRICS}


def resolution_signal(delta: dict[str, float], aps_gate: float) -> dict:
    """Classify the diagnostic signal without treating it as a formal model result."""
    aps_delta = delta["AP_small"]
    ap_delta = delta["AP_all"]
    passed = aps_delta >= aps_gate
    return {
        "aps_gate_points": aps_gate,
        "aps_delta_points": aps_delta,
        "ap_delta_points": ap_delta,
        "passed": passed,
        "interpretation": (
            "resolution-sensitive: higher-resolution features are a plausible bottleneck"
            if passed
            else "weak resolution signal: do not assume feature resolution is the main bottleneck"
        ),
        "scope": "diagnostic only; changing evaluation resolution is not an architecture ablation",
    }


def sparse_sahi_evaluate(args: argparse.Namespace, coco_gt, filename_to_id: dict[str, int]) -> dict:
    """Evaluate the repository's integrated Sparse SAHI path on the validation directory."""
    from ultralytics import YOLO

    model = YOLO(str(args.checkpoint.resolve()))
    predictions: list[dict] = []
    image_paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    started = time.perf_counter()
    for index, image_path in enumerate(image_paths, start=1):
        image_id = filename_to_id[image_path.name]
        result = model.predict(
            source=str(image_path),
            imgsz=args.sahi_imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
            save=False,
            sparse_sahi=True,
            slice_size=args.slice_size,
            overlap_ratio=args.overlap_ratio,
            objectness_threshold=args.objectness_threshold,
            sparse_sahi_fallback=True,
        )[0]
        if result.boxes is not None:
            for xyxy, confidence, class_id in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
            ):
                x1, y1, x2, y2 = map(float, xyxy)
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": int(class_id) + 1,
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(confidence),
                    }
                )
        print(f"Sparse SAHI: {index}/{len(image_paths)}", end="\r", flush=True)
    print(flush=True)
    if not predictions:
        raise ValueError("Sparse SAHI produced no predictions")
    return {
        "checkpoint": str(args.checkpoint.resolve()),
        "prediction_count": len(predictions),
        "wall_seconds": time.perf_counter() - started,
        "coco_max_dets_100": coco_metrics(coco_gt, predictions, 100),
        "dense_max_dets_300": coco_metrics(coco_gt, predictions, 300),
    }


def print_table(stages: dict, deltas: dict, reference_name: str) -> None:
    """Print compact primary metrics and deltas."""
    print("\nCOCO maxDets=100 (percentage points)")
    print(f"reference={reference_name}")
    print("stage                 metric        value       delta")
    for stage_name, result in stages.items():
        stage_delta = deltas.get(stage_name, {})
        for label, key in METRICS:
            value = result["coco_max_dets_100"][key] * 100
            delta = stage_delta.get(key, 0.0)
            print(f"{stage_name:<21} {label:<8} {value:10.3f} {delta:+10.3f}")


def main() -> None:
    """Run the frozen-checkpoint resolution probe and write an auditable report."""
    args = parse_args()
    validate_args(args)
    output = args.output or args.checkpoint.resolve().parent.parent / "resolution_probe.json"
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["output"] = str(output)
    if args.print_config:
        print(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True))
        return

    from faster_coco_eval import COCO

    output.parent.mkdir(parents=True, exist_ok=True)
    dataset, filename_to_id = yolo_ground_truth(args.images.resolve(), args.labels.resolve())
    coco_gt = COCO(dataset, print_function=lambda *_: None)
    stages: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="a2or_resolution_probe_", dir=output.parent) as temporary_directory:
        temporary_root = Path(temporary_directory)
        for size in args.imgsz:
            stage_name = f"imgsz_{size}"
            stage_args = argparse.Namespace(**vars(args))
            stage_args.imgsz = size
            started = time.perf_counter()
            stages[stage_name] = validate_checkpoint(
                stage_name,
                args.checkpoint.resolve(),
                stage_args,
                temporary_root,
                coco_gt,
                filename_to_id,
            )
            stages[stage_name]["wall_seconds"] = time.perf_counter() - started

    reference_name = f"imgsz_{args.reference_imgsz}"
    reference = stages[reference_name]
    deltas = {name: delta_points(result, reference) for name, result in stages.items()}
    signals = {
        name: resolution_signal(deltas[name], args.aps_gate)
        for name in stages
        if name != reference_name
    }

    if args.sparse_sahi:
        stages["sparse_sahi"] = sparse_sahi_evaluate(args, coco_gt, filename_to_id)
        deltas["sparse_sahi"] = delta_points(stages["sparse_sahi"], reference)
        signals["sparse_sahi"] = resolution_signal(deltas["sparse_sahi"], args.aps_gate)

    report = {
        "protocol": {
            "checkpoint": str(args.checkpoint.resolve()),
            "data": str(args.data.resolve()),
            "images": str(args.images.resolve()),
            "labels": str(args.labels.resolve()),
            "validation_images": len(dataset["images"]),
            "ground_truth_boxes": len(dataset["annotations"]),
            "area_ranges_original_pixels": {"small": [0, 1024], "medium": [1024, 9216], "large": [9216, 1e10]},
            "primary": "COCO maxDets=100",
            "supplemental": "VisDrone-dense maxDets=300",
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "max_det_prediction_cap": args.max_det,
        },
        "reference_stage": reference_name,
        "stages": stages,
        "delta_vs_reference_points": deltas,
        "signals": signals,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_table(stages, deltas, reference_name)
    for name, signal in signals.items():
        print(f"\n{name}: {signal['interpretation']} (delta APs={signal['aps_delta_points']:+.3f} points)")
    print(f"\nSaved auditable probe report to {output.resolve()}")


if __name__ == "__main__":
    main()
