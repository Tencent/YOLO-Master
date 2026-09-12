#!/usr/bin/env python3
"""Export deterministic VisDrone TXT predictions; official MATLAB scores them."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

from scripts.d1.artifacts import encoded, file_sha, immutable

TOOLKIT_COMMIT = "005445782213e20cb91bc50a597db3dd949e749a"
METRICS = ("AP_all", "AP_50", "AP_75", "AR_1", "AR_10", "AR_100", "AR_500")


def export_predictions(predictions, image_ids, output, *, conf=0.001, max_det=500):
    """Input bbox is ORIGINAL-image xywh, category_id is zero-based 0..9."""
    output = Path(output)
    if len(image_ids) != len(set(image_ids)) or not image_ids:
        raise ValueError("Image list must be nonempty and unique")
    if conf != 0.001 or max_det != 500:
        raise ValueError("E2 requires the registered export thresholds")
    groups = {sid: [] for sid in image_ids}
    for sid in image_ids:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", sid):
            raise ValueError("Unsafe VisDrone image ID")
    for row in predictions:
        sid, category, score = row["image_id"], row["category_id"], row["score"]
        box = row["bbox"]
        if sid not in groups or type(category) is not int or not 0 <= category < 10:
            raise ValueError("Prediction image/class mismatch")
        if len(box) != 4 or not all(math.isfinite(v) for v in [*box, score]):
            raise ValueError("Nonfinite or malformed detection")
        if not 0 <= score <= 1 or box[2] <= 0 or box[3] <= 0:
            raise ValueError("Invalid confidence or box size")
        if score >= conf:
            groups[sid].append((score, category, tuple(box)))
    files = {}
    expected_names = {sid + ".txt" for sid in image_ids}
    if output.exists() and {p.name for p in output.glob("*.txt")} - expected_names:
        raise ValueError("Extra prediction files not in the evaluation split")
    for sid, detections in groups.items():
        # Stable score ties retain input order. Do not apply a second NMS here.
        detections.sort(key=lambda x: -x[0])
        text = ""
        for score, category, box in detections[:max_det]:
            text += ",".join(f"{v:.10g}" for v in (*box, score)) + f",{category + 1},-1,-1\n"
        immutable(output / f"{sid}.txt", text.encode())
        files[f"{sid}.txt"] = file_sha(output / f"{sid}.txt")
    result = {
        "schema_version": "d1-visdrone-export-v1",
        "image_count": len(image_ids),
        "conf": conf,
        "max_det": max_det,
        "coordinates": "original-image xywh",
        "category_mapping": "0..9 -> 1..10",
        "sort": "stable descending score before global per-image top500",
        "nms": "none in exporter; upstream end2end decoding or NMS must be recorded separately",
        "toolkit_commit": TOOLKIT_COMMIT,
        "files_sha256": files,
    }
    immutable(output / "export.json", encoded(result))
    return result


def validate_official_report(report):
    if report.get("toolkit_commit") != TOOLKIT_COMMIT or report.get("backend") != "official-matlab":
        raise ValueError("Not the pinned official MATLAB evaluator")
    raw = report.get("metrics_percent", {})
    values = report.get("metrics", {})
    if set(raw) != set(METRICS) or set(values) != set(METRICS):
        raise ValueError("Incomplete official metrics")
    for name in METRICS:
        if not math.isfinite(raw[name]) or not 0 <= raw[name] <= 100 or abs(values[name] - raw[name] / 100) > 1e-12:
            raise ValueError("Invalid metric scale or nonfinite official result")
    return report


def make_fixture(source, output):
    """Eight fixed val images with GT-derived predictions, never model accuracy."""
    source, output = Path(source), Path(output)
    if source.name != "VisDrone2019-DET-val":
        raise ValueError("Evaluator fixture may only use official val")
    images = sorted((source / "images").glob("*.jpg"))[:8]
    if len(images) != 8:
        raise ValueError("Need eight fixed validation images")
    predictions, files = [], {}
    for image in images:
        annotation = source / "annotations" / (image.stem + ".txt")
        for original, relative in [(image, "images/" + image.name), (annotation, "annotations/" + annotation.name)]:
            immutable(output / "dataset" / relative, original.read_bytes())
            files[relative] = file_sha(original)
        for row in csv.reader(annotation.read_text().splitlines()):
            values = [float(x) for x in row if x != ""]
            if values[4] == 1 and 1 <= values[5] <= 10 and values[2] > 0 and values[3] > 0:
                predictions.append(
                    {"image_id": image.stem, "category_id": int(values[5]) - 1, "score": 0.9, "bbox": values[:4]}
                )
    immutable(output / "predictions.json", encoded(predictions))
    exported = export_predictions(predictions, [p.stem for p in images], output / "predictions")
    result = {
        "fixture_not_model_accuracy": True,
        "source_split": "val",
        "image_ids": [p.stem for p in images],
        "files_sha256": files,
        "export": exported,
    }
    immutable(output / "fixture.json", encoded(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    export = sub.add_parser("export")
    export.add_argument("--predictions", type=Path, required=True)
    export.add_argument("--image-list", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("check")
    check.add_argument("--report", type=Path, required=True)
    fixture = sub.add_parser("fixture")
    fixture.add_argument("--source", type=Path, required=True)
    fixture.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "export":
        ids = [Path(p).stem for p in args.image_list.read_text().splitlines() if p.strip()]
        result = export_predictions(json.loads(args.predictions.read_text()), ids, args.output)
    elif args.mode == "check":
        result = validate_official_report(json.loads(args.report.read_text()))
    else:
        result = make_fixture(args.source, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
