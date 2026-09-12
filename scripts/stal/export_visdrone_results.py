#!/usr/bin/env python3
"""Convert Ultralytics predictions.json into the official VisDrone DET submission layout."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from ultralytics.data.utils import IMG_FORMATS


def export_visdrone_results(prediction_path: Path, image_dir: Path, output_dir: Path) -> dict[str, int]:
    """Write one VisDrone DET result TXT per image, including empty files for images without detections."""
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise TypeError("predictions JSON must contain a list")
    image_paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix[1:].lower() in IMG_FORMATS)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {image_dir}")
    known_stems = {path.stem for path in image_paths}
    if len(known_stems) != len(image_paths):
        raise ValueError("Duplicate image stems would overwrite the same result TXT")
    numeric_stems = {int(stem): stem for stem in known_stems if stem.isnumeric()}
    if len(numeric_stems) != sum(stem.isnumeric() for stem in known_stems):
        raise ValueError("Ambiguous numeric image IDs in the image directory")
    detections: dict[str, list[str]] = defaultdict(list)
    for index, prediction in enumerate(predictions):
        file_name = prediction.get("file_name")
        image_id = prediction.get("image_id", "")
        id_stem = str(image_id)
        if id_stem not in known_stems and id_stem.isnumeric():
            id_stem = numeric_stems.get(int(id_stem), id_stem)
        stem = Path(file_name).stem if file_name else id_stem
        if stem not in known_stems:
            raise ValueError(f"Prediction {index} refers to unknown image stem '{stem}'")
        if file_name and "image_id" in prediction and id_stem != stem:
            raise ValueError(f"Prediction {index} has conflicting file_name and image_id")
        bbox = prediction.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Prediction {index} must contain bbox=[left, top, width, height]")
        raw_category = prediction["category_id"]
        category = float(raw_category)
        if not math.isfinite(category) or not category.is_integer() or not 1 <= category <= 10:
            raise ValueError(f"Prediction {index} has non-evaluated VisDrone category_id={category}; expected 1..10")
        category = int(category)
        left, top, width, height = (float(value) for value in bbox)
        score = float(prediction["score"])
        if (
            not all(math.isfinite(value) for value in (left, top, width, height, score))
            or min(width, height) <= 0
            or not 0 <= score <= 1
        ):
            raise ValueError(f"Prediction {index} must have a score in [0, 1] and a finite positive-size bbox")
        detections[stem].append(f"{left:.3f},{top:.3f},{width:.3f},{height:.3f},{score:.8f},{category},-1,-1")

    # A directory from another split must not silently turn into a mixed official submission.
    expected_files = {f"{stem}.txt" for stem in known_stems}
    unexpected = [p.name for p in output_dir.glob("*") if p.suffix.lower() == ".txt" and p.name not in expected_files]
    if unexpected:
        raise ValueError(
            f"Unexpected result TXT files in output directory: {sorted(unexpected)}; use a clean directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        content = "\n".join(detections[image_path.stem])
        (output_dir / f"{image_path.stem}.txt").write_text(f"{content}\n" if content else "", encoding="utf-8")
    return {
        "images": len(image_paths),
        "detections": len(predictions),
        "empty_images": sum(not detections.get(path.stem) for path in image_paths),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Ultralytics predictions.json")
    parser.add_argument("--images", type=Path, required=True, help="Original VisDrone split image directory")
    parser.add_argument("--output", type=Path, required=True, help="Official-toolkit result TXT directory")
    return parser.parse_args()


def main() -> None:
    """Export predictions and print a compact manifest."""
    args = parse_args()
    summary = export_visdrone_results(args.predictions, args.images, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
