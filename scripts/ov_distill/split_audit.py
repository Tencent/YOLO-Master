"""Audit image-level separation between grounding training and validation data."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.ov_distill.class_split import load_class_split
from scripts.ov_distill.reliability import assert_disjoint_manifests, image_key, load_image_manifest


def yolo_val_image_paths(data_yaml: Path) -> set[str]:
    """Resolve the validation images exactly from a YOLO detection dataset YAML."""
    from ultralytics.data.utils import IMG_FORMATS, check_det_dataset

    data = check_det_dataset(str(Path(data_yaml).resolve()), autodownload=False, split="val")
    values = data["val"] if isinstance(data["val"], list) else [data["val"]]
    paths: set[str] = set()
    for value in values:
        split_path = Path(value)
        if split_path.is_dir():
            candidates = split_path.rglob("*.*")
        elif split_path.is_file() and split_path.suffix.lower() == ".txt":
            entries = split_path.read_text(encoding="utf-8").splitlines()
            parent = split_path.parent
            candidates = (
                (parent / entry[2:] if entry.startswith("./") else Path(entry)) for entry in entries if entry.strip()
            )
        elif split_path.is_file():
            candidates = [split_path]
        else:
            raise FileNotFoundError(f"validation path from {data_yaml} does not exist: {split_path}")
        paths.update(
            image_key(str(path)) for path in candidates if path.is_file() and path.suffix[1:].lower() in IMG_FORMATS
        )
    if not paths:
        raise ValueError(f"validation YAML contains no readable images: {data_yaml}")
    return paths


def grounding_image_paths(grounding_json: Path, image_root: Path) -> set[str]:
    """Resolve image paths referenced by a GoldG-style grounding JSON."""
    payload = json.loads(Path(grounding_json).read_text(encoding="utf-8"))
    images = payload.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError(f"grounding JSON contains no images: {grounding_json}")
    root = Path(image_root).resolve()
    paths = set()
    for row in images:
        file_name = row.get("file_name")
        if not file_name:
            raise ValueError(f"grounding JSON has an image without file_name: {grounding_json}")
        path = Path(file_name).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"grounding image does not exist: {path}")
        paths.add(str(path))
    if len(paths) != len(images):
        raise ValueError(f"grounding JSON contains duplicate image paths: {grounding_json}")
    return paths


def grounding_class_names(grounding_json: Path) -> set[str]:
    """Read positive phrase spans and explicit negatives from a GoldG grounding JSON."""
    payload = json.loads(Path(grounding_json).read_text(encoding="utf-8"))
    image_by_id = {row.get("id"): row for row in payload.get("images", [])}
    names = set()
    for image in image_by_id.values():
        caption = str(image.get("caption", "")).strip()
        if caption.endswith("."):
            caption = caption[:-1]
        names.update(" ".join(name.strip().lower().split()) for name in caption.split(". ") if name.strip())
    for annotation in payload.get("annotations", []):
        image = image_by_id.get(annotation.get("image_id"))
        if image is None:
            raise ValueError(f"grounding annotation references an unknown image: {annotation.get('image_id')}")
        caption = image.get("caption", "")
        spans = annotation.get("tokens_positive")
        if not isinstance(spans, list) or not spans:
            raise ValueError("grounding annotation is missing tokens_positive class spans")
        for span_group in spans:
            if not isinstance(span_group, list):
                raise TypeError("grounding tokens_positive entries must be lists")
            span = span_group if len(span_group) == 2 and isinstance(span_group[0], int) else None
            if span is None:
                for candidate in span_group:
                    if isinstance(candidate, list) and len(candidate) == 2:
                        start, end = candidate
                        if isinstance(start, int) and isinstance(end, int):
                            names.add(" ".join(caption[start:end].strip().lower().split()))
            elif span[0] < 0 or span[1] > len(caption) or span[0] >= span[1]:
                raise ValueError("grounding tokens_positive contains an invalid caption span")
            else:
                start, end = span
                names.add(" ".join(caption[start:end].strip().lower().split()))
        negatives = image.get("negative_captions", [])
        if not isinstance(negatives, list):
            raise TypeError("grounding image negative_captions must be a list")
        names.update(" ".join(str(name).strip().lower().split()) for name in negatives)
    return names


def validate_training_eval_split(
    train_manifest: Path,
    eval_manifest: Path,
    grounding_json: Path,
    image_root: Path,
    class_split: Path,
    val_yaml: Path | None = None,
) -> dict:
    """Reject image/class leakage and mismatches between manifests and actual training inputs."""
    train_images, eval_images = assert_disjoint_manifests(
        train_manifest, eval_manifest, first_name="training", second_name="evaluation"
    )
    missing = sorted(path for path in train_images | eval_images if not Path(path).is_file())
    if missing:
        raise FileNotFoundError(f"split manifests reference {len(missing)} missing images (examples: {missing[:5]})")

    grounding_images = grounding_image_paths(grounding_json, image_root)
    outside_train = grounding_images - train_images
    if outside_train:
        examples = sorted(outside_train)[:5]
        raise ValueError(
            f"grounding JSON contains {len(outside_train)} images outside the training manifest (examples: {examples})"
        )
    leaked = grounding_images & eval_images
    if leaked:
        examples = sorted(leaked)[:5]
        raise ValueError(f"grounding JSON overlaps evaluation images on {len(leaked)} images: {examples}")

    split = load_class_split(class_split)
    seen = {" ".join(name.lower().split()) for name in split["seen"]}
    training_classes = grounding_class_names(grounding_json)
    forbidden = training_classes - seen
    if forbidden:
        raise ValueError(
            f"grounding JSON contains {len(forbidden)} classes outside the seen split "
            f"(unseen/unknown: {sorted(forbidden)[:10]})"
        )

    grounding_payload = json.loads(Path(grounding_json).read_text(encoding="utf-8"))
    metadata = grounding_payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("grounding JSON metadata must be an object when provided")
    if not metadata.get("class_split"):
        raise ValueError("grounding JSON is missing class-split provenance; regenerate it with convert --class-split")
    if metadata.get("provenance_version") != 1:
        raise ValueError("grounding JSON has unsupported or missing split-provenance metadata")
    if metadata.get("weight_mode") not in {"agreement", "calibrated", "uniform"}:
        raise ValueError("grounding JSON has missing or unsupported reliability weight-mode metadata")
    if metadata.get("training_manifest"):
        source_train_manifest = Path(metadata["training_manifest"])
        if not source_train_manifest.is_absolute():
            source_train_manifest = Path(grounding_json).parent / source_train_manifest
        if load_image_manifest(source_train_manifest) != train_images:
            raise ValueError("grounding JSON was converted from a different training manifest")
    if set(metadata.get("seen_classes", [])) != set(split["seen"]):
        raise ValueError("grounding JSON class split does not match the training --class-split")
    if set(metadata.get("unseen_classes", [])) != set(split["unseen"]):
        raise ValueError("grounding JSON class split does not match the training --class-split")
    if metadata.get("weight_mode") == "calibrated" and not metadata.get("calibration_manifest"):
        raise ValueError("calibrated grounding JSON is missing calibration-manifest provenance")
    if metadata.get("calibration_manifest"):
        calibration_manifest = Path(metadata["calibration_manifest"])
        if not calibration_manifest.is_absolute():
            calibration_manifest = Path(grounding_json).parent / calibration_manifest
        calibration_images = load_image_manifest(calibration_manifest)
        if calibration_images & train_images:
            raise ValueError("calibration manifest overlaps the training split")
        if calibration_images & eval_images:
            raise ValueError("calibration manifest overlaps the evaluation split")

    if val_yaml is not None:
        yaml_images = yolo_val_image_paths(val_yaml)
        if yaml_images != eval_images:
            missing_from_yaml = sorted(eval_images - yaml_images)[:5]
            outside_manifest = sorted(yaml_images - eval_images)[:5]
            raise ValueError(
                "validation YAML and evaluation manifest do not describe the same images "
                f"(manifest-only examples: {missing_from_yaml}; YAML-only examples: {outside_manifest})"
            )

    return {
        "train_images": len(train_images),
        "eval_images": len(eval_images),
        "grounding_images": len(grounding_images),
        "image_overlap": 0,
        "class_split": str(Path(class_split).resolve()),
        "training_classes": len(training_classes),
        "unseen_classes_in_training": 0,
        "calibration_images": len(calibration_images) if metadata.get("calibration_manifest") else 0,
        "val_yaml_checked": val_yaml is not None,
    }
