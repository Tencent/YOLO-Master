"""Build a provenance-preserving full COCO 2017 manifest for unified multi-task training.

The generated artifacts reference the original COCO images and annotations in place. They do not duplicate images,
polygons, keypoints, or dense maps. It configures detection, instance segmentation, pose, image multi-label
classification, optional local depth/normal maps, and optional official Panoptic semantic supervision.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from ultralytics.data.converter import coco91_to_coco80_class


DEFAULT_ROOT = Path(os.environ.get("YOLO_MASTER_COCO_ROOT", Path.home() / "datasets" / "coco2017"))
SCHEMA_VERSION = "coco2017-unified-multitask-v2"
SPLITS = ("train", "val")


def parse_args() -> argparse.Namespace:
    """Parse manifest generation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT, help="COCO 2017 root directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <dataset-root>/unified_multitask.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load a COCO JSON file and fail with an actionable message when it is absent."""
    if not path.is_file():
        raise FileNotFoundError(f"Required COCO annotation file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def valid_polygon(segmentation: Any) -> bool:
    """Return whether a COCO segmentation is a trainable polygon rather than crowd RLE."""
    return isinstance(segmentation, list) and any(isinstance(poly, list) and len(poly) >= 6 for poly in segmentation)


def category_index(category_id: int, class_map: list[int | None]) -> int | None:
    """Map a COCO category id to the contiguous COCO-80 class id."""
    return class_map[category_id - 1] if 0 < category_id <= len(class_map) else None


def image_map(image_dir: Path) -> dict[str, Path]:
    """Index images by file name while checking that the expected image split exists."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"COCO image directory not found: {image_dir}")
    images = {path.name: path for path in image_dir.glob("*.jpg")}
    if not images:
        raise RuntimeError(f"No JPG images found in {image_dir}")
    return images


def auxiliary_path(root: Path, folder: str, image_name: str, suffix: str) -> str | None:
    """Return a root-relative auxiliary-label path only when the local file exists."""
    candidate = root / folder / f"{Path(image_name).stem}{suffix}"
    return str(candidate.relative_to(root)) if candidate.is_file() else None


def config_path(root: Path, path: Path) -> str:
    """Prefer a COCO-root-relative config path while accepting an external output directory."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def panoptic_availability(root: Path, split_name: str) -> dict[int, dict[str, Any]]:
    """Return official Panoptic annotations indexed by COCO image ID when installed."""
    path = root / "annotations" / f"panoptic_{split_name}.json"
    if not path.is_file():
        return {}
    return {int(annotation["image_id"]): annotation for annotation in load_json(path).get("annotations", [])}


def build_split(
    root: Path,
    split: str,
    output_dir: Path,
    class_map: list[int | None],
) -> dict[str, Any]:
    """Create one split image list and one JSONL manifest, returning integrity statistics."""
    split_name = f"{split}2017"
    annotations = root / "annotations"
    instances = load_json(annotations / f"instances_{split_name}.json")
    keypoints = load_json(annotations / f"person_keypoints_{split_name}.json")
    captions = load_json(annotations / f"captions_{split_name}.json")
    available_images = image_map(root / "images" / split_name)

    images_by_id = {int(image["id"]): image for image in instances["images"]}
    instance_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in instances["annotations"]:
        instance_by_image[int(annotation["image_id"])].append(annotation)
    keypoint_by_annotation_id = {int(annotation["id"]): annotation for annotation in keypoints["annotations"]}
    captions_by_image: dict[int, list[str]] = defaultdict(list)
    for annotation in captions["annotations"]:
        caption = str(annotation.get("caption", "")).strip()
        if caption:
            captions_by_image[int(annotation["image_id"])].append(caption)
    panoptic_by_image = panoptic_availability(root, split_name)

    manifest_path = output_dir / f"{split_name}.jsonl"
    image_list_path = output_dir / f"{split_name}.txt"
    stats: Counter[str] = Counter()
    class_counts: Counter[int] = Counter()
    with manifest_path.open("w", encoding="utf-8") as manifest, image_list_path.open(
        "w", encoding="utf-8"
    ) as image_list:
        for image_id in sorted(images_by_id):
            image = images_by_id[image_id]
            image_name = str(image["file_name"])
            image_path = available_images.get(image_name)
            if image_path is None:
                stats["missing_images"] += 1
                continue

            stats["images"] += 1
            bbox_count = 0
            polygon_count = 0
            pose_person_count = 0
            class_ids: set[int] = set()
            for annotation in instance_by_image[image_id]:
                if annotation.get("iscrowd", 0):
                    stats["crowd_annotations_excluded"] += 1
                    continue
                bbox = annotation.get("bbox", ())
                if len(bbox) != 4 or float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
                    stats["invalid_boxes_excluded"] += 1
                    continue
                cls = category_index(int(annotation.get("category_id", 0)), class_map)
                if cls is None:
                    stats["unknown_categories_excluded"] += 1
                    continue
                bbox_count += 1
                class_ids.add(cls)
                class_counts[cls] += 1
                if valid_polygon(annotation.get("segmentation")):
                    polygon_count += 1
                    keypoints_annotation = keypoint_by_annotation_id.get(int(annotation["id"]))
                    if keypoints_annotation and int(keypoints_annotation.get("num_keypoints", 0)) > 0:
                        pose_person_count += 1

            caption_values = captions_by_image[image_id]
            depth = auxiliary_path(root, "depth", image_name, "_depth.png")
            normal = auxiliary_path(root, "normal", image_name, "_normal.png")
            panoptic = panoptic_by_image.get(image_id)
            panoptic_map = (
                root / "annotations" / f"panoptic_{split_name}" / str(panoptic["file_name"])
                if panoptic is not None
                else None
            )
            record = {
                "schema_version": SCHEMA_VERSION,
                "split": split_name,
                "image_id": image_id,
                "image": str(image_path.relative_to(root)),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "tasks": {
                    "detect": bbox_count > 0,
                    "instance_segment": polygon_count > 0,
                    "person_pose": pose_person_count > 0,
                    "image_multilabel_classification": bool(class_ids),
                    "caption": bool(caption_values),
                    "depth_auxiliary": depth is not None,
                    "surface_normal_auxiliary": normal is not None,
                    "panoptic_semantic": panoptic_map is not None and panoptic_map.is_file(),
                },
                "targets": {
                    "detect_instances": bbox_count,
                    "instance_polygons": polygon_count,
                    "pose_person_instances": pose_person_count,
                    "class_ids": sorted(class_ids),
                    "captions": caption_values,
                },
                "auxiliary": {"depth": depth, "surface_normal": normal},
            }
            manifest.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            image_list.write(str(image_path) + "\n")
            stats["detect_images"] += bbox_count > 0
            stats["segment_images"] += polygon_count > 0
            stats["pose_images"] += pose_person_count > 0
            stats["caption_images"] += bool(caption_values)
            stats["depth_auxiliary_images"] += depth is not None
            stats["surface_normal_auxiliary_images"] += normal is not None
            stats["panoptic_semantic_images"] += panoptic_map is not None and panoptic_map.is_file()
            stats["detect_instances"] += bbox_count
            stats["instance_polygons"] += polygon_count
            stats["pose_person_instances"] += pose_person_count
            stats["captions"] += len(caption_values)

    return {
        "image_list": image_list_path.name,
        "manifest": manifest_path.name,
        "annotation_sources": {
            "instances": str((annotations / f"instances_{split_name}.json").relative_to(root)),
            "person_keypoints": str((annotations / f"person_keypoints_{split_name}.json").relative_to(root)),
            "captions": str((annotations / f"captions_{split_name}.json").relative_to(root)),
        },
        "statistics": dict(sorted(stats.items())),
        "instance_class_counts": {str(key): value for key, value in sorted(class_counts.items())},
    }


def write_training_yaml(root: Path, output_dir: Path, names: dict[int, str]) -> Path:
    """Write the complete config when official Panoptic annotations are available."""
    panoptic_ready = all(
        (root / "annotations" / path).exists()
        for path in (
            "panoptic_train2017.json",
            "panoptic_val2017.json",
            "panoptic_train2017",
            "panoptic_val2017",
        )
    )
    tasks = ["detect", "segment", "pose", "classify", "depth", "normal"]
    config = {
        "path": str(root),
        "train": "images/train2017",
        "val": "images/val2017",
        "multitask_format": "coco",
        "train_instances": "annotations/instances_train2017.json",
        "val_instances": "annotations/instances_val2017.json",
        "train_keypoints": "annotations/person_keypoints_train2017.json",
        "val_keypoints": "annotations/person_keypoints_val2017.json",
        "depth_dir": "depth",
        "normal_dir": "normal",
        "depth_scale": 1.0 / 255.0,
        "depth_valid_min": 0,
        "normal_valid_min": 0.1,
        "tasks": tasks,
        "kpt_shape": [17, 3],
        "flip_idx": [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
        "unified_manifest": config_path(root, output_dir / "manifest.json"),
        "caption_annotations": {
            "train": "annotations/captions_train2017.json",
            "val": "annotations/captions_val2017.json",
        },
        "derived_image_multilabel": {
            "source": "instances annotations",
            "field": "targets.class_ids",
            "status": "enabled_as_80_class_multi_hot_bce",
        },
        "auxiliary_dense_labels": {
            "depth": "depth/<image_id>_depth.png",
            "surface_normal": "normal/<image_id>_normal.png",
            "status": "enabled_when_local_png_is_present; missing samples are ignored",
        },
        "names": names,
    }
    if panoptic_ready:
        config.update(
            {
                "semantic_source": "panoptic",
                "semantic_nc": 133,
                "panoptic_train_masks": "annotations/panoptic_train2017",
                "panoptic_val_masks": "annotations/panoptic_val2017",
                "panoptic_train_annotations": "annotations/panoptic_train2017.json",
                "panoptic_val_annotations": "annotations/panoptic_val2017.json",
            }
        )
        tasks.append("semantic")
    path = output_dir / "coco2017_mot_multitask.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def main() -> None:
    """Generate a complete, non-destructive COCO unified multi-task manifest."""
    args = parse_args()
    root = args.dataset_root.resolve()
    output_dir = (args.output_dir or root / "unified_multitask").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"COCO dataset root not found: {root}")
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output exists: {output_dir}. Re-run with --overwrite to replace only this directory."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    categories = load_json(root / "annotations" / "instances_train2017.json")["categories"]
    class_map = coco91_to_coco80_class()
    names = {category_index(int(category["id"]), class_map): str(category["name"]) for category in categories}
    if None in names or len(names) != 80:
        raise RuntimeError("COCO category mapping did not produce the expected 80 contiguous classes")

    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_root": str(root),
        "provenance": {
            "official_coco_annotations": [
                "instances_train2017.json",
                "instances_val2017.json",
                "person_keypoints_train2017.json",
                "person_keypoints_val2017.json",
                "captions_train2017.json",
                "captions_val2017.json",
            ],
            "current_trainer_enabled_tasks": [
                "detect",
                "instance_segment",
                "person_pose",
                "image_multilabel_classification",
                "depth_auxiliary",
                "surface_normal_auxiliary",
                "panoptic_semantic_when_installed",
            ],
            "prepared_but_not_enabled_tasks": ["caption"],
            "unsupported_without_new_labels_and_losses": ["oriented_box"],
            "local_auxiliary_labels": "Depth and normal PNGs are optional local auxiliary supervision, not official COCO labels.",
        },
        "splits": {},
    }
    for split in SPLITS:
        report["splits"][f"{split}2017"] = build_split(root, split, output_dir, class_map)
    yaml_path = write_training_yaml(root, output_dir, names)
    report["training_config"] = yaml_path.name
    (output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Prepared {SCHEMA_VERSION}")
    print(f"Output: {output_dir}")
    for split, summary in report["splits"].items():
        stats = summary["statistics"]
        print(
            f"{split}: images={stats['images']}, detect={stats['detect_images']}, "
            f"segment={stats['segment_images']}, pose={stats['pose_images']}, captions={stats['caption_images']}"
        )
    print(f"Train with: {yaml_path}")


if __name__ == "__main__":
    main()
