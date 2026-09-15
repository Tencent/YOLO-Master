"""Shared synthetic datasets for tests that cannot assume a full COCO checkout."""

from __future__ import annotations

from pathlib import Path


def tiny_coco_multitask_yaml(root: Path) -> str:
    """Create a minimal COCO-format detect+segment multi-task dataset YAML.

    Four 32×32 images with one box/mask each. Used by engine and python tests so
    ``task=multitask`` does not require the full COCO 2017 checkout referenced
    by ``coco-multitask.yaml``.

    Args:
        root (Path): Directory that will hold images, annotations, and the YAML.

    Returns:
        (str): Absolute path to the generated dataset YAML.
    """
    import json

    import cv2
    import numpy as np
    import yaml

    img_dir = root / "images" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for i in range(4):
        cv2.imwrite(str(img_dir / f"img{i}.jpg"), np.full((32, 32, 3), i * 40, dtype=np.uint8))
        images.append({"id": i, "file_name": f"img{i}.jpg", "width": 32, "height": 32})
        annotations.append(
            {
                "id": i,
                "image_id": i,
                "category_id": 1,
                "bbox": [4.0, 4.0, 16.0, 16.0],
                "area": 256.0,
                "iscrowd": 0,
                "segmentation": [[4.0, 4.0, 20.0, 4.0, 20.0, 20.0, 4.0, 20.0]],
            }
        )
    (root / "instances.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "person"}]})
    )
    yaml_file = root / "data.yaml"
    yaml_file.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/train",
                "names": {0: "person"},
                "multitask_format": "coco",
                "tasks": ["detect", "segment"],
                "train_instances": "instances.json",
                "val_instances": "instances.json",
                "kpt_shape": [17, 3],
            }
        )
    )
    return str(yaml_file)
