"""Create a reproducible image-list subset from a local COCO2017 training split."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(os.environ.get("YOLO_MASTER_COCO_ROOT", Path.home() / "datasets" / "coco2017"))


def parse_args() -> argparse.Namespace:
    """Parse subset generation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--size", type=int, default=2000, help="Number of train2017 images to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for sampling.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "scripts")
    return parser.parse_args()


def main() -> None:
    """Sample COCO images and write an image list plus matching dataset YAML."""
    args = parse_args()
    image_dir = args.dataset_root / "images" / "train2017"
    source_yaml = REPO_ROOT / "scripts" / "coco2017.yaml"
    if args.size <= 0:
        raise ValueError("--size must be positive")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"COCO train image directory not found: {image_dir}")

    images = sorted(image_dir.glob("*.jpg"))
    if len(images) < args.size:
        raise ValueError(f"Requested {args.size} images but only found {len(images)} in {image_dir}")

    selected = sorted(random.Random(args.seed).sample(images, args.size))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"coco2017_train{args.size}_seed{args.seed}"
    list_path = args.output_dir / f"{stem}.txt"
    yaml_path = args.output_dir / f"{stem}.yaml"
    list_path.write_text("\n".join(str(path) for path in selected) + "\n", encoding="utf-8")

    data = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    data["train"] = str(list_path)
    data["subset"] = {"split": "train2017", "size": args.size, "seed": args.seed}
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"sampled {len(selected)} images with seed={args.seed}")
    print(f"image list: {list_path}")
    print(f"dataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()
