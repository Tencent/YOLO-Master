#!/usr/bin/env python3
"""Convert VisDrone annotations to YOLO format."""

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def convert_box(size, box):
    """Convert VisDrone box to YOLO xywh box."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh


def visdrone2yolo(dir_path):
    """Convert VisDrone annotations to YOLO labels."""
    dir_path = Path(dir_path)
    labels_dir = dir_path / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)

    annotations_dir = dir_path / 'annotations'
    images_dir = dir_path / 'images'

    if not annotations_dir.exists():
        print(f"No annotations directory found in {dir_path}")
        return

    pbar = tqdm(list(annotations_dir.glob('*.txt')), desc=f'Converting {dir_path.name}')
    for f in pbar:
        img_path = (images_dir / f.name).with_suffix('.jpg')
        if not img_path.exists():
            continue

        img_size = Image.open(img_path).size
        lines = []

        with open(f, 'r') as file:
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if len(row) < 6:
                    continue
                if row[4] == '0':  # ignored regions
                    continue
                cls = int(row[5]) - 1
                if cls < 0 or cls > 9:
                    continue
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")

        label_path = labels_dir / f.name
        with open(label_path, 'w') as fl:
            fl.writelines(lines)


if __name__ == '__main__':
    base_dir = Path('/root/autodl-tmp/datasets/VisDrone')

    for split in ['VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev']:
        split_dir = base_dir / split / split
        if split_dir.exists():
            visdrone2yolo(split_dir)
