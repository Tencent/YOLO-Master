#!/usr/bin/env python3
"""VisDrone2019-DET → YOLO 标签转换。

把 VisDrone 原始标注（逗号分隔：x,y,w,h,score,category,truncation,occlusion）
转成 YOLO 格式（class x_center y_center w h，归一化）。产出结构：

    <root>/images/{train,val}/*.jpg
    <root>/labels/{train,val}/*.txt

默认 copy 图片（保留原始目录，避免破坏其他脚本的路径依赖）。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# VisDrone 官方类别 → 0-based（class - 1）
CATEGORIES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}


def visdrone2yolo(src: Path, out_img: Path, out_lab: Path) -> int:
    """Convert one VisDrone split. Returns number of images processed."""
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path in (src / "images").glob("*.jpg"):
        shutil.copy2(img_path, out_img / img_path.name)
        count += 1
    skipped_boxes = 0
    for ann in tqdm(sorted((src / "annotations").glob("*.txt")), desc=f"labels {src.parent.name}/{src.name}", leave=False):
        img_name = ann.with_suffix(".jpg").name
        img_path = out_img / img_name
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            W, H = im.size
        dw, dh = 1.0 / W, 1.0 / H
        lines = []
        for row in (x.split(",") for x in ann.read_text().strip().splitlines()):
            if len(row) < 6:
                continue
            if row[4] != "0":  # score==0 → ignored region，跳过
                x, y, w, h = map(int, row[:4])
                cls = int(row[5]) - 1  # VisDrone 1-based → 0-based
                if cls not in CATEGORIES:
                    continue
                lines.append(f"{cls} {(x + w / 2) * dw:.6f} {(y + h / 2) * dh:.6f} {w * dw:.6f} {h * dh:.6f}")
            else:
                skipped_boxes += 1
        (out_lab / ann.name).write_text("\n".join(lines))
    return count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/visdrone"))
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--remove-source", action="store_true", help="转换后删除原始 VisDrone2019-DET-* 目录（保留图片副本）")
    args = ap.parse_args()

    root = args.root.resolve()
    for split in args.splits:
        src = root / f"VisDrone2019-DET-{split}"
        if not src.exists():
            print(f"[skip] {src} 不存在（{split} 可能未解压）")
            continue
        out_img = root / "images" / split
        out_lab = root / "labels" / split
        n = visdrone2yolo(src, out_img, out_lab)
        print(f"[{split}] {n} images copied, labels -> {out_lab}")

    # 生成 data.yaml
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CATEGORIES.values()))
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )
    print(f"\n[data.yaml] wrote {root / 'data.yaml'}")

    # 统计
    for split in args.splits:
        img = root / "images" / split
        lab = root / "labels" / split
        if img.exists():
            print(f"[{split}] images={len(list(img.glob('*.jpg')))} labels={len(list(lab.glob('*.txt')))}")

    if args.remove_source:
        for split in args.splits:
            src = root / f"VisDrone2019-DET-{split}"
            if src.exists():
                shutil.rmtree(src)
                print(f"[remove] {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
