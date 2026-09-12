from __future__ import annotations

"""A1 S2：COCO train2017 固定 seed 抽样 20000 张 + 全量 val2017 5k 评测目录。

数据源 /root/gpufree-data/datasets/coco/（train2017 由 hf-mirror parquet 提取，
val2017 为魔搭镜像，均为 ultralytics 结构 images/{split} + labels/{split}）。

产物：
1. coco-train-2017/  —— images/train 20k 软链 + labels/train 复制（同 val 拆分口径），
   images/val + labels/val = val2017 全量 5k（20k 复验的 eval 口径，用户拍板）
2. 清单 JSON：seed + train/val 完整文件列表（可复现）
3. 数据集 YAML：ultralytics/cfg/datasets/coco-train-2017.yaml

分发：本地拆分完成后 rsync -aL 解引用上传共享盘（训练机无源目录，软链会断）。

用法：
    python scripts/a1/prepare_coco_train_split.py [--n-train 20000]
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import yaml

SRC = Path("/root/gpufree-data/datasets/coco")
SRC_TRAIN_IMAGES = SRC / "images/train2017"
SRC_TRAIN_LABELS = SRC / "labels/train2017"
SRC_VAL_IMAGES = SRC / "images/val2017"
SRC_VAL_LABELS = SRC / "labels/val2017"
DST = Path("/root/gpufree-data/datasets/coco-train-2017")
MANIFEST = Path("/root/workspace/docs/experiments/2026-09-02/split_manifest.json")
YAML_OUT = Path("/root/workspace/YOLO-Master/ultralytics/cfg/datasets/coco-train-2017.yaml")
SEED = 0
N_VAL = 5000  # eval 用全量 val2017


def link_split(dst_root: Path, files: list[str], src_images: Path, src_labels: Path, split: str) -> None:
    """软链图片 + 复制标签（训练缓存会写 labels.cache，标签不能软链）。"""
    img_dir = dst_root / "images" / split
    lbl_dir = dst_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        img_dir.joinpath(name).symlink_to(src_images / name)
        src_lbl = src_labels / f"{Path(name).stem}.txt"
        dst_lbl = lbl_dir / f"{Path(name).stem}.txt"
        if src_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)
        else:
            dst_lbl.touch()
    print(f"{split}: {len(files)} 张图")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=20000)
    args = ap.parse_args()

    train_images = sorted(p.name for p in SRC_TRAIN_IMAGES.glob("*.jpg"))
    val_images = sorted(p.name for p in SRC_VAL_IMAGES.glob("*.jpg"))
    print(f"源图数: train2017={len(train_images)} (官方 118287), val2017={len(val_images)}")
    assert len(val_images) == N_VAL, f"val2017 应为 {N_VAL} 张, 实际 {len(val_images)}"
    assert args.n_train <= len(train_images)

    rng = random.Random(SEED)
    rng.shuffle(train_images)
    train_files = train_images[: args.n_train]

    link_split(DST, train_files, SRC_TRAIN_IMAGES, SRC_TRAIN_LABELS, "train")
    link_split(DST, val_images, SRC_VAL_IMAGES, SRC_VAL_LABELS, "val")

    manifest = {
        "seed": SEED,
        "source_train": str(SRC_TRAIN_IMAGES),
        "source_val": str(SRC_VAL_IMAGES),
        "train": train_files,
        "val": val_images,
        "n_train": len(train_files),
        "n_val": len(val_images),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"清单已存档: {MANIFEST}")

    names = yaml.safe_load(Path("/root/workspace/YOLO-Master/ultralytics/cfg/datasets/coco.yaml").read_text())["names"]
    cfg = {
        "path": str(DST),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    YAML_OUT.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    print(f"数据集 YAML 已生成: {YAML_OUT}")
    print("拆分完成 ✅")


if __name__ == "__main__":
    main()
