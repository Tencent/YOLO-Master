from __future__ import annotations

"""A1 S1 阶段2：COCO val2017 固定 seed 拆分 4000 train / 1000 val。

从 /root/gpufree-data/datasets/coco/（魔搭镜像下载，含 images/val2017 5000 张与
labels/val2017 YOLO 标签）按 seed=0 shuffle 拆分，用软链组织数据集目录，
并生成清单 JSON（可复现）与数据集 YAML。

用法：
    python scripts/a1/prepare_coco_val_split.py
"""

import json
import random
import shutil
from pathlib import Path

import yaml

SRC_IMAGES = Path("/root/gpufree-data/datasets/coco/images/val2017")
SRC_LABELS = Path("/root/gpufree-data/datasets/coco/labels/val2017")
DST = Path("/root/gpufree-data/datasets/coco-val-2017")
MANIFEST = Path("/root/workspace/docs/experiments/2026-08-31/split_manifest.json")
YAML_OUT = Path("/root/workspace/YOLO-Master/ultralytics/cfg/datasets/coco-val-2017.yaml")
SEED = 0
N_TRAIN, N_VAL = 4000, 1000


def main() -> None:
    images = sorted(p.name for p in SRC_IMAGES.glob("*.jpg"))
    assert len(images) == 5000, f"预期 5000 张图，实际 {len(images)}"

    rng = random.Random(SEED)
    rng.shuffle(images)
    train_files, val_files = images[:N_TRAIN], images[N_TRAIN:]

    for split, files in (("train", train_files), ("val", val_files)):
        img_dir = DST / "images" / split
        lbl_dir = DST / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            img_dir.joinpath(name).symlink_to(SRC_IMAGES / name)
            src_lbl = SRC_LABELS / f"{Path(name).stem}.txt"
            dst_lbl = lbl_dir / f"{Path(name).stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)  # 复制而非软链：训练缓存会写 labels.cache
            else:
                dst_lbl.touch()  # 空标注图：生成空标签文件
        print(f"{split}: {len(files)} 张图")

    manifest = {
        "seed": SEED,
        "source": str(SRC_IMAGES),
        "train": train_files,
        "val": val_files,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"清单已存档: {MANIFEST}")

    # 数据集 YAML（names 复用内置 coco.yaml 的 80 类）
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
