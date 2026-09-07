#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C3 stage1 · DeepPCB 数据准备 + few-shot 划分

DeepPCB:印刷电路板缺陷数据集,1500 张 test 图(template+test 对),6 类缺陷:
  open / short / mousebite / spur / pinhole / spurious_copper
标注原始格式:每行 `x1 y1 x2 y2 cls`,cls=1..6,像素坐标(原图 1500x1500)。
本脚本转换为 YOLO 格式(cls=0..5, 归一化 cx cy w h)并做 few-shot 分层采样。

用法:
  python prepare_deeppcb.py --src <DeepPCB/PCBData> --out <out_dir> \
      --shots 5,10,50,100 --seed 824 --val-ratio 0.1
"""
import argparse
import json
import hashlib
import random
import shutil
from pathlib import Path
from PIL import Image

CLASSES = ["open", "short", "mousebite", "spur", "pinhole", "spurious_copper"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_test_images(src: Path):
    """遍历 PCBData/group*/<sample>/<id>_test.jpg,返回 (img_path, label_path, stem)。"""
    items = []
    for img in sorted(src.rglob("*_test.jpg")):
        stem = img.stem[:-5]  # 去掉 "_test"
        parent = img.parent  # .../group*/<sample>
        # 标注在同级 <sample>_not/<stem>.txt
        lab_dir = parent.parent / (parent.name + "_not")
        lab = lab_dir / f"{stem}.txt"
        if not lab.exists():
            # 兜底:全局按 stem 查找
            hits = list(src.rglob(f"{stem}.txt"))
            lab = hits[0] if hits else None
        if lab is None:
            print(f"  ! 缺标注 {lab_dir}/{stem}.txt,跳过 {img.name}")
            continue
        items.append((img, lab, stem))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="DeepPCB/PCBData 目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shots", default="5,10,50,100")
    ap.add_argument("--seed", type=int, default=824)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    shots = [int(s) for s in args.shots.split(",") if s]
    rng = random.Random(args.seed)

    items = find_test_images(src)
    if not items:
        raise SystemExit(f"未在 {src} 找到任何 *_test.jpg")
    print(f"[DeepPCB] 共 {len(items)} 张 test 图")

    images_dir, labels_dir = out / "images", out / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # ---- 单图尺寸(首个样本) ----
    with Image.open(items[0][0]) as im:
        img_w, img_h = im.size
    print(f"[DeepPCB] 图尺寸: {img_w}x{img_h}")

    ids = []
    for img, lab, stem in items:
        shutil.copy2(img, images_dir / (stem + img.suffix))
        # 校验图片后缀统一 .jpg 后写入 txt
        boxes = []
        for ln in lab.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 5:
                continue
            x1, y1, x2, y2 = (float(v) for v in parts[:4])
            cls = int(parts[4]) - 1  # 1..6 -> 0..5
            if cls < 0 or cls > 5:
                continue
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            cx, cy = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
            w, h = (x2 - x1) / img_w, (y2 - y1) / img_h
            boxes.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (labels_dir / f"{stem}.txt").write_text("\n".join(boxes) + ("\n" if boxes else ""))
        ids.append(stem)
    print(f"[DeepPCB] 转换完成,{len(ids)} 条")

    # ---- train/val 划分(seed 打乱) ----
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * args.val_ratio))
    val_ids, train_ids = set(ids[:n_val]), ids[n_val:]
    print(f"[DeepPCB] train={len(train_ids)} val={len(val_ids)}")

    def img_abspath(stem):
        for suf in (".jpg", ".png", ".jpeg"):
            p = images_dir / (stem + suf)
            if p.exists():
                return p
        raise FileNotFoundError(stem)

    (out / "train.txt").write_text("\n".join(str(img_abspath(i)) for i in train_ids))
    (out / "val.txt").write_text("\n".join(str(img_abspath(i)) for i in val_ids))

    split_report = {"classes": CLASSES, "seed": args.seed, "n_images": len(ids), "img_size": [img_w, img_h]}

    # ---- 每类图片索引(few-shot 分层采样) ----
    class_to_imgs = {}
    for i in train_ids:
        lab = labels_dir / f"{i}.txt"
        for c in {ln.split()[0] for ln in lab.read_text().splitlines() if ln.strip()}:
            class_to_imgs.setdefault(c, []).append(i)
    split_report["class_histogram_train"] = {c: len(v) for c, v in sorted(class_to_imgs.items())}

    shot_dir = out / "shots"
    for k in shots:
        d = shot_dir / f"k{k}"
        d.mkdir(parents=True, exist_ok=True)
        picked = []
        for c in sorted(class_to_imgs):
            picked.extend(class_to_imgs[c][:k])
        picked = sorted(set(picked))
        (d / "train.txt").write_text("\n".join(str(img_abspath(i)) for i in picked))
        (d / "val.txt").write_text("\n".join(str(img_abspath(i)) for i in val_ids))
        split_report[f"k{k}"] = {"train": len(picked), "per_class": k, "val": len(val_ids)}
        print(f"[DeepPCB] few-shot k{k}: train={len(picked)}")

    yaml_txt = f"""path: {out}
train: train.txt
val: val.txt
names:
  0: open
  1: short
  2: mousebite
  3: spur
  4: pinhole
  5: spurious_copper
"""
    (out / "deeppcb.yaml").write_text(yaml_txt)

    sample = random.Random(0).sample(ids, min(5, len(ids)))
    split_report["sample_images_sha256"] = {s: sha256_file(img_abspath(s)) for s in sample}
    (out / "split_report.json").write_text(json.dumps(split_report, indent=2, ensure_ascii=False))
    print(f"[DeepPCB] 完成 -> {out}\n  deeppcb.yaml + split_report.json 已生成")


if __name__ == "__main__":
    main()
