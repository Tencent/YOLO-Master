#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C3 工业缺陷 V-PEFT 小样本实战 —— NEU-DET 数据准备 + few-shot 划分 (GPU 机使用)

NEU-DET:东北大学钢材表面缺陷数据集(1800 张,6 类,crazing / inclusion / patches /
pitted_surface / rolled-in_scale / scratches)。

用法(在 GPU 机器上):
  python smoke/c3/prepare_neu_det.py --src /path/to/NEU-DET --out /path/to/neu_det_yolo \
      --shots 5,10,50,100 --seed 824

输入目录要求(两种格式任选):
  1) 原始格式:  --src 内含 IMAGES/*.jpg 与 ANNOTATIONS/*.xml
  2) kaggle 解包: --src 内含 images/*.jpg 与 annotations/*.xml(文件名小写)

输出:
  <out>/
    images/*.jpg          (YOLO 训练用图)
    labels/*.txt          (YOLO 标签: cls cx cy w h)
    train.txt / val.txt   (训练/验证图清单)
    shots/k5/  k10/  k50/  k100/  (每个目录内含 train.txt + val.txt 指向 <out>)
    neu_det.yaml          (Ultralytics 数据集配置)
    split_report.json     (划分统计 + SHA-256,供准入文档引用)

标注许可:NEU-DET 学术开放,下载页面请确认许可条款(任务书要求锁定数据许可)。
"""

import argparse
import json
import hashlib
import random
import shutil
import sys
from pathlib import Path

try:
    import xml.etree.ElementTree as ET
except Exception as e:  # pragma: no cover
    raise SystemExit(f"缺少依赖: {e}")

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_srcs(src: Path):
    img_dir = None
    xml_dir = None
    # 常见布局
    candidates = [
        (src / "IMAGES", src / "ANNOTATIONS"),
        (src / "images", src / "annotations"),
        (src / "JPEGImages", src / "Annotations"),
        (src, src / "annotations"),
    ]
    for idir, xdir in candidates:
        if idir.exists() and xdir.exists():
            img_dir, xml_dir = idir, xdir
            break
    if img_dir is None or xml_dir is None:
        # 兜底:src 下直接平铺 images 与 xml
        imgs = sorted(src.rglob("*.jpg"))
        xmls = sorted(src.rglob("*.xml"))
        if imgs and xmls:
            img_dir, xml_dir = src, src
        else:
            raise SystemExit(f"未在 {src} 找到 NEU-DET 图片与标注(需要 IMAGES/*.jpg + ANNOTATIONS/*.xml)")
    return img_dir, xml_dir


def convert_xml(xml_file: Path, img_w: int, img_h: int):
    """NEU-DET XML -> [(cls_id, cx, cy, w, h)] 归一化坐标"""
    root = ET.parse(xml_file).getroot()
    boxes = []
    for obj in root.findall("object"):
        cls = obj.findtext("name")
        if cls not in CLASSES:
            continue
        b = obj.find("bndbox")
        xmin, ymin = float(b.findtext("xmin")), float(b.findtext("ymin"))
        xmax, ymax = float(b.findtext("xmax")), float(b.findtext("ymax"))
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)
        ymin, ymax = min(ymin, ymax), max(ymin, ymax)
        cx, cy = (xmin + xmax) / 2 / img_w, (ymin + ymax) / 2 / img_h
        w, h = (xmax - xmin) / img_w, (ymax - ymin) / img_h
        boxes.append((CLASSES.index(cls), cx, cy, w, h))
    return boxes


def run_yolo_mode(src: Path, out: Path, shots, rng, val_ratio: float):
    """YOLO 格式目录:train|val/{images,labels} 或 images/{labels}。复制到 out 并做 few-shot 划分。"""
    images_dir, labels_dir = out / "images", out / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    def copy_split(sub: str):
        img_sub = src / sub / "images"
        if not img_sub.is_dir():
            return []
        lab_sub = src / sub / "labels"
        ids = []
        for img in sorted(img_sub.glob("*.jpg")) + sorted(img_sub.glob("*.png")):
            lab = lab_sub / (img.stem + ".txt")
            if not lab.exists():
                print(f"  ! 缺标签 {lab},跳过")
                continue
            shutil.copy2(img, images_dir / img.name)
            shutil.copy2(lab, labels_dir / img.name.replace(img.suffix, ".txt"))
            ids.append(img.stem)
        return ids

    train_ids = copy_split("train")
    val_ids = copy_split("val")
    # 常见仓库无 val 目录,用 test 作为验证集(如 Marfbin/NEU-DET-with-yolov8)
    if not val_ids:
        val_ids = copy_split("test")
        if val_ids:
            print(f"[NEU-DET] 未发现 val,使用 test 作为验证集(test={len(val_ids)})")
    # 只有 images/ 平铺时:自行划分
    if not train_ids and (src / "images").is_dir():
        all_ids = [img.stem for img in (src / "images").glob("*.jpg")]
        rng.shuffle(all_ids)
        n_val = max(1, int(len(all_ids) * val_ratio))
        val_ids, train_ids = all_ids[:n_val], all_ids[n_val:]

    if not train_ids:
        raise SystemExit(f"train 划分为空: {src}/train/images 不存在或为空")
    print(f"[NEU-DET] YOLO 格式: train={len(train_ids)} val={len(val_ids)}")

    (out / "train.txt").write_text("\n".join(str(images_dir / f"{i}.jpg") for i in train_ids))
    (out / "val.txt").write_text("\n".join(str(images_dir / f"{i}.jpg") for i in val_ids))

    split_report = {"classes": CLASSES, "seed": rng.randrange(10**9), "n_images": len(train_ids) + len(val_ids),
                    "mode": "yolo", "src": str(src), "files": {}}

    # 类别-图片索引(每张图可能含多类)
    def image_classes(stem: str):
        lab = labels_dir / f"{stem}.txt"
        if not lab.exists():
            return []
        return sorted({ln.split()[0] for ln in lab.read_text().splitlines() if ln.strip()})

    class_to_imgs = {}
    for i in train_ids:
        for c in image_classes(i):
            class_to_imgs.setdefault(c, []).append(i)
    split_report["class_histogram_train"] = {c: len(v) for c, v in sorted(class_to_imgs.items())}

    # few-shot 采用"每类 k 张"的分层采样(小样本实验纪律:必须覆盖全部类别)
    shot_dir = out / "shots"
    for k in shots:
        d = shot_dir / f"k{k}"
        d.mkdir(parents=True, exist_ok=True)
        picked = []
        for c in sorted(class_to_imgs):
            picked.extend(class_to_imgs[c][:k])
        (d / "train.txt").write_text("\n".join(str(images_dir / f"{i}.jpg") for i in picked))
        (d / "val.txt").write_text("\n".join(str(images_dir / f"{i}.jpg") for i in val_ids))
        split_report[f"k{k}"] = {"train": len(picked), "per_class": k, "val": len(val_ids)}
        print(f"[NEU-DET] few-shot k{k}: train={len(picked)} (每类{k}张, {len(class_to_imgs)}类)")

    yaml_txt = f"""path: {out}
train: train.txt
val: val.txt
names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
"""
    (out / "neu_det.yaml").write_text(yaml_txt)
    sample = random.Random(0).sample(train_ids, min(5, len(train_ids)))
    split_report["sample_images_sha256"] = {s: sha256_file(images_dir / f"{s}.jpg") for s in sample}
    (out / "split_report.json").write_text(json.dumps(split_report, indent=2, ensure_ascii=False))
    print(f"[NEU-DET] 完成 -> {out}\n  neu_det.yaml 与 split_report.json 已生成(含样本 SHA-256)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="NEU-DET 原始目录(IMAGES+ANNOTATIONS)或 YOLO 格式目录(train|val/{images,labels})")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--shots", default="5,10,50,100", help="few-shot 图片数(逗号分隔)")
    ap.add_argument("--seed", type=int, default=824)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shots = [int(s) for s in args.shots.split(",") if s]
    rng = random.Random(args.seed)

    # ---- 模式判定:XML 原始格式 vs 已转换 YOLO 格式 ----
    yolo_mode = (src / "train" / "images").is_dir() or (src / "images").is_dir()
    if yolo_mode:
        print(f"[NEU-DET] 检测到 YOLO 格式目录,跳过 XML 转换")
        run_yolo_mode(src, out, shots, rng, args.val_ratio)
        return

    img_dir, xml_dir = find_srcs(src)
    imgs = sorted(img_dir.glob("*.jpg"))
    print(f"[NEU-DET] 图片 {len(imgs)} 张,来自 {img_dir}")

    # ---- 1. 复制图片 + 转换标签 ----
    images_dir, labels_dir = out / "images", out / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    split_report = {"classes": CLASSES, "seed": args.seed, "n_images": len(imgs), "files": {}}
    ids = []
    for img in imgs:
        xml = xml_dir / (img.stem + ".xml")
        if not xml.exists():
            xml = xml_dir / (img.stem + ".XML")
        if not xml.exists():
            print(f"  ! 缺标注 {xml},跳过")
            continue
        shutil.copy2(img, images_dir / img.name)
        # 读取图片尺寸
        try:
            from PIL import Image
            with Image.open(img) as im:
                w, h = im.size
        except Exception:
            import cv2
            im = cv2.imread(str(img))
            h, w = im.shape[:2]
        boxes = convert_xml(xml, w, h)
        lab = labels_dir / (img.stem + ".txt")
        with open(lab, "w") as f:
            for b in boxes:
                f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
        ids.append(img.stem)
    print(f"[NEU-DET] 转换完成,共 {len(ids)} 条可用")

    # ---- 2. 划分 train/val ----
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * args.val_ratio))
    val_ids, train_ids = set(ids[:n_val]), ids[n_val:]
    (out / "train.txt").write_text("\n".join(str(images_dir / f"{i}.jpg") for i in train_ids))
    (out / "val.txt").write_text("\n".join(str(images_dir / f"{i}.jpg") for i in val_ids))
    print(f"[NEU-DET] train={len(train_ids)} val={len(val_ids)}")

    # ---- 3. few-shot 子集 ----
    shot_dir = out / "shots"
    for k in shots:
        d = shot_dir / f"k{k}"
        d.mkdir(parents=True, exist_ok=True)
        picked = train_ids[:k]  # 固定顺序采样,seed 已打乱
        (d / "train.txt").write_text("\n".join(str(images_dir / f"{i}.jpg") for i in picked))
        (d / "val.txt").write_text("\n".join(str(images_dir / f"{i}.jpg") for i in val_ids))
        split_report[f"k{k}"] = {"train": len(picked), "val": len(val_ids)}
        print(f"[NEU-DET] few-shot k{k}: train={len(picked)}")

    # ---- 4. yaml + 报告 ----
    yaml_txt = f"""path: {out}
train: train.txt
val: val.txt
names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
"""
    (out / "neu_det.yaml").write_text(yaml_txt)
    sample = random.Random(0).sample(ids, min(5, len(ids)))
    split_report["sample_images_sha256"] = {s: sha256_file(images_dir / f"{s}.jpg") for s in sample}
    (out / "split_report.json").write_text(json.dumps(split_report, indent=2, ensure_ascii=False))
    print(f"[NEU-DET] 完成 -> {out}\n  neu_det.yaml 与 split_report.json 已生成")
    print("  请检查 split_report.json 中的 SHA-256 并在准入文档中引用")


if __name__ == "__main__":
    main()
