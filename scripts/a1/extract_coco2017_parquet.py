from __future__ import annotations

"""从 HuggingFace detection-datasets/coco 的 parquet 中提取 train2017 原始图片与 YOLO 标签。

源格式（HF det 规范，经 2026-09-02 采样核实）：
    image_id: int64           # COCO 图片 id，文件名 = f"{image_id:012d}.jpg"
    image:    struct<bytes>   # 内嵌原始 jpg
    width/height: int64       # 图片原始尺寸
    objects:  struct<category: list<int64>, bbox: list<fixed_size_list<double>[4]>, ...>
        category 为 0-79 的 YOLO 类索引（非 COCO 原始 category_id）
        bbox 为 xyxy 绝对像素坐标（相对 width/height）

输出（与 val2017 魔搭包同结构）：
    <out_dir>/images/train2017/{image_id:012d}.jpg
    <out_dir>/labels/train2017/{image_id:012d}.txt   # 每行: cls cx cy w h（归一化）
    空标注图写空 txt（与 val2017 口径一致）

用法：
    python extract_coco2017_parquet.py <parquet_dir> <out_dir> [--keep-parquet]

对 40 个文件边提取边删除 parquet（--keep-parquet 则保留），控制磁盘峰值。
"""

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq


def extract_one(parquet_path: Path, images_dir: Path, labels_dir: Path) -> tuple[int, int, set[int]]:
    """提取单个 parquet：返回 (图数, 空标注图数, 类别集合)。"""
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(parquet_path)
    n_img = n_empty = 0
    cats: set[int] = set()
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg)
        ids = t.column(0).to_pylist()
        imgs = t.column(1).to_pylist()
        ws = t.column(2).to_pylist()
        hs = t.column(3).to_pylist()
        objs = t.column(4).to_pylist()
        for iid, img, w, h, obj in zip(ids, imgs, ws, hs, objs):
            img_bytes = img["bytes"]
            if not img_bytes:  # 防御：空 bytes 跳过
                continue
            img_path = images_dir / f"{iid:012d}.jpg"
            img_path.write_bytes(img_bytes)
            lines = []
            for cat, box in zip(obj["category"], obj["bbox"]):
                x1, y1, x2, y2 = box
                # 裁剪到图内，过滤无效框（ultralytics 对越界/退化框敏感）
                x1 = max(0.0, min(float(x1), float(w)))
                x2 = max(0.0, min(float(x2), float(w)))
                y1 = max(0.0, min(float(y1), float(h)))
                y2 = max(0.0, min(float(y2), float(h)))
                if x2 <= x1 or y2 <= y1:
                    continue
                cx = (x1 + x2) / 2.0 / w
                cy = (y1 + y2) / 2.0 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{cat} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                cats.add(int(cat))
            label_path = labels_dir / f"{iid:012d}.txt"
            if lines:
                label_path.write_text("\n".join(lines) + "\n")
            else:
                label_path.write_text("")  # 空标注图补空文件（与 val2017 口径一致）
                n_empty += 1
            n_img += 1
    return n_img, n_empty, cats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parquet_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--keep-parquet", action="store_true")
    args = ap.parse_args()

    images_dir = args.out_dir / "images" / "train2017"
    labels_dir = args.out_dir / "labels" / "train2017"
    total_img = total_empty = 0
    all_cats: set[int] = set()
    # aria2c 跟随 HF CDN 重定向后文件名为 sha256 hex(无 .parquet 后缀),按内容读取而非扩展名
    pq_files = sorted(p for p in args.parquet_dir.iterdir() if p.is_file() and not p.name.endswith(".aria2"))
    for pq_path in pq_files:
        n_img, n_empty, cats = extract_one(pq_path, images_dir, labels_dir)
        total_img += n_img
        total_empty += n_empty
        all_cats |= cats
        if not args.keep_parquet:
            pq_path.unlink()
        print(f"  {pq_path.name}: {n_img} 图 / {n_empty} 空标注 | 累计 {total_img}", flush=True)

    print(f"\n完成: 共 {total_img} 图, 空标注 {total_empty} 个")
    print(f"类别范围: min={min(all_cats)} max={max(all_cats)} (期望 0..79)")
    if all_cats - set(range(80)):
        print(f"⚠️ 发现异常类别(超出 0-79): {sorted(all_cats - set(range(80)))}")
        sys.exit(1)


if __name__ == "__main__":
    main()
