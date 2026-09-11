"""UAVDT (UAV-benchmark-M) → YOLO 格式转换(F11 T6R.3 跨域评估专用)。

跨域协议(v2 设计):
    源域  : VisDrone 训练(T4R 主线产物,不重训)
    目标域: UAVDT 仅评估不训练(zero-shot 跨域泛化)
    类映射: UAVDT car(1)→VisDrone car(idx 3) / truck(2)→truck(idx 5) / bus(3)→bus(idx 8)
    评估  : YOLO val 时 classes=[3,5,8] 过滤预测,只评公共类

UAVDT 官方标注格式(每序列一个 GT 文件,逗号分隔):
    <frame_index>,<target_id>,<box_left>,<box_top>,<box_w>,<box_h>,<score>,<object_category>,<truncation_ratio>
    frame_index 为 1-based;object_category: 1=car, 2=truck, 3=bus

用法示例:
    python scripts/convert_uavdt_to_yolo.py --src "G:\\DATASETS\\UAV-benchmark-M" --stride 30
    # 输出: <out>/images/val + <out>/labels/val + UAVDT.yaml
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image

# UAVDT 类别 → VisDrone 10 类 yaml 中的索引(1-based 官方类目 → 0-based YOLO idx)
CATEGORY_MAP = {1: 3, 2: 5, 3: 8}  # car→car, truck→truck, bus→bus


def find_gt_file(seq_dir: Path, seq_name: str) -> Path | None:
    """在多种官方/解压布局中定位该序列的 DET 标注文件。"""
    candidates = [
        seq_dir / "GT" / f"{seq_name}_GT_DET.txt",
        seq_dir / "GT" / f"{seq_name}_GT.txt",
        seq_dir / "GT_DET.txt",
        seq_dir / f"{seq_name}_GT_DET.txt",
        seq_dir / f"{seq_name}_GT.txt",
    ]
    gt_dir = seq_dir.parent / "GT"
    if gt_dir.is_dir():
        candidates += [gt_dir / f"{seq_name}_GT_DET.txt", gt_dir / f"{seq_name}_GT.txt"]
    for cand in candidates:
        if cand.is_file():
            return cand
    # 兜底:按模式全局搜一次(解压布局多变)
    hits = sorted(seq_dir.glob("**/*GT*.txt"))
    return hits[0] if hits else None


def parse_gt_lines(gt_path: Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """解析 GT 文件,返回 {frame_index: [(yolo_cls, x, y, w, h), ...]}(像素坐标)。"""
    frames: dict[int, list[tuple[int, float, float, float, float]]] = {}
    for raw in gt_path.read_text(errors="ignore").splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 9:
            continue  # 跳过表头或畸形行
        try:
            frame_idx = int(float(parts[0]))
            x, y, w, h = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
            category = int(float(parts[7]))
        except ValueError:
            continue  # 非数值行(表头等)
        if category not in CATEGORY_MAP or w <= 0 or h <= 0:
            continue  # 非公共类或无效框
        frames.setdefault(frame_idx, []).append((CATEGORY_MAP[category], x, y, w, h))
    return frames


def convert(args: argparse.Namespace) -> None:
    """执行转换:抽样帧复制 + 生成 YOLO 标签 + 写 UAVDT.yaml。"""
    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    img_dir = out / "images" / "val"
    lbl_dir = out / "labels" / "val"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # 1) 发现全部序列目录(含 img*.jpg 的目录)
    seq_dirs = sorted(d for d in src.iterdir() if d.is_dir() and list(d.glob("img*.jpg")))
    if not seq_dirs:
        raise SystemExit(f"未在 {src} 下发现含 img*.jpg 的序列目录,请检查解压结构")
    print(f"[UAVDT] 发现 {len(seq_dirs)} 个序列")

    total_frames, total_boxes = 0, 0
    per_class = {3: 0, 5: 0, 8: 0}
    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        gt_path = find_gt_file(seq_dir, seq_name)
        if gt_path is None:
            print(f"[UAVDT][WARN] {seq_name}: 未找到 GT 文件,跳过")
            continue
        gt_frames = parse_gt_lines(gt_path)
        images = sorted(seq_dir.glob("img*.jpg"))

        # 2) 均匀抽帧(stride),仅保留有标注的帧
        selected = images[:: args.stride]
        kept = 0
        for img_path in selected:
            frame_idx = int(img_path.stem.replace("img", ""))
            boxes = gt_frames.get(frame_idx, [])
            if not boxes:
                continue  # 无公共类标注的帧不进评估集
            stem = f"{seq_name}_{img_path.stem}"
            shutil.copy2(img_path, img_dir / f"{stem}.jpg")

            # 3) 像素坐标 → YOLO 归一化 cx cy w h
            with Image.open(img_path) as im:
                img_w, img_h = im.size
            lines = []
            for cls, x, y, w, h in boxes:
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w / img_w:.6f} {h / img_h:.6f}")
                per_class[cls] += 1
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="ascii")
            total_boxes += len(lines)
            kept += 1
        total_frames += kept
        print(f"[UAVDT] {seq_name}: 抽样 {len(selected)} 帧 → 保留 {kept} 帧(有标注)")

    # 4) 写评估用 yaml(names 与 VisDrone 10 类完全对齐)
    yaml_path = out / "UAVDT.yaml"
    yaml_path.write_text(
        "# F11 T6R.3 跨域评估专用(由 convert_uavdt_to_yolo.py 生成)\n"
        "# 仅评估不训练;公共类 GT: car(3)/truck(5)/bus(8),预测需 classes=[3,5,8] 过滤\n"
        f"path: {out.as_posix()}\n"
        "train: images/val\n"
        "val: images/val\n"
        "names:\n"
        "  0: pedestrian\n  1: people\n  2: bicycle\n  3: car\n  4: van\n"
        "  5: truck\n  6: tricycle\n  7: awning-tricycle\n  8: bus\n  9: motor\n",
        encoding="utf-8",
    )
    print(f"[UAVDT] 完成: {total_frames} 帧 / {total_boxes} 框 | car={per_class[3]} truck={per_class[5]} bus={per_class[8]}")
    print(f"[UAVDT] yaml → {yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="UAVDT(UAV-benchmark-M) → YOLO 转换")
    parser.add_argument("--src", required=True, help="UAV-benchmark-M 解压根目录")
    parser.add_argument("--out", default=r"G:\Codes\OpenSource\Rhino-bird\DATASETS\UAVDT", help="输出目录")
    parser.add_argument("--stride", type=int, default=30, help="每 stride 帧抽 1 帧(默认 30)")
    convert(parser.parse_args())


if __name__ == "__main__":
    main()
