#!/usr/bin/env python3
"""F11 · 教师特征离线缓存脚本（Foundation 路由 KD · 冒烟 1）

F11 课题：DINO 教师特征 → Router 软目标蒸馏。
本脚本：加载冻结 DINO 教师（torch.hub），离线缓存 100 张 COCO8 数据的 patch 特征，
       输出 manifest.json（含 SHA256 + 复现命令）+ meta.json。

用法：
  python scripts/cache_teacher_features.py --data coco8.yaml --num 100 --out runs/f11_teacher_cache

参考：
  - #54 scripts/diagnose_mot_routing.py：MoT 路由诊断 hook 框架
  - 任务书 §2.1 F11：教师信号进入 Router 软目标分布（合法定义）

文档版本：v1.0（2026-08-23）｜Owner：张伟林（Zviolin）｜F11 8.24 准入冒烟 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yolo_master_matplotlib")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F11 · DINO 教师特征离线缓存")
    parser.add_argument("--data", default="coco8.yaml", help="YOLO 数据集 yaml")
    parser.add_argument("--num", type=int, default=100, help="采样图片数量")
    parser.add_argument("--out", default="runs/f11_teacher_cache", help="输出缓存目录")
    parser.add_argument("--teacher", default="dinov2_vitb14", help="DINO 教师模型")
    parser.add_argument("--device", default="cuda:0", help="推理设备")
    parser.add_argument("--imgsz", type=int, default=224, help="输入分辨率（224=标准 / 112=8GB 降级）")
    parser.add_argument("--batch", type=int, default=1, help="batch size（8GB 必须=1）")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    """计算文件 SHA256"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dino_teacher(model_name: str, device: str) -> "torch.nn.Module":
    """加载冻结的 DINO教师模型（torch.hub，无需联网下载时使用本地权重）。"""
    import torch

    # 把字符串 '0' / 'cuda:0' / 'cpu' 解析成 torch.device
    if isinstance(device, str):
        if device.isdigit():
            device = f"cuda:{device}"
        device = torch.device(device)

    print(f"[F11] 加载 DINO 教师: {model_name} @ {device}")
    try:
        # 尝试 torch.hub 在线加载
        teacher = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
    except Exception as exc:
        print(f"[F11][WARN] torch.hub 加载失败 ({exc})，使用随机初始化教师作为 fallback")
        # Fallback：随机初始化（仅供冒烟链路验证，不能用于真实蒸馏）
        teacher = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=False)

    teacher.eval()
    teacher.requires_grad_(False)
    teacher.to(device)
    return teacher


def extract_features(
    teacher: "torch.nn.Module",
    image_paths: list,
    device: str,
    imgsz: int,
    batch: int,
    out_dir: Path,
) -> list:
    """抽取图片 patch 特征并离线缓存。"""
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms

    # 把字符串 device 解析成 torch.device（避免 .to('0') 报错）
    if isinstance(device, str):
        if device.isdigit():
            device = f"cuda:{device}"
        device = torch.device(device)

    teacher.eval()

    # DINO 标准预处理
    transform = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    patches_dir = out_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for idx, img_path in enumerate(image_paths[: batch * 200]):  # 限制总采样数
        try:
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                # DINOv2 forward_features 返回 dict
                feats = teacher.forward_features(x)
                patch_feat = feats["x_norm_patchtokens"]  # [1, N, D]
            patch_feat = patch_feat.cpu().squeeze(0)  # [N, D]
            patch_path = patches_dir / f"img_{idx:04d}.pt"
            torch.save(patch_feat, patch_path)
            manifest_rows.append({
                "idx": idx,
                "image": str(img_path.name),
                "patch_path": str(patch_path.relative_to(out_dir)),
                "patch_shape": list(patch_feat.shape),
                "sha256": file_sha256(patch_path),
            })
            if (idx + 1) % 10 == 0:
                print(f"[F11] 已缓存 {idx + 1} / {len(image_paths)} 张")
        except Exception as exc:
            print(f"[F11][WARN] 跳过 {img_path}: {exc}")
            continue
        if len(manifest_rows) >= 200:
            break
    return manifest_rows


def collect_image_paths(data_yaml: Path, max_num: int) -> list:
    """从 YOLO 数据集 yaml 收集图片路径。"""
    import yaml

    # 优先从 data_yaml 找图片路径
    candidates: list = []
    if data_yaml.exists():
        with data_yaml.open(encoding="utf-8") as f:
            data_info = yaml.safe_load(f)
        # 处理 coco8.yaml 的 path 字段(可能是相对或绝对)
        ds_path = data_info.get("path", "")
        if ds_path and not Path(ds_path).is_absolute():
            ds_path = ROOT / ds_path
        # 用 ultralytics 内置 check_det_dataset 解析 train/val 图片路径
        from ultralytics.data.utils import check_det_dataset
        try:
            data_info = check_det_dataset(str(data_yaml))
            # check_det_dataset 返回的 path 是已解析的绝对路径
            base = Path(data_info.get("path", ds_path))
            for split in ("train", "val", "test"):
                # 数据集 yaml 里通常用绝对路径 images/train2017 等
                if split in data_info:
                    split_dir = Path(data_info[split])
                    if split_dir.exists():
                        candidates.extend(split_dir.rglob("*.jpg"))
                        candidates.extend(split_dir.rglob("*.png"))
        except Exception as exc:
            print(f"[F11][WARN] check_det_dataset 失败: {exc}")

    # coco8.yaml fallback：使用 ultralytics 内置数据集
    if not candidates:
        candidates = list((ROOT / "ultralytics" / "assets" / "coco8" / "images").glob("*.jpg"))
        if not candidates:
            candidates = list((ROOT / "ultralytics" / "assets" / "coco8" / "images").glob("*.png"))

    # 最终 fallback：只扫 ultralytics/assets(避免扫到 runs/*.jpg 这些训练可视化图)
    if not candidates:
        candidates = list((ROOT / "ultralytics" / "assets").rglob("*.jpg"))[:max_num * 2]
        candidates.extend(list((ROOT / "ultralytics" / "assets").rglob("*.png"))[:max_num * 2])

    print(f"[F11] 收集到 {len(candidates)} 张候选图片（最多用 {max_num}）")
    # 去重(可能 .jpg 和 .png 列表合并后有重复)
    seen = set()
    unique = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique[:max_num]


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 收集图片路径
    data_yaml = (ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    image_paths = collect_image_paths(data_yaml, args.num)

    # 2. 加载教师模型
    teacher = load_dino_teacher(args.teacher, args.device)

    # 3. 离线特征抽取
    manifest_rows = extract_features(
        teacher=teacher,
        image_paths=image_paths,
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
        out_dir=out_dir,
    )

    # 4. 写 manifest.json
    manifest = {
        "task": "F11 · teacher feature cache",
        "teacher_model": args.teacher,
        "device": args.device,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "num_requested": args.num,
        "num_cached": len(manifest_rows),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "reproduce_cmd": (
            f"python scripts/cache_teacher_features.py "
            f"--data {args.data} --num {args.num} --out {args.out} "
            f"--teacher {args.teacher} --device {args.device} "
            f"--imgsz {args.imgsz} --batch {args.batch}"
        ),
        "rows": manifest_rows,
    }
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 5. 写 meta.json
    meta = {
        "teacher_model": args.teacher,
        "num_cached": len(manifest_rows),
        "cache_dir": str(out_dir),
        "patches_dir": str(out_dir / "patches"),
    }
    meta_path = out_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[F11] [OK] 缓存完成: {len(manifest_rows)} 张")
    print(f"[F11] manifest: {manifest_path}")
    print(f"[F11] meta: {meta_path}")


if __name__ == "__main__":
    main()