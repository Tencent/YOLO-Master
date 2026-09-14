#!/usr/bin/env python3
"""F11 · 8.24 准入证据包归档脚本（Foundation 路由 KD · 冒烟 4）

F11 课题：收集冒烟 1-3 的所有证据（训练命令 + loss 日志 + 路由分布图 + 显存估算），
       打包到 reports/f11_evidence/，用于 8.24 准入提交。

用法：
  python scripts/archive_evidence.py --output reports/f11_evidence/

参考：
  - #54 scripts/compare_mot_ablation.py：实验产物归档方法
  - 任务书 §2.1 F11：提交训练命令 + loss 日志 + 路由分布图 + 显存估算

文档版本：v1.0（2026-08-23）｜Owner：张伟林（Zviolin）｜F11 8.24 准入冒烟 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F11 · 8.24 准入证据包归档")
    parser.add_argument("--output", default="reports/f11_evidence", help="证据包输出目录")
    parser.add_argument(
        "--cache_dir", default="runs/f11_teacher_cache", help="冒烟 1 输出"
    )
    parser.add_argument(
        "--q_teacher_dir", default="runs/f11_q_teacher_check", help="冒烟 2 输出"
    )
    parser.add_argument("--smoke_kd_dir", default="runs/f11_smoke_kd", help="冒烟 3 输出")
    parser.add_argument("--smoke_dir", default="runs/smoke", help="基线 smoke 输出")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_file(path: Path, dest_dir: Path, category: str) -> dict:
    """复制文件到证据包，返回元信息。"""
    if not path.exists():
        return {"path": str(path), "exists": False, "sha256": None, "size": 0}
    rel = path.relative_to(ROOT) if path.is_absolute() and path.is_relative_to(ROOT) else path.name
    dest = dest_dir / category / path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "dest": str(dest.relative_to(dest_dir)),
        "exists": True,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def collect_directory(src_dir: Path, dest_dir: Path, category: str, pattern: str = "*") -> list:
    """复制目录下匹配 pattern 的文件。"""
    if not src_dir.exists():
        return []
    files = []
    for src in sorted(src_dir.rglob(pattern)):
        if src.is_file():
            rel = src.relative_to(src_dir)
            dest = dest_dir / category / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            files.append(str(rel))
    return files


def collect_evidence(args) -> dict:
    """收集冒烟 1-4 的所有证据文件。"""
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    evidence = {
        "task": "F11 · 8.24 准入证据包",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "outputs": {},
    }

    # 1. 基线 smoke（冒烟 0）
    smoke_dir = Path(args.smoke_dir).resolve()
    if smoke_dir.exists():
        evidence["outputs"]["smoke"] = {
            "description": "基线 smoke（coco8 1 epoch）",
            "files": collect_directory(smoke_dir, output, "smoke", "results.csv"),
        }

    # 2. 冒烟 1：教师特征缓存
    cache_dir = Path(args.cache_dir).resolve()
    if cache_dir.exists():
        evidence["outputs"]["teacher_cache"] = {
            "description": "DINO 教师特征离线缓存（100 张）",
            "files": [
                collect_file(cache_dir / "manifest.json", output, "teacher_cache"),
                collect_file(cache_dir / "meta.json", output, "teacher_cache"),
            ] + collect_directory(cache_dir, output, "teacher_cache", "*.pt")[:50],  # 限制 patch 数量
        }

    # 3. 冒烟 2：q_teacher
    q_dir = Path(args.q_teacher_dir).resolve()
    if q_dir.exists():
        evidence["outputs"]["q_teacher"] = {
            "description": "q_teacher 软目标 + 熵校验",
            "files": [
                collect_file(q_dir / "report.json", output, "q_teacher"),
                collect_file(q_dir / "entropy.csv", output, "q_teacher"),
                collect_file(q_dir / "q_teacher.pt", output, "q_teacher"),
            ],
        }

    # 4. 冒烟 3：Router KD 闭环
    kd_dir = Path(args.smoke_kd_dir).resolve()
    if kd_dir.exists():
        evidence["outputs"]["router_kd"] = {
            "description": "Router KD 训练闭环（单 batch 冒烟）",
            "files": [
                collect_file(kd_dir / "smoke_report.json", output, "router_kd"),
            ],
        }

    # 5. 提交命令清单
    commands = [
        "# 冒烟 0：基线 smoke",
        "python -c \"from ultralytics import YOLO; m=YOLO('ultralytics/cfg/models/master/v0_8/det/yolo-master-n.yaml'); m.train(data='coco8.yaml', epochs=1, imgsz=640, batch=8, device=0, project='runs/smoke')\"",
        "",
        "# 冒烟 1：教师特征缓存",
        "python scripts/cache_teacher_features.py --data coco8.yaml --num 100 --out runs/f11_teacher_cache",
        "",
        "# 冒烟 2：q_teacher 非退化",
        "python scripts/gen_q_teacher.py --cache runs/f11_teacher_cache --model yolo-master-n.yaml --device cpu",
        "",
        "# 冒烟 3：Router KD 闭环",
        "python scripts/smoke_router_kd.py --epochs 1 --layers 1 --experts 2 --device 0",
        "",
        "# 冒烟 4：证据归档（本脚本）",
        "python scripts/archive_evidence.py --output reports/f11_evidence/",
    ]
    commands_path = output / "commands.sh"
    with commands_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(commands))

    evidence["outputs"]["commands"] = {
        "description": "冒烟 1-4 完整命令清单",
        "files": [{"path": "commands.sh", "sha256": file_sha256(commands_path)}],
    }

    return evidence


def write_summary(evidence: dict, output: Path) -> None:
    """生成摘要报告。"""
    summary = {
        "F11 8.24 准入证据包": {
            "创建时间": evidence["created_at"],
            "基线": "acce839c7e895d6b179de7f7093fa879e237cc7b",
            "提交人": "张伟林（Zviolin）",
            "课题": "F11 Foundation 路由 KD",
            "证据清单": {
                "冒烟 0（基线 smoke）": "coco8 1 epoch 训练日志",
                "冒烟 1（教师特征缓存）": "DINO 教师特征 + manifest.json",
                "冒烟 2（q_teacher 非退化）": "H_norm ∈ (0.3, 0.8) 校验报告",
                "冒烟 3（Router KD 闭环）": "KD loss + Router grad ≠ 0 验证",
                "冒烟 4（证据归档）": "本脚本",
            },
        }
    }
    summary_path = output / "00-证据包摘要.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    print(f"[F11] === 证据包归档 ===")
    print(f"[F11] output: {output}")

    evidence = collect_evidence(args)

    # 写证据清单
    manifest_path = output / "evidence_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)

    # 写摘要
    write_summary(evidence, output)

    print(f"\n[F11] OK 证据包归档完成")
    print(f"[F11] manifest: {manifest_path}")
    print(f"[F11] 摘要: {output / '00-证据包摘要.json'}")
    print(f"[F11] 命令清单: {output / 'commands.sh'}")
    print(f"[F11] \n各任务证据：")
    for category, info in evidence["outputs"].items():
        file_count = len(info.get("files", []))
        print(f"[F11]   {category}: {file_count} 文件 - {info.get('description', '')}")


if __name__ == "__main__":
    main()