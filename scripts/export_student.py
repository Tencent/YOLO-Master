#!/usr/bin/env python3
"""F11 · T5 零教师依赖证明 · 步骤 1: 导出纯学生模型

依据 06 文档 §7.3:
  1. export_student.py   --ckpt 训练权重 → 剥离 Foundation 包装 → 纯学生
  2. verify_no_teacher.py --student 验证教师完全移除
  3. eval_student.py      --student 推理期精度对比

用法:
  python scripts/export_student.py --ckpt runs/f11_ablation/c-router-kd/weights/best.pt \
      --out runs/f11_student.pt

参考:
  - ultralytics.nn.foundation_distill_model.strip_foundation_distillation_model

文档版本：v1.0（2026-08-24）｜Owner：张伟林（Zviolin）｜F11 T5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yolo_master_matplotlib")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F11 · 导出纯学生模型(剥离教师)")
    parser.add_argument("--ckpt", required=True, help="训练权重路径(best.pt)")
    parser.add_argument("--out", default="runs/f11_student.pt", help="导出路径")
    parser.add_argument("--device", default="cpu", help="导出设备(默认 cpu)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = Path(args.ckpt).resolve()
    out = Path(args.out).resolve()
    if not ckpt.exists():
        raise SystemExit(f"权重不存在: {ckpt}")
    out.parent.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(str(ckpt))
    student = model.model
    # 剥离 Foundation 蒸馏包装,返回纯学生模型(教师彻底移除)
    from ultralytics.nn.foundation_distill_model import strip_foundation_distillation_model

    stripped = strip_foundation_distillation_model(student)

    # 保存为可加载权重(以 student 结构重建 YOLO 兼容 ckpt)
    stripped_ckpt = {
        "model": stripped,
        "train_args": getattr(model.train_args, "dict", lambda: {})() if hasattr(model, "train_args") else {},
        "epoch": -1,
        "best_fitness": None,
    }
    torch.save(stripped_ckpt, out)
    print(f"[F11][T5] ✅ 纯学生模型已导出: {out}")
    print(f"[F11][T5] 参数量: {sum(p.numel() for p in stripped.parameters()):,}")


if __name__ == "__main__":
    main()
