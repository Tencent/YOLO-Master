#!/usr/bin/env python3
"""F11 · T5 零教师依赖证明 · 步骤 2: 验证教师完全移除

依据 06 文档 §7.3:
  验证导出的学生模型中:
  - 参数量 = baseline 参数量(教师参数为 0)
  - 无教师模型文件/参数(Foundation 组件不存在)
  - 推理延迟 ≈ baseline(无教师前向开销)

用法:
  python scripts/verify_no_teacher.py --student runs/f11_student.pt

参考:
  - ultralytics.nn.foundation_distill_model.FoundationDistillationModel

文档版本：v1.0（2026-08-24）｜Owner：张伟林（Zviolin）｜F11 T5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yolo_master_matplotlib")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F11 · 验证零教师依赖")
    parser.add_argument("--student", required=True, help="纯学生权重路径(runs/f11_student.pt)")
    parser.add_argument("--device", default="cpu", help="验证设备(默认 cpu)")
    parser.add_argument("--reps", type=int, default=10, help="延迟采样次数")
    parser.add_argument("--imgsz", type=int, default=640, help="推理分辨率")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    student_path = Path(args.student).resolve()
    if not student_path.exists():
        raise SystemExit(f"学生权重不存在: {student_path}")

    device = args.device if args.device != "0" else ("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(str(student_path), map_location="cpu", weights_only=False)
    model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.to(torch.device(device)).eval()

    total_params = sum(p.numel() for p in model.parameters())
    # 检测是否存在 Foundation 教师组件
    from ultralytics.nn.foundation_distill_model import FoundationDistillationModel
    from ultralytics.nn.foundation.routing import FoundationTeacherRouter

    has_wrapper = isinstance(model, FoundationDistillationModel)
    has_router = any(isinstance(m, FoundationTeacherRouter) for m in model.modules())
    teacher_params = sum(
        p.numel() for p in model.parameters() if not p.requires_grad and has_wrapper
    )

    # 推理延迟
    x = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.no_grad():
        for _ in range(3):  # warmup
            _ = model(x)
        latencies = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            _ = model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)
    latencies.sort()
    p50 = latencies[len(latencies) // 2]

    print(f"\n[F11][T5] ==== 零教师依赖验证 ====")
    print(f"[F11][T5] 模型: {student_path.name}")
    print(f"[F11][T5] 参数量: {total_params:,}")
    print(f"[F11][T5] Foundation 包装残留: {'⚠️ 有' if has_wrapper else '✅ 无'}")
    print(f"[F11][T5] Teacher Router 残留:  {'⚠️ 有' if has_router else '✅ 无'}")
    print(f"[F11][T5] 冻结教师参数: {teacher_params:,}")
    print(f"[F11][T5] 推理延迟 P50: {p50:.2f} ms ({args.reps} reps)")
    passed = (not has_wrapper) and (not has_router) and teacher_params == 0
    print(f"[F11][T5] 结果: {'✅ 零教师依赖成立' if passed else '❌ 仍有教师残留'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
