#!/usr/bin/env python3
"""F11 · T5 零教师依赖证明 · 步骤 3: 推理期精度对比

依据 06 文档 §7.3:
  用纯学生模型推理,mAP 应与训练期(教师存在)一致(差 < 0.1)。

用法:
  python scripts/eval_student.py --student runs/f11_student.pt --data coco8.yaml

参考:
  - ultralytics 标准 val 流程

文档版本：v1.0（2026-08-24）｜Owner：张伟林（Zviolin）｜F11 T5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yolo_master_matplotlib")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F11 · 推理期精度对比")
    parser.add_argument("--student", required=True, help="纯学生权重路径")
    parser.add_argument("--data", default="coco8.yaml", help="数据集 yaml")
    parser.add_argument("--device", default="0", help="验证设备(默认 0)")
    parser.add_argument("--imgsz", type=int, default=640, help="推理分辨率")
    parser.add_argument("--out", default="runs/f11_student_eval", help="输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    student_path = Path(args.student).resolve()
    if not student_path.exists():
        raise SystemExit(f"学生权重不存在: {student_path}")

    device = args.device if args.device != "0" else ("cuda:0" if __import__("torch").cuda.is_available() else "cpu")
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(student_path))
    results = model.val(data=args.data, device=device, imgsz=args.imgsz, project=str(out_dir), name="student")

    summary = {
        "student": str(student_path),
        "data": args.data,
        "mAP50-95": float(results.box.map if hasattr(results, "box") else results.results_dict.get("metrics/mAP50-95(B)", 0)),
        "mAP50": float(results.box.map50 if hasattr(results, "box") else results.results_dict.get("metrics/mAP50(B)", 0)),
    }
    report_path = out_dir / "eval_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[F11][T5] ✅ 推理期评估完成")
    print(f"[F11][T5] mAP50-95 = {summary['mAP50-95']:.4f}")
    print(f"[F11][T5] mAP50    = {summary['mAP50']:.4f}")
    print(f"[F11][T5] report: {report_path}")


if __name__ == "__main__":
    main()
