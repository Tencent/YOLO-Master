"""A1 S2 部署闭环:20k 四格 PT vs ONNX 导出 parity 验证。

对比方式(沿用 S1 verify_export_parity.py 口径):
同一真实图输入(640)分别经 PT forward 与 onnxruntime 推理,
比较输出最大绝对误差。FP32 下应 < 1e-2(slim 与 BN 折叠引入的小幅误差可容忍)。

- 真实图输入:分数连续无并列,topk 顺序确定(随机输入下 PT/ONNX 并列排序不同)
- 端到端格(C/D)输出 BNC (1,300,6);A/B 格输出稠密 (1,84,8400)
- ONNX 由 ultralytics 标准导出(imgsz 640,FP32)

用法:
    python scripts/a1/verify_export_parity_20k.py --model <A.pt> <B.pt> <C.pt> <D.pt>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # YOLO-Master 仓库根
sys.path.insert(0, str(REPO_ROOT))

os.environ["WANDB_MODE"] = "disabled"
os.environ.setdefault("YOLO_VERBOSE", "false")

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

IMGSZ = 640


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", nargs="+", required=True, help="权重路径(四格)")
    ap.add_argument("--source", default=f"{REPO_ROOT}/ultralytics/assets/bus.jpg", help="真实图输入")
    return ap.parse_args()


def preprocess(path: str) -> torch.Tensor:
    """与推理管线一致的预处理:letterbox → /255 → BCHW float32。"""
    img = cv2.imread(path)
    img = LetterBox(new_shape=(IMGSZ, IMGSZ), auto=False, stride=32)(image=img)
    x = torch.from_numpy(img.transpose(2, 0, 1).copy()).unsqueeze(0).float() / 255.0
    return x


def main() -> None:
    """逐格导出 ONNX 并对比 PT/ORT 输出。"""
    args = parse_args()
    x = preprocess(args.source)

    for pt in args.model:
        yolo = YOLO(pt)
        # PT 前向(未 fuse,与 S1 parity 口径一致)
        model_pt = yolo.model
        model_pt.eval()
        with torch.no_grad():
            y_pt = model_pt(x)
            if isinstance(y_pt, (tuple, list)):
                y_pt = y_pt[0]
        y_pt = y_pt.float().numpy()

        # 导出 ONNX(标准导出路径,内部会 fuse)
        onnx_path = yolo.export(format="onnx", imgsz=IMGSZ, opset=13, dynamic=False)
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        y_onnx = sess.run(None, {input_name: x.numpy()})[0]

        print(f"\n== {pt} ==")
        print(f"  PT 输出 shape: {y_pt.shape} | ONNX 输出 shape: {y_onnx.shape}")
        if y_pt.shape != y_onnx.shape:
            print("  ⚠️ 输出 shape 不一致,无法逐元素对比")
            continue
        max_abs = float(np.abs(y_pt - y_onnx).max())
        # 框级匹配率(BNC 6 列:前 4 框坐标 + conf + cls)
        n_rows = min(y_pt.shape[1], y_onnx.shape[1])
        row_diff = np.abs(y_pt[:, :n_rows] - y_onnx[:, :n_rows]).max(axis=-1)
        matched = float((row_diff < 1e-2).mean() * 100)
        status = "✅" if max_abs < 1e-2 else "⚠️ 超阈值"
        print(f"  最大绝对误差: {max_abs:.6f} | 行级匹配率(<1e-2): {matched:.1f}% | {status}")

        Path(onnx_path).unlink(missing_ok=True)  # 清理导出产物(仅验证用)
        print("  ONNX 已清理")


if __name__ == "__main__":
    main()
