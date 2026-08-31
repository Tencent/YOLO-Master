"""A1 S1 阶段1：导出 parity——C 格 PT 与 ONNX 同图输出一致性对比。

对比方式：同一输入张量分别经 PT 模型 forward 与 onnxruntime 推理，
比较 BNC 输出（shape (1, max_det, 6)，即 x1y1x2y2/conf/cls）的最大绝对误差。
FP32 导出下 PT vs ONNX 数值误差应 < 1e-2（slim 与算子差异引入的小幅误差可容忍）。

用法：
    python scripts/a1/verify_export_parity.py
"""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

_ROOT = Path(__file__).resolve().parents[2]
_PT = "runs/detect/runs/agent/a1-s1-smoke-c/weights/best.pt"
_ONNX = "runs/detect/runs/agent/a1-s1-smoke-c/weights/best.onnx"
_SOURCE = "ultralytics/assets/bus.jpg"


def preprocess(path: str, imgsz: int = 64) -> np.ndarray:
    """与推理管线一致的预处理：letterbox → /255 → BCHW（float32）。"""
    img = cv2.imread(path)
    img = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)(image=img)
    x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    return x


def main() -> None:
    # 真实图输入：分数连续无并列，topk 顺序确定，行对齐对比才有意义
    # （随机输入下分数全并列，PT 与 ONNX 的 TopK 并列顺序不同导致行序错位）
    x = preprocess(str(_ROOT / _SOURCE))
    assert x.shape == (1, 3, 64, 64), f"输入 shape 异常: {x.shape}"

    # PT 前向（eval 推理路径，end2end 时返回 (y, preds) 元组，取 BNC 输出 y）
    # 注意：YOLO(...).model 加载后权重在 CPU，输入也放 CPU 保持一致
    model = YOLO(str(_ROOT / _PT)).model
    model.eval()
    with torch.no_grad():
        y_pt = model(x)
        if isinstance(y_pt, (tuple, list)):
            y_pt = y_pt[0]
    y_pt = y_pt.float().numpy()

    # ONNX 推理（CPU provider，与 PT 侧对齐）
    sess = ort.InferenceSession(str(_ROOT / _ONNX), providers=["CPUExecutionProvider"])
    y_onnx = sess.run(None, {"images": x.float().numpy()})[0]

    max_diff = float(np.abs(y_pt - y_onnx).max())
    print(f"PT 输出 shape: {y_pt.shape}")
    print(f"ONNX 输出 shape: {y_onnx.shape}")
    print(f"最大绝对误差（行对齐）: {max_diff:.6f}")
    # 诊断：全部数值展平排序后对比。
    # 误差≈0 → 数值集合相同仅顺序不同（topk 排序差异）；
    # 误差仍大 → 两边数值集合真实不同（计算路径/权重差异）
    flat_diff = float(np.abs(np.sort(y_pt.reshape(-1)) - np.sort(y_onnx.reshape(-1))).max())
    print(f"最大绝对误差（展平排序后）: {flat_diff:.6f}")
    assert y_pt.shape == y_onnx.shape, f"输出 shape 不一致: {y_pt.shape} vs {y_onnx.shape}"
    assert max_diff < 1e-2, f"误差超容差: {max_diff}"
    print("导出 parity 通过 ✅  PT 与 ONNX 输出一致")


if __name__ == "__main__":
    main()
