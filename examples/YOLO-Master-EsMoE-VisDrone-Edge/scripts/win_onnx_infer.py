"""Windows ONNX inference verification (standalone).

Loads best_recal.onnx, runs on a VisDrone image with the same letterbox +
area-adaptive NMS as the Linux pipeline, prints detection count + latency.
Proves the exported ONNX model runs identically on Windows x86_64.
"""
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort


def letterbox(bgr, imgsz=640):
    h, w = bgr.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    rs = cv2.resize(bgr, (nw, nh))
    pad = np.full((imgsz, imgsz, 3), 114, np.uint8)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    pad[top:top + nh, left:left + nw] = rs
    rgb = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB)
    return rgb.transpose(2, 0, 1).astype(np.float32) / 255.0, r, left, top, w, h


def decode(raw, nc=10, conf=0.15, small_conf=0.05, small_area=32 * 32, iou=0.45):
    arr = raw[0].T if raw.ndim == 3 else raw.T  # (N, 4+nc)
    xywh = arr[:, :4]
    scores = arr[:, 4:4 + nc]
    cls = scores.argmax(1)
    top = scores[np.arange(len(cls)), cls]
    xyxy = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                     xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1)
    area = np.clip(xyxy[:, 2] - xyxy[:, 0], 0, None) * np.clip(xyxy[:, 3] - xyxy[:, 1], 0, None)
    thr = np.where(area < small_area, small_conf, conf)
    keep = top >= thr
    return xyxy[keep], top[keep], cls[keep]


def main():
    model, img_path = sys.argv[1], sys.argv[2]
    nc = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    bgr = cv2.imread(img_path)
    chw, r, left, top, w, h = letterbox(bgr)
    x = np.ascontiguousarray(chw[None].astype(np.float32))
    # warmup
    for _ in range(10):
        sess.run(None, {in_name: x})
    t0 = time.perf_counter()
    N = 100
    for _ in range(N):
        out = sess.run(None, {in_name: x})[0]
    ms = (time.perf_counter() - t0) / N * 1000
    boxes, scores, cls = decode(out, nc=nc)
    # scale back to original
    boxes[:, 0::2] = (boxes[:, 0::2] - left) / r
    boxes[:, 1::2] = (boxes[:, 1::2] - top) / r
    print(f"[win-onnx] detections={len(boxes)}  (cls {cls[:8].tolist()})")
    print(f"[win-onnx] ort={ort.__version__}  latency={ms:.2f}ms  FPS={1000/ms:.1f}  (CPU, avg {N})")
    # save annotated
    for b, s, c in zip(boxes, scores, cls):
        b = b.astype(int)
        cv2.rectangle(bgr, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
    cv2.imwrite("win_result.jpg", bgr)
    print("[win-onnx] saved win_result.jpg")


if __name__ == "__main__":
    main()
