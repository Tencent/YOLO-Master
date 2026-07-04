#!/usr/bin/env python3
# Run the MNN model over the val set and dump per-image preds ('class conf x1 y1 x2 y2')
# using the SAME decode as the C++ runner (multi-label, conf 0.001, per-class NMS iou 0.7,
# cap 300) so eval_map.py yields a mAP directly comparable to the ONNX/ncnn/PyTorch numbers.
import MNN, numpy as np, cv2, glob, os, argparse

def letterbox(path, sz=640):
    img = cv2.imread(path); h, w = img.shape[:2]
    r = min(sz / w, sz / h); nw, nh = round(w * r), round(h * r)
    c = np.full((sz, sz, 3), 114, np.uint8); px, py = (sz - nw) // 2, (sz - nh) // 2
    c[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh))
    x = np.ascontiguousarray(np.transpose(c[:, :, ::-1].astype(np.float32) / 255, (2, 0, 1))[None])
    return x, r, px, py, w, h

def nms(boxes, scores, iou_thr):                     # greedy, score-desc; matches C++ nms_greedy
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]; keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep

ap = argparse.ArgumentParser()
ap.add_argument("--mnn", default="models/esmoe_n_visdrone.mnn")
ap.add_argument("--out", default="preds_mnn")
ap.add_argument("--conf", type=float, default=0.001)
ap.add_argument("--iou", type=float, default=0.7)
ap.add_argument("--images", default="/data/datasets/VisDrone/images/val")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

interp = MNN.Interpreter(a.mnn); sess = interp.createSession({"numThread": 4, "backend": "CPU"})
inp = interp.getSessionInput(sess)
def run(x):
    t = MNN.Tensor((1, 3, 640, 640), MNN.Halide_Type_Float, x, MNN.Tensor_DimensionType_Caffe)
    inp.copyFrom(t); interp.runSession(sess)
    o = interp.getSessionOutput(sess); sh = o.getShape()
    ot = MNN.Tensor(sh, MNN.Halide_Type_Float, np.zeros(sh, np.float32), MNN.Tensor_DimensionType_Caffe)
    o.copyToHostTensor(ot); return np.array(ot.getData(), np.float32).reshape(sh)[0]   # (14, 8400)

OFF = 8192.0
for p in sorted(glob.glob(os.path.join(a.images, "*.jpg"))):
    x, r, px, py, w, h = letterbox(p)
    y = run(x); box, cls = y[:4], y[4:]                 # (4,8400),(10,8400)
    cs, an = np.where(cls >= a.conf)                    # multi-label candidates
    if cs.size:
        sc = cls[cs, an]
        cx, cy, bw, bh = box[0, an], box[1, an], box[2, an], box[3, an]
        x1 = (cx - 0.5 * bw - px) / r; y1 = (cy - 0.5 * bh - py) / r
        x2 = (cx + 0.5 * bw - px) / r; y2 = (cy + 0.5 * bh - py) / r
        xyxy = np.stack([x1, y1, x2, y2], 1)
        off = xyxy + (cs[:, None] * OFF)               # per-class NMS via offset
        keep = nms(off, sc, a.iou)[:300]               # cap 300 (ultralytics val)
    else:
        keep = []
    with open(os.path.join(a.out, os.path.splitext(os.path.basename(p))[0] + ".txt"), "w") as f:
        for i in keep:
            bx1 = max(0.0, min(x1[i], w)); by1 = max(0.0, min(y1[i], h))
            bx2 = max(0.0, min(x2[i], w)); by2 = max(0.0, min(y2[i], h))
            if bx2 > bx1 and by2 > by1:
                f.write(f"{cs[i]} {sc[i]:.6f} {bx1:.2f} {by1:.2f} {bx2:.2f} {by2:.2f}\n")
print(f"dumped preds -> {a.out}")
