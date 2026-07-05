#!/usr/bin/env python3
# MNN correctness + latency: feed identical letterboxed inputs to the MNN model and
# the source ONNX, compare raw output tensors (same graph -> same mAP, already <0.5%).
import MNN, numpy as np, cv2, glob, time, onnxruntime as ort, argparse

def letterbox(path, sz=640):
    img = cv2.imread(path); h, w = img.shape[:2]
    r = min(sz / h, sz / w); nw, nh = round(w * r), round(h * r)
    c = np.full((sz, sz, 3), 114, np.uint8); px, py = (sz - nw) // 2, (sz - nh) // 2
    c[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh))
    return np.ascontiguousarray(np.transpose(c[:, :, ::-1].astype(np.float32) / 255, (2, 0, 1))[None])

ap = argparse.ArgumentParser()
ap.add_argument("--mnn", default="models/esmoe_n_visdrone.mnn")
ap.add_argument("--onnx", default="models/esmoe_n_visdrone_sim.onnx")
ap.add_argument("--n", type=int, default=100)
a = ap.parse_args()
imgs = sorted(glob.glob("/data/datasets/VisDrone/images/val/*.jpg"))[:a.n]

interp = MNN.Interpreter(a.mnn); sess = interp.createSession({"numThread": 4, "backend": "CPU"})
inp = interp.getSessionInput(sess)
def mnn_run(x):
    t = MNN.Tensor((1, 3, 640, 640), MNN.Halide_Type_Float, x, MNN.Tensor_DimensionType_Caffe)
    inp.copyFrom(t); interp.runSession(sess)
    o = interp.getSessionOutput(sess); sh = o.getShape()
    ot = MNN.Tensor(sh, MNN.Halide_Type_Float, np.zeros(sh, np.float32), MNN.Tensor_DimensionType_Caffe)
    o.copyToHostTensor(ot)
    return np.array(ot.getData(), np.float32).reshape(sh)

so = ort.SessionOptions(); so.intra_op_num_threads = 4   # match MNN's 4 threads for a fair compare
s = ort.InferenceSession(a.onnx, sess_options=so, providers=["CPUExecutionProvider"])
nm = s.get_inputs()[0].name

maxd = 0.0; meand = 0.0
for p in imgs:
    x = letterbox(p)
    ym = mnn_run(x); yo = s.run(None, {nm: x})[0]
    d = np.abs(ym - yo); maxd = max(maxd, d.max()); meand += d.mean()
meand /= len(imgs)

tm, to = [], []
for p in imgs[:min(50, len(imgs))]:
    x = letterbox(p)
    t0 = time.time(); mnn_run(x);          tm.append((time.time() - t0) * 1000)
    t0 = time.time(); s.run(None, {nm: x}); to.append((time.time() - t0) * 1000)
print(f"MNN vs ONNX over {len(imgs)} imgs:  max|delta|={maxd:.6f}  mean|delta|={meand:.2e}")
print(f"same-box CPU (4 threads):")
print(f"  MNN : {np.mean(tm):5.1f} ms/img  ->  {1000/np.mean(tm):5.1f} FPS")
print(f"  ONNX: {np.mean(to):5.1f} ms/img  ->  {1000/np.mean(to):5.1f} FPS")
