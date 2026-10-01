#!/usr/bin/env python3
# MNN correctness + latency: feed identical letterboxed inputs to the MNN model and
# the source ONNX, compare raw output tensors (same graph -> same mAP, already <0.5%).
import argparse, glob, os, time
import MNN, numpy as np, cv2, onnxruntime as ort

def letterbox(path, sz=640):
    img = cv2.imread(path)
    if img is None:                                   # missing / corrupt file
        return None
    h, w = img.shape[:2]
    r = min(sz / h, sz / w); nw, nh = round(w * r), round(h * r)
    c = np.full((sz, sz, 3), 114, np.uint8); px, py = (sz - nw) // 2, (sz - nh) // 2
    c[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh))
    return np.ascontiguousarray(np.transpose(c[:, :, ::-1].astype(np.float32) / 255, (2, 0, 1))[None])

ap = argparse.ArgumentParser(description="MNN vs ONNX raw-output parity + CPU latency on identical letterboxed inputs.")
ap.add_argument("--mnn", default="models/esmoe_n_visdrone.mnn")
ap.add_argument("--onnx", default="models/esmoe_n_visdrone_sim.onnx")
ap.add_argument("--images", default="/data/datasets/VisDrone/images/val", help="dir of validation images (*.jpg)")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--n", type=int, default=100)
a = ap.parse_args()
imgs = sorted(glob.glob(os.path.join(a.images, "*.jpg")))[:a.n]
if not imgs:
    raise SystemExit(f"no *.jpg images found under {a.images}")

interp = MNN.Interpreter(a.mnn); sess = interp.createSession({"numThread": 4, "backend": "CPU"})
inp = interp.getSessionInput(sess)
def mnn_run(x):
    # multi-output aware (seg protos, mixture heads): returns the rank-3 detection tensor,
    # falling back to the sole output for single-output detection models.
    t = MNN.Tensor((1, 3, a.imgsz, a.imgsz), MNN.Halide_Type_Float, x, MNN.Tensor_DimensionType_Caffe)
    inp.copyFrom(t); interp.runSession(sess)
    outs = interp.getSessionOutputAll(sess)
    o = None
    for name, cand in outs.items():
        if len(cand.getShape()) == 3: o = cand; break
    if o is None: o = interp.getSessionOutput(sess)
    sh = o.getShape()
    ot = MNN.Tensor(sh, MNN.Halide_Type_Float, np.zeros(sh, np.float32), MNN.Tensor_DimensionType_Caffe)
    o.copyToHostTensor(ot)
    return np.array(ot.getData(), np.float32).reshape(sh)

so = ort.SessionOptions(); so.intra_op_num_threads = 4   # match MNN's 4 threads for a fair compare
s = ort.InferenceSession(a.onnx, sess_options=so, providers=["CPUExecutionProvider"])
nm = s.get_inputs()[0].name

maxd = 0.0; meand = 0.0; n_ok = 0
for p in imgs:
    x = letterbox(p, a.imgsz)
    if x is None:
        print(f"  skip unreadable image: {p}")
        continue
    ym = mnn_run(x)
    det_out = next((o.name for o in s.get_outputs() if len(o.shape) == 3), s.get_outputs()[0].name)
    yo = s.run([det_out], {nm: x})[0]
    d = np.abs(ym - yo); maxd = max(maxd, d.max()); meand += d.mean(); n_ok += 1
if not n_ok:
    raise SystemExit("no readable images to compare")
meand /= n_ok

tm, to = [], []
for p in imgs[:min(50, len(imgs))]:
    x = letterbox(p, a.imgsz)
    if x is None:
        continue
    t0 = time.perf_counter(); mnn_run(x);          tm.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter(); s.run(None, {nm: x}); to.append((time.perf_counter() - t0) * 1000)
print(f"MNN vs ONNX over {n_ok} imgs:  max|delta|={maxd:.6f}  mean|delta|={meand:.2e}")
print(f"same-box CPU (4 threads):")
print(f"  MNN : {np.mean(tm):5.1f} ms/img  ->  {1000/np.mean(tm):5.1f} FPS")
print(f"  ONNX: {np.mean(to):5.1f} ms/img  ->  {1000/np.mean(to):5.1f} FPS")
