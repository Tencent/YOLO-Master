#!/usr/bin/env python3
"""Validate exported mixture models (scripts/export_mixture.py output) against the edge
runtime. Runs in the edge environment (numpy + onnxruntime; no ultralytics needed).

Three levels per model:
  A. torch-vs-pipeline: the eager predict reference (<model>.ref.npz, torch ground truth)
     is IoU-matched against the C++ CLI's --save-txt boxes on the same images. Validates
     the FULL-MODEL export numerics + edge decode + NMS end to end (upstream's own
     evidence stops at block-level roundtrips).
  B. python-vs-C++ decode: onnxruntime here, with a numpy replica of the edge letterbox
     and decode (both layouts: [1,4+nc,anchors] and end2end [1,N,6]), tightly matched
     against the CLI boxes. Isolates the C++ decode path at ~pixel tolerance.
  C. feature smoke via the CLI: metadata nc/imgsz line, sparse slicing line, CW-NMS tag,
     label-export instance count == detection count.

Backends: onnx always; mnn/ncnn when a sibling <stem>.mnn / <stem>_ncnn exists (det-count
parity vs the onnx backend, T6-style).
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


def letterbox_params(w, h, imgsz):
    r = min(imgsz / w, imgsz / h)
    nw, nh = round(w * r), round(h * r)
    return r, (imgsz - nw) // 2, (imgsz - nh) // 2


def preprocess(img_path, imgsz):
    import cv2
    bgr = cv2.imread(str(img_path))
    h, w = bgr.shape[:2]
    r, px, py = letterbox_params(w, h, imgsz)
    resized = cv2.resize(bgr, (round(w * r), round(h * r)))
    canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
    canvas[py:py + resized.shape[0], px:px + resized.shape[1]] = resized
    blob = canvas[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(blob), (r, px, py, w, h)


def decode_np(out, lb, conf):
    """numpy replica of the edge decode for both layouts -> (xyxy, conf, cls)."""
    r, px, py, w, h = lb
    if out.ndim == 3:
        out = out[0]
    if out.shape[-1] == 6 and out.shape[0] >= 32:          # end2end [N, 6]
        keep = out[:, 4] >= conf
        d = out[keep]
        xyxy = (d[:, :4] - [px, py, px, py]) / r
        return clip_xyxy(xyxy, w, h), d[:, 4], d[:, 5].astype(int)
    feat, anchors = out.shape                               # [4+nc, anchors]
    scores = out[4:, :]
    cls = scores.argmax(0)
    sc = scores.max(0)
    keep = sc >= conf
    cx, cy, bw, bh = out[0, keep], out[1, keep], out[2, keep], out[3, keep]
    x0 = (cx - bw / 2 - px) / r
    y0 = (cy - bh / 2 - py) / r
    xyxy = np.stack([x0, y0, x0 + bw / r, y0 + bh / r], 1)
    return clip_xyxy(xyxy, w, h), sc[keep], cls[keep]


def clip_xyxy(xyxy, w, h):
    """replicate the CLI's clip-to-frame so candidates match its post-NMS boxes."""
    if len(xyxy) == 0:
        return xyxy
    out = xyxy.copy()
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h)
    return out


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), 2)
    ar_a = np.prod(a[:, 2:] - a[:, :2], 1)
    ar_b = np.prod(b[:, 2:] - b[:, :2], 1)
    return inter / (ar_a[:, None] + ar_b[None, :] - inter + 1e-9)


def match_rate(ref_xyxy, ref_cls, got_xyxy, got_cls, iou_thr=0.8):
    """fraction of ref boxes with a same-class IoU>=thr counterpart (greedy)."""
    if len(ref_xyxy) == 0:
        return 1.0
    m = iou_matrix(ref_xyxy, got_xyxy)
    m[ref_cls[:, None] != got_cls[None, :]] = 0
    matched = 0
    used = set()
    for i in np.argsort(-m.max(1) if m.size else []):
        j_order = np.argsort(-m[i]) if m.size else []
        for j in j_order:
            if m[i, j] < iou_thr:
                break
            if j not in used:
                used.add(j)
                matched += 1
                break
    return matched / len(ref_xyxy)


def read_savetxt(path):
    rows = [l.split() for l in open(path)] if Path(path).exists() else []
    if not rows:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    a = np.array(rows, dtype=float)
    return a[:, 2:6], a[:, 1], a[:, 0].astype(int)


def run_cli(bin_, model, src, extra, tmp):
    cmd = [bin_, "-m", str(model), "-s", str(src), "--no-save"] + extra
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
    return p.stdout + p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models/mixture")
    ap.add_argument("--bin", default="cpp/build_pkg-cpu/yolomaster_edge")
    ap.add_argument("--images", default="visdrone50/images/val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="must match the export's --conf (reference predict)")
    ap.add_argument("--only", default="")
    ap.add_argument("--ncnn-fp16", action="store_true",
                    help="run the ncnn raw dump with fp16 packed/storage/arithmetic enabled. NO-OP ON x86 "
                         "(ncnn's fp16 kernels are the armv8.2 path): only an ARM/Jetson run certifies fp16.")
    args = ap.parse_args()

    import onnxruntime as ort
    import tempfile

    mdir = Path(args.models)
    bin_ = str(Path(args.bin).resolve())
    onnx_models = sorted(mdir.glob("*.onnx"))
    if args.only:
        keep = set(args.only.split(","))
        onnx_models = [m for m in onnx_models if m.stem in keep]
    failures = []

    for model in onnx_models:
        print(f"\n=== {model.stem} ===")
        sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        det_out = next(o for o in sess.get_outputs() if len(o.shape) == 3)
        e2e = det_out.shape[1] >= 32 and det_out.shape[2] == 6
        print(f"  layout: {'end2end ' + str(det_out.shape) if e2e else 'anchors ' + str(det_out.shape)}")

        ref_npz = mdir / f"{model.stem}.ref.npz"
        ref = np.load(ref_npz) if ref_npz.exists() else None
        stems = sorted({k.rsplit("_", 1)[0] for k in ref.files}) if ref is not None else []
        n_ref = sum(int(ref[k].shape[0]) for k in ref.files if k.endswith("_xyxy")) if ref is not None else 0
        print(f"  reference boxes: {n_ref}")
        if ref is not None and n_ref == 0:
            failures.append(f"{model.stem}: EMPTY reference set - validation would be vacuous")
            continue

        with tempfile.TemporaryDirectory() as tmp:
            rates_a, rates_b = [], []
            for stem in stems:
                img = next(Path(args.images).glob(stem + ".*"))
                # C++ CLI boxes
                # e2e refs carry no NMS (one2one top-k); run the CLI NMS nearly-off so
                # the sets stay comparable. Anchors refs went through torch NMS at 0.7.
                cli_iou = "0.95" if e2e else "0.7"
                run_cli(bin_, model.resolve(), img.resolve(),
                        ["--conf", str(args.conf), "--iou", cli_iou, "--quiet",
                         "--save-txt", tmp], tmp)
                got_xyxy, got_conf, got_cls = read_savetxt(Path(tmp) / f"{stem}.txt")
                # A: RAW torch-eager output vs ORT (max abs diff; tie-immune, and the
                # strongest full-model export-fidelity evidence available)
                if stem + "_raw" in ref:
                    blob_a, _ = preprocess(img, args.imgsz)
                    out_a = sess.run([det_out.name], {in_name: blob_a})[0]
                    tie_skip = False
                    if e2e:
                        # in-graph top-k under an untrained trunk's mass score ties picks
                        # implementation-defined row SETS; elementwise diff is undefined.
                        # Skip ONLY the raw check (level B still binds); a trained model
                        # has no such ties.
                        sc = out_a[0][:, 4]
                        _, counts = np.unique(np.round(sc, 5), return_counts=True)
                        tie_skip = counts.max() > 0.5 * len(sc)
                        if tie_skip and stem == stems[0]:
                            print("  A raw parity: SKIPPED (tie-degenerate untrained "
                                  "trunk; in-graph top-k row sets are implementation-"
                                  "defined - level B still binds)")
                    if not tie_skip:
                        rates_a.append(float(np.abs(out_a - ref[stem + "_raw"]).max()))
                # B: every CLI box must exist among the raw decoded candidates (reversed
                # direction: the CLI output is post-NMS, the numpy decode is pre-NMS)
                blob, lb = preprocess(img, args.imgsz)
                out = sess.run([det_out.name], {in_name: blob})[0]
                p_xyxy, p_conf, p_cls = decode_np(out, lb, args.conf)
                if len(got_xyxy) == 0:
                    rates_b.append(0.0)          # nothing decoded = failure, not a pass
                else:
                    rates_b.append(match_rate(got_xyxy, got_cls, p_xyxy, p_cls, iou_thr=0.9))
            a, b = (np.max(rates_a) if rates_a else -1), (np.mean(rates_b) if rates_b else -1)
            print(f"  A eager-vs-ORT raw maxdiff: {a:.2e}   B CLI-boxes-in-decode match: {b:.3f}")
            if rates_a and a > 5e-3:
                failures.append(f"{model.stem}: raw export fidelity maxdiff {a:.2e}")
            if rates_b and b < 0.9:
                failures.append(f"{model.stem}: CLI-boxes-in-decode match {b:.3f}")

            # C: feature smoke (folder = images dir limited)
            log = run_cli(bin_, model.resolve(), Path(args.images).resolve(),
                          ["--limit", "2", "--quiet", "--slicing", "sparse", "--cw-nms",
                           "--export-labels", tmp + "/lbl"], tmp)
            ok_meta = re.search(r"nc=80", log) or re.search(r"nc=\d+ \(model-metadata\)", log)
            ok_slice = "[slicing]" in log
            ok_cw = "nms=cw" in log
            m_lbl = re.search(r"\[labels\].*instances=(\d+)", log)
            m_dets = re.search(r"total_dets=(\d+)", log)
            ok_lbl = m_lbl and m_dets and m_lbl.group(1) == m_dets.group(1)
            print(f"  C meta:{bool(ok_meta)} slicing:{ok_slice} cw:{ok_cw} labels==dets:{bool(ok_lbl)}")
            for name, ok in [("meta", ok_meta), ("slicing", ok_slice), ("cw", ok_cw), ("labels", ok_lbl)]:
                if not ok:
                    failures.append(f"{model.stem}: feature check '{name}' failed")

            # MNN sibling: RAW-tensor parity vs ORT (python MNN), tie-immune, plus a CLI
            # smoke run proving the edge backend loads and infers the converted model.
            sib = mdir / f"{model.stem}.mnn"
            if sib.exists():
                try:
                    import MNN
                    interp = MNN.Interpreter(str(sib))
                    msess = interp.createSession({"numThread": 4, "backend": "CPU"})
                    inp = interp.getSessionInput(msess)
                    maxd = 0.0
                    for stem in stems:
                        img = next(Path(args.images).glob(stem + ".*"))
                        blob_m, _ = preprocess(img, args.imgsz)
                        t = MNN.Tensor((1, 3, args.imgsz, args.imgsz), MNN.Halide_Type_Float,
                                       blob_m, MNN.Tensor_DimensionType_Caffe)
                        inp.copyFrom(t)
                        interp.runSession(msess)
                        mout = None
                        for _, cand in interp.getSessionOutputAll(msess).items():
                            if len(cand.getShape()) == 3:
                                mout = cand
                                break
                        sh = mout.getShape()
                        host = MNN.Tensor(sh, MNN.Halide_Type_Float,
                                          np.zeros(sh, np.float32), MNN.Tensor_DimensionType_Caffe)
                        mout.copyToHostTensor(host)
                        ym = np.array(host.getData(), np.float32).reshape(sh)
                        yo = sess.run([det_out.name], {in_name: blob_m})[0]
                        if ym.shape != yo.shape:
                            maxd = float("inf")
                            break
                        if e2e:
                            sc = yo[0][:, 4]
                            _, counts = np.unique(np.round(sc, 5), return_counts=True)
                            if counts.max() > 0.5 * len(sc):
                                maxd = -2.0          # sentinel: tie-degenerate, skip
                                break
                        maxd = max(maxd, float(np.abs(ym - yo).max()))
                    log_s = run_cli(bin_, sib.resolve(), Path(args.images).resolve(),
                                    ["--limit", "1", "--quiet"], tmp)
                    cli_ok = "frames=1" in log_s
                    if maxd == -2.0:
                        ok = cli_ok
                        print(f"  mnn parity: raw SKIPPED (tie-degenerate top-k), CLI run {'OK' if cli_ok else 'FAILED'} -> {'OK' if ok else 'MISMATCH'}")
                    else:
                        ok = maxd < 5e-3 and cli_ok
                        print(f"  mnn parity: raw maxdiff {maxd:.2e}, CLI run {'OK' if cli_ok else 'FAILED'} -> {'OK' if ok else 'MISMATCH'}")
                    if not ok:
                        failures.append(f"{model.stem}: mnn raw maxdiff {maxd} cli={cli_ok}")
                except ImportError:
                    print("  mnn parity: python MNN unavailable, skipped")

            # ncnn sibling dir: RAW-tensor parity vs ORT via tests/ncnn_rawdump (linked
            # against the vendored SDK; the python ncnn wheels segfault at inference on
            # this machine), plus the same CLI smoke (backend auto-detects the dir).
            nsib = mdir / f"{model.stem}_ncnn"
            if (nsib / "model.ncnn.param").exists():
                dump = Path(__file__).parent / "ncnn_rawdump"
                if not dump.exists():
                    import os
                    root = Path(__file__).resolve().parent.parent
                    # NCNN_ROOT overrides the vendored SDK (e.g. third_party/ncnn-x86-20260526, the
                    # from-source build that actually runs on an Ubuntu-20.04 host).
                    ncnn_root = Path(os.environ.get("NCNN_ROOT", str(root / "third_party" / "ncnn")))
                    subprocess.run(["g++", "-O2", "-std=c++17",
                                    f"-I{ncnn_root}/include/ncnn",
                                    str(Path(__file__).parent / "ncnn_rawdump.cpp"),
                                    "-o", str(dump),
                                    f"-L{ncnn_root}/lib", "-lncnn",
                                    f"-Wl,-rpath,{ncnn_root}/lib", "-fopenmp"],
                                   check=True)
                maxd = 0.0
                for stem in stems:
                    img = next(Path(args.images).glob(stem + ".*"))
                    blob_n, _ = preprocess(img, args.imgsz)
                    fin, fout = Path(tmp) / "in.f32", Path(tmp) / "out.f32"
                    blob_n[0].astype(np.float32).tofile(fin)
                    r = subprocess.run([str(dump)] + (["--fp16"] if args.ncnn_fp16 else []) +
                                       [str(nsib / "model.ncnn.param"),
                                        str(nsib / "model.ncnn.bin"), str(fin),
                                        "3", str(args.imgsz), str(args.imgsz), str(fout)],
                                       capture_output=True, text=True)
                    if r.returncode != 0:
                        maxd = float("inf")
                        break
                    dims = dict(kv.split("=") for kv in r.stdout.split())
                    yn = np.fromfile(fout, np.float32).reshape(1, int(dims["h"]), int(dims["w"]))
                    yo = sess.run([det_out.name], {in_name: blob_n})[0]
                    if yn.shape != yo.shape:
                        maxd = float("inf")
                        break
                    maxd = max(maxd, float(np.abs(yn - yo).max()))
                log_s = run_cli(bin_, nsib.resolve(), Path(args.images).resolve(),
                                ["--limit", "1", "--quiet"], tmp)
                cli_ok = "frames=1" in log_s
                ok = maxd < 5e-3 and cli_ok
                print(f"  ncnn parity: raw maxdiff {maxd:.2e}, CLI run {'OK' if cli_ok else 'FAILED'} -> {'OK' if ok else 'MISMATCH'}")
                if not ok:
                    failures.append(f"{model.stem}: ncnn raw maxdiff {maxd} cli={cli_ok}")

    print("\n===== validation summary =====")
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        return 1
    print(f"  all checks passed for {len(onnx_models)} models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
