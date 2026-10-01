#!/usr/bin/env python3
"""Compare model-input tensors dumped by a runtime (`--dump-input`, raw float32 NCHW, one .f32 per
image) against the Linux reference preprocessing (cpp/src/common.cpp preprocess_nchw: cv::resize
INTER_LINEAR letterbox, integer pads (imgsz - out) / 2, 114 gray, BGR -> RGB, / 255).

    python scripts/preproc_compare.py <images_dir> <dumps_dir> [--imgsz 640] [--tol 0.0039216]

Prints the max abs difference per tensor and fails when any exceeds --tol (1/255 by default, the
contract the CUDA kernel is held to by cpp/tests/preproc_parity.cpp).
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np


def load_source(stem, images_dir, dumps_dir):
    """The BGR source: the runtime's own decoded pixels when it dumped them (<stem>.WxH.rgba|bgra next
    to the tensor, so the JPEG decoder is out of the comparison), else the file decoded by cv2."""
    for f in glob.glob(os.path.join(dumps_dir, stem + ".*x*.*")):
        dims, order = f.rsplit(".", 2)[1:]
        if order not in ("rgba", "bgra"):
            continue
        w, h = (int(v) for v in dims.split("x"))
        px = np.fromfile(f, np.uint8).reshape(h, w, 4)
        return (px[:, :, 2::-1] if order == "rgba" else px[:, :, :3]).copy(), "runtime pixels"
    imgs = [p for p in glob.glob(os.path.join(images_dir, stem + ".*")) if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    if not imgs:
        return None, "no source"
    return cv2.imread(imgs[0], cv2.IMREAD_COLOR), "cv2 decode"


def reference(img, imgsz):
    h, w = img.shape[:2]
    r = min(imgsz / w, imgsz / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((imgsz, imgsz, 3), 114, np.uint8)
    px, py = (imgsz - nw) // 2, (imgsz - nh) // 2
    out[py:py + nh, px:px + nw] = resized
    rgb = out[:, :, ::-1].astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("images")
    ap.add_argument("dumps")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tol", type=float, default=1.0 / 255.0)
    a = ap.parse_args()
    dumps = sorted(glob.glob(os.path.join(a.dumps, "*.f32")))
    if not dumps:
        print("no .f32 dumps in", a.dumps, file=sys.stderr)
        return 2
    worst = 0.0
    bad = 0
    source_kind = ""
    for d in dumps:
        stem = os.path.splitext(os.path.basename(d))[0]
        img, source_kind = load_source(stem, a.images, a.dumps)
        if img is None:
            print(f"  {stem}: no source image", file=sys.stderr)
            bad += 1
            continue
        got = np.fromfile(d, np.float32)
        ref = reference(img, a.imgsz)
        if got.size != ref.size:
            print(f"  {stem}: size {got.size} != {ref.size}", file=sys.stderr)
            bad += 1
            continue
        diff = np.abs(got.reshape(ref.shape) - ref)
        m = float(diff.max())
        pad_diff = float(np.abs(got.reshape(ref.shape)[ref == 114 / 255.0] - 114 / 255.0).max()) if np.any(ref == 114 / 255.0) else 0.0
        worst = max(worst, m)
        flag = "" if m <= a.tol else "  EXCEEDS"
        print(f"  {stem}: max|diff|={m:.5f} mean={float(diff.mean()):.6f} pad={pad_diff:.5f}{flag}")
        if m > a.tol:
            bad += 1
    print(f"{len(dumps)} tensors ({source_kind}), worst {worst:.5f} (tol {a.tol:.5f}): {'OK' if bad == 0 else str(bad) + ' failed'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
