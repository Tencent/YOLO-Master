#!/usr/bin/env python3
"""Render one real image into the raw float32 CHW blob cpp/tools/ncnn_bench feeds to `in0`.

Reproduces the edge runtime preprocessing exactly (the numpy replica in
tests/validate_mixture.py::preprocess): aspect-preserving resize, centred gray-114 letterbox,
BGR->RGB, /255, CHW float32. Writes the blob plus a JSON sidecar (source, imgsz, letterbox
ratio/pad, md5 of the blob) so a device log can be tied back to the exact probe.

    /root/anaconda3/envs/yolo_master/bin/python scripts/make_probe_f32.py \
        --image /data/datasets/coco/images/val2017/000000000139.jpg --imgsz 640 \
        --out probe_640.f32 [--meta probe_640.json]

numpy + cv2 only (no ultralytics, no ncnn wheel).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def letterbox_params(w, h, imgsz):
    r = min(imgsz / w, imgsz / h)
    nw, nh = round(w * r), round(h * r)
    return r, (imgsz - nw) // 2, (imgsz - nh) // 2


def preprocess(img_path, imgsz):
    """runtime letterbox -> (blob [1,3,imgsz,imgsz] float32 RGB /255, (r, px, py, w, h))."""
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise SystemExit(f"cannot read image: {img_path}")
    h, w = bgr.shape[:2]
    r, px, py = letterbox_params(w, h, imgsz)
    resized = cv2.resize(bgr, (round(w * r), round(h * r)))
    canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
    canvas[py:py + resized.shape[0], px:px + resized.shape[1]] = resized
    blob = canvas[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(blob), (r, px, py, w, h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="source jpg/png")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", required=True, help="output raw float32 CHW file (e.g. probe_640.f32)")
    ap.add_argument("--meta", default=None, help="JSON sidecar (default: <out>.json)")
    args = ap.parse_args()

    src = Path(args.image)
    out = Path(args.out)
    meta = Path(args.meta) if args.meta else out.with_suffix(out.suffix + ".json")

    blob, (r, px, py, w, h) = preprocess(src, args.imgsz)
    chw = blob[0]                                   # [3, imgsz, imgsz], RGB, /255
    assert chw.dtype == np.float32 and chw.flags.c_contiguous
    raw = chw.tobytes()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)

    info = {
        "source": str(src.resolve()),
        "source_md5": hashlib.md5(src.read_bytes()).hexdigest(),
        "source_hw": [h, w],
        "imgsz": args.imgsz,
        "shape": list(chw.shape),                   # C,H,W -> ncnn_bench --shape 3,640,640
        "dtype": "float32",
        "layout": "CHW",
        "channel_order": "RGB",
        "normalize": "x/255",
        "letterbox": {"ratio": r, "pad_x": px, "pad_y": py, "fill": 114, "resized_hw": [round(h * r), round(w * r)]},
        "f32_bytes": len(raw),
        "f32_md5": hashlib.md5(raw).hexdigest(),
        "blob_min": float(chw.min()),
        "blob_max": float(chw.max()),
        "blob_mean": float(chw.mean()),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": "scripts/make_probe_f32.py",
    }
    meta.write_text(json.dumps(info, indent=2) + "\n")
    print(f"[make_probe_f32] {src} ({w}x{h}) -> {out} shape={','.join(map(str, chw.shape))} "
          f"r={r:.6f} pad=({px},{py}) md5={info['f32_md5']} meta={meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
