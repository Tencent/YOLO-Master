#!/usr/bin/env python
"""Mac-side A8 step for the project03 iPhone INT8 path.

Takes the fp16 .mlpackage produced on Linux by scripts/export_coreml_p03.py and
produces the W8A8 (INT8 weights + INT8 activations) package. Activation
calibration runs the model, which needs the Core ML runtime - macOS only.

Also runs a parity check of every package against the shipped torch reference
(p03_ref.npz) and a quick latency loop per compute unit.

Usage (Mac, coremltools >= 8):
  python3 mac/quantize_a8_p03.py --dir models/p03_coreml --calib <folder-of-jpgs>
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def letterbox(img, sz=640):
    import cv2
    h, w = img.shape[:2]
    r = min(sz / h, sz / w)
    nh, nw = round(h * r), round(w * r)
    im = cv2.resize(img, (nw, nh))
    top, left = (sz - nh) // 2, (sz - nw) // 2
    canvas = np.full((sz, sz, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = im
    blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="models/p03_coreml")
    ap.add_argument("--calib", required=True, help="folder of jpg/png for A8 calibration")
    ap.add_argument("--n-calib", type=int, default=64)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    d = Path(args.dir)

    import cv2  # noqa: F401  (letterbox)
    import coremltools as ct
    from coremltools.optimize.coreml import OptimizationConfig
    # the A8 API moved between ct releases: stable in some, experimental in others
    try:
        from coremltools.optimize.coreml import (
            OpActivationLinearQuantizerConfig, linear_quantize_activations)
    except ImportError:
        from coremltools.optimize.coreml.experimental import (
            OpActivationLinearQuantizerConfig, linear_quantize_activations)

    fp16 = ct.models.MLModel(str(d / "p03_v01n_coco_fp16.mlpackage"))

    imgs = sorted(Path(args.calib).glob("*.[jp][pn]g"))[: args.n_calib]
    assert imgs, f"no images in {args.calib}"
    sample = [{"images": letterbox(cv2.imread(str(p)))} for p in imgs]
    print(f"[a8] calibrating on {len(sample)} images...")
    a8_cfg = OptimizationConfig(global_config=OpActivationLinearQuantizerConfig(
        mode="linear_symmetric"))
    w8a8 = linear_quantize_activations(fp16, a8_cfg, sample_data=sample)
    # weights too (A8 pass quantizes activations only)
    from coremltools.optimize.coreml import OpLinearQuantizerConfig, linear_quantize_weights
    w8a8 = linear_quantize_weights(w8a8, config=OptimizationConfig(
        global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                              granularity="per_channel")))
    out = d / "p03_v01n_coco_w8a8.mlpackage"
    w8a8.save(str(out))
    print(f"[a8] {out}")

    # parity + latency for every package present
    ref = np.load(d / "p03_ref.npz")
    for pkg in sorted(d.glob("*.mlpackage")):
        for cu, cuname in ((ct.ComputeUnit.CPU_AND_NE, "ANE"),
                           (ct.ComputeUnit.CPU_AND_GPU, "GPU"),
                           (ct.ComputeUnit.CPU_ONLY, "CPU")):
            try:
                m = ct.models.MLModel(str(pkg), compute_units=cu)
                pred = m.predict({"images": ref["input"]})
                y = list(pred.values())[0]
                err = float(np.abs(np.asarray(y) - ref["output"]).max())
                t0 = time.perf_counter()
                for _ in range(args.iters):
                    m.predict({"images": ref["input"]})
                ms = (time.perf_counter() - t0) * 1000 / args.iters
                print(f"[bench {pkg.stem} @{cuname}] {ms:.2f}ms  max|err| {err:.4f}")
            except Exception as e:
                print(f"[bench {pkg.stem} @{cuname}] failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
