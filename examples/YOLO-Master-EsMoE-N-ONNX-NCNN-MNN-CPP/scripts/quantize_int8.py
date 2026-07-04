#!/usr/bin/env python3
"""Static INT8 post-training quantization of the EsMoE-N ONNX via ONNXRuntime.

Calibrates on VisDrone TRAIN images (no val leakage), QDQ per-channel. The
preprocessing (letterbox 640, BGR->RGB, /255, NCHW) matches the C++ runner and
ultralytics. Metadata (class names) is restored so `YOLO(int8).val()` works.
"""
import argparse, glob, os
from pathlib import Path
import cv2, numpy as np, onnx
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod)
from onnxruntime.quantization.shape_inference import quant_pre_process

FP32 = "/data/yolo-master-edge/models/esmoe_n_visdrone_sim.onnx"
TRAIN = "/data/datasets/VisDrone/images/train"


def letterbox(img_path, imgsz=640):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nw, nh = round(w * r), round(h * r)
    canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
    px, py = (imgsz - nw) // 2, (imgsz - nh) // 2
    canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh))
    x = canvas[:, :, ::-1].astype(np.float32) / 255.0            # BGR->RGB, /255
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None])  # (1,3,640,640)


class Calib(CalibrationDataReader):
    def __init__(self, images, input_name):
        self.input_name, self.images = input_name, images
        self.it = iter(images)
    def get_next(self):
        p = next(self.it, None)
        return None if p is None else {self.input_name: letterbox(p)}
    def rewind(self):
        self.it = iter(self.images)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/yolo-master-edge/models/esmoe_n_visdrone_int8.onnx")
    ap.add_argument("--n-calib", type=int, default=500)
    ap.add_argument("--method", default="MinMax", choices=["MinMax", "Entropy", "Percentile"])
    ap.add_argument("--format", default="QDQ", choices=["QDQ", "QOperator"],
                    help="QOperator fuses to QLinear* (fast on ORT CPU); QDQ is TensorRT-preferred")
    ap.add_argument("--exclude", nargs="*", default=[], help="node-name substrings to keep in fp32")
    args = ap.parse_args()

    # evenly-spaced calibration sample across the train set
    allimg = sorted(glob.glob(os.path.join(TRAIN, "*.jpg")))
    step = max(1, len(allimg) // args.n_calib)
    calib = allimg[::step][:args.n_calib]
    print(f"[quant] calibration images: {len(calib)} (method={args.method}, per_channel=True, QDQ)")

    inp = onnx.load(FP32).graph.input[0].name
    prep = FP32.replace(".onnx", ".prep.onnx")
    quant_pre_process(FP32, prep, skip_symbolic_shape=True)  # model is fully static-shaped

    # per-channel QDQ emits DequantizeLinear with an 'axis' attr -> needs opset >= 13
    from onnx import version_converter
    mp = onnx.load(prep)
    cur = next((o.version for o in mp.opset_import if o.domain in ("", "ai.onnx")), 0)
    if cur < 13:
        print(f"[quant] upgrading opset {cur} -> 17 (needed for per-channel QDQ)")
        onnx.save(version_converter.convert_version(mp, 17), prep)

    # optional: exclude quant-sensitive nodes (router/attention) -> keep fp32
    nodes_to_exclude = []
    if args.exclude:
        m = onnx.load(prep)
        for n in m.graph.node:
            if any(s in n.name for s in args.exclude):
                nodes_to_exclude.append(n.name)
        print(f"[quant] excluding {len(nodes_to_exclude)} nodes from INT8 (fp32): {args.exclude}")

    quantize_static(
        prep, args.out,
        calibration_data_reader=Calib(calib, inp),
        quant_format=getattr(QuantFormat, args.format),
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=getattr(CalibrationMethod, args.method),
        nodes_to_exclude=nodes_to_exclude,
    )

    # restore ultralytics metadata (names/imgsz) so YOLO(int8).val() works
    src = {p.key: p.value for p in onnx.load(FP32).metadata_props}
    q = onnx.load(args.out)
    del q.metadata_props[:]
    for k, v in src.items():
        e = q.metadata_props.add(); e.key, e.value = k, v
    onnx.save(q, args.out)

    fp32_mb = os.path.getsize(FP32) / 1e6
    int8_mb = os.path.getsize(args.out) / 1e6
    print(f"[quant] done -> {args.out}")
    print(f"[quant] size: fp32 {fp32_mb:.1f} MB -> int8 {int8_mb:.1f} MB ({fp32_mb/int8_mb:.1f}x smaller)")


if __name__ == "__main__":
    main()
