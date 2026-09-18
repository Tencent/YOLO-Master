#!/usr/bin/env python
"""A3 P0: PyTorch vs ONNX Runtime accuracy delta under ONE protocol (ultralytics .val,
imgsz 640, batch 1, val2017). The TensorRT rungs come from quantize_trt.py's ladder.csv
with the same .val protocol, so all deltas in the README are apples to apples.

  python scripts/a3/backend_val.py --model <pt> --onnx <onnx> --data a3/config/coco-a3.yaml \
      --json a3/results/backend_<m>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "False")
sys.path.insert(0, str(Path(__file__).parent))
from diagnose_moe import _fix_add_residual, _force_dense_esmoe  # noqa: E402


def val(m, data, imgsz, workers):
    t = time.time()
    r = m.val(data=data, split="val", imgsz=imgsz, batch=1, device=0, workers=workers,
              plots=False, verbose=False)
    return {"mAP50": round(float(r.box.map50), 5), "mAP50_95": round(float(r.box.map), 5),
            "val_s": round(time.time() - t, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--data", default="a3/config/coco-a3.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    import onnxruntime as ort
    from ultralytics import YOLO

    out = {"model": args.model, "onnx": args.onnx, "protocol":
           f"ultralytics .val imgsz={args.imgsz} batch=1 split=val ({args.data})"}
    yolo = YOLO(args.model)
    out["repairs"] = {"add_residual_forced_false": _fix_add_residual(yolo),
                      "esmoe_dense_forced": _force_dense_esmoe(yolo)}
    print(f"[repair] {out['repairs']}")
    out["pytorch"] = val(yolo, args.data, args.imgsz, args.workers)
    print(f"[pytorch] {out['pytorch']}")

    # "available" only means the provider library exists; it can still fail to dlopen
    # (missing cuDNN on LD_LIBRARY_PATH) and silently fall back to CPU, after which
    # ultralytics' GPU io-binding crashes. Prove the EP really loads on this ONNX.
    probe = ort.InferenceSession(args.onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    loaded = probe.get_providers()
    out["onnxruntime"] = {"version": ort.__version__, "providers_available": ort.get_available_providers(),
                          "providers_loaded": loaded}
    assert loaded[0] == "CUDAExecutionProvider", \
        f"ORT CUDA EP did not load (got {loaded}); set LD_LIBRARY_PATH to torch's nvidia/cudnn/lib"
    del probe
    out["ort"] = val(YOLO(args.onnx, task="detect"), args.data, args.imgsz, args.workers)
    out["ort"]["delta_mAP50_95"] = round(out["ort"]["mAP50_95"] - out["pytorch"]["mAP50_95"], 5)
    print(f"[ort] {out['ort']}")
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"[backend_val] -> {args.json}")


if __name__ == "__main__":
    main()
