#!/usr/bin/env python
"""Model-only vs end-to-end latency for TRT engines, plus mAP re-validation.

MODEL-ONLY: median of pure enqueue->sync executions (no host I/O), the number
quantization ladders report. END-TO-END: per-frame wall time of the full
deployment pipeline on real val images: letterbox preprocess -> H2D copy ->
inference -> D2H copy -> NMS postprocess. Both reported per engine, with the
fp32 engine as the reference row.

Usage (GPU pod, PYTHONPATH=/data/YOLO-Master):
  python scripts/project03/measure_latency.py \
    --data /data/tmp/coco-qat.yaml \
    --engine fp32=path/to/model_fp32.engine \
    --engine qdq-calib=path/to/model_int8-qdq.engine \
    --n-images 300 --revalidate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import quantize_trt as qt


def e2e_bench(engine_path: Path, images: list[str], imgsz: int, conf: float,
              iou: float) -> dict:
    import tensorrt as trt
    import torch
    from ultralytics.utils import nms as _nms_mod

    logger = trt.Logger(trt.Logger.ERROR)
    rt = trt.Runtime(logger)
    raw = engine_path.read_bytes()
    meta_len = int.from_bytes(raw[:4], byteorder="little", signed=True)
    engine = rt.deserialize_cuda_engine(raw[4 + meta_len:])
    ctx = engine.create_execution_context()
    bufs, out_name = {}, None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = ctx.get_tensor_shape(name)
        t = torch.empty(max(int(np.prod(shape)), 1), dtype=torch.float32, device="cuda")
        ctx.set_tensor_address(name, t.data_ptr())
        bufs[name] = (t, tuple(shape))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            in_name = name
        else:
            out_name = name
    handle = torch.cuda.Stream().cuda_stream

    # warmup
    for _ in range(20):
        ctx.execute_async_v3(handle)
    torch.cuda.synchronize()

    pre_ms, inf_ms, post_ms, tot_ms = [], [], [], []
    for p in images:
        t0 = time.perf_counter()
        blob = qt.letterbox_blob(p, imgsz)                       # preprocess
        t1 = time.perf_counter()
        bufs[in_name][0].copy_(torch.from_numpy(blob.ravel()))   # H2D
        ctx.execute_async_v3(handle)
        out_t, out_shape = bufs[out_name]
        pred = out_t.reshape(out_shape).clone()                  # D2H implicit via NMS on gpu?
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        _nms_mod.non_max_suppression(pred, conf_thres=conf, iou_thres=iou,
                                max_det=300)                     # postprocess (gpu->cpu inside)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        pre_ms.append((t1 - t0) * 1000)
        inf_ms.append((t2 - t1) * 1000)
        post_ms.append((t3 - t2) * 1000)
        tot_ms.append((t3 - t0) * 1000)

    def stats(v):
        v = sorted(v)
        return {"median": round(v[len(v) // 2], 3), "p90": round(v[int(len(v) * 0.9)], 3)}

    return {"pre": stats(pre_ms), "infer": stats(inf_ms),
            "post": stats(post_ms), "e2e": stats(tot_ms)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--engine", action="append", required=True,
                    help="name=path, repeatable; include fp32=... as reference")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n-images", type=int, default=300)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--revalidate", action="store_true",
                    help="also run full val (mAP) per engine")
    ap.add_argument("--out", default="runs/project03/trt/latency-e2e")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import yaml as pyyaml
    cfg = pyyaml.safe_load(Path(args.data).read_text())
    val_dir = Path(cfg["path"]) / cfg["val"]
    imgs = sorted(str(p) for p in val_dir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    step = max(1, len(imgs) // args.n_images)
    imgs = imgs[::step][:args.n_images]
    print(f"[e2e] {len(imgs)} val images, conf={args.conf} iou={args.iou}")

    rows = {}
    for spec in args.engine:
        name, path = spec.split("=", 1)
        ep = Path(path)
        lat = qt.time_engine(ep, args.imgsz)
        e2e = e2e_bench(ep, imgs, args.imgsz, args.conf, args.iou)
        row = {"model_only_ms": lat["lat_ms_median"],
               "model_only_p90": lat["lat_ms_p90"], **{f"e2e_{k}": v for k, v in e2e.items()}}
        if args.revalidate:
            m50, mm = qt.eval_engine(ep, args.data, args.imgsz, 1, args.workers, 0, None)
            row["mAP50"] = round(m50, 5)
            row["mAP50_95"] = round(mm, 5)
        rows[name] = row
        print(f"[lat {name}] model-only {row['model_only_ms']}ms | "
              f"e2e {e2e['e2e']['median']}ms (pre {e2e['pre']['median']} / "
              f"infer {e2e['infer']['median']} / post {e2e['post']['median']})"
              + (f" | mAP50-95 {row.get('mAP50_95')}" if args.revalidate else ""))

    if "fp32" in rows:
        base_m = rows["fp32"]["model_only_ms"]
        base_e = rows["fp32"]["e2e_e2e"]["median"]
        for name, r in rows.items():
            r["model_only_cut_pct"] = round(100 * (1 - r["model_only_ms"] / base_m), 1)
            r["e2e_cut_pct"] = round(100 * (1 - r["e2e_e2e"]["median"] / base_e), 1)
            print(f"[vs fp32] {name}: model-only {r['model_only_cut_pct']:+.1f}% | "
                  f"e2e {r['e2e_cut_pct']:+.1f}%")
    (out / "latency_e2e.json").write_text(json.dumps(rows, indent=1))
    print(f"[done] {out}/latency_e2e.json")


if __name__ == "__main__":
    main()
