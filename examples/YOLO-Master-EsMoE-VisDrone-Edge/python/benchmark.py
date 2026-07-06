#!/usr/bin/env python3
"""Latency / throughput micro-benchmark for each export format.

Warmup rounds + measured rounds, reporting mean / p50 / p95 latency (ms/frame)
and FPS. Pre-/post-processing are included so numbers reflect real edge cost.

Usage (from edge-src/python):
    python benchmark.py --onnx exports/best.onnx --ncnn exports/best_ncnn_model \
        --mnn exports/best.mnn --pytorch runs/.../best.pt --warmup 20 --rounds 200
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from backends import create_detector
from postprocess import NmsConfig, decode_and_nms
from preprocess import letterbox


PROJ_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pytorch", type=Path, default=None)
    p.add_argument("--onnx", type=Path, default=None)
    p.add_argument("--ncnn", type=Path, default=None)
    p.add_argument("--mnn", type=Path, default=None)
    p.add_argument("--image", type=Path, default=PROJ_ROOT.parent / "datasets" / "VisDrone" / "images" / "val")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--out", type=Path, default=PROJ_ROOT.parent / "results" / "benchmark.json")
    return p.parse_args()


def benchmark_backend(kind: str, path: Path, lb_image: np.ndarray, args) -> dict:
    det = create_detector(kind, str(path), args.device, args.num_classes)
    cfg = NmsConfig(num_classes=args.num_classes)

    for _ in range(args.warmup):
        raw = det.forward(lb_image)
        decode_and_nms(raw, cfg)

    times = []
    for _ in range(args.rounds):
        t0 = time.perf_counter()
        raw = det.forward(lb_image)
        decode_and_nms(raw, cfg)
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times)
    return {
        "backend": kind,
        "rounds": args.rounds,
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "fps": float(1000.0 / arr.mean()),
    }


def main():
    args = parse_args()
    # use first val image as the benchmark input (fixed input -> stable timing)
    img_dir = args.image
    if img_dir.is_dir():
        imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
        sample = cv2.imread(str(imgs[0])) if imgs else np.zeros((640, 640, 3), np.uint8)
    else:
        sample = cv2.imread(str(img_dir))
    lb = letterbox(sample, args.imgsz)

    targets = [("pytorch", args.pytorch), ("onnx", args.onnx), ("ncnn", args.ncnn), ("mnn", args.mnn)]
    results = []
    print(f"{'backend':10s} {'mean(ms)':>10s} {'p50(ms)':>10s} {'p95(ms)':>10s} {'FPS':>8s}")
    print("-" * 52)
    for kind, path in targets:
        if path is None or not Path(path).exists():
            print(f"{kind:10s} (skipped — no model)")
            continue
        r = benchmark_backend(kind, Path(path), lb.image, args)
        results.append(r)
        print(f"{kind:10s} {r['mean_ms']:10.3f} {r['p50_ms']:10.3f} {r['p95_ms']:10.3f} {r['fps']:8.1f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n[bench] written -> {args.out}")


if __name__ == "__main__":
    main()
