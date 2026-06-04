#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPS benchmark for YOLO-Master MoE configs.

Usage (在你自己的终端直接跑):
    cd /Users/gatilin/PycharmProjects/YOLO-Master-v260601-for-MoE
    python scripts/bench_moe_mps.py
    # 默认: bs=1, imgsz=640, runs=30, warmup=5
    # 自定义:
    python scripts/bench_moe_mps.py --bs 4 --imgsz 640 --runs 50 --warmup 10
"""
import argparse, time, warnings, os, sys
warnings.filterwarnings("ignore")

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ultralytics.nn.tasks import DetectionModel  # noqa: E402

CFGS = [
    ("v0_1 ModularRouter (n)", "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml"),
    ("v0.2 UltraOptimized (n)", "ultralytics/cfg/models/master/exp/yolo-master-v0_2.yaml"),
    ("v0.3 UltimateOptimized (n)", "ultralytics/cfg/models/master/exp/yolo-master-v0_3.yaml"),
    ("v0.4 AdaptiveGate (n)", "ultralytics/cfg/models/master/exp/yolo-master-v0_4.yaml"),
    ("v0.5 FusedAdaptive (n)", "ultralytics/cfg/models/master/exp/yolo-master-v0_5.yaml"),
    ("v0.6 HybridAdaptive (n)", "ultralytics/cfg/models/master/exp/yolo-master-v0_6.yaml"),
    ("v0.7 LowRankHybrid (n)", "ultralytics/cfg/models/master/exp/yolo-master-v0_7.yaml"),
    ("v0.8 RefinedLowRank (n)", "ultralytics/cfg/models/master/exp/yolo-master-v0_8.yaml"),
    ("v0.9 DetailAware (n)", "ultralytics/cfg/models/master/exp/yolo-master-v0_9.yaml"),
    ("v0.10 VisualEnhanced (n)", "ultralytics/cfg/models/master/exp/yolo-master-v0_10.yaml"),
    ("v0 stable HybridAdaptive (n)", "ultralytics/cfg/models/master/exp/yolo-master-v0_stable.yaml"),
]


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def bench(model, x, runs, warmup, device):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    trim = max(1, len(ts) // 10)
    core = ts[trim:-trim] if len(ts) > 2 * trim else ts
    return {
        "median_ms": ts[len(ts) // 2],
        "mean_ms":   sum(core) / len(core),
        "min_ms":    min(ts),
        "p95_ms":    ts[int(len(ts) * 0.95)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    args = ap.parse_args()

    dev = pick_device()
    print(f"[bench] device = {dev}, torch = {torch.__version__}")
    if dev.type == "mps":
        print(f"[bench] mps name = {torch.backends.mps.get_name()}, "
              f"core count = {torch.backends.mps.get_core_count()}")
    print(f"[bench] bs={args.bs} imgsz={args.imgsz} runs={args.runs} warmup={args.warmup} dtype={args.dtype}")
    print("-" * 90)
    print(f"{'config':<30}{'params(M)':>11}{'median':>9}{'mean':>9}{'min':>9}{'p95':>9}{'fps':>8}")
    print("-" * 90)

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    x = torch.randn(args.bs, 3, args.imgsz, args.imgsz, device=dev, dtype=dtype)

    for name, cfg in CFGS:
        try:
            m = DetectionModel(cfg=os.path.join(ROOT, cfg), ch=3, nc=80, verbose=False)
            m = m.to(dev).to(dtype)
            n_p = sum(p.numel() for p in m.parameters()) / 1e6
            r = bench(m, x, args.runs, args.warmup, dev)
            fps = 1000.0 / r["median_ms"] * args.bs
            print(f"{name:<30}{n_p:>11.3f}{r['median_ms']:>9.2f}{r['mean_ms']:>9.2f}"
                  f"{r['min_ms']:>9.2f}{r['p95_ms']:>9.2f}{fps:>8.1f}")
            del m
            if dev.type == "mps":
                torch.mps.empty_cache()
            elif dev.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"{name:<30}  FAIL: {type(e).__name__}: {e}")

    print("-" * 90)
    print("[hint] 若 MPS 比 CPU 慢，检查:\n"
          "       1) 是否 imgsz 太小 / bs=1 (warm-up 之外的小 kernel 启动开销主导)\n"
          "       2) export PYTORCH_ENABLE_MPS_FALLBACK=1 是否启用了过多 fallback 算子\n"
          "       3) 是否有 MoE 内部用了 MPS 不支持的 op (会回落 CPU 反而拖慢)")


if __name__ == "__main__":
    main()
