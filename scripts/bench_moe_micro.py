#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MoE micro-benchmark on MPS / CUDA / CPU.

只跑 MoE 层（不带 backbone/head），更能暴露 MoE 模块之间的真实差异。
模拟 v0.3 各层的工作点：
  - P3: c=128, H=W=80   (n-scale 时 width=0.25 → 512*0.25=128)
  - P4: c=128, H=W=40
  - P5: c=256, H=W=20

Usage:
    python scripts/bench_moe_micro.py
    python scripts/bench_moe_micro.py --bs 4 --runs 50 --warmup 10
"""
import argparse, os, sys, time, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ultralytics.nn.modules.moe import (   # noqa: E402
    UltraOptimizedMoE,
    HyperSplitMoE,
    HyperFusedMoE,
    HyperUltimateMoE,
    UltimateOptimizedMoE,
    ModularRouterExpertMoE,
    AdaptiveGateMoE,
    FusedAdaptiveGateMoE,
    HybridAdaptiveGateMoE,
    LowRankHybridAdaptiveGateMoE,
    RefinedLowRankHybridAdaptiveGateMoE,
    DetailAwareLowRankHybridAdaptiveGateMoE,
    VisualEnhancedAdaptiveGateMoE,
)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def bench(layer, x, runs, warmup, dev):
    layer.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = layer(x)
        sync(dev)
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = layer(x)
            sync(dev)
            ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {
        "median": ts[len(ts) // 2],
        "min":    min(ts),
        "p95":    ts[int(len(ts) * 0.95)],
        "params": sum(p.numel() for p in layer.parameters()),
    }


def make(klass, c, k_extra):
    """工厂：按各 MoE 类的 ctor 签名构造 (in==out)"""
    if klass in (
        HyperSplitMoE,
        HyperUltimateMoE,
        UltimateOptimizedMoE,
        AdaptiveGateMoE,
        FusedAdaptiveGateMoE,
        HybridAdaptiveGateMoE,
        LowRankHybridAdaptiveGateMoE,
        RefinedLowRankHybridAdaptiveGateMoE,
        DetailAwareLowRankHybridAdaptiveGateMoE,
        VisualEnhancedAdaptiveGateMoE,
    ):
        # 4-arg 版本 (..., split_ratio=0.5)
        return klass(c, c, k_extra["num_experts"], k_extra["top_k"], 0.5)
    return klass(c, c, k_extra["num_experts"], k_extra["top_k"])


# 与 v0.3 配置完全对齐的工作点
WORK_POINTS = [
    ("P3 (c=128, 80x80, ne=4)",  128,  80,  dict(num_experts=4,  top_k=2)),
    ("P4 (c=128, 40x40, ne=8)",  128,  40,  dict(num_experts=8,  top_k=2)),
    ("P5 (c=256, 20x20, ne=16)", 256,  20,  dict(num_experts=16, top_k=2)),
]

CANDIDATES = [
    ("ModularRouter",   ModularRouterExpertMoE),
    ("UltraOptimized",  UltraOptimizedMoE),
    ("HyperSplit",      HyperSplitMoE),
    ("HyperFused",      HyperFusedMoE),
    ("HyperUltimate",   HyperUltimateMoE),
    ("UltimateOpt",     UltimateOptimizedMoE),
    ("AdaptiveGate",    AdaptiveGateMoE),
    ("FusedAdaptive",   FusedAdaptiveGateMoE),
    ("HybridAdaptive",  HybridAdaptiveGateMoE),
    ("LowRankHybrid",   LowRankHybridAdaptiveGateMoE),
    ("RefinedLowRank",  RefinedLowRankHybridAdaptiveGateMoE),
    ("DetailAware",     DetailAwareLowRankHybridAdaptiveGateMoE),
    ("VisualEnhanced",  VisualEnhancedAdaptiveGateMoE),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    dev = pick_device()
    print(f"[micro-bench] device = {dev}, torch = {torch.__version__}")
    if dev.type == "mps":
        print(f"[micro-bench] mps name = {torch.backends.mps.get_name()}, "
              f"core count = {torch.backends.mps.get_core_count()}")
    print(f"[micro-bench] bs={args.bs} runs={args.runs} warmup={args.warmup}")
    print()

    for wp_name, c, hw, kx in WORK_POINTS:
        print(f"### {wp_name}")
        print(f"{'MoE':<18}{'params(K)':>11}{'median(ms)':>12}{'min':>9}{'p95':>9}")
        print("-" * 60)
        x = torch.randn(args.bs, c, hw, hw, device=dev)
        for name, cls in CANDIDATES:
            try:
                layer = make(cls, c, kx).to(dev)
                r = bench(layer, x, args.runs, args.warmup, dev)
                print(f"{name:<18}{r['params']/1e3:>11.1f}{r['median']:>12.3f}"
                      f"{r['min']:>9.3f}{r['p95']:>9.3f}")
                del layer
                if dev.type == "mps": torch.mps.empty_cache()
                elif dev.type == "cuda": torch.cuda.empty_cache()
            except Exception as e:
                print(f"{name:<18}  FAIL: {type(e).__name__}: {e}")
        print()

    print("[v0.3 hybrid 选型回顾]")
    print("  P3 -> HyperSplit    (浅层 FLOPs 主导, 通道分裂省算力)")
    print("  P4 -> UltraOptimized (中层稳态, 池化路由)")
    print("  P5 -> UltimateOpt   (深层语义主导, 复杂度跳过+融合)")


if __name__ == "__main__":
    main()
