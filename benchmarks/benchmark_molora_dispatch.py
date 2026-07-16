"""Local benchmark for MoLoRA grouped expert dispatch."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from ultralytics.nn.peft.molora.layer import MoLoRALayer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--experts", type=int, default=4)
    args = parser.parse_args()
    layer = MoLoRALayer(
        nn.Conv2d(args.channels, args.channels, 3, padding=1),
        r=4,
        alpha=8,
        num_experts=args.experts,
        top_k=1,
    ).eval()
    x = torch.randn(args.batch, args.channels, 32, 32)
    for _ in range(5):
        layer(x)
    start = time.perf_counter()
    for _ in range(args.steps):
        layer(x)
    elapsed = (time.perf_counter() - start) / args.steps
    print({"seconds": elapsed, "dispatch": getattr(layer, "_last_dispatch_stats", {})})


if __name__ == "__main__":
    main()
