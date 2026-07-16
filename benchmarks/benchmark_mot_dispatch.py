"""Small local benchmark for MoT dense vs sample-sparse dispatch."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ultralytics.nn.modules.mot import MoTBlock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--size", type=int, default=32)
    args = parser.parse_args()
    x = torch.randn(args.batch, 32, args.size, args.size)
    for sparse in (False, True):
        block = MoTBlock(32, num_heads=4, top_k=1, sparse_train=False).eval()
        if sparse:
            block.sparse_train = False
        for _ in range(3):
            block(x)
        start = time.perf_counter()
        for _ in range(args.steps):
            block(x)
        elapsed = (time.perf_counter() - start) / args.steps
        print({"mode": "sparse" if sparse else "dense", "seconds": elapsed, "dispatch": block._last_dispatch_stats})


if __name__ == "__main__":
    main()
