#!/usr/bin/env python3
"""Benchmark Regional MoA attention with explicit KV-token budgets.

The micro benchmark is intentionally independent of a checkpoint.  Optional
validation arguments run real mAP evaluation only when both weights and data
are supplied; otherwise the JSON reports ``not_run`` instead of guessing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics.nn.modules.moa.heads import _RegionalAttnHead


def _device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if value == "mps" and not getattr(torch.backends, "mps", None):
        return torch.device("cpu")
    return torch.device(value)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _peak_memory(device: torch.device) -> int | None:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return None


def _evaluate_map(args: argparse.Namespace, budget: int | None) -> dict:
    if not args.weights or not args.data:
        return {"status": "not_run", "reason": "provide --weights and --data for matched validation"}
    try:
        from ultralytics import YOLO
        from ultralytics.nn.modules.moa.heads import _RegionalAttnHead

        model = YOLO(args.weights)
        for module in model.model.modules():
            if isinstance(module, _RegionalAttnHead):
                module.max_kv_tokens = budget
        metrics = model.val(
            data=args.data,
            imgsz=args.imgsz,
            split=args.split,
            device=args.device,
            batch=args.batch,
            # Pass the budget through the normal validator config path too;
            # direct module mutation alone can be overwritten by runtime
            # mixture-config resolution during validation.
            moa_regional_max_kv_tokens=0 if budget is None else budget,
            workers=0,
            verbose=False,
            plots=False,
        )
        box = getattr(metrics, "box", None)
        return {
            "status": "ok",
            "map50": float(box.map50) if box is not None else None,
            "map50_95": float(box.map) if box is not None else None,
        }
    except Exception as exc:  # pragma: no cover - depends on optional data/checkpoint
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def benchmark_one(args: argparse.Namespace, budget: int | None) -> dict:
    device = _device(args.device)
    head = (
        _RegionalAttnHead(
            args.channels,
            args.heads,
            head_dim=args.head_dim,
            pool_stride=args.pool_stride,
            max_kv_tokens=budget,
        )
        .eval()
        .to(device)
    )
    x = torch.randn(args.batch, args.channels, args.height, args.width, device=device)
    pooled: list[tuple[int, int]] = []
    hook = head.kv_proj.register_forward_hook(lambda _, inputs, __: pooled.append(tuple(inputs[0].shape[-2:])))
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            for _ in range(args.warmup):
                head(x)
                _sync(device)
            times = []
            for _ in range(args.runs):
                start = time.perf_counter()
                head(x)
                _sync(device)
                times.append((time.perf_counter() - start) * 1000.0)
    finally:
        hook.remove()
    return {
        "max_kv_tokens": budget,
        "device": str(device),
        "shape": [args.batch, args.channels, args.height, args.width],
        "latency_ms_mean": sum(times) / len(times),
        "latency_ms_p50": sorted(times)[len(times) // 2],
        "peak_memory_bytes": _peak_memory(device),
        "pooled_shape": list(pooled[-1]) if pooled else None,
        "pooled_tokens": pooled[-1][0] * pooled[-1][1] if pooled else None,
        "mAP": _evaluate_map(args, budget),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-budget", nargs="+", default=["4096", "8192", "none"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--pool-stride", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--weights", help="Optional trained YOLO checkpoint for matched mAP validation")
    parser.add_argument("--data", help="Optional dataset YAML used with --weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()
    budgets = [None if value.lower() == "none" else int(value) for value in args.kv_budget]
    report = {
        "benchmark": "moa_regional_head",
        "results": [benchmark_one(args, budget) for budget in budgets],
        "mAP_note": "Run matched validation separately with a trained checkpoint and dataset; no random-input mAP is reported.",
    }
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
