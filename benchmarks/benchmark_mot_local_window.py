#!/usr/bin/env python3
"""P3/P4-style LocalConv global-vs-window attention ablation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics.nn.modules.mot.experts import _LocalConvTransformerExpert


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _evaluate_map(args: argparse.Namespace, window: int) -> dict:
    if not args.weights or not args.data:
        return {"status": "not_run", "reason": "provide --weights and --data for matched validation"}
    try:
        from ultralytics import YOLO
        from ultralytics.nn.modules.mot.experts import _LocalConvTransformerExpert

        model = YOLO(args.weights)
        for module in model.model.modules():
            if isinstance(module, _LocalConvTransformerExpert):
                module.local_window_size = window
        metrics = model.val(
            data=args.data,
            imgsz=args.imgsz,
            split=args.split,
            device=args.device,
            batch=args.batch,
            # Keep the ablation value when the validator reapplies runtime
            # mixture settings after loading the checkpoint.
            mot_local_attn_window=window,
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


def run_case(args: argparse.Namespace, window: int) -> dict:
    device = torch.device(args.device)
    expert = _LocalConvTransformerExpert(args.channels, args.heads, local_window_size=window).eval().to(device)
    x = torch.randn(args.batch, args.channels, args.height, args.width, device=device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            expert(x)
            _sync(device)
        times = []
        for _ in range(args.runs):
            start = time.perf_counter()
            output = expert(x)
            _sync(device)
            times.append((time.perf_counter() - start) * 1000.0)
    return {
        "stage": args.stage,
        "local_attn_window": window,
        "shape": [args.batch, args.channels, args.height, args.width],
        "latency_ms_mean": sum(times) / len(times),
        "latency_ms_p50": sorted(times)[len(times) // 2],
        "output_shape": list(output.shape),
        "mAP": _evaluate_map(args, window),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("P3", "P4"), default="P3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--window", nargs="+", type=int, default=[0, 7])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--weights", help="Optional trained YOLO checkpoint for matched mAP validation")
    parser.add_argument("--data", help="Optional dataset YAML used with --weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()
    if args.stage == "P4" and args.height == 80 and args.width == 80:
        args.height = args.width = 40
    payload = json.dumps({"benchmark": "mot_local_window", "results": [run_case(args, w) for w in args.window]}, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
