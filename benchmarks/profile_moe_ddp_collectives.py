#!/usr/bin/env python3
"""Profile MoE auxiliary-loss collectives before changing DDP protocols.

Run with ``torchrun --nproc_per_node=2 benchmarks/profile_moe_ddp_collectives.py``.
Single-process runs remain useful as a compute baseline but explicitly report
that they cannot establish an all-reduce bottleneck.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics.nn.modules.moe.loss import MoELoss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    initialized_here = False
    if world > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        initialized_here = True
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    criterion = MoELoss(num_experts=args.experts, top_k=args.top_k).to(device).train()
    elapsed = 0.0
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
        + ([torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else []),
        record_shapes=False,
        profile_memory=True,
    ) as profiler:
        for _ in range(args.steps):
            probs = torch.softmax(torch.randn(args.tokens, args.experts, device=device), dim=-1).requires_grad_()
            logits = torch.log(probs.clamp_min(1e-6))
            indices = torch.topk(probs, args.top_k, dim=1).indices
            start = time.perf_counter()
            loss = criterion(probs, logits, indices)
            loss.backward()
            elapsed += time.perf_counter() - start
            profiler.step()

    events = profiler.key_averages()
    collectives = [event for event in events if "all_reduce" in event.key.lower() or "allreduce" in event.key.lower()]
    collective_self_us = sum(float(event.self_cpu_time_total) for event in collectives)
    collective_cpu_us = sum(float(event.cpu_time_total) for event in collectives)
    step_us = max(elapsed / max(args.steps, 1) * 1e6, 1.0)
    report = {
        "benchmark": "moe_ddp_collectives",
        "rank": rank,
        "world_size": world,
        "steps": args.steps,
        "tokens": args.tokens,
        "experts": args.experts,
        "all_reduce_event_count": sum(int(event.count) for event in collectives),
        "all_reduce_self_cpu_ms": collective_self_us / 1000.0,
        "all_reduce_cpu_ms": collective_cpu_us / 1000.0,
        "mean_step_wall_ms": step_us / 1000.0,
        "all_reduce_fraction_of_step_cpu": collective_cpu_us / step_us,
        "collective_timing_status": (
            "ok" if collective_cpu_us > 0 else "profiler_no_cpu_time_use_wall_measurement"
        ),
        "evidence_status": "proven_candidate" if world > 1 and collectives else "not_proven_single_process",
        "recommendation": "Do not change the communication protocol until a multi-rank run shows a material fraction."
        if world <= 1
        else "Inspect all_reduce fraction and event count before protocol changes.",
    }
    if rank == 0:
        payload = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
        print(payload)
    if initialized_here:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
