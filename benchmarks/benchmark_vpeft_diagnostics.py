#!/usr/bin/env python3
"""Record V-PEFT candidate count, solve time, and discrete solution quality."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics.vpeft import ComputationGraphBuilder, ConstraintRegistry
from ultralytics.vpeft.solver import AlternatingOptimizationSolver, DifferentiableOptimizationSolver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=("ao", "dco"), default="dco")
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--budget", type=int, default=100_000)
    parser.add_argument("--optimize-variant", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    model = nn.Sequential(nn.Conv2d(8, 16, 3), nn.Conv2d(16, 16, 1), nn.Linear(16, 16))
    graph = ComputationGraphBuilder().build(model)
    constraints = ConstraintRegistry.default({"max_params": args.budget})
    if args.solver == "dco":
        solver = DifferentiableOptimizationSolver(
            max_iter=args.max_iter,
            optimize_variant=args.optimize_variant,
        )
    else:
        solver = AlternatingOptimizationSolver(max_iter=args.max_iter)
    started = time.perf_counter()
    decision = solver.solve(graph, args.budget, "lora", constraints)
    elapsed = time.perf_counter() - started
    metadata = dict(decision.metadata or {})
    metadata.setdefault("elapsed_seconds", elapsed)
    metadata.setdefault("n_nodes", graph.n_nodes)
    metadata.setdefault("n_variant_candidates", 1)
    metadata.setdefault("final_utility", float(decision.utility))
    metadata.setdefault("budget_used", int(decision.budget_used))
    metadata.setdefault("budget_remaining", int(decision.budget_remaining))
    metadata.setdefault("target_module_count", len(decision.target_modules))
    metadata["quality"] = {
        "status": decision.status,
        "utility": float(decision.utility),
        "budget_feasible": decision.budget_used <= args.budget,
        "target_module_count": len(decision.target_modules),
    }
    payload = json.dumps({"benchmark": "vpeft_diagnostics", "metadata": metadata}, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
