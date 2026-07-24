#!/usr/bin/env python3
"""One-shot runner for YOLO-Master-EsMoE-N issue #52 experiments.

Pipeline:
  1. Train a baseline EsMoE-N on VisDrone (or another dataset).
  2. MoEPruner threshold sweep {0.05,0.10,0.15,0.20,0.30} on the baseline:
     direct inference + LoRA 10-epoch recovery.
  3. Dynamic schedule ablation: baseline fixed / Gini dynamic / low-coeff ablation.
  4. Generate threshold curves, Pareto front, and summary CSVs.

All hyperparameters are CLI-controllable. Intended to run on a single idle GPU
(e.g. device 6) with large batch / image size to saturate VRAM.

Example:
    ./yolo/bin/python scripts/run_issue52_full.py \
        --model-cfg ultralytics/cfg/models/master/v0/det/yolo-master-esmoe-n-visdrone.yaml \
        --data VisDrone.yaml --device 6 --batch 36 --imgsz 1344 \
        --baseline-epochs 100 --schedule-epochs 100
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.nn.modules.moe.analysis import ExpertUsageTracker  # noqa: E402
from ultralytics.nn.modules.moe.pruning import MoEPruner  # noqa: E402
from ultralytics.nn.modules.moe.scheduler import compute_gini  # noqa: E402
from ultralytics.nn.modules.moe.utils import is_core_moe_block  # noqa: E402
from ultralytics.utils.lora import apply_lora  # noqa: E402


DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30)
DEFAULT_BATCH = 36
DEFAULT_IMGSZ = 1344
PRUNE_FIELDS = (
    "threshold",
    "recovery",
    "checkpoint",
    "map50_95",
    "map50",
    "gflops",
    "latency_ms",
    "params_m",
    "mean_gini",
    "experts_per_layer",
    "layer_gini",
)
SCHEDULE_VARIANTS = {
    "baseline": {
        "name": "baseline_fixed",
        "args": {"moe_dynamic_schedule": "none", "moe_balance_loss": 1.0},
    },
    "dynamic": {
        "name": "dynamic_gini_balance",
        "args": {
            "moe_dynamic_schedule": "gini",
            "moe_balance_loss": 1.0,
            "moe_dynamic_gini_target": 0.25,
            "moe_dynamic_gini_alpha": 1.0,
            "moe_dynamic_gini_beta": 0.8,
            "moe_dynamic_balance_min": 0.5,
            "moe_dynamic_balance_max": 2.0,
        },
    },
    "ablation": {
        "name": "ablation_low_coeff",
        "args": {"moe_dynamic_schedule": "none", "moe_balance_loss": 0.3},
    },
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _benchmark_latency(model: torch.nn.Module, imgsz: int, device: torch.device, warmup: int, runs: int) -> float:
    model = model.to(device).eval()
    sample = torch.zeros(1, 3, imgsz, imgsz, device=device)
    timings = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        _sync(device)
        for _ in range(runs):
            start = time.perf_counter()
            model(sample)
            _sync(device)
            timings.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(timings)


def _expert_counts(model: torch.nn.Module) -> dict[str, int]:
    return {
        name: int(getattr(module, "num_experts", len(module.experts)))
        for name, module in model.named_modules()
        if is_core_moe_block(module) and hasattr(module, "experts")
    }


def _canonical_moe_layer_name(name: str) -> str:
    """Normalize PEFT wrapper prefixes so pre/post-recovery signatures are comparable."""
    for prefix in ("model.base_model.model.", "base_model.model."):
        if name.startswith(prefix):
            return f"model.{name.removeprefix(prefix)}"
    return name


def _expert_signature(model: torch.nn.Module) -> dict[str, dict[str, int]]:
    """Return a stable structural signature for every core MoE layer."""
    return {
        _canonical_moe_layer_name(name): {
            "num_experts": int(getattr(module, "num_experts", len(module.experts))),
            "top_k": int(getattr(module, "top_k", 0)),
        }
        for name, module in model.named_modules()
        if is_core_moe_block(module) and hasattr(module, "experts")
    }


def _freeze_detection_heads(model: torch.nn.Module) -> int:
    """Lock all detector heads after adapter injection and return their parameter count."""
    frozen = 0
    for module in model.modules():
        if not any(base.__name__ == "Detect" for base in module.__class__.mro()):
            continue
        for parameter in module.parameters():
            parameter.requires_grad = False
            frozen += parameter.numel()
    return frozen


def _detection_head_parameter_count(model: torch.nn.Module) -> int:
    """Return the number of parameters owned by concrete Detect heads."""
    return sum(
        parameter.numel()
        for module in model.modules()
        if any(base.__name__ == "Detect" for base in module.__class__.mro())
        for parameter in module.parameters()
    )


def _prepare_structure_preserving_lora(
    recovered: YOLO, args: argparse.Namespace
) -> tuple[dict[str, dict[str, int]], int]:
    """Inject LoRA into the loaded pruned graph before Trainer can rebuild it from YAML."""
    before = _expert_signature(recovered.model)
    lora_args = get_cfg(
        overrides={
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_include_moe": True,
            "lora_include_head": False,
            "lora_freeze_bn": True,
            "lora_save_adapters": False,
            "lora_gradient_checkpointing": False,
            "lora_lr_mult": args.lora_lr_mult,
            "lora_alpha_warmup": 3,
        }
    )
    recovered.model = apply_lora(recovered.model, lora_args)
    frozen_head_params = _freeze_detection_heads(recovered.model)
    after = _expert_signature(recovered.model)
    if not before or after != before:
        raise RuntimeError(f"LoRA injection changed the pruned MoE graph: before={before}, after={after}")
    if not getattr(recovered.model, "lora_enabled", False):
        raise RuntimeError("LoRA injection did not activate an adapter model")
    if frozen_head_params <= 0:
        raise RuntimeError("No detection-head parameters were found to lock")
    return before, frozen_head_params


def _tracker_gini(tracker: ExpertUsageTracker, model: torch.nn.Module) -> dict[str, float]:
    result = {}
    for layer_name, module in model.named_modules():
        if not is_core_moe_block(module):
            continue
        router_name = f"{layer_name}.routing" if layer_name else "routing"
        stats = tracker.usage_stats.get(router_name, {})
        num_experts = int(getattr(module, "num_experts", len(stats)))
        hits = [float(stats[i].hits) if i in stats else 0.0 for i in range(num_experts)]
        result[layer_name] = compute_gini(torch.tensor(hits, dtype=torch.float32))
    return result


def _evaluate(
    checkpoint: Path,
    data: str,
    device_arg: str,
    imgsz: int,
    batch: int,
    workers: int,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    yolo = YOLO(str(checkpoint))
    with ExpertUsageTracker(yolo.model) as tracker:
        metrics = yolo.val(
            data=data,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            device=device_arg,
            verbose=False,
            plots=False,
        )
    device = next(yolo.model.parameters()).device
    if device.type == "cpu" and torch.cuda.is_available():
        ordinal = int(device_arg) if device_arg.isdigit() and int(device_arg) < torch.cuda.device_count() else 0
        device = torch.device(f"cuda:{ordinal}")
    layer_gini = _tracker_gini(tracker, yolo.model)
    _, params, _, gflops = yolo.info()
    return {
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "gflops": float(gflops),
        "latency_ms": _benchmark_latency(yolo.model, imgsz, device, warmup, runs),
        "params_m": float(params) / 1e6,
        "mean_gini": sum(layer_gini.values()) / len(layer_gini) if layer_gini else 0.0,
        "experts_per_layer": json.dumps(_expert_counts(yolo.model), sort_keys=True),
        "layer_gini": json.dumps(layer_gini, sort_keys=True),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_pruning(results_csv: Path, out_dir: Path, max_map_drop: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[dict[str, str]] = []
    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    def f(key: str, row: dict[str, str]) -> float | None:
        try:
            return float(row[key])
        except (KeyError, ValueError, TypeError):
            return None

    for metric in ("map50_95", "gflops", "latency_ms"):
        usable = [r for r in rows if f("threshold", r) is not None and f(metric, r) is not None]
        if not usable:
            continue
        plt.figure(figsize=(7, 4))
        for recovery in ("direct", "lora10"):
            group = sorted(
                (r for r in usable if r["recovery"] == recovery),
                key=lambda r: f("threshold", r) or 0.0,
            )
            if not group:
                continue
            plt.plot(
                [f("threshold", r) for r in group],
                [f(metric, r) for r in group],
                marker="o",
                label=recovery,
            )
        plt.xlabel("Pruning threshold")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out = out_dir / f"threshold_{metric}.png"
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"[plot] {out}")

    scored = [r for r in rows if f("latency_ms", r) is not None and f("map50_95", r) is not None]
    if scored:
        scored.sort(key=lambda r: (f("latency_ms", r) or 0.0, -(f("map50_95", r) or 0.0)))
        front: list[dict[str, str]] = []
        best_map = -1.0
        for row in scored:
            map_val = f("map50_95", row) or 0.0
            if map_val > best_map:
                front.append(row)
                best_map = map_val
        plt.figure(figsize=(6, 5))
        for row in scored:
            x = f("latency_ms", row)
            y = f("map50_95", row)
            if x is None or y is None:
                continue
            plt.scatter(x, y, alpha=0.6)
            plt.text(x, y, f"{row['threshold']}/{row['recovery']}", fontsize=8)
        plt.plot(
            [f("latency_ms", r) for r in front],
            [f("map50_95", r) for r in front],
            linewidth=2,
            label="Pareto front",
        )
        dense = next((r for r in rows if r["recovery"] == "dense"), None)
        dense_map = f("map50_95", dense) if dense else None
        feasible = [
            row
            for row in front
            if row["recovery"] != "dense"
            and dense_map is not None
            and (dense_map - (f("map50_95", row) or 0.0)) <= max_map_drop
            and row.get("experts_per_layer") != dense.get("experts_per_layer")
        ]
        sweet = (
            min(feasible, key=lambda r: (f("latency_ms", r) or float("inf"), -(f("map50_95", r) or 0.0)))
            if feasible
            else None
        )
        if sweet:
            plt.scatter([f("latency_ms", sweet)], [f("map50_95", sweet)], s=160, marker="*", label="sweet spot")
        plt.xlabel("Latency (ms)")
        plt.ylabel("mAP50-95")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out = out_dir / "pareto_accuracy_latency.png"
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"[plot] {out}")
        if sweet:
            print(f"[plot] sweet spot: threshold={sweet['threshold']} recovery={sweet['recovery']}")
        else:
            print(f"[plot] no non-trivial sweet spot within mAP50-95 drop <= {max_map_drop}")

    points_3d = [
        row
        for row in rows
        if row["recovery"] != "dense"
        and all(f(key, row) is not None for key in ("threshold", "gflops", "latency_ms", "map50_95"))
    ]
    if points_3d:
        fig = plt.figure(figsize=(8, 6))
        axis = fig.add_subplot(111, projection="3d")
        scatter = axis.scatter(
            [f("threshold", row) for row in points_3d],
            [f("gflops", row) for row in points_3d],
            [f("latency_ms", row) for row in points_3d],
            c=[f("map50_95", row) for row in points_3d],
            cmap="viridis",
        )
        axis.set_xlabel("Threshold")
        axis.set_ylabel("GFLOPs")
        axis.set_zlabel("Latency (ms)")
        fig.colorbar(scatter, ax=axis, label="mAP50-95", shrink=0.7)
        fig.tight_layout()
        out = out_dir / "threshold_map_flops_latency_3d.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"[plot] {out}")


def _analyze_pruning(results_csv: Path, out_dir: Path, max_map_drop: float) -> Path:
    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    dense = next((row for row in rows if row["recovery"] == "dense"), None)
    if dense is None:
        raise RuntimeError("Pruning analysis requires a dense baseline row")

    def number(row: dict[str, str], key: str) -> float:
        return float(row[key])

    scored = sorted(rows, key=lambda row: (number(row, "latency_ms"), -number(row, "map50_95")))
    pareto, best_map = [], -1.0
    for row in scored:
        if number(row, "map50_95") > best_map:
            pareto.append(row)
            best_map = number(row, "map50_95")
    pareto_path = out_dir / "pareto.csv"
    _write_csv(pareto_path, pareto)

    dense_map = number(dense, "map50_95")
    feasible = [
        row
        for row in pareto
        if row["recovery"] != "dense"
        and dense_map - number(row, "map50_95") <= max_map_drop
        and row["experts_per_layer"] != dense["experts_per_layer"]
    ]
    sweet = min(feasible, key=lambda row: (number(row, "latency_ms"), -number(row, "map50_95"))) if feasible else None
    recommendation = {
        "quality_gate": {"metric": "mAP50-95", "max_absolute_drop": max_map_drop},
        "dense_baseline": dense,
        "sweet_spot": sweet,
        "sweet_spot_status": "observed" if sweet else "not_observed",
        "server": sweet or dense,
        "edge": sweet or dense,
        "note": (
            "Use the fastest structurally pruned Pareto point within the quality gate."
            if sweet
            else "No structurally pruned point passed the quality gate; retain the dense model for both scenarios."
        ),
    }
    out = out_dir / "recommendations.json"
    out.write_text(json.dumps(recommendation, indent=2), encoding="utf-8")
    return out


def _summarize_schedule(project: Path) -> Path:
    rows_by_variant: dict[str, list[dict[str, str]]] = {}
    for variant in SCHEDULE_VARIANTS.values():
        csv_path = project / variant["name"] / "results.csv"
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows_by_variant[variant["name"]] = [
                    {k.strip(): v for k, v in row.items()} for row in csv.DictReader(handle)
                ]
        else:
            rows_by_variant[variant["name"]] = []

    metric_key = "metrics/mAP50-95(B)"
    baseline_rows = rows_by_variant[SCHEDULE_VARIANTS["baseline"]["name"]]
    baseline_final = float(baseline_rows[-1].get(metric_key, "nan")) if baseline_rows else float("nan")
    target = baseline_final * 0.95 if baseline_final == baseline_final else float("nan")

    def first_at(rows: list[dict[str, str]], tgt: float) -> int | None:
        for idx, row in enumerate(rows, start=1):
            try:
                if float(row.get(metric_key, "nan")) >= tgt:
                    return idx
            except (TypeError, ValueError):
                continue
        return None

    baseline_epoch = first_at(baseline_rows, target) if target == target else None
    summary_rows = []
    for key, variant in SCHEDULE_VARIANTS.items():
        rows = rows_by_variant[variant["name"]]
        final = rows[-1] if rows else {}
        best = max(rows, key=lambda r: float(r.get(metric_key, "nan") or "nan")) if rows else {}
        reach = first_at(rows, target) if target == target else None
        trace_path = project / variant["name"] / "moe_dynamic_schedule.csv"
        trace_rows = []
        if trace_path.exists():
            with trace_path.open(newline="", encoding="utf-8") as handle:
                trace_rows = list(csv.DictReader(handle))
        summary_rows.append(
            {
                "variant": key,
                "run_dir": str(project / variant["name"]),
                "epochs": len(rows),
                "final_mAP50-95": final.get(metric_key, ""),
                "best_mAP50-95": best.get(metric_key, ""),
                "target_95pct_baseline_mAP50-95": target if target == target else "",
                "epoch_to_target": reach or "",
                "convergence_epoch_ratio": (reach / baseline_epoch) if reach and baseline_epoch else "",
                "convergence_speedup": (baseline_epoch / reach) if reach and baseline_epoch else "",
                "mean_gini": (
                    sum(float(row["mean_gini"]) for row in trace_rows) / len(trace_rows) if trace_rows else ""
                ),
                "final_gini": trace_rows[-1]["mean_gini"] if trace_rows else "",
                "mean_balance_loss_coeff": (
                    sum(float(row["balance_loss_coeff"]) for row in trace_rows) / len(trace_rows)
                    if trace_rows
                    else variant["args"]["moe_balance_loss"]
                ),
                "final_balance_loss_coeff": (
                    trace_rows[-1]["balance_loss_coeff"] if trace_rows else variant["args"]["moe_balance_loss"]
                ),
            }
        )
    out = project / "schedule_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "run_dir",
                "epochs",
                "final_mAP50-95",
                "best_mAP50-95",
                "target_95pct_baseline_mAP50-95",
                "epoch_to_target",
                "convergence_epoch_ratio",
                "convergence_speedup",
                "mean_gini",
                "final_gini",
                "mean_balance_loss_coeff",
                "final_balance_loss_coeff",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    return out


def _train_baseline(args: argparse.Namespace) -> Path:
    if args.baseline_checkpoint is not None:
        checkpoint = args.baseline_checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Baseline checkpoint not found: {checkpoint}")
        print(f"[baseline] use supplied checkpoint {checkpoint}")
        return checkpoint
    project = args.output / "baseline"
    project.mkdir(parents=True, exist_ok=True)
    best = project / "train" / "weights" / "best.pt"
    last = project / "train" / "weights" / "last.pt"
    if args.skip_existing and (best.exists() or last.exists()):
        checkpoint = best if best.exists() else last
        print(f"[baseline] reuse {checkpoint}")
        return checkpoint
    model = YOLO(str(args.model_cfg))
    model.train(
        data=str(args.data),
        epochs=args.baseline_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        project=str(project),
        name="train",
        exist_ok=True,
        pretrained=False,
        val=True,
        plots=False,
        amp=args.amp,
        moe_dynamic_schedule="none",
        moe_balance_loss=1.0,
    )
    return best


def _run_pruning_sweep(args: argparse.Namespace, baseline_ckpt: Path) -> None:
    sweep_dir = args.output / "pruning"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    thresholds = tuple(args.thresholds)

    results_csv = sweep_dir / "results.csv"
    rows: list[dict[str, Any]] = []
    if args.skip_existing and results_csv.exists():
        with results_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    if args.rerun_lora:
        rows = [
            row
            for row in rows
            if row["recovery"] != "lora10" or "lora_recovery_structural" in row.get("checkpoint", "")
        ]
        if rows:
            _write_csv(results_csv, rows)

    def completed(threshold: float, recovery: str) -> bool:
        return any(float(row["threshold"]) == threshold and row["recovery"] == recovery for row in rows)

    if not completed(0.0, "dense"):
        dense = _evaluate(
            baseline_ckpt,
            str(args.data),
            args.device,
            args.imgsz,
            args.batch,
            args.workers,
            args.warmup,
            args.runs,
        )
        rows.append({"threshold": 0.0, "recovery": "dense", "checkpoint": str(baseline_ckpt), **dense})
        _write_csv(results_csv, rows)

    calibration = MoEPruner(
        str(baseline_ckpt), thresholds[0], str(args.data), device=args.device, importance_mode=args.importance_mode
    )
    usage_stats = calibration.collect_usage()

    for threshold in thresholds:
        tag = f"t{int(round(threshold * 100)):02d}"
        point_dir = sweep_dir / f"threshold_{tag}"
        point_dir.mkdir(parents=True, exist_ok=True)
        pruned_path = point_dir / f"pruned_{tag}.pt"

        if not pruned_path.exists() or not args.skip_existing:
            pruner = MoEPruner(
                str(baseline_ckpt),
                threshold,
                str(args.data),
                device=args.device,
                usage_stats=usage_stats,
                importance_mode=args.importance_mode,
            )
            if not pruner.prune(str(pruned_path)):
                raise RuntimeError(f"Pruning failed at threshold={threshold}")

        if not completed(threshold, "direct"):
            direct = _evaluate(
                pruned_path,
                str(args.data),
                args.device,
                args.imgsz,
                args.batch,
                args.workers,
                args.warmup,
                args.runs,
            )
            rows.append({"threshold": threshold, "recovery": "direct", "checkpoint": str(pruned_path), **direct})
            _write_csv(results_csv, rows)
            print(f"[pruning t={threshold}] direct map50-95={direct['map50_95']:.4f}")

        recovery_dir = point_dir / "lora_recovery_structural"
        recovered_path = recovery_dir / "lora10" / "weights" / "best.pt"
        source_signature = _expert_signature(YOLO(str(pruned_path)).model)
        frozen_head_params = 0
        if not recovered_path.exists() or not args.skip_existing:
            recovered = YOLO(str(pruned_path))
            source_signature, frozen_head_params = _prepare_structure_preserving_lora(recovered, args)
            recovered.train(
                data=str(args.data),
                epochs=args.lora_epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                workers=args.workers,
                device=args.device,
                seed=args.seed,
                project=str(recovery_dir),
                name="lora10",
                exist_ok=True,
                val=True,
                plots=False,
                amp=args.amp,
                pretrained=True,
                lr0=args.lora_lr0,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_lr_mult=args.lora_lr_mult,
                lora_include_moe=True,
                lora_include_head=False,
                lora_freeze_bn=True,
                lora_gradient_checkpointing=False,
                lora_alpha_warmup=3,
                lora_save_adapters=False,
            )
        recovered_model = YOLO(str(recovered_path))
        frozen_head_params = frozen_head_params or _detection_head_parameter_count(recovered_model.model)
        recovered_signature = _expert_signature(recovered_model.model)
        if recovered_signature != source_signature:
            raise RuntimeError(
                f"Recovered checkpoint changed the pruned MoE graph: source={source_signature}, "
                f"recovered={recovered_signature}"
            )
        signature_path = recovery_dir / "lora10" / "structure_signature.json"
        signature_path.write_text(
            json.dumps(
                {
                    "status": "preserved",
                    "source_checkpoint": str(pruned_path),
                    "recovered_checkpoint": str(recovered_path),
                    "source_signature": source_signature,
                    "recovered_signature": recovered_signature,
                    "detection_head_locked": True,
                    "frozen_head_params_at_injection": frozen_head_params,
                    "lr0": args.lora_lr0,
                    "lora_lr_mult": args.lora_lr_mult,
                    "lora_rank": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if not completed(threshold, "lora10"):
            lora = _evaluate(
                recovered_path,
                str(args.data),
                args.device,
                args.imgsz,
                args.batch,
                args.workers,
                args.warmup,
                args.runs,
            )
            rows.append({"threshold": threshold, "recovery": "lora10", "checkpoint": str(recovered_path), **lora})
            _write_csv(results_csv, rows)
            print(f"[pruning t={threshold}] lora10 map50-95={lora['map50_95']:.4f}")

    rows.sort(key=lambda row: (float(row["threshold"]), {"dense": 0, "direct": 1, "lora10": 2}[row["recovery"]]))
    _write_csv(results_csv, rows)
    _plot_pruning(results_csv, sweep_dir / "plots", args.max_map_drop)
    recommendation = _analyze_pruning(results_csv, sweep_dir, args.max_map_drop)
    print(f"[pruning] recommendation -> {recommendation}")


def _run_schedule_ablation(args: argparse.Namespace, baseline_ckpt: Path) -> None:
    project = args.output / "schedule"
    project.mkdir(parents=True, exist_ok=True)
    init_checkpoint = project / "initial_state.pt"
    if not init_checkpoint.exists():
        initial_model = YOLO(str(baseline_ckpt))
        initial_model.save(str(init_checkpoint))
        print(f"[schedule] shared baseline state -> {init_checkpoint}")
    for key, variant in SCHEDULE_VARIANTS.items():
        run_dir = project / variant["name"]
        if args.skip_existing and (run_dir / "results.csv").exists():
            print(f"[schedule] skip existing {variant['name']}")
            continue
        print(f"[schedule] training {key}: {variant['args']}")
        model = YOLO(str(init_checkpoint))
        model.train(
            data=str(args.data),
            epochs=args.schedule_epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            seed=args.seed,
            project=str(project),
            name=variant["name"],
            exist_ok=True,
            pretrained=True,
            val=True,
            plots=False,
            amp=args.amp,
            moe_map_saturation_enabled=False,
            **variant["args"],
        )
    summary = _summarize_schedule(project)
    print(f"[schedule] summary -> {summary}")


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generate_report(args: argparse.Namespace, baseline_ckpt: Path) -> Path:
    pruning_csv = args.output / "pruning" / "results.csv"
    schedule_csv = args.output / "schedule" / "schedule_summary.csv"
    recommendation_path = args.output / "pruning" / "recommendations.json"

    pruning_rows = []
    if pruning_csv.exists():
        with pruning_csv.open(newline="", encoding="utf-8") as handle:
            pruning_rows = list(csv.DictReader(handle))
    schedule_rows = []
    if schedule_csv.exists():
        with schedule_csv.open(newline="", encoding="utf-8") as handle:
            schedule_rows = list(csv.DictReader(handle))
    recommendation = json.loads(recommendation_path.read_text(encoding="utf-8")) if recommendation_path.exists() else {}

    lines = [
        "# Issue #52 — MoE 专家剪枝与动态超参数调度实验报告",
        "",
        "本报告由 `scripts/run_issue52_full.py` 从机器可读实验产物生成，不填充或推测缺失指标。",
        "",
        "## 实验设置",
        "",
        f"- 数据集：`{args.data}`",
        f"- 模型配置：`{args.model_cfg}`",
        f"- 基线 checkpoint：`{baseline_ckpt}`",
        f"- checkpoint SHA-256：`{_checkpoint_sha256(baseline_ckpt)}`",
        f"- 阈值：`{', '.join(f'{value:.2f}' for value in args.thresholds)}`",
        f"- LoRA 恢复：`{args.lora_epochs}` epochs，rank `{args.lora_r}`，alpha `{args.lora_alpha}`",
        f"- 延迟：batch 1、输入 `{args.imgsz}×{args.imgsz}`、median of `{args.runs}` runs after `{args.warmup}` warmups",
        f"- 随机种子：`{args.seed}`",
        f"- 复现分支：[{args.repo_url}]({args.repo_url})",
        "",
        "## 动态调度公式",
        "",
        "每个训练 epoch 累计所有 batch、所有核心 MoE 层的专家利用率，逐层计算 Gini 后取均值：",
        "",
        "```text",
        "ema_t = beta * ema_(t-1) + (1 - beta) * gini_t",
        "coeff_(t+1) = clip(base_coeff * exp(alpha * (ema_t - target_gini)), min_coeff, max_coeff)",
        "```",
        "",
        "Gini 高于目标时增强均衡损失以抑制路由坍塌；低于目标时减弱约束以允许专家分化。功能默认关闭，",
        "恢复失败或 NaN recovery 拒绝的 epoch 不推进 EMA、系数或 trace。",
        "",
        "## 剪枝结果",
        "",
    ]
    if pruning_rows:
        lines.extend(
            [
                "| threshold | recovery | mAP50-95 | mAP50 | GFLOPs | latency ms | Params M | mean Gini | experts/layer |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in pruning_rows:
            lines.append(
                f"| {float(row['threshold']):.2f} | {row['recovery']} | {float(row['map50_95']):.6f} | "
                f"{float(row['map50']):.6f} | {float(row['gflops']):.3f} | {float(row['latency_ms']):.3f} | "
                f"{float(row['params_m']):.3f} | {float(row['mean_gini']):.6f} | `{row['experts_per_layer']}` |"
            )
    else:
        lines.append("尚未生成剪枝结果；运行时不要使用 `--skip-pruning`。")

    lines.extend(["", "## 动态调度三组对照", ""])
    if schedule_rows:
        lines.extend(
            [
                "| variant | final mAP50-95 | best mAP50-95 | epoch→95% | ratio | speedup | mean/final Gini | mean/final coeff |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for row in schedule_rows:
            lines.append(
                f"| {row['variant']} | {row['final_mAP50-95']} | {row['best_mAP50-95']} | "
                f"{row['epoch_to_target']} | {row['convergence_epoch_ratio']} | {row['convergence_speedup']} | "
                f"{row['mean_gini']}/{row['final_gini']} | "
                f"{row['mean_balance_loss_coeff']}/{row['final_balance_loss_coeff']} |"
            )
    else:
        lines.append("尚未生成三组调度结果；运行时不要使用 `--skip-schedule`。")

    side_effects = []
    for row in schedule_rows:
        try:
            final_map, best_map = float(row["final_mAP50-95"]), float(row["best_mAP50-95"])
        except (KeyError, TypeError, ValueError):
            continue
        if best_map > 0 and final_map < best_map * 0.8:
            side_effects.append(f"`{row['variant']}` 后期精度坍塌：final 低于 best 的 80%。")
    if schedule_rows:
        by_variant = {row["variant"]: row for row in schedule_rows}
        if "baseline" in by_variant and "dynamic" in by_variant:
            try:
                delta = float(by_variant["dynamic"]["final_mAP50-95"]) - float(by_variant["baseline"]["final_mAP50-95"])
                side_effects.append(f"动态组相对固定基线 final mAP50-95 差值：`{delta:+.6f}`。")
            except (TypeError, ValueError):
                pass
    lines.extend(["", "## 副作用与改进建议", ""])
    lines.extend(f"- {item}" for item in side_effects)
    lines.extend(
        [
            "- 若系数振荡，增大 `moe_dynamic_gini_beta`；若过度均衡导致专家同质化，降低 `alpha` 或提高 target。",
            "- 收敛结论至少应追加 3 个随机种子；单 seed 结果只能视为候选现象。",
            "- 若 baseline final 已发生坍塌，95% final 指标会失去区分力，应同时报告 best-checkpoint 目标。",
            "",
            "## 场景推荐与 Sweet Spot",
            "",
            f"- 状态：`{recommendation.get('sweet_spot_status', 'pending')}`",
            f"- 结论：{recommendation.get('note', '等待剪枝实验结果。')}",
            f"- 服务器端：`{json.dumps(recommendation.get('server'), ensure_ascii=False)}`",
            f"- 边缘端：`{json.dumps(recommendation.get('edge'), ensure_ascii=False)}`",
            "",
            "## 复现",
            "",
            "```bash",
            "yolo/bin/python scripts/run_issue52_full.py --device 0 --skip-existing",
            "```",
            "",
            "主要产物：`pruning/results.csv`、`pruning/pareto.csv`、`pruning/recommendations.json`、",
            "`pruning/plots/`、`schedule/schedule_summary.csv` 以及 dynamic 组的 `moe_dynamic_schedule.csv`。",
        ]
    )
    out = args.output / "issue52_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "dataset": str(args.data),
        "model_cfg": str(args.model_cfg),
        "baseline_checkpoint": str(baseline_ckpt),
        "baseline_sha256": _checkpoint_sha256(baseline_ckpt),
        "thresholds": list(args.thresholds),
        "seed": args.seed,
        "report": str(out),
    }
    (args.output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def run(args: argparse.Namespace) -> int:
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    required_thresholds = set(DEFAULT_THRESHOLDS)
    if not required_thresholds.issubset(set(args.thresholds)):
        raise ValueError(f"Issue #52 requires thresholds {sorted(required_thresholds)}; got {args.thresholds}")
    if args.lora_epochs != 10:
        raise ValueError(f"Issue #52 requires LoRA recovery for 10 epochs; got {args.lora_epochs}")

    print(f"[issue52] model-cfg={args.model_cfg}")
    print(f"[issue52] data={args.data}")
    print(f"[issue52] device={args.device} batch={args.batch} imgsz={args.imgsz}")
    print(f"[issue52] output={args.output}")
    print(f"[issue52] stages: pruning={not args.skip_pruning} schedule={not args.skip_schedule}")
    if args.dry_run:
        return 0

    baseline_ckpt = _train_baseline(args)
    print(f"[issue52] baseline checkpoint -> {baseline_ckpt}")

    if not args.skip_pruning:
        _run_pruning_sweep(args, baseline_ckpt)
    if not args.skip_schedule:
        _run_schedule_ablation(args, baseline_ckpt)
    report = _generate_report(args, baseline_ckpt)

    print(f"[issue52] report -> {report}")
    print(f"[issue52] all done. Results in {args.output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-cfg",
        type=Path,
        default=ROOT / "ultralytics/cfg/models/master/v0/det/yolo-master-esmoe-n-visdrone.yaml",
    )
    parser.add_argument("--data", default="VisDrone.yaml")
    parser.add_argument("--baseline-checkpoint", type=Path, help="Reuse a trained baseline instead of training one.")
    parser.add_argument("--device", default="6")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--baseline-epochs", type=int, default=100)
    parser.add_argument("--schedule-epochs", type=int, default=100)
    parser.add_argument("--lora-epochs", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-lr0", type=float, default=1e-3)
    parser.add_argument("--lora-lr-mult", type=float, default=1.0)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--importance-mode", choices=("usage", "usage_weight"), default="usage_weight")
    parser.add_argument("--max-map-drop", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/issue52_full_visdrone")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--rerun-lora",
        action="store_true",
        help="Discard legacy LoRA rows and resume/re-run structure-preserving recovery outputs.",
    )
    parser.add_argument("--skip-pruning", action="store_true")
    parser.add_argument("--skip-schedule", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--repo-url",
        default="https://github.com/vankari/YOLO-Master/tree/moe-schedule-study",
        help="Public experiment-script link embedded in the generated report.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
