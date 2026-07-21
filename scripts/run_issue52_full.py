#!/usr/bin/env python3
"""One-shot runner for YOLO-Master-EsMoE-N issue #52 experiments.

Pipeline:
  1. Train a baseline EsMoE-N on VisDrone (or another dataset).
  2. MoEPruner threshold sweep {0.05,0.10,0.15,0.20,0.30} on the baseline:
     direct inference + LoRA 10-epoch recovery.
  3. Dynamic schedule ablation: baseline fixed / map-saturation dynamic / low-coeff ablation.
  4. Generate threshold curves, Pareto front, and summary CSVs.

All hyperparameters are CLI-controllable. Intended to run on a single idle GPU
(e.g. device 6) with large batch / image size to saturate VRAM.

Example:
    ./yolo/bin/python scripts/run_issue52_full.py \
        --model-cfg ultralytics/cfg/models/master/v0/det/yolo-master-esmoe-n-visdrone.yaml \
        --data VisDrone.yaml --device 6 --batch 32 --imgsz 1280 \
        --baseline-epochs 100 --schedule-epochs 100
"""

from __future__ import annotations

import argparse
import csv
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
from ultralytics.nn.modules.moe.analysis import ExpertUsageTracker  # noqa: E402
from ultralytics.nn.modules.moe.pruning import MoEPruner  # noqa: E402
from ultralytics.nn.modules.moe.scheduler import compute_gini  # noqa: E402


DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30)
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
        "args": {"moe_map_saturation_enabled": False, "moe_balance_loss": 1.0},
    },
    "dynamic": {
        "name": "dynamic_map_saturation",
        "args": {
            "moe_map_saturation_enabled": True,
            "moe_balance_loss": 1.0,
            "moe_map_saturation_window_size": 5,
            "moe_map_saturation_threshold": 0.001,
            "moe_map_saturation_decay_factor": 0.8,
            "moe_map_saturation_min_scale": 0.1,
        },
    },
    "ablation": {
        "name": "ablation_low_coeff",
        "args": {"moe_map_saturation_enabled": False, "moe_balance_loss": 0.3},
    },
}


def _device_tensor(value: str) -> torch.device:
    if value in {"", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda:0")
    if value.isdigit() and torch.cuda.is_available():
        return torch.device(f"cuda:{value}")
    if value == "mps":
        return torch.device("mps")
    return torch.device("cpu")


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
        name: len(module.experts)
        for name, module in model.named_modules()
        if hasattr(module, "experts") and hasattr(module, "routing")
    }


def _tracker_gini(tracker: ExpertUsageTracker, model: torch.nn.Module) -> dict[str, float]:
    modules = dict(model.named_modules())
    result = {}
    for router_name, stats in tracker.usage_stats.items():
        num_experts = int(getattr(modules.get(router_name), "num_experts", len(stats)))
        hits = [float(stats[i].hits) if i in stats else 0.0 for i in range(num_experts)]
        result[router_name] = compute_gini(torch.tensor(hits, dtype=torch.float32))
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
    device = _device_tensor(device_arg)
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


def _plot_pruning(results_csv: Path, out_dir: Path) -> None:
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
        sweet = max(
            front,
            key=lambda r: (f("map50_95", r) or 0.0) / max(f("latency_ms", r) or 1.0, 1e-6),
        )
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
        print(f"[plot] sweet spot: threshold={sweet['threshold']} recovery={sweet['recovery']}")


def _summarize_schedule(project: Path) -> Path:
    key_map = {v["name"]: k for k, v in SCHEDULE_VARIANTS.items()}
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
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    return out


def _train_baseline(args: argparse.Namespace) -> Path:
    project = args.output / "baseline"
    project.mkdir(parents=True, exist_ok=True)
    last = project / "train" / "weights" / "last.pt"
    if last.exists() and args.skip_existing:
        print(f"[baseline] reuse {last}")
        return last
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
        moe_map_saturation_enabled=False,
        moe_balance_loss=1.0,
    )
    return project / "train" / "weights" / "best.pt"


def _run_pruning_sweep(args: argparse.Namespace, baseline_ckpt: Path) -> None:
    sweep_dir = args.output / "pruning"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    thresholds = tuple(args.thresholds)

    calibration = MoEPruner(
        str(baseline_ckpt), thresholds[0], str(args.data), device=args.device, importance_mode=args.importance_mode
    )
    usage_stats = calibration.collect_usage()

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        tag = f"t{int(round(threshold * 100)):02d}"
        point_dir = sweep_dir / f"threshold_{tag}"
        point_dir.mkdir(parents=True, exist_ok=True)
        pruned_path = point_dir / f"pruned_{tag}.pt"

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
        _write_csv(sweep_dir / "results.csv", rows)
        print(f"[pruning t={threshold}] direct map50-95={direct['map50_95']:.4f}")

        recovered = YOLO(str(pruned_path))
        recovered.train(
            data=str(args.data),
            epochs=args.lora_epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            project=str(point_dir / "lora_recovery"),
            name="lora10",
            exist_ok=True,
            val=True,
            plots=False,
            amp=args.amp,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_include_moe=True,
            lora_include_head=False,
            lora_freeze_bn=True,
            lora_save_adapters=False,
        )
        recovered_path = point_dir / "lora_recovery" / "lora10" / "weights" / "best.pt"
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
        _write_csv(sweep_dir / "results.csv", rows)
        print(f"[pruning t={threshold}] lora10 map50-95={lora['map50_95']:.4f}")

    _plot_pruning(sweep_dir / "results.csv", sweep_dir / "plots")


def _run_schedule_ablation(args: argparse.Namespace) -> None:
    project = args.output / "schedule"
    project.mkdir(parents=True, exist_ok=True)
    for key, variant in SCHEDULE_VARIANTS.items():
        run_dir = project / variant["name"]
        if args.skip_existing and (run_dir / "results.csv").exists():
            print(f"[schedule] skip existing {variant['name']}")
            continue
        print(f"[schedule] training {key}: {variant['args']}")
        model = YOLO(str(args.model_cfg))
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
            pretrained=False,
            val=True,
            plots=False,
            amp=args.amp,
            **variant["args"],
        )
    summary = _summarize_schedule(project)
    print(f"[schedule] summary -> {summary}")


def run(args: argparse.Namespace) -> int:
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"[issue52] model-cfg={args.model_cfg}")
    print(f"[issue52] data={args.data}")
    print(f"[issue52] device={args.device} batch={args.batch} imgsz={args.imgsz}")
    print(f"[issue52] output={args.output}")

    baseline_ckpt = _train_baseline(args)
    print(f"[issue52] baseline checkpoint -> {baseline_ckpt}")

    _run_pruning_sweep(args, baseline_ckpt)
    _run_schedule_ablation(args)

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
    parser.add_argument("--device", default="6")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--baseline-epochs", type=int, default=100)
    parser.add_argument("--schedule-epochs", type=int, default=100)
    parser.add_argument("--lora-epochs", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--importance-mode", choices=("usage", "usage_weight"), default="usage_weight")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/issue52_full_visdrone")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
