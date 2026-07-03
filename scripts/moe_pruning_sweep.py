#!/usr/bin/env python3
"""Reproduce Rhino-Bird issue #52 MoE pruning and dynamic-schedule experiments.

The default setup uses the VisDrone EsMoE checkpoint produced by issue #49:

    python scripts/moe_pruning_sweep.py --dry-run
    python scripts/moe_pruning_sweep.py --thresholds 0.10 --direct-only --limit-val-batches smoke
    python scripts/moe_pruning_sweep.py --plot-only
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs/reproduce/_runtime/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs/reproduce/_runtime/matplotlib"))

import torch  # noqa: E402

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules.moe.analysis import ExpertUsageTracker  # noqa: E402
from ultralytics.nn.modules.moe.pruning import MoEPruner  # noqa: E402
from ultralytics.nn.modules.moe.schedule import usage_gini  # noqa: E402
from ultralytics.utils import SETTINGS, YAML  # noqa: E402


DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30)
METRIC_KEYS = ("metrics/mAP50-95(B)", "metrics/mAP50(B)")


@dataclass(frozen=True)
class EvalResult:
    map5095: float
    map50: float
    params_m: float
    gflops: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    gini_mean: float
    usage_stats: dict[str, dict[int, Any]]


def default_model_path() -> Path:
    candidates = [
        # Path produced by the merged issue #49 reproduction scripts when running
        # VisDrone EsMoE-N with --no-sparse-eval.
        ROOT / "runs/reproduce/visdrone/VisDrone_EsMoE-N/weights/best.pt",
        # Backward-compatible local path from the original issue #52 experiments.
        ROOT / "runs/reproduce/visdrone/visdrone_100e_denseeval_esmoe/weights/best.pt",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def default_data_yaml() -> Path:
    local = ROOT / "runs/reproduce/visdrone/_data/VisDrone.local.yaml"
    return local if local.exists() else ROOT / "ultralytics/cfg/datasets/VisDrone.yaml"


def ensure_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def threshold_tag(threshold: float) -> str:
    return f"thr{threshold:.2f}".replace(".", "p")


def configure_wandb(mode: str) -> None:
    """Configure W&B without requiring global user settings changes."""
    if mode == "disabled":
        SETTINGS.update({"wandb": False})
        os.environ.setdefault("WANDB_DISABLED", "true")
        return

    SETTINGS.update({"wandb": True})
    os.environ.pop("WANDB_DISABLED", None)
    os.environ["WANDB_MODE"] = "offline" if mode == "offline" else "online"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def metrics_to_dict(metrics: Any) -> dict[str, float]:
    results = getattr(metrics, "results_dict", None) or {}
    out = {}
    for key in METRIC_KEYS:
        value = results.get(key, 0.0)
        out[key] = float(value.item() if hasattr(value, "item") else value)
    return out


def model_params_m(model: YOLO) -> float:
    return sum(p.numel() for p in model.model.parameters()) / 1e6


def resolve_torch_device(device: str) -> torch.device:
    """Resolve an Ultralytics device string to a torch device for local probes."""
    value = str(device or "").strip()
    if value in {"", "-1"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value.isdigit():
        return torch.device(f"cuda:{value}")
    if "," in value:
        first = value.split(",", 1)[0].strip()
        return torch.device(f"cuda:{first}" if first.isdigit() else first)
    return torch.device(value)


def configure_sparse_inference(model: YOLO, enabled: bool) -> None:
    """Force ES-MoE sparse inference mode for reproducible eval semantics."""
    for module in model.model.modules():
        if hasattr(module, "enable_sparse_inference"):
            module.enable_sparse_inference(enabled)
        elif hasattr(module, "use_sparse_inference"):
            module.use_sparse_inference = enabled


def load_model_for_probe(model_path: Path, device: str, sparse_eval: bool) -> YOLO:
    """Load a fresh model instance for FLOPs/latency probes."""
    model = YOLO(str(model_path))
    configure_sparse_inference(model, enabled=sparse_eval)
    model.model.to(resolve_torch_device(device))
    model.model.eval()
    return model


@contextmanager
def capture_moe_input_shapes(model: torch.nn.Module):
    shapes: dict[str, tuple[int, ...]] = {}
    hooks = []

    def make_hook(name: str):
        def hook(module, inputs, _output):
            if inputs and isinstance(inputs[0], torch.Tensor) and hasattr(module, "get_gflops"):
                shapes[name] = tuple(inputs[0].shape)

        return hook

    for name, module in model.named_modules():
        if hasattr(module, "get_gflops"):
            hooks.append(module.register_forward_hook(make_hook(name)))
    try:
        yield shapes
    finally:
        for hook in hooks:
            hook.remove()


def estimate_moe_gflops(model: YOLO, imgsz: int, device: str) -> float:
    torch_device = next(model.model.parameters()).device
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=torch_device)
    model.model.eval()
    with torch.no_grad(), capture_moe_input_shapes(model.model) as shapes:
        model.model(dummy)

    total = 0.0
    modules = dict(model.model.named_modules())
    for name, shape in shapes.items():
        module = modules[name]
        try:
            value = module.get_gflops(shape)
        except Exception:
            continue
        if isinstance(value, dict):
            total += float(value.get("total_gflops", 0.0))
        else:
            total += float(value)
    return total


def benchmark_latency(model: YOLO, imgsz: int, warmup: int, reps: int) -> tuple[float, float, float]:
    torch_device = next(model.model.parameters()).device
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=torch_device)
    model.model.eval()

    def sync():
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)

    with torch.no_grad():
        for _ in range(max(warmup, 0)):
            model.model(dummy)
        sync()
        times = []
        for _ in range(max(reps, 1)):
            start = time.perf_counter()
            model.model(dummy)
            sync()
            times.append((time.perf_counter() - start) * 1000.0)

    return statistics.mean(times), statistics.median(times), sorted(times)[int(0.9 * (len(times) - 1))]


def usage_rows_from_stats(threshold: float, stage: str, usage_stats: dict[str, dict[int, Any]]) -> list[dict[str, Any]]:
    rows = []
    for layer_name, stats in sorted(usage_stats.items()):
        total_hits = sum(float(s.hits) for s in stats.values())
        usages = []
        for expert_id, expert_stats in sorted(stats.items()):
            usage = float(expert_stats.hits) / total_hits if total_hits > 0 else 0.0
            usages.append(usage)
            rows.append(
                {
                    "threshold": threshold,
                    "stage": stage,
                    "layer_name": layer_name,
                    "expert_id": expert_id,
                    "usage": usage,
                    "hits": float(expert_stats.hits),
                    "avg_weight": float(expert_stats.avg_weight),
                    "layer_gini": usage_gini(usages),
                }
            )
        layer_gini = usage_gini(usages)
        for row in rows:
            if row["stage"] == stage and row["layer_name"] == layer_name:
                row["layer_gini"] = layer_gini
    return rows


def collect_usage_stats(model_path: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, dict[int, Any]]:
    model = YOLO(str(model_path))
    configure_sparse_inference(model, enabled=args.sparse_eval)
    with ExpertUsageTracker(model.model) as tracker:
        model.val(data=str(data_yaml), imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers, plots=False)
        return tracker.usage_stats


def per_layer_expert_rows(threshold: float, stage: str, model_path: Path) -> list[dict[str, Any]]:
    model = YOLO(str(model_path))
    rows = []
    for name, module in model.model.named_modules():
        if hasattr(module, "num_experts") and int(getattr(module, "num_experts", 0)) > 0:
            rows.append(
                {
                    "threshold": threshold,
                    "stage": stage,
                    "layer_name": name,
                    "module_type": type(module).__name__,
                    "num_experts": int(getattr(module, "num_experts", 0)),
                    "top_k": int(getattr(module, "top_k", 0)),
                }
            )
    return rows


def evaluate_model(model_path: Path, data_yaml: Path, args: argparse.Namespace) -> EvalResult:
    val_model = YOLO(str(model_path))
    configure_sparse_inference(val_model, enabled=args.sparse_eval)
    metrics = val_model.val(data=str(data_yaml), imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers, plots=False)
    metric_dict = metrics_to_dict(metrics)
    del val_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    probe_model = load_model_for_probe(model_path, args.device, args.sparse_eval)
    params_m = model_params_m(probe_model)
    gflops = estimate_moe_gflops(probe_model, args.imgsz, args.device)
    latency = benchmark_latency(probe_model, args.imgsz, args.latency_warmup, args.latency_reps)
    del probe_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    usage_stats = collect_usage_stats(model_path, data_yaml, args)
    layer_ginis = []
    for stats in usage_stats.values():
        total_hits = sum(float(s.hits) for s in stats.values())
        usage = [float(s.hits) / total_hits if total_hits > 0 else 0.0 for _, s in sorted(stats.items())]
        layer_ginis.append(usage_gini(usage))
    return EvalResult(
        map5095=metric_dict["metrics/mAP50-95(B)"],
        map50=metric_dict["metrics/mAP50(B)"],
        params_m=params_m,
        gflops=gflops,
        latency_mean_ms=latency[0],
        latency_p50_ms=latency[1],
        latency_p90_ms=latency[2],
        gini_mean=sum(layer_ginis) / len(layer_ginis) if layer_ginis else 0.0,
        usage_stats=usage_stats,
    )


def make_smoke_data_yaml(data_yaml: Path, project: Path, limit_images: int = 16) -> Path:
    data = YAML.load(data_yaml)
    root = Path(data.get("path", ""))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve() if (data_yaml.parent / root).exists() else (ROOT / "datasets" / root).resolve()
    val_dir = root / data.get("val", "images/val")
    images = sorted([*val_dir.glob("*.jpg"), *val_dir.glob("*.png")])[:limit_images]
    if not images:
        raise FileNotFoundError(f"No validation images found for smoke YAML: {val_dir}")

    smoke_dir = project / "_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    image_list = smoke_dir / "val_images.txt"
    image_list.write_text("\n".join(str(p) for p in images) + "\n", encoding="utf-8")
    smoke_yaml = smoke_dir / "VisDrone.smoke.yaml"
    smoke_data = dict(data)
    smoke_data["path"] = str(root)
    smoke_data["train"] = str(image_list)
    smoke_data["val"] = str(image_list)
    YAML.save(smoke_yaml, smoke_data)
    return smoke_yaml


def run_lora_recovery(pruned_path: Path, data_yaml: Path, threshold: float, args: argparse.Namespace) -> Path:
    run_name = f"{threshold_tag(threshold)}_lora10"
    model = YOLO(str(pruned_path))
    configure_sparse_inference(model, enabled=args.sparse_eval)
    model.train(
        data=str(data_yaml),
        epochs=args.lora_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=run_name,
        exist_ok=args.exist_ok,
        plots=args.plots,
        patience=0,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_include_moe=True,
        lora_save_adapters=True,
        pretrained=False,
        val=True,
    )
    return args.project / run_name / "weights" / "best.pt"


def choose_sweet_spot(rows: list[dict[str, Any]], baseline_map: float) -> float | None:
    direct_rows = [r for r in rows if r["stage"] == "direct" and r["status"] == "ok"]
    if not direct_rows:
        return None
    candidates = [r for r in direct_rows if float(r["mAP50-95"]) >= baseline_map * 0.99]
    if candidates:
        return float(min(candidates, key=lambda r: float(r["latency_mean_ms"]))["threshold"])

    maps = [float(r["mAP50-95"]) for r in direct_rows]
    lats = [float(r["latency_mean_ms"]) for r in direct_rows]
    map_min, map_max = min(maps), max(maps)
    lat_min, lat_max = min(lats), max(lats)

    def score(row: dict[str, Any]) -> float:
        map_norm = 0.0 if map_max == map_min else (float(row["mAP50-95"]) - map_min) / (map_max - map_min)
        lat_norm = 0.0 if lat_max == lat_min else (float(row["latency_mean_ms"]) - lat_min) / (lat_max - lat_min)
        return map_norm - lat_norm

    return float(max(direct_rows, key=score)["threshold"])


def plot_results(project: Path, summary_rows: list[dict[str, Any]], baseline_map: float) -> None:
    if not summary_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = project / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    direct = [r for r in summary_rows if r["stage"] == "direct" and r["status"] == "ok"]
    if not direct:
        return

    xs = [float(r["threshold"]) for r in direct]
    maps = [float(r["mAP50-95"]) for r in direct]
    flops = [float(r["gflops"]) for r in direct]
    lats = [float(r["latency_mean_ms"]) for r in direct]
    sweet = choose_sweet_spot(summary_rows, baseline_map)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, maps, lats, marker="o", label="direct")
    ax.set_xlabel("threshold")
    ax.set_ylabel("mAP50-95")
    ax.set_zlabel("latency mean (ms)")
    ax.set_title("Threshold vs mAP vs Latency")
    if sweet is not None:
        idx = xs.index(sweet)
        ax.scatter([xs[idx]], [maps[idx]], [lats[idx]], s=90, label="Sweet Spot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "threshold_map_latency_3d.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(xs, maps, marker="o")
    axes[0].axhline(baseline_map * 0.99, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("mAP50-95")
    axes[1].plot(xs, flops, marker="o", color="tab:orange")
    axes[1].set_ylabel("MoE GFLOPs")
    axes[2].plot(xs, lats, marker="o", color="tab:green")
    axes[2].set_ylabel("Latency mean (ms)")
    for ax in axes:
        ax.set_xlabel("threshold")
        if sweet is not None:
            ax.axvline(sweet, color="red", linestyle=":", linewidth=1)
    fig.tight_layout()
    fig.savefig(plot_dir / "threshold_curves.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(lats, maps)
    for x, y, threshold in zip(lats, maps, xs):
        ax.annotate(f"{threshold:.2f}", (x, y), textcoords="offset points", xytext=(4, 4))
    if sweet is not None:
        idx = xs.index(sweet)
        ax.scatter([lats[idx]], [maps[idx]], color="red", s=90, label="Sweet Spot")
    ax.set_xlabel("Latency mean (ms)")
    ax.set_ylabel("mAP50-95")
    ax.set_title("Pareto: Accuracy vs Latency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "pareto_accuracy_latency.png", dpi=200)
    plt.close(fig)


def write_readme(project: Path, rows: list[dict[str, Any]], baseline_map: float) -> None:
    sweet = choose_sweet_spot(rows, baseline_map)
    lines = [
        "# Issue #52 MoE Pruning Reproduction",
        "",
        f"- Baseline mAP50-95: `{baseline_map:.5f}`",
        f"- Sweet Spot: `{sweet:.2f}`" if sweet is not None else "- Sweet Spot: pending",
        "",
        "## Outputs",
        "",
        "- `summary.csv`: metrics per threshold and stage.",
        "- `per_layer_experts.csv`: retained experts and top-k per MoE layer.",
        "- `expert_usage_gini.csv`: expert usage and per-layer Gini.",
        "- `latency.csv`: latency and size metrics.",
        "- `plots/`: threshold curves and Pareto figure.",
        "",
    ]
    (project / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=default_model_path())
    parser.add_argument("--data", type=Path, default=default_data_yaml())
    parser.add_argument("--project", type=Path, default=ROOT / "runs/reproduce/issue52_moe_pruning")
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--latency-reps", type=int, default=100)
    parser.add_argument("--lora-epochs", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--baseline-map", type=float, help="Baseline mAP50-95; inferred from baseline model if omitted.")
    parser.add_argument("--limit-val-batches", choices=("none", "smoke"), default="none")
    parser.add_argument("--direct-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--wandb", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--sparse-eval", action="store_true", help="Use sparse ES-MoE eval. Default is dense eval for accuracy reporting.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.model_path = args.model_path if args.model_path.is_absolute() else ROOT / args.model_path
    args.data = args.data if args.data.is_absolute() else ROOT / args.data
    args.project = args.project if args.project.is_absolute() else ROOT / args.project
    configure_wandb(args.wandb)

    data_yaml = args.data
    if args.limit_val_batches == "smoke":
        data_yaml = make_smoke_data_yaml(args.data, args.project)

    print(f"[issue52] model={ensure_relative(args.model_path)}")
    print(f"[issue52] data={ensure_relative(data_yaml)}")
    print(f"[issue52] project={ensure_relative(args.project)}")
    print(f"[issue52] thresholds={', '.join(f'{x:.2f}' for x in args.thresholds)}")

    if args.dry_run:
        for threshold in args.thresholds:
            print(f"  - {threshold:.2f}: {ensure_relative(args.project / threshold_tag(threshold) / 'pruned.pt')}")
        return 0

    summary_path = args.project / "summary.csv"
    summary_fields = [
        "threshold", "stage", "status", "checkpoint", "mAP50-95", "mAP50", "gflops", "latency_mean_ms",
        "latency_p50_ms", "latency_p90_ms", "params_m", "gini_mean", "error",
    ]

    if args.plot_only:
        rows = list(csv.DictReader(summary_path.open())) if summary_path.exists() else []
        baseline_map = args.baseline_map if args.baseline_map is not None else max(float(r["mAP50-95"]) for r in rows)
        plot_results(args.project, rows, baseline_map)
        write_readme(args.project, rows, baseline_map)
        return 0

    args.project.mkdir(parents=True, exist_ok=True)
    baseline_map = args.baseline_map
    if baseline_map is None:
        print("[baseline] evaluating baseline checkpoint")
        baseline_map = evaluate_model(args.model_path, data_yaml, args).map5095

    all_summary_rows: list[dict[str, Any]] = []
    expert_rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []

    for threshold in args.thresholds:
        tag = threshold_tag(threshold)
        run_dir = args.project / tag
        pruned_path = run_dir / "pruned.pt"
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            if not (args.skip_existing and pruned_path.exists()):
                pruner = MoEPruner(str(args.model_path), threshold=threshold, dataset=str(data_yaml), device=args.device)
                ok = pruner.prune(str(pruned_path))
                if not ok:
                    raise RuntimeError("MoEPruner returned failure")
                usage_rows.extend(usage_rows_from_stats(threshold, "pre_prune", pruner.usage_stats))

            direct = evaluate_model(pruned_path, data_yaml, args)
            row = {
                "threshold": threshold,
                "stage": "direct",
                "status": "ok",
                "checkpoint": ensure_relative(pruned_path),
                "mAP50-95": direct.map5095,
                "mAP50": direct.map50,
                "gflops": direct.gflops,
                "latency_mean_ms": direct.latency_mean_ms,
                "latency_p50_ms": direct.latency_p50_ms,
                "latency_p90_ms": direct.latency_p90_ms,
                "params_m": direct.params_m,
                "gini_mean": direct.gini_mean,
                "error": "",
            }
            append_csv(summary_path, row, summary_fields)
            all_summary_rows.append(row)
            expert_rows.extend(per_layer_expert_rows(threshold, "direct", pruned_path))
            usage_rows.extend(usage_rows_from_stats(threshold, "direct", direct.usage_stats))

            if args.direct_only:
                continue

            lora_best = run_lora_recovery(pruned_path, data_yaml, threshold, args)
            lora_eval = evaluate_model(lora_best, data_yaml, args)
            row = {
                "threshold": threshold,
                "stage": "lora10",
                "status": "ok",
                "checkpoint": ensure_relative(lora_best),
                "mAP50-95": lora_eval.map5095,
                "mAP50": lora_eval.map50,
                "gflops": lora_eval.gflops,
                "latency_mean_ms": lora_eval.latency_mean_ms,
                "latency_p50_ms": lora_eval.latency_p50_ms,
                "latency_p90_ms": lora_eval.latency_p90_ms,
                "params_m": lora_eval.params_m,
                "gini_mean": lora_eval.gini_mean,
                "error": "",
            }
            append_csv(summary_path, row, summary_fields)
            all_summary_rows.append(row)
            expert_rows.extend(per_layer_expert_rows(threshold, "lora10", lora_best))
            usage_rows.extend(usage_rows_from_stats(threshold, "lora10", lora_eval.usage_stats))

        except Exception as exc:
            row = {
                "threshold": threshold,
                "stage": "failed",
                "status": "failed",
                "checkpoint": "",
                "mAP50-95": "",
                "mAP50": "",
                "gflops": "",
                "latency_mean_ms": "",
                "latency_p50_ms": "",
                "latency_p90_ms": "",
                "params_m": "",
                "gini_mean": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
            append_csv(summary_path, row, summary_fields)
            all_summary_rows.append(row)
            print(f"[fail] threshold={threshold:.2f}: {row['error']}")

    write_csv(
        args.project / "per_layer_experts.csv",
        expert_rows,
        ["threshold", "stage", "layer_name", "module_type", "num_experts", "top_k"],
    )
    write_csv(
        args.project / "expert_usage_gini.csv",
        usage_rows,
        ["threshold", "stage", "layer_name", "expert_id", "usage", "hits", "avg_weight", "layer_gini"],
    )
    latency_rows = [
        {
            "threshold": r["threshold"],
            "stage": r["stage"],
            "latency_mean_ms": r["latency_mean_ms"],
            "latency_p50_ms": r["latency_p50_ms"],
            "latency_p90_ms": r["latency_p90_ms"],
            "params_m": r["params_m"],
            "gflops": r["gflops"],
        }
        for r in all_summary_rows
        if r["status"] == "ok"
    ]
    write_csv(args.project / "latency.csv", latency_rows, list(latency_rows[0].keys()) if latency_rows else ["threshold"])
    plot_results(args.project, all_summary_rows, baseline_map)
    write_readme(args.project, all_summary_rows, baseline_map)
    return 0 if all(r["status"] == "ok" for r in all_summary_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
