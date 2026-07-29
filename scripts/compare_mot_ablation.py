#!/usr/bin/env python3
"""Run reproducible YOLO-Master MoT, MoA, and hybrid ablations.

Examples:
    python3 scripts/compare_mot_ablation.py --check-build
    python3 scripts/compare_mot_ablation.py --benchmark --imgsz 256 --reps 5 --device cpu
    python3 scripts/compare_mot_ablation.py --train --epochs 50 --imgsz 640 --batch 8 --device 0 --models v10 v10_mot v10_moa
    python3 scripts/compare_mot_ablation.py --summary-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from ultralytics import YOLO
from ultralytics.nn.modules.moa import C2fMoA, MoABlock, anneal_moa_temperature
from ultralytics.nn.modules.mot import C2fMoT, MoTBlock, anneal_mot_temperature
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    cfg: Path


SPECS = {
    "v10": ModelSpec(
        key="v10",
        label="YOLO-Master-v0.10-EsMoE-N",
        cfg=ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-n.yaml",
    ),
    "v10_mot": ModelSpec(
        key="v10_mot",
        label="YOLO-Master-v0.10-MoT-N",
        cfg=ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-n.yaml",
    ),
    "v10_mot_p5": ModelSpec(
        key="v10_mot_p5",
        label="YOLO-Master-v0.10-MoT-P5-N",
        cfg=ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-p5-n.yaml",
    ),
    "v10_moa": ModelSpec(
        key="v10_moa",
        label="YOLO-Master-v0.10-MoA-N",
        cfg=ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-n.yaml",
    ),
    "v10_moa_mot": ModelSpec(
        key="v10_moa_mot",
        label="YOLO-Master-v0.10-MoA+MoT-N",
        cfg=ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-mot-n.yaml",
    ),
    "v08": ModelSpec(
        key="v08",
        label="YOLO-Master v0.8 baseline",
        cfg=ROOT / "ultralytics/cfg/models/master/v0_8/det/yolo-master-n.yaml",
    ),
    "v08_moa": ModelSpec(
        key="v08_moa",
        label="YOLO-Master v0.8 MoA",
        cfg=ROOT / "ultralytics/cfg/models/master/v0_8/det/yolo-master-moa-n.yaml",
    ),
    "v08_mot": ModelSpec(
        key="v08_mot",
        label="YOLO-Master v0.8 MoT",
        cfg=ROOT / "ultralytics/cfg/models/master/v0_8/det/yolo-master-mot-n.yaml",
    ),
    "v08_moa_mot": ModelSpec(
        key="v08_moa_mot",
        label="YOLO-Master v0.8 MoA+MoT",
        cfg=ROOT / "ultralytics/cfg/models/master/v0_8/det/yolo-master-moa-mot-n.yaml",
    ),
}

METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "train/moe_loss",
    "train/moa_loss",
    "train/mot_loss",
)

LOSS_KEYS = (
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "train/moe_loss",
    "train/moa_loss",
    "train/mot_loss",
)


def default_data_yaml() -> Path:
    local = ROOT / "datasets/coco128/dataset.yaml"
    has_local_images = any(
        (ROOT / rel).exists()
        for rel in (
            "datasets/coco128/images/train",
            "datasets/coco128/images/val",
            "datasets/coco128/images/train2017",
        )
    )
    if local.exists() and has_local_images:
        return local
    return ROOT / "ultralytics/cfg/datasets/coco128.yaml"


def select_specs(keys: list[str]) -> list[ModelSpec]:
    specs = []
    for key in keys:
        if key not in SPECS:
            raise SystemExit(f"unknown model key: {key}. Choices: {', '.join(SPECS)}")
        spec = SPECS[key]
        if not spec.cfg.exists():
            raise SystemExit(f"missing config for {key}: {spec.cfg}")
        specs.append(spec)
    return specs


def count_modules(model: torch.nn.Module, cls: type[torch.nn.Module]) -> int:
    return sum(1 for m in model.modules() if isinstance(m, cls))


def normalize_torch_device(device: str) -> str:
    if not device:
        return "cpu"
    if device.isdigit():
        return f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    return device


def parse_float(value: object) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed


def finite_float(value: object) -> float | None:
    parsed = parse_float(value)
    if parsed is None or not math.isfinite(parsed):
        return None
    return parsed


def percentile(values: list[float], q: float) -> float:
    """Return a simple linear-interpolated percentile for latency samples."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[int(rank)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def deterministic_benchmark_input(model: torch.nn.Module, imgsz: int, seed: int) -> torch.Tensor:
    """Create one reproducible synthetic input without changing global RNG state."""
    parameter = next(model.parameters())
    target_device = parameter.device
    generator_device = target_device if target_device.type in {"cpu", "cuda"} else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    inputs = torch.randn(
        (1, 3, imgsz, imgsz),
        device=generator_device,
        dtype=parameter.dtype,
        generator=generator,
    )
    return inputs if generator_device == target_device else inputs.to(target_device)


def profile_flops(
    model: torch.nn.Module,
    imgsz: int,
    actual: bool = False,
    input_seed: int = 0,
) -> tuple[float, str]:
    """Return GFLOPs and method; actual=True uses torch profiler on full input size."""
    if not actual:
        return float(get_flops(model, imgsz=imgsz)), "thop_stride_scaled"

    try:
        model = model.eval()
        x = deterministic_benchmark_input(model, imgsz=imgsz, seed=input_seed)
        with torch.no_grad(), torch.profiler.profile(with_flops=True) as prof:
            _ = model(x)
        return sum(evt.flops for evt in prof.key_averages()) / 1e9, "torch_profile_actual"
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return float(get_flops(model, imgsz=imgsz)), "thop_stride_scaled_fallback"


def build_model(spec: ModelSpec, device: str = "cpu") -> DetectionModel:
    model = DetectionModel(str(spec.cfg), ch=3, nc=80, verbose=False).eval()
    if device:
        model.to(torch.device(normalize_torch_device(device)))
    return model


def model_structure_row(spec: ModelSpec, model: torch.nn.Module) -> dict[str, str]:
    """Describe the exact model object used by training or benchmarking."""
    params = sum(p.numel() for p in model.parameters())
    return {
        "key": spec.key,
        "label": spec.label,
        "cfg": str(spec.cfg.relative_to(ROOT)),
        "nc": str(getattr(model, "nc", "")),
        "params": str(params),
        "params_m": f"{params / 1e6:.6f}",
        "moablocks": str(count_modules(model, MoABlock)),
        "c2fmoa": str(count_modules(model, C2fMoA)),
        "motblocks": str(count_modules(model, MoTBlock)),
        "c2fmot": str(count_modules(model, C2fMoT)),
    }


def build_row(spec: ModelSpec, device: str = "cpu", imgsz: int = 640, include_flops: bool = False) -> dict[str, str]:
    model = build_model(spec, device=device)
    row = model_structure_row(spec, model)
    if include_flops:
        flops, method = profile_flops(model, imgsz=imgsz, actual=False)
        row.update({"imgsz": str(imgsz), "flops_g": f"{flops:.6f}", "flops_method": method})
    return row


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_model(path: Path, device: str) -> torch.nn.Module:
    model = YOLO(str(path)).model
    return model.to(torch.device(normalize_torch_device(device))).eval()


def sync_device(device: str) -> None:
    device = normalize_torch_device(device)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def benchmark_row(
    spec: ModelSpec,
    device: str,
    imgsz: int,
    warmup: int,
    reps: int,
    actual_flops: bool = False,
    weights: Path | None = None,
    input_seed: int = 0,
) -> dict[str, str]:
    model = load_checkpoint_model(weights, device) if weights else build_model(spec, device=device)
    device_name = normalize_torch_device(device)
    x = deterministic_benchmark_input(model, imgsz=imgsz, seed=input_seed)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
            sync_device(device)

        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            _ = model(x)
            sync_device(device)
            times.append((time.perf_counter() - t0) * 1000.0)

    flops, flops_method = profile_flops(model, imgsz=imgsz, actual=actual_flops, input_seed=input_seed)
    base = model_structure_row(spec, model)
    base.update(
        {
            "device": device_name,
            "imgsz": str(imgsz),
            "latency_ms_mean": f"{sum(times) / len(times):.3f}",
            "latency_ms_p50": f"{percentile(times, 0.50):.3f}",
            "latency_ms_p95": f"{percentile(times, 0.95):.3f}",
            "latency_ms_p99": f"{percentile(times, 0.99):.3f}",
            "latency_ms_min": f"{min(times):.3f}",
            "latency_ms_max": f"{max(times):.3f}",
            "flops_g": f"{flops:.6f}",
            "flops_method": flops_method,
            "reps": str(reps),
            "input_seed": str(input_seed),
            "input_distribution": "standard_normal",
            "weights": str(weights) if weights else "",
            "weights_sha256": sha256_file(weights) if weights else "",
        }
    )
    return base


def add_mixture_callbacks(model: YOLO, spec: ModelSpec, args: argparse.Namespace) -> None:
    if "moa" in spec.key:

        def on_moa_epoch_end(trainer):
            anneal_moa_temperature(trainer.model, factor=args.moa_temp_factor, min_temp=args.moa_min_temp)

        model.add_callback("on_train_epoch_end", on_moa_epoch_end)

    if "mot" in spec.key:

        def on_mot_epoch_end(trainer):
            anneal_mot_temperature(trainer.model, factor=args.mot_temp_factor, min_temp=args.mot_min_temp)

        model.add_callback("on_train_epoch_end", on_mot_epoch_end)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]


def read_last_metrics(results_csv: Path) -> dict[str, str]:
    rows = read_csv_rows(results_csv)
    return rows[-1] if rows else {}


def read_best_observed_metrics(results_csv: Path) -> dict[str, str]:
    """Return the CSV row with the highest observed mAP50-95."""
    rows = read_csv_rows(results_csv)
    metric = "metrics/mAP50-95(B)"
    finite_rows = []
    for row in rows:
        try:
            value = float(row.get(metric, "nan"))
        except ValueError:
            continue
        if math.isfinite(value):
            finite_rows.append((value, row))
    return max(finite_rows, key=lambda item: item[0])[1] if finite_rows else {}


def row_total_loss(row: dict[str, str]) -> float | None:
    values = [finite_float(row.get(key)) for key in LOSS_KEYS]
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values)


def read_recovery_events(path: Path) -> list[dict[str, object]]:
    """Read structured non-finite recovery events, ignoring malformed lines."""
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def stability_from_results(results_csv: Path, recovery_events_jsonl: Path | None = None) -> dict[str, str]:
    events = read_recovery_events(recovery_events_jsonl) if recovery_events_jsonl else []
    recovery = {
        "recovery_events": str(len(events)),
        "nonfinite_recovery_detected": str(bool(events)),
        "recovery_reasons": "|".join(sorted({str(event.get("reason", "")) for event in events if event.get("reason")})),
        "amp_fallback_triggered": str(any(bool(event.get("amp_fallback_triggered")) for event in events)),
    }
    rows = read_csv_rows(results_csv)
    if not rows:
        return recovery | {
            "nan_detected": "",
            "loss_diverged": "",
            "final_train_total_loss": "",
            "best_train_total_loss": "",
        }

    nan_detected = False
    train_losses = []
    for row in rows:
        for value in row.values():
            parsed = parse_float(value)
            if parsed is not None and not math.isfinite(parsed):
                nan_detected = True
        total = row_total_loss(row)
        if total is None:
            continue
        train_losses.append(total)
        if not math.isfinite(total):
            nan_detected = True

    finite_losses = [v for v in train_losses if math.isfinite(v)]
    if not finite_losses:
        return recovery | {
            "nan_detected": str(nan_detected),
            "loss_diverged": str(nan_detected),
            "final_train_total_loss": "",
            "best_train_total_loss": "",
        }

    final_loss = finite_losses[-1]
    best_loss = min(finite_losses)
    tail = finite_losses[-5:] if len(finite_losses) >= 5 else finite_losses
    tail_mean = sum(tail) / len(tail)
    diverged = nan_detected or (best_loss > 0 and tail_mean > best_loss * 1.5 and final_loss > best_loss * 1.5)
    return recovery | {
        "nan_detected": str(nan_detected),
        "loss_diverged": str(diverged),
        "final_train_total_loss": f"{final_loss:.6f}",
        "best_train_total_loss": f"{best_loss:.6f}",
    }


def benchmark_rows_by_key(project: Path) -> dict[str, dict[str, str]]:
    rows_by_key: dict[str, dict[str, str]] = {}
    for path in sorted(project.glob("latency_*.csv")):
        for row in read_csv_rows(path):
            key = row.get("key", "")
            if key:
                rows_by_key[key] = row
    return rows_by_key


def train_spec(args: argparse.Namespace, spec: ModelSpec, data_yaml: Path, project: Path) -> None:
    resume_ckpt = project / spec.key / "weights" / "last.pt"
    resume = bool(args.resume and resume_ckpt.exists())
    model = YOLO(str(resume_ckpt if resume else spec.cfg))
    add_mixture_callbacks(model, spec, args)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=args.deterministic,
        project=str(project),
        name=spec.key,
        exist_ok=args.exist_ok,
        pretrained=False,
        val=True,
        plots=args.plots,
        cache=args.cache,
        patience=args.patience,
        amp=args.amp,
        resume=resume,
        verbose=args.verbose,
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(project: Path, specs: list[ModelSpec]) -> Path:
    rows = []
    benchmark_rows = benchmark_rows_by_key(project)
    for spec in specs:
        run_dir = project / spec.key
        metrics = read_last_metrics(run_dir / "results.csv")
        best_observed = read_best_observed_metrics(run_dir / "results.csv")
        checkpoint = run_dir / "weights/best.pt"
        if not checkpoint.exists():
            checkpoint = run_dir / "weights/last.pt"
        if checkpoint.exists():
            trained_model = load_checkpoint_model(checkpoint, "cpu")
            row = model_structure_row(spec, trained_model)
            row.update({"weights": str(checkpoint), "weights_sha256": sha256_file(checkpoint)})
        else:
            row = build_row(spec, device="cpu")
            row.update({"weights": "", "weights_sha256": ""})
        row.update(
            {
                "run_dir": str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
                "epoch": metrics.get("epoch", ""),
                "final_epoch": metrics.get("epoch", ""),
                "best_observed_epoch": best_observed.get("epoch", ""),
                "training_time_seconds": metrics.get("time", ""),
            }
        )
        for key, value in benchmark_rows.get(spec.key, {}).items():
            if key not in {"key", "label", "cfg", "params", "params_m", "moablocks", "c2fmoa", "motblocks", "c2fmot"}:
                row[key] = value
        for key in METRIC_KEYS:
            row[key] = metrics.get(key, "")
            row[f"final/{key}"] = metrics.get(key, "")
            row[f"best_observed/{key}"] = best_observed.get(key, "")
        row.update(stability_from_results(run_dir / "results.csv", run_dir / "recovery_events.jsonl"))
        rows.append(row)
    out = project / "summary.csv"
    write_csv(out, rows)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["v10", "v10_mot", "v10_moa"], choices=tuple(SPECS))
    parser.add_argument("--project", type=Path, default=ROOT / "runs/mot_ablation")
    parser.add_argument("--data", type=Path, default=default_data_yaml())
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--check-build", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--trained-weights",
        action="store_true",
        help="Benchmark PROJECT/<key>/weights/best.pt instead of an untrained YAML model.",
    )
    parser.add_argument(
        "--actual-flops", action="store_true", help="Use torch profiler on the full input size for FLOPs."
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--benchmark-seed", type=int, default=0)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resume", action="store_true", help="Resume each model from PROJECT/<key>/weights/last.pt when present."
    )
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip per-model summary writes. Useful when independent model runs share a project directory.",
    )
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--moa-temp-factor", type=float, default=0.97)
    parser.add_argument("--moa-min-temp", type=float, default=0.3)
    parser.add_argument("--mot-temp-factor", type=float, default=0.97)
    parser.add_argument("--mot-min-temp", type=float, default=0.3)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.imgsz <= 0 or args.warmup < 0 or args.reps <= 0:
        raise SystemExit("--imgsz and --reps must be positive; --warmup must be non-negative")
    specs = select_specs(args.models)
    project = args.project if args.project.is_absolute() else ROOT / args.project
    data_yaml = args.data if args.data.is_absolute() else ROOT / args.data

    if args.check_build:
        rows = [build_row(spec, device=args.device, imgsz=args.imgsz, include_flops=True) for spec in specs]
        out = project / "build_summary.csv"
        write_csv(out, rows)
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"[build] wrote {out}")

    if args.benchmark:
        weight_paths = {}
        if args.trained_weights:
            for spec in specs:
                path = project / spec.key / "weights/best.pt"
                if not path.exists():
                    raise SystemExit(f"trained checkpoint not found for {spec.key}: {path}")
                weight_paths[spec.key] = path
        rows = [
            benchmark_row(
                spec,
                args.device,
                args.imgsz,
                args.warmup,
                args.reps,
                args.actual_flops,
                weights=weight_paths.get(spec.key),
                input_seed=args.benchmark_seed,
            )
            for spec in specs
        ]
        out = project / f"latency_{args.device}_{args.imgsz}.csv"
        write_csv(out, rows)
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"[benchmark] wrote {out}")

    if args.train:
        project.mkdir(parents=True, exist_ok=True)
        for spec in specs:
            train_spec(args, spec, data_yaml, project)
            if not args.no_summary:
                out = write_summary(project, specs)
                print(f"[summary] wrote {out}")

    if args.summary_only:
        out = write_summary(project, specs)
        print(f"[summary] wrote {out}")

    if not any((args.check_build, args.benchmark, args.train, args.summary_only)):
        raise SystemExit("choose one or more actions: --check-build, --benchmark, --train, --summary-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
