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
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from ultralytics import YOLO  # noqa: E402
from ultralytics.engine.telemetry import (  # noqa: E402
    TELEMETRY_ENV,
    TELEMETRY_LOSS_STEPS_ENV,
    device_memory_sample,
    reset_device_memory_peak,
    sync_device,
)
from ultralytics.nn.modules.moa import C2fMoA, MoABlock  # noqa: E402
from ultralytics.nn.modules.mot import C2fMoT, MoTBlock  # noqa: E402
from ultralytics.utils import YAML  # noqa: E402
from ultralytics.utils.torch_utils import get_flops  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    cfg: Path
    task: str = "detect"
    variant: str = "standard"
    runtime_overrides: tuple[tuple[str, object], ...] = ()


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
    "mt_off": ModelSpec(
        key="mt_off",
        label="YOLO26-MT MPS three-task MoT off",
        cfg=ROOT / "scripts/yolo26-master-mt-off-mps-local.yaml",
        task="multitask",
        variant="mot_off",
    ),
    "mt_mot_dense": ModelSpec(
        key="mt_mot_dense",
        label="YOLO26-MT MPS three-task MoT dense",
        cfg=ROOT / "scripts/yolo26-master-mt-mot-dense-mps-local.yaml",
        task="multitask",
        variant="mot_dense",
    ),
    "mt_mot_sparse": ModelSpec(
        key="mt_mot_sparse",
        label="YOLO26-MT MPS three-task MoT sparse-train",
        cfg=ROOT / "scripts/yolo26-master-mt-mot-runtime-mps-local.yaml",
        task="multitask",
        variant="mot_sparse_train",
        runtime_overrides=(("mot_sparse_train", True), ("mot_local_attn_window", 7)),
    ),
}

METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/precision(M)",
    "metrics/recall(M)",
    "metrics/mAP50(M)",
    "metrics/mAP50-95(M)",
    "metrics/precision(P)",
    "metrics/recall(P)",
    "metrics/mAP50(P)",
    "metrics/mAP50-95(P)",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "val/seg_loss",
    "val/pose_loss",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "train/seg_loss",
    "train/pose_loss",
    "train/moe_loss",
    "train/moa_loss",
    "train/mot_loss",
)

LOSS_KEYS = (
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "train/seg_loss",
    "train/pose_loss",
    "train/moe_loss",
    "train/moa_loss",
    "train/mot_loss",
)

MULTITASK_ABLATION_TASKS = frozenset(("detect", "segment", "pose"))


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


def validate_training_specs(specs: list[ModelSpec], data_yaml: Path) -> None:
    """Reject mixed task families and non-comparable data contracts before training."""
    task_families = {spec.task for spec in specs}
    if len(task_families) != 1:
        raise SystemExit(
            "--train cannot mix task families. Run detection and three-task multi-task ablations in separate projects."
        )
    if task_families != {"multitask"}:
        return
    if not data_yaml.exists():
        raise SystemExit(f"missing multi-task data YAML: {data_yaml}")
    data = YAML.load(data_yaml)
    data_tasks = data.get("tasks") if isinstance(data, dict) else None
    if not isinstance(data_tasks, (list, tuple, set)) or set(data_tasks) != MULTITASK_ABLATION_TASKS:
        raise SystemExit(
            "three-task MoT ablation requires the data YAML to declare exactly tasks: [detect, segment, pose]."
        )
    if data.get("multitask_format") != "coco":
        raise SystemExit(
            "three-task MoT ablation requires multitask_format: coco for aligned mask and pose supervision."
        )


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


def profile_flops(model: torch.nn.Module, imgsz: int, actual: bool = False) -> tuple[float, str]:
    """Return GFLOPs and method; actual=True uses torch profiler on full input size."""
    if not actual:
        return float(get_flops(model, imgsz=imgsz)), "thop_stride_scaled"

    try:
        model = model.eval()
        param = next(model.parameters())
        x = torch.empty((1, 3, imgsz, imgsz), device=param.device)
        with torch.no_grad(), torch.profiler.profile(with_flops=True) as prof:
            _ = model(x)
        return sum(evt.flops for evt in prof.key_averages()) / 1e9, "torch_profile_actual"
    except Exception:
        return float(get_flops(model, imgsz=imgsz)), "thop_stride_scaled_fallback"


def build_model(spec: ModelSpec, device: str = "cpu") -> torch.nn.Module:
    """Build the exact task model declared by one ablation specification."""
    model = YOLO(str(spec.cfg), task=spec.task, verbose=False).model.eval()
    if device:
        model.to(torch.device(normalize_torch_device(device)))
    return model


def build_row(spec: ModelSpec, device: str = "cpu", imgsz: int = 640, include_flops: bool = False) -> dict[str, str]:
    model = build_model(spec, device=device)
    params = sum(p.numel() for p in model.parameters())
    row = {
        "key": spec.key,
        "label": spec.label,
        "cfg": str(spec.cfg.relative_to(ROOT)),
        "task": spec.task,
        "variant": spec.variant,
        "params": str(params),
        "params_m": f"{params / 1e6:.6f}",
        "moablocks": str(count_modules(model, MoABlock)),
        "c2fmoa": str(count_modules(model, C2fMoA)),
        "motblocks": str(count_modules(model, MoTBlock)),
        "c2fmot": str(count_modules(model, C2fMoT)),
    }
    if include_flops:
        flops, method = profile_flops(model, imgsz=imgsz, actual=False)
        row.update({"imgsz": str(imgsz), "flops_g": f"{flops:.6f}", "flops_method": method})
    return row


def benchmark_row(
    spec: ModelSpec,
    device: str,
    imgsz: int,
    warmup: int,
    reps: int,
    actual_flops: bool = False,
) -> dict[str, str]:
    torch.set_grad_enabled(False)
    model = build_model(spec, device=device)
    device_name = normalize_torch_device(device)
    x = torch.randn(1, 3, imgsz, imgsz, device=torch.device(device_name))
    reset_device_memory_peak(device_name)
    memory = device_memory_sample(device_name)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
            sync_device(device)
            sample = device_memory_sample(device_name)
            if sample["peak_device_memory_bytes"] is not None:
                memory["peak_device_memory_bytes"] = max(
                    int(memory.get("peak_device_memory_bytes") or 0), int(sample["peak_device_memory_bytes"])
                )
                total = sample.get("device_total_memory_bytes")
                if total:
                    memory["device_total_memory_bytes"] = int(total)
                    memory["peak_device_memory_fraction"] = memory["peak_device_memory_bytes"] / total
            if sample["sampled_current_memory_bytes"] is not None:
                memory["sampled_current_memory_bytes_max"] = max(
                    int(memory.get("sampled_current_memory_bytes_max") or 0),
                    int(sample["sampled_current_memory_bytes"]),
                )

        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            _ = model(x)
            sync_device(device)
            times.append((time.perf_counter() - t0) * 1000.0)
            sample = device_memory_sample(device_name)
            if sample["peak_device_memory_bytes"] is not None:
                memory["peak_device_memory_bytes"] = max(
                    int(memory.get("peak_device_memory_bytes") or 0), int(sample["peak_device_memory_bytes"])
                )
                total = sample.get("device_total_memory_bytes")
                if total:
                    memory["device_total_memory_bytes"] = int(total)
                    memory["peak_device_memory_fraction"] = memory["peak_device_memory_bytes"] / total
            if sample["sampled_current_memory_bytes"] is not None:
                memory["sampled_current_memory_bytes_max"] = max(
                    int(memory.get("sampled_current_memory_bytes_max") or 0),
                    int(sample["sampled_current_memory_bytes"]),
                )

    flops, flops_method = profile_flops(model, imgsz=imgsz, actual=actual_flops)
    base = build_row(spec, device=device)
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
            "memory_measurement": str(memory["measurement"]),
            "memory_is_true_peak": str(memory["is_true_peak"]),
            "peak_device_memory_bytes": str(memory["peak_device_memory_bytes"] or ""),
            "device_total_memory_bytes": str(memory.get("device_total_memory_bytes") or ""),
            "peak_device_memory_fraction": str(memory.get("peak_device_memory_fraction") or ""),
            "sampled_current_memory_bytes_max": str(memory.get("sampled_current_memory_bytes_max") or ""),
        }
    )
    return base


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]


def read_last_metrics(results_csv: Path) -> dict[str, str]:
    rows = read_csv_rows(results_csv)
    return rows[-1] if rows else {}


def row_total_loss(row: dict[str, str]) -> float | None:
    values = [finite_float(row.get(key)) for key in LOSS_KEYS]
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values)


def stability_from_results(results_csv: Path) -> dict[str, str]:
    rows = read_csv_rows(results_csv)
    if not rows:
        return {
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
        return {
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
    return {
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


def read_json(path: Path) -> dict[str, object]:
    """Read a JSON object or return an empty mapping for absent/invalid optional artifacts."""
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _telemetry_value(value: object) -> str:
    """Format optional telemetry values for the string-oriented CSV writer."""
    return "" if value is None else str(value)


def telemetry_summary_fields(telemetry: dict[str, object]) -> dict[str, str]:
    """Flatten the rank-zero telemetry record into stable summary CSV fields."""
    aggregate = telemetry.get("aggregation", {})
    ranks = telemetry.get("ranks", [])
    if not isinstance(aggregate, dict) or not isinstance(ranks, list) or not ranks:
        return {}
    rank_zero = next(
        (record for record in ranks if isinstance(record, dict) and record.get("metadata", {}).get("rank") == 0),
        ranks[0],
    )
    if not isinstance(rank_zero, dict):
        return {}
    steps = rank_zero.get("steps", {})
    memory = rank_zero.get("memory", {})
    if not isinstance(steps, dict) or not isinstance(memory, dict):
        return {}
    return {
        "train_world_size": _telemetry_value(aggregate.get("world_size")),
        "train_rank_step_counts_consistent": _telemetry_value(aggregate.get("rank_step_counts_consistent")),
        "train_rank_loss_relative_spread_max": _telemetry_value(aggregate.get("rank_loss_relative_spread_max")),
        "train_global_samples_per_second": _telemetry_value(aggregate.get("global_samples_per_second")),
        "train_samples_per_second_rank0": _telemetry_value(steps.get("samples_per_second")),
        "train_step_ms_p50": _telemetry_value(steps.get("p50_milliseconds")),
        "train_step_ms_p95": _telemetry_value(steps.get("p95_milliseconds")),
        "train_memory_measurement": _telemetry_value(memory.get("measurement")),
        "train_memory_is_true_peak": _telemetry_value(memory.get("is_true_peak")),
        "train_peak_device_memory_bytes": _telemetry_value(memory.get("peak_device_memory_bytes")),
        "train_device_total_memory_bytes": _telemetry_value(memory.get("device_total_memory_bytes")),
        "train_peak_device_memory_fraction": _telemetry_value(memory.get("peak_device_memory_fraction")),
        "train_rank_peak_device_memory_fraction_max": _telemetry_value(
            aggregate.get("rank_peak_device_memory_fraction_max")
        ),
        "train_sampled_current_memory_bytes_max": _telemetry_value(memory.get("sampled_current_memory_bytes_max")),
    }


def routing_manifest(
    model: YOLO,
    runtime_overrides: tuple[tuple[str, object], ...] = (),
) -> list[dict[str, object]]:
    """Capture construction state and requested run-time routing policy separately."""
    overrides = dict(runtime_overrides)
    rows = []
    for name, module in model.model.named_modules():
        if not isinstance(module, (MoABlock, MoTBlock)):
            continue
        local_expert = module.experts[0] if isinstance(module, MoTBlock) and module.experts else None
        sparse_at_construction = bool(getattr(module, "sparse_train", False))
        local_window_at_construction = getattr(local_expert, "local_window_size", None)
        rows.append(
            {
                "name": name,
                "module_type": type(module).__name__,
                "top_k": getattr(module, "top_k", None),
                "num_experts": getattr(module, "num_experts", None),
                "sparse_train_at_construction": sparse_at_construction,
                "sparse_train_requested": bool(overrides.get("mot_sparse_train", sparse_at_construction)),
                "local_attn_window_at_construction": local_window_at_construction,
                "local_attn_window_requested": overrides.get("mot_local_attn_window", local_window_at_construction),
                "ddp_contract_source": getattr(module, "_ddp_contract_source", None),
            }
        )
    return rows


def write_run_manifest(args: argparse.Namespace, spec: ModelSpec, project: Path, seed: int, model: YOLO) -> Path:
    """Write the reproducibility contract for one model/seed training run."""
    run_dir = project / spec.key
    manifest = {
        "schema_version": 1,
        "model": {
            "key": spec.key,
            "label": spec.label,
            "cfg": str(spec.cfg),
            "task": spec.task,
            "variant": spec.variant,
        },
        "seed": seed,
        "data": str(args.data),
        "training": {
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "optimizer": args.optimizer,
            "mosaic": args.mosaic,
            "mixup": args.mixup,
            "cutmix": args.cutmix,
            "copy_paste": args.copy_paste,
            "device": args.device,
            "amp_requested": args.amp,
            "deterministic": args.deterministic,
            "resume_requested": args.resume,
            "telemetry_enabled": args.telemetry,
            "telemetry_loss_steps": args.telemetry_loss_steps,
            "moa_mot_temperature_factor": args.temperature_factor,
            "moa_mot_min_temperature": args.temperature_min,
            "runtime_overrides_requested": dict(spec.runtime_overrides),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        },
        "routing_requested": routing_manifest(model, spec.runtime_overrides),
        "evidence_boundary": (
            "This manifest records requested configuration only. Runtime dispatch, DDP consistency, and memory "
            "evidence are recorded in telemetry.json after training."
        ),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "run_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def train_spec(args: argparse.Namespace, spec: ModelSpec, data_yaml: Path, project: Path, seed: int) -> None:
    resume_ckpt = project / spec.key / "weights" / "last.pt"
    resume = bool(args.resume and resume_ckpt.exists())
    model = YOLO(str(resume_ckpt if resume else spec.cfg), task=spec.task)
    if args.telemetry:
        os.environ[TELEMETRY_ENV] = "1"
        os.environ[TELEMETRY_LOSS_STEPS_ENV] = str(args.telemetry_loss_steps)
    else:
        os.environ.pop(TELEMETRY_ENV, None)
        os.environ.pop(TELEMETRY_LOSS_STEPS_ENV, None)
    manifest = write_run_manifest(args, spec, project, seed, model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=seed,
        deterministic=args.deterministic,
        project=str(project),
        name=spec.key,
        exist_ok=args.exist_ok,
        pretrained=False,
        val=True,
        plots=args.plots,
        cache=args.cache,
        patience=args.patience,
        optimizer=args.optimizer,
        mosaic=args.mosaic,
        mixup=args.mixup,
        cutmix=args.cutmix,
        copy_paste=args.copy_paste,
        amp=args.amp,
        moa_mot_temperature_factor=args.temperature_factor,
        moa_mot_min_temperature=args.temperature_min,
        resume=resume,
        verbose=args.verbose,
        **dict(spec.runtime_overrides),
    )
    print(f"[manifest] wrote {manifest}")


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
        row = build_row(spec, device="cpu")
        row.update(
            {
                "run_dir": str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
                "epoch": metrics.get("epoch", ""),
            }
        )
        for key, value in benchmark_rows.get(spec.key, {}).items():
            if key not in {
                "key",
                "label",
                "cfg",
                "task",
                "variant",
                "params",
                "params_m",
                "moablocks",
                "c2fmoa",
                "motblocks",
                "c2fmot",
            }:
                row[key] = value
        for key in METRIC_KEYS:
            row[key] = metrics.get(key, "")
        row.update(stability_from_results(run_dir / "results.csv"))
        row.update(telemetry_summary_fields(read_json(run_dir / "telemetry.json")))
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
        "--actual-flops", action="store_true", help="Use torch profiler on the full input size for FLOPs."
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--cutmix", type=float, default=0.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Run each model for these seeds under PROJECT/seed_<seed>. Overrides --seed for --train.",
    )
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resume", action="store_true", help="Resume each model from PROJECT/<key>/weights/last.pt when present."
    )
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-rank training timing, memory, routing, and DDP-consistency telemetry JSON artifacts.",
    )
    parser.add_argument(
        "--telemetry-loss-steps",
        type=int,
        default=20,
        help="Number of initial rank-local training losses retained for DDP consistency analysis.",
    )
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--temperature-factor", type=float, default=0.97, help="Shared MoA/MoT per-epoch router temperature multiplier."
    )
    parser.add_argument("--temperature-min", type=float, default=0.3, help="Shared MoA/MoT router temperature floor.")
    parser.add_argument("--moa-temp-factor", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--moa-min-temp", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--mot-temp-factor", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--mot-min-temp", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = select_specs(args.models)
    project = args.project if args.project.is_absolute() else ROOT / args.project
    data_yaml = args.data if args.data.is_absolute() else ROOT / args.data
    if args.telemetry_loss_steps < 0:
        raise SystemExit("--telemetry-loss-steps must be non-negative")
    if args.seeds is not None and not args.seeds:
        raise SystemExit("--seeds must contain at least one seed")
    legacy_factors = [value for value in (args.moa_temp_factor, args.mot_temp_factor) if value is not None]
    legacy_mins = [value for value in (args.moa_min_temp, args.mot_min_temp) if value is not None]
    if len(set(legacy_factors)) > 1 or len(set(legacy_mins)) > 1:
        raise SystemExit(
            "Separate MoA and MoT temperature schedules are not compatible with DDP-safe execution; "
            "use one shared --temperature-factor and --temperature-min."
        )
    if legacy_factors:
        args.temperature_factor = legacy_factors[0]
    if legacy_mins:
        args.temperature_min = legacy_mins[0]
    if args.temperature_factor <= 0 or args.temperature_min <= 0:
        raise SystemExit("--temperature-factor and --temperature-min must be positive")

    if args.check_build:
        rows = [build_row(spec, device=args.device, imgsz=args.imgsz, include_flops=True) for spec in specs]
        out = project / "build_summary.csv"
        write_csv(out, rows)
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"[build] wrote {out}")

    if args.benchmark:
        rows = [
            benchmark_row(spec, args.device, args.imgsz, args.warmup, args.reps, args.actual_flops) for spec in specs
        ]
        out = project / f"latency_{args.device}_{args.imgsz}.csv"
        write_csv(out, rows)
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"[benchmark] wrote {out}")

    if args.train:
        validate_training_specs(specs, data_yaml)
        project.mkdir(parents=True, exist_ok=True)
        seeds = args.seeds or [args.seed]
        multi_seed = args.seeds is not None
        for seed in seeds:
            seed_project = project / f"seed_{seed}" if multi_seed else project
            for spec in specs:
                train_spec(args, spec, data_yaml, seed_project, seed)
                out = write_summary(seed_project, specs)
                print(f"[summary] wrote {out}")

    if args.summary_only:
        summary_projects = [project / f"seed_{seed}" for seed in args.seeds] if args.seeds else [project]
        for summary_project in summary_projects:
            out = write_summary(summary_project, specs)
            print(f"[summary] wrote {out}")

    if not any((args.check_build, args.benchmark, args.train, args.summary_only)):
        raise SystemExit("choose one or more actions: --check-build, --benchmark, --train, --summary-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
