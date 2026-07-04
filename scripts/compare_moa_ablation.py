#!/usr/bin/env python3
"""Run reproducible YOLO-Master MoA ablations.

Examples:
    python3 scripts/compare_moa_ablation.py --check-build
    python3 scripts/compare_moa_ablation.py --benchmark --imgsz 256 --reps 5 --device cpu
    python3 scripts/compare_moa_ablation.py --train --epochs 50 --imgsz 640 --batch 8 --device 0
    python3 scripts/compare_moa_ablation.py --summary-only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules.moa import C2fMoA, MoABlock  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    cfg: Path


SPECS = {
    "v07": ModelSpec(
        key="v07",
        label="YOLO-Master v0.7 baseline",
        cfg=ROOT / "ultralytics/cfg/models/master/v0_7/det/yolo-master-n.yaml",
    ),
    "v08_moa": ModelSpec(
        key="v08_moa",
        label="YOLO-Master v0.8 MoA",
        cfg=ROOT / "ultralytics/cfg/models/master/v0_8/det/yolo-master-moa-n.yaml",
    ),
    "v10": ModelSpec(
        key="v10",
        label="YOLO-Master v0.10 baseline",
        cfg=ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-n.yaml",
    ),
    "v10_moa": ModelSpec(
        key="v10_moa",
        label="YOLO-Master v0.10 MoA",
        cfg=ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-n.yaml",
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


def build_model(spec: ModelSpec, device: str = "cpu") -> DetectionModel:
    model = DetectionModel(str(spec.cfg), ch=3, nc=80, verbose=False).eval()
    if device:
        model.to(torch.device(device))
    return model


def build_row(spec: ModelSpec, device: str = "cpu") -> dict[str, str]:
    model = build_model(spec, device=device)
    params = sum(p.numel() for p in model.parameters())
    return {
        "key": spec.key,
        "label": spec.label,
        "cfg": str(spec.cfg.relative_to(ROOT)),
        "params": str(params),
        "params_m": f"{params / 1e6:.6f}",
        "moablocks": str(count_modules(model, MoABlock)),
        "c2fmoa": str(count_modules(model, C2fMoA)),
    }


def benchmark_row(spec: ModelSpec, device: str, imgsz: int, warmup: int, reps: int) -> dict[str, str]:
    torch.set_grad_enabled(False)
    model = build_model(spec, device=device)
    x = torch.randn(1, 3, imgsz, imgsz, device=torch.device(device))

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

    base = build_row(spec, device=device)
    base.update(
        {
            "device": device or "cpu",
            "imgsz": str(imgsz),
            "latency_ms_mean": f"{sum(times) / len(times):.3f}",
            "latency_ms_min": f"{min(times):.3f}",
            "latency_ms_max": f"{max(times):.3f}",
            "reps": str(reps),
        }
    )
    return base


def sync_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def read_last_metrics(results_csv: Path) -> dict[str, str]:
    if not results_csv.exists():
        return {}
    with results_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return {k.strip(): v for k, v in rows[-1].items()}


def train_spec(args: argparse.Namespace, spec: ModelSpec, data_yaml: Path, project: Path) -> None:
    model = YOLO(str(spec.cfg))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        project=str(project),
        name=spec.key,
        exist_ok=args.exist_ok,
        pretrained=False,
        val=True,
        plots=args.plots,
        cache=args.cache,
        patience=args.patience,
        amp=args.amp,
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
    for spec in specs:
        run_dir = project / spec.key
        metrics = read_last_metrics(run_dir / "results.csv")
        row = {
            "key": spec.key,
            "label": spec.label,
            "cfg": str(spec.cfg.relative_to(ROOT)),
            "run_dir": str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
            "epoch": metrics.get("epoch", ""),
        }
        for key in METRIC_KEYS:
            row[key] = metrics.get(key, "")
        rows.append(row)
    out = project / "summary.csv"
    write_csv(out, rows)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["v07", "v08_moa", "v10", "v10_moa"], choices=tuple(SPECS))
    parser.add_argument("--project", type=Path, default=ROOT / "runs/moa_ablation")
    parser.add_argument("--data", type=Path, default=default_data_yaml())
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--check-build", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--moa-temp-factor", type=float, default=0.97)
    parser.add_argument("--moa-min-temp", type=float, default=0.3)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = select_specs(args.models)
    project = args.project if args.project.is_absolute() else ROOT / args.project
    data_yaml = args.data if args.data.is_absolute() else ROOT / args.data

    if args.check_build:
        rows = [build_row(spec, device=args.device) for spec in specs]
        out = project / "build_summary.csv"
        write_csv(out, rows)
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"[build] wrote {out}")

    if args.benchmark:
        rows = [benchmark_row(spec, args.device, args.imgsz, args.warmup, args.reps) for spec in specs]
        out = project / f"latency_{args.device}_{args.imgsz}.csv"
        write_csv(out, rows)
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"[benchmark] wrote {out}")

    if args.train:
        project.mkdir(parents=True, exist_ok=True)
        for spec in specs:
            train_spec(args, spec, data_yaml, project)
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
