#!/usr/bin/env python3
"""Train and compare YOLO-Master MoE versions on local COCO128.

Examples:
    /usr/bin/python3 scripts/compare_moe_coco128.py --dry-run
    /usr/bin/python3 scripts/compare_moe_coco128.py --check-build
    /usr/bin/python3 scripts/compare_moe_coco128.py --epochs 30 --imgsz 640 --batch 8 --device mps
    /usr/bin/python3 scripts/compare_moe_coco128.py --resume-existing --extra-epochs 5 --device mps
    /usr/bin/python3 scripts/compare_moe_coco128.py --init-from-project runs/moe_coco128_compare_5e_128 --epochs 5
    /usr/bin/python3 scripts/compare_moe_coco128.py --summary-only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402


DEFAULT_VERSIONS = [f"v0_{i}" for i in range(1, 11)]
METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "val/moe_loss",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "train/moe_loss",
)


@dataclass(frozen=True)
class VersionSpec:
    version: str
    cfg: Path
    run_name: str


def normalize_version(value: str) -> str:
    value = value.strip().replace(".", "_")
    if value in {"stable", "v0_stable", "v_stable"}:
        return "v0_stable"
    if value.startswith("v"):
        return value
    return f"v{value}"


def version_index(version: str) -> int:
    return int(version.split("_", 1)[1])


def find_cfg(version: str) -> Path:
    if version == "v0_stable":
        candidates = [
            ROOT / "ultralytics/cfg/models/master/exp/yolo-master-v0_stable.yaml",
            ROOT / "ultralytics/cfg/models/master/v0_stable/det/yolo-master-n.yaml",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"No config found for {version}: {', '.join(str(p) for p in candidates)}")

    idx = version_index(version)
    candidates = [
        ROOT / f"ultralytics/cfg/models/master/exp/yolo-master-v0_{idx}.yaml",
        ROOT / f"ultralytics/cfg/models/master/{version}/det/yolo-master-n.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No config found for {version}: {', '.join(str(p) for p in candidates)}")


def discover_specs(versions: Iterable[str], name_prefix: str = "") -> list[VersionSpec]:
    specs = []
    for version in versions:
        version = normalize_version(version)
        run_name = f"{name_prefix}{version}" if name_prefix else version
        specs.append(VersionSpec(version=version, cfg=find_cfg(version), run_name=run_name))
    return specs


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


def read_last_metrics(results_csv: Path) -> dict[str, str]:
    if not results_csv.exists():
        return {}
    with results_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return {k.strip(): v for k, v in rows[-1].items()}


def completed_epoch(run_dir: Path) -> int | None:
    metrics = read_last_metrics(run_dir / "results.csv")
    value = metrics.get("epoch")
    if value in {None, ""}:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def float_or_blank(value: str | None) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6g}"
    except ValueError:
        return value


def collect_summary(project: Path, specs: list[VersionSpec]) -> list[dict[str, str]]:
    rows = []
    for spec in specs:
        run_dir = project / spec.run_name
        results = read_last_metrics(run_dir / "results.csv")
        row = {
            "version": spec.version,
            "cfg": str(spec.cfg.relative_to(ROOT)),
            "run_dir": str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
            "epoch": results.get("epoch", ""),
        }
        for key in METRIC_KEYS:
            row[key] = float_or_blank(results.get(key))
        rows.append(row)
    return rows


def write_summary(project: Path, specs: list[VersionSpec]) -> Path:
    rows = collect_summary(project, specs)
    project.mkdir(parents=True, exist_ok=True)
    out = project / "summary.csv"
    fieldnames = ["version", "cfg", "run_dir", "epoch", *METRIC_KEYS]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def safe_write_summary(project: Path, specs: list[VersionSpec]) -> None:
    try:
        out = write_summary(project, specs)
        print(f"[summary] updated {out}")
    except OSError as exc:
        print(f"[summary-warning] failed to write summary: {exc}")


def check_build(specs: list[VersionSpec]) -> None:
    for spec in specs:
        model = DetectionModel(str(spec.cfg), ch=3, nc=80, verbose=False)
        params = sum(p.numel() for p in model.parameters())
        print(f"[build-ok] {spec.version:<5} params={params / 1e6:.3f}M cfg={spec.cfg.relative_to(ROOT)}")


def train_one(args: argparse.Namespace, spec: VersionSpec, data_yaml: Path, project: Path) -> dict[str, str]:
    start = time.time()
    run_dir = project / spec.run_name
    resume_existing = args.resume_existing or args.extra_epochs is not None
    completed = completed_epoch(run_dir)
    target_epochs = args.epochs
    if args.extra_epochs is not None and completed is not None:
        target_epochs = completed + args.extra_epochs

    if args.skip_existing and (run_dir / "results.csv").exists() and not resume_existing:
        print(f"[skip] {spec.version}: existing {run_dir / 'results.csv'}")
        return {"version": spec.version, "status": "skipped", "duration_s": "0"}

    last_pt = run_dir / "weights/last.pt"
    init_weights = None
    if args.init_from_project:
        init_project = args.init_from_project if args.init_from_project.is_absolute() else ROOT / args.init_from_project
        candidate = init_project / spec.version / "weights" / f"{args.init_weight}.pt"
        if candidate.exists():
            init_weights = candidate
        else:
            print(f"[init-miss] {spec.version}: {candidate} not found, starting from cfg")

    if resume_existing and last_pt.exists() and completed is not None:
        if target_epochs <= completed:
            print(f"[skip] {spec.version}: already at epoch {completed}, target={target_epochs}")
            return {"version": spec.version, "status": "skipped", "duration_s": "0"}
        print(f"[resume] {spec.version}: {last_pt} epoch={completed} -> target={target_epochs}")
        model = YOLO(str(last_pt))
        resume = True
    else:
        if init_weights is not None:
            print(f"[finetune] {spec.version}: init={init_weights} epochs={target_epochs}")
            model = YOLO(str(init_weights))
        else:
            if resume_existing:
                print(f"[resume-miss] {spec.version}: no completed checkpoint, starting from cfg")
            print(f"[train] {spec.version}: cfg={spec.cfg.relative_to(ROOT)} data={data_yaml}")
            model = YOLO(str(spec.cfg))
        resume = False

    model.train(
        data=str(data_yaml),
        epochs=target_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        project=str(project),
        name=spec.run_name,
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
    duration = time.time() - start
    status = "resumed" if resume else "finetuned" if init_weights is not None else "ok"
    return {"version": spec.version, "status": status, "duration_s": f"{duration:.2f}"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--versions", nargs="+", default=DEFAULT_VERSIONS, help="Versions to run, e.g. v0_1 v0_2 ...")
    parser.add_argument("--data", type=Path, default=default_data_yaml(), help="Dataset YAML. Defaults to local coco128.")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/moe_coco128_compare")
    parser.add_argument("--name-prefix", default="", help="Prefix for run directories, e.g. smoke_")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of the training set to use (1.0 = all).")
    parser.add_argument("--device", default="", help="Ultralytics device string: '', cpu, mps, 0, 0,1 ...")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume-existing", action="store_true", help="Resume from each run's weights/last.pt when present.")
    parser.add_argument("--extra-epochs", type=int, help="Additional epochs to run from existing results; implies --resume-existing.")
    parser.add_argument("--init-from-project", type=Path, help="Initialize each version from another project's <version>/weights/*.pt.")
    parser.add_argument("--init-weight", default="last", choices=("last", "best"), help="Weight file to use with --init-from-project.")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved configs and exit.")
    parser.add_argument("--check-build", action="store_true", help="Instantiate all selected DetectionModels and exit.")
    parser.add_argument("--summary-only", action="store_true", help="Only aggregate existing results.csv files.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = discover_specs(args.versions, args.name_prefix)
    data_yaml = args.data if args.data.is_absolute() else ROOT / args.data
    project = args.project if args.project.is_absolute() else ROOT / args.project

    print("[compare] versions:", ", ".join(s.version for s in specs))
    print("[compare] data:", data_yaml)
    print("[compare] project:", project)
    for spec in specs:
        print(f"  - {spec.version:<5} -> {spec.cfg.relative_to(ROOT)}  run={spec.run_name}")

    if args.dry_run:
        return 0
    if args.check_build:
        check_build(specs)
        write_summary(project, specs)
        return 0
    if args.summary_only:
        out = write_summary(project, specs)
        print(f"[summary] wrote {out}")
        return 0

    statuses = []
    project.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        try:
            statuses.append(train_one(args, spec, data_yaml, project))
        except Exception as exc:
            print(f"[fail] {spec.version}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            statuses.append({"version": spec.version, "status": "failed", "error": str(exc)})
            if args.stop_on_failure:
                break
        finally:
            safe_write_summary(project, specs)

    with (project / "status.json").open("w") as f:
        json.dump(statuses, f, indent=2, ensure_ascii=False)
    success_states = {"ok", "skipped", "resumed", "finetuned"}
    return 0 if all(s.get("status") in success_states for s in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
