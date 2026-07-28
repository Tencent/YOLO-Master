#!/usr/bin/env python3
"""Run the complete Issue #54 ablation and same-checkpoint routing experiment.

The default protocol trains four variants concurrently on four GPUs, benchmarks
them sequentially on one GPU, and audits one trained MoT checkpoint across
VisDrone, COCO128, and brain-tumor.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pickle
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = ROOT / "scripts/compare_mot_ablation.py"
ROUTING_SCRIPT = ROOT / "scripts/analyze_mot_cross_domain.py"
SCENE_SCRIPT = ROOT / "scripts/prepare_mot_routing_scenes.py"
DEFAULT_MODELS = ("v10", "v10_mot", "v10_moa", "v10_mot_p5")
STAGES = ("prepare", "check", "train", "benchmark", "audit")


def resolve_dataset_yaml(value: str | Path) -> Path:
    """Resolve a dataset YAML from an explicit path or the built-in catalog."""
    path = Path(value).expanduser()
    candidates = (
        path,
        ROOT / path,
        ROOT / "ultralytics/cfg/datasets" / path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"dataset YAML not found: {value}")


def run_logged(command: list[str], log_path: Path, cwd: Path = ROOT) -> None:
    """Run one command with a durable log and raise on failure."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = shlex.join(command)
    print(f"[run] {rendered}")
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {rendered}\n\n")
        log.flush()
        process = subprocess.run(
            command,
            cwd=cwd,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    print(f"[done] {log_path}")


def prepare_dataset(yaml_path: Path) -> dict[str, Any]:
    """Download/validate one official dataset and return JSON-safe paths."""
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(yaml_path), autodownload=True)
    return json_safe(
        {
            "yaml": str(yaml_path),
            "path": str(data["path"]),
            "train": data.get("train"),
            "val": data.get("val"),
            "test": data.get("test"),
            "names": data.get("names"),
            "nc": data.get("nc"),
        }
    )


def json_safe(value: Any) -> Any:
    """Convert paths and nested dataset metadata into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def first_image_directory(value: Any) -> Path:
    """Return a concrete image directory from an Ultralytics split value."""
    candidates = value if isinstance(value, list) else [value]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_dir():
            return path.resolve()
    raise FileNotFoundError(f"dataset split does not resolve to an image directory: {value!r}")


def repository_state() -> dict[str, Any]:
    """Capture source revision and dirty state for experiment provenance."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"git_commit": commit, "git_dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def write_protocol(project: Path, args: argparse.Namespace, datasets: dict[str, Any]) -> None:
    """Persist the exact orchestration parameters before expensive stages run."""
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "command": [sys.executable, *sys.argv],
        **repository_state(),
        "stages": args.stages,
        "models": args.models,
        "devices": args.devices,
        "training_dataset": str(resolve_dataset_yaml(args.data)),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers_per_model": args.workers,
        "seed": args.seed,
        "benchmark_device": args.benchmark_device,
        "benchmark_warmup": args.warmup,
        "benchmark_repetitions": args.reps,
        "audit_device": args.audit_device,
        "audit_images_per_domain": args.audit_images,
        "smoke": args.smoke,
        "datasets": datasets,
    }
    serialized = json.dumps(payload, ensure_ascii=False)
    (project / "experiment_protocol.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (project / "experiment_invocations.jsonl").open("a", encoding="utf-8") as file:
        file.write(serialized + "\n")


def prepare_datasets(args: argparse.Namespace, project: Path) -> dict[str, dict[str, Any]]:
    """Prepare the training and audit datasets before concurrent workers start."""
    train_yaml = resolve_dataset_yaml(args.data)
    coco_yaml = resolve_dataset_yaml("coco128.yaml")
    brain_yaml = resolve_dataset_yaml("brain-tumor.yaml")
    requested = {
        "training": train_yaml,
        "coco128": coco_yaml,
        "brain_tumor": brain_yaml,
    }
    datasets: dict[str, dict[str, Any]] = {}
    for name, yaml_path in requested.items():
        if any(item.get("yaml") == str(yaml_path) for item in datasets.values()):
            continue
        print(f"[prepare] validating {name}: {yaml_path.name}")
        datasets[name] = prepare_dataset(yaml_path)
    manifest = project / "prepared_datasets.json"
    manifest.write_text(json.dumps(datasets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return datasets


def load_prepared_datasets(project: Path) -> dict[str, dict[str, Any]]:
    path = project / "prepared_datasets.json"
    if not path.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {path}; run the 'prepare' stage first")
    return json.loads(path.read_text(encoding="utf-8"))


def run_check(args: argparse.Namespace, project: Path) -> None:
    command = [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--check-build",
        "--models",
        *args.models,
        "--imgsz",
        str(args.imgsz),
        "--project",
        str(project / "build"),
    ]
    run_logged(command, project / "logs/check_build.log")


def training_command(
    args: argparse.Namespace,
    project: Path,
    model: str,
    device: str,
    data_yaml: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--train",
        "--models",
        model,
        "--project",
        str(project / "training"),
        "--data",
        str(data_yaml),
        "--device",
        device,
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
        "--patience",
        "0",
        "--exist-ok",
        "--no-summary",
        "--plots",
    ]
    if args.resume:
        command.append("--resume")
    if not args.amp:
        command.append("--no-amp")
    if not args.deterministic:
        command.append("--no-deterministic")
    return command


def ensure_amp_reference() -> Path:
    """Download and validate the AMP reference model once before workers fork.

    Ultralytics performs an AMP compatibility check in every trainer. Without
    this preflight, first-time parallel runs can write the same asset concurrently.
    """
    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset

    reference = ROOT / "yolo26n.pt"
    try:
        attempt_download_asset(reference)
        YOLO(str(reference))
    except (EOFError, OSError, RuntimeError, ValueError, pickle.UnpicklingError):
        reference.unlink(missing_ok=True)
        attempt_download_asset(reference)
        YOLO(str(reference))
    return reference


def run_parallel_training(args: argparse.Namespace, project: Path) -> None:
    """Assign one model to each GPU and wait for every training process."""
    if not args.devices:
        raise ValueError("at least one --devices value is required")
    if args.amp:
        reference = ensure_amp_reference()
        print(f"[preflight] validated shared AMP reference: {reference}")
    assignments = [(model, args.devices[index % len(args.devices)]) for index, model in enumerate(args.models)]
    data_yaml = resolve_dataset_yaml(args.data)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(assignments)) as executor:
        futures = {
            executor.submit(
                run_logged,
                training_command(args, project, model, device, data_yaml),
                project / f"logs/train_{model}_gpu{device}.log",
            ): (model, device)
            for model, device in assignments
        }
        for future in concurrent.futures.as_completed(futures):
            model, device = futures[future]
            try:
                future.result()
            except (OSError, subprocess.SubprocessError) as error:
                failures.append((model, device, str(error)))
    if failures:
        details = "; ".join(f"{model}@GPU{device}: {error}" for model, device, error in failures)
        raise RuntimeError(f"one or more training jobs failed: {details}")

    summary_command = [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--summary-only",
        "--models",
        *args.models,
        "--project",
        str(project / "training"),
    ]
    run_logged(summary_command, project / "logs/training_summary.log")


def run_benchmark(args: argparse.Namespace, project: Path) -> None:
    """Benchmark all variants sequentially on one GPU for comparability."""
    command = [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--benchmark",
        "--actual-flops",
        "--trained-weights",
        "--models",
        *args.models,
        "--project",
        str(project / "training"),
        "--device",
        args.benchmark_device,
        "--imgsz",
        str(args.imgsz),
        "--warmup",
        str(args.warmup),
        "--reps",
        str(args.reps),
    ]
    run_logged(command, project / "logs/benchmark.log")
    summary_command = [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--summary-only",
        "--models",
        *args.models,
        "--project",
        str(project / "training"),
    ]
    run_logged(summary_command, project / "logs/benchmark_summary.log")


def find_dataset(datasets: dict[str, dict[str, Any]], yaml_name: str) -> dict[str, Any]:
    for dataset in datasets.values():
        if Path(dataset["yaml"]).name == yaml_name:
            return dataset
    raise KeyError(f"prepared dataset not found for {yaml_name}")


def run_audit(args: argparse.Namespace, project: Path, datasets: dict[str, dict[str, Any]]) -> None:
    """Run cross-domain and within-VisDrone scene audits with one checkpoint."""
    checkpoint = project / "training/v10_mot/weights/best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"trained MoT checkpoint not found: {checkpoint}")

    train_dataset = find_dataset(datasets, Path(resolve_dataset_yaml(args.data)).name)
    coco_dataset = find_dataset(datasets, "coco128.yaml")
    brain_dataset = find_dataset(datasets, "brain-tumor.yaml")
    train_val = first_image_directory(train_dataset["val"])
    coco_val = first_image_directory(coco_dataset["val"])
    brain_val = first_image_directory(brain_dataset["val"])

    # 同一 checkpoint 跨域比较：消除“数据域”和“模型参数”同时变化的混淆。
    domains = [
        f"{'training-domain' if args.smoke else 'VisDrone'}={train_val}",
        f"COCO128={coco_val}",
        f"brain-tumor={brain_val}",
    ]
    if args.smoke and train_val == coco_val:
        domains = [f"COCO128={coco_val}", f"brain-tumor={brain_val}"]
    cross_command = [
        sys.executable,
        str(ROUTING_SCRIPT),
        "--model",
        str(checkpoint),
        "--device",
        args.audit_device,
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.audit_batch),
        "--max-images",
        str(args.audit_images),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--permutations",
        str(args.permutations),
        "--seed",
        str(args.seed),
        "--output",
        str(project / "routing/cross_domain"),
    ]
    for domain in domains:
        cross_command.extend(("--domain", domain))
    run_logged(cross_command, project / "logs/audit_cross_domain.log")

    if args.smoke or not args.scene_audit:
        return
    visdrone_root = Path(train_dataset["path"])
    scene_root = project / "routing/visdrone_scene_inputs"
    scene_command = [
        sys.executable,
        str(SCENE_SCRIPT),
        "--dataset",
        str(visdrone_root),
        "--split",
        "val",
        "--output",
        str(scene_root),
        "--max-images-per-scene",
        str(args.audit_images),
    ]
    run_logged(scene_command, project / "logs/prepare_visdrone_scenes.log")

    scene_names = ("dense", "sparse", "small_objects", "large_objects", "irregular_occluded")
    scene_audit_command = [
        sys.executable,
        str(ROUTING_SCRIPT),
        "--model",
        str(checkpoint),
        "--device",
        args.audit_device,
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.audit_batch),
        "--max-images",
        str(args.audit_images),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--permutations",
        str(args.permutations),
        "--seed",
        str(args.seed),
        "--no-perturbations",
        "--output",
        str(project / "routing/visdrone_scenes"),
    ]
    for scene in scene_names:
        scene_audit_command.extend(("--domain", f"{scene}={scene_root / scene}"))
    run_logged(scene_audit_command, project / "logs/audit_visdrone_scenes.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--data", default="VisDrone.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable AMP. The reference protocol defaults to FP32 after an AMP non-finite-gradient calibration.",
    )
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--benchmark-device", default="0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--audit-device", default="0")
    parser.add_argument("--audit-batch", type=int, default=4)
    parser.add_argument("--audit-images", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--scene-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true", help="Use a one-epoch, low-resolution COCO128 protocol.")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/mot_cross_domain")
    args = parser.parse_args()
    if args.smoke:
        args.data = "coco128.yaml"
        args.epochs = 1
        args.imgsz = min(args.imgsz, 160)
        args.batch = min(args.batch, 4)
        args.workers = min(args.workers, 2)
        args.warmup = min(args.warmup, 2)
        args.reps = min(args.reps, 5)
        args.audit_images = min(args.audit_images, 8)
        args.bootstrap_samples = min(args.bootstrap_samples, 200)
        args.permutations = min(args.permutations, 200)
    return args


def main() -> int:
    args = parse_args()
    project = args.project if args.project.is_absolute() else ROOT / args.project
    project.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, dict[str, Any]] = {}
    if "prepare" in args.stages:
        datasets = prepare_datasets(args, project)
    elif "audit" in args.stages or (project / "prepared_datasets.json").is_file():
        datasets = load_prepared_datasets(project)
    write_protocol(project, args, datasets)

    if "check" in args.stages:
        run_check(args, project)
    if "train" in args.stages:
        run_parallel_training(args, project)
    if "benchmark" in args.stages:
        run_benchmark(args, project)
    if "audit" in args.stages:
        run_audit(args, project, datasets)
    print(f"[complete] experiment artifacts: {project}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
