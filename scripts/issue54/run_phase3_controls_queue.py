#!/usr/bin/env python3
"""Run formal EsMoE and MoA Phase 3 controls in a strict serial queue."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.issue54.run_phase3_control_seed import (  # noqa: E402
    CONFIGS,
    experiment_root,
    load_checkpoint,
    read_final_metrics,
    sha256_file,
)
from scripts.issue54.schema import load_json, validate_experiment_manifest  # noqa: E402

DEFAULT_RUNS = (("v10", 0), ("v10", 1), ("v10", 2), ("v10_moa", 0))
DATA = Path("configs/visdrone_issue54.yaml")
EXPECTED = {"dataset": "VisDrone2019-DET", "dataset_version": "2019-DET", "split": "val-full"}
MIN_FREE_BYTES = 20 * 1024**3
_ACTIVE_PROCESS: subprocess.Popen[str] | None = None


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def state(root: Path, status: str, run: str | None, detail: str | None) -> None:
    payload = {"status": status, "run": run, "detail": detail, "pid": os.getpid(), "timestamp": now()}
    atomic_write(root / "controller_state.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    atomic_write(root / "heartbeat.txt", f"{payload['timestamp']} status={status} run={run}\n")


def log(root: Path, message: str) -> None:
    with (root / "controller.log").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{now()} {message}\n")
        handle.flush()


def read_root(path: Path) -> Path:
    value = path.read_text(encoding="utf-8").strip()
    if not value or "\n" in value or "\r" in value:
        raise ValueError(f"root file must contain one path: {path}")
    root = Path(value)
    if not root.is_dir():
        raise FileNotFoundError(f"result root not found: {root}")
    return root


def parse_runs(values: list[str] | None) -> tuple[tuple[str, int], ...]:
    if values is None:
        return DEFAULT_RUNS
    parsed = []
    for token in values:
        model, separator, seed = token.partition(":")
        if not separator or model not in CONFIGS:
            raise ValueError("--runs entries must be v10:SEED or v10_moa:SEED")
        parsed.append((model, int(seed)))
    if not parsed or len(set(parsed)) != len(parsed):
        raise ValueError("--runs must be non-empty and unique")
    return tuple(parsed)


def runner_command(root: Path, model: str, seed: int, *, data: Path, dataset_manifest: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/issue54/run_phase3_control_seed.py"),
        "--seed",
        str(seed),
        "--model",
        model,
        "--data",
        str(data),
        "--output-root",
        str(root),
        "--dataset-name",
        EXPECTED["dataset"],
        "--dataset-version",
        EXPECTED["dataset_version"],
        "--dataset-manifest",
        str(dataset_manifest),
        "--validation-split",
        EXPECTED["split"],
        "--epochs",
        "30",
        "--device",
        "0",
        "--batch",
        "8",
        "--imgsz",
        "640",
        "--formal",
    ]


def mot_hashes(root: Path) -> set[str]:
    hashes = set()
    for seed in range(5):
        manifest = validate_experiment_manifest(
            load_json(root / f"phase3_v10_mot_seed{seed}" / "experiment_manifest.json")
        )
        if (
            manifest["model_variant"] != "v10_mot"
            or manifest["seed"] != seed
            or manifest["status"] != "passed"
            or not manifest["checkpoint_sha256"]
        ):
            raise ValueError(f"MoT seed {seed} is not passed evidence")
        hashes.add(manifest["checkpoint_sha256"])
    if len(hashes) != 5:
        raise ValueError("MoT checkpoint hashes are not unique")
    return hashes


def resolve_checkpoint(run_root: Path, checkpoint_path: str | Path) -> Path:
    """Resolve an absolute checkpoint directly or a manifest path under its run directory."""
    checkpoint = Path(checkpoint_path)
    return checkpoint if checkpoint.is_absolute() else run_root / checkpoint


def validate_run(root: Path, model: str, seed: int, used_hashes: set[str]) -> str | None:
    run = experiment_root(root, model, seed)
    results = run / "training" / model / "results.csv"
    last_checkpoint = run / "training" / model / "weights" / "last.pt"
    manifest_path = run / "experiment_manifest.json"
    exitcode_path = run / "exitcode"
    for path in (results, last_checkpoint, manifest_path, exitcode_path):
        if not path.is_file():
            return f"{model} seed {seed}: missing {path.name}"
    if exitcode_path.read_text(encoding="utf-8").strip() != "0":
        return f"{model} seed {seed}: runner exit code is not 0"
    try:
        manifest = validate_experiment_manifest(load_json(manifest_path))
        checkpoint = resolve_checkpoint(run, manifest["checkpoint_path"])
        if not checkpoint.is_file():
            return f"{model} seed {seed}: missing checkpoint"
        load_checkpoint(checkpoint)
        read_final_metrics(results, 30)
        expected = {
            "model_variant": model,
            "seed": seed,
            "status": "passed",
            "failure_reason": None,
            "requested_epochs": 30,
            "requested_batch": 8,
            "batch": 8,
            "effective_batch": 8,
            "imgsz": 640,
            "precision_mode": "amp",
            **EXPECTED,
        }
        for field, value in expected.items():
            if manifest.get(field) != value:
                return f"{model} seed {seed}: manifest {field} mismatch"
        checkpoint_hash = sha256_file(checkpoint)
        if checkpoint_hash != manifest["checkpoint_sha256"]:
            return f"{model} seed {seed}: checkpoint SHA256 mismatch"
        if checkpoint_hash in used_hashes:
            return f"{model} seed {seed}: duplicate checkpoint SHA256"
        used_hashes.add(checkpoint_hash)
    except Exception as error:
        return f"{model} seed {seed}: validation failed: {error}"
    return None


def gpu_idle() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0 and not result.stdout.strip()


def preflight(root: Path, data: Path, dataset_manifest: Path) -> str | None:
    if not gpu_idle():
        return "GPU is busy or nvidia-smi is unavailable"
    if shutil.disk_usage(root).free < MIN_FREE_BYTES:
        return "insufficient disk space"
    resolved_data = data if data.is_absolute() else ROOT / data
    resolved_manifest = dataset_manifest if dataset_manifest.is_absolute() else ROOT / dataset_manifest
    for path in (resolved_data, resolved_manifest, *(ROOT / value for value in CONFIGS.values())):
        if not path.is_file():
            return f"required input missing: {path}"
    subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
    return None


def acquire_lock(root: Path) -> Path:
    lock = root / "controller.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"another controls controller exists: {lock}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()} started={now()}\n")
    return lock


def stop_handler(signum: int, _frame: Any) -> None:
    if _ACTIVE_PROCESS is not None and _ACTIVE_PROCESS.poll() is None:
        _ACTIVE_PROCESS.terminate()
    raise RuntimeError(f"received termination signal {signum}")


def run_one(
    root: Path,
    model: str,
    seed: int,
    poll_seconds: float,
    *,
    data: Path,
    dataset_manifest: Path,
) -> int:
    global _ACTIVE_PROCESS
    name = f"{model}_seed{seed}"
    with (root / f"{name}.log").open("x", encoding="utf-8", newline="\n") as output:
        _ACTIVE_PROCESS = subprocess.Popen(
            runner_command(root, model, seed, data=data, dataset_manifest=dataset_manifest),
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
        )
        atomic_write(root / f"{name}.pid", f"{_ACTIVE_PROCESS.pid}\n")
        while _ACTIVE_PROCESS.poll() is None:
            state(root, "running", name, "runner active")
            time.sleep(poll_seconds)
        code = _ACTIVE_PROCESS.returncode
    _ACTIVE_PROCESS = None
    atomic_write(root / f"{name}.exitcode", f"{code}\n")
    return code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls-root-file", type=Path, required=True)
    parser.add_argument("--mot-root-file", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    controls_root = read_root(args.controls_root_file)
    mot_root = read_root(args.mot_root_file)
    runs = parse_runs(args.runs)
    if not args.validate_only and args.dataset_manifest is None:
        raise ValueError("--dataset-manifest is required for launch or dry-run")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "controls_root": str(controls_root),
                    "runs": runs,
                    "commands": [
                        runner_command(controls_root, *run, data=args.data, dataset_manifest=args.dataset_manifest)
                        for run in runs
                    ],
                },
                indent=2,
            )
        )
        return 0
    used_hashes = mot_hashes(mot_root)
    if args.validate_only:
        for model, seed in runs:
            error = validate_run(controls_root, model, seed, used_hashes)
            if error:
                print(error, file=sys.stderr)
                return 1
        return 0
    lock = acquire_lock(controls_root)
    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    try:
        state(controls_root, "starting", None, None)
        for model, seed in runs:
            run = experiment_root(controls_root, model, seed)
            if run.exists():
                error = validate_run(controls_root, model, seed, used_hashes)
                if error:
                    raise RuntimeError(error)
                log(controls_root, f"{model} seed {seed} already passed; skipped")
                continue
            issue = preflight(controls_root, args.data, args.dataset_manifest)
            if issue:
                raise RuntimeError(issue)
            if (
                run_one(
                    controls_root,
                    model,
                    seed,
                    args.poll_seconds,
                    data=args.data,
                    dataset_manifest=args.dataset_manifest,
                )
                != 0
            ):
                raise RuntimeError(f"{model} seed {seed} runner failed")
            error = validate_run(controls_root, model, seed, used_hashes)
            if error:
                raise RuntimeError(error)
            log(controls_root, f"{model} seed {seed} passed")
        state(controls_root, "passed", None, "all requested controls passed")
        return 0
    except Exception as error:
        state(controls_root, "failed", None, str(error))
        log(controls_root, f"queue failed: {error}")
        print(error, file=sys.stderr)
        return 1
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
