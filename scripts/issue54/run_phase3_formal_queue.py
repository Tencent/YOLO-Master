#!/usr/bin/env python3
"""Run the Issue #54 formal MoT seeds serially and stop at the first failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.issue54.schema import SchemaValidationError, load_json, validate_experiment_manifest  # noqa: E402


MODEL = "v10_mot"
SEEDS = (0, 1, 2, 3, 4)
DEFAULT_RUN_SEEDS = (1, 2, 3, 4)
DATA = Path("configs/visdrone_issue54.yaml")
EXPECTED = {
    "model_variant": MODEL,
    "dataset": "VisDrone2019-DET",
    "dataset_version": "2019-DET",
    "split": "val-fixed32",
    "status": "passed",
    "failure_reason": None,
}
MIN_FREE_BYTES = 20 * 1024**3
_ACTIVE_PROCESS: subprocess.Popen[str] | None = None


def now() -> str:
    """Return a UTC timestamp suitable for state files."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_write(path: Path, text: str) -> None:
    """Atomically replace one small UTF-8 state file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write controller state atomically."""
    atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    """Return a file SHA256 without loading the checkpoint into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def formal_root(path: Path) -> Path:
    """Read the sole cloud-owned Phase 3 root without creating a replacement."""
    value = path.read_text(encoding="utf-8").strip()
    if not value or "\n" in value or "\r" in value:
        raise ValueError(f"formal root file must contain exactly one path: {path}")
    root = Path(value)
    if not root.is_dir():
        raise FileNotFoundError(f"formal root does not exist: {root}")
    return root


def experiment_root(root: Path, seed: int) -> Path:
    """Return the protocol-owned directory for one formal seed."""
    return root / f"phase3_{MODEL}_seed{seed}"


def routing_line_count(path: Path) -> int:
    """Count JSON-object routing records and reject malformed JSONL."""
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"routing.jsonl:{line_number}: invalid JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"routing.jsonl:{line_number}: record must be an object")
            count += 1
    return count


def resolve_checkpoint(run_root: Path, checkpoint_path: str | Path) -> Path:
    """Resolve an absolute checkpoint directly or a manifest path under its run directory."""
    checkpoint = Path(checkpoint_path)
    return checkpoint if checkpoint.is_absolute() else run_root / checkpoint


def validate_seed(
    root: Path, seed: int, checkpoint_hashes: set[str] | None = None
) -> tuple[dict[str, Any], str | None]:
    """Validate one completed formal seed and optionally reject duplicate checkpoints."""
    run_root = experiment_root(root, seed)
    manifest_path = run_root / "experiment_manifest.json"
    routes = run_root / "routing" / "routing.jsonl"
    for path, label in (
        (manifest_path, "experiment_manifest.json"),
        (routes, "routing.jsonl"),
    ):
        if not path.is_file():
            return {}, f"seed {seed}: missing {label}"
    try:
        manifest = validate_experiment_manifest(load_json(manifest_path))
    except (OSError, SchemaValidationError, ValueError) as error:
        return {}, f"seed {seed}: invalid manifest: {error}"
    checkpoint = resolve_checkpoint(run_root, manifest["checkpoint_path"])
    if not checkpoint.is_file():
        return {}, f"seed {seed}: missing checkpoint: {checkpoint}"
    for field, expected in EXPECTED.items():
        if manifest.get(field) != expected:
            return {}, f"seed {seed}: manifest {field}={manifest.get(field)!r}, expected {expected!r}"
    if manifest["seed"] != seed:
        return {}, f"seed {seed}: manifest seed={manifest['seed']!r}"
    checkpoint_hash = sha256_file(checkpoint)
    if manifest["checkpoint_sha256"] != checkpoint_hash:
        return {}, f"seed {seed}: checkpoint SHA256 does not match manifest"
    if routing_line_count(routes) != 384:
        return {}, f"seed {seed}: routing record count is not 384"
    if checkpoint_hashes is not None:
        if checkpoint_hash in checkpoint_hashes:
            return {}, f"seed {seed}: checkpoint SHA256 duplicates another seed"
        checkpoint_hashes.add(checkpoint_hash)
    return manifest, None


def parse_seeds(values: list[str] | None) -> tuple[int, ...]:
    """Parse unique requested formal seeds, defaulting to the unfinished queue."""
    if not values:
        return DEFAULT_RUN_SEEDS
    parsed = tuple(int(value) for item in values for value in item.split(","))
    if not parsed or any(seed not in SEEDS for seed in parsed) or len(set(parsed)) != len(parsed):
        raise ValueError("--seeds must be unique members of 0,1,2,3,4")
    return tuple(sorted(parsed))


def gpu_is_idle() -> bool:
    """Return whether CUDA reports no active compute process."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0 and not result.stdout.strip()


def preflight(root: Path) -> str | None:
    """Reject a new training launch when GPU or storage is unavailable."""
    if not gpu_is_idle():
        return "GPU is not idle; another compute process is present or nvidia-smi is unavailable"
    if shutil.disk_usage(root).free < MIN_FREE_BYTES:
        return f"insufficient free space under formal root; require at least {MIN_FREE_BYTES} bytes"
    return None


def runner_command(
    root: Path,
    seed: int,
    *,
    data: Path,
    image_manifest: Path,
    data_root: Path,
    dataset_manifest: Path,
) -> list[str]:
    """Return the exact formal runner command for one seed."""
    return [
        sys.executable,
        str(ROOT / "scripts/issue54/run_phase3_seed.py"),
        "--seed",
        str(seed),
        "--model",
        MODEL,
        "--data",
        str(data),
        "--dataset-name",
        EXPECTED["dataset"],
        "--dataset-version",
        EXPECTED["dataset_version"],
        "--dataset-manifest",
        str(dataset_manifest),
        "--routing-split",
        EXPECTED["split"],
        "--output-root",
        str(root),
        "--image-manifest",
        str(image_manifest),
        "--data-root",
        str(data_root),
        "--epochs",
        "30",
        "--device",
        "0",
        "--batch",
        "8",
        "--imgsz",
        "640",
        "--no-amp",
        "--formal",
    ]


def write_state(root: Path, *, status: str, current_seed: int | None, detail: str | None) -> None:
    """Persist the minimal controller state and liveness heartbeat."""
    payload = {"status": status, "current_seed": current_seed, "detail": detail, "pid": os.getpid(), "timestamp": now()}
    write_json(root / "controller_state.json", payload)
    atomic_write(root / "heartbeat.txt", f"{payload['timestamp']} status={status} seed={current_seed}\n")


def append_log(root: Path, message: str) -> None:
    """Append one flushed controller event line."""
    with (root / "controller.log").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{now()} {message}\n")
        handle.flush()


def acquire_lock(root: Path) -> Path:
    """Atomically acquire the singleton controller lock without stealing stale state."""
    lock = root / "controller.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"another controller lock exists: {lock}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()} started={now()}\n")
    return lock


def stop_handler(signum: int, _frame: Any) -> None:
    """Convert a termination signal into the normal failure-stop path."""
    if _ACTIVE_PROCESS is not None and _ACTIVE_PROCESS.poll() is None:
        _ACTIVE_PROCESS.terminate()
    raise RuntimeError(f"received termination signal {signum}")


def run_seed(root: Path, seed: int, poll_seconds: float, command: list[str]) -> int:
    """Run one runner subprocess while preserving its stdout, PID, and exit code."""
    global _ACTIVE_PROCESS
    log_path = root / f"seed{seed}.log"
    pid_path = root / f"seed{seed}.pid"
    exit_path = root / f"seed{seed}.exitcode"
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        _ACTIVE_PROCESS = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        atomic_write(pid_path, f"{_ACTIVE_PROCESS.pid}\n")
        while _ACTIVE_PROCESS.poll() is None:
            write_state(root, status="running", current_seed=seed, detail="runner active")
            time.sleep(poll_seconds)
    exit_code = _ACTIVE_PROCESS.returncode
    _ACTIVE_PROCESS = None
    atomic_write(exit_path, f"{exit_code}\n")
    return exit_code


def finalize(root: Path) -> int:
    """Build the registry and cross-seed analysis only after all five seeds pass."""
    manifests = [experiment_root(root, seed) / "experiment_manifest.json" for seed in SEEDS]
    routes = [experiment_root(root, seed) / "routing" / "routing.jsonl" for seed in SEEDS]
    registry = root / "phase3_formal_registry.json"
    analysis = root / "phase3_cross_seed_routing.json"
    if registry.exists() or analysis.exists():
        raise RuntimeError("formal registry or analysis output already exists; refusing to overwrite")
    registry_command = [sys.executable, str(ROOT / "scripts/issue54/build_experiment_registry.py")]
    for manifest in manifests:
        registry_command.extend(["--manifest", str(manifest)])
    registry_command.extend(["--output", str(registry)])
    if subprocess.run(registry_command, cwd=ROOT, check=False).returncode != 0:
        return 1
    analysis_command = [
        sys.executable,
        str(ROOT / "scripts/issue54/analyze_cross_seed_routing.py"),
        "--registry",
        str(registry),
    ]
    for route in routes:
        analysis_command.extend(["--routes", str(route)])
    analysis_command.extend(["--output", str(analysis)])
    return subprocess.run(analysis_command, cwd=ROOT, check=False).returncode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse controller options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root-file", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--image-manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument(
        "--seeds", nargs="+", help="Seeds to queue (comma-separated values accepted); defaults to 1 2 3 4."
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--finalize", action="store_true", help="Explicitly build registry and routing analysis.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the serial formal queue, stopping immediately on any unsafe condition."""
    args = parse_args(argv)
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    root = formal_root(args.formal_root_file)
    requested = parse_seeds(args.seeds)
    if not args.validate_only and any(
        value is None for value in (args.image_manifest, args.data_root, args.dataset_manifest)
    ):
        raise ValueError("--image-manifest, --data-root, and --dataset-manifest are required for launch or dry-run")

    def command_for(seed: int) -> list[str]:
        return runner_command(
            root,
            seed,
            data=args.data,
            image_manifest=args.image_manifest,
            data_root=args.data_root,
            dataset_manifest=args.dataset_manifest,
        )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "formal_root": str(root),
                    "seeds": requested,
                    "commands": [command_for(seed) for seed in requested],
                },
                indent=2,
            )
        )
        return 0

    hashes: set[str] = set()
    if args.validate_only:
        for seed in requested:
            _, error = validate_seed(root, seed, hashes)
            if error:
                print(error, file=sys.stderr)
                return 1
        print(f"validated seeds={requested}")
        return 0
    if 0 not in requested:
        _, error = validate_seed(root, 0, hashes)
        if error:
            print(error, file=sys.stderr)
            return 1

    lock = acquire_lock(root)
    try:
        signal.signal(signal.SIGINT, stop_handler)
        signal.signal(signal.SIGTERM, stop_handler)
        write_state(root, status="starting", current_seed=None, detail=None)
        append_log(root, f"queue started seeds={requested}")
        for seed in requested:
            _, existing_error = validate_seed(root, seed, hashes)
            if existing_error is None:
                append_log(root, f"seed {seed} already passed; skipped")
                continue
            if experiment_root(root, seed).exists():
                raise RuntimeError(f"seed {seed} exists but is incomplete or failed: {existing_error}")
            issue = preflight(root)
            if issue:
                raise RuntimeError(f"seed {seed} preflight failed: {issue}")
            write_state(root, status="running", current_seed=seed, detail="launching runner")
            append_log(root, f"seed {seed} launched")
            exit_code = run_seed(root, seed, args.poll_seconds, command_for(seed))
            if exit_code != 0:
                raise RuntimeError(f"seed {seed} runner exited with code {exit_code}")
            _, error = validate_seed(root, seed, hashes)
            if error:
                raise RuntimeError(error)
            append_log(root, f"seed {seed} passed")
        detail = "all requested seeds passed"
        if args.finalize:
            final_hashes: set[str] = set()
            for seed in SEEDS:
                _, error = validate_seed(root, seed, final_hashes)
                if error:
                    raise RuntimeError(error)
            write_state(root, status="finalizing", current_seed=None, detail="explicit finalization requested")
            exit_code = finalize(root)
            if exit_code:
                raise RuntimeError(f"registry or analysis exited with code {exit_code}")
            detail = "registry and analysis completed"
        write_state(root, status="passed", current_seed=None, detail=detail)
        append_log(root, "queue completed")
        return 0
    except (OSError, RuntimeError, SchemaValidationError, ValueError) as error:
        write_state(root, status="failed", current_seed=None, detail=str(error))
        append_log(root, f"queue failed: {error}")
        print(error, file=sys.stderr)
        return 1
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
