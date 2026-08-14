"""Run the bounded Issue #54 Phase 2.2 overnight pilot on one CUDA GPU.

The controller is self-contained once launched on the remote host. It owns
training subprocesses, time admission, heartbeat/state files, routing exports,
checkpoint indexing, and final compact reports. Results are diagnostic pilot
evidence, not a formal multi-seed conclusion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SMOKE = importlib.import_module("scripts.issue54.run_phase2_one_hour_smoke")
DIAGNOSTIC_LABEL = _SMOKE.DIAGNOSTIC_LABEL
IMAGE_SUFFIXES = _SMOKE.IMAGE_SUFFIXES
compare_routes = _SMOKE.compare_routes
create_dataset_snapshot = _SMOKE.create_dataset_snapshot
discover_official_configs = _SMOKE.discover_official_configs
export_routing = _SMOKE.export_routing
parse_last_results_row = _SMOKE.parse_last_results_row
sha256_file = _SMOKE.sha256_file

ALLOWED_ROOT = Path("/root/autodl-tmp/MoT").resolve()
ALLOWED_RESULTS_ROOT = ALLOWED_ROOT / "results"
ALLOWED_STATUSES = {
    "passed_pilot",
    "failed",
    "interrupted_with_checkpoint",
    "not_started",
    "not_started_insufficient_time",
    "implementation_failed",
}
TRANSIENT_FAILURE_MARKERS = (
    "resource temporarily unavailable",
    "connection reset",
    "connection aborted",
    "dataloader worker",
    "worker exited unexpectedly",
    "timed out",
)
REPORT_NAMES = {
    "markdown": "PHASE2_2_OVERNIGHT_REPORT.md",
    "csv": "PHASE2_2_OVERNIGHT_RUNS.csv",
    "manifest": "PHASE2_2_OVERNIGHT_MANIFEST.json",
}
PHASE2_3_REPORT_NAMES = {
    "markdown": "PHASE2_3_CONTROLLER_FIX_REPORT.md",
    "csv": "PHASE2_3_CALIBRATION_RUNS.csv",
    "manifest": "PHASE2_3_MANIFEST.json",
}
RECOVERED_MOT_CHECKPOINT_SHA256 = {
    "calibration_mot_amp": "70ff9e97539a772cb009539f16159d7eb9a623d1a8df28c8f5ee68af2ab19b8b",
    "calibration_mot_fp32": "d2d81d184519ca0dc14a78b93e1c2ca310a6aeb7557cbe505639842903d99361",
}
EXPECTED_SOURCE_MANIFEST_SHA256 = "1781db149db322a2e18704fda13dea01df84c28893b975074db9762346fc36a3"
EXPECTED_DATASET_INVENTORY_SHA256 = "4a7a03b08cf21d913ab85a86b40a75eee13579881fba9bc0d979b72b7f0a96fa"
MINIMUM_FREE_DISK_BYTES = 20 * 1024**3
BOOTSTRAP_CALIBRATION_ESTIMATE_SECONDS = 3600.0


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_text(path: Path, text: str) -> None:
    """Atomically replace a UTF-8 text artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    """Atomically write deterministic JSON."""
    atomic_text(path, json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def ensure_under(path: Path, root: Path) -> Path:
    """Resolve and require a path to be a strict child of root."""
    resolved = path.resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"path must be a unique child of {root}: {resolved}")
    return resolved


def file_sha256(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_snapshot() -> dict[str, Any]:
    """Collect the execution environment without changing it."""
    import platform

    import torch

    gpu_query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=20,
    ).strip()
    gpu_name, gpu_memory, driver = [part.strip() for part in gpu_query.split(",")]
    return {
        "hostname": platform.node(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": {
            "name": gpu_name,
            "memory_total_mib": int(gpu_memory),
            "driver": driver,
        },
    }


def checkpoint_info(run_dir: Path) -> dict[str, Any] | None:
    """Return the best available checkpoint metadata."""
    candidates = (
        ("best", run_dir / "weights/best.pt"),
        ("last", run_dir / "weights/last.pt"),
        ("last_healthy", run_dir / "weights/last_healthy.pt"),
    )
    for kind, path in candidates:
        if path.is_file():
            return {
                "kind": kind,
                "path": path.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    epoch_checkpoints = sorted((run_dir / "weights").glob("epoch*.pt")) if (run_dir / "weights").is_dir() else []
    if epoch_checkpoints:
        path = epoch_checkpoints[-1]
        return {
            "kind": "latest_epoch",
            "path": path.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return None


def contains_non_finite(values: list[Any]) -> bool:
    """Return whether any present scalar is NaN or infinite."""
    for value in values:
        if value is None or value == "":
            continue
        try:
            if not math.isfinite(float(value)):
                return True
        except (TypeError, ValueError):
            continue
    return False


def merge_training_payload(
    payload: dict[str, Any],
    parsed: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Merge parsed metrics and controller metadata into one child payload."""
    payload.update(parsed)
    payload.update(metadata)
    return payload


def recover_training_payload(
    run_dir: Path,
    *,
    expected_epochs: int,
    expected_amp: bool,
    expected_checkpoint_sha256: str | None = None,
    checkpoint_loader: Any | None = None,
) -> dict[str, Any]:
    """Reconstruct a completed child payload without launching training."""
    import yaml

    payload: dict[str, Any] = {
        "evidence_label": DIAGNOSTIC_LABEL,
        "recovered_without_training": True,
        "status": "failed",
        "failure_reason": None,
        "checkpoint_loadable": False,
        "nan_or_inf": False,
    }
    results_path = run_dir / "results.csv"
    args_path = run_dir / "args.yaml"
    parsed = parse_last_results_row(results_path)
    actual_args = yaml.safe_load(args_path.read_text(encoding="utf-8")) if args_path.is_file() else {}
    checkpoint = checkpoint_info(run_dir)
    completed = int(float(parsed["epoch"])) if parsed.get("epoch") is not None else 0
    raw = parsed.get("raw_last_row", {})
    nan_or_inf = contains_non_finite(
        [
            parsed.get("map50"),
            parsed.get("map50_95"),
            parsed.get("loss"),
            *raw.values(),
        ]
    )
    failures = []
    if completed != expected_epochs:
        failures.append(f"expected {expected_epochs} completed epochs, found {completed}")
    if int(actual_args.get("epochs", -1)) != expected_epochs:
        failures.append(f"args.yaml epochs mismatch: {actual_args.get('epochs')}")
    if int(actual_args.get("batch", -1)) != 8:
        failures.append(f"args.yaml batch mismatch: {actual_args.get('batch')}")
    if int(actual_args.get("imgsz", -1)) != 640:
        failures.append(f"args.yaml imgsz mismatch: {actual_args.get('imgsz')}")
    if bool(actual_args.get("cache")):
        failures.append("args.yaml cache must be false")
    if bool(actual_args.get("amp")) != expected_amp:
        failures.append(f"args.yaml amp mismatch: {actual_args.get('amp')}")
    if checkpoint is None:
        failures.append("no checkpoint found")
    elif checkpoint["kind"] != "best":
        failures.append(f"expected best checkpoint, found {checkpoint['kind']}")
    elif expected_checkpoint_sha256 and checkpoint["sha256"] != expected_checkpoint_sha256:
        failures.append(f"checkpoint SHA256 mismatch: {checkpoint['sha256']} != {expected_checkpoint_sha256}")
    if nan_or_inf:
        failures.append("non-finite training metrics")

    checkpoint_error = None
    if checkpoint and not failures:
        try:
            if checkpoint_loader is None:
                from ultralytics import YOLO

                checkpoint_loader = YOLO
            checkpoint_loader(checkpoint["path"])
            payload["checkpoint_loadable"] = True
        except Exception as error:  # noqa: BLE001  # Recovery evidence must preserve loader failures.
            checkpoint_error = f"{type(error).__name__}: {error}"
            failures.append(checkpoint_error)

    if failures and checkpoint and completed < expected_epochs:
        status = "interrupted_with_checkpoint"
    else:
        status = "failed" if failures else "passed_pilot"
    return merge_training_payload(
        payload,
        parsed,
        {
            "status": status,
            "failure_reason": "; ".join(failures) or None,
            "epochs_completed": completed,
            "requested_batch": 8,
            "actual_batch": int(actual_args.get("batch", 8)),
            "optimizer_actual": actual_args.get("optimizer", "auto"),
            "lr0_actual": actual_args.get("lr0", 0.01),
            "checkpoint": checkpoint,
            "checkpoint_error": checkpoint_error,
            "nan_or_inf": nan_or_inf,
            "args_path": args_path.as_posix(),
            "results_path": results_path.as_posix(),
        },
    )


def child_train(args: argparse.Namespace) -> int:
    """Run one isolated Ultralytics training process."""
    import torch
    import yaml

    from ultralytics import YOLO

    result_path = Path(args.child_result)
    run_dir = Path(args.project) / args.name
    started = time.monotonic()
    payload: dict[str, Any] = {
        "evidence_label": DIAGNOSTIC_LABEL,
        "pid": os.getpid(),
        "started_at": utc_now(),
        "requested_batch": 8,
        "actual_batch": 8,
    }
    try:
        torch.cuda.set_device(0)
        torch.cuda.init()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        model = YOLO(args.model)
        model.train(
            data=args.data,
            epochs=args.epochs,
            fraction=1.0,
            imgsz=640,
            batch=8,
            workers=8,
            seed=args.seed,
            deterministic=True,
            cache=False,
            device=0,
            amp=args.amp,
            optimizer="auto",
            lr0=0.01,
            pretrained=False,
            project=args.project,
            name=args.name,
            exist_ok=False,
            plots=False,
            verbose=True,
            save=True,
            save_period=1,
            patience=max(args.epochs + 1, 100),
        )
        peak_memory = int(torch.cuda.max_memory_allocated(0))
        optimizer = getattr(model.trainer, "optimizer", None)
        optimizer_name = type(optimizer).__name__ if optimizer is not None else None
        optimizer_lr0 = None
        if optimizer is not None and optimizer.param_groups:
            optimizer_lr0 = optimizer.param_groups[0].get(
                "initial_lr",
                optimizer.param_groups[0].get("lr"),
            )
        parsed = parse_last_results_row(run_dir / "results.csv")
        args_yaml = run_dir / "args.yaml"
        actual_args = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) if args_yaml.is_file() else {}
        checkpoint = checkpoint_info(run_dir)
        checkpoint_loadable = False
        checkpoint_error = None
        if checkpoint:
            try:
                del model
                torch.cuda.empty_cache()
                YOLO(checkpoint["path"])
                checkpoint_loadable = True
            except Exception as error:  # noqa: BLE001  # Diagnostic result, never swallowed.
                checkpoint_error = f"{type(error).__name__}: {error}"
        raw = parsed.get("raw_last_row", {})
        completed = int(float(parsed["epoch"])) if parsed.get("epoch") is not None else 0
        merge_training_payload(
            payload,
            parsed,
            {
                "status": "passed_pilot" if checkpoint_loadable else "failed",
                "failure_reason": checkpoint_error,
                "peak_gpu_memory_bytes": peak_memory,
                "optimizer_actual": optimizer_name or actual_args.get("optimizer"),
                "lr0_actual": optimizer_lr0 if optimizer_lr0 is not None else actual_args.get("lr0"),
                "epochs_completed": completed,
                "checkpoint_loadable": checkpoint_loadable,
                "nan_or_inf": contains_non_finite(
                    [
                        parsed.get("map50"),
                        parsed.get("map50_95"),
                        parsed.get("loss"),
                        *raw.values(),
                    ]
                ),
            },
        )
        return 0 if payload["status"] == "passed_pilot" and not payload["nan_or_inf"] else 1
    except BaseException as error:
        payload.update(
            {
                "status": "failed",
                "failure_reason": f"{type(error).__name__}: {error}",
                "peak_gpu_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(0)) if torch.cuda.is_available() else None
                ),
                "checkpoint_loadable": False,
            }
        )
        raise
    finally:
        payload["ended_at"] = utc_now()
        payload["duration_seconds"] = time.monotonic() - started
        result_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(result_path, payload)


class OvernightController:
    """Bounded single-GPU experiment controller."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output = ensure_under(Path(args.output), ALLOWED_RESULTS_ROOT)
        resume_from = getattr(args, "resume_from", None)
        self.resume_from = ensure_under(Path(resume_from), ALLOWED_RESULTS_ROOT) if resume_from else None
        self.report_names = PHASE2_3_REPORT_NAMES if self.resume_from else REPORT_NAMES
        self.data_root = Path(args.data_root).resolve()
        self.source_manifest_path = Path(args.source_manifest).resolve()
        self.started_monotonic = time.monotonic()
        self.hard_deadline = self.started_monotonic + args.budget_seconds
        self.training_cutoff = self.started_monotonic + args.no_new_training_seconds
        self.started_at = utc_now()
        self.expected_end_at = (
            (datetime.now(timezone.utc) + timedelta(seconds=args.budget_seconds)).isoformat().replace("+00:00", "Z")
        )
        self.runs: list[dict[str, Any]] = []
        self.routing: list[dict[str, Any]] = []
        self.precision_selection: dict[str, Any] = {}
        self.admission_decisions: list[dict[str, Any]] = []
        self.current_task = "initializing"
        self.stop_heartbeat = threading.Event()
        self.heartbeat_thread: threading.Thread | None = None
        self.environment: dict[str, Any] = {}
        self.dataset: dict[str, Any] = {}
        self.configs: dict[str, Path] = {}
        self.git_commit = ""
        self.controller_failure: str | None = None
        self.route_comparison: dict[str, Any] = {"available": False, "reason": "not evaluated"}

    @property
    def running_path(self) -> Path:
        return self.output / "RUNNING.json"

    @property
    def heartbeat_path(self) -> Path:
        return self.output / "heartbeat.txt"

    @property
    def manifest_path(self) -> Path:
        return self.output / "manifest.json"

    def elapsed(self) -> float:
        return time.monotonic() - self.started_monotonic

    def remaining_to_cutoff(self) -> float:
        return max(self.training_cutoff - time.monotonic(), 0.0)

    def remaining_total(self) -> float:
        return max(self.hard_deadline - time.monotonic(), 0.0)

    def state_payload(self, status: str = "running") -> dict[str, Any]:
        """Build the live state payload."""
        return {
            "evidence_label": DIAGNOSTIC_LABEL,
            "status": status,
            "controller_pid": os.getpid(),
            "started_at": self.started_at,
            "expected_end_at": self.expected_end_at,
            "updated_at": utc_now(),
            "current_task": self.current_task,
            "elapsed_seconds": self.elapsed(),
            "remaining_total_seconds": self.remaining_total(),
            "remaining_training_admission_seconds": self.remaining_to_cutoff(),
            "runs_recorded": len(self.runs),
            "routing_exports_recorded": len(self.routing),
            "precision_selection": self.precision_selection,
            "admission_decisions": self.admission_decisions,
            "controller_failure": self.controller_failure,
        }

    def write_state(self, status: str = "running") -> None:
        atomic_json(self.running_path, self.state_payload(status))
        atomic_json(self.manifest_path, self.manifest_payload(status))

    def heartbeat_loop(self) -> None:
        """Update heartbeat once per minute until the bounded controller exits."""
        while not self.stop_heartbeat.is_set():
            atomic_text(
                self.heartbeat_path,
                "\n".join(
                    [
                        f"timestamp={utc_now()}",
                        f"controller_pid={os.getpid()}",
                        f"current_task={self.current_task}",
                        f"elapsed_seconds={self.elapsed():.1f}",
                        f"remaining_total_seconds={self.remaining_total():.1f}",
                    ]
                )
                + "\n",
            )
            self.stop_heartbeat.wait(60)

    def manifest_payload(self, status: str) -> dict[str, Any]:
        return {
            "evidence_label": DIAGNOSTIC_LABEL,
            "status": status,
            "git_commit": self.git_commit,
            "environment": self.environment,
            "dataset": self.dataset,
            "budget": {
                "total_seconds": self.args.budget_seconds,
                "no_new_training_after_seconds": self.args.no_new_training_seconds,
                "admission_buffer": 1.2,
                "uncalibrated_model_buffer": 1.3,
                "started_at": self.started_at,
                "expected_end_at": self.expected_end_at,
            },
            "precision_selection": self.precision_selection,
            "admission_decisions": self.admission_decisions,
            "runs": self.runs,
            "routing": self.routing,
            "controller_failure": self.controller_failure,
            "updated_at": utc_now(),
        }

    def initialize(self) -> None:
        """Create a new isolated output and validate immutable inputs."""
        if self.output.exists():
            existing = [item.relative_to(self.output).as_posix() for item in self.output.rglob("*") if item.is_file()]
            if any(item != "logs/controller.log" for item in existing):
                raise FileExistsError(f"refusing to reuse non-empty output: {self.output}")
        else:
            self.output.mkdir(parents=True)
        for name in ("logs", "checkpoints", "routing", "reports", "training"):
            (self.output / name).mkdir(exist_ok=True)
        self.git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if self.git_commit != self.args.expected_commit:
            raise RuntimeError(f"git commit mismatch: {self.git_commit} != {self.args.expected_commit}")
        if subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=False).returncode != 0:
            raise RuntimeError("tracked cloud worktree is dirty")
        self.environment = environment_snapshot()
        if (
            not self.environment["cuda_available"]
            or "4090" not in self.environment["gpu"]["name"]
            or self.environment["gpu"]["memory_total_mib"] < 23000
        ):
            raise RuntimeError(f"unexpected CUDA environment: {self.environment}")
        free_bytes = shutil.disk_usage(ALLOWED_ROOT).free
        if free_bytes < MINIMUM_FREE_DISK_BYTES:
            raise RuntimeError(f"insufficient free disk: {free_bytes} < {MINIMUM_FREE_DISK_BYTES}")
        self.environment["free_disk_bytes_at_start"] = free_bytes
        source_sha256 = sha256_file(self.source_manifest_path)
        if source_sha256 != EXPECTED_SOURCE_MANIFEST_SHA256:
            raise RuntimeError(
                f"source VisDrone manifest SHA changed: {source_sha256} != {EXPECTED_SOURCE_MANIFEST_SHA256}"
            )
        source_manifest = json.loads(self.source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest.get("splits", {}).get("train", {}).get("images") != 6471:
            raise RuntimeError("source VisDrone manifest train count changed")
        if source_manifest.get("splits", {}).get("val", {}).get("images") != 548:
            raise RuntimeError("source VisDrone manifest val count changed")
        self.configs = discover_official_configs(ROOT)
        config_snapshot = self.output / "configs/models"
        config_snapshot.mkdir(parents=True)
        for path in self.configs.values():
            shutil.copy2(path, config_snapshot / path.name)
        self.dataset = create_dataset_snapshot(self.data_root, self.output)
        if self.dataset["inventory_sha256"] != EXPECTED_DATASET_INVENTORY_SHA256:
            raise RuntimeError(
                f"VisDrone inventory SHA changed: {self.dataset['inventory_sha256']} != "
                f"{EXPECTED_DATASET_INVENTORY_SHA256}"
            )
        self.dataset.update(
            {
                "source_manifest_path": self.source_manifest_path.as_posix(),
                "source_manifest_sha256": source_sha256,
            }
        )
        self.current_task = "calibration_mot_amp"
        self.write_state()
        self.heartbeat_thread = threading.Thread(target=self.heartbeat_loop, name="issue54-heartbeat", daemon=True)
        self.heartbeat_thread.start()

    def run_command(
        self,
        *,
        experiment_id: str,
        model_variant: str,
        config: Path,
        precision: str,
        seed: int,
        epochs: int,
        phase: str,
        admission_estimate_seconds: float,
    ) -> dict[str, Any]:
        """Run one experiment with one narrowly gated transient retry."""
        amp = precision == "amp"
        attempts = []
        interrupted = False
        final_child: dict[str, Any] = {}
        final_run_dir: Path | None = None
        final_exit_code = 1
        total_started = time.monotonic()
        for retry_index in range(2):
            suffix = "" if retry_index == 0 else "_retry1"
            name = f"{experiment_id}{suffix}"
            run_dir = self.output / "training" / name
            child_result = self.output / "logs" / f"{name}.result.json"
            log_path = self.output / "logs" / f"{name}.log"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child-run",
                "--model",
                str(config),
                "--data",
                self.dataset["yaml_path"],
                "--project",
                str(self.output / "training"),
                "--name",
                name,
                "--epochs",
                str(epochs),
                "--seed",
                str(seed),
                "--amp",
                str(amp),
                "--child-result",
                str(child_result),
            ]
            started_at = utc_now()
            with log_path.open("x", encoding="utf-8") as log:
                log.write(f"# evidence_label={DIAGNOSTIC_LABEL}\n# command={command!r}\n# started_at={started_at}\n")
                log.flush()
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                attempt = {
                    "attempt": retry_index,
                    "command": command,
                    "pid": process.pid,
                    "started_at": started_at,
                    "log_path": log_path.relative_to(self.output).as_posix(),
                }
                while process.poll() is None:
                    if time.monotonic() >= self.hard_deadline:
                        interrupted = True
                        try:
                            os.killpg(process.pid, signal.SIGINT)
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=120)
                        except subprocess.TimeoutExpired:
                            process.terminate()
                            try:
                                process.wait(timeout=60)
                            except subprocess.TimeoutExpired:
                                process.kill()
                        break
                    time.sleep(30)
                    self.write_state()
                final_exit_code = process.returncode if process.returncode is not None else 1
                attempt.update({"ended_at": utc_now(), "exit_code": final_exit_code, "interrupted": interrupted})
                attempts.append(attempt)
            final_child = json.loads(child_result.read_text(encoding="utf-8")) if child_result.is_file() else {}
            final_run_dir = run_dir
            if final_exit_code == 0 and final_child.get("status") == "passed_pilot":
                break
            if interrupted:
                break
            log_tail = log_path.read_text(encoding="utf-8", errors="replace").lower()[-20000:]
            transient = any(marker in log_tail for marker in TRANSIENT_FAILURE_MARKERS)
            if retry_index == 0 and transient:
                attempts[-1]["retry_reason"] = "recognized transient environment/worker failure"
                retry_admitted = admission_estimate_seconds <= self.remaining_to_cutoff()
                self.admission_decisions.append(
                    {
                        "experiment_id": experiment_id,
                        "kind": "retry",
                        "predicted_seconds_with_buffers": admission_estimate_seconds,
                        "available_seconds": self.remaining_to_cutoff(),
                        "admitted": retry_admitted,
                        "timestamp": utc_now(),
                    }
                )
                self.write_state()
                if retry_admitted:
                    continue
            break

        checkpoint = checkpoint_info(final_run_dir) if final_run_dir else None
        status = final_child.get("status", "failed")
        if interrupted:
            status = "interrupted_with_checkpoint" if checkpoint else "failed"
        elif status == "passed_pilot" and final_child.get("nan_or_inf"):
            status = "failed"
        if status not in ALLOWED_STATUSES:
            status = "implementation_failed"
        if checkpoint:
            link = self.output / "checkpoints" / f"{experiment_id}_{checkpoint['kind']}.pt"
            if not link.exists():
                link.symlink_to(os.path.relpath(checkpoint["path"], link.parent))
            checkpoint["index_path"] = link.as_posix()
        training_time = final_child.get("training_time_seconds")
        completed = int(final_child.get("epochs_completed") or 0)
        seconds_per_epoch = float(training_time) / completed if training_time is not None and completed > 0 else None
        result = {
            "evidence_label": DIAGNOSTIC_LABEL,
            "experiment_id": experiment_id,
            "phase": phase,
            "model_variant": model_variant,
            "seed": seed,
            "precision": precision,
            "git_commit": self.git_commit,
            "config_path": config.relative_to(ROOT).as_posix(),
            "config_sha256": sha256_file(config),
            "dataset_manifest_path": self.dataset["manifest_path"],
            "dataset_manifest_sha256": self.dataset["manifest_sha256"],
            "epochs_requested": epochs,
            "epochs_completed": completed,
            "requested_batch": 8,
            "actual_batch": int(final_child.get("actual_batch", 8)),
            "imgsz": 640,
            "optimizer": final_child.get("optimizer_actual", "auto"),
            "lr0": final_child.get("lr0_actual", 0.01),
            "started_at": attempts[0]["started_at"],
            "ended_at": attempts[-1]["ended_at"],
            "seconds_per_epoch": seconds_per_epoch,
            "training_time_seconds": training_time,
            "total_duration_seconds": time.monotonic() - total_started,
            "peak_gpu_memory_bytes": final_child.get("peak_gpu_memory_bytes"),
            "loss": final_child.get("loss"),
            "map50": final_child.get("map50"),
            "map50_95": final_child.get("map50_95"),
            "nan_or_inf": bool(final_child.get("nan_or_inf", False)),
            "checkpoint_loadable": bool(final_child.get("checkpoint_loadable", False)),
            "checkpoint": checkpoint,
            "exit_code": final_exit_code,
            "status": status,
            "failure_reason": final_child.get("failure_reason"),
            "attempts": attempts,
        }
        self.runs.append(result)
        self.write_state()
        return result

    def recover_calibration(
        self,
        *,
        experiment_id: str,
        precision: str,
        config: Path,
    ) -> dict[str, Any]:
        """Recover one verified three-epoch MoT calibration without retraining it."""
        for run in self.runs:
            if run["experiment_id"] == experiment_id:
                return run
        if self.resume_from is None:
            raise RuntimeError("recovery requires --resume-from")
        source_run_dir = self.resume_from / "training" / experiment_id
        if not source_run_dir.is_dir():
            raise FileNotFoundError(source_run_dir)
        source_manifest_path = self.resume_from / "manifest.json"
        source_manifest = (
            json.loads(source_manifest_path.read_text(encoding="utf-8")) if source_manifest_path.is_file() else {}
        )
        source_run = next(
            (run for run in source_manifest.get("runs", []) if run.get("experiment_id") == experiment_id),
            {},
        )
        source_config_sha = source_run.get("config_sha256")
        current_config_sha = sha256_file(config)
        if source_config_sha and source_config_sha != current_config_sha:
            raise RuntimeError(f"recovered config SHA256 mismatch: {source_config_sha} != {current_config_sha}")
        source_inventory_sha = source_manifest.get("dataset", {}).get("inventory_sha256")
        if source_inventory_sha and source_inventory_sha != self.dataset["inventory_sha256"]:
            raise RuntimeError(
                f"recovered dataset inventory mismatch: {source_inventory_sha} != {self.dataset['inventory_sha256']}"
            )
        child = recover_training_payload(
            source_run_dir,
            expected_epochs=3,
            expected_amp=precision == "amp",
            expected_checkpoint_sha256=RECOVERED_MOT_CHECKPOINT_SHA256[experiment_id],
        )
        checkpoint = child.get("checkpoint")
        if checkpoint:
            link = self.output / "checkpoints" / f"{experiment_id}_{checkpoint['kind']}.pt"
            if not link.exists():
                link.symlink_to(os.path.relpath(checkpoint["path"], link.parent))
            checkpoint["index_path"] = link.as_posix()
        training_time = child.get("training_time_seconds")
        completed = int(child.get("epochs_completed") or 0)
        attempts = source_run.get("attempts") or []
        result = {
            "evidence_label": DIAGNOSTIC_LABEL,
            "experiment_id": experiment_id,
            "phase": "calibration",
            "model_variant": "mot",
            "seed": 0,
            "precision": precision,
            "git_commit": self.git_commit,
            "config_path": config.relative_to(ROOT).as_posix(),
            "config_sha256": current_config_sha,
            "dataset_manifest_path": self.dataset["manifest_path"],
            "dataset_manifest_sha256": self.dataset["manifest_sha256"],
            "epochs_requested": 3,
            "epochs_completed": completed,
            "requested_batch": 8,
            "actual_batch": int(child.get("actual_batch", 8)),
            "imgsz": 640,
            "optimizer": child.get("optimizer_actual", "auto"),
            "lr0": child.get("lr0_actual", 0.01),
            "started_at": source_run.get("started_at"),
            "ended_at": source_run.get("ended_at"),
            "seconds_per_epoch": (
                float(training_time) / completed if training_time is not None and completed > 0 else None
            ),
            "training_time_seconds": training_time,
            "total_duration_seconds": 0.0,
            "historical_total_duration_seconds": source_run.get("total_duration_seconds"),
            "peak_gpu_memory_bytes": source_run.get("peak_gpu_memory_bytes"),
            "loss": child.get("loss"),
            "map50": child.get("map50"),
            "map50_95": child.get("map50_95"),
            "nan_or_inf": bool(child.get("nan_or_inf", False)),
            "checkpoint_loadable": bool(child.get("checkpoint_loadable", False)),
            "checkpoint": checkpoint,
            "exit_code": 0 if child["status"] == "passed_pilot" else 1,
            "status": child["status"],
            "failure_reason": child.get("failure_reason"),
            "attempts": attempts,
            "recovered_without_training": True,
            "source_run_dir": source_run_dir.as_posix(),
        }
        self.runs.append(result)
        self.write_state()
        return result

    def record_not_started_formal_runs(self, selected_precision: str) -> None:
        """Record every explicitly prohibited formal run without launching it."""
        specs = (
            ("mot_seed0_30e", "mot", selected_precision, 0),
            ("esmoe_seed0_30e", "esmoe", "amp", 0),
            ("moa_seed0_30e", "moa", "amp", 0),
            ("mot_seed1_30e", "mot", selected_precision, 1),
        )
        existing = {run["experiment_id"] for run in self.runs}
        for experiment_id, variant, precision, seed in specs:
            if experiment_id in existing:
                continue
            self.runs.append(
                {
                    "evidence_label": DIAGNOSTIC_LABEL,
                    "experiment_id": experiment_id,
                    "phase": "formal_long_run",
                    "model_variant": variant,
                    "seed": seed,
                    "precision": precision,
                    "git_commit": self.git_commit,
                    "config_path": self.configs[variant].relative_to(ROOT).as_posix(),
                    "config_sha256": sha256_file(self.configs[variant]),
                    "dataset_manifest_path": self.dataset["manifest_path"],
                    "dataset_manifest_sha256": self.dataset["manifest_sha256"],
                    "epochs_requested": 30,
                    "epochs_completed": 0,
                    "requested_batch": 8,
                    "actual_batch": None,
                    "imgsz": 640,
                    "optimizer": "auto",
                    "lr0": 0.01,
                    "status": "not_started",
                    "failure_reason": "formal training is outside Phase 2.3 authorization",
                    "checkpoint": None,
                    "exit_code": None,
                }
            )
        self.write_state()

    def route_checkpoint(self, run: dict[str, Any]) -> dict[str, Any]:
        """Export and validate the fixed 32-image MoT routing sample."""
        self.current_task = f"routing_{run['experiment_id']}"
        self.write_state()
        try:
            checkpoint = dict(run["checkpoint"])
            if checkpoint.get("index_path"):
                checkpoint["path"] = checkpoint["index_path"]
            route_run = {
                "experiment_id": run["experiment_id"],
                "model_variant": "mot",
                "epochs": run["epochs_requested"],
                "requested_batch": run["requested_batch"],
                "actual_batch": run["actual_batch"],
                "optimizer": run["optimizer"],
                "precision_mode": run["precision"],
                "config_path": run["config_path"],
                "config_sha256": run["config_sha256"],
                "checkpoint": checkpoint,
            }
            result = export_routing(
                route_run,
                output=self.output,
                data_root=self.data_root,
                dataset_manifest=self.dataset,
                git_commit=self.git_commit,
            )
            result.update(
                {
                    "status": "passed_pilot",
                    "seed": run["seed"],
                    "precision": run["precision"],
                }
            )
        except BaseException as error:  # noqa: BLE001
            result = {
                "experiment_id": run["experiment_id"],
                "status": "implementation_failed",
                "failure_reason": f"{type(error).__name__}: {error}",
                "seed": run["seed"],
                "precision": run["precision"],
                "duration_seconds": 0.0,
            }
        self.routing.append(result)
        self.write_state()
        return result

    @staticmethod
    def route_is_stable(route: dict[str, Any] | None) -> bool:
        return bool(
            route
            and route.get("status") == "passed_pilot"
            and route.get("deterministic_repeats")
            and route.get("hooks_cleaned")
            and route.get("probabilities_valid")
        )

    def select_precision(
        self,
        amp_run: dict[str, Any],
        fp32_run: dict[str, Any],
        amp_route: dict[str, Any] | None,
        fp32_route: dict[str, Any] | None,
    ) -> str | None:
        """Apply the predeclared AMP/FP32 stability and 10% speed rule."""
        amp_stable = bool(
            amp_run["status"] == "passed_pilot"
            and not amp_run["nan_or_inf"]
            and amp_run["checkpoint_loadable"]
            and self.route_is_stable(amp_route)
        )
        fp32_stable = bool(
            fp32_run["status"] == "passed_pilot"
            and not fp32_run["nan_or_inf"]
            and fp32_run["checkpoint_loadable"]
            and self.route_is_stable(fp32_route)
        )
        amp_speed = amp_run.get("seconds_per_epoch")
        fp32_speed = fp32_run.get("seconds_per_epoch")
        speed_ratio = amp_speed / fp32_speed if amp_speed and fp32_speed else None
        if not fp32_stable:
            selected = None
            reason = "FP32 calibration/routing was not stable; protocol calibration is incomplete"
        elif not amp_stable:
            selected = "fp32"
            reason = "AMP failed a stability, finiteness, checkpoint, or routing determinism gate"
        elif speed_ratio is not None and speed_ratio <= 0.9:
            selected = "amp"
            reason = "AMP was at least 10% faster and passed all stability/routing gates"
        else:
            selected = "fp32"
            reason = "AMP speed advantage was below 10% or AMP was slower"
        self.precision_selection = {
            "selected": selected,
            "reason": reason,
            "threshold_amp_max_speed_ratio": 0.9,
            "amp_stable": amp_stable,
            "fp32_stable": fp32_stable,
            "amp_seconds_per_epoch": amp_speed,
            "fp32_seconds_per_epoch": fp32_speed,
            "amp_over_fp32_speed_ratio": speed_ratio,
            "same_seed_independent_runs": False,
            "interpretation": "precision calibration only; AMP and FP32 are not independent seeds",
        }
        self.write_state()
        return selected

    def skipped_run(
        self,
        *,
        experiment_id: str,
        model_variant: str,
        seed: int,
        precision: str,
        config: Path,
        predicted_seconds: float,
        reason: str,
    ) -> dict[str, Any]:
        """Record a long run rejected by bounded time admission."""
        result = {
            "evidence_label": DIAGNOSTIC_LABEL,
            "experiment_id": experiment_id,
            "phase": "long_pilot",
            "model_variant": model_variant,
            "seed": seed,
            "precision": precision,
            "git_commit": self.git_commit,
            "config_path": config.relative_to(ROOT).as_posix(),
            "config_sha256": sha256_file(config),
            "dataset_manifest_path": self.dataset["manifest_path"],
            "dataset_manifest_sha256": self.dataset["manifest_sha256"],
            "epochs_requested": 30,
            "epochs_completed": 0,
            "requested_batch": 8,
            "actual_batch": None,
            "imgsz": 640,
            "optimizer": "auto",
            "lr0": 0.01,
            "predicted_seconds_with_buffers": predicted_seconds,
            "available_admission_seconds": self.remaining_to_cutoff(),
            "status": "not_started_insufficient_time",
            "failure_reason": reason,
            "checkpoint": None,
            "exit_code": None,
            "nan_or_inf": False,
        }
        self.runs.append(result)
        self.write_state()
        return result

    def admit_training(self, experiment_id: str, predicted_seconds: float) -> bool:
        """Record and return a bounded training admission decision."""
        available = self.remaining_to_cutoff()
        admitted = bool(time.monotonic() < self.training_cutoff and predicted_seconds <= available)
        self.admission_decisions.append(
            {
                "experiment_id": experiment_id,
                "kind": "initial",
                "predicted_seconds_with_buffers": predicted_seconds,
                "available_seconds": available,
                "admitted": admitted,
                "timestamp": utc_now(),
            }
        )
        self.write_state()
        return admitted

    def compare_mot_seeds(self, routes: list[dict[str, Any]]) -> dict[str, Any]:
        """Compare two completed MoT seeds by name-aligned fixed routing rows."""
        if len(routes) != 2:
            return {
                "available": False,
                "reason": "two successful long-run MoT routing exports were not available",
            }

        def load_first(path: str) -> dict[tuple[str, str], dict[str, Any]]:
            rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
            return {(row["image_id"], row["layer_name"]): row for row in rows if row["inference_repeat"] == 0}

        def named_probs(row: dict[str, Any]) -> dict[str, float]:
            return dict(zip(row["expert_names"], row["expert_probabilities"]))

        def entropy(probabilities: dict[str, float]) -> float:
            return -sum(value * math.log(value) for value in probabilities.values() if value > 0)

        def jsd(left: dict[str, float], right: dict[str, float]) -> float:
            names = sorted(set(left) | set(right))
            midpoint = {name: (left.get(name, 0.0) + right.get(name, 0.0)) / 2 for name in names}

            def kl(probabilities: dict[str, float]) -> float:
                return sum(
                    value * math.log(value / midpoint[name]) for name, value in probabilities.items() if value > 0
                )

            return (kl(left) + kl(right)) / 2

        left, right = (load_first(route["jsonl_path"]) for route in routes)
        keys = sorted(set(left) & set(right))
        if not keys:
            return {"available": False, "reason": "no aligned image/layer rows"}
        left_probs = {key: named_probs(left[key]) for key in keys}
        right_probs = {key: named_probs(right[key]) for key in keys}
        by_layer: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for key in keys:
            by_layer[key[1]].append(key)
        utilization_rows = []
        layers = []
        for layer_name, layer_keys in sorted(by_layer.items()):
            left_counts = Counter(left[key]["selected_expert"] for key in layer_keys)
            right_counts = Counter(right[key]["selected_expert"] for key in layer_keys)
            expert_names = sorted(set().union(*(set(left_probs[key]) | set(right_probs[key]) for key in layer_keys)))
            for expert_name in expert_names:
                left_usage = left_counts[expert_name] / len(layer_keys)
                right_usage = right_counts[expert_name] / len(layer_keys)
                utilization_rows.append(
                    {
                        "layer_name": layer_name,
                        "expert_name": expert_name,
                        "seed0_top1_utilization": left_usage,
                        "seed1_top1_utilization": right_usage,
                        "difference_seed0_minus_seed1": left_usage - right_usage,
                    }
                )
            left_entropy = sum(entropy(left_probs[key]) for key in layer_keys) / len(layer_keys)
            right_entropy = sum(entropy(right_probs[key]) for key in layer_keys) / len(layer_keys)
            layers.append(
                {
                    "layer_name": layer_name,
                    "top1_agreement": sum(
                        left[key]["selected_expert"] == right[key]["selected_expert"] for key in layer_keys
                    )
                    / len(layer_keys),
                    "mean_jsd": sum(jsd(left_probs[key], right_probs[key]) for key in layer_keys) / len(layer_keys),
                    "mean_entropy_seed0": left_entropy,
                    "mean_entropy_seed1": right_entropy,
                    "entropy_difference_seed0_minus_seed1": left_entropy - right_entropy,
                }
            )
        comparison = {
            "available": True,
            "aligned_image_layer_rows": len(keys),
            "top1_agreement": sum(left[key]["selected_expert"] == right[key]["selected_expert"] for key in keys)
            / len(keys),
            "mean_jsd": sum(jsd(left_probs[key], right_probs[key]) for key in keys) / len(keys),
            "mean_entropy_seed0": sum(entropy(left_probs[key]) for key in keys) / len(keys),
            "mean_entropy_seed1": sum(entropy(right_probs[key]) for key in keys) / len(keys),
            "layers": layers,
            "interpretation": "two-seed pilot only; insufficient for a formal population-level conclusion",
        }
        atomic_json(self.output / "routing/mot_seed0_seed1_comparison.json", comparison)
        utilization_path = self.output / "routing/mot_seed0_seed1_utilization.csv"
        with utilization_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(utilization_rows[0]))
            writer.writeheader()
            writer.writerows(utilization_rows)
        comparison["utilization_csv"] = utilization_path.as_posix()
        return comparison

    def render_reports(self, final_status: str, seed_comparison: dict[str, Any]) -> None:
        """Render compact machine-readable and Markdown summaries."""
        gpu_seconds = sum(
            float(run.get("total_duration_seconds") or 0)
            for run in self.runs
            if run.get("status") in {"passed_pilot", "failed", "interrupted_with_checkpoint"}
        ) + sum(float(route.get("duration_seconds") or 0) for route in self.routing)
        completed_long = [
            run for run in self.runs if run.get("phase") == "long_pilot" and run["status"] == "passed_pilot"
        ]
        passed_mot_ids = {run["experiment_id"] for run in completed_long if run["model_variant"] == "mot"}
        stable_long_route_ids = {
            route["experiment_id"]
            for route in self.routing
            if route.get("experiment_id") in passed_mot_ids and self.route_is_stable(route)
        }
        if self.resume_from:
            calibrations = {run["experiment_id"]: run for run in self.runs if run.get("phase") == "calibration"}
            stable_route_ids = {route["experiment_id"] for route in self.routing if self.route_is_stable(route)}
            pilot_ready = bool(
                self.precision_selection.get("selected")
                and all(
                    calibrations.get(experiment_id, {}).get("status") == "passed_pilot"
                    for experiment_id in (
                        "calibration_mot_amp",
                        "calibration_mot_fp32",
                        "calibration_esmoe_amp",
                        "calibration_moa_amp",
                    )
                )
                and {"calibration_mot_amp", "calibration_mot_fp32"} <= stable_route_ids
                and self.route_comparison.get("available")
            )
        else:
            pilot_ready = bool(
                self.precision_selection.get("selected")
                and "mot_seed0_30e" in passed_mot_ids
                and "mot_seed0_30e" in stable_long_route_ids
            )
        final_manifest = self.manifest_payload(final_status)
        final_manifest.update(
            {
                "ended_at": utc_now(),
                "wall_seconds": self.elapsed(),
                "gpu_process_and_routing_seconds": gpu_seconds,
                "mot_seed_comparison": seed_comparison,
                "precision_route_comparison": self.route_comparison,
                "ready_for_formal_mvp": pilot_ready,
                "formal_protocol_frozen": bool(self.resume_from and pilot_ready),
                "formal_evidence": False,
            }
        )
        atomic_json(self.output / "reports" / self.report_names["manifest"], final_manifest)
        fields = [
            "experiment_id",
            "phase",
            "model_variant",
            "seed",
            "precision",
            "epochs_requested",
            "epochs_completed",
            "requested_batch",
            "actual_batch",
            "seconds_per_epoch",
            "total_duration_seconds",
            "peak_gpu_memory_bytes",
            "loss",
            "map50",
            "map50_95",
            "nan_or_inf",
            "exit_code",
            "status",
            "failure_reason",
            "checkpoint_path",
            "checkpoint_size_bytes",
            "checkpoint_sha256",
        ]
        csv_path = self.output / "reports" / self.report_names["csv"]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for run in self.runs:
                checkpoint = run.get("checkpoint") or {}
                row = {field: run.get(field) for field in fields}
                row.update(
                    {
                        "checkpoint_path": checkpoint.get("path"),
                        "checkpoint_size_bytes": checkpoint.get("size_bytes"),
                        "checkpoint_sha256": checkpoint.get("sha256"),
                    }
                )
                writer.writerow(row)
        lines = [
            (
                "# Issue #54 Phase 2.3 Controller Recovery and Calibration"
                if self.resume_from
                else "# Issue #54 Phase 2.2 Overnight Pilot"
            ),
            "",
            f"Evidence classification: `{DIAGNOSTIC_LABEL}`.",
            "",
            "This is a bounded long-run pilot, not a formal multi-seed conclusion.",
            "",
            "## Environment and protocol",
            "",
            f"- Git: `{self.git_commit}`",
            f"- GPU: `{self.environment.get('gpu', {})}`",
            f"- PyTorch/CUDA: `{self.environment.get('torch')}` / `{self.environment.get('torch_cuda')}`",
            f"- Dataset inventory: `{self.dataset.get('inventory_sha256')}`",
            f"- Wall time: `{self.elapsed():.1f}` seconds",
            f"- GPU process + routing time: `{gpu_seconds:.1f}` seconds",
            "",
            "## Precision selection",
            "",
            f"- Selected: `{self.precision_selection.get('selected')}`",
            f"- Reason: {self.precision_selection.get('reason')}",
            f"- Evidence: `{self.precision_selection}`",
            "",
            "## Runs",
            "",
            "| Run | Variant | Seed | Precision | Epochs | s/epoch | Peak GiB | mAP50 | mAP50-95 | Status |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---|",
        ]
        for run in self.runs:
            peak = run.get("peak_gpu_memory_bytes")
            peak_gib = "" if peak is None else f"{peak / 1024**3:.2f}"
            seconds = run.get("seconds_per_epoch")
            lines.append(
                f"| {run['experiment_id']} | {run['model_variant']} | {run['seed']} | "
                f"{run['precision']} | {run.get('epochs_completed', 0)}/{run['epochs_requested']} | "
                f"{'' if seconds is None else f'{seconds:.2f}'} | {peak_gib} | "
                f"{run.get('map50')} | {run.get('map50_95')} | {run['status']} |"
            )
        lines.extend(
            [
                "",
                "## Routing and conclusion",
                "",
                f"- Routing exports: `{self.routing}`",
                f"- Precision-route comparison: `{self.route_comparison}`",
                f"- Two-seed MoT comparison: `{seed_comparison}`",
                f"- Ready for an explicitly approved formal MVP: `{pilot_ready}`",
                "",
                (
                    "Metrics are pilot-only. Two seeds, if available, are still insufficient for a formal aggregate "
                    "claim. No formal 5-run protocol was launched."
                ),
                "",
            ]
        )
        atomic_text(self.output / "reports" / self.report_names["markdown"], "\n".join(lines))
        repo_report_dir = ROOT / "reports/issue54"
        for filename in self.report_names.values():
            destination = repo_report_dir / filename
            if destination.exists():
                raise FileExistsError(f"refusing to overwrite repository report: {destination}")
            shutil.copy2(self.output / "reports" / filename, destination)

    def execute_resume(self) -> int:
        """Recover MoT calibrations, validate routing, and calibrate EsMoE/MoA only."""
        final_status = "complete"
        seed_comparison: dict[str, Any] = {
            "available": False,
            "reason": "Phase 2.3 does not treat precision conditions as independent seeds",
        }
        try:
            self.initialize()
            mot_config = self.configs["mot"]
            amp_run = self.recover_calibration(
                experiment_id="calibration_mot_amp",
                precision="amp",
                config=mot_config,
            )
            amp_route = self.route_checkpoint(amp_run) if amp_run["status"] == "passed_pilot" else None
            fp32_run = self.recover_calibration(
                experiment_id="calibration_mot_fp32",
                precision="fp32",
                config=mot_config,
            )
            fp32_route = self.route_checkpoint(fp32_run) if fp32_run["status"] == "passed_pilot" else None
            if self.route_is_stable(amp_route) and self.route_is_stable(fp32_route):
                self.route_comparison = compare_routes(self.output, [amp_route, fp32_route])
            selected = self.select_precision(amp_run, fp32_run, amp_route, fp32_route)
            if selected is None:
                final_status = "failed"
                self.controller_failure = self.precision_selection["reason"]
            else:
                calibration_runs = []
                for experiment_id, variant in (
                    ("calibration_esmoe_amp", "esmoe"),
                    ("calibration_moa_amp", "moa"),
                ):
                    self.current_task = experiment_id
                    self.write_state()
                    calibration_runs.append(
                        self.run_command(
                            experiment_id=experiment_id,
                            model_variant=variant,
                            config=self.configs[variant],
                            precision="amp",
                            seed=0,
                            epochs=3,
                            phase="calibration",
                            admission_estimate_seconds=BOOTSTRAP_CALIBRATION_ESTIMATE_SECONDS,
                        )
                    )
                if any(run["status"] != "passed_pilot" for run in calibration_runs):
                    final_status = "failed"
                    self.controller_failure = "one or more EsMoE/MoA calibrations failed"
            self.record_not_started_formal_runs(selected or "unselected")
        except BaseException as error:  # noqa: BLE001
            final_status = "implementation_failed"
            self.controller_failure = f"{type(error).__name__}: {error}"
        finally:
            self.current_task = "finalizing_reports"
            try:
                self.render_reports(final_status, seed_comparison)
            except BaseException as error:  # noqa: BLE001
                final_status = "implementation_failed"
                report_error = f"{type(error).__name__}: {error}"
                self.controller_failure = (
                    f"{self.controller_failure}; report_error={report_error}"
                    if self.controller_failure
                    else f"report_error={report_error}"
                )
                atomic_json(
                    self.output / "reports/report_failure.json",
                    {
                        "evidence_label": DIAGNOSTIC_LABEL,
                        "error": report_error,
                        "controller_failure": self.controller_failure,
                    },
                )
            self.current_task = "complete" if final_status == "complete" else "failed"
            self.write_state(final_status)
            self.stop_heartbeat.set()
            if self.heartbeat_thread:
                self.heartbeat_thread.join(timeout=5)
            atomic_text(
                self.heartbeat_path,
                f"timestamp={utc_now()}\ncontroller_pid={os.getpid()}\ncurrent_task={self.current_task}\n",
            )
        return 0 if final_status == "complete" else 1

    def execute(self) -> int:
        """Run calibrations, choose precision, admit long runs, and finalize."""
        final_status = "complete"
        seed_comparison: dict[str, Any] = {"available": False, "reason": "not evaluated"}
        try:
            self.initialize()
            mot_config = self.configs["mot"]
            amp_calibration_estimate = BOOTSTRAP_CALIBRATION_ESTIMATE_SECONDS
            if not self.admit_training("calibration_mot_amp", amp_calibration_estimate):
                raise RuntimeError("insufficient time to start AMP calibration")
            amp_run = self.run_command(
                experiment_id="calibration_mot_amp",
                model_variant="mot",
                config=mot_config,
                precision="amp",
                seed=0,
                epochs=3,
                phase="calibration",
                admission_estimate_seconds=amp_calibration_estimate,
            )
            amp_route = self.route_checkpoint(amp_run) if amp_run["status"] == "passed_pilot" else None
            amp_epoch_seconds = amp_run.get("seconds_per_epoch")
            fp32_calibration_estimate = (
                amp_epoch_seconds * 3 * 1.2 if amp_epoch_seconds is not None else BOOTSTRAP_CALIBRATION_ESTIMATE_SECONDS
            )
            if not self.admit_training("calibration_mot_fp32", fp32_calibration_estimate):
                raise RuntimeError("insufficient time to start FP32 calibration")
            self.current_task = "calibration_mot_fp32"
            self.write_state()
            fp32_run = self.run_command(
                experiment_id="calibration_mot_fp32",
                model_variant="mot",
                config=mot_config,
                precision="fp32",
                seed=0,
                epochs=3,
                phase="calibration",
                admission_estimate_seconds=fp32_calibration_estimate,
            )
            fp32_route = self.route_checkpoint(fp32_run) if fp32_run["status"] == "passed_pilot" else None
            selected = self.select_precision(amp_run, fp32_run, amp_route, fp32_route)
            if selected is None:
                raise RuntimeError(self.precision_selection["reason"])
            selected_calibration = amp_run if selected == "amp" else fp32_run
            selected_epoch_seconds = selected_calibration.get("seconds_per_epoch")
            calibration_speeds = [
                run.get("seconds_per_epoch") for run in (amp_run, fp32_run) if run.get("seconds_per_epoch") is not None
            ]
            if not selected_epoch_seconds or not calibration_speeds:
                raise RuntimeError("calibration did not produce usable epoch timing")
            closest_conservative_speed = max(calibration_speeds)
            long_specs = (
                ("mot_seed0_30e", "mot", selected, 0),
                ("esmoe_seed0_30e", "esmoe", "amp", 0),
                ("moa_seed0_30e", "moa", "amp", 0),
                ("mot_seed1_30e", "mot", selected, 1),
            )
            long_mot_routes = []
            for experiment_id, variant, precision, seed in long_specs:
                self.current_task = f"admission_{experiment_id}"
                self.write_state()
                base_seconds = selected_epoch_seconds if variant == "mot" else closest_conservative_speed * 1.3
                predicted = base_seconds * 30 * 1.2
                if not self.admit_training(experiment_id, predicted):
                    self.skipped_run(
                        experiment_id=experiment_id,
                        model_variant=variant,
                        seed=seed,
                        precision=precision,
                        config=self.configs[variant],
                        predicted_seconds=predicted,
                        reason=(
                            f"predicted {predicted:.1f}s exceeds remaining training-admission "
                            f"budget {self.remaining_to_cutoff():.1f}s"
                        ),
                    )
                    continue
                self.current_task = experiment_id
                self.write_state()
                run = self.run_command(
                    experiment_id=experiment_id,
                    model_variant=variant,
                    config=self.configs[variant],
                    precision=precision,
                    seed=seed,
                    epochs=30,
                    phase="long_pilot",
                    admission_estimate_seconds=predicted,
                )
                if variant == "mot" and run["status"] == "passed_pilot":
                    route = self.route_checkpoint(run)
                    if self.route_is_stable(route):
                        long_mot_routes.append(route)
            seed_comparison = self.compare_mot_seeds(long_mot_routes)
        except BaseException as error:  # noqa: BLE001
            final_status = "implementation_failed"
            self.controller_failure = f"{type(error).__name__}: {error}"
        finally:
            self.current_task = "finalizing_reports"
            try:
                self.render_reports(final_status, seed_comparison)
            except BaseException as error:  # noqa: BLE001
                final_status = "implementation_failed"
                report_error = f"{type(error).__name__}: {error}"
                self.controller_failure = (
                    f"{self.controller_failure}; report_error={report_error}"
                    if self.controller_failure
                    else f"report_error={report_error}"
                )
                atomic_json(
                    self.output / "reports/report_failure.json",
                    {
                        "evidence_label": DIAGNOSTIC_LABEL,
                        "error": report_error,
                        "controller_failure": self.controller_failure,
                    },
                )
            self.current_task = "complete" if final_status == "complete" else "failed"
            self.write_state(final_status)
            self.stop_heartbeat.set()
            if self.heartbeat_thread:
                self.heartbeat_thread.join(timeout=5)
            atomic_text(
                self.heartbeat_path,
                f"timestamp={utc_now()}\ncontroller_pid={os.getpid()}\ncurrent_task={self.current_task}\n",
            )
        return 0 if final_status == "complete" else 1


def validate_only(args: argparse.Namespace) -> int:
    """Validate immutable paths, configs, data manifest, commit, and CUDA."""
    output = ensure_under(Path(args.output), ALLOWED_RESULTS_ROOT)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"validation output must not be a non-empty existing directory: {output}")
    data_root = Path(args.data_root).resolve()
    source_manifest = Path(args.source_manifest).resolve()
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    source_sha256 = sha256_file(source_manifest)
    if source_sha256 != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise RuntimeError(f"source manifest SHA mismatch: {source_sha256} != {EXPECTED_SOURCE_MANIFEST_SHA256}")
    if payload.get("splits", {}).get("train", {}).get("images") != 6471:
        raise RuntimeError("unexpected VisDrone train count")
    if payload.get("splits", {}).get("val", {}).get("images") != 548:
        raise RuntimeError("unexpected VisDrone val count")
    for split, expected in (("train", 6471), ("val", 548)):
        count = sum(
            path.suffix.lower() in IMAGE_SUFFIXES for path in (data_root / "images" / split).iterdir() if path.is_file()
        )
        if count != expected:
            raise RuntimeError(f"VisDrone {split} count mismatch: {count} != {expected}")
    configs = discover_official_configs(ROOT)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if commit != args.expected_commit:
        raise RuntimeError(f"git commit mismatch: {commit}")
    environment = environment_snapshot()
    if not environment["cuda_available"] or "4090" not in environment["gpu"]["name"]:
        raise RuntimeError(f"invalid CUDA environment: {environment}")
    free_bytes = shutil.disk_usage(ALLOWED_ROOT).free
    if free_bytes < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError(f"insufficient free disk: {free_bytes} < {MINIMUM_FREE_DISK_BYTES}")
    print(
        json.dumps(
            {
                "status": "validated",
                "output": output.as_posix(),
                "git_commit": commit,
                "environment": environment,
                "source_manifest_sha256": source_sha256,
                "free_disk_bytes": free_bytes,
                "configs": {name: path.relative_to(ROOT).as_posix() for name, path in configs.items()},
            },
            sort_keys=True,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    """Parse controller and child arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--data-root", default="/root/autodl-tmp/datasets/VisDrone")
    parser.add_argument(
        "--source-manifest",
        default=("/root/autodl-tmp/MoT/results/phase2_smoke_20260730T015316Z/configs/visdrone_manifest.json"),
    )
    parser.add_argument("--expected-commit", default="cf233abb630ab6490bb2fb7a47e4b80cb9ab4822")
    parser.add_argument("--budget-seconds", type=int, default=6 * 3600 + 20 * 60)
    parser.add_argument("--no-new-training-seconds", type=int, default=5 * 3600 + 50 * 60)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--resume-from",
        help="Recover completed MoT calibrations from an earlier result root and run Phase 2.3 only",
    )
    parser.add_argument("--child-run", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--data")
    parser.add_argument("--project")
    parser.add_argument("--name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--amp", type=lambda value: value.lower() == "true")
    parser.add_argument("--child-result")
    return parser.parse_args()


def main() -> int:
    """Dispatch validation, child training, or the overnight controller."""
    args = parse_args()
    if args.child_run:
        required = ("model", "data", "project", "name", "epochs", "seed", "amp", "child_result")
        missing = [field for field in required if getattr(args, field) is None]
        if missing:
            raise ValueError(f"child-run missing arguments: {missing}")
        return child_train(args)
    if not args.output:
        raise ValueError("--output is required")
    if args.budget_seconds <= 0 or args.no_new_training_seconds <= 0:
        raise ValueError("time budgets must be positive")
    if args.no_new_training_seconds >= args.budget_seconds:
        raise ValueError("no-new-training cutoff must precede total deadline")
    if args.validate_only:
        return validate_only(args)
    controller = OvernightController(args)
    return controller.execute_resume() if args.resume_from else controller.execute()


if __name__ == "__main__":
    raise SystemExit(main())
