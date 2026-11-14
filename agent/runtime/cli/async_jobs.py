from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import psutil

from .contract import (
    _atomic_write_json,
    ensure_manifest_dir,
    json_safe,
    redact_job_value,
    request_secret_values,
    response,
    write_manifest,
)

SKILL_ROOT = Path(__file__).resolve().parents[2]
DISPATCHER = SKILL_ROOT / "scripts" / "run_yolo_master_skill.py"
WORKER = SKILL_ROOT / "scripts" / "run_async_job.py"
EXECUTION_REQUEST_NAME = ".execution-request.json"


def async_requested(request: dict[str, Any]) -> bool:
    value = request.get("policy", {}).get("async")
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class AsyncJobManager:
    """Manage subprocess-backed long-running skill jobs."""

    def __init__(self, root: Path | None = None):
        self.root = root

    def _jobs_dir(self, request: dict[str, Any] | None = None) -> Path:
        base = self.root or SKILL_ROOT / "logs" / "async-jobs"
        return base.resolve()

    def _job_dir(self, job_id: str, request: dict[str, Any] | None = None) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", job_id):
            raise ValueError("Invalid job_id")
        base = self._jobs_dir(request)
        path = (base / job_id).resolve()
        if path.parent != base:
            raise ValueError("Job path escapes the jobs directory")
        return path

    def submit(self, skill: str, request: dict[str, Any], callback_url: str | None = None) -> dict[str, Any]:
        job_id = uuid4().hex[:12]
        job_dir = self._job_dir(job_id, request)
        job_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        job_dir.chmod(0o700)
        child_request = json_safe(request)
        child_request["request_id"] = f"{request.get('request_id', skill.replace('.', '-'))}-{job_id}"
        child_request.setdefault("policy", {})
        child_request["policy"]["async"] = False
        request_path = job_dir / "request.json"
        status_path = job_dir / "status.json"
        stdout_path = job_dir / "stdout.jsonl"
        stderr_path = job_dir / "stderr.log"
        execution_path = job_dir / EXECUTION_REQUEST_NAME
        secrets = request_secret_values(child_request)
        proc = None
        try:
            progress_path = str((ensure_manifest_dir(child_request) / "progress.jsonl").resolve())
            _atomic_write_json(request_path, redact_job_value(child_request, secrets))
            _atomic_write_json(execution_path, child_request)
            execution_path.chmod(0o600)
            with os.fdopen(
                os.open(stdout_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8"
            ) as stdout_handle, os.fdopen(
                os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8"
            ) as stderr_handle:
                proc = subprocess.Popen(
                    [sys.executable, str(WORKER), str(job_dir), str(DISPATCHER), str(execution_path)],
                    cwd=SKILL_ROOT.parent,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    start_new_session=True,
                )
            try:
                created = psutil.Process(proc.pid).create_time()
            except psutil.NoSuchProcess:
                created = None  # A fast worker may already have persisted its result.
            status = {
                "job_id": job_id,
                "skill": skill,
                "status": "running",
                "pid": proc.pid,
                "process_create_time": created,
                "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "request_path": str(request_path.resolve()),
                "stdout_path": str(stdout_path.resolve()),
                "stderr_path": str(stderr_path.resolve()),
                "callback_url": redact_job_value(callback_url, secrets),
                "progress_path": progress_path,
            }
            _atomic_write_json(status_path, status)
            return {**status, "status_path": str(status_path.resolve())}
        except BaseException:
            execution_path.unlink(missing_ok=True)
            if proc is not None:
                try:
                    self._terminate_process_tree(psutil.Process(proc.pid))
                except psutil.NoSuchProcess:
                    pass
            raise

    @staticmethod
    def _terminate_process_tree(process: psutil.Process) -> bool:
        """Stop an owned supervisor and its descendants before publishing cancellation."""
        children = process.children(recursive=True)
        for target in reversed(children):
            try:
                target.terminate()
            except psutil.NoSuchProcess:
                pass
        # Let the supervisor reap exited children before stopping the supervisor itself.
        _, alive = psutil.wait_procs(children, timeout=3)
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
        alive.append(process)
        for target in alive:
            try:
                target.kill()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(alive, timeout=3)
        return not alive

    @staticmethod
    def _process(status: dict[str, Any]) -> psutil.Process | None:
        """Find the recorded process without signaling it or accepting a reused PID."""
        pid, created = status.get("pid"), status.get("process_create_time")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or created is None:
            return None
        process = psutil.Process(pid)
        if process.create_time() != created or not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return None
        return process

    def status(self, job_id: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
        """Read worker results or probe process identity; reads never create files or send signals."""
        job_dir = self._job_dir(job_id, request)
        status_path = job_dir / "status.json"
        if not status_path.exists():
            return {"job_id": job_id, "status": "missing"}
        status = json.loads(status_path.read_text(encoding="utf-8"))
        for name in ("cancelled.json", "result.json"):
            result_path = job_dir / name
            if result_path.exists():
                return {**status, **json.loads(result_path.read_text(encoding="utf-8"))}
        try:
            running = self._process(status) is not None
        except psutil.NoSuchProcess:
            running = False
        except psutil.AccessDenied:
            return {**status, "status": "unknown"}
        # A missing process is not evidence of a successful task (including legacy records).
        return {**status, "status": "running" if running else "unknown"}

    def cancel(self, job_id: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
        """Terminate only the recorded worker and its descendants, then persist cancellation."""
        status = self.status(job_id, request)
        if status.get("status") != "running":
            return {**status, "cancelled": False}
        try:
            process = self._process(status)
            if process is None:
                return {**status, "status": "unknown", "cancelled": False}
            if not self._terminate_process_tree(process):
                return {**status, "status": "unknown", "cancelled": False}
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {**status, "status": "unknown", "cancelled": False}
        result = {"status": "cancelled", "cancelled": True, "cancelled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        (self._job_dir(job_id, request) / EXECUTION_REQUEST_NAME).unlink(missing_ok=True)
        _atomic_write_json(self._job_dir(job_id, request) / "cancelled.json", result)
        return {**status, **result}


def submit_async_skill(request: dict[str, Any]) -> dict[str, Any]:
    callback_url = request.get("policy", {}).get("callback_url") or request.get("runtime", {}).get("callback_url")
    job = AsyncJobManager().submit(str(request["skill"]), request, callback_url=callback_url)
    payload = response(
        request["skill"],
        "running",
        "asynchronous job submitted",
        job={
            "mode": "async",
            "job_id": job["job_id"],
            "pid": job["pid"],
            "status_path": job["status_path"],
            "progress_path": job["progress_path"],
            "stdout_path": job["stdout_path"],
            "stderr_path": job["stderr_path"],
        },
        next_actions=["yolo.job.status", "tail progress.jsonl"],
    )
    payload["manifest"] = str(write_manifest(request, payload))
    return payload
