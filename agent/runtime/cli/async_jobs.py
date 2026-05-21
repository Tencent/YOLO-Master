from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from runtime.cli.contract import ensure_manifest_dir, json_safe, response, write_manifest


SKILL_ROOT = Path(__file__).resolve().parents[2]
DISPATCHER = SKILL_ROOT / "scripts" / "run_yolo_master_skill.py"


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
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _job_dir(self, job_id: str, request: dict[str, Any] | None = None) -> Path:
        path = self._jobs_dir(request) / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def submit(self, skill: str, request: dict[str, Any], callback_url: str | None = None) -> dict[str, Any]:
        job_id = uuid4().hex[:12]
        job_dir = self._job_dir(job_id, request)
        child_request = json_safe(request)
        child_request["request_id"] = f"{request.get('request_id', skill.replace('.', '-'))}-{job_id}"
        child_request.setdefault("policy", {})
        child_request["policy"]["async"] = False
        request_path = job_dir / "request.json"
        status_path = job_dir / "status.json"
        stdout_path = job_dir / "stdout.jsonl"
        stderr_path = job_dir / "stderr.log"
        request_path.write_text(json.dumps(child_request, ensure_ascii=False, indent=2), encoding="utf-8")

        stdout_handle = stdout_path.open("a", encoding="utf-8")
        stderr_handle = stderr_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, str(DISPATCHER), "--request", str(request_path)],
            cwd=Path(__file__).resolve().parents[3],
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        stdout_handle.close()
        stderr_handle.close()
        status = {
            "job_id": job_id,
            "skill": skill,
            "status": "running",
            "pid": proc.pid,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "request_path": str(request_path.resolve()),
            "stdout_path": str(stdout_path.resolve()),
            "stderr_path": str(stderr_path.resolve()),
            "callback_url": callback_url,
            "progress_path": str((ensure_manifest_dir(child_request) / "progress.jsonl").resolve()),
        }
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        return {**status, "status_path": str(status_path.resolve())}

    def status(self, job_id: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
        status_path = self._job_dir(job_id, request) / "status.json"
        if not status_path.exists():
            return {"job_id": job_id, "status": "missing"}
        status = json.loads(status_path.read_text(encoding="utf-8"))
        pid = status.get("pid")
        running = False
        if isinstance(pid, int):
            try:
                os.kill(pid, 0)
                running = True
            except OSError:
                running = False
        if not running and status.get("status") == "running":
            status["status"] = "completed"
            status["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        return status

    def cancel(self, job_id: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
        status = self.status(job_id, request)
        pid = status.get("pid")
        if status.get("status") != "running" or not isinstance(pid, int):
            return {**status, "cancelled": False}
        try:
            os.killpg(pid, signal.SIGTERM)
        except Exception:
            os.kill(pid, signal.SIGTERM)
        status["status"] = "cancelled"
        status["cancelled"] = True
        status["cancelled_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        status_path = self._job_dir(job_id, request) / "status.json"
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        return status


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
