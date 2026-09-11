"""Cross-platform lifecycle checks using disposable workers, never user processes."""

import importlib
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import psutil
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
jobs = importlib.import_module("runtime.cli.async_jobs")


def wait_terminal(manager, job_id):
    """Wait a bounded interval for the supervised worker's atomic result."""
    for _ in range(100):
        status = manager.status(job_id)
        if status["status"] in {"completed", "failed", "cancelled"}:
            return status
        time.sleep(0.05)
    pytest.fail(f"worker did not finish: {status}")


@pytest.mark.parametrize("code", [0, 7])
def test_worker_records_exit_code(tmp_path, monkeypatch, code):
    """Fast success and failure must not be overwritten by submission metadata."""
    dispatcher = tmp_path / "dispatcher.py"
    dispatcher.write_text(f"raise SystemExit({code})")
    monkeypatch.setattr(jobs, "DISPATCHER", dispatcher)
    monkeypatch.setattr(jobs, "ensure_manifest_dir", lambda request: tmp_path)
    manager = jobs.AsyncJobManager(tmp_path / "jobs")
    submitted = manager.submit("fixture", {"policy": {}})
    status = wait_terminal(manager, submitted["job_id"])
    assert status["returncode"] == code
    assert status["status"] == ("completed" if code == 0 else "failed")


def test_status_is_read_only_and_cancel_stops_worker(tmp_path, monkeypatch):
    """Windows queries must leave a running worker alive; cancellation is explicit."""
    dispatcher = tmp_path / "dispatcher.py"
    ready = tmp_path / "ready"
    dispatcher.write_text(f"import time\nfrom pathlib import Path\nPath({str(ready)!r}).touch()\ntime.sleep(60)")
    monkeypatch.setattr(jobs, "DISPATCHER", dispatcher)
    monkeypatch.setattr(jobs, "ensure_manifest_dir", lambda request: tmp_path)
    manager = jobs.AsyncJobManager(tmp_path / "jobs")
    job = manager.submit("fixture", {"policy": {}})
    try:
        for _ in range(100):
            if ready.exists():
                break
            time.sleep(0.05)
        assert ready.exists()
        with monkeypatch.context() as context:
            context.setattr(os, "kill", Mock(side_effect=AssertionError("status must not send signals")))
            assert manager.status(job["job_id"])["status"] == "running"
        assert psutil.Process(job["pid"]).is_running()
        assert manager.cancel(job["job_id"])["cancelled"]
        assert manager.status(job["job_id"])["status"] == "cancelled"
    finally:
        manager.cancel(job["job_id"])


@pytest.mark.parametrize("job_id", ["../escape", "/absolute", "C:\\escape", "a/b", "a\\b", ".", ""])
def test_invalid_job_id_cannot_create_paths(tmp_path, job_id):
    """Reject traversal before any mkdir or status read."""
    root = tmp_path / "jobs"
    with pytest.raises(ValueError):
        jobs.AsyncJobManager(root).status(job_id)
    assert not root.exists()


def test_missing_status_creates_nothing(tmp_path):
    """Existing safe legacy IDs remain queryable without creating directories."""
    root = tmp_path / "jobs"
    assert jobs.AsyncJobManager(root).status("missing-job-contract")["status"] == "missing"
    assert not root.exists()


@pytest.mark.parametrize("created", [None, -1])
def test_legacy_or_reused_pid_cannot_be_cancelled(tmp_path, created):
    """A PID alone is not enough to authorize termination, even when it exists."""
    manager = jobs.AsyncJobManager(tmp_path)
    path = manager._job_dir("fixture")
    path.mkdir()
    (path / "status.json").write_text(
        json.dumps({"status": "running", "pid": os.getpid(), "process_create_time": created})
    )
    assert manager.status("fixture")["status"] == "unknown"
    assert not manager.cancel("fixture")["cancelled"]


def test_access_denied_is_unknown(tmp_path, monkeypatch):
    """Permission failure is neither a successful exit nor a cancellable process."""
    manager = jobs.AsyncJobManager(tmp_path)
    path = manager._job_dir("fixture")
    path.mkdir()
    (path / "status.json").write_text(json.dumps({"status": "running", "pid": 123, "process_create_time": 1}))
    monkeypatch.setattr(jobs.psutil, "Process", Mock(side_effect=psutil.AccessDenied(123)))
    assert manager.status("fixture")["status"] == "unknown"
    assert not manager.cancel("fixture")["cancelled"]
