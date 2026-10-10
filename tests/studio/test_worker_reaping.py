"""Real Linux zombies and adopted children must be reaped before releasing ownership."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import pytest

from core.schema import TERMINAL_STATUSES, JobRequest, JobStatus
from studio.jobs_manager import JobsManager
from studio.worker_runtime import ManagedWorker, _worker_main


def reaping_executor(job):
    """Publish the actual compute identity and a detached child, without importing YOLO."""
    root = Path(job.output.output_dir)
    if job.job_id == "queued":
        (root / "queued.started").touch()
        job.status = JobStatus.COMPLETED
        return job
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(120)",
        ],
        start_new_session=True,
    )
    compute = psutil.Process()
    temporary = root / "identities.tmp"
    temporary.write_text(
        json.dumps({"compute": describe(compute), "child": describe(psutil.Process(child.pid))}), encoding="utf-8"
    )
    temporary.replace(root / "identities.json")
    while True:
        time.sleep(0.02)


def audited_guardian(connection, stop, raw_job, executor, parent_pid):
    """Record successful native waits without changing the production guardian loop."""
    original_waitpid = os.waitpid
    path = Path(raw_job["output"]["output_dir"]) / "reap-events.jsonl"

    def waitpid(pid, options):
        result = original_waitpid(pid, options)
        if result[0] > 0:
            event = {
                "reaper": os.getpid(),
                "target": result[0],
                "wait_status": result[1],
                "wall": datetime.now(timezone.utc).isoformat(),
                "monotonic": time.monotonic(),
            }
            with path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(event) + "\n")
        return result

    os.waitpid = waitpid
    _worker_main(connection, stop, raw_job, executor, parent_pid)


def describe(process):
    """Capture identity, parent, session and state together with observation time."""
    return {
        "pid": process.pid,
        "created": process.create_time(),
        "ppid": process.ppid(),
        "pgid": os.getpgid(process.pid),
        "sid": os.getsid(process.pid),
        "status": process.status(),
        "wall": datetime.now(timezone.utc).isoformat(),
        "monotonic": time.monotonic(),
    }


def wait_for(predicate, timeout=20):
    """Bound readiness and cleanup without retrying a failed test run."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Real process condition did not arrive before the deadline")


def present(process):
    """Count a matching zombie as present; do not treat ZombieProcess as disappearance."""
    try:
        return process.is_running()
    except psutil.ZombieProcess:
        return True
    except psutil.NoSuchProcess:
        return False


@pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux guardian/subreaper and real zombie states")
@pytest.mark.parametrize("reason", ["cancel", "timeout", "shutdown"])
def test_zombie_retains_guardian_owner_and_slot_until_reaped(tmp_path, monkeypatch, reason):
    """Stop a known reaper, kill its known compute, and require fail-closed cleanup."""
    manager = JobsManager(
        storage_path=str(tmp_path / "state.json"),
        output_root=tmp_path,
        cpu_concurrency=1,
        gpu_concurrency=1,
        stop_grace_seconds=0.05,
        shutdown_grace_seconds=0.05,
    )
    manager._worker_executor = reaping_executor
    stop_entered, allow_stop, premature_guardian_kill = threading.Event(), threading.Event(), threading.Event()
    original_stop, original_killpg = ManagedWorker.stop, os.killpg
    guardian = None
    processes, facts, errors = [], [], []
    closing = None

    def record(label, **extra):
        observation = {"label": label, "reason": reason, **extra}
        facts.append(observation)
        (tmp_path / "reaping-facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
        print(json.dumps(observation), flush=True)

    def stop_at_boundary(worker, grace):
        if worker.process.pid == guardian.pid and not allow_stop.is_set():
            stop_entered.set()
            assert allow_stop.wait(10), "Test did not release the cleanup barrier"
        return original_stop(worker, grace)

    def protect_reaper(pgid, sig):
        # Keep the failing baseline observable and cleanable, instead of orphaning its zombies.
        if (
            guardian is not None
            and pgid == guardian.pid
            and sig == signal.SIGKILL
            and any(present(process) for process in processes[1:])
        ):
            premature_guardian_kill.set()
            raise RuntimeError("Attempted guardian kill before owned children were reaped")
        return original_killpg(pgid, sig)

    def close():
        try:
            manager.shutdown()
        except RuntimeError as error:
            errors.append(str(error))

    try:
        monkeypatch.setattr("studio.worker_runtime._worker_main", audited_guardian)
        manager.submit_job_request(
            JobRequest(
                job_id="reaping",
                task_type="predict",
                output={"output_dir": str(tmp_path)},
                runtime_tracking={"timeout_seconds": 12 if reason == "timeout" else 60},
            )
        )
        wait_for(lambda: (tmp_path / "identities.json").exists())
        identities = json.loads((tmp_path / "identities.json").read_text(encoding="utf-8"))
        worker = manager._workers["reaping"]
        guardian = psutil.Process(worker.process.pid)
        compute, child = (psutil.Process(identities[key]["pid"]) for key in ("compute", "child"))
        processes = [guardian, compute, child]
        assert compute.create_time() == identities["compute"]["created"] and compute.ppid() == guardian.pid
        assert child.create_time() == identities["child"]["created"] and child.ppid() == compute.pid
        assert os.getsid(child.pid) == child.pid and os.getsid(compute.pid) == guardian.pid
        worker._capture_descendants()
        record("ready", guardian=describe(guardian), compute=describe(compute), child=describe(child))
        manager.submit_job_request(
            JobRequest(job_id="queued", task_type="predict", output={"output_dir": str(tmp_path)})
        )
        monkeypatch.setattr(ManagedWorker, "stop", stop_at_boundary)
        monkeypatch.setattr(os, "killpg", protect_reaper)
        guardian.suspend()
        wait_for(lambda: guardian.status() == psutil.STATUS_STOPPED)
        if reason == "cancel":
            manager.request_cancel("reaping")
        elif reason == "shutdown":
            closing = threading.Thread(target=close)
            closing.start()
        wait_for(stop_entered.is_set)
        compute.kill()
        wait_for(lambda: compute.status() == psutil.STATUS_ZOMBIE and child.ppid() == guardian.pid)
        record(
            "compute_zombie_adopted_child",
            guardian=describe(guardian),
            compute=describe(compute),
            child=describe(child),
        )
        allow_stop.set()
        wait_for(lambda: manager.get_job("reaping").error is not None)
        snapshot = manager.get_job("reaping")
        record(
            "cleanup_blocked",
            guardian=describe(guardian),
            compute=describe(compute),
            child=describe(child),
            status=snapshot.status.value,
            error=snapshot.error.code,
            owner_retained="reaping" in manager._workers,
            premature_guardian_kill=premature_guardian_kill.is_set(),
        )
        assert not premature_guardian_kill.is_set(), "Runtime attempted to kill its reaper before reap completed"
        assert snapshot.status == JobStatus.RUNNING and snapshot.error.code == "WORKER_STOP_FAILED"
        assert snapshot.metadata.completed_at is None and "reaping" in manager._workers
        assert guardian.status() == psutil.STATUS_STOPPED and compute.status() == psutil.STATUS_ZOMBIE
        persisted = json.loads((tmp_path / "state.json").read_text())["jobs"]["reaping"]
        assert persisted["status"] == "running" and persisted["metadata"]["completed_at"] is None
        assert not (tmp_path / "queued.started").exists()
        assert manager.get_job("queued").status == (JobStatus.FAILED if reason == "shutdown" else JobStatus.PENDING)
        guardian.resume()
        wait_for(lambda: manager.get_job("reaping").status in TERMINAL_STATUSES)
        result = manager.get_job("reaping")
        assert not any(present(process) for process in processes)
        assert "reaping" not in manager._workers
        assert (
            result.error.code
            == {"cancel": "USER_CANCELLED", "timeout": "TIMEOUT", "shutdown": "SERVICE_SHUTDOWN"}[reason]
        )
        assert result.status == (JobStatus.CANCELLED if reason == "cancel" else JobStatus.FAILED)
        reaps = [json.loads(line) for line in (tmp_path / "reap-events.jsonl").read_text().splitlines()]
        assert {compute.pid, child.pid}.issubset({event["target"] for event in reaps})
        assert all(event["reaper"] == guardian.pid for event in reaps)
        record("terminal_after_reap", terminal=result.model_dump(mode="json"), identities_gone=True, native_waits=reaps)
        if closing is not None:
            closing.join(5)
            assert not closing.is_alive() and not errors
        manager.shutdown()
        restored = JobsManager(storage_path=str(tmp_path / "state.json"), output_root=tmp_path)
        try:
            assert restored.get_job("reaping").model_dump() == result.model_dump()
            assert not restored._workers
        finally:
            restored.shutdown()
    finally:
        allow_stop.set()
        if guardian is not None and present(guardian):
            guardian.resume()
            wait_for(lambda: not any(present(process) for process in processes[1:]))
        manager.shutdown()
        if closing is not None:
            closing.join(5)
        assert not any(present(process) for process in processes)
