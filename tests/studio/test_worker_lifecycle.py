"""Real CPU-only process lifecycle tests: no torch, GPU, model or dataset downloads."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from core.schema import TERMINAL_STATUSES, ErrorInfo, JobRequest, JobStatus, TaskType
from studio.jobs_manager import JobsManager
from studio.worker_runtime import ManagedWorker, execute_job

TASK_TIMEOUT = 12 if sys.platform == "linux" else 3


def dummy_executor(job):
    """Spawn a real child and grandchild, then finish, crash or ignore cancellation."""
    root = Path(job.output.output_dir)
    if os.name != "nt":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if job.params.get("children", True):
        # Constant program + separate argv paths; no shell and no command interpolation.
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import os,signal,subprocess,sys,time; from pathlib import Path; "
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'], "
                    "start_new_session=(os.name!='nt')); "
                    "Path(sys.argv[1]).write_text(str(os.getpid())+','+str(p.pid)); time.sleep(120)"
                ),
                str(root / f"{job.job_id}.children"),
            ],
            start_new_session=os.name != "nt",  # Match torchrun's detached rank sessions.
        )
    (root / f"{job.job_id}.started").write_text(str(os.getpid()))
    for message in job.params.get("live_logs", []):
        job.append_log(message)
    if job.params.get("crash"):
        deadline = time.monotonic() + 5
        while not (root / f"{job.job_id}.children").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        os._exit(17)
    gate = root / f"{job.job_id}.release"
    while not gate.exists():
        time.sleep(0.02)
    job.status = JobStatus.COMPLETED
    if terminal_log := job.params.get("terminal_log"):
        job.append_log(terminal_log)
    return job


def mock_yolo_executor(job):
    """Exercise the real dispatcher and each real handler using an in-process fake YOLO."""

    class FakeYOLO:
        def __init__(self, *args):
            self.model = SimpleNamespace()

        def add_callback(self, event, callback):
            self.callback = (event, callback)

        def compute(self, **kwargs):
            dummy_executor(job)

        train = val = predict = export = compute

    sys.modules["ultralytics"] = SimpleNamespace(YOLO=FakeYOLO)
    from studio.handlers.train import TrainHandler

    TrainHandler._inject_determinism = lambda *args, **kwargs: 42
    return execute_job(job)


def wait_for(predicate, timeout=15):
    """Poll real process/file state with a bounded test deadline."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.03)
    raise AssertionError("Lifecycle condition did not become true before the test deadline")


def live(pid):
    """Zombies have stopped computing; POSIX init is responsible for their final reap."""
    try:
        return psutil.Process(pid).is_running() and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.fixture
def manager(tmp_path):
    instance = JobsManager(
        storage_path=str(tmp_path / "state.json"),
        output_root=tmp_path,
        model_roots=[tmp_path],
        data_roots=[tmp_path],
        cpu_concurrency=2,
        gpu_concurrency=1,
        stop_grace_seconds=0.2,
        shutdown_grace_seconds=0.2,
    )
    instance._worker_executor = dummy_executor
    yield instance
    instance.shutdown()


def submit(manager, root, job_id, device="cpu", timeout=30, **params):
    return manager.submit_job_request(
        JobRequest(
            job_id=job_id,
            task_type=TaskType.PREDICT,
            params={"device": device, **params},
            output={"output_dir": str(root)},
            runtime_tracking={"timeout_seconds": timeout},
        )
    )


def pids(root, job_id):
    wait_for(lambda: (root / f"{job_id}.started").exists() and (root / f"{job_id}.children").exists())
    return [
        int((root / f"{job_id}.started").read_text()),
        *map(int, (root / f"{job_id}.children").read_text().split(",")),
    ]


def terminal(manager, job_id):
    wait_for(lambda: manager.get_job(job_id).status in TERMINAL_STATUSES)
    assert job_id not in manager._workers
    return manager.get_job(job_id)


def test_normal_completion_cleans_leftover_children(manager, tmp_path):
    submit(manager, tmp_path, "normal")
    owned = pids(tmp_path, "normal")
    (tmp_path / "normal.release").touch()
    result = terminal(manager, "normal")
    assert result.status == JobStatus.COMPLETED
    assert result.metadata.started_at is not None
    assert result.metadata.completed_at is not None
    assert manager.get_job_status("normal")["duration"] >= 0
    assert not any(live(pid) for pid in owned)


def test_log_sequence_reordering_dedup_and_terminal_tail(monkeypatch, tmp_path):
    """Parent orders out-of-order events, drops duplicate seq values, and merges only the final tail."""
    import studio.jobs_manager as jobs_manager_module

    manager = JobsManager(output_root=tmp_path, model_roots=[tmp_path], data_roots=[tmp_path])
    job_id = "log-sequence"
    job = JobRequest(job_id=job_id, task_type=TaskType.PREDICT, output={"output_dir": str(tmp_path)})
    manager._jobs[job_id] = job
    manager._logs.entries[job_id] = ["submitted"]

    result = job.model_copy(deep=True)
    result.logs = ["zero", "one", "terminal"]
    result.status = JobStatus.COMPLETED
    cleanup_started = threading.Event()
    allow_cleanup = threading.Event()
    events = iter(
        [
            ("log", {"seq": 1, "text": "one", "terminal": False}),
            ("log", {"seq": 0, "text": "zero", "terminal": False}),
            ("log", {"seq": 1, "text": "duplicate one", "terminal": False}),
            ("log", {"seq": 2, "text": "terminal", "terminal": True}),
            ("result", result.model_dump(mode="json")),
        ]
    )

    class FakeProcess:
        @staticmethod
        def is_alive():
            return True

    class FakeWorker:
        def __init__(self, *_args, **_kwargs):
            self.process = FakeProcess()

        def start(self):
            return None

        def receive(self):
            return next(events)

        def stop(self, _grace):
            cleanup_started.set()
            assert allow_cleanup.wait(timeout=5)

        def close(self):
            return None

    monkeypatch.setattr(jobs_manager_module, "ManagedWorker", FakeWorker)
    execution = threading.Thread(target=manager._execute_job, args=(job_id,), daemon=True)
    execution.start()
    assert cleanup_started.wait(timeout=5)
    assert manager._jobs[job_id].status == JobStatus.RUNNING
    assert manager._logs.entries[job_id] == ["submitted", "zero", "one"]
    allow_cleanup.set()
    execution.join(timeout=5)

    assert not execution.is_alive()
    assert manager._jobs[job_id].status == JobStatus.COMPLETED
    assert manager._logs.entries[job_id] == ["submitted", "zero", "one", "terminal"]


@pytest.mark.parametrize("reason", ["cancel", "timeout", "shutdown", "crash"])
def test_stop_confirms_entire_tree_before_persisting(manager, tmp_path, reason):
    live_line = f"{reason} live log"
    submit(
        manager,
        tmp_path,
        reason,
        timeout=TASK_TIMEOUT if reason == "timeout" else 30,
        crash=reason == "crash",
        live_logs=[live_line],
    )
    owned = pids(tmp_path, reason)
    wait_for(lambda: live_line in manager.get_job_log_lines(reason))
    if reason == "cancel":
        manager.cancel_job(reason)
    elif reason == "shutdown":
        manager.shutdown()
    result = terminal(manager, reason)
    expected = {
        "cancel": "USER_CANCELLED",
        "timeout": "TIMEOUT",
        "shutdown": "SERVICE_SHUTDOWN",
        "crash": "WORKER_LOST",
    }
    assert result.error.code == expected[reason]
    assert result.status == (JobStatus.CANCELLED if reason == "cancel" else JobStatus.FAILED)
    assert result.metadata.started_at is not None
    assert result.metadata.completed_at is not None
    assert manager.get_job_status(reason)["duration"] >= 0
    assert manager.get_job_log_lines(reason).count(live_line) == 1
    assert not any(live(pid) for pid in owned)
    saved = json.loads((tmp_path / "state.json").read_text())["jobs"][reason]
    assert saved["status"] == ("cancelled" if reason == "cancel" else "failed")
    assert saved["error"]["code"] == expected[reason]
    assert "duration" not in saved and "duration" not in saved["metadata"]


def test_cpu_gpu_limits_and_pending_cancellation(manager, tmp_path):
    for job_id, device in [
        ("cpu1", "cpu"),
        ("cpu2", "cpu"),
        ("gpu1", "0"),
        ("cpu3", "cpu"),
        ("gpu2", "0,1"),
        ("gpu3", ""),
    ]:
        submit(manager, tmp_path, job_id, device=device)
    owned = {job_id: pids(tmp_path, job_id) for job_id in ("cpu1", "cpu2", "gpu1")}
    assert len(manager._workers) == 3
    assert len(manager._supervisors) == 3
    for job_id in ("cpu3", "gpu2", "gpu3"):
        assert manager.get_job(job_id).status == JobStatus.PENDING
        assert not (tmp_path / f"{job_id}.started").exists()
    manager.cancel_job("gpu2")
    cancelled = terminal(manager, "gpu2")
    assert cancelled.error.code == "USER_CANCELLED"
    assert cancelled.metadata.started_at is None
    assert cancelled.metadata.completed_at is not None
    assert manager.get_job_status("gpu2")["duration"] is None
    for job_id in ("cpu1", "gpu1"):
        manager.cancel_job(job_id)
        terminal(manager, job_id)
        assert not any(live(pid) for pid in owned[job_id])
    pids(tmp_path, "cpu3")
    pids(tmp_path, "gpu3")
    assert len(manager._workers) == 3
    assert not (tmp_path / "gpu2.started").exists()


@pytest.mark.parametrize("device", ["cpu", "0"])
@pytest.mark.parametrize("outcome", ["complete", "cancel", "timeout", "crash"])
def test_capacity_released_only_after_tree_exit(manager, tmp_path, device, outcome):
    manager._limits = {"cpu": 1, "gpu": 1}
    submit(
        manager,
        tmp_path,
        "first",
        device=device,
        timeout=TASK_TIMEOUT if outcome == "timeout" else 30,
        crash=outcome == "crash",
    )
    submit(manager, tmp_path, "next", device=device)
    assert manager.get_job("next").status == JobStatus.PENDING
    owned = pids(tmp_path, "first")
    if outcome == "complete":
        (tmp_path / "first.release").touch()
    elif outcome == "cancel":
        manager.cancel_job("first")
    result = terminal(manager, "first")
    if outcome == "complete":
        assert result.status == JobStatus.COMPLETED
    else:
        assert result.error.code == {"cancel": "USER_CANCELLED", "timeout": "TIMEOUT", "crash": "WORKER_LOST"}[outcome]
    assert not any(live(pid) for pid in owned)
    pids(tmp_path, "next")
    assert len(manager._workers) == 1
    assert manager.get_job("next").status == JobStatus.RUNNING


@pytest.mark.parametrize(
    "device,expected", [("cpu", "cpu"), (0, "gpu"), (None, "gpu"), ("", "gpu"), ("mps", "gpu"), ([0, 1], "gpu")]
)
def test_resource_classification(device, expected):
    job = JobRequest(job_id="device", task_type=TaskType.TRAIN, params={"device": device})
    assert JobsManager._resource_class(job) == expected


def test_server_configuration(monkeypatch):
    monkeypatch.setenv("STUDIO_CPU_CONCURRENCY", "3")
    monkeypatch.setenv("STUDIO_GPU_CONCURRENCY", "2")
    instance = JobsManager()
    assert instance._limits == {"cpu": 3, "gpu": 2}
    monkeypatch.setenv("STUDIO_CPU_CONCURRENCY", "0")
    with pytest.raises(ValueError, match="concurrency"):
        JobsManager()


def test_launch_failure_cleans_gated_process(manager, tmp_path, monkeypatch):
    launched = []

    def fail_after_spawn(worker):
        worker.process.start()
        launched.append(worker.process.pid)
        raise OSError("Containment setup failed")

    monkeypatch.setattr(ManagedWorker, "start", fail_after_spawn)
    submit(manager, tmp_path, "launch-failed")
    assert terminal(manager, "launch-failed").error.code == "EXECUTION_FAILED"
    assert launched and not any(live(pid) for pid in launched)
    assert not (tmp_path / "launch-failed.started").exists()


def test_cleanup_failure_retains_slot_and_nonterminal_state(manager, tmp_path, monkeypatch):
    manager._limits["cpu"] = 1
    submit(manager, tmp_path, "stopping")
    owned = pids(tmp_path, "stopping")
    original_stop = ManagedWorker.stop

    def deferred_stop(worker, grace):
        if not (tmp_path / "allow-cleanup").exists():
            raise OSError("Temporary termination failure")
        return original_stop(worker, grace)

    monkeypatch.setattr(ManagedWorker, "stop", deferred_stop)
    try:
        submit(manager, tmp_path, "waiting")
        manager.cancel_job("stopping")
        wait_for(lambda: manager.get_job("stopping").error is not None)
        assert manager.get_job("stopping").error.code == "WORKER_STOP_FAILED"
        assert manager.get_job("stopping").status == JobStatus.RUNNING
        assert "stopping" in manager._workers
        assert manager.get_job("waiting").status == JobStatus.PENDING
        assert all(live(pid) for pid in owned)
    finally:
        (tmp_path / "allow-cleanup").touch()
    assert terminal(manager, "stopping").error.code == "USER_CANCELLED"
    assert not any(live(pid) for pid in owned)
    pids(tmp_path, "waiting")


@pytest.mark.parametrize("exit_mode", ["kill", "normal"])
def test_parent_exit_cleans_tree_and_restart_is_honest(tmp_path, exit_mode):
    program = (
        "import sys,time; from pathlib import Path; "
        "sys.path.insert(0,sys.argv[1]); "
        "from test_worker_lifecycle import dummy_executor,submit; "
        "from studio.jobs_manager import JobsManager; "
        "root=Path(sys.argv[2]); "
        "manager=JobsManager(storage_path=str(root/'state.json'), output_root=root,stop_grace_seconds=0.1); "
        "manager._worker_executor=dummy_executor; submit(manager,root,'parent'); "
        "\nwhile not (root/'exit').exists(): time.sleep(0.05)\n"
    )
    parent = subprocess.Popen([sys.executable, "-c", program, str(Path(__file__).parent), str(tmp_path)])
    identities = []
    try:
        owned = pids(tmp_path, "parent")
        identities = [psutil.Process(pid) for pid in owned]
        if exit_mode == "kill":
            parent.kill()
        else:
            (tmp_path / "exit").touch()
        parent.wait(timeout=15)
        wait_for(lambda: not any(live(pid) for pid in owned))
        restored = JobsManager(storage_path=str(tmp_path / "state.json"))
        assert restored.get_job("parent").error.code == (
            "SERVICE_RESTARTED" if exit_mode == "kill" else "SERVICE_SHUTDOWN"
        )
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        for process in identities:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass


def test_cancel_while_running_wins_over_late_worker_completion(monkeypatch, tmp_path):
    """A result arriving after an accepted cancellation cannot replace CANCELLED."""
    import studio.jobs_manager as jobs_manager_module

    manager = JobsManager(output_root=tmp_path, model_roots=[tmp_path], data_roots=[tmp_path])
    job_id = "cancel-late-result"
    job = JobRequest(job_id=job_id, task_type=TaskType.PREDICT, output={"output_dir": str(tmp_path)})
    manager._jobs[job_id] = job
    manager._logs.entries[job_id] = []
    completed_result = job.model_copy(deep=True)
    completed_result.status = JobStatus.COMPLETED

    worker_started = threading.Event()
    result_waiting = threading.Event()
    release_late_result = threading.Event()

    class FakeProcess:
        @staticmethod
        def is_alive():
            return True

    class FakeWorker:
        def __init__(self, *_args, **_kwargs):
            self.process = FakeProcess()

        def start(self):
            worker_started.set()

        def receive(self):
            result_waiting.set()
            assert release_late_result.wait(timeout=5)
            return "result", completed_result.model_dump(mode="json")

        def stop(self, _grace):
            return None

        def close(self):
            return None

    monkeypatch.setattr(jobs_manager_module, "ManagedWorker", FakeWorker)
    execution = threading.Thread(target=manager._execute_job, args=(job_id,), daemon=True)
    execution.start()
    assert worker_started.wait(timeout=5)
    assert result_waiting.wait(timeout=5)

    assert "Cancellation requested" in manager.cancel_job(job_id)
    release_late_result.set()
    execution.join(timeout=5)

    assert not execution.is_alive()
    assert manager._jobs[job_id].status == JobStatus.CANCELLED
    assert manager._jobs[job_id].error.code == "USER_CANCELLED"
    assert manager._jobs[job_id].runtime_tracking.cancel_requested is True
    manager.shutdown()


def test_timeout_wins_without_consuming_late_worker_completion(monkeypatch, tmp_path):
    """Once the deadline expires, a completion waiting behind it cannot replace TIMEOUT."""
    import studio.jobs_manager as jobs_manager_module

    manager = JobsManager(output_root=tmp_path, model_roots=[tmp_path], data_roots=[tmp_path])
    job_id = "timeout-late-result"
    job = JobRequest(
        job_id=job_id,
        task_type=TaskType.PREDICT,
        output={"output_dir": str(tmp_path)},
        runtime_tracking={"timeout_seconds": 1},
    )
    manager._jobs[job_id] = job
    manager._logs.entries[job_id] = []
    completed_result = job.model_copy(deep=True)
    completed_result.status = JobStatus.COMPLETED
    receive_calls = 0

    class FakeProcess:
        @staticmethod
        def is_alive():
            return True

    class FakeWorker:
        def __init__(self, *_args, **_kwargs):
            self.process = FakeProcess()

        def start(self):
            return None

        def receive(self):
            nonlocal receive_calls
            receive_calls += 1
            if receive_calls == 1:
                return None, None
            return "result", completed_result.model_dump(mode="json")

        def stop(self, _grace):
            return None

        def close(self):
            return None

    monkeypatch.setattr(jobs_manager_module, "ManagedWorker", FakeWorker)
    monkeypatch.setattr(
        jobs_manager_module,
        "time",
        SimpleNamespace(monotonic=lambda: 2.0 if receive_calls else 0.0, sleep=time.sleep),
    )

    manager._execute_job(job_id)

    assert receive_calls == 1
    assert manager._jobs[job_id].status == JobStatus.FAILED
    assert manager._jobs[job_id].error.code == "TIMEOUT"
    manager.shutdown()


def test_old_persistence_without_execution_timestamps_loads(tmp_path):
    job = JobRequest(job_id="legacy", task_type=TaskType.PREDICT, status=JobStatus.COMPLETED).model_dump(mode="json")
    job["metadata"].pop("started_at")
    job["metadata"].pop("completed_at")
    path = tmp_path / "legacy-state.json"
    path.write_text(json.dumps({"jobs": {"legacy": job}, "job_logs": {}}))

    restored = JobsManager(storage_path=str(path)).get_job("legacy")

    assert restored.status == JobStatus.COMPLETED
    assert restored.metadata.started_at is None
    assert restored.metadata.completed_at is None


def test_shutdown_preserves_terminal_job_and_never_launches_queued_job(monkeypatch, tmp_path):
    """Shutdown stops the active slot, drains queued IDs inertly, and leaves prior terminal state untouched."""
    import studio.jobs_manager as jobs_manager_module

    manager = JobsManager(
        output_root=tmp_path,
        model_roots=[tmp_path],
        data_roots=[tmp_path],
        cpu_concurrency=1,
        gpu_concurrency=1,
    )
    active_started = threading.Event()
    allow_active_poll = threading.Event()
    queued_failed = threading.Event()
    constructed: list[str] = []

    class FakeProcess:
        @staticmethod
        def is_alive():
            return True

    class FakeWorker:
        def __init__(self, job, *_args, **_kwargs):
            constructed.append(job.job_id)
            self.process = FakeProcess()

        def start(self):
            active_started.set()

        def receive(self):
            assert allow_active_poll.wait(timeout=5)
            return None, None

        def stop(self, _grace):
            return None

        def close(self):
            return None

    monkeypatch.setattr(jobs_manager_module, "ManagedWorker", FakeWorker)
    original_fail_job = manager._fail_job

    def observe_queued_shutdown(job, code, message):
        original_fail_job(job, code, message)
        if job.job_id == "queued":
            queued_failed.set()

    monkeypatch.setattr(manager, "_fail_job", observe_queued_shutdown)
    manager.submit_job_request(
        JobRequest(
            job_id="active",
            task_type=TaskType.PREDICT,
            params={"device": "cpu"},
            output={"output_dir": str(tmp_path)},
        )
    )
    assert active_started.wait(timeout=5)
    manager.submit_job_request(
        JobRequest(
            job_id="queued",
            task_type=TaskType.PREDICT,
            params={"device": "cpu"},
            output={"output_dir": str(tmp_path)},
        )
    )
    terminal_job = JobRequest(job_id="existing-terminal", task_type=TaskType.PREDICT, status=JobStatus.COMPLETED)
    manager._jobs[terminal_job.job_id] = terminal_job
    manager._logs.entries[terminal_job.job_id] = ["already complete"]
    terminal_snapshot = terminal_job.model_dump(mode="json")

    shutdown = threading.Thread(target=manager.shutdown, daemon=True)
    shutdown.start()
    assert queued_failed.wait(timeout=5)
    assert manager._jobs["queued"].status == JobStatus.FAILED
    assert manager._jobs["queued"].error.code == "SERVICE_SHUTDOWN"
    assert manager._jobs["queued"].metadata.started_at is None
    allow_active_poll.set()
    shutdown.join(timeout=5)

    assert not shutdown.is_alive()
    assert constructed == ["active"]
    assert manager._jobs["active"].status == JobStatus.FAILED
    assert manager._jobs["active"].error.code == "SERVICE_SHUTDOWN"
    assert manager._jobs["existing-terminal"].model_dump(mode="json") == terminal_snapshot
    assert manager._logs.entries["existing-terminal"] == ["already complete"]
    assert manager._workers == {}
    assert all(not thread.is_alive() for thread in manager._supervisors)


def test_terminal_jobs_reload_without_being_reexecuted(tmp_path):
    """Persisted terminal jobs remain terminal and are never added back to execution queues."""
    jobs = {
        "completed": JobRequest(job_id="completed", task_type=TaskType.PREDICT, status=JobStatus.COMPLETED).model_dump(
            mode="json"
        ),
        "failed": JobRequest(
            job_id="failed",
            task_type=TaskType.PREDICT,
            status=JobStatus.FAILED,
            error=ErrorInfo(code="EXPECTED_FAILURE", message="already failed"),
        ).model_dump(mode="json"),
        "cancelled": JobRequest(
            job_id="cancelled",
            task_type=TaskType.PREDICT,
            status=JobStatus.CANCELLED,
            error=ErrorInfo(code="USER_CANCELLED", message="already cancelled"),
        ).model_dump(mode="json"),
    }
    state_path = tmp_path / "terminal-state.json"
    state_path.write_text(json.dumps({"jobs": jobs, "job_logs": {}}))
    restored = JobsManager(storage_path=str(state_path), output_root=tmp_path)
    executed: list[str] = []
    restored._execute_job = executed.append

    with restored.lock:
        restored._start_supervisors()
    restored.shutdown()

    assert executed == []
    assert restored._workers == {}
    assert restored._jobs["completed"].status == JobStatus.COMPLETED
    assert restored._jobs["failed"].error.code == "EXPECTED_FAILURE"
    assert restored._jobs["cancelled"].error.code == "USER_CANCELLED"


@pytest.mark.parametrize("task", ["train", "val", "predict", "export"])
@pytest.mark.parametrize("reason", ["cancel", "timeout"])
def test_real_handlers_are_process_isolated(manager, tmp_path, task, reason):
    manager._worker_executor = mock_yolo_executor
    model, data = tmp_path / "mock.pt", tmp_path / ("data.yaml" if task in ("train", "val") else "image.jpg")
    model.touch()
    data.touch()
    job_id = f"{task}-{reason}"
    request = JobRequest(
        job_id=job_id,
        task_type=TaskType(task),
        params={"model_path": str(model), "data_source": str(data), "device": "cpu", "format": "onnx"},
        output={"output_dir": str(tmp_path)},
        runtime_tracking={"timeout_seconds": TASK_TIMEOUT if reason == "timeout" else 30},
    )
    manager.submit_job_request(request)
    owned = pids(tmp_path, job_id)
    if reason == "cancel":
        manager.cancel_job(job_id)
    result = terminal(manager, job_id)
    assert result.error.code == ("USER_CANCELLED" if reason == "cancel" else "TIMEOUT")
    assert not any(live(pid) for pid in owned)
