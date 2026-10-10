"""Real IPC cooperation and independent checkpoint recovery checks; no implicit downloads."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from core.schema import JobRequest, JobStatus
from studio.jobs_manager import JobsManager
from studio.training_shutdown import CHECKPOINT_ID, CURRENT_SHUTDOWN, TrainingShutdown, install_training_shutdown
from studio.worker_runtime import execute_job


def wait_for(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Shutdown condition did not arrive before the test deadline")


def cooperative_executor(job):
    """Use actual worker stop IPC with a deterministic epoch checkpoint stand-in."""
    root = Path(job.output.output_dir)
    control = CURRENT_SHUTDOWN.get()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    (root / "child.pid").write_text(str(child.pid))
    (root / "started").touch()
    wait_for(control.requested.is_set)
    job.append_log("stop IPC reached training thread password=hidden-value")
    for index in range(job.params.get("log_count", 0)):
        job.append_log(f"shutdown-stream-{index:04d}")
    if job.params.get("ignore"):
        while True:
            time.sleep(0.02)
    target = root / job.job_id / CHECKPOINT_ID
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"checkpoint-test-data")
    (target.parent / "unlisted.pt").write_bytes(b"unauthorized")
    event = {"path": CHECKPOINT_ID, "epoch": 2, "size": target.stat().st_size}
    event.update(job.params.get("event", {}))
    control.emit("checkpoint", event)
    wait_for(control.acknowledged.is_set)
    (root / "acknowledged").touch()
    if job.params.get("linger"):
        while True:
            time.sleep(0.02)

    def finish_child():
        if job.params.get("wait_for_result"):
            wait_for(lambda: (root / "result-received").exists())
        time.sleep(job.params.get("exit_delay", 0))
        child.terminate()
        child.wait(timeout=5)
        (root / "finalized").touch()

    if job.params.get("exit_delay"):
        threading.Thread(target=finish_child).start()
    else:
        finish_child()
    job.status = JobStatus.COMPLETED
    return job


def new_manager(root, grace=1):
    manager = JobsManager(
        storage_path=str(root / "state.json"),
        output_root=root,
        model_roots=[root],
        data_roots=[root],
        cpu_concurrency=1,
        gpu_concurrency=1,
        stop_grace_seconds=0.1,
        shutdown_grace_seconds=grace,
    )
    manager._worker_executor = cooperative_executor
    return manager


def submit(manager, root, **params):
    return manager.submit_job_request(
        JobRequest(
            job_id="shutdown-train",
            task_type="train",
            params={"device": "cpu", **params},
            output={"output_dir": str(root)},
            runtime_tracking={"timeout_seconds": 120, "stream_logs": True},
        )
    )


def assert_child_gone(root):
    pid = int((root / "child.pid").read_text())
    try:
        process = psutil.Process(pid)
        assert not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass


def test_checkpoint_ack_shutdown_manifest_and_restart(tmp_path):
    manager = new_manager(tmp_path, grace=5)
    try:
        submit(manager, tmp_path)
        wait_for(lambda: (tmp_path / "started").exists())
        manager.shutdown()
        job = manager.get_job("shutdown-train")
        assert job.status == JobStatus.FAILED
        assert job.error.code == "SERVICE_SHUTDOWN"
        assert not job.runtime_tracking.cancel_requested
        assert job.metadata.completed_at and not manager._workers
        assert (tmp_path / "acknowledged").exists()
        assert job.output.artifacts == [CHECKPOINT_ID]
        assert manager.resolve_artifact(job.job_id, CHECKPOINT_ID).is_file()
        assert manager.resolve_artifact(job.job_id, "weights/unlisted.pt") is None
        logs = manager.get_job_logs(job.job_id)
        assert "CHECKPOINT_CONFIRMED" in logs and "hidden-value" not in logs
        assert "stop IPC reached training thread" in logs
        assert "SHUTDOWN_FORCED" not in logs
        assert_child_gone(tmp_path)
        restored = new_manager(tmp_path)
        try:
            assert restored.get_job(job.job_id).model_dump() == job.model_dump()
        finally:
            restored.shutdown()
    finally:
        manager.shutdown()


@pytest.mark.parametrize(
    "params",
    [
        {"ignore": True},
        {"linger": True},
        {"event": {"path": "../escape.pt"}},
        {"event": {"size": 1}},
        {"event": {"epoch": -1}},
    ],
)
def test_uncooperative_or_invalid_checkpoint_forces_only_at_deadline(tmp_path, params):
    manager = new_manager(tmp_path, grace=0.4)
    try:
        submit(manager, tmp_path, **params)
        wait_for(lambda: (tmp_path / "started").exists())
        started = time.monotonic()
        manager.shutdown()
        assert time.monotonic() - started >= 0.4
        job = manager.get_job("shutdown-train")
        assert job.status == JobStatus.FAILED and job.error.code == "SERVICE_SHUTDOWN"
        assert "SHUTDOWN_FORCED" in manager.get_job_logs(job.job_id)
        assert job.output.artifacts == ([CHECKPOINT_ID] if params.get("linger") else [])
        assert_child_gone(tmp_path)
    finally:
        manager.shutdown()


def test_epoch_callback_saves_before_ack_without_watcher_saving(tmp_path):
    checkpoint = tmp_path / "weights" / "last.pt"
    checkpoint.parent.mkdir()
    calls = []
    control = TrainingShutdown(lambda *event: calls.append(event))
    trainer = SimpleNamespace(last=checkpoint, save_dir=tmp_path, epoch=3, stop=False, world_size=1)

    def serialize(*, include_online_model):
        assert include_online_model
        return b"unstripped-optimizer-scaler"

    trainer._serialize_checkpoint = serialize
    control.on_epoch_end(trainer)
    assert calls == []
    control.requested.set()
    callback = threading.Thread(target=control.on_epoch_end, args=(trainer,))
    callback.start()
    wait_for(lambda: bool(calls))
    assert callback.is_alive() and not trainer.stop
    assert (tmp_path / CHECKPOINT_ID).read_bytes() == b"unstripped-optimizer-scaler"
    checkpoint.write_bytes(b"stripped")
    control.acknowledged.set()
    callback.join(5)
    assert not callback.is_alive() and trainer.stop
    control.on_epoch_end(trainer)
    assert len(calls) == 1


def test_ddp_is_explicitly_outside_cooperation(monkeypatch):
    events, callbacks = [], []
    control = TrainingShutdown(lambda *event: events.append(event))
    token = CURRENT_SHUTDOWN.set(control)
    try:
        model = SimpleNamespace(add_callback=lambda *args: callbacks.append(args))
        install_training_shutdown(model, "0,1")
        assert not callbacks and events[0][0] == "shutdown_limit"
        monkeypatch.setenv("WORLD_SIZE", "2")
        install_training_shutdown(model, "0")
        assert not callbacks
    finally:
        CURRENT_SHUTDOWN.reset(token)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 3601])
def test_shutdown_budget_rejects_nonfinite_or_unbounded_values(tmp_path, value):
    with pytest.raises(ValueError, match="Shutdown grace"):
        JobsManager(output_root=tmp_path, shutdown_grace_seconds=value)


def real_training_executor(job):
    """Actual YOLO handler, bounded test settings and no model/data downloads in spawn."""
    import torch

    from ultralytics import YOLO
    from ultralytics.utils import checks, downloads

    torch.set_num_threads(2)

    def reject(*args, **kwargs):
        raise AssertionError("No downloads permitted during shutdown acceptance")

    downloads.safe_download = downloads.download = reject
    checks.check_font = lambda *args, **kwargs: None

    def local_yolo(path):
        model = YOLO(path)
        original_train = model.train

        def started(trainer):
            (Path(job.output.output_dir) / "training.started").write_text(str(os.getpid()))

        model.add_callback("on_train_epoch_start", started)

        def bounded_train(**kwargs):
            return original_train(**kwargs, workers=1, amp=False, plots=False)

        model.train = bounded_train
        return model

    import ultralytics

    ultralytics.YOLO = local_yolo
    return execute_job(job)


@pytest.mark.slow
@pytest.mark.studio_integration
def test_real_shutdown_checkpoint_load_and_resume(tmp_path):
    import torch
    import yaml
    from PIL import Image

    from ultralytics import YOLO

    model = Path(os.environ["STUDIO_TEST_MODEL"]).resolve(strict=True)
    for split in ("train", "val"):
        images, labels = tmp_path / "images" / split, tmp_path / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for index in range(4):
            Image.new("RGB", (64, 64), "white").save(images / f"{index}.jpg")
            (labels / f"{index}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    data = tmp_path / "data.yaml"
    data.write_text(
        yaml.safe_dump({"path": str(tmp_path), "train": "images/train", "val": "images/val", "names": {0: "object"}})
    )
    manager = new_manager(tmp_path, grace=120)
    manager._admission.model_roots = [model.parent]
    manager._worker_executor = real_training_executor
    try:
        submit(manager, tmp_path, model_path=str(model), data_source=str(data), epochs=10, batch_size=2, imgsz=64)
        wait_for(lambda: (tmp_path / "training.started").exists(), timeout=120)
        # Capture actual dataloader descendants before closing; track identity, not recycled PIDs.
        worker = manager._workers["shutdown-train"]
        descendants = psutil.Process(worker.process.pid).children(recursive=True)
        manager.shutdown()
        job = manager.get_job("shutdown-train")
        assert job.status == JobStatus.FAILED and job.error.code == "SERVICE_SHUTDOWN"
        assert not job.runtime_tracking.cancel_requested
        checkpoint = manager.resolve_artifact(job.job_id, CHECKPOINT_ID)
        assert checkpoint is not None, manager.get_job_logs(job.job_id)
        assert "SHUTDOWN_FORCED" not in manager.get_job_logs(job.job_id)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert 0 <= state["epoch"] < 9
        assert state["model"] is not None
        assert state["optimizer"] is not None and state["scaler"] is not None
        assert state["ema"] is not None and state["updates"] >= 0
        YOLO(str(checkpoint))
        for child in descendants:
            assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
        resumed = YOLO(str(checkpoint))
        epochs_seen = []
        resumed.add_callback("on_train_epoch_end", lambda trainer: epochs_seen.append(trainer.epoch))
        resumed.train(
            resume=True, epochs=state["epoch"] + 2, device="cpu", workers=0, amp=False, plots=False, val=False
        )
        assert epochs_seen == [state["epoch"] + 1]
        restored = new_manager(tmp_path)
        try:
            assert restored.get_job(job.job_id).model_dump() == job.model_dump()
        finally:
            restored.shutdown()
    finally:
        manager.shutdown()


def test_checkpoint_temp_hardlink_cannot_overwrite_external_file(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    outside = tmp_path.parent / (tmp_path.name + "-external.pt")
    outside.write_bytes(b"external-original")
    os.link(outside, weights / "shutdown.pt.tmp")
    os.link(outside, weights / "shutdown.pt")
    events = []
    control = TrainingShutdown(lambda *event: events.append(event))
    control.requested.set()
    control.acknowledged.set()
    trainer = SimpleNamespace(
        save_dir=tmp_path, epoch=0, stop=False, world_size=1, _serialize_checkpoint=lambda **kwargs: b"new-checkpoint"
    )
    control.on_epoch_end(trainer)
    assert outside.read_bytes() == b"external-original"
    assert (weights / "shutdown.pt").read_bytes() == b"new-checkpoint"
    assert events[0][0] == "checkpoint"


def test_checkpoint_write_rejects_weights_escape_before_serializing(tmp_path):
    # Directory junctions on Windows and symlinks on Linux exercise the same
    # canonical containment check without granting write authority from events.
    from unittest.mock import patch

    (tmp_path / "weights").mkdir()
    control = TrainingShutdown(lambda *args: None)
    control.requested.set()
    serialized = []
    trainer = SimpleNamespace(
        save_dir=tmp_path, world_size=1, _serialize_checkpoint=lambda **kwargs: serialized.append(True)
    )
    original_resolve = Path.resolve

    def escaping(path, *args, **kwargs):
        if path == tmp_path / "weights":
            return tmp_path.parent
        return original_resolve(path, *args, **kwargs)

    with patch.object(Path, "resolve", escaping), pytest.raises(ValueError):
        control.on_epoch_end(trainer)
    assert serialized == []


@pytest.mark.parametrize("scan_delay", [0, 0.004])
def test_result_waits_for_natural_exit_and_drains_beyond_queue_capacity(tmp_path, monkeypatch, scan_delay):
    from studio.worker_runtime import ManagedWorker

    manager = new_manager(tmp_path, grace=5)
    result_seen = threading.Event()
    original_receive = ManagedWorker.receive
    original_capture = ManagedWorker._capture_descendants

    def slow_capture(worker):
        time.sleep(scan_delay)
        return original_capture(worker)

    def receive(worker):
        kind, payload = original_receive(worker)
        if kind == "result":
            assert not (tmp_path / "finalized").exists()
            job = manager.get_job("shutdown-train")
            assert job.status == JobStatus.RUNNING and job.metadata.completed_at is None
            result_seen.set()
            (tmp_path / "result-received").touch()
        return kind, payload

    monkeypatch.setattr(ManagedWorker, "_capture_descendants", slow_capture)
    monkeypatch.setattr(ManagedWorker, "receive", receive)
    try:
        submit(manager, tmp_path, exit_delay=1, log_count=700, wait_for_result=True)
        wait_for(lambda: (tmp_path / "started").exists())
        start = time.monotonic()
        manager.shutdown()
        assert time.monotonic() - start >= 1
        assert result_seen.is_set() and (tmp_path / "acknowledged").exists()
        assert (tmp_path / "finalized").exists()
        logs = manager.get_job_log_lines("shutdown-train")
        assert sum("shutdown-stream-" in line for line in logs) == 700
        assert not any("SHUTDOWN_FORCED" in line for line in logs)
        assert manager.get_job("shutdown-train").error.code == "SERVICE_SHUTDOWN"
        assert_child_gone(tmp_path)
    finally:
        manager.shutdown()


def test_busy_ipc_samples_descendants_but_exit_checks_capture_immediately(tmp_path, monkeypatch):
    import studio.worker_runtime as runtime

    worker = runtime.ManagedWorker(JobRequest(job_id="busy-logs", task_type="train"))
    clock, captures = [100.0], []
    monkeypatch.setattr(runtime, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(worker, "_capture_descendants", lambda: captures.append(clock[0]))
    monkeypatch.setattr(worker, "_descendants_alive", lambda: True)
    try:
        for index in range(700):
            worker._messages.put(("log", {"seq": index}))
            assert worker.receive() == ("log", {"seq": index})
        assert captures == [100.0]
        assert not worker.tree_exited()
        assert captures == [100.0, 100.0]
        clock[0] = 100.051
        worker._messages.put(("log", {"seq": 700}))
        assert worker.receive() == ("log", {"seq": 700})
        assert captures == [100.0, 100.0, 100.051]
    finally:
        worker.close()


def test_shutdown_cleanup_failure_retains_owner_until_retry(tmp_path, monkeypatch):
    from studio.worker_runtime import ManagedWorker

    manager = new_manager(tmp_path, grace=3)
    released = threading.Event()
    original = ManagedWorker.stop

    def fail_until_released(worker, grace):
        if not released.is_set():
            raise OSError("injected cleanup failure password=hidden-value")
        return original(worker, grace)

    monkeypatch.setattr(ManagedWorker, "stop", fail_until_released)
    try:
        submit(manager, tmp_path)
        wait_for(lambda: (tmp_path / "started").exists())
        with monkeypatch.context() as budget:
            budget.setattr(JobsManager, "shutdown_budget_seconds", property(lambda self: 0.5))
            with pytest.raises(RuntimeError, match="ownership has been retained"):
                manager.shutdown()
        wait_for(lambda: manager.get_job("shutdown-train").error is not None)
        job = manager.get_job("shutdown-train")
        assert job.status == JobStatus.RUNNING and job.error.code == "WORKER_STOP_FAILED"
        assert job.metadata.completed_at is None and "shutdown-train" in manager._workers
        assert "hidden-value" not in job.error.message
    finally:
        released.set()
        manager.shutdown()
    assert manager.get_job("shutdown-train").error.code == "SERVICE_SHUTDOWN"
    assert not manager._workers


def test_accepted_cancel_wins_during_cooperative_shutdown(tmp_path, monkeypatch):
    from studio.worker_runtime import ManagedWorker

    manager = new_manager(tmp_path, grace=5)
    stop_entered, allow_cleanup = threading.Event(), threading.Event()
    stop_calls, errors, identities = [], [], []
    original_stop = ManagedWorker.stop

    def observed_stop(worker, grace):
        stop_calls.append(grace)
        stop_entered.set()
        assert allow_cleanup.wait(5)
        return original_stop(worker, grace)

    def close():
        try:
            manager.shutdown()
        except BaseException as error:  # noqa: BLE001 - Forward shutdown thread failures to the assertions.
            errors.append(error)

    monkeypatch.setattr(ManagedWorker, "stop", observed_stop)
    shutdown = threading.Thread(target=close)
    try:
        submit(manager, tmp_path, linger=True)
        wait_for(lambda: (tmp_path / "started").exists())
        root = psutil.Process(manager._workers["shutdown-train"].process.pid)
        identities = [root, *root.children(recursive=True)]
        manager.submit_job_request(
            JobRequest(job_id="queued", task_type="predict", output={"output_dir": str(tmp_path)})
        )
        shutdown.start()
        wait_for(lambda: (tmp_path / "acknowledged").exists())
        manager.request_cancel("shutdown-train")
        assert stop_entered.wait(5)
        assert stop_calls == [manager._stop_grace]
        pending = manager.get_job("shutdown-train")
        assert pending.status == JobStatus.RUNNING and pending.metadata.completed_at is None
        assert "shutdown-train" in manager._workers and any(process.is_running() for process in identities)
        assert manager.get_job("queued").metadata.started_at is None
        allow_cleanup.set()
        shutdown.join(10)
        assert not shutdown.is_alive() and not errors
        job = manager.get_job("shutdown-train")
        assert job.status == JobStatus.CANCELLED and job.error.code == "USER_CANCELLED"
        assert job.output.artifacts == [CHECKPOINT_ID] and not manager._workers
        assert not any(process.is_running() for process in identities)
        persisted = json.loads((tmp_path / "state.json").read_text())["jobs"][job.job_id]
        assert persisted["status"] == "cancelled" and persisted["metadata"]["completed_at"]
        (tmp_path / "cancel-grace-facts.json").write_text(
            json.dumps(
                {
                    "stop_graces": stop_calls,
                    "identities": [{"pid": process.pid, "created": process.create_time()} for process in identities],
                    "identities_gone": True,
                    "terminal": job.model_dump(mode="json"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        allow_cleanup.set()
        manager.shutdown()
        if shutdown.ident is not None:
            shutdown.join(10)


@pytest.mark.parametrize(
    ("mode", "now", "expected_graces"),
    [
        ("shutdown", 101.0, [0]),
        ("shutdown_deadline", 110.0, [0]),
        ("cancel_full", 101.0, [2.0]),
        ("cancel_at_deadline", 110.0, [2.0]),
        ("cancel_partial", 111.5, [0.5]),
        ("cancel_exhausted", 112.5, [0]),
        ("cancel_late_result", 101.0, [2.0]),
        ("cancel_retries", 101.0, [2.0, 0.5, 0]),
    ],
)
def test_shutdown_cancel_cleanup_grace_uses_remaining_budget(tmp_path, monkeypatch, mode, now, expected_graces):
    """Drive real owner arbitration at fixed clock/barrier boundaries; no OS exit simulation claim."""
    import studio.jobs_manager as module

    manager = JobsManager(
        storage_path=tmp_path / "state.json",
        output_root=tmp_path,
        cpu_concurrency=1,
        gpu_concurrency=1,
        stop_grace_seconds=2,
        shutdown_grace_seconds=10,
    )
    clock = [100.0]
    receiving, allow_closing = threading.Event(), threading.Event()
    cooperation, allow_cooperation = threading.Event(), threading.Event()
    late_result, allow_result = threading.Event(), threading.Event()
    stopping, allow_cleanup = threading.Event(), threading.Event()
    stop_calls, closed, errors = [], [], []

    class Worker:
        def __init__(self, job, *_):
            self.result = job.model_copy(deep=True)
            self.result.status = JobStatus.COMPLETED
            self.process = SimpleNamespace(is_alive=lambda: True)

        def start(self):
            return None

        def request_shutdown(self):
            cooperation.set()
            assert allow_cooperation.wait(5)

        def receive(self):
            if not manager._closing:
                receiving.set()
                assert allow_closing.wait(5)
                return None, None
            if mode == "cancel_late_result":
                late_result.set()
                assert allow_result.wait(5)
            return "result", self.result.model_dump(mode="json")

        def tree_exited(self):
            return mode == "shutdown"

        def stop(self, grace):
            stop_calls.append(grace)
            if mode == "cancel_retries" and len(stop_calls) < 3:
                clock[0] = 111.5 if len(stop_calls) == 1 else 112.5
                raise OSError("Deterministic cleanup retry")
            stopping.set()
            assert allow_cleanup.wait(5)

        def close(self):
            closed.append(True)

    def close():
        try:
            manager.shutdown()
        except BaseException as error:  # noqa: BLE001 - Forward shutdown thread failures to the assertions.
            errors.append(error)

    monkeypatch.setattr(module, "ManagedWorker", Worker)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=time.sleep))
    shutdown = threading.Thread(target=close)
    try:
        submit(manager, tmp_path)
        assert receiving.wait(5)
        manager.submit_job_request(
            JobRequest(job_id="queued", task_type="predict", output={"output_dir": str(tmp_path)})
        )
        shutdown.start()
        wait_for(lambda: manager._closing)
        allow_closing.set()
        assert cooperation.wait(5)
        assert manager._shutdown_deadline == 110.0 and manager.shutdown_budget_seconds == 28.0
        if mode == "cancel_late_result":
            allow_cooperation.set()
            assert late_result.wait(5)
        if mode.startswith("cancel"):
            manager.request_cancel("shutdown-train")
        clock[0] = now
        allow_cooperation.set()
        allow_result.set()
        assert stopping.wait(5)
        assert stop_calls == expected_graces
        snapshot = manager.get_job("shutdown-train")
        assert snapshot.status == JobStatus.RUNNING and snapshot.metadata.completed_at is None
        assert "shutdown-train" in manager._workers and not closed
        if mode == "cancel_retries":
            assert snapshot.error.code == "WORKER_STOP_FAILED"
        persisted = json.loads((tmp_path / "state.json").read_text())["jobs"]["shutdown-train"]
        assert persisted["status"] == "running" and persisted["metadata"]["completed_at"] is None
        queued = manager.get_job("queued")
        assert queued.error.code == "SERVICE_SHUTDOWN" and queued.metadata.started_at is None
        allow_cleanup.set()
        shutdown.join(5)
        assert not shutdown.is_alive() and not errors
        result = manager.get_job("shutdown-train")
        assert result.status == (JobStatus.CANCELLED if mode.startswith("cancel") else JobStatus.FAILED)
        assert result.error.code == ("USER_CANCELLED" if mode.startswith("cancel") else "SERVICE_SHUTDOWN")
        assert result.metadata.completed_at and not manager._workers and closed == [True]
        assert manager._shutdown_deadline == 110.0
        restored = JobsManager(storage_path=tmp_path / "state.json", output_root=tmp_path)
        try:
            assert restored.get_job(result.job_id).model_dump() == result.model_dump()
            assert not restored._workers
        finally:
            restored.shutdown()
    finally:
        allow_closing.set()
        allow_cooperation.set()
        allow_result.set()
        allow_cleanup.set()
        manager.shutdown()
        if shutdown.ident is not None:
            shutdown.join(5)


def natural_result_executor(job):
    root = Path(job.output.output_dir)

    def finish():
        time.sleep(1)
        (root / "teardown.finished").touch()

    threading.Thread(target=finish).start()
    job.status = JobStatus.COMPLETED
    return job


def test_closing_between_receive_and_result_cleanup_uses_grace(tmp_path, monkeypatch):
    from studio.worker_runtime import ManagedWorker

    manager = new_manager(tmp_path, grace=5)
    manager._worker_executor = natural_result_executor
    received, release = threading.Event(), threading.Event()
    original_receive = ManagedWorker.receive

    def paused_receive(worker):
        event = original_receive(worker)
        if event[0] == "result":
            received.set()
            assert release.wait(5)
        return event

    monkeypatch.setattr(ManagedWorker, "receive", paused_receive)
    shutdown = threading.Thread(target=manager.shutdown)
    try:
        submit(manager, tmp_path)
        assert received.wait(15)
        shutdown.start()
        wait_for(lambda: manager._closing)
        release.set()
        shutdown.join(10)
        assert not shutdown.is_alive()
        assert (tmp_path / "teardown.finished").exists()
        assert manager.get_job("shutdown-train").error.code == "SERVICE_SHUTDOWN"
        logs = manager.get_job_logs("shutdown-train")
        assert "SHUTDOWN_REQUESTED" in logs and "SHUTDOWN_FORCED" not in logs
    finally:
        release.set()
        manager.shutdown()
        if shutdown.ident is not None:
            shutdown.join(10)


def loader_facts(trainer):
    """Record actual loader processes, not the requested workers setting."""
    import json

    facts = {
        "device": str(trainer.device),
        "amp": bool(trainer.amp),
        "epoch": getattr(trainer, "epoch", None),
        "loaders": {},
        "trainer_source": __import__("inspect").getfile(type(trainer)),
        "torch_version": __import__("torch").__version__,
    }
    for name in ("train_loader", "test_loader"):
        loader = getattr(trainer, name)
        workers = getattr(loader.iterator, "_workers", ())
        facts["loaders"][name] = {
            "num_workers": loader.num_workers,
            "workers": [
                {
                    "pid": process.pid,
                    "created": psutil.Process(process.pid).create_time(),
                    "ppid": psutil.Process(process.pid).ppid(),
                    "status": psutil.Process(process.pid).status(),
                }
                for process in workers
            ],
        }
    # Ensure facts are JSON compatible before publishing the readiness marker.
    json.dumps(facts)
    return facts


def real_loader_executor(job):
    """Exercise the actual handler and upstream GPU loaders; inject only timing faults."""
    import json

    import torch

    from ultralytics import YOLO
    from ultralytics.utils import checks, downloads

    torch.set_num_threads(2)

    def reject(*args, **kwargs):
        raise AssertionError("No downloads permitted during loader acceptance")

    downloads.safe_download = downloads.download = reject
    checks.check_font = lambda *args, **kwargs: None
    root = Path(job.output.output_dir)
    mode = job.params["loader_fault"]

    def local_yolo(path):
        model = YOLO(path)
        original_train = model.train

        def consumed(trainer):
            if (root / "loader-ready.json").exists():
                return
            facts = loader_facts(trainer)
            facts["batch_consumed"] = True
            temporary = root / "loader-ready.tmp"
            temporary.write_text(json.dumps(facts), encoding="utf-8")
            temporary.replace(root / "loader-ready.json")
            # Freeze immediately after a real optimizer batch, while workers live.
            control = CURRENT_SHUTDOWN.get()
            wait_for(control.requested.is_set, timeout=180)
            if mode == "before_checkpoint":
                while True:
                    time.sleep(0.02)

        def after_checkpoint(trainer):
            control = CURRENT_SHUTDOWN.get()
            if mode == "after_ack" and control.saved:
                assert control.acknowledged.is_set()
                (root / "loader-acknowledged").touch()
                while True:
                    time.sleep(0.02)

        model.add_callback("on_train_batch_end", consumed)

        def bounded_train(**kwargs):
            # Handler installs its checkpoint callback before invoking this wrapper.
            model.add_callback("on_fit_epoch_end", after_checkpoint)
            return original_train(**kwargs, workers=1, amp=False, plots=False, mosaic=0)

        model.train = bounded_train
        return model

    import ultralytics

    ultralytics.YOLO = local_yolo
    return execute_job(job)


def assert_process_identities_gone(identities):
    """Require disappearance, including zombies; a recycled PID is a different process."""
    for identity in identities:
        try:
            process = psutil.Process(identity["pid"])
            assert process.create_time() != identity["created"], (identity, process.status())
        except psutil.NoSuchProcess:
            pass


@pytest.mark.slow
@pytest.mark.studio_integration
@pytest.mark.parametrize("mode", ["cooperative", "before_checkpoint", "after_ack"])
def test_real_loader_shutdown_deadline_and_resume(tmp_path, mode):
    """Require live workers, real batches, bounded shutdown, exact manifest and real resume."""
    import hashlib
    import json

    import torch
    import yaml
    from PIL import Image

    from ultralytics import YOLO

    device = os.environ.get("STUDIO_LOADER_DEVICE", "0")
    assert torch.cuda.is_available(), "This acceptance requires the supported CUDA workers>0 path"
    model = Path(os.environ["STUDIO_TEST_MODEL"]).resolve(strict=True)
    for split in ("train", "val"):
        images, labels = tmp_path / "images" / split, tmp_path / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for index in range(8):
            Image.new("RGB", (64, 64), "white").save(images / f"{index}.jpg")
            (labels / f"{index}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    data = tmp_path / "data.yaml"
    data.write_text(
        yaml.safe_dump({"path": str(tmp_path), "train": "images/train", "val": "images/val", "names": {0: "object"}})
    )
    grace = {"before_checkpoint": 1, "after_ack": 15, "cooperative": 60}[mode]
    manager = new_manager(tmp_path, grace=grace)
    manager._admission.model_roots = [model.parent]
    manager._worker_executor = real_loader_executor
    facts = {"mode": mode, "grace": grace}
    try:
        submit(
            manager,
            tmp_path,
            device=device,
            loader_fault=mode,
            model_path=str(model),
            data_source=str(data),
            epochs=10,
            batch_size=2,
            imgsz=64,
        )
        wait_for(lambda: (tmp_path / "loader-ready.json").exists(), timeout=180)
        facts["training"] = json.loads((tmp_path / "loader-ready.json").read_text(encoding="utf-8"))
        for loader in facts["training"]["loaders"].values():
            assert loader["num_workers"] > 0 and len(loader["workers"]) == loader["num_workers"]
        worker = manager._workers["shutdown-train"]
        owner = psutil.Process(worker.process.pid)
        identities = [
            {"pid": process.pid, "created": process.create_time()}
            for process in [owner, *owner.children(recursive=True)]
        ]
        loader_pids = {w["pid"] for loader in facts["training"]["loaders"].values() for w in loader["workers"]}
        assert loader_pids.issubset({p["pid"] for p in identities})
        facts["owned_tree"] = identities
        observations, failures = [], []

        def close():
            try:
                manager.shutdown()
            except BaseException as error:  # noqa: BLE001 - Forward thread failures to the test assertion.
                failures.append(error)

        started = time.monotonic()
        closing = threading.Thread(target=close)
        closing.start()
        while closing.is_alive():
            with manager.lock:
                snapshot = manager._snapshot(manager._jobs["shutdown-train"])
                if snapshot.status == JobStatus.RUNNING:
                    assert "shutdown-train" in manager._workers
            observations.append({"elapsed": time.monotonic() - started, "status": snapshot.status.value})
            if snapshot.status == JobStatus.FAILED:
                assert_process_identities_gone(identities)
            assert time.monotonic() - started < manager.shutdown_budget_seconds + 15
            closing.join(0.02)
        assert not failures, failures
        facts["shutdown_elapsed"] = time.monotonic() - started
        facts["observations"] = observations
        assert_process_identities_gone(identities)
        job = manager.get_job("shutdown-train")
        assert job.status == JobStatus.FAILED and job.error.code == "SERVICE_SHUTDOWN"
        assert job.metadata.completed_at and not job.runtime_tracking.cancel_requested and not manager._workers
        logs = manager.get_job_logs(job.job_id)
        (tmp_path / "shutdown.log").write_text(logs, encoding="utf-8")
        assert ("SHUTDOWN_FORCED" in logs) == (mode != "cooperative")
        if mode != "cooperative":
            assert facts["shutdown_elapsed"] >= grace
        checkpoint = manager.resolve_artifact(job.job_id, CHECKPOINT_ID)
        if mode == "before_checkpoint":
            assert checkpoint is None and job.output.artifacts == []
            assert "CHECKPOINT_CONFIRMED" not in logs
        else:
            assert checkpoint is not None and job.output.artifacts == [CHECKPOINT_ID]
            assert "CHECKPOINT_CONFIRMED" in logs
            if mode == "after_ack":
                assert (tmp_path / "loader-acknowledged").exists()
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert 0 <= state["epoch"] < 9
            assert all(state[key] is not None for key in ("model", "optimizer", "scaler", "ema", "updates"))
            assert state["optimizer"]["state"] and state["updates"] > 0
            facts["checkpoint"] = {
                "epoch": state["epoch"],
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "optimizer_states": len(state["optimizer"]["state"]),
                "scaler": state["scaler"],
                "updates": state["updates"],
            }
            YOLO(str(checkpoint))
            resumed = YOLO(str(checkpoint))
            epochs_seen, resume_tree = [], []

            def resumed_started(trainer):
                facts["resume"] = loader_facts(trainer)
                facts["resume"]["start_epoch"] = trainer.start_epoch
                facts["resume"]["optimizer_states"] = len(trainer.optimizer.state)
                actual = trainer.optimizer.state_dict()["state"]
                assert actual.keys() == state["optimizer"]["state"].keys()
                for key, expected in state["optimizer"]["state"].items():
                    for name, value in expected.items():
                        restored_value = actual[key][name]
                        if isinstance(value, torch.Tensor):
                            assert torch.equal(restored_value.cpu().float(), value.cpu().float())
                        else:
                            assert restored_value == value
                facts["resume"]["optimizer_values_equal"] = True
                facts["resume"]["ema_updates"] = trainer.ema.updates
                assert trainer.ema.updates == state["updates"]
                resume_tree.extend(facts["resume"]["loaders"]["train_loader"]["workers"])
                resume_tree.extend(facts["resume"]["loaders"]["test_loader"]["workers"])

            resumed.add_callback("on_train_start", resumed_started)
            resumed.add_callback("on_train_epoch_end", lambda trainer: epochs_seen.append(trainer.epoch))
            resumed.train(
                resume=True, epochs=state["epoch"] + 2, device=device, workers=1, amp=False, plots=False, val=False
            )
            assert epochs_seen == [state["epoch"] + 1]
            assert facts["resume"]["start_epoch"] == state["epoch"] + 1
            assert facts["resume"]["optimizer_states"] == len(state["optimizer"]["state"])
            assert facts["resume"]["loaders"]["train_loader"]["num_workers"] > 0
            assert_process_identities_gone(resume_tree)
            facts["resume"]["epochs_seen"] = epochs_seen
        restored = new_manager(tmp_path)
        try:
            assert restored.get_job(job.job_id).model_dump() == job.model_dump()
        finally:
            restored.shutdown()
        facts["terminal"] = job.model_dump(mode="json")
        facts["result"] = "PASS"
        (tmp_path / "loader-facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    finally:
        manager.shutdown()
