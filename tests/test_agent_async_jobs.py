"""Cross-platform lifecycle checks using disposable workers, never user processes."""

import importlib
import io
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
worker = importlib.import_module("agent.scripts.run_async_job")


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


def fake_submission(tmp_path, monkeypatch, request):
    """Submit only to a mocked supervisor; keep all task files inside tmp_path."""
    process = Mock()
    process.create_time.return_value = 1
    monkeypatch.setattr(jobs.psutil, "Process", Mock(return_value=process))
    monkeypatch.setattr(jobs, "ensure_manifest_dir", lambda request: tmp_path)
    create = Mock(return_value=Mock(pid=12345))
    monkeypatch.setattr(jobs.subprocess, "Popen", create)
    submitted = jobs.AsyncJobManager(tmp_path / "jobs").submit("fixture", request)
    return submitted, create


def test_private_execution_request_preserves_auth_and_public_files_are_redacted(tmp_path, monkeypatch):
    """The audit copy is safe, while the private execution input retains exact credential values."""
    secret = "SYNTHETIC_REQUEST_CREDENTIAL"
    submitted, create = fake_submission(
        tmp_path,
        monkeypatch,
        {
            "inputs": {"api_key": secret},
            "runtime": {"headers": {"Authorization": f"Bearer {secret}"}},
            "params": {"description": secret, "max_output_tokens": 128},
            "policy": {"async": True},
        },
    )
    job_dir = Path(submitted["request_path"]).parent
    execution_path = job_dir / jobs.EXECUTION_REQUEST_NAME
    actual = json.loads(execution_path.read_text())
    assert actual["inputs"]["api_key"] == secret
    assert actual["runtime"]["headers"]["Authorization"] == f"Bearer {secret}"
    assert actual["policy"]["async"] is False
    audit = json.loads(Path(submitted["request_path"]).read_text())
    assert audit["params"]["max_output_tokens"] == 128
    assert create.call_args.args[0][-1] == str(execution_path)
    assert secret not in repr(create.call_args)
    for path in job_dir.iterdir():
        if path != execution_path:
            assert secret not in path.read_text()
    if os.name == "posix":
        assert job_dir.stat().st_mode & 0o777 == 0o700
        assert execution_path.stat().st_mode & 0o777 == 0o600


class RecordingInput(io.StringIO):
    """Keep a copy of the dispatcher input after the supervisor closes its pipe."""

    def close(self):
        if not self.closed:
            self.payload = self.getvalue()
        super().close()


@pytest.mark.parametrize("returncode", [0, 7])
def test_worker_consumes_private_request_cleans_it_and_scrubs_streams(tmp_path, monkeypatch, capsys, returncode):
    """Both outcomes preserve request/env authentication while publishing only redacted logs."""
    secret = "SYNTHETIC_INLINE_CREDENTIAL"
    environment_secret = "SYNTHETIC_ENV_CREDENTIAL"
    monkeypatch.setenv("REVIEW_API_KEY", environment_secret)
    execution_path = tmp_path / jobs.EXECUTION_REQUEST_NAME
    payload = {"inputs": {"api_key": secret}, "runtime": {"headers": {"Authorization": f"Bearer {secret}"}}}
    jobs._atomic_write_json(execution_path, payload)
    incoming = RecordingInput()
    child = Mock(
        stdin=incoming,
        stdout=io.StringIO(f"raw value {secret}\nAPI env {environment_secret}\n"),
        stderr=io.StringIO(f"failed request {secret}\n"),
    )
    child.wait.return_value = returncode
    child.poll.return_value = returncode

    def create(args, **kwargs):
        assert not execution_path.exists(), "Remove the private file before launching the dispatcher"
        assert "--request" not in args
        assert kwargs.get("env") is None, "Keep provider authentication inherited from the caller environment"
        assert os.environ["REVIEW_API_KEY"] == environment_secret
        return child

    monkeypatch.setattr(worker.subprocess, "Popen", create)
    assert worker.run(tmp_path, tmp_path / "dispatcher.py", execution_path) == (0 if returncode == 0 else 1)
    assert json.loads(incoming.payload) == payload
    assert not execution_path.exists()
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["returncode"] == returncode
    assert result["status"] == ("completed" if returncode == 0 else "failed")
    captured = capsys.readouterr()
    public = captured.out + captured.err + (tmp_path / "result.json").read_text()
    assert secret not in public and environment_secret not in public


def test_worker_launch_failure_removes_private_input_and_redacts_error(tmp_path, monkeypatch):
    """A failed dispatcher launch cannot retain its execution credential or expose it in result.json."""
    secret = "SYNTHETIC_FAILURE_CREDENTIAL"
    execution_path = tmp_path / jobs.EXECUTION_REQUEST_NAME
    jobs._atomic_write_json(execution_path, {"inputs": {"api_key": secret}})
    monkeypatch.setattr(worker.subprocess, "Popen", Mock(side_effect=OSError(f"launch failed {secret}")))
    assert worker.run(tmp_path, tmp_path / "dispatcher.py", execution_path) == 1
    assert not execution_path.exists()
    assert secret not in (tmp_path / "result.json").read_text()


def test_worker_preserves_fast_exit_when_stdin_write_and_close_raise(tmp_path, monkeypatch):
    """A dispatcher that exits before reading stdin still gets its actual exit status."""
    execution_path = tmp_path / jobs.EXECUTION_REQUEST_NAME
    jobs._atomic_write_json(execution_path, {"inputs": {"api_key": "SYNTHETIC_FAST_EXIT_CREDENTIAL"}})
    incoming = Mock()
    incoming.write.side_effect = BrokenPipeError("child has exited")
    incoming.close.side_effect = BrokenPipeError("flush after child exit")
    child = Mock(stdin=incoming, stdout=io.StringIO(""), stderr=io.StringIO(""))
    child.wait.return_value = child.poll.return_value = 7
    monkeypatch.setattr(worker.subprocess, "Popen", Mock(return_value=child))
    assert worker.run(tmp_path, tmp_path / "dispatcher.py", execution_path) == 1
    assert json.loads((tmp_path / "result.json").read_text())["returncode"] == 7
    assert not execution_path.exists()


def test_submission_launch_failure_removes_private_input(tmp_path, monkeypatch):
    """If the supervisor cannot start, only the redacted audit remains on disk."""
    secret = "SYNTHETIC_SUBMIT_CREDENTIAL"
    monkeypatch.setattr(jobs, "ensure_manifest_dir", lambda request: tmp_path)
    monkeypatch.setattr(jobs.subprocess, "Popen", Mock(side_effect=OSError("fixture launch failure")))
    with pytest.raises(OSError, match="fixture launch failure"):
        jobs.AsyncJobManager(tmp_path / "jobs").submit("fixture", {"inputs": {"api_key": secret}})
    assert not list(tmp_path.rglob(jobs.EXECUTION_REQUEST_NAME))
    assert all(secret not in path.read_text() for path in tmp_path.rglob("*") if path.is_file())


def test_submission_status_failure_cleans_private_input_and_owned_worker(tmp_path, monkeypatch):
    """A post-launch persistence error must not retain credentials or owned computation."""
    original = jobs._atomic_write_json

    def write(path, payload):
        if path.name == "status.json":
            raise OSError("fixture status failure")
        original(path, payload)

    stop = Mock(return_value=True)
    monkeypatch.setattr(jobs, "_atomic_write_json", write)
    monkeypatch.setattr(jobs.AsyncJobManager, "_terminate_process_tree", staticmethod(stop))
    with pytest.raises(OSError, match="fixture status failure"):
        fake_submission(tmp_path, monkeypatch, {"inputs": {"api_key": "SYNTHETIC_PERSIST_CREDENTIAL"}})
    stop.assert_called_once()
    assert not list(tmp_path.rglob(jobs.EXECUTION_REQUEST_NAME))


def test_cancel_cleans_unconsumed_private_request(tmp_path, monkeypatch):
    """Confirmed cancellation removes a private file even if the supervisor never consumed it."""
    submitted, _ = fake_submission(tmp_path, monkeypatch, {"inputs": {"api_key": "SYNTHETIC_CANCEL_CREDENTIAL"}})
    process = Mock()
    process.children.return_value = []
    monkeypatch.setattr(jobs.AsyncJobManager, "_process", staticmethod(lambda status: process))
    monkeypatch.setattr(jobs.psutil, "wait_procs", lambda processes, timeout: (processes, []))
    manager = jobs.AsyncJobManager(tmp_path / "jobs")
    assert manager.cancel(submitted["job_id"])["cancelled"]
    assert not list(tmp_path.rglob(jobs.EXECUTION_REQUEST_NAME))
