"""Bounded runtime with one parent-owned public lifecycle decision maker.

Worker copies report provisional outcomes. Only JobsManager commits lifecycle,
timestamps, final manifests and logical slot release after process-tree cleanup.
"""

from __future__ import annotations

import atexit
import math
import os
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from core.schema import TERMINAL_STATUSES, ErrorInfo, JobRequest, JobStatus
from studio.admission import AdmissionPolicy
from studio.artifacts import IMAGE_EXTENSIONS, ArtifactResolver
from studio.job_logs import JobLogs, sanitize_log_text, sanitize_snapshot_value
from studio.job_store import JobStore
from studio.training_shutdown import CHECKPOINT_ID
from studio.worker_runtime import ManagedWorker, execute_job

MAX_PENDING_JOBS = 100
MAX_PENDING_JOBS_ENV = "STUDIO_MAX_PENDING_JOBS"


class QueueFullError(RuntimeError):
    """The atomic pending-capacity check rejected admission."""

    code = "QUEUE_FULL"


class StatePersistenceError(RuntimeError):
    """Admission could not be persisted and has not been accepted or queued."""

    code = "PERSISTENCE_FAILED"


class CancelCode(str, Enum):
    """Stable decisions for transport adapters; messages are presentation only."""

    ACCEPTED = "accepted"
    NOT_FOUND = "not_found"
    ALREADY_TERMINAL = "already_terminal"
    NOT_CANCELLABLE = "not_cancellable"


@dataclass(frozen=True)
class CancelDecision:
    """A cancellation decision and detached record captured under one lock."""

    code: CancelCode
    job: JobRequest | None
    message: str


def compute_duration(started_at, completed_at):
    """Return execution seconds, excluding queue wait and preserving zero."""
    if not started_at or not completed_at:
        return None
    try:
        return (datetime.fromisoformat(completed_at) - datetime.fromisoformat(started_at)).total_seconds()
    except (ValueError, TypeError):
        return None


class JobsManager:
    """Own bounded scheduling, public state decisions and restart reconciliation."""

    def __init__(
        self,
        storage_path=None,
        *,
        model_roots=None,
        data_roots=None,
        output_root=None,
        network_input_hosts=None,
        cpu_concurrency=None,
        gpu_concurrency=None,
        max_pending_jobs=None,
        stop_grace_seconds=None,
        shutdown_grace_seconds=None,
    ):
        self._jobs = {}
        self._logs = JobLogs()
        self.lock = threading.Lock()
        self._limits = {
            "cpu": int(os.environ.get("STUDIO_CPU_CONCURRENCY", "2")) if cpu_concurrency is None else cpu_concurrency,
            "gpu": int(os.environ.get("STUDIO_GPU_CONCURRENCY", "1")) if gpu_concurrency is None else gpu_concurrency,
        }
        if any(type(value) is not int or value < 1 for value in self._limits.values()):
            raise ValueError("CPU/GPU concurrency must be positive integers")
        self.max_pending_jobs = (
            int(os.environ.get(MAX_PENDING_JOBS_ENV, str(MAX_PENDING_JOBS)))
            if max_pending_jobs is None
            else max_pending_jobs
        )
        if type(self.max_pending_jobs) is not int or self.max_pending_jobs < 1:
            raise ValueError("Maximum pending jobs must be a positive integer")
        self._stop_grace = (
            float(os.environ.get("STUDIO_STOP_GRACE_SECONDS", "2"))
            if stop_grace_seconds is None
            else stop_grace_seconds
        )
        if not 0 <= self._stop_grace <= 30:
            raise ValueError("Stop grace must be between 0 and 30 seconds")
        self._shutdown_grace = (
            float(os.environ.get("STUDIO_SHUTDOWN_GRACE_SECONDS", "60"))
            if shutdown_grace_seconds is None
            else float(shutdown_grace_seconds)
        )
        if not math.isfinite(self._shutdown_grace) or not 0 <= self._shutdown_grace <= 3600:
            raise ValueError("Shutdown grace must be finite and between 0 and 3600 seconds")
        self._shutdown_deadline = None
        self._queues = {resource: queue.Queue() for resource in self._limits}
        self._supervisors = []
        self._workers = {}
        self._closing = False
        self._worker_executor = execute_job  # Importable, server-owned injection seam, never request data.
        self._admission = AdmissionPolicy(model_roots, data_roots, output_root, network_input_hosts)
        self._artifacts = ArtifactResolver(self._admission.output_root)
        self._store = JobStore(storage_path)
        self.persistence_error = None
        self._load()

    @staticmethod
    def _snapshot(job):
        return JobRequest.model_validate(sanitize_snapshot_value(job.model_dump(mode="json"))) if job else None

    def _save(self, *, required=False):
        """Require durable admission; report later write failures without losing workers."""
        try:
            self._store.save(self._jobs, self._logs.entries)
            self.persistence_error = None
        except (OSError, ValueError, TypeError) as exc:
            self.persistence_error = sanitize_log_text(str(exc))
            if required:
                raise StatePersistenceError(f"Job admission could not be persisted: {self.persistence_error}") from None

    def _load(self):
        """Reconcile restored active records without launching or signaling old workers."""
        payload = self._store.load()
        self._jobs = {key: JobRequest.model_validate(raw) for key, raw in payload["jobs"].items()}
        self._logs.entries.update(payload.get("job_logs", {}))
        for job in self._jobs.values():
            if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
                job.status = JobStatus.FAILED
                job.metadata.completed_at = datetime.now(timezone.utc).isoformat()
                job.error = ErrorInfo(
                    code="SERVICE_RESTARTED",
                    message="Service restarted; the previous job has no owned worker and will not be resumed",
                )
                job.append_log(f"[SERVICE_RESTARTED] {job.error.message}")
                self._logs.append(job.job_id, f"[SERVICE_RESTARTED] {job.error.message}")
        # Startup must not succeed if reconciliation cannot be made durable.
        if self._store.path is not None and (self._jobs or self._store.path.exists()):
            self._store.save(self._jobs, self._logs.entries)

    def submit_job_request(self, request):
        """Atomically admit a detached request and return a sanitized snapshot."""
        job = self._admission.prepare(request)
        with self.lock:
            if self._closing:
                raise ValueError("Job manager is shutting down")
            if job.job_id in self._jobs:
                raise ValueError(f"Duplicate job_id '{job.job_id}': a job with this identifier already exists")
            pending = sum(current.status == JobStatus.PENDING for current in self._jobs.values())
            if pending >= self.max_pending_jobs:
                raise QueueFullError(f"Pending job capacity exhausted ({pending}/{self.max_pending_jobs})")
            job.status, job.error, job.logs = JobStatus.PENDING, None, []
            job.metadata.created_at = datetime.now(timezone.utc).isoformat()
            job.metadata.started_at = job.metadata.completed_at = None
            job.runtime_tracking.cancel_requested = False
            job.output.artifacts = []
            self._jobs[job.job_id] = job
            previous_logs = self._logs.entries.get(job.job_id)
            self._logs.entries[job.job_id] = list(previous_logs or ())
            self._logs.append(job.job_id, f"[{job.metadata.created_at}] Job {job.job_id} submitted")
            try:
                self._save(required=True)
            except StatePersistenceError:
                self._jobs.pop(job.job_id)
                if previous_logs is None:
                    self._logs.entries.pop(job.job_id, None)
                else:
                    self._logs.entries[job.job_id] = previous_logs
                raise
            try:
                self._start_supervisors()
            except Exception as exc:
                self._fail_job(job, "EXECUTION_FAILED", "Supervisor launch failed")
                self._closing = True
                self._shutdown_deadline = time.monotonic() + self._shutdown_grace
                for resource, capacity in self._limits.items():
                    for _ in range(capacity):
                        self._queues[resource].put(None)
                atexit.register(self.shutdown)
                raise RuntimeError("Supervisor launch failed; manager is closing") from exc
            self._queues[self._resource_class(job)].put(job.job_id)
            return self._snapshot(job)

    @staticmethod
    def _resource_class(job: JobRequest) -> str:
        """Explicit CPU uses CPU slots; auto, CUDA, multi-GPU and MPS share GPU slots."""
        device = job.params.get("device", "cpu")
        return "cpu" if str(device).strip().lower() == "cpu" else "gpu"

    def _start_supervisors(self) -> None:
        """Start a fixed number of supervisors once, while holding the manager lock."""
        if self._supervisors:
            return
        for resource, capacity in self._limits.items():
            for index in range(capacity):
                thread = threading.Thread(
                    target=self._consume_queue, args=(resource,), name=f"studio-{resource}-{index}", daemon=True
                )
                thread.start()
                self._supervisors.append(thread)
        atexit.register(self.shutdown)

    def _consume_queue(self, resource: str) -> None:
        """Own one capacity slot; no thread or process is allocated to waiting jobs."""
        pending = self._queues[resource]
        while True:
            job_id = pending.get()
            try:
                if job_id is None:
                    return
                try:
                    self._execute_job(job_id)
                except Exception as exc:
                    with self.lock:
                        if job_id in self._workers:
                            raise  # Never discard ownership of an unexpectedly live worker.
                        self._fail_job(self._jobs[job_id], "EXECUTION_FAILED", str(exc))
            finally:
                pending.task_done()

    def _fail_job(self, job: JobRequest, code: str, message: str) -> None:
        """Set a terminal result only when there is no live owned computation; lock held."""
        if job.status in TERMINAL_STATUSES:
            return
        job.status = JobStatus.FAILED
        job.metadata.completed_at = datetime.now(timezone.utc).isoformat()
        job.error = ErrorInfo(code=code, message=sanitize_log_text(message))
        job.append_log(f"[{code}] {message}")
        self._logs.entries.setdefault(job.job_id, []).append(sanitize_log_text(f"[{code}] {message}"))
        self._save()

    def _cancel_job(self, job: JobRequest, message: str) -> None:
        """Publish confirmed cancellation after no owned computation remains; lock held."""
        if job.status in TERMINAL_STATUSES:
            return
        job.status = JobStatus.CANCELLED
        job.metadata.completed_at = datetime.now(timezone.utc).isoformat()
        job.error = ErrorInfo(code="USER_CANCELLED", message=sanitize_log_text(message))
        job.append_log(f"[USER_CANCELLED] {message}")
        self._logs.entries.setdefault(job.job_id, []).append(sanitize_log_text(f"[USER_CANCELLED] {message}"))
        self._save()

    def _execute_job(self, job_id: str) -> None:
        """Supervise one process through execution, tree cleanup and final persistence."""
        with self.lock:
            job = self._jobs.get(job_id)
            if not job or job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
                return
            if self._closing or job.runtime_tracking.cancel_requested:
                if job.runtime_tracking.cancel_requested:
                    self._cancel_job(job, "Job cancelled before worker launch")
                else:
                    self._fail_job(job, "SERVICE_SHUTDOWN", "Job stopped before worker launch")
                return
            job.metadata.started_at = datetime.now(timezone.utc).isoformat()
            try:
                worker = ManagedWorker(job, self._worker_executor)
            except Exception as exc:  # noqa: BLE001 - constructor failure must end admission honestly
                self._fail_job(job, "EXECUTION_FAILED", str(exc))
                return
            self._workers[job_id] = worker
            # The child's request remains PENDING for the existing dispatcher FSM.
            job.status = JobStatus.RUNNING
            self._save()
        result = None
        checkpoint = None
        code, message = None, None
        deadline = time.monotonic() + max(float(job.runtime_tracking.timeout_seconds), 0)

        def receive_log_event(payload, *, persist=True):
            with self.lock:
                self._logs.receive(job_id, payload)
                if persist:
                    self._save()

        worker.log_sink = receive_log_event

        try:
            worker.start()
            # Run our cleanup before multiprocessing's interpreter-exit join.
            atexit.unregister(self.shutdown)
            atexit.register(self.shutdown)
            while True:
                with self.lock:
                    if job.runtime_tracking.cancel_requested:
                        code, message = "USER_CANCELLED", "Job execution cancelled by user request"
                    elif self._closing:
                        code, message = "SERVICE_SHUTDOWN", "Service is shutting down"
                    elif time.monotonic() >= deadline:
                        code, message = "TIMEOUT", "Job execution exceeded its configured timeout"
                if code:
                    break
                kind, payload = worker.receive()
                if kind == "log":
                    receive_log_event(payload)
                    continue
                if kind == "shutdown_limit":
                    self._append_log(job_id, "[SHUTDOWN_LIMIT] DDP checkpoint cooperation unsupported")
                    continue
                if kind == "lost":
                    code, message = "WORKER_LOST", "Computation process exited without reporting a result"
                    break
                if kind == "result":
                    result = JobRequest.model_validate(payload)
                    if result.job_id != job_id or result.task_type != job.task_type:
                        code, message = "WORKER_LOST", "Worker returned an unrelated job identity"
                    elif result.status not in TERMINAL_STATUSES:
                        code, message = "WORKER_LOST", "Worker returned without a terminal result"
                    break
                if not worker.process.is_alive():
                    code, message = "WORKER_LOST", "Worker exited without reporting a result"
                    break
        except (EOFError, BrokenPipeError):
            code, message = "WORKER_LOST", "Worker connection closed without a result"
        except Exception as exc:  # noqa: BLE001 - parent must always clean up the tree
            code, message = "EXECUTION_FAILED", str(exc)
            self._append_log(job_id, traceback.format_exc())
        finally:
            # Closing/cancel can arrive while receive() returns a provisional
            # result. Re-arbitrate before choosing any process cleanup policy.
            with self.lock:
                current = self._jobs[job_id]
                if current.status not in TERMINAL_STATUSES:
                    if current.runtime_tracking.cancel_requested:
                        code, message = "USER_CANCELLED", "Job execution cancelled by user request"
                    elif self._closing:
                        code, message = "SERVICE_SHUTDOWN", "Service is shutting down"
            if code == "SERVICE_SHUTDOWN" and job.task_type.value == "train":
                # Per-line atomic writes can block drain beyond the grace/cleanup
                # budget. Checkpoint confirmation and terminal publication flush
                # the buffered, ordered/sanitized tail under the same owner lock.
                worker.log_sink = lambda payload: receive_log_event(payload, persist=False)
                try:
                    checkpoint = self._cooperate_training_shutdown(job_id, worker, worker.log_sink)
                except Exception as exc:  # noqa: BLE001 - shutdown must still clean up on broken IPC
                    self._append_log(job_id, f"[CHECKPOINT_UNCONFIRMED] {exc}")
            # Never detach a worker or release its resource slot on cleanup failure.
            # Keep RUNNING and a structured reason while retaining/retrying ownership.
            while True:
                try:
                    with self.lock:
                        if self._jobs[job_id].runtime_tracking.cancel_requested:
                            code, message = "USER_CANCELLED", "Job execution cancelled by user request"
                        grace = (
                            0
                            if code == "SERVICE_SHUTDOWN" and job.task_type.value == "train"
                            else self._stop_grace
                            if code
                            else 0
                        )
                        if self._closing and code == "USER_CANCELLED":
                            # Use the original cooperation + stop window, leaving
                            # the budget's 16 seconds for OS cleanup/handle close.
                            # Recompute on retries; cancellation never renews it.
                            remaining = self._shutdown_deadline + self._stop_grace - time.monotonic()
                            grace = min(grace, max(0, remaining))
                    worker.stop(grace)
                    break
                except Exception as exc:  # noqa: BLE001 - retain ownership on OS cleanup failure
                    with self.lock:
                        if self._jobs[job_id].status not in TERMINAL_STATUSES:
                            self._jobs[job_id].error = ErrorInfo(
                                code="WORKER_STOP_FAILED", message=sanitize_log_text(str(exc))
                            )
                        self._save()
                    time.sleep(0.2)
            while True:
                try:
                    worker.close()
                    break
                except Exception as exc:  # noqa: BLE001 - retain the slot until handles can be released
                    with self.lock:
                        if self._jobs[job_id].status not in TERMINAL_STATUSES:
                            self._jobs[job_id].error = ErrorInfo(
                                code="WORKER_STOP_FAILED", message=sanitize_log_text(str(exc))
                            )
                        self._save()
                    time.sleep(0.2)
            with self.lock:
                self._workers.pop(job_id, None)
                self._logs.finish(job_id, result.logs if result is not None else ())
                # Publish only from RUNNING. A natural terminal result already
                # published under this lock wins over a later cancellation.
                current = self._jobs[job_id]
                if current.status not in TERMINAL_STATUSES:
                    if current.runtime_tracking.cancel_requested:
                        code, message = "USER_CANCELLED", "Job execution cancelled by user request"
                    elif self._closing:
                        code, message = "SERVICE_SHUTDOWN", "Service is shutting down"
                    if checkpoint is not None:
                        # Revalidate after cleanup; a provisional fact never authorizes a live file.
                        try:
                            current.output.artifacts = self._artifacts.normalize(current, [checkpoint])
                        except Exception as exc:  # noqa: BLE001 - do not strand a cleaned job on resolver failure
                            self._logs.append(job_id, f"[CHECKPOINT_REJECTED] Final containment check failed: {exc}")
                            current.output.artifacts = []
                    if code == "USER_CANCELLED":
                        self._cancel_job(current, message)
                    elif code:
                        self._fail_job(current, code, message)
                    else:
                        # Prepare fallible fields before publishing any terminal state.
                        try:
                            error = (
                                ErrorInfo.model_validate(sanitize_snapshot_value(result.error.model_dump(mode="json")))
                                if result.error
                                else None
                            )
                            if result.status == JobStatus.FAILED and error is None:
                                error = ErrorInfo(
                                    code="EXECUTION_FAILED", message="Worker failed without an error detail"
                                )
                            logs = [sanitize_log_text(line) for line in result.logs]
                            artifacts = self._artifacts.normalize(current, result.output.artifacts)
                        except Exception as exc:  # noqa: BLE001 - finalization must publish a complete failure
                            self._fail_job(current, "EXECUTION_FAILED", f"Worker result finalization failed: {exc}")
                        else:
                            # Merge only provisional execution fields, never replace the owned record.
                            current.error = error
                            current.logs = logs
                            current.output.artifacts = artifacts
                            current.metadata.completed_at = datetime.now(timezone.utc).isoformat()
                            current.status = result.status
                self._save()

    def _cooperate_training_shutdown(self, job_id, worker, log_sink):
        """Drain facts until checkpoint ack plus execution exit, or the shared deadline."""
        try:
            worker.request_shutdown()
        except OSError:
            self._append_log(job_id, "[SHUTDOWN_IPC_CLOSED] Waiting for actual tree exit or deadline")
        checkpoint = None
        self._append_log(job_id, "[SHUTDOWN_REQUESTED] Waiting for a safe epoch checkpoint boundary")
        while time.monotonic() < self._shutdown_deadline:
            with self.lock:
                if self._jobs[job_id].runtime_tracking.cancel_requested:
                    return checkpoint
            kind, payload = worker.receive()
            if kind == "log":
                log_sink(payload)
                # Keep draining a busy log queue before scanning the whole tree again.
                continue
            elif kind == "shutdown_limit":
                self._append_log(job_id, "[SHUTDOWN_LIMIT] DDP checkpoint cooperation unsupported")
            elif kind == "checkpoint":
                with self.lock:
                    current = self._jobs[job_id]
                    try:
                        valid = (
                            isinstance(payload, dict)
                            and payload.get("path") == CHECKPOINT_ID
                            and type(payload.get("epoch")) is int
                            and payload["epoch"] >= 0
                            and type(payload.get("size")) is int
                            and payload["size"] > 0
                            and self._artifacts.normalize(current, [CHECKPOINT_ID]) == [CHECKPOINT_ID]
                        )
                        root = self._artifacts.root(current)
                        valid = valid and root is not None and (root / CHECKPOINT_ID).stat().st_size == payload["size"]
                    except Exception as exc:  # noqa: BLE001 - reject invalid facts without cutting grace short
                        valid = False
                        self._logs.append(job_id, f"[CHECKPOINT_REJECTED] Validation failed: {exc}")
                    if valid:
                        checkpoint = CHECKPOINT_ID
                        self._logs.append(
                            job_id, f"[CHECKPOINT_CONFIRMED] epoch={payload['epoch']} artifact={checkpoint}"
                        )
                        self._save()
                if valid:
                    try:
                        worker.acknowledge_checkpoint()
                    except OSError:
                        self._append_log(
                            job_id, "[SHUTDOWN_IPC_CLOSED] Checkpoint accepted; waiting for exit or deadline"
                        )
                else:
                    self._append_log(job_id, "[CHECKPOINT_REJECTED] Untrusted or incomplete checkpoint fact")
            elif kind in {"result", "lost"} and checkpoint is None:
                self._append_log(job_id, "[CHECKPOINT_UNCONFIRMED] Worker returned without an accepted checkpoint")
            # Result/EOF is provisional: Python teardown, loaders or descendants
            # may still be running. Let them exit cooperatively until the deadline.
            if worker.tree_exited():
                return checkpoint
        self._append_log(job_id, "[SHUTDOWN_FORCED] Grace deadline expired; forcing process-tree cleanup")
        return checkpoint

    @property
    def shutdown_budget_seconds(self):
        """Minimum outer caller budget for cooperation and bounded OS cleanup (excluding retries)."""
        return self._shutdown_grace + self._stop_grace + 16

    def shutdown(self) -> None:
        """Reject submissions, end pending jobs and join all owned execution slots."""
        with self.lock:
            if not self._closing:
                self._closing = True
                self._shutdown_deadline = time.monotonic() + self._shutdown_grace
                for job_id, job in self._jobs.items():
                    if job.status == JobStatus.PENDING and job_id not in self._workers:
                        self._fail_job(job, "SERVICE_SHUTDOWN", "Service stopped before worker launch")
                for resource, capacity in self._limits.items():
                    for _ in range(capacity):
                        self._queues[resource].put(None)
        deadline = time.monotonic() + self.shutdown_budget_seconds
        for thread in self._supervisors:
            thread.join(timeout=max(0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in self._supervisors):
            raise RuntimeError("Worker cleanup is still pending; ownership has been retained")
        atexit.unregister(self.shutdown)

    def _append_log(self, job_id, message):
        """Keep late logs separate from immutable terminal records."""
        with self.lock:
            self._logs.append(job_id, message)
            self._save()

    def get_job(self, job_id):
        """Return a detached sanitized record captured under the owner lock."""
        with self.lock:
            return self._snapshot(self._jobs.get(job_id))

    def get_job_status(self, job_id):
        """Read one coherent lifecycle summary, including execution-only duration."""
        with self.lock:
            job = self._snapshot(self._jobs.get(job_id))
            if job is None:
                return {"status": "NOT_FOUND", "message": "Job not found"}
            return {
                "status": job.status.value.upper(),
                "created_at": job.metadata.created_at,
                "started_at": job.metadata.started_at,
                "completed_at": job.metadata.completed_at,
                "duration": compute_duration(job.metadata.started_at, job.metadata.completed_at),
                "error_code": job.error.code if job.error else None,
                "error_message": job.error.message if job.error else None,
                "artifact_count": len(job.output.artifacts),
            }

    def get_job_log_lines(self, job_id):
        """Read a detached copy of live and terminal-tail log entries."""
        with self.lock:
            return list(self._logs.entries.get(job_id, []))

    def get_job_logs(self, job_id):
        """Read logs as text for local callers."""
        return "\n".join(self.get_job_log_lines(job_id)) or "No logs available"

    def get_job_log_page(self, job_id, offset=0, limit=500):
        """Read a coherent cursor page; reaching the tail does not stop logging."""
        with self.lock:
            return self._logs.page(job_id, offset, limit)

    def get_job_artifacts(self, job_id):
        """Return only exact manifest IDs that pass fresh containment checks."""
        with self.lock:
            job = self._jobs.get(job_id)
            return self._artifacts.list(job) if job else []

    def resolve_artifact(self, job_id, identifier):
        """Resolve a single exact manifest ID with read-time authorization."""
        with self.lock:
            job = self._jobs.get(job_id)
            return self._artifacts.resolve(job, identifier) if job else None

    def get_job_image_artifacts(self, job_id):
        """Return manifest IDs of validated image files without scanning directories."""
        return [
            identifier
            for identifier, path in self.get_job_artifacts(job_id)
            if Path(path).suffix.lower() in IMAGE_EXTENSIONS
        ]

    def request_cancel(self, job_id):
        """Decide cancellation atomically; accepted running cancellation stays active."""
        with self.lock:
            job = self._jobs.get(job_id)
            if job is None:
                return CancelDecision(CancelCode.NOT_FOUND, None, "Job not found")
            if job.status in TERMINAL_STATUSES:
                return CancelDecision(
                    CancelCode.ALREADY_TERMINAL,
                    self._snapshot(job),
                    f"Job already in terminal state: {job.status.value}",
                )
            if not job.runtime_tracking.cancellable:
                return CancelDecision(CancelCode.NOT_CANCELLABLE, self._snapshot(job), "Job is not cancellable")
            job.runtime_tracking.cancel_requested = True
            if job_id not in self._workers:
                self._cancel_job(job, "Job cancelled before worker launch")
            else:
                self._save()
            self._logs.append(job_id, f"[{datetime.now(timezone.utc).isoformat()}] Cancellation requested")
            self._save()
            return CancelDecision(CancelCode.ACCEPTED, self._snapshot(job), f"Cancellation requested for {job_id}")

    def cancel_job(self, job_id):
        """Compatibility text; adapters should consume request_cancel's typed decision."""
        return self.request_cancel(job_id).message

    def list_recent_jobs(self, limit=10):
        """Return detached summaries ordered by server acceptance timestamp."""
        with self.lock:
            return [
                {
                    "job_id": job.job_id,
                    "task_type": job.task_type.value,
                    "status": job.status.value.upper(),
                    "created_at": job.metadata.created_at,
                    "started_at": job.metadata.started_at,
                    "completed_at": job.metadata.completed_at,
                    "duration": compute_duration(job.metadata.started_at, job.metadata.completed_at),
                }
                for job in sorted(self._jobs.values(), key=lambda job: job.metadata.created_at, reverse=True)[:limit]
            ]
