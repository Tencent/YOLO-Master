"""Owned, spawn-based Studio workers and OS process-tree containment.

No YOLO code runs before the parent's start handshake. Windows uses a non-inherited
kill-on-close Job Object; POSIX uses a new session. Results are provisional until
the parent has stopped the entire tree. Only server code supplies the executor.
"""

from __future__ import annotations

import ctypes
import multiprocessing
import os
import queue
import signal
import sys
import threading
import time
import traceback
from ctypes import wintypes

import psutil

from core.schema import ErrorInfo, JobRequest, JobStatus
from core.security import sanitize_log_text
from studio.training_shutdown import CURRENT_SHUTDOWN, TrainingShutdown


def execute_job(job: JobRequest) -> JobRequest:
    """Run the existing Studio dispatcher synchronously inside an owned worker."""
    from studio.dispatcher import JobDispatcherStateMachine

    return JobDispatcherStateMachine().execute(job, managed=True)


def _compute_job(connection, stop, raw_job, executor):
    """Run task code in a process with no shared synchronization locks."""
    if os.name != "nt":
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    job = JobRequest.model_validate(raw_job)
    stopped = threading.Event()
    send_lock = threading.Lock()

    def emit(kind, payload):
        with send_lock:
            connection.send((kind, payload))

    shutdown = TrainingShutdown(emit, job.output.output_dir, job.job_id)
    context_token = CURRENT_SHUTDOWN.set(shutdown)

    if job.runtime_tracking.stream_logs:

        def emit_log(sequence, text, terminal):
            """Send one sanitized log entry over the existing result pipe."""
            try:
                with send_lock:
                    connection.send(("log", {"seq": sequence, "text": text, "terminal": terminal}))
            except (BrokenPipeError, EOFError, OSError):
                pass

        job._set_log_event_sink(emit_log)

    def watch_stop():
        while True:
            try:
                command = stop.recv_bytes()
            except EOFError:
                command = b"stop"
            if command == b"shutdown":
                shutdown.requested.set()
                stopped.set()
            elif command == b"checkpoint-accepted":
                shutdown.acknowledged.set()
            else:
                job.runtime_tracking.cancel_requested = True
                stopped.set()
                return

    threading.Thread(target=watch_stop, daemon=True).start()
    try:
        result = executor(job)
    except BaseException as exc:  # noqa: BLE001 - report even SystemExit at the process boundary
        job.status = JobStatus.FAILED
        job.error = ErrorInfo(code="EXECUTION_FAILED", message=sanitize_log_text(str(exc)))
        job.append_log(traceback.format_exc())
        result = job
    job._set_log_event_sink(None)
    CURRENT_SHUTDOWN.reset(context_token)
    emit("result", result.model_dump(mode="json"))
    if shutdown.requested.is_set():
        return
    stopped.wait()
    if shutdown.requested.is_set():
        return
    while True:
        time.sleep(1)


def _descendants(pid):
    """Find descendants even if torchrun placed ranks in different sessions."""
    try:
        return psutil.Process(pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def _signal_process(process, operation):
    """Use psutil's identity checks to avoid signaling a reused PID."""
    try:
        getattr(process, operation)()
    except psutil.NoSuchProcess:
        pass


def _kill_descendants(pid):
    """Freeze the Linux subreaper's descendants before killing, closing spawn races."""
    frozen = set()
    while True:
        children = _descendants(pid)
        new = [child for child in children if child.pid not in frozen]
        if not new:
            break
        for child in new:
            _signal_process(child, "suspend")
            frozen.add(child.pid)
    for child in children:
        _signal_process(child, "kill")
    return children


def _worker_main(connection, stop, raw_job, executor, parent_pid):
    """Gate computation behind containment; Linux root remains a subreaper guardian."""
    if os.name != "nt":
        os.setsid()
    linux = sys.platform == "linux"
    if linux:
        libc = ctypes.CDLL(None, use_errno=True)
        # PR_SET_CHILD_SUBREAPER: orphaned ranks/loaders reparent here even after setsid().
        if libc.prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "Cannot establish worker subreaper")
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    # Windows handle closure kills the Job Object on parent death. POSIX needs
    # a parent watchdog too, including while YOLO blocks the main Python thread.
    def watch_parent():
        while True:
            if os.getppid() != parent_pid:
                if linux:
                    _kill_descendants(os.getpid())
                if os.name != "nt":
                    os.killpg(os.getpid(), signal.SIGKILL)
                os._exit(1)
            time.sleep(0.1)

    threading.Thread(target=watch_parent, daemon=True).start()
    connection.send(("ready", None))
    if connection.recv() != "start":
        return
    if not linux:
        _compute_job(connection, stop, raw_job, executor)
        return
    compute = multiprocessing.get_context("spawn").Process(
        target=_compute_job,
        args=(connection, stop, raw_job, executor),
        daemon=False,
    )
    compute.start()
    # Only compute owns the result endpoint now. Retaining a guardian writer
    # would hide EOF when compute dies in the middle of a frame.
    connection.close()
    while True:
        compute.join(timeout=0.05)
        # Reap adopted children without stealing multiprocessing's direct child.
        for child in psutil.Process().children():
            if child.pid != compute.pid:
                try:
                    os.waitpid(child.pid, os.WNOHANG)
                except ChildProcessError:
                    pass
        if compute.exitcode is not None and not psutil.Process().children():
            return
        time.sleep(0.05)


class _WindowsJob:
    """Minimal typed Win32 Job Object binding; no breakaway is permitted."""

    def __init__(self):
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, args, result in (
            ("CreateJobObjectW", [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            (
                "SetInformationJobObject",
                [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD],
                wintypes.BOOL,
            ),
            (
                "QueryInformationJobObject",
                [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p],
                wintypes.BOOL,
            ),
            ("OpenProcess", [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            ("AssignProcessToJobObject", [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            ("TerminateJobObject", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
        ):
            function = getattr(self.api, name)
            function.argtypes, function.restype = args, result

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("process_time", ctypes.c_int64),
                ("job_time", ctypes.c_int64),
                ("flags", wintypes.DWORD),
                ("min_ws", ctypes.c_size_t),
                ("max_ws", ctypes.c_size_t),
                ("active_limit", wintypes.DWORD),
                ("affinity", ctypes.c_size_t),
                ("priority", wintypes.DWORD),
                ("scheduling", wintypes.DWORD),
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("basic", BasicLimits),
                ("io", ctypes.c_uint64 * 6),
                ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t),
                ("peak_process", ctypes.c_size_t),
                ("peak_job", ctypes.c_size_t),
            ]

        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        try:
            self._check(self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
        except OSError:
            self.close()
            raise

    @staticmethod
    def _check(ok):
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, pid):
        """Assign a gated worker before allowing any task computation."""
        handle = self.api.OpenProcess(0x0100 | 0x0001, False, pid)  # SET_QUOTA | TERMINATE
        self._check(handle)
        try:
            self._check(self.api.AssignProcessToJobObject(self.handle, handle))
        finally:
            self.api.CloseHandle(handle)

    def active(self):
        """Return kernel-owned active process count, including all descendants."""

        class Accounting(ctypes.Structure):
            _fields_ = [
                ("times", ctypes.c_int64 * 4),
                ("faults", wintypes.DWORD),
                ("total", wintypes.DWORD),
                ("active", wintypes.DWORD),
                ("terminated", wintypes.DWORD),
            ]

        info = Accounting()
        self._check(self.api.QueryInformationJobObject(self.handle, 1, ctypes.byref(info), ctypes.sizeof(info), None))
        return info.active

    def kill(self):
        """Terminate every process in the job, including racing child spawns."""
        self._check(self.api.TerminateJobObject(self.handle, 1))

    def close(self):
        """Release the parent's sole Job Object handle."""
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


class ManagedWorker:
    """One process tree; the owning supervisor alone calls start/stop/close."""

    def __init__(self, job, executor=execute_job):
        context = multiprocessing.get_context("spawn")
        self.connection, self._child_connection = context.Pipe()
        # A killed process can strand a multiprocessing.Event's shared lock.
        # A private one-way pipe does not require acquiring a child-owned lock.
        self._stop_receiver, self._stop_sender = context.Pipe(duplex=False)
        self._stop_sent = False
        self.process = context.Process(
            target=_worker_main,
            args=(self._child_connection, self._stop_receiver, job.model_dump(mode="json"), executor, os.getpid()),
            name=f"studio-{job.job_id}",
            daemon=False,  # YOLO must be allowed to create dataloader/DDP children.
        )
        self._group_ready = False
        self._windows_job = None
        self._assigned = False
        self._descendants = {}
        self.log_sink = None
        self._messages = queue.Queue(maxsize=256)
        self._reader = None
        self._next_descendant_capture = 0

    def start(self):
        """Create the root, still blocked on the containment handshake."""
        if os.name == "nt":
            self._windows_job = _WindowsJob()
        self.process.start()
        self._child_connection.close()
        self._stop_receiver.close()
        if self._windows_job:
            self._windows_job.assign(self.process.pid)
            self._assigned = True
        reader = threading.Thread(target=self._read_messages, name=f"studio-ipc-{self.process.pid}", daemon=True)
        reader.start()
        self._reader = reader

    def request_shutdown(self):
        """Send a distinct service-shutdown request without setting user cancellation."""
        self._stop_sender.send_bytes(b"shutdown")

    def acknowledge_checkpoint(self):
        """Release the training callback after the owner validates its checkpoint fact."""
        self._stop_sender.send_bytes(b"checkpoint-accepted")

    def _read_messages(self):
        """Isolate blocking frame reads from supervision, deadlines and OS cleanup."""
        try:
            while True:
                self._messages.put(self.connection.recv())
        except Exception:  # noqa: BLE001 - malformed or truncated worker IPC is worker loss
            self._messages.put(("lost", None))

    def receive(self):
        """Read one queued event without blocking on a partial Pipe frame."""
        # A full process scan per queued log can starve checkpoint ACKs.
        # Exit/stop checks still capture immediately; busy IPC samples every poll interval.
        if time.monotonic() >= self._next_descendant_capture:
            self._capture_descendants()
            self._next_descendant_capture = time.monotonic() + 0.05
        try:
            kind, payload = self._messages.get(timeout=0.05)
        except queue.Empty:
            return None, None
        if kind == "ready":
            self._group_ready = True
            self.connection.send("start")
        return kind, payload

    def _capture_descendants(self):
        if os.name != "nt" and self.process.pid is not None:
            for child in _descendants(self.process.pid):
                self._descendants[child.pid] = child

    def _descendants_alive(self):
        """Keep ownership of matching identities until reaped, including zombies."""
        for child in self._descendants.values():
            try:
                if child.is_running():
                    return True
            except psutil.ZombieProcess:
                # ZombieProcess subclasses NoSuchProcess but does not prove reap.
                return True
            except psutil.NoSuchProcess:
                pass
        return False

    def _group_alive(self):
        self._capture_descendants()
        if self._descendants_alive():
            return True
        if self._windows_job and self._assigned:
            return bool(self._windows_job.active())
        if os.name == "nt":
            return self.process.is_alive()
        if not self._group_ready:
            # setsid may have happened before the ready message was consumed.
            try:
                self._group_ready = os.getpgid(self.process.pid) == self.process.pid
            except ProcessLookupError:
                pass
        if self._group_ready:
            for process in psutil.process_iter(["pid"]):
                try:
                    # The parent can reap its own root via multiprocessing;
                    # every other group identity still needs its actual reaper.
                    if os.getpgid(process.pid) == self.process.pid and (
                        process.pid != self.process.pid or self.process.is_alive()
                    ):
                        return True
                except (ProcessLookupError, psutil.NoSuchProcess):
                    continue
            return False
        return self.process.is_alive()

    def tree_exited(self):
        """Report actual owned tree exit; a result message alone is not exit evidence."""
        return not self._group_alive()

    def stop(self, grace_seconds):
        """Request stop, wait grace, force kill, and verify tree exit before returning.

        Failure deliberately raises: callers must retain the occupied slot and
        worker handle and retry, never publish a false terminal status.
        """
        if self.process.pid is None:
            return
        if not self._stop_sent:
            self._stop_sent = True
            try:
                self._stop_sender.send_bytes(b"stop")
            except OSError:
                pass
        # Continue draining IPC during the bounded stop window. This is process
        # cleanup/log delivery, not a training-checkpoint shutdown guarantee.
        deadline = time.monotonic() + grace_seconds
        while self._group_alive() and time.monotonic() < deadline:
            self._drain_logs()
            time.sleep(0.02)
        self._drain_logs()
        alive = self._group_alive()
        for child in self._descendants.values():
            _signal_process(child, "terminate")
        if os.name != "nt" and self._group_ready and alive:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if self._group_alive():
            if self._windows_job and self._assigned:
                self._windows_job.kill()
            elif self._group_ready:
                # On Linux the still-live subreaper retains children that escape
                # our session. Freeze/kill those before killing the guardian.
                if sys.platform == "linux" and self.process.is_alive():
                    for child in _kill_descendants(self.process.pid):
                        # Include ranks spawned during the grace period in the
                        # final exit verification, not just earlier snapshots.
                        self._descendants[child.pid] = child
                for child in self._descendants.values():
                    _signal_process(child, "kill")
                child_deadline = time.monotonic() + 5
                while self._descendants_alive() and time.monotonic() < child_deadline:
                    time.sleep(0.02)
                if self._descendants_alive():
                    raise RuntimeError("Worker descendants have not been reaped; guardian retained")
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                self.process.kill()
        self._drain_logs()
        self.process.join(timeout=5)
        deadline = time.monotonic() + 5
        while self._group_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        if self.process.is_alive() or self._group_alive():
            raise RuntimeError("Worker process tree has not exited")
        self._drain_logs()

    def _drain_logs(self):
        """Drain bounded queued IPC; a truncated frame cannot block tree cleanup."""
        for _ in range(256):
            try:
                kind, payload = self._messages.get_nowait()
            except queue.Empty:
                return
            if kind == "log" and self.log_sink is not None:
                self.log_sink(payload)

    def close(self):
        """Release IPC and OS handles only after stop has confirmed exit."""
        if self._reader is not None:
            self._drain_logs()
            self._reader.join(timeout=1)
            if self._reader.is_alive():
                raise RuntimeError("Worker IPC reader has not exited; ownership retained")
            self._drain_logs()
        if self._windows_job:
            self._windows_job.close()
        self.connection.close()
        self._child_connection.close()
        self._stop_receiver.close()
        self._stop_sender.close()
        self.process.close()
