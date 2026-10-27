# Studio Runtime

Runtime schedules jobs and runs the synchronous Core engine in managed worker
processes on a trusted local machine. It owns job state, process cleanup and
persistence. See [architecture](architecture.md) for the service integration and
[security](security.md) for deployment limits.

```python
from core.schema import JobRequest
from studio.jobs_manager import JobsManager

def main():
    manager = JobsManager(
        storage_path="runs/jobs_state.json",
        model_roots=["models"],
        data_roots=["datasets"],
        output_root="runs",
    )
    try:
        submitted = manager.submit_job_request(
            JobRequest(job_id="local-diagnose", task_type="diagnose", output={"output_dir": "runs"})
        )
        snapshot = manager.get_job(submitted.job_id)
    finally:
        manager.shutdown()


if __name__ == "__main__":
    main()
```

Applications using spawn must construct and call the manager inside an
`if __name__ == "__main__":` guard. Submission returns a detached, sanitized
snapshot. Queries also return copies captured under the manager lock. Worker
results never replace the parent record: the manager commits execution status,
error, sanitized logs and a validated manifest after confirming process-tree exit.
Worker changes to submission inputs, paths, policy or lifecycle timestamps are
discarded. `request_cancel()` returns a `CancelDecision` with a `CancelCode` and
snapshot captured under the same lock; adapters should use its code rather than
parse `cancel_job()`'s compatibility message.

`JobsManager` alone decides admission registration, status, timestamps, terminal
publication, restart reconciliation, final manifest and logical slot release.
`AdmissionPolicy` validates inputs, `JobStore` restores valid records, `JobLogs`
orders and buffers inter-process communication (IPC) logs, `ArtifactResolver` resolves manifest entries, and
`ManagedWorker` reports containment and exit facts. Helpers make no lifecycle
decisions.

## Capacity and stopping

Defaults are two CPU slots, one GPU slot, a shared capacity of 100 pending jobs,
and two seconds of stop grace. Set server-owned `STUDIO_CPU_CONCURRENCY`,
`STUDIO_GPU_CONCURRENCY`, `STUDIO_MAX_PENDING_JOBS`, and
`STUDIO_STOP_GRACE_SECONDS`, or pass constructor arguments. Exact `cpu` devices
use CPU slots; auto, CUDA, multi-GPU and MPS share GPU slots. A fixed set of
supervisor threads consumes queues; waiting jobs have no dedicated process or
thread. Each active worker also has one IPC reader with a bounded 256-message
queue, keeping partial frames outside deadline and cleanup loops. The capacity check and registration occur under one lock. Execution
timeouts start at slot assignment, excluding queue time.

Workers always use spawn. Windows places the gated worker in a kill-on-close Job
Object before execution starts. Linux uses a session/process group, subreaper
guardian and parent-death watcher, including descendants that create their own
sessions. Completion, cancellation, timeout, worker loss, launch failure and
shutdown all require tree cleanup before terminal publication or slot reuse.
`WORKER_STOP_FAILED` retains the worker handle and occupied slot while cleanup
retries; `shutdown()` raises if supervisors still own cleanup after its deadline.

Pending cancellation publishes `cancelled / USER_CANCELLED` without launch.
Running cancellation is accepted while still running, then published only after
cleanup. Shutdown publishes `failed / SERVICE_SHUTDOWN` and never marks the
parent request as user-cancelled. Arbitration preserves an already-published
terminal record, then accepted user cancellation, then service closing, then an
observed timeout, worker failure or result. Stop windows continue to drain IPC
logs. Single-process training additionally uses the checkpoint protocol below.
Cancellation and timeout use bounded process cleanup. Checkpoint cooperation
applies to single-process training during service shutdown;
DistributedDataParallel (DDP) uses deadline-based cleanup.

## Training shutdown and recovery

`shutdown()` rejects admission and closes pending jobs, then requests cooperative
shutdown for running training. A separate stop message/token preserves the
distinction from user cancellation. The IPC watcher only updates tokens; the
training thread's `on_fit_epoch_end` callback saves after the epoch's training,
validation and metrics, before `final_eval()` can strip recovery state.

The callback uses the upstream checkpoint serializer with the online model
aligned with optimizer state, plus EMA, updates and scaler. It verifies the owned
job directory and weights containment before writing an exclusive random
temporary file, flushes/fsyncs and atomically replaces `weights/shutdown.pt`.
This preserves a recovery copy independently of stripped `last.pt`/`best.pt`.
It reports a checkpoint fact (relative ID, epoch, size) and waits for the parent
to validate exact containment and size and acknowledge it. Parent validation
does not deserialize worker-supplied pickle files. The acknowledgment records
the checkpoint's location and size. To check recovery, load it and run the
resume procedure below.

Throughout cooperation, the owner drains ordered, sanitized IPC. It buffers log
updates in memory instead of doing a JSON atomic replacement per line; checkpoint
confirmation and final publication flush that tail. The existing persistence
failure boundary still applies. A worker result is provisional: the owner waits
for actual process-tree exit, including Python teardown and descendants. Only the
absolute grace deadline permits forced shutdown cleanup. Accepted user
cancellation can supersede shutdown. The owner rechecks cancellation after
cooperation and before each cleanup attempt, restoring `stop_grace_seconds`.
During closing this wait is capped by the first shutdown's cooperation deadline
plus its reserved stop window; elapsed retries reduce the remaining wait to zero.
The budget's 16 seconds for OS cleanup and handle closure are not added to this
cancellation wait. Cancellation never renews the shutdown deadline.
Cleanup failure retains the owner and slot as `WORKER_STOP_FAILED` while cleanup
retries. `shutdown()` raises if cleanup remains unfinished at its deadline.

After cleanup, the manager revalidates the checkpoint candidate and commits only
that exact ID to the shutdown manifest. It publishes `failed / SERVICE_SHUTDOWN`,
including when the checkpoint is saved, acknowledged and resumable. It never
converts checkpoint success into public `completed` or `USER_CANCELLED`.
Child dispatcher completion logs remain provisional execution facts. Directory
contents are not extra download authority. A forced stop can still retain a
previously confirmed checkpoint, but logs distinguish it from natural tree exit.
Without confirmation the shutdown manifest grants no checkpoint access.

Set server-owned `STUDIO_SHUTDOWN_GRACE_SECONDS` or constructor
`shutdown_grace_seconds` (default 60, finite range 0 through 3600). The first
shutdown request anchors one deadline for all active training jobs. Existing
`stop_grace_seconds` still controls cancel/timeout cleanup. An outer caller must
allow at least `manager.shutdown_budget_seconds` (default 78 seconds) and handle
raised cleanup failures rather than kill the manager early. This budget includes
the grace and bounded OS cleanup; it does not bound persistent cleanup retries.

The [Service lifespan](service.md#history-and-lifecycle) and
[combined launcher](frontend.md#trusted-local-configuration) consume this shutdown
budget.

Single-process CPU and single-device training share the callback protocol.
CPU training forces dataloader workers to zero upstream; verification of
dataloader child shutdown therefore needs a device/configuration that uses them.
Multiple devices or `WORLD_SIZE > 1` skip the outer checkpoint callback;
the owned tree is forcibly cleaned if it is still running at the deadline.
Checkpoint cooperation runs at epoch boundaries in single-process training.

To validate recovery of a trusted local checkpoint, inspect its nonnegative
`epoch`, online model, optimizer, scaler and EMA fields, load it using
`YOLO(path)`, then train with `resume=True` and a total epoch count greater than
`saved_epoch + 1` (at least `saved_epoch + 2`). Verify the next epoch actually executes.
Resume follows upstream epoch-boundary semantics; in-flight batches and exact
RNG/dataloader positions are not retained. Restarting the manager preserves the
already-published shutdown terminal; it never
automatically resumes training.

## Persistence and recovery

`jobs_state.json` and `jobs_state.json.bak` contain version-1 JSON snapshots.
The store validates every `JobRequest`, dictionary identity and log list, then
sanitizes and serializes before touching disk. Writes use sibling `.tmp` files,
flush, best-effort fsync and atomic replacement. Only a schema-valid primary may
rotate into the backup, so a damaged primary cannot overwrite a good backup.

A valid primary takes precedence. A missing/invalid primary with a valid backup
is repaired from that backup. Two absent snapshots mean a new environment;
existing snapshots with no valid copy raise `StateRecoveryError`. Broken snapshot
symlinks count as existing invalid history; inaccessible existence checks fail
closed. Legacy records
without optional execution timestamps remain supported. The manager changes
restored pending/running jobs to `failed / SERVICE_RESTARTED` and persists this
reconciliation before startup succeeds. It never relaunches those jobs or signals
historical PIDs. With a configured store, submission must persist before starting
supervisors or entering a queue. A failure raises `StatePersistenceError`
(`PERSISTENCE_FAILED`) and removes the unaccepted record and submission log;
the same identifier can be retried. In-memory mode (`storage_path=None`) stores
state only in memory. After acceptance, runtime write failures are exposed
through `persistence_error`;
owned process cleanup continues, and later mutations retry persistence. Such a
failure leaves the disk at its last successfully saved snapshot. Accepted
cancellation and actual execution results remain valid in memory even when their
snapshot cannot be written; a later successful mutation clears the error.

Result fields are prepared before terminal publication under the owner lock.
An unexpected artifact-normalization error publishes a complete
`failed / EXECUTION_FAILED` record after worker cleanup, with a completion time
and no newly authorized manifest entries. Expected invalid candidates continue
to be rejected by the resolver without granting file access.

## Paths, logs and artifacts

Model, data and output roots are server configuration; see
[service settings](service.md#trusted-configuration-and-existing-ui-entry-points).
Network hosts use comma-separated `STUDIO_NETWORK_INPUT_HOSTS`. Admission
canonicalizes local model/data paths before execution and rejects credential-shaped
job IDs and URL userinfo. The [security guide](security.md#paths-and-manifest-authorization)
describes path containment and artifact authorization.

Runtime artifact IDs are exact job-relative paths, including nested names.
The [HTTP service](api.md#log-cursors-and-artifacts) uses those manifest IDs and
checks containment and opened-file identity before streaming.

Logs are sanitized, ordered by child sequence and deduplicated. Terminal and
stop tails remain in an independent buffer, so late logs never mutate a published
terminal job snapshot. `get_job_log_page(job_id, offset, limit)` returns copied
entries; `next_offset=None` marks the current tail. Later logs remain available
through subsequent polls. A caller continuing after a tail read resumes from
`offset + len(logs)`.

## Verification

Use the commands and prerequisites in [testing](testing.md). Process tests use
importable server-owned executors and real child/grandchild processes, with
separate deterministic race tests. Parent dispatcher patches do not propagate
automatically into spawned children.

Implementation references: [Python multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
and [psutil process APIs](https://psutil.readthedocs.io/).
