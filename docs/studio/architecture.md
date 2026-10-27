# Studio architecture

[Studio documentation](README.md)

Studio separates browser interaction, HTTP handling and job execution. A job is
described by the shared `core.schema.JobRequest` model and runs in a worker
process managed by the API service.

```text
React / built-in console / Gradio Jobs
                 | HTTP
main_engine.py + api/v1/jobs.py
                 |
studio.jobs_manager.JobsManager
                 |
CPU/GPU queues -> studio.worker_runtime.ManagedWorker
                 |
studio.dispatcher -> studio.handlers -> Ultralytics engines
```

## Layer responsibilities

- [Core](core.md) validates job inputs, removes credentials from public text and
  dispatches each task to its handler. Handlers adapt synchronous calls to the
  Ultralytics engines. Direct callers supply trusted policy and validate inputs.
- [Runtime](runtime.md) admits and schedules jobs, owns public status and
  timestamps, persists history and authorizes output files in an artifact
  manifest. `AdmissionPolicy`, `JobStore`, `JobLogs` and `ArtifactResolver`
  support these operations. `ManagedWorker` contains the OS process tree and
  reports its outcome; the manager confirms cleanup before publishing a final
  status or releasing a slot.
- [Service](service.md) creates one manager during application startup and waits
  for its cleanup during shutdown (FastAPI's lifespan). HTTP requests read
  snapshots and cancellation decisions from that manager.
- [Clients](frontend.md) submit and monitor jobs through HTTP. Gradio's
  **Inference Studio** retains a separate synchronous inference path beside Jobs.

## Compatibility

Production workers use `studio.dispatcher` and the `studio.handlers` registry.
The retained `f1` backend and direct-call adapters are separate compatibility
implementations. `f1/ui` provides lazy compatibility imports for `studio/ui`.

The service accepts legacy `F1_*` configuration aliases and retains the health
identity `YOLO-Master F1 Task Engine`. Version-1 snapshots and safe historical
artifact entries remain readable; see [service compatibility](service.md#history-and-lifecycle).
Each state file has one service owner. Restart reconciles interrupted jobs
without running them again.

`agent/` has its own runtime, dispatcher and Skills. It does not participate in
Studio job execution. Deployment and access limits are documented in
[security](security.md).
