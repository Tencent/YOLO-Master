# Studio Core Task Engine

Core translates a validated job into a synchronous Ultralytics engine call.
It supplies the `JobRequest` input model, path validation, credential sanitization,
a dispatcher that tracks execution state, and handlers for `train`, `val`,
`predict`, `export` and `diagnose`. Install with `python -m pip install -e ".[studio]"`.

## Dispatch and handlers

Import `studio.dispatcher.JobDispatcherStateMachine` or `studio.handlers` before
concurrent execution. Importing handlers registers all five classes; extensions
belong in `studio/handlers/__init__.py` and must be registered before dispatch
threads start. The separate `f1.handlers` registry serves legacy direct calls.

Direct handler calls have two steps: `validate_params(params, constraints)`, then
`execute(job_id, params, output_dir)`. Execution returns a dictionary containing
`success`, `artifacts`, `metadata` and `error`. The dispatcher returns an updated
`JobRequest`. Adding a public task requires a corresponding schema/API contract
change as well as handler registration.

## Trusted caller and safety boundary

Core is an in-process library for trusted callers. The caller supplies path roots
and regex rules and validates `output_dir` and the final job directory. Handlers
validate model/data paths; direct `execute()` calls assume validated inputs.
The dispatcher requires `allow_shell=false` and `path_whitelisted=true`.
Containment resolves symlinks and rejects broken links, sibling-prefix escapes
and malformed-only regex rules. Empty roots/patterns reject input paths.

Returned artifact paths are candidates for a runtime's authorized file list.
The consuming runtime checks containment before authorizing or reading them.
`append_log` and dispatcher error/traceback handling sanitize credentials;
direct handler results require sanitization before public responses or storage.
See [security](security.md) for the production service's access rules.

The legacy direct dispatcher may report cancellation/timeout while a detached
daemon thread still computes. Use [Runtime](runtime.md) for managed process-tree
cleanup, queues and persistence. Core's `managed=True` path executes synchronously
under external supervision.

## Upstream compatibility

`TrainHandler` forwards explicitly supplied `imgsz` after the same integer
normalization used by val/export. Omitting it preserves the upstream default.
Optimizer metadata summarizes `trainer.optimizer_group_audit`, recorded after
upstream optimizer construction. A missing snapshot reports `audited=false`.

## Verification

Use the [offline lane](testing.md#offline-core-runtime-and-service-lane) for unit
checks and [local-model integration](testing.md#explicit-local-model-integration)
for real engine calls. Core integration uses generated images/data, one CPU
training epoch at `imgsz=64` and TorchScript export. Service lifecycle and
checkpoint resume are covered by the separate production integration tests.
