# Legacy F1 task handlers

The F1 handler package adapts the canonical job contract to the existing
Ultralytics YOLO Python API. Handlers validate task parameters, call the engine,
and return a structured result and artifact manifest. Queueing, process ownership,
timeouts, terminal publication, and persistence belong to `JobsManager`.

Use this reference for the legacy `f1.handlers` package. For production
handlers and registration, see [Studio Core](../../docs/studio/core.md).

## Runtime position

```text
Legacy direct caller -> f1 dispatcher -> f1 registry -> f1 handler
                                                               |
                                                               v
                                                    Ultralytics YOLO API
```

Direct dispatcher/handler calls support tests and embedding. Use Studio Runtime
when process-tree cleanup is required; see [architecture](../../docs/studio/architecture.md).

## Package structure

```text
f1/handlers/
|-- __init__.py       # imports legacy handlers and triggers registration
|-- base.py           # BaseTaskHandler and validation/cancellation helpers
|-- registry.py       # TaskHandlerRegistry
|-- predict.py        # predict handler
|-- train.py          # train handler
|-- val.py            # validation handler
|-- export.py         # model export handler
|-- diagnose.py       # system diagnostics handler
|-- demo_handlers.py  # direct-use examples, not the production entry point
|-- README.md
`-- USAGE.md
```

Importing `f1.handlers` registers exactly these public task types:

```text
diagnose | export | predict | train | val
```

`core/schema.py::TaskType` is the API contract. Adding another registry entry is
not sufficient to expose a new public task: the schema, API/client surfaces, task
catalog, documentation, and tests must also be updated.

## Handler contract

Every concrete handler inherits `BaseTaskHandler` and implements:

```python
def validate_params(
    self,
    params: dict[str, Any],
    security_constraints: dict[str, Any],
) -> tuple[bool, str | None]: ...


def execute(
    self,
    job_id: str,
    params: dict[str, Any],
    output_dir: str,
) -> dict[str, Any]: ...
```

`execute()` returns:

```python
{
    "success": bool,
    "artifacts": list[str],
    "metadata": dict[str, Any],
    "error": str | None,
}
```

Artifacts are existing files collected from the job-specific output directory.
The manager normalizes them into safe relative IDs before the API exposes them.

## Task behavior

- `predict`: requires model and data source; accepts a file, a non-recursive media
  directory, or a list; processes deterministic chunks with cancellation checks.
- `train`: requires model and dataset YAML; applies validated training parameters
  and collects the full job output tree.
- `val`: requires model and dataset YAML; returns JSON-serializable metrics when
  the underlying result exposes them and collects validation outputs.
- `export`: requires a model and a closed-allowlist format; copies the model into
  the job directory before calling `YOLO.export()` so concurrent exports do not
  modify the source directory.
- `diagnose`: writes JSON and text environment reports; it does not invoke a YOLO
  training/inference operation.

## Security boundary

Handlers reject enabled shell access, disabled path whitelisting, missing trusted
roots, and unsafe model/data paths. Path failures raise
`PathWhitelistViolationError`; ordinary parameter errors return `(False, message)`.
The dispatcher maps these categories to structured errors.

For production API submissions, `JobsManager` is the authoritative admission
boundary. It discards client path lists/patterns, applies server model/data/output
roots, checks network input hosts, clears client artifacts, and forces shell off
and path whitelisting on before queueing. Direct handler calls do not perform all
of those manager-level checks.

`_check_cancelled()` is a cooperative checkpoint injected by the dispatcher.
Forced cancellation and timeout are provided by `ManagedWorker`/`JobsManager`,
not by the handler method itself.

## Registry API

- `@TaskHandlerRegistry.register(task_type)` registers a handler at import time.
- `TaskHandlerRegistry.get(task_type)` returns the registered class.
- `TaskHandlerRegistry.list_registered()` returns sorted names.
- `TaskHandlerRegistry.clear()` exists for isolated tests only.

Registration is not thread-safe and must finish before concurrent dispatch.
Duplicate names raise `ValueError`; non-`BaseTaskHandler` classes raise `TypeError`.

## Verification

Run the focused handler and dispatcher tests from the repository root:

```bash
python -m pytest \
  tests/f1/test_handlers_framework.py \
  tests/f1/test_handler_inventory.py \
  tests/f1/test_phase1_handlers.py \
  tests/f1/test_predict_diagnose.py \
  tests/f1/test_val_batch_runtime.py \
  tests/f1/test_dispatcher.py -v
```

For the current compatibility and production selections, see
[testing](../../docs/studio/testing.md).

See [USAGE.md](USAGE.md) for extension guidance and the Agent/F1 boundary.

---

**Document version**: 1.2.0

**Last synchronized**: 2026-09-15
