# Legacy F1 direct-use handler extension guide

Use these examples to extend the legacy F1 direct-call implementation.
For production handler registration, see [Studio Core](../../docs/studio/core.md).
For job submission, use the [Studio user guide](../../docs/studio/user_guide.md).

## 1. Understand the ownership boundary

```text
Legacy direct caller -> f1 dispatcher -> f1 registry -> f1 handler
```

Handlers own task-specific validation and engine adaptation. They do not own:

- API admission or server-root normalization
- CPU/GPU queueing and concurrency slots
- process creation, timeout, or forced termination
- public cancellation publication
- state persistence, cursor logs, or artifact download authorization

Those responsibilities stay in `JobsManager`, `ManagedWorker`, and the API. A
handler should never create another manager or mutate the shared state file.

## 2. Implement the base contract

```python
from typing import Any

from f1.handlers.base import BaseTaskHandler, PathWhitelistViolationError
from f1.handlers.registry import TaskHandlerRegistry


@TaskHandlerRegistry.register("example")
class ExampleHandler(BaseTaskHandler):
    def validate_params(
        self,
        params: dict[str, Any],
        security_constraints: dict[str, Any],
    ) -> tuple[bool, str | None]:
        if security_constraints.get("allow_shell", False):
            return False, "Shell execution is not allowed"
        if not security_constraints.get("path_whitelisted", False):
            return False, "Path whitelisting must be enabled"

        allowed_paths = security_constraints.get("allowed_paths", [])
        allowed_patterns = security_constraints.get("allowed_path_patterns", [])
        if not allowed_paths and not allowed_patterns:
            return False, "Trusted path whitelist cannot be empty"

        model_path = params.get("model_path")
        if not model_path:
            return False, "Required parameter 'model_path' is missing"
        if not self._is_path_safe(model_path, allowed_paths, allowed_patterns):
            raise PathWhitelistViolationError(f"model_path '{model_path}' is not within the trusted whitelist")
        return True, None

    def execute(
        self,
        job_id: str,
        params: dict[str, Any],
        output_dir: str,
    ) -> dict[str, Any]:
        self._check_cancelled()
        # Delegate to a Python API here; do not build a shell command.
        self._check_cancelled()
        return {
            "success": True,
            "artifacts": [],
            "metadata": {},
            "error": None,
        }
```

This example demonstrates the method shape. To expose `example` as a public API
task, update the canonical schema and public surfaces as described in section 6.

## 3. Validation and error classification

Use these categories consistently:

- Return `(False, message)` for missing fields, invalid types, unsupported values,
  or out-of-range parameters. The dispatcher maps this to
  `PARAM_VALIDATION_FAILED`.
- Raise `PathWhitelistViolationError` for path containment failures. The
  dispatcher maps it to `SEC_ERR_001`.
- Return a structured `success=False` result for controlled engine failures.
- Let `CooperativeCancellationError` from `_check_cancelled()` propagate. Do not
  catch it in the handler's generic engine exception block.

The production API overwrites client security fields before dispatch. Handler
validation is still fail-closed because handlers are also used directly by tests
and compatibility callers.

`_is_path_safe()` resolves relative paths, rejects unsafe symlink components and
traversal, requires containment in a trusted root, and supports full-match regex
patterns for non-API compatibility callers. The production FastAPI path clears
client regex patterns and uses server-owned roots.

## 4. Execution rules

1. Call `_check_cancelled()` before expensive engine work.
2. For chunked work, call it between chunks. It cannot interrupt a single blocking
   Ultralytics call; manager-owned process termination provides the hard stop.
3. Use the Python API (`YOLO.train`, `YOLO.val`, `YOLO.predict`, `YOLO.export`) and
   do not invoke a shell.
4. Write only below `Path(output_dir) / job_id`.
5. Return absolute paths for files that actually exist. Production handlers use a
   sorted recursive scan of the job directory.
6. Keep metadata JSON-serializable. Convert tensors, NumPy values, and engine
   result objects to plain scalars, lists, and dictionaries.
7. Do not publish public lifecycle states directly. The dispatcher produces its
   worker result; `JobsManager` arbitrates the final public snapshot.
8. Do not assume that printing to stdout enters the structured job log. Use the
   controlled job logging path when working at dispatcher/manager level.

## 5. Existing handler reference

### Predict

`f1/handlers/predict.py` accepts one source path, a non-recursive directory, or a
list. Directory expansion is sorted and filtered to supported media extensions.
Input chunks default to `batch_size=8`, with cancellation checks between chunks.
Network sources require API-level host authorization.

### Train and val

`train.py` and `val.py` require a dataset `.yaml`/`.yml` and validate model/data
paths. The API injects `epochs=1` and `imgsz=640` defaults for train, and
`imgsz=640` for val when omitted. Both collect the complete job output tree.

### Export

`export.py` validates format against `SUPPORTED_EXPORT_FORMATS`, copies the source
model into the job directory, calls `YOLO.export()`, and scans that directory for
artifacts. The copy prevents concurrent jobs from writing next to the original
checkpoint.

### Diagnose

`diagnose.py` collects environment information and writes
`system_diagnostics.json` and `system_diagnostics.txt`. It still follows the same
result and artifact contract, but does not call a YOLO compute method.

## 6. Expose a new public task

A new public task is a cross-layer contract change. Update all applicable areas:

1. Add the task to `core/schema.py::TaskType`.
2. Add the concrete handler module and registration decorator.
3. Import it from `f1/handlers/__init__.py` so registration occurs in workers.
4. Update `core/task_catalog.py` and any Agent catalog only if the task is also an
   Agent-facing capability.
5. Update REST/Gradio/zero-build/React form validation and presets as needed.
6. Add handler inventory, dispatcher, API contract, lifecycle, and UI tests.
7. Document the new task and the public surfaces that accept it.

The dispatcher normally needs no task-specific branch because it uses the
registry, but the schema and clients are closed over the current five task types.

## 7. Registry behavior

```python
from f1.handlers import TaskHandlerRegistry

handler_class = TaskHandlerRegistry.get("predict")
handler = handler_class()
available = TaskHandlerRegistry.list_registered()
```

- Registration occurs at module import time.
- Duplicate task names raise `ValueError`.
- Registered classes must inherit `BaseTaskHandler`.
- The registry is not thread-safe; finish registration before dispatch begins.
- `TaskHandlerRegistry.clear()` is for isolated tests only.

## 8. Verification

From the repository root, run focused handler checks:

```bash
python -m pytest \
  tests/f1/test_handlers_framework.py \
  tests/f1/test_handler_inventory.py \
  tests/f1/test_phase1_handlers.py \
  tests/f1/test_predict_diagnose.py \
  tests/f1/test_val_batch_runtime.py \
  tests/f1/test_dispatcher.py -v
```

For changes affecting admission, cancellation, timeout, or artifacts, also run:

```bash
python -m pytest \
  tests/f1/test_worker_lifecycle.py \
  tests/f1/test_jobs_queue_capacity.py \
  tests/api/test_api_v1.py \
  tests/api/test_security_boundaries.py \
  tests/api/test_live_logs_ablation.py -v
```

The CI-equivalent F1 scope is:

```bash
python -m pytest tests/f1/ tests/api/ --cov=f1 --cov=core
python -m smoke.test_f1_smoke
ruff check f1/ tests/f1/ smoke/ core/ api/ main_engine.py tests/api/ app.py
ruff format --check f1/ tests/f1/ smoke/ core/ api/ main_engine.py tests/api/ app.py
```

## 9. Agent/F1 boundary

F1 handlers are not Agent Skill handlers. Legacy direct-use F1 routing ends in
`f1/handlers/`; Agent CLI dispatch and multimodal skill behavior live under
`agent/`. Share canonical task names through `core/` only where the code already
defines an explicit shared contract. F1 lifecycle checks use the F1 runtime and
its test suite.

## Current limitations

- Synchronous handler methods only
- Cooperative checkpoints only between blocking engine calls
- Static import-time registry; no hot reload or handler version negotiation
- Resource scheduling is manager-level and static, not handler-selected
- No per-handler retry policy or isolated security sandbox

---

**Document version**: 1.2.0

**Last synchronized**: 2026-09-15

For current production and compatibility verification selections, see
[testing and reproduction](../../docs/studio/testing.md).
