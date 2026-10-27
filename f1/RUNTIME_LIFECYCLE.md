# Runtime documentation moved

See [Studio Runtime](../docs/studio/runtime.md) for production queues, cleanup,
persistence and checkpoint recovery. [Service](../docs/studio/service.md) and
[launcher](../docs/studio/frontend.md) describe startup/shutdown integration.

The [previous lifecycle guide](https://github.com/Tencent/YOLO-Master/blob/b10cdf11c9f7aa6cb949595391572e0eed1bf0f1/f1/RUNTIME_LIFECYCLE.md)
covers the retained F1 backend. Current verification methods are in
[testing](../docs/studio/testing.md).

## Verification record: 2026-09-15

This record describes the historical F1 revision named below.

- The official baseline `acce839c7e895d6b179de7f7093fa879e237cc7b` is an
  ancestor of the reviewed commit `0e7a8f83b53c97abc8eb3caedc532f5779fbb086`.
- Windows/Python 3.12 in the repository `.venv`: `412 passed, 11 skipped,
  3 warnings`; combined `f1` + `core` coverage was 78%. The Windows skips require
  symlink privileges. One warning was host-specific pytest cache permission noise.
- Ubuntu/Python 3.10 source-branch CI at the same HEAD: `422 passed, 1 skipped,
  3 warnings`, 78% coverage, smoke 5/5, Ruff lint and format passed.
- CI run: <https://github.com/9-71/YOLO-Master/actions/runs/34793607227>.
  This is source-branch push evidence, not a Tencent required PR check.
- The CI workflow exercises Python F1/API tests and smoke. It does not run Node
  lint, TypeScript checks, or a Vite production build.

Remaining limits: concurrency is per service process, pending/history records are
not memory-capped, and no distributed coordination is provided. Process checks
establish absence of live computation; POSIX zombie reaping ultimately depends on
the guardian/OS init. A platform-level refusal to terminate remains visible as
`WORKER_STOP_FAILED` with the slot retained. Only Windows and Linux have been
exercised; other POSIX systems do not have the Linux subreaper guarantee. Run
task execution through `JobsManager` to obtain these lifecycle guarantees.

---

**Document version**: 1.2.0

**Last synchronized**: 2026-09-15
