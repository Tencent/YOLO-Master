# Legacy F1 compatibility

For Studio setup and daily use, start at the [quick start](../docs/studio/README.md)
and [user guide](../docs/studio/user_guide.md).

The `f1` backend and direct-call adapters support legacy integrations. Production
HTTP jobs use the `studio` runtime, dispatcher and handlers; `f1/ui` provides lazy
imports for `studio/ui`. See [architecture](../docs/studio/architecture.md),
[service configuration](../docs/studio/service.md) and [legacy handler extensions](handlers/USAGE.md).

The [earlier F1 platform guide](https://github.com/Tencent/YOLO-Master/blob/b10cdf11c9f7aa6cb949595391572e0eed1bf0f1/f1/README.md)
remains available for historical reference.

## Historical verification record

This original record describes the revision named below. For current test
selections, see [testing](../docs/studio/testing.md).

The following results were recorded on commit
`0e7a8f83b53c97abc8eb3caedc532f5779fbb086`, reviewed against the official
baseline `acce839c7e895d6b179de7f7093fa879e237cc7b` (2026-08-21 23:59:59 UTC+8).

Windows verification used the repository `.venv` with Python 3.12 and a fresh
repository-local pytest temp directory:

```text
pytest tests/f1/ tests/api/ --cov=f1 --cov=core
412 passed, 11 skipped, 3 warnings
TOTAL: 2369 statements, 518 missed, 78% coverage
```

The 11 Windows skips require symlink privileges. Two warnings are ONNX exporter
deprecations; the third was a host-specific pytest cache permission warning.

The Ubuntu/Python 3.10 source-branch CI run at that commit completed successfully:

```text
422 passed, 1 skipped, 3 warnings
TOTAL: 2369 statements, 516 missed, 78% coverage
F1 smoke: 5/5 passed
Ruff lint and format: passed
```

CI evidence:
<https://github.com/9-71/YOLO-Master/actions/runs/34793607227>

This is source-branch push CI evidence, not a Tencent repository required PR
check. The F1 workflow covers Python Ruff, smoke, unit tests, and coverage. It
does not currently run Node lint, TypeScript checking, or a Vite production build.

Local checks at the same commit additionally passed `npm run lint` and TypeScript
`--noEmit` checks for the React app and Node configuration. No claim is made here
that a production React bundle is exercised by CI.
