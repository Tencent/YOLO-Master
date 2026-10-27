# Testing and reproduction

[Studio documentation](README.md)

Run commands from the source root unless a section says `web/`. Use Python
3.10+ and a virtual environment. Examples below use PowerShell; in POSIX shells
use `export NAME=value` instead of `$env:NAME="value"`. Record the tested
revision/patch and whether the checkout and environment were fresh.

## Offline Core, Runtime and Service lane

```powershell
python -m pip install -e ".[studio]" pytest httpx ruff codespell
$env:YOLO_AUTOINSTALL="false"
$env:YOLO_OFFLINE="true"
python -m pytest tests/studio --ignore=tests/studio/ui -q
python -m pytest --doctest-modules core studio api main_engine.py -q
ruff check core studio api main_engine.py tests/studio
ruff format --check core studio api main_engine.py tests/studio
codespell core studio api main_engine.py tests/studio
```

[Studio Core CI](https://github.com/Tencent/YOLO-Master/blob/main/.github/workflows/studio-core.yml)
runs the Python selection on Ubuntu and Windows with Python 3.10. It includes
Core, Runtime and Service checks and excludes Gradio UI tests. Studio tests need
explicit selection; ordinary upstream collection excludes them.
`tests/studio/conftest.py` prohibits implicit model/data downloads and removes
real-model tests unless explicitly enabled below.

## Frontend lane

```powershell
python -m pip install -e ".[studio]" gradio pandas pytest httpx ruff
$env:YOLO_AUTOINSTALL="false"
$env:YOLO_OFFLINE="true"
$env:NO_PROXY="127.0.0.1,localhost,::1"
python -m pytest tests/studio tests/f1/ui -q
```

In `web/`, with Node.js 24:

```bash
npm ci
npm test
npm run lint
npm run build
```

[Studio Frontend](https://github.com/Tencent/YOLO-Master/blob/main/.github/workflows/studio-frontend.yml) runs the Python
clients/launcher on Ubuntu and Windows and the browser checks on Ubuntu. Build
includes TypeScript checking. Keep `package-lock.json`; do not add generated
`dist/`, `node_modules/`, logs or model files to a contribution.

## Compatibility lane

[Studio Compatibility CI](https://github.com/Tencent/YOLO-Master/blob/main/.github/workflows/f1_ci.yml)
covers the retained F1 package, API compatibility, smoke and coverage. It fetches
a model for legacy export/smoke checks and has its own triggers. Use that workflow
for the compatibility test selection; browser/Node checks belong to the Frontend lane.

## Explicit local-model integration

Use an existing trusted detection `.pt` file; no download is part of this
procedure. Keep the source checkout and local model/data under trusted control.
Install the frontend dependencies above for the combined launcher case.

```powershell
$env:STUDIO_TEST_MODEL="D:\path\to\existing-local-model.pt"
$env:YOLO_AUTOINSTALL="false"
$env:YOLO_OFFLINE="true"
$env:NO_PROXY="127.0.0.1,localhost,::1"
python -m pytest tests/studio/test_core_integration.py --slow --studio-integration -q
python -m pytest tests/studio/test_mainline_integration.py --slow --studio-integration -q
```

Both flags are required. Core integration uses tiny generated data, CPU
prediction/training and TorchScript export. Production integration exercises
loopback HTTP, five handlers, the legacy HTTP client, cancellation, service and
launcher shutdown, checkpoint load/resume and restart reconciliation. It uses
CPU with effective dataloader workers zero and AMP disabled. Record a separate
device/configuration when checking GPU, AMP-scaler or dataloader-child recovery.

## Documentation checks

Regenerate [job_schema.json](job_schema.json) using the [API instructions](api.md#regenerate-the-domain-schema)
and compare again in a fresh process. Check every changed Markdown link and
anchor, parse/compile the examples, and run the diagnostic against a local API.
Run `codespell` on the changed Markdown/JSON files and `git diff --check`.

The repository's `mkdocs.yml` serves `docs/en/`, not `docs/studio/`. For the
full-site toolchain in a fresh environment, install the documentation entries
from the `dev` extra together with MkDocs and the configured Material theme:

```bash
python -m pip install mkdocs mkdocs-material "zensical>=0.0.15" "mkdocs-ultralytics-plugin>=0.2.5" "minijinja>=2.0.0"
python -m mkdocs build --strict
zensical build --strict -f mkdocs.yml
```

MkDocs is the repository's `AGENTS.md` check. Zensical is also used by
`docs/build_docs.py`, which prepares generated/external pages and macros first;
the direct command above builds only existing checkout pages. Record versions
and compare any failure with the unchanged base using the same command/environment.
Record baseline and candidate failures in the PR report.

To build the Studio documentation, run the following Python block from the
repository root. It copies Studio pages to a temporary tree and runs MkDocs
strict with default warning handling. Record the Studio documentation build
and full-site build separately.

```python
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

root = Path.cwd()
work = Path(tempfile.mkdtemp(prefix="studio-docs-"))
for item in (root / "docs/studio").rglob("*"):
    if item.is_file() and item.suffix in {".md", ".json"}:
        target = work / "docs" / item.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
config = work / "mkdocs.yml"
config.write_text(
    'site_name: Studio documentation\ndocs_dir: docs\nsite_dir: site\n'
    'markdown_extensions: [fenced_code, tables]\n',
    encoding="utf-8",
)
subprocess.run(
    [sys.executable, "-m", "mkdocs", "build", "--strict", "--config-file", str(config)],
    check=True,
)
print(work)
```

## Manual clean-checkout checklist

Use a fresh checkout/environment and follow the [quick start](README.md) and
[user guide](user_guide.md). Record the revision/patch, OS, dependency
versions and effective device/AMP/loader settings. Keep commands, outcomes,
skips and artifact hashes in the PR report; mark unperformed checks explicitly.

- [ ] Install and start using only the documented prerequisites. Check `/health`,
  `/docs`, `/openapi.json` and the built-in console from the source checkout.
- [ ] Complete a diagnostic in each client; follow logs, download both reports,
  and inspect their contents. Check a duplicate API ID returns 409 and an unknown
  ID returns 404.
- [ ] With local assets inside configured roots, submit model tasks, cancel
  running work and wait for final status. Check an outside path is rejected.
  Download nested/non-image artifacts and compare a large file's hash.
- [ ] Switch jobs/APIs while refreshing; check old responses do not replace the
  new selection. Disconnect and reconnect the API. Check Inference Studio,
  browser-close behavior and launcher-owned versus external API shutdown as
  described in [clients and launcher](frontend.md).
- [ ] When changing recovery behavior, check confirmed checkpoint download,
  `failed/SERVICE_SHUTDOWN`, cleanup, trusted load and actual next-epoch resume.
  Restart and check `failed/SERVICE_RESTARTED` without automatic relaunch.

## Test scope

Report automated checks, manual UI review and platform/model tests separately,
including skipped and untested cases. Put run-specific results in the PR report.

Use a matching device and configuration for GPU, AMP and dataloader-child tests.
Cooperative checkpoint shutdown supports single-process training. Process tests
check for remaining owned processes after exit; failure-injection cases restore
cleanup before checking shutdown. Persistent cleanup failures retain the worker
and slot for retry. See [Runtime](runtime.md) and [launcher](frontend.md) for
shutdown budgets and cleanup behavior.
