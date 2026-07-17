# Standard BenchmarkSuite Design

## Scope

The first BenchmarkSuite phase standardizes fast, read-only model comparison. It builds detection models from local YAML or checkpoint files, measures construction metadata and PyTorch eager inference, captures routing health, and writes reproducible reports. It does not start training, validation datasets, export jobs, or network downloads.

## Architecture

The canonical suite catalog is a declarative YAML file under `benchmarks/`. A suite contains model cases plus shared settings for device, dtype, input size, batch size, warmup, timed iterations, seed, FLOPs, and routing analysis. Command-line overrides produce a resolved run specification without mutating the catalog.

`BenchmarkSuiteRunner` executes each case independently. It records environment and Git metadata once, builds a deterministic synthetic input, loads the model through the repository-local `DetectionModel` or direct checkpoint path, and collects parameters, GFLOPs, P50/P95/P99 latency, throughput, module-family counts, and routing-collapse indicators. Case failures are serialized and do not stop remaining cases.

Results use one canonical JSON document. CSV and Markdown are deterministic projections of that document. JSON writes use a temporary sibling followed by `Path.replace()` so interruption cannot leave a partially written result. Each case has a fingerprint derived from the resolved suite settings and model source digest. Resume mode reuses successful matching cases, while failed or changed cases run again.

## Data Flow

1. Load and validate the YAML catalog.
2. Resolve repository-relative model paths and CLI overrides.
3. Capture environment, Git commit, dirty state, Python, PyTorch, and device information.
4. For every selected case, reuse a matching successful result or execute it.
5. Persist `results.json`, `results.csv`, and `report.md` after every case.
6. Return a non-zero CLI status only when one or more selected cases fail.

## Safety And Boundaries

- Only local `.yaml`, `.yml`, `.pt`, and `.pth` model sources are accepted.
- Suite task type is limited to `inference` in phase one.
- Synthetic input avoids implicit dataset download and preprocessing variance.
- Checkpoints are trusted local PyTorch artifacts and follow the repository's existing model-loading assumptions.
- Training and export will be later executors using the same result protocol rather than flags hidden in this runner.

## Verification

Unit tests cover schema validation, percentile math, atomic reports, failed-case continuation, and resume fingerprints. An integration smoke test runs the built-in no-MoE/MoE/MoA/MoT suite at a small CPU input size and verifies that all three reports contain comparable metrics.
