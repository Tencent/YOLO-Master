# Standard BenchmarkSuite

The standard BenchmarkSuite provides one reproducible interface for fast YOLO-Master architecture comparisons. Phase one covers local model construction, PyTorch eager inference, parameters, GFLOPs, latency percentiles, throughput, routed-module counts, and routing health. It deliberately does not train, validate datasets, export models, or download assets.

## Built-in suites

List available suites:

```bash
python benchmarks/run.py --list
```

The initial catalog at `benchmarks/suites.yaml` includes:

- `mixture_smoke`: v0.10 no-MoE, MoE, MoA, and MoT.
- `moa_vs_mot`: focused MoA/MoT comparison.
- `mot_scene_router`: legacy MoT versus the experimental scene-aware router.

Run the CPU-safe smoke defaults:

```bash
python benchmarks/run.py --suite mixture_smoke
```

Override hardware and timing settings without editing the catalog:

```bash
python benchmarks/run.py \
  --suite moa_vs_mot \
  --device mps \
  --imgsz 320 \
  --warmup 10 \
  --iterations 50 \
  --output runs/benchmarks/moa_vs_mot_mps
```

Use repeated `--case` options to select cases. `--no-flops` and `--no-routing` disable their respective collectors. `--force` ignores matching successful results and reruns every selected case.

## Outputs

Every run writes three synchronized views after each case:

- `results.json`: canonical structured result, including resolved settings, environment, Git commit, case fingerprints, metrics, routing diagnostics, failures, and timestamps.
- `results.csv`: flat comparison table suitable for spreadsheets.
- `report.md`: human-readable comparison table plus environment metadata.

Latency is accelerator-synchronized and reports mean, P50, P95, P99, minimum, maximum, and P50-derived item throughput. Compare latency only when device, dtype, batch size, image size, warmup, iterations, and software environment match.

## Resume and failure behavior

A case fingerprint includes its resolved settings, runtime environment, Git state, and the complete local model file contents. A successful case is reused only when the fingerprint still matches. Failed cases always run again. A model build or benchmark failure is recorded and later cases continue; the CLI exits with status 1 when any selected case failed.

JSON writes are atomic, so an interrupted run keeps the last complete case result. CSV and Markdown are generated from the same canonical in-memory report.

## Catalog schema

Catalog defaults merge in this order: global defaults, suite settings, command-line overrides, then case-specific settings. Case-specific values intentionally win so suites can preserve exceptional requirements. Duplicate YAML mapping keys are rejected instead of being silently overwritten.

Only the `inference` task and local `.yaml`, `.yml`, `.pt`, or `.pth` models are supported in phase one. Training accuracy, export parity, memory, and backend-specific runners should be added as explicit future executors under the same result schema rather than hidden flags in the inference runner.
