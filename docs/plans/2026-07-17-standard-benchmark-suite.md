# Standard BenchmarkSuite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reproducible, resumable inference benchmark suite for local YOLO-Master model variants.

**Architecture:** Load declarative suites from one YAML catalog, resolve each model into a typed case, and execute cases independently through a common runner. Store one canonical JSON result atomically and generate CSV/Markdown projections after every case so partial runs remain useful.

**Tech Stack:** Python 3.11, PyTorch, PyYAML, Ultralytics `DetectionModel`, RoutingInterpreter, pytest

---

### Task 1: Define suite and result contracts

**Files:**
- Create: `benchmarks/__init__.py`
- Create: `tests/test_benchmark_suite.py`

**Step 1: Test YAML schema loading**

Assert defaults merge into model cases, repository-relative paths resolve correctly, duplicate IDs fail, unsupported task types fail, and invalid timing settings produce actionable errors.

**Step 2: Test latency summaries**

Assert mean, P50, P95, P99, minimum, maximum, and throughput are deterministic for a known sample vector.

**Step 3: Test result serialization**

Assert JSON records are machine-readable and CSV/Markdown reports use a stable model order and comparable metric columns.

**Step 4: Test continuation and resume**

Use an injected case executor to assert one failed case does not block later cases, successful fingerprints are reused, and changed settings rerun the case.

**Step 5: Run tests to verify they fail**

Run: `python -m pytest -q tests/test_benchmark_suite.py --tb=long`

Expected: FAIL because `benchmarks.suite` does not exist.

### Task 2: Implement the suite core

**Files:**
- Create: `benchmarks/suite.py`

**Step 1: Add immutable typed specifications**

Define suite settings, model cases, resolved suite, case results, and run report records with JSON-compatible conversion.

**Step 2: Add strict YAML loading**

Validate schema version, task, unique IDs, model suffixes, positive dimensions, supported dtype/device strings, and non-empty cases.

**Step 3: Add environment and fingerprint capture**

Record platform/Python/PyTorch/device/Git information and hash the resolved case plus local model content.

**Step 4: Add model execution**

Build YAML models with `DetectionModel`, load trusted local checkpoints directly, create deterministic synthetic input, synchronize accelerators, and measure parameter, FLOPs, latency, throughput, module-family, and routing metrics.

**Step 5: Add resilient run orchestration**

Catch case failures, persist after each case, resume only matching successful fingerprints, and clear accelerator caches between cases.

**Step 6: Add deterministic reports**

Atomically write `results.json`, then render `results.csv` and `report.md` from the same in-memory report.

### Task 3: Add built-in suites and CLI

**Files:**
- Create: `benchmarks/suites.yaml`
- Create: `benchmarks/run.py`

**Step 1: Define `mixture_smoke`**

Compare v0.10 no-MoE, MoE, MoA, and MoT model YAMLs with CPU-safe small defaults.

**Step 2: Define focused suites**

Add `moa_vs_mot` and `mot_scene_router` as aliases over the same executor with different case lists.

**Step 3: Add CLI overrides**

Support suite listing, case filtering, device, dtype, batch, image size, warmup, iterations, seed, output directory, resume, force, FLOPs toggle, and routing toggle.

**Step 4: Add exit semantics**

Print report paths and a compact case table. Exit 1 if any selected case failed and 0 otherwise.

### Task 4: Document and verify

**Files:**
- Create: `docs/governance/benchmark-suite.md`

**Step 1: Document the catalog and outputs**

Explain fields, metrics, resume fingerprints, failure behavior, hardware comparability, and the no-training boundary.

**Step 2: Run focused tests**

Run: `python -m pytest -q tests/test_benchmark_suite.py --tb=long`

Expected: PASS.

**Step 3: Run a real smoke suite**

Run: `python benchmarks/run.py --suite mixture_smoke --device cpu --imgsz 64 --warmup 0 --iterations 1 --output /tmp/yolo-benchmark-smoke`

Expected: four successful cases and JSON/CSV/Markdown outputs.

**Step 4: Run related regressions**

Run: `python -m pytest -q tests/test_routing_interpreter.py tests/test_master_model_configs.py --tb=long`

Expected: PASS.

**Step 5: Run static checks**

Run: `ruff check benchmarks/suite.py benchmarks/run.py tests/test_benchmark_suite.py`

Run: `python -m compileall -q benchmarks/suite.py benchmarks/run.py`

Expected: PASS with no output.
