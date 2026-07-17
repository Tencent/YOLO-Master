# Export Capability Matrix Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn the declarative export capability matrix into the runtime source of truth for preflight decisions, CI validation, and generated user documentation.

**Architecture:** Store the canonical packaged YAML under `ultralytics/cfg`, load it through a typed utility, and intersect matrix policy with each routed module's runtime declaration. Generate the governance Markdown table from the canonical YAML so documentation cannot drift independently.

**Tech Stack:** Python 3.11, PyYAML, PyTorch, pytest, GitHub Actions

---

### Task 1: Define matrix and preflight contracts

**Files:**
- Create: `tests/test_export_capability_matrix.py`

**Step 1: Test schema validation**

Assert every format and module entry has required fields and invalid matrices raise actionable errors.

**Step 2: Test aliases and module classification**

Assert TensorRT aliases normalize to `engine`, routed classes classify as MoE/MoA/MoT/MoLoRA, and unknown formats are refused.

**Step 3: Test matrix-backed preflight**

Use a temporary matrix to block one module/backend and assert `export_preflight` reports the matrix limitation before export.

**Step 4: Test merge policy**

Assert a matrix entry with `requires_merge: true` refuses an unmerged MoLoRA layer and accepts a merged layer.

**Step 5: Test generated documentation**

Render Markdown from the canonical matrix and assert the checked-in governance document matches exactly.

### Task 2: Add the canonical packaged matrix

**Files:**
- Create: `ultralytics/cfg/export-capability-matrix.yaml`
- Create: `ultralytics/utils/export_capabilities.py`

**Step 1: Define all exporter formats**

Declare eager/dynamic or dense-fallback policy for every format accepted by `export_formats()`.

**Step 2: Define module policies**

Declare `supported`, `dense_fallback`, `requires_merge`, and `known_error` for MoE, MoA, MoT, and MoLoRA, with optional per-format overrides.

**Step 3: Implement typed loading and validation**

Reject missing fields, unknown override formats, invalid booleans, and missing routed module families.

**Step 4: Add query helpers**

Expose format normalization, routed module classification, and effective capability resolution.

### Task 3: Integrate runtime preflight

**Files:**
- Modify: `ultralytics/utils/export_preflight.py`

**Step 1: Load the canonical matrix**

Accept an optional matrix/path for testing and default to the packaged file.

**Step 2: Intersect runtime and matrix capabilities**

Refuse unsupported formats or modules, enforce `requires_merge`, use dense fallback only when both declarations allow it, and report matrix limitations in every decision.

**Step 3: Include matrix metadata in reports**

Add schema version and source path to the JSON-compatible preflight report.

### Task 4: Generate governance documentation

**Files:**
- Create: `scripts/generate_export_capability_docs.py`
- Create: `docs/governance/export-capability-matrix.md`
- Remove: `docs/governance/export-capability-matrix.yaml`

**Step 1: Render deterministic Markdown tables**

Generate format defaults and module/backend effective policies from the canonical matrix.

**Step 2: Add `--check` mode**

Exit non-zero if the checked-in Markdown differs from generated output.

### Task 5: Integrate CI and verify

**Files:**
- Modify: `.github/workflows/ci.yml`

**Step 1: Add schema and generated-doc checks**

Run `python scripts/generate_export_capability_docs.py --check` and focused matrix/preflight tests in the PR gate.

**Step 2: Run related regressions**

Run export preflight, model registry, component export, config drift, ruff, compileall, and diff checks.
