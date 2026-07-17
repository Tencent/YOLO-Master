# Configuration Drift Detector Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a fast CI guard that detects drift between public YAML defaults, configuration dataclasses, runtime CLI mappings, and YOLO-Master model YAML declarations.

**Architecture:** Implement a static detector in `tools/config_drift_detector.py` that parses YAML and Python ASTs without constructing models. Keep runtime model smoke tests as a separate semantic layer, while this detector provides fast repository-wide structural checks on every pull request.

**Tech Stack:** Python 3.11, `ast`, PyYAML, pytest, GitHub Actions

---

### Task 1: Define the drift contracts with failing tests

**Files:**
- Create: `tests/test_config_drift_detector.py`

**Step 1: Write tests for duplicate YAML keys**

Create a temporary YAML file with the same top-level key twice and assert that the detector returns `YAML_DUPLICATE_KEY`.

**Step 2: Write tests for dataclass-to-CLI mapping drift**

Create minimal `LoRAConfig` and `MoLoRAConfig` sources and assert that an unmapped dataclass field and a mapped CLI key missing from `default.yaml` are both reported.

**Step 3: Write tests for runtime resolver drift**

Create mismatched `MIXTURE_DEFAULTS`, `CLI_FIELDS`, and `default.yaml` values and assert that `MIXTURE_DEFAULT_MISMATCH` is reported.

**Step 4: Write tests for CLI type registry drift**

Declare a typed CLI key that does not exist in `default.yaml` and assert that `CLI_TYPE_KEY_MISSING` is reported.

**Step 5: Write tests for model YAML drift**

Create a model YAML with an unknown module and another with too many constructor arguments. Assert that `MODEL_MODULE_UNKNOWN` and `MODEL_ARGS_INCOMPATIBLE` are reported.

**Step 6: Run the tests to verify they fail**

Run: `python -m pytest -q tests/test_config_drift_detector.py --tb=long`

Expected: collection fails because `tools.config_drift_detector` does not exist yet.

### Task 2: Implement the static detector

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/config_drift_detector.py`

**Step 1: Add structured issue reporting**

Implement an immutable `DriftIssue` record with a stable code, path, optional line, message, text formatting, and JSON serialization.

**Step 2: Add duplicate-safe YAML loading**

Use a custom `yaml.SafeLoader` mapping constructor that rejects duplicate keys and preserves the duplicate key line for diagnostics.

**Step 3: Add dataclass and mapping AST checks**

Parse class-level annotated fields and literal `from_args` mapping dictionaries. Verify local dataclass fields are mapped, mappings target real fields, and every mapped CLI key exists in `default.yaml`.

**Step 4: Add runtime resolver checks**

Parse literal `MIXTURE_DEFAULTS` and `CLI_FIELDS` assignments. Verify the field sets match, every CLI key is public, and resolver defaults equal public YAML defaults.

**Step 5: Add CLI type registry checks**

Parse `CFG_FLOAT_KEYS`, `CFG_FRACTION_KEYS`, `CFG_INT_KEYS`, and `CFG_BOOL_KEYS`. Verify registered keys exist, do not appear in multiple registries, and their default values have compatible types.

**Step 6: Add Master model YAML checks**

Index Python class constructor signatures statically, extract `parse_model` base/repeat/channel-append module groups, and validate all `ultralytics/cfg/models/master/**/*.yaml` layer records, module names, and effective positional argument counts.

**Step 7: Add the CLI entry point**

Support `python tools/config_drift_detector.py`, optional `--root`, and optional `--json`. Return exit code 1 when issues are found.

**Step 8: Run focused tests**

Run: `python -m pytest -q tests/test_config_drift_detector.py --tb=long`

Expected: PASS.

### Task 3: Repair detected repository drift and add the CI gate

**Files:**
- Modify: `ultralytics/nn/modules/moe/config.py`
- Modify: `.github/workflows/ci.yml`
- Create: `docs/governance/config-drift-detection.md`

**Step 1: Run the detector on the repository**

Run: `python tools/config_drift_detector.py`

Expected: FAIL with the existing `moe_router_z_loss` resolver/default mismatch.

**Step 2: Align the runtime resolver default**

Change `MIXTURE_DEFAULTS["moe"]["router_z_loss_coeff"]` from `1.0` to the public `default.yaml` value `0.1`.

**Step 3: Add the pull-request gate**

Run `python tools/config_drift_detector.py` in the existing `mixture-p0-regression` configuration step before pytest.

**Step 4: Document the contract**

Document the command, checks, issue format, and boundary between static drift detection and runtime smoke tests.

**Step 5: Re-run the detector**

Run: `python tools/config_drift_detector.py`

Expected: `Configuration drift check: PASS`.

### Task 4: Verify the integration

**Files:**
- Test: `tests/test_config_drift_detector.py`
- Test: `tests/test_default_config_integrity.py`
- Test: `tests/test_mixture_config_resolution.py`
- Test: `tests/test_master_model_configs.py`

**Step 1: Run focused drift tests**

Run: `python -m pytest -q tests/test_config_drift_detector.py --tb=long`

Expected: PASS.

**Step 2: Run related configuration regressions**

Run: `python -m pytest -q tests/test_default_config_integrity.py tests/test_mixture_config_resolution.py tests/test_master_model_configs.py --tb=long`

Expected: PASS.

**Step 3: Compile the new tool**

Run: `python -m compileall -q tools/config_drift_detector.py`

Expected: PASS with no output.
