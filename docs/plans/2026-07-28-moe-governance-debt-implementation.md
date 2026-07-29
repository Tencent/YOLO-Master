# MoE Governance and Engineering Debt Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add auditable MoE variant retirement and remove three high-risk MoLoRA maintenance ambiguities without breaking checkpoint structure.

**Architecture:** Extend the existing usage audit instead of adding a second scanner. Keep the legacy weak registry as a write-only compatibility surface while routing all internal loss computation through canonical step records. Move per-expert construction into the MoLoRA base initializer and centralize capacity semantics there and in configuration validation.

**Tech Stack:** Python 3.8+, PyTorch, dataclasses, JSON, pytest, Ruff.

---

### Task 1: Version-Aware Variant Retirement

**Files:**
- Modify: `ultralytics/nn/modules/moe/__init__.py`
- Modify: `scripts/audit_moe_usage.py`
- Create: `docs/governance/moe-variant-usage.json`
- Modify: `docs/governance/moe-class-lifecycle.md`
- Test: `tests/test_moe_ssot.py`

**Steps:**

1. Add failing tests for a disjoint `DEPRECATED_MOE_CLASSES` tier and two-snapshot eligibility.
2. Run `pytest tests/test_moe_ssot.py -v` and confirm the new tests fail.
3. Add the empty deprecated tier, snapshot loading/recording, YAML registry usage collection, and eligibility calculation.
4. Record the `8.4.101` baseline snapshot without declaring any class deprecated.
5. Run `pytest tests/test_moe_ssot.py -v` and `python scripts/audit_moe_usage.py` and confirm both pass.

### Task 2: Canonical Auxiliary-Loss Consumers

**Files:**
- Modify: `ultralytics/nn/modules/moe/_common.py`
- Modify: `ultralytics/nn/peft/molora/layer.py`
- Modify: `ultralytics/nn/peft/molora/model.py`
- Modify: `ultralytics/nn/mixture_loss.py`
- Modify: `ultralytics/engine/extensions/recovery.py`
- Test: `tests/test_routing_aux_contract.py`
- Test: `tests/test_p2_fixes.py`
- Test: `tests/test_ddp_lifecycle_ema_nan.py`

**Steps:**

1. Add failing tests proving internal consumers ignore legacy-only injected values while `_registry_set` still publishes to both stores.
2. Run the focused tests and confirm the legacy-only expectation fails.
3. Replace internal registry reads with `get_aux_record`, `iter_aux_records`, or `collect_aux_loss`.
4. Retain legacy writes and clearing for compatibility, but remove the composite and MoLoRA loss fallbacks.
5. Run the focused tests and confirm graph connectivity, step filtering, NaN recovery, and no double counting.

### Task 3: Parent-Owned MoE-Aware Initialization

**Files:**
- Modify: `ultralytics/nn/peft/molora/layer.py`
- Modify: `ultralytics/nn/peft/molora/moe_aware.py`
- Test: `tests/test_moe_aware_peft.py`

**Steps:**

1. Add a failing regression test that asserts fields introduced by the parent initializer exist on the aware layer.
2. Run `pytest tests/test_moe_aware_peft.py -v` and confirm failure against the duplicated initializer.
3. Add optional `expert_ranks` and `router_calibration` arguments to `MoLoRALayer.__init__`, including rank validation.
4. Replace the aware layer's manual initialization with a complete `super().__init__` call.
5. Run the aware-layer tests and state-dict round-trip tests.

### Task 4: Capacity Factor Contract

**Files:**
- Modify: `ultralytics/nn/peft/molora/config.py`
- Modify: `ultralytics/nn/peft/molora/layer.py`
- Modify: `ultralytics/nn/peft/molora/moe_aware.py`
- Modify: `ultralytics/cfg/default.yaml`
- Test: `tests/test_molora.py`
- Test: `tests/test_default_config_integrity.py`

**Steps:**

1. Add failing tests for non-finite rejection, default-unlimited behavior, an explicit factor of `1.0`, and the `num_experts` unlimited boundary.
2. Run the focused tests and confirm current `1.0` handling fails the new contract.
3. Set defaults to `0.0`, centralize the unlimited predicate, and apply the capacity calculation for the full active interval.
4. Update configuration comments and validation.
5. Run `pytest tests/test_molora.py tests/test_moe_aware_peft.py tests/test_default_config_integrity.py -v`.

### Task 5: Regression Gates

**Files:**
- Verify all modified source and test files.

**Steps:**

1. Run `pytest tests/test_routing_aux_contract.py tests/test_p2_fixes.py tests/test_ddp_lifecycle_ema_nan.py -v`.
2. Run `pytest tests/test_molora_routing_aware_merge.py tests/test_molora_dtype.py tests/test_molora_backend_roundtrip.py tests/test_vpeft.py -v`.
3. Run `pytest tests/test_moe_router_boundaries.py tests/test_master_model_configs.py -v`.
4. Run `ruff check` and `ruff format --check` on touched Python files.
5. Run `codespell` on touched documentation and source files.
