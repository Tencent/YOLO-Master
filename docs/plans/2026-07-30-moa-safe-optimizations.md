# MoA Safe Optimizations Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Apply the report items that are directly supported by the current code while preserving model interfaces and export behavior.

**Architecture:** Keep dense routing semantics unchanged, but reduce unnecessary intermediate state and bound Regional Head KV resolution. Export/tracing paths remain numerically dense and omit Python-only diagnostics.

**Tech Stack:** PyTorch, pytest, Ruff.

---

### Task 1: Correct linear-attention accumulator scaling

**Files:**
- Modify: `ultralytics/nn/modules/moa/heads.py:288-293`
- Test: `tests/test_moa.py`

Remove the extra L2 normalization of `k^T @ v`; retain the existing fp32 accumulation and denominator safeguards. Add a regression assertion that the implementation follows the documented numerator/denominator computation without an accumulator normalization factor.

### Task 2: Bound Regional Head KV tokens

**Files:**
- Modify: `ultralytics/nn/modules/moa/heads.py:147-199`
- Test: `tests/test_moa.py`

Add an optional `max_kv_tokens` limit (default 4096). At runtime, double the configured pooling stride until the pooled map fits the limit, while preserving the existing adaptive pooling and small-map fallback. Verify stride selection and output shape.

### Task 3: Make sequential MoA heads the default

**Files:**
- Modify: `ultralytics/nn/modules/moa/block.py:46-58`
- Modify: `ultralytics/nn/modules/moa/wrappers.py:58-72`
- Test: `tests/test_moa.py`, `tests/test_p2_fixes.py`

Change only the default argument to `True`; explicit `False` remains supported for compatibility and equivalence tests remain valid.

### Task 4: Skip Python-only routing diagnostics while exporting

**Files:**
- Modify: `ultralytics/nn/modules/routing_protocol.py`
- Modify: `ultralytics/nn/modules/moa/block.py`
- Modify: `ultralytics/nn/modules/moa/wrappers.py`
- Test: `tests/test_moa.py`

Expose a small export/tracing predicate and use it to skip auxiliary-loss publication and detached snapshot construction in MoA modules. Exported numerical tensors continue through the existing dense path.

### Task 5: Verify

Run:

```bash
pytest tests/test_moa.py tests/test_p2_fixes.py tests/test_mixture_export.py -q
ruff check ultralytics/nn/modules/moa ultralytics/nn/modules/routing_protocol.py tests/test_moa.py tests/test_p2_fixes.py
ruff format --check ultralytics/nn/modules/moa ultralytics/nn/modules/routing_protocol.py tests/test_moa.py tests/test_p2_fixes.py
```
