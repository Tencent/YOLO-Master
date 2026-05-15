# LoRA Package Refactor Spec

**Goal:** Replace the monolithic `ultralytics/utils/lora.py` module with a `ultralytics/utils/lora/` package while preserving the external import path `ultralytics.utils.lora`.

**Architecture:** This is a conservative package split. The public import surface stays stable, while the most self-contained responsibilities move into dedicated modules first. The first implementation phase splits out adapter I/O and training strategy logic, and keeps the config, fallback, wrapper, and `apply_lora()` orchestration path together in `api.py`.

**Non-goals:** No behavior changes to training defaults, target selection, optimizer grouping, checkpoint format, or CLI arguments. No renaming of existing public symbols in this phase.

---

## Why This Refactor

The current `ultralytics/utils/lora.py` file is ~3.8k lines and mixes:

- PEFT capability detection
- low-level adapter/fallback implementations
- config parsing and builder logic
- runtime safety guards
- adapter save/load/merge I/O
- training-time strategy helpers
- diagnostics and recommendation utilities

This makes the file expensive to review and risky to modify, because unrelated concerns live in the same import and edit surface.

## Target Package Layout

The refactor keeps the import path `ultralytics.utils.lora` stable by turning it into a package:

- `ultralytics/utils/lora/__init__.py`
  - public facade
  - re-exports public API from package modules
- `ultralytics/utils/lora/api.py`
  - PEFT import bootstrap
  - shared utilities and small helpers
  - fallback wrappers and top-level wrapper classes
  - `LoRAConfig`, `LoRAConfigBuilder`
  - runtime safety helpers
  - `apply_lora()`
  - param grouping and memory/stat helpers
- `ultralytics/utils/lora/io.py`
  - `save_lora_adapters()`
  - `load_lora_adapters()`
  - `merge_lora_weights()`
  - helper used only by adapter I/O and merge path
- `ultralytics/utils/lora/training.py`
  - `LoraTrainingStrategy`
  - `get_lora_training_stats()`
  - `suggest_lora_config_for_dataset()`

## Dependency Rules

Dependency direction for this phase:

- `__init__.py` depends on `api.py`, `io.py`, `training.py`
- `io.py` may depend on `api.py`
- `training.py` may depend on `api.py`
- `api.py` must not depend on `io.py` or `training.py`

This keeps `api.py` as the lowest-level orchestration module in phase 1 and avoids circular imports.

## Public Compatibility Requirements

These imports must continue to work unchanged after the split:

- `from ultralytics.utils.lora import apply_lora`
- `from ultralytics.utils.lora import LoRAConfig`
- `from ultralytics.utils.lora import LoraTrainingStrategy`
- `from ultralytics.utils.lora import save_lora_adapters, load_lora_adapters, merge_lora_weights`
- `from ultralytics.utils.lora import FewShotLoRAConv, ManualLoRAConv`
- `from ultralytics.utils.lora import _is_adapter_param, _unfreeze_detection_head`

The existing `trainer.py`, `model.py`, tests, and agent runtime imports should not need path changes.

## Migration Plan

Phase 1 in this implementation:

1. Create `docs/plans/2026-05-15-lora-package-spec.md`.
2. Replace `ultralytics/utils/lora.py` with `ultralytics/utils/lora/` package.
3. Move the most independent logic into:
   - `io.py`
   - `training.py`
4. Keep the remaining config/fallback/apply mainline in `api.py`.
5. Re-export the public API from `__init__.py`.
6. Run import-level and syntax-level verification.

Deferred to a later phase:

- Split fallback implementation into `fallback.py`
- Split config/builder into `config.py`
- Split target selection and safety guards into dedicated modules
- Reduce `api.py` further once the package boundary is stable

## Validation Requirements

Minimum validation for this phase:

- `python3 -m py_compile` passes for the new package files and dependent tests
- direct imports from `ultralytics.utils.lora` still resolve
- previously added unit tests for adapter load/merge behavior still compile against the new package layout

## Risks

- Circular imports introduced by moving package symbols too aggressively
- Import-order issues around PEFT bootstrap and `PeftProxy`
- Accidental loss of private helper availability used by tests or trainer internals

## Rollback Plan

If package migration causes import instability, rollback is simple:

1. Restore the single-file `ultralytics/utils/lora.py`
2. Remove `ultralytics/utils/lora/`
3. Keep the spec document as the design reference for a second attempt
