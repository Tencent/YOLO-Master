<!--
PR 标题（GitHub 上填这一行，对齐 #253 的 [犀牛鸟-Ax] 风格）：
[犀牛鸟-C3]：Fix V-PEFT capacity guard for layers narrower than the smallest candidate rank
分支：lycyhrc/YOLO-Master:fix/vpeft-capacity-guard（base: Tencent:main，2 commits，tip daed306）
下面的内容从 "## Summary" 开始整段粘贴到 PR 描述框。
-->

## Summary

- Add the missing `C_cap` hard constraint (`RankCapacityConstraint`: a rank is feasible only when `rank <= min(in, out)`) so the solver no longer projects narrow layers as adapter targets.
- Re-check the selected targets while the placement plan is built: skipped layers are recorded in `metadata["capacity_excluded"]` and logged as a greppable `[V-PEFT] capacity-excluded: ...` line.
- `PlacementPlan.validate_model` keeps its strict check but now reports every violating target with its capacity in one pass.
- Freeze the layers that stay unadapted when an explicit target list or a profiling filter leaves them out, so the manual backend matches the PEFT backend (adapters are the only trainable weights).

This is a bug fix only: it adds no configuration option, changes no default (`rank_min` stays `4`) and leaves the legacy planner untouched.

## Problem

None of the seven hard constraints (`C_op` / `C_sem` / `C_budget` / `C_deploy` / `C_compat` / `C_moe` / `C_div`) models layer capacity, while `PlacementPlan.validate_model` computes `min(in, out)` and rejects any rank above it. The projection stage and the validation stage therefore disagree:

- projection: `routing.routing_network.2` (capacity 3), `25.dfl.conv`, `0.conv` are feasible and get selected;
- validation: `apply_lora` raises `ValueError` on exactly those targets.

With `lora_planner_backend=vpeft` the exception is swallowed by `except (ValueError, TypeError)` and the run silently falls back to the legacy planner (`vpeft_strict=True` fails instead), so V-PEFT was not actually in effect. The workaround is a manual `lora_exclude_modules` list, which matches by substring (`0.conv` also matches `10.conv`).

Reproduction before the fix:

```bash
python -c "
import torch.nn as nn
from ultralytics.utils.lora.api import apply_lora
from ultralytics.utils.lora.config import LoRAConfig
m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.Conv2d(8, 8, 3, padding=1))
apply_lora(m, LoRAConfig(r=2, alpha=4, backend='fallback', planner_backend='vpeft',
                         adapter_budget=100_000, vpeft_strict=True))
"
```

Audit output after the fix (EsMoE-N shaped narrow-layer model):

```
[V-PEFT] capacity-excluded: 2 layer(s) below the requested rank: 0(rank=4>capacity=3), 3(rank=4>capacity=3)
plan: ACCEPT targets=['1', '2'] capacity_excluded=[{'name': '0', 'rank': 4, 'capacity': 3}, ...]
```

## Behaviour

| Scenario | Before | After |
|---|---|---|
| Narrow-layer model + `vpeft` | plan validation raises → silent fallback to legacy (raises under `vpeft_strict=True`) | narrow layers excluded, remaining targets placed, audit recorded |
| Capacity logging | only the exception / fallback warning | `[V-PEFT] capacity-excluded: N layer(s) below the requested rank: ...` |
| Hand-written plan over capacity | first violating target raises | all violating targets listed in one message, audit field referenced |
| Explicit `target_modules` | skipped layers stayed trainable | unadapted layers are frozen |
| `rank_min=4` semantics / existing config keys | — | unchanged (no new key, `default.yaml` and `cfg/__init__.py` untouched) |

## Validation

```bash
python -m pytest tests/test_vpeft_capacity_guard.py tests/test_vpeft.py tests/test_placement_plan_schema.py \
  tests/test_vpeft_lora_e2e.py tests/test_lora_fallback_effective_config.py tests/test_model_adapter_facade.py \
  tests/test_adapter_backend_contract.py tests/test_lora_training_strategy.py tests/test_p0_system_gates.py -q
# 70 passed, 1 failed
```

- The single failure is `test_p0_system_gates.py::test_cpu_gloo_two_rank_routed_continuous_training`, which asserts on `torchrun` subprocess stdout; on this host that stdout carries an `Ultralytics settings updated to the latest schema` warning. No code path in this PR is involved and the test was not skipped or relaxed.
- New `tests/test_vpeft_capacity_guard.py` (8 cases): constraint semantics, unknown capacity is not excluded, registry and legacy-alias registration, solver excludes narrow layers, plan audit field, single-pass strict validation, `rank == capacity` boundary on both sides.
- `tests/test_vpeft.py` hard-constraint list assertion updated to include `C_cap` (intentional: the default hard set grows).
- Ruff on the changed files and `git diff --check` pass.

## Scope

Two commits, source and tests only — no experiment scaffolding, data or reports:

1. `fix(vpeft): align solver rank projection with plan capacity validation` — `ultralytics/vpeft/constraints.py` (`C_cap` + registry/alias registration), `ultralytics/vpeft/placement_plan.py` (single-pass error report), `ultralytics/utils/lora/api.py` (second capacity pass + audit field + log), `tests/test_vpeft_capacity_guard.py`, `tests/test_vpeft.py`.
2. `fix(lora): freeze layers left out of an explicit target list` — `ultralytics/utils/lora/api.py`. `_replace_conv_with_manual_lora` only froze the layers it wrapped, so layers skipped by `target_modules`, by the depthwise/`only_3x3` filters or by head-like names kept training (and were missing from `save_adapters`). The official P0 gate `test_planner_adapter_full_lifecycle` covers that path; before commit 2 it only "passed" because the plan validation failure degraded the whole run to the legacy planner.

## Experiments (context, not the acceptance basis for this PR)

NEU-DET (1800 images) and DeepPCB with YOLO-Master EsMoE-N (2.8M parameters), 3 strategies × 3 seeds, same budget (epochs 100, batch 8, imgsz 640, amp off):

| Strategy | NEU mAP50 (best) | DeepPCB mAP50 (best) | Trainable parameters | Peak VRAM |
|---|---|---|---|---|
| full_sft | 0.767±0.014 | 0.989±0.000 | 2,813,626 (100%) | ~5.3G |
| vpeft (planner, 6/6 ACCEPT) | 0.694±0.052* | 0.881±0.072 | 116,736 (4.15%) | ~4.1G |
| frozen_backbone | 0.694±0.041* | 0.769±0.151 | 1,906,822 (67.8%) | ~3.7G |

\* robust estimate (one sporadic seed removed, n=3); including it, vpeft is 0.612±0.167.
Those runs worked around the defect above with `lora_exclude_modules`; **no training was re-run with the fixed code**.

## Limitations

- No accuracy claim. This PR only removes the validation failure and the silent fallback; the ablation numbers still come from the pre-fix workaround.
- Seed stability: on NEU, vpeft and frozen each produced one extreme seed (0.369 / 0.210) while the re-run with a fresh seed was normal, and the DeepPCB frozen oscillation at seed 824 reproduces deterministically with the same seed. The observation is limited to this seed pool and does not estimate a failure probability.
- EsMoE-N is a 2.8M model, so the parameter-efficiency comparison does not automatically transfer to larger models; under equal epochs the three strategies take about the same wall-clock (data pipeline bound).
- `capacity_excluded` is audit metadata only and does not change the plan schema version, but `C_cap` joins the default hard-constraint set and therefore the plan fingerprint.

## Non-goals

- No silent rank clipping (`ADAPT` / `REFUSE` semantics unchanged).
- `exclude_modules` substring matching is left as is.
- Legacy planner behaviour unchanged.
- Non-Conv2d unadapted modules (Linear / BN) are not frozen yet.
- No experiment scaffolding, data or reports are added to the repository.

## Reproduction

```bash
git fetch origin && git checkout af961b9
git checkout fix/vpeft-capacity-guard   # or check out this PR branch
python -m pytest tests/test_vpeft_capacity_guard.py -q
python -c "
import torch.nn as nn
from ultralytics.utils.lora.api import apply_lora
from ultralytics.utils.lora.config import LoRAConfig
m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.Conv2d(8, 16, 3, padding=1),
                  nn.Conv2d(16, 4, 1), nn.Conv2d(4, 3, 1))
w = apply_lora(m, LoRAConfig(r=4, alpha=8, backend='fallback', planner_backend='vpeft',
                             adapter_budget=1_000_000, vpeft_strict=True))
print(w.lora_target_modules)                                  # ['1', '2']
print(w.lora_placement_plan['metadata']['capacity_excluded'])  # 0/3 excluded by capacity
"
```

## PPT

https://drive.google.com/file/d/1dg8tZyI11xJR1JrCCKBWqbC4_kFgmsVp/view?usp=sharing
