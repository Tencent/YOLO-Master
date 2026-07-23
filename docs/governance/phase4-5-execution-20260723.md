# YOLO-Master Phase 4-5 Execution Record (2026-07-23)

## Phase 4: V-PEFT opt-in closure

Implemented the V-PEFT backend in `ultralytics/utils/lora/api.py`.

- `lora_planner_backend=legacy` remains the default and does not build a V-PEFT graph.
- `lora_planner_backend=vpeft` builds a `ComputationGraph`, applies the configured AO/DCO/MIP solver, and emits a versioned `PlacementPlan`.
- The plan records model fingerprint, solver, budget usage, selected targets, ranks, constraints, status, and a plan fingerprint.
- The existing LoRA injection path consumes the selected target names and uses the minimum planned rank as the current global-rank compatibility bridge.
- Empty/refused plans and solver errors fall back to the legacy/fixed-rank path; the plan remains attached to runtime metadata for auditability.

Verification:

```text
python3 -m pytest -q tests/test_vpeft_lora_e2e.py tests/test_placement_plan_schema.py tests/test_planner.py
80 passed, 1 warning
```

## Phase 5: Controlled MoA/MoT smoke

Commands:

```text
python3 benchmarks/run.py --suite moa_vs_mot --device cpu --imgsz 64 --warmup 0 --iterations 1
python3 benchmarks/run.py --suite mot_scene_router --device cpu --imgsz 64 --warmup 0 --iterations 1
python3 scripts/diagnose_mot_routing.py --model ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-scene-n.yaml --device cpu --imgsz 64 --synthetic --max-images 3 --project runs/benchmarks/mot_scene_diagnostics
```

Results are structural CPU smoke measurements, not publication-quality latency:

| Suite | Case | Params | P50 ms | Expert calls | Mean Gini | Mean entropy | Collapsed layers |
|---|---|---:|---:|---:|---:|---:|---:|
| `moa_vs_mot` | v10 MoA | 3,576,587 | 91.72 | 27 | 0.2802 | 0.7869 | 3 |
| `moa_vs_mot` | v10 MoT | 4,055,333 | 153.28 | 19 | 0.2743 | 0.7870 | 3 |
| `mot_scene_router` | v10 MoT | 4,055,333 | 111.54 | 19 | 0.2743 | 0.7870 | 3 |
| `mot_scene_router` | v10 scene-aware MoT | 4,055,687 | 125.34 | 19 | 0.2741 | 0.7870 | 3 |

Interpretation:

- MoT uses `sample_sparse` dispatch in expert blocks, but the current smoke still shows three collapsed routed layers.
- Scene-aware routing is numerically close to the legacy router on synthetic probes and does not yet demonstrate scene separation; keep it experimental and opt-in.
- No default model configuration was changed as a result of this smoke.

Artifacts:

- `runs/benchmarks/moa_vs_mot/report.md`
- `runs/benchmarks/mot_scene_router/report.md`
- `runs/benchmarks/mot_scene_diagnostics/mot_routing_scenarios.csv`
- `runs/benchmarks/mot_scene_diagnostics/mot_deformable_activation_check.csv`
