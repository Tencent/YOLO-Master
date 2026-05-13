---
name: yolo-master-agent
description: Use when the user wants to train, validate, predict, track, export, benchmark, tune, inspect, or orchestrate YOLO-Master / Ultralytics experiments in this repository, including LoRA, MoE, and solutions workflows.
---

# YOLO-Master Agent Skill

## Use This Skill

Use this skill for any repository task that should drive the YOLO-Master stack end-to-end:

- `train`, `val`, `predict`, `track`, `export`, `benchmark`, `tune`
- model inspection and task detection
- LoRA save/load/merge
- MoE diagnose/prune
- `solutions` workflows
- launchers for Gradio / Streamlit

## Execution Rule

First make sure the local Ultralytics framework is installed and the `yolo` CLI is available. Prefer the CLI over raw Python API for supported commands, and use the bundled dispatcher when you want a deterministic, structured run:

```bash
python -m pip install -e .
yolo version
python yolo-master-agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.train","inputs":{"model":"yolo11n.pt","data":"coco8.yaml"},"params":{"epochs":1,"imgsz":32}}'
```

On Apple Silicon hosts with PyTorch MPS support, the dispatcher now defaults heavy compute modes such as `train`, `val`, `benchmark`, `predict`, and `track` to `device=mps` when no explicit device is provided. Override with `runtime.device` or `params.device` if needed.

If the CLI run is auto-selected onto MPS/CUDA and fails for a device-level runtime reason, the dispatcher will retry once on CPU and return a structured `recovery` record with the full attempt trail.

When you need fast coverage across many skills or requests, use the AutoTrain-style validator first:

```bash
python yolo-master-agent/scripts/validate_yolo_master_skill.py --suite quick --pretty --summary-only
```

`quick` is the default agent loop. It combines `fast-smoke`, `dry-run`, and `contract` so agents can iterate without waiting on real model inspection or CLI cold-start probes. Use `all` only when you explicitly want the slower full non-manual regression pass.

For quick regression checks, prefer the tiered suites:

```bash
python yolo-master-agent/scripts/validate_yolo_master_skill.py --suite quick --pretty --summary-only
python yolo-master-agent/scripts/validate_yolo_master_skill.py --suite fast-smoke --pretty --summary-only
python yolo-master-agent/scripts/validate_yolo_master_skill.py --suite cli-smoke --pretty --summary-only
python yolo-master-agent/scripts/validate_yolo_master_skill.py --suite deep-smoke --pretty --summary-only
python yolo-master-agent/scripts/validate_yolo_master_skill.py --suite extended --pretty --summary-only
python yolo-master-agent/scripts/validate_yolo_master_skill.py --suite dry-run --pretty --summary-only
python yolo-master-agent/scripts/validate_yolo_master_skill.py --suite contract --pretty --summary-only
python yolo-master-agent/scripts/validate_yolo_master_skill.py --suite all --pretty --summary-only
```

## Workflow

1. Inspect the request and normalize paths.
2. Install/refresh the local Ultralytics package first when the `yolo` CLI is missing.
3. Prefer `policy.dry_run=true` while validating or evolving the skill surface.
4. On Apple Silicon, let the dispatcher pick `mps` by default for train/val/eval unless the request already sets `device`.
5. Let the dispatcher auto-complete safe runtime defaults such as `workers=0` on macOS train/val paths when the request leaves them unset.
6. Use `yolo` CLI for supported tasks; fall back to Python API only when the CLI does not cover the action.
7. For `predict` and `track`, accept `source` in either `inputs.source` or `params.source`; the dispatcher will normalize it before CLI emission.
8. Pass all task-specific options through `params` unchanged.
9. Return structured artifacts, metrics, evaluation summaries, environment reports, and next actions.
10. For long jobs, use `async`/launcher behavior and write a manifest.

## AutoTrain Loop

Use the bundled validator and case pack to keep this skill honest:

- case pack: `assets/autotrain_cases.json`
- report: `logs/autotrain-report.json`
- bootstrap: `python -m pip install -e .`
- dispatcher supports `policy.dry_run=true` for cheap coverage before real runs
- `yolo` CLI is the preferred execution surface for supported actions
- `quick` is the default iteration suite: `fast-smoke` + `dry-run` + `contract`
- the validator enables a short-lived runtime cache for Torch/MPS detection so repeated subprocess cases do not re-import the stack
- `fast-smoke` protects bootstrap and planning paths with tight timing budgets
- `cli-smoke` validates real `yolo` CLI cold-start execution
- `deep-smoke` holds heavyweight real-model inspection and local `.pt` inference checks
- `all` runs every non-manual case, including slower `cli-smoke` and `deep-smoke`
- `extended-cli` carries slower real CLI validation probes such as mini-dataset `yolo train` and `yolo val` on `mps`, and is marked `manual_only`
- `contract` verifies failure-path behavior and manifest emission
- `contract` now also includes in-process recovery probes so auto device fallback semantics stay covered without adding test-only hooks to the dispatcher
- CLI failures now carry categorized hints so the agent can recover instead of stopping at a raw traceback.
- Built-in dataset YAML names such as `coco128.yaml` are auto-resolved against the local repository before execution.
- `doctor` returns environment, device selection source, and agent-facing recommendations.
- CLI train/val/predict/benchmark/export responses now carry environment metadata, and auto-selected runs can include a recovery trail when a device fallback occurs.

## Manual Probes

Use these when you want stronger confidence than the default smoke suites without pulling slow jobs into routine validation.

Environment doctor and adaptive install probe:

```bash
python yolo-master-agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.system","action":"doctor","params":{"ensure_cli":true}}' --pretty
```

Real CLI training and validation probes on the bundled mini dataset:

```bash
python yolo-master-agent/scripts/validate_yolo_master_skill.py --suite extended --pretty --summary-only
```

Equivalent direct CLI train command:

```bash
yolo train model=scripts/peft_validation/yolo11n.pt data=yolo-master-agent/assets/mini-detect/mini_detect.yaml imgsz=64 epochs=1 batch=1 device=mps workers=0 plots=False verbose=False patience=1 project=runs/agent name=train-mini-mps-manual
```

Equivalent direct CLI val command:

```bash
yolo val model=scripts/peft_validation/yolo11n.pt data=yolo-master-agent/assets/mini-detect/mini_detect.yaml imgsz=16 batch=1 device=mps workers=0 plots=False verbose=False project=runs/agent name=val-mini-mps-manual
```

Structured dispatcher example with automatic MPS selection:

```bash
python yolo-master-agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.train","runtime":{"prefer_cli":true,"prefer_mps":true},"inputs":{"model":"scripts/peft_validation/yolo11n.pt","data":"yolo-master-agent/assets/mini-detect/mini_detect.yaml"},"params":{"epochs":1,"imgsz":64,"batch":1,"workers":0,"plots":false,"verbose":false,"patience":1},"artifacts":{},"policy":{"dry_run":false}}' --pretty
```

## References

- Read [`references/skill-architecture.md`](references/skill-architecture.md) for the full architecture map, skill registry, request/response contract, and execution logic.

## Guardrails

- Do not hardcode new CLI strings when a Python API exists.
- Keep `params` as the pass-through layer for new Ultralytics arguments.
- Prefer `yolo` CLI for supported commands; use Python API only as fallback.
- On Apple Silicon, prefer `mps` for training and validation unless the request explicitly overrides the device.
- Consume `evaluation` in addition to `metrics` when judging train/val runs.
- Use `yolo.system doctor` before long runs when the agent needs to confirm install state, selected device, and local repo activation.
- Prefer the `recovery` field over raw stderr when a run auto-falls back from MPS/CUDA to CPU.
- Keep slow real training out of default validator suites; use the manual probe path instead.
- Treat UI launchers and research scripts as launcher-style skills, not plain sync functions.
