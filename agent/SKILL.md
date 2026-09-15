---
name: yolo-master-agent
description: Use when the user wants to run a YOLO-Master task (train/val/predict/track/export/benchmark) or use the Agent Skill dispatcher.
---

# YOLO-Master Agent Skill

## Use This Skill

- `train`, `val`, `predict`, `track`, `export`, `benchmark`, `tune`
- LoRA diagnose, PEFT compare, MoE diagnose/prune
- Multimodal visual inference and batch evaluation
- End-to-end orchestration via `yolo.pipeline.experiment`

## Execution Rule

If `yolo version` fails, run `pip install -e .` first. Prefer the `yolo` CLI over raw Python API for supported commands. Use the bundled dispatcher for deterministic, structured runs:

```bash
python agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.train","inputs":{"model":"yolo11n.pt","data":"coco8.yaml"},"params":{"epochs":1,"imgsz":32}}'
```

On Apple Silicon, the dispatcher defaults heavy compute (`train`, `val`, `benchmark`, `predict`, `track`) to `device=mps`. Override with `runtime.device` or `params.device`. If MPS/CUDA fails, the dispatcher retries once on CPU and returns a `recovery` record.

Use `policy.dry_run=true` for cheap coverage before real runs. For long jobs, set `policy.async=true` on train/tune/pipeline to submit a subprocess job.

## Workflow

1. Ensure `yolo` CLI is available; install if missing.
2. Dispatch the task via `run_yolo_master_skill.py` with appropriate `skill`, `inputs`, and `params`.
3. Return structured artifacts, metrics, and next actions from the dispatcher response.

## Validation

```bash
python agent/scripts/validate_yolo_master_skill.py --suite quick --pretty --summary-only
```

`quick` = `fast-smoke` + `dry-run` + `contract`. Use `--suite all` for the full regression pass.

## References

- [Skill architecture](references/skill-architecture.md) — skill registry, request/response contract, execution logic
- [Multimodal inference](references/multimodal-inference.md) — VLM/LLM parameters, prompt templates, fusion, open-world taxonomy
- [Multimodal evaluation](references/multimodal-evaluation.md) — batch evaluation, fusion guardrails, metric guardrails
- [AutoTrain loop](references/autotrain.md) — validator suites, case pack, tiered coverage
- [Pipeline & PEFT](references/pipeline-peft.md) — pipeline experiments, release audit, LoRA diagnose, PEFT compare
- [Manual probes](references/manual-probes.md) — extended CLI suite, direct train/val, environment doctor
- [Thinking with image](references/thinking-with-image.md) — VLM visual reasoning, marked-image prompting, crop/zoom search
- Open-world taxonomy assets in `assets/open-world-taxonomy` — `LVIS 1203` and `V3Det 13204` category lists

## Guardrails

- New Ultralytics CLI args must go through `params` dict, not as top-level dispatcher keys.
- Consume `evaluation` in addition to `metrics` when judging train/val runs.
- Prefer the `recovery` field over raw stderr when a run auto-falls back from MPS/CUDA to CPU.
- Treat UI launchers and research scripts as launcher-style skills, not plain sync functions.
