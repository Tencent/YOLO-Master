---
name: yolo-master-agent
description: Use when the user wants to train, validate, predict, track, export, benchmark, tune, inspect, or orchestrate YOLO-Master / Ultralytics experiments in this repository, including LoRA, MoE, multimodal inference/evaluation, and solutions workflows.
---

# YOLO-Master Agent Skill

## Use This Skill

Use this skill for any repository task that should drive the YOLO-Master stack end-to-end:

- `train`, `val`, `predict`, `track`, `export`, `benchmark`, `tune`
- model inspection and task detection
- LoRA save/load/merge and `yolo.lora.diagnose`
- PEFT comparison via `yolo.eval.peft_compare`
- Sparse SAHI comparison via `yolo.eval.sparse_sahi_compare`
- MoE diagnose/prune
- multimodal visual inference with OpenAI VLM/LLM cooperation
- multimodal batch evaluation over a dataset or image folder
- `solutions` workflows
- launchers for Gradio / Streamlit
- end-to-end orchestration via `yolo.pipeline.experiment`

## Execution Rule

First make sure the local Ultralytics framework is installed and the `yolo` CLI is available. Prefer the CLI over raw Python API for supported commands, and use the bundled dispatcher when you want a deterministic, structured run:

```bash
python -m pip install -e .
yolo version
python agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.train","inputs":{"model":"yolo11n.pt","data":"coco8.yaml"},"params":{"epochs":1,"imgsz":32}}'
```

On Apple Silicon hosts with PyTorch MPS support, the dispatcher now defaults heavy compute modes such as `train`, `val`, `benchmark`, `predict`, and `track` to `device=mps` when no explicit device is provided. Override with `runtime.device` or `params.device` if needed.

If the CLI run is auto-selected onto MPS/CUDA and fails for a device-level runtime reason, the dispatcher will retry once on CPU and return a structured `recovery` record with the full attempt trail.

When you need fast coverage across many skills or requests, use the AutoTrain-style validator first:

```bash
python agent/scripts/validate_yolo_master_skill.py --suite quick --pretty --summary-only
```

`quick` is the default agent loop. It combines `fast-smoke`, `dry-run`, and `contract` so agents can iterate without waiting on real model inspection or CLI cold-start probes. Use `all` only when you explicitly want the slower full non-manual regression pass.
The case pack now lives in `assets/autotrain_cases/` as skill-grouped JSON files. It includes multimodal dry-run/contract probes plus dry-run coverage for `yolo.pipeline.experiment`, `yolo.lora.diagnose`, and `yolo.eval.peft_compare`.

For quick regression checks, prefer the tiered suites:

```bash
python agent/scripts/validate_yolo_master_skill.py --suite quick --pretty --summary-only
python agent/scripts/validate_yolo_master_skill.py --suite fast-smoke --pretty --summary-only
python agent/scripts/validate_yolo_master_skill.py --suite cli-smoke --pretty --summary-only
python agent/scripts/validate_yolo_master_skill.py --suite deep-smoke --pretty --summary-only
python agent/scripts/validate_yolo_master_skill.py --suite extended --pretty --summary-only
python agent/scripts/validate_yolo_master_skill.py --suite dry-run --pretty --summary-only
python agent/scripts/validate_yolo_master_skill.py --suite contract --pretty --summary-only
python agent/scripts/validate_yolo_master_skill.py --suite all --pretty --summary-only
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
10. For long jobs, set `policy.async=true` on train/tune/pipeline requests to submit a subprocess job and return `job_id`, status, stdout/stderr, manifest, and progress paths.

## Multimodal Inference

`yolo.multimodal.infer` is an optional enhancement layer for visual reasoning. It does not replace `yolo.predict`: it runs YOLO first, condenses detections into reasoning evidence, then calls the OpenAI Responses API with `input_text` plus `input_image`, and optionally runs a second LLM refinement pass.

Environment variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` optional
- `OPENAI_API_MODE` optional, `auto`, `responses`, or `chat.completions`
- `OPENAI_VLM_MODEL` optional
- `OPENAI_LLM_MODEL` optional
- `structured_output=true` asks the VLM/LLM to return a strict JSON verdict that can be parsed into `verdict`
- `prompt_template=vlm_coco_multitask` asks the VLM to output caption, global classification, COCO-style object proposals, rough segmentation proxies, YOLO cross-checks, and fusion hints
- `prompt_template=vlm_open_world_detection` asks the VLM to preserve open-world objects, optional COCO mappings, rough segmentation proxies, captioning, and fusion hints for novel categories
- `prompt_template=vlm_open_world_detection_compact` is the compact open-world schema tuned for providers that tend to truncate long JSON, especially `qwen-vl-plus`
- `prompt_template=vlm_open_world_detect_classify_compact` focuses on a few grounded object proposals plus scene-level classes
- `prompt_template=vlm_open_world_caption_misses_compact` focuses on scene captioning plus the most important likely misses
- `max_output_tokens` defaults to `3500` in COCO multitask template mode; avoid lowering it below this unless you also reduce schema fields
- `use_marked_image=true` draws numbered YOLO boxes onto a lightweight marked copy before VLM inspection
- `visual_search_mode=auto` lets the VLM request crop-and-zoom follow-ups through `visual_search.needs_zoom` and `search_regions`
- `fusion_mode=preview` converts parsed VLM/LLM hints into metric-safe keep/suppress/add/relabel/adjust proposals plus COCO-style prediction records; use `fusion_mode=off` to disable it
- `fusion_policy=add_only` is now the default. It only allows filtered high-confidence VLM additions that look like genuine misses; use `balanced` or `aggressive` only when you explicitly want VLM-driven suppress/adjust/relabel actions
- `fusion_policy=open_world_assist` is the opt-in exploratory path. It keeps the normal metric preview for COCO-mappable outputs, but also preserves unmapped open-world objects in `multimodal.fusion.open_world_predictions_preview`
- open-world normalization now tries to anchor novel labels against the bundled `LVIS 1203` and `V3Det 13204` taxonomies, and batch reports expose `taxonomy.best`, `taxonomy.candidates`, dataset hit counts, and unmatched totals
- taxonomy matching is now intentionally conservative by default:
  - `open_world_taxonomy_min_score=40`
  - `open_world_taxonomy_require_exact_for_generic=true`
  This prevents weak generic matches such as `grass -> bear grass` from being treated as confirmed taxonomy anchors unless you explicitly loosen the policy
- open-world report aggregation now separates:
  - `enhancement_stats`: labels allowed to enter open-world enhancement statistics
  - `reasoning_only`: labels preserved for agent reasoning but filtered from aggregate enhancement stats
  Default filters are:
  - `open_world_filter_unmatched_taxonomy=true`
  - `open_world_filter_generic_labels=true`
- `open_world_assist_profile` gives the agent a higher-level mode switch for open-world runs without forcing you to set every threshold by hand:
  - `strict` default for evaluation-oriented runs: stronger taxonomy gate, generic labels filtered, unmatched labels kept as reasoning-only
  - `balanced`: keeps generic filtering, but allows unmatched taxonomy labels to remain in enhancement stats
  - `exploratory`: lowest taxonomy gate, generic/unmatched labels stay in enhancement stats for broad discovery passes
  Explicit param values still win over the profile defaults
- core implementation now lives in [`runtime/`](runtime); `scripts/` contains only thin executable wrappers that delegate into [`runtime/cli/`](runtime/cli)
- opt-in hooks now exist for:
  - IoU-based open-world relabeling via `open_world_iou_relabel_enabled`
  - WordNet hypernym fallback when taxonomy matching misses
  - cross-profile verified-list merging for prompt ensemble style arbitration
- When `vlm_model` looks like `qwen-vl-*` and `prompt_template=vlm_open_world_detection`, the dispatcher now auto-switches to a compact task profile unless you explicitly request a different template path:
  - default: `vlm_open_world_detect_classify_compact`
  - caption/miss emphasis: `vlm_open_world_caption_misses_compact`
  - override with `compact_open_world_profile=detect_classify|caption_misses`

Behavior:

- `thinking_with_image=true` attaches the image to the VLM request
- `enable_llm_refine=true` or `OPENAI_LLM_MODEL` enables the refinement pass
- missing `OPENAI_API_KEY` returns a structured `blocked` result
- Provider-aware defaults can be externalized under `runtime/multimodal/providers/*.yaml`; built-in configs currently include `openai` and `dashscope`.
- DashScope/OpenAI-compatible chat endpoints can use `params.provider="dashscope"` plus `DASHSCOPE_API_KEY`, or set `OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1` with `params.openai_api_mode="chat.completions"`.
- the manifest now preserves the `multimodal` block, including parsed verdicts, visual-search crop passes, fusion preview, and artifact paths when available
- every response envelope includes `usage.tokens` and `cost_estimate`; cost is `null` when provider pricing is not configured.

Example:

```bash
python agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.multimodal.infer","inputs":{"model":"yolo11n.pt","source":"ultralytics/assets/bus.jpg","prompt":"What matters most in this image?"},"params":{"thinking_with_image":true,"vlm_model":"gpt-4.1-mini","llm_model":"gpt-4.1-mini","max_reasoning_items":3,"max_reasoning_boxes":20},"policy":{"dry_run":true}}' --pretty
```

COCO multitask VLM prompt:

```bash
python agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.multimodal.infer","inputs":{"model":"yolo11n.pt","source":"ultralytics/assets/bus.jpg","prompt":"Detect, classify, segment roughly, caption, and propose metric-safe fusion changes."},"params":{"thinking_with_image":true,"structured_output":true,"prompt_template":"vlm_coco_multitask","use_marked_image":true,"visual_search_mode":"auto","fusion_mode":"preview","vlm_model":"qwen-vl-plus","llm_model":"qwen-plus","openai_api_mode":"chat.completions"}}' --pretty
```

Open-world VLM prompt:

```bash
python agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.multimodal.infer","inputs":{"model":"yolo11n.pt","source":"ultralytics/assets/bus.jpg","prompt":"Find visible objects, including novel categories outside COCO, and preserve them for downstream reasoning."},"params":{"thinking_with_image":true,"structured_output":true,"prompt_template":"vlm_open_world_detection","use_marked_image":true,"visual_search_mode":"auto","fusion_mode":"preview","fusion_policy":"open_world_assist","vlm_model":"qwen-vl-plus","llm_model":"qwen-plus","openai_api_mode":"chat.completions"}}' --pretty
```

## Multimodal Batch Evaluation

Use `yolo.multimodal.evaluate` when the agent needs to evaluate a real image sample or dataset split with YOLO first, then VLM/LLM cross-checks.

- `inputs.data` selects a dataset YAML such as `coco128.yaml`; `params.split` defaults to `val`
- `inputs.source` may point to a local image folder, image file, or image-list text file
- `params.limit`, `offset`, `stride`, `shuffle`, and `seed` control sampling; `limit=0` means all resolved images
- `params.run_yolo_val=true` also runs a YOLO-only validation baseline when `inputs.data` is available
- Ground-truth labels are read for reporting when available; they are not added to the VLM prompt unless `include_ground_truth_in_prompt=true`
- `params.prompt_template="vlm_coco_multitask"` enables VLM-side detection/classification/rough segmentation/caption output for downstream fusion experiments
- `params.prompt_template="vlm_open_world_detection"` enables a less conservative open-world path where novel categories are preserved even when they cannot be mapped into COCO metrics
- `params.use_marked_image=true` and `params.visual_search_mode=auto` enable Set-of-Mark-style box grounding and crop/zoom follow-up calls
- `params.fusion_mode="preview"` writes conservative fused prediction previews and, for batch evaluation, a `fusion-preview-coco-predictions.json` artifact for downstream COCO scoring
- Fusion is policy- and confidence-guarded by default: `add_only` blocks suppress/relabel/adjust, and even in broader policies high-confidence YOLO boxes are protected while box adjustments must stay close to the original box
- In `open_world_assist`, unmapped novel objects are emitted to `open_world_predictions_preview` instead of being dropped; only COCO-mappable predictions participate in metric preview and guardrail selection
- `open_world_assist` now defaults to an add-first posture: it prefers preserving novel objects and no longer enables suppress/adjust/relabel by default
- When YOLO-format labels are available, `evaluation.metric_preview` compares YOLO-only vs fused predictions on the sampled images and writes `fusion-metric-preview.json`; treat it as a fast same-sample guardrail, not an official benchmark
- `metric_guardrail` writes `metric-guarded-coco-predictions.json`: it keeps fused predictions only when there is a material change and same-sample `map50_95` shows a positive delta without recall regression; otherwise it falls back to YOLO-only predictions

Example:

```bash
python agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.multimodal.evaluate","runtime":{"prefer_cli":true,"prefer_mps":true},"inputs":{"model":"yolo11n.pt","data":"coco128.yaml","prompt":"Cross-check detector outputs and summarize obvious false positives, misses, duplicates, and uncertainty."},"params":{"limit":5,"split":"val","imgsz":640,"batch":1,"thinking_with_image":true,"prompt_template":"vlm_coco_multitask","use_marked_image":true,"visual_search_mode":"auto","fusion_mode":"preview","vlm_model":"qwen-vl-plus","llm_model":"qwen-plus","openai_base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1","openai_api_mode":"chat.completions"},"policy":{"dry_run":false}}' --pretty
```

## AutoTrain Loop

Use the bundled validator and case pack to keep this skill honest:

- case pack: `assets/autotrain_cases/` grouped by skill, with `assets/autotrain_cases.json` retained for compatibility
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
- `contract` includes single-image and batch multimodal stub probes so OpenAI-compatible request shaping, structured verdict parsing, and aggregation stay covered
- CLI failures now carry categorized hints so the agent can recover instead of stopping at a raw traceback.
- Built-in dataset YAML names such as `coco128.yaml` are auto-resolved against the local repository before execution.
- `doctor` returns environment, device selection source, and agent-facing recommendations.
- CLI train/val/predict/benchmark/export responses now carry environment metadata, and auto-selected runs can include a recovery trail when a device fallback occurs.

## Pipeline And PEFT Tools

Use `yolo.pipeline.experiment` for end-to-end train/val/export/benchmark flows. It accepts either stage keys such as `train`, `val`, `export`, `benchmark`, or an explicit `params.stages` list, and can include `inspect`, `lora_diagnose`, `moe_diagnose`, and `peft_compare`. Real runs write `progress.jsonl` next to the manifest for file-tail progress monitoring.

Use `yolo.lora.diagnose` to inspect active or loaded adapters:

```bash
python agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.lora.diagnose","inputs":{"model":"yolo11n.pt"},"params":{"path":"runs/train/exp/weights/lora_adapter_best","svd_max_layers":20,"spectrum_max_layers":12},"policy":{"dry_run":true}}' --pretty
```

Use `yolo.eval.peft_compare` to compare Full-SFT and PEFT variants with the same base train/val settings:

```bash
python agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.eval.peft_compare","inputs":{"model":"yolo11n.pt","data":"coco8.yaml"},"params":{"train":{"epochs":1,"imgsz":32,"batch":1},"variants":[{"name":"full_sft","train":{"lora_r":0}},{"name":"lora_r8","train":{"lora_type":"lora","lora_r":8,"lora_alpha":16}}]},"policy":{"dry_run":true}}' --pretty
```

## Manual Probes

Use these when you want stronger confidence than the default smoke suites without pulling slow jobs into routine validation.

Environment doctor and adaptive install probe:

```bash
python agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.system","action":"doctor","params":{"ensure_cli":true}}' --pretty
```

Real CLI training and validation probes on the bundled mini dataset:

```bash
python agent/scripts/validate_yolo_master_skill.py --suite extended --pretty --summary-only
```

Equivalent direct CLI train command:

```bash
yolo train model=scripts/peft_validation/yolo11n.pt data=agent/assets/mini-detect/mini_detect.yaml imgsz=64 epochs=1 batch=1 device=mps workers=0 plots=False verbose=False patience=1 project=runs/agent name=train-mini-mps-manual
```

Equivalent direct CLI val command:

```bash
yolo val model=scripts/peft_validation/yolo11n.pt data=agent/assets/mini-detect/mini_detect.yaml imgsz=16 batch=1 device=mps workers=0 plots=False verbose=False project=runs/agent name=val-mini-mps-manual
```

Structured dispatcher example with automatic MPS selection:

```bash
python agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.train","runtime":{"prefer_cli":true,"prefer_mps":true},"inputs":{"model":"scripts/peft_validation/yolo11n.pt","data":"agent/assets/mini-detect/mini_detect.yaml"},"params":{"epochs":1,"imgsz":64,"batch":1,"workers":0,"plots":false,"verbose":false,"patience":1},"artifacts":{},"policy":{"dry_run":false}}' --pretty
```

Regenerate the taxonomy-enriched small-batch open-world report from an existing real-run log:

```bash
python agent/scripts/regenerate_open_world_report.py \
  --input agent/logs/qwen-open-world-small-batch.json \
  --json-out agent/logs/qwen-open-world-small-batch-report.json \
  --md-out agent/logs/qwen-open-world-small-batch-report.md
```

## References

- Read [`README.md`](README.md) for the concise directory map and maintenance boundaries.
- Read [`references/skill-architecture.md`](references/skill-architecture.md) for the full architecture map, skill registry, request/response contract, and execution logic.
- Read [`references/thinking-with-image.md`](references/thinking-with-image.md) when improving VLM visual reasoning, marked-image prompting, crop/zoom search, or COCO metric fusion.
- Read the open-world taxonomy assets in [`assets/open-world-taxonomy`](assets/open-world-taxonomy) for reusable class references, including `LVIS 1203` and `V3Det 13204` category lists plus source metadata.

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
