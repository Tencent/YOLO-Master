# Multimodal Inference

`yolo.multimodal.infer` is an optional enhancement layer for visual reasoning. It does not replace `yolo.predict`: it runs YOLO first, condenses detections into reasoning evidence, then calls the OpenAI Responses API with `input_text` plus `input_image`, and optionally runs a second LLM refinement pass.

## Parameters

- `thinking_with_image=true` attaches the image to the VLM request
- `enable_llm_refine=true` or `OPENAI_LLM_MODEL` enables the refinement pass
- `structured_output=true` asks the VLM/LLM to return a strict JSON verdict parseable into `verdict`
- `max_output_tokens` defaults to `3500` in COCO multitask template mode

### Prompt Templates

- `vlm_coco_multitask` — caption, global classification, COCO-style object proposals, rough segmentation proxies, YOLO cross-checks, and fusion hints
- `vlm_open_world_detection` — preserve open-world objects, optional COCO mappings, captioning, and fusion hints for novel categories
- `vlm_open_world_detection_compact` — compact schema for providers that truncate long JSON (e.g. `qwen-vl-plus`)
- `vlm_open_world_detect_classify_compact` — grounded object proposals plus scene-level classes
- `vlm_open_world_caption_misses_compact` — scene captioning plus important likely misses

### Visual Search & Fusion

- `use_marked_image=true` draws numbered YOLO boxes onto a lightweight marked copy before VLM inspection
- `visual_search_mode=auto` lets the VLM request crop-and-zoom follow-ups
- `fusion_mode=preview` converts parsed VLM/LLM hints into metric-safe keep/suppress/add/relabel/adjust proposals; `fusion_mode=off` disables
- `fusion_policy=add_only` (default) — only allows filtered high-confidence VLM additions; use `balanced` or `aggressive` for VLM-driven suppress/adjust/relabel
- `fusion_policy=open_world_assist` — preserves unmapped open-world objects in `multimodal.fusion.open_world_predictions_preview`

### Open-World Taxonomy

- Anchors novel labels against bundled `LVIS 1203` and `V3Det 13204` taxonomies
- Conservative defaults: `open_world_taxonomy_min_score=40`, `open_world_taxonomy_require_exact_for_generic=true`
- `open_world_assist_profile`: `strict` | `balanced` | `exploratory`
- Report aggregation separates `enhancement_stats` (labels in stats) from `reasoning_only` (labels for reasoning only)
- Default filters: `open_world_filter_unmatched_taxonomy=true`, `open_world_filter_generic_labels=true`
- Opt-in hooks: IoU-based open-world relabeling, WordNet hypernym fallback, cross-profile verified-list merging

### Provider Configuration

- Environment variables: `OPENAI_API_KEY`, `OPENAI_BASE_URL` (optional), `OPENAI_API_MODE` (`auto`/`responses`/`chat.completions`), `OPENAI_VLM_MODEL`, `OPENAI_LLM_MODEL`
- Provider-aware defaults in `runtime/multimodal/providers/*.yaml` (currently `openai` and `dashscope`)
- DashScope: `params.provider="dashscope"` + `DASHSCOPE_API_KEY`, or set `OPENAI_BASE_URL` + `params.openai_api_mode="chat.completions"`
- When `vlm_model` looks like `qwen-vl-*` with `prompt_template=vlm_open_world_detection`, auto-switches to compact profile; override with `compact_open_world_profile=detect_classify|caption_misses`
- Missing `OPENAI_API_KEY` returns structured `blocked` result
- Every response envelope includes `usage.tokens` and `cost_estimate`

## Examples

Basic inference:

```bash
python agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.multimodal.infer","inputs":{"model":"yolo11n.pt","source":"ultralytics/assets/bus.jpg","prompt":"What matters most in this image?"},"params":{"thinking_with_image":true,"vlm_model":"gpt-4.1-mini","llm_model":"gpt-4.1-mini","max_reasoning_items":3,"max_reasoning_boxes":20},"policy":{"dry_run":true}}' --pretty
```

COCO multitask:

```bash
python agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.multimodal.infer","inputs":{"model":"yolo11n.pt","source":"ultralytics/assets/bus.jpg","prompt":"Detect, classify, segment roughly, caption, and propose metric-safe fusion changes."},"params":{"thinking_with_image":true,"structured_output":true,"prompt_template":"vlm_coco_multitask","use_marked_image":true,"visual_search_mode":"auto","fusion_mode":"preview","vlm_model":"qwen-vl-plus","llm_model":"qwen-plus","openai_api_mode":"chat.completions"}}' --pretty
```

Open-world detection:

```bash
python agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.multimodal.infer","inputs":{"model":"yolo11n.pt","source":"ultralytics/assets/bus.jpg","prompt":"Find visible objects, including novel categories outside COCO, and preserve them for downstream reasoning."},"params":{"thinking_with_image":true,"structured_output":true,"prompt_template":"vlm_open_world_detection","use_marked_image":true,"visual_search_mode":"auto","fusion_mode":"preview","fusion_policy":"open_world_assist","vlm_model":"qwen-vl-plus","llm_model":"qwen-plus","openai_api_mode":"chat.completions"}}' --pretty
```
