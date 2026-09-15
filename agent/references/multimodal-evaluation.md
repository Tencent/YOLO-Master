# Multimodal Batch Evaluation

Use `yolo.multimodal.evaluate` to evaluate a real image sample or dataset split with YOLO first, then VLM/LLM cross-checks.

## Parameters

- `inputs.data` selects a dataset YAML such as `coco128.yaml`; `params.split` defaults to `val`
- `inputs.source` may point to a local image folder, image file, or image-list text file
- `params.limit`, `offset`, `stride`, `shuffle`, and `seed` control sampling; `limit=0` means all resolved images
- `params.run_yolo_val=true` also runs a YOLO-only validation baseline
- Ground-truth labels are read for reporting when available; not added to VLM prompt unless `include_ground_truth_in_prompt=true`
- `params.prompt_template` — same templates as multimodal inference (`vlm_coco_multitask`, `vlm_open_world_detection`, etc.)
- `params.use_marked_image=true` and `params.visual_search_mode=auto` enable Set-of-Mark-style box grounding

## Fusion & Guardrails

- `fusion_mode="preview"` writes conservative fused prediction previews and `fusion-preview-coco-predictions.json` for downstream COCO scoring
- Default policy `add_only` blocks suppress/relabel/adjust; high-confidence YOLO boxes are protected
- `open_world_assist` preserves unmapped novel objects in `open_world_predictions_preview`; defaults to add-first posture
- `evaluation.metric_preview` compares YOLO-only vs fused predictions → `fusion-metric-preview.json` (same-sample guardrail, not official benchmark)
- `metric_guardrail` → `metric-guarded-coco-predictions.json`: keeps fused predictions only when `map50_95` shows positive delta without recall regression

## Example

```bash
python agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.multimodal.evaluate","runtime":{"prefer_cli":true,"prefer_mps":true},"inputs":{"model":"yolo11n.pt","data":"coco128.yaml","prompt":"Cross-check detector outputs and summarize obvious false positives, misses, duplicates, and uncertainty."},"params":{"limit":5,"split":"val","imgsz":640,"batch":1,"thinking_with_image":true,"prompt_template":"vlm_coco_multitask","use_marked_image":true,"visual_search_mode":"auto","fusion_mode":"preview","vlm_model":"qwen-vl-plus","llm_model":"qwen-plus","openai_base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1","openai_api_mode":"chat.completions"},"policy":{"dry_run":false}}' --pretty
```
