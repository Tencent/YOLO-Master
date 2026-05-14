# Thinking-With-Image Research Notes

This note maps mainstream visual reasoning methods to practical YOLO-Master agent upgrades. The goal is not to expose private chain-of-thought, but to make image reasoning inspectable through structured evidence, region proposals, and fusion hints that can be scored against COCO.

## Current Baseline

The current agent runs YOLO first, compresses detections into `index/label/confidence/xyxy`, can render a marked image, attaches the image to an OpenAI-compatible VLM, optionally runs crop-and-zoom visual search, and can run an LLM refine pass. It now emits a conservative fusion preview, but metric improvement is not proven until the fused COCO-style predictions are evaluated against the same ground truth split.

## Mainstream Method Families

### 1. Multimodal Chain-of-Thought

Representative work: Multimodal Chain-of-Thought reasoning, LLaVA-CoT-style stepwise reasoning.

Core idea: ask the model to decompose visual reasoning into perception, grounding, and answer synthesis. For production agents, keep the hidden reasoning private and request only structured intermediate artifacts: evidence, detected objects, uncertainty, and recommended actions.

Best fit here: keep `answer`, `visual_evidence`, `yolo_cross_check`, and `uncertainty`, but add explicit `vlm_detections` and `fusion_hints`. For open-world experiments, preserve `open_label` even when no COCO class exists.

Risk: free-form CoT can hallucinate and is hard to evaluate. Use strict JSON schemas and COCO class constraints.

### 2. Set-of-Mark / Visual Prompting

Representative work: Set-of-Mark prompting for GPT-4V and follow-on visual prompting systems.

Core idea: overlay marks, region IDs, boxes, or masks on the image so the VLM can refer to stable visual anchors. This improves grounding because the model can say "region 3 is wrong" instead of describing vague positions.

Best fit here:

- Render YOLO boxes with stable indices.
- Optionally render high-confidence masks or SAM proposals later.
- Ask VLM to return `linked_yolo_indices`, `suppress_yolo_indices`, `adjust_boxes`, and `add_vlm_detections`.

Risk: excessive marks can hide small objects or bias the VLM. Keep overlays lightweight and preserve an unmarked image fallback.

### 3. Visual Search / Crop-and-Zoom

Representative work: V* and guided visual search methods.

Core idea: whole-image VLM passes miss small, occluded, or crowded objects. The agent first asks for uncertain search regions, crops/zooms them, re-queries the VLM, and merges the evidence.

Best fit here:

- Add `visual_search.needs_zoom` and `visual_search.search_regions`.
- Run second-pass crop VLM calls only for high-priority regions.
- Use crop evidence to suppress, relabel, or add boxes before COCO evaluation.

Risk: cost grows quickly. Gate crop calls by uncertainty, small-object classes, or disagreement between YOLO and VLM.

### 4. Tool-Augmented Visual Reasoning

Representative work: tool-using VLM pipelines and visual programming approaches.

Core idea: do not ask the VLM to do geometry alone. Let deterministic tools handle resizing, box normalization, IoU, NMS, cropping, and COCO JSON conversion; let the VLM provide semantic judgments.

Best fit here:

- Use Python for IoU, NMS, and COCO result emission.
- Use VLM only for `keep/suppress/add/relabel/adjust` proposals.
- Always report whether a VLM-derived box is `exact`, `estimated`, or `rough`.
- For open-world mode, keep a second lane where the VLM can emit `open_world_predictions_preview` that are not forced into COCO metrics.

Risk: VLM geometry is approximate. Do not trust rough polygons for segmentation mAP without mask refinement.

### 5. Self-Consistency and Critic Passes

Representative work: test-time scaling, self-consistency, reflection/critic passes for MLLMs.

Core idea: run multiple prompts or a VLM+LLM critic and accept changes only when independent passes agree.

Best fit here:

- Require `confidence >= threshold` for VLM-added detections.
- Keep original YOLO boxes unless VLM and LLM agree on suppression.
- Track per-image deltas and evaluate before accepting a fusion rule globally.

Risk: more calls can amplify shared model bias. Measure against held-out COCO subsets.

## Recommended Enhancement Path

### Phase A: Structured Multitask Prompt

Implemented via `assets/prompts/vlm_coco_multitask.md` and `assets/prompts/vlm_open_world_detection.md`.

It asks the VLM for:

- `caption`
- `global_classification`
- `vlm_detections`
- `vlm_segmentation`
- `visual_search`
- `yolo_cross_check`
- `fusion_hints`

Use:

```json
{
  "params": {
    "prompt_template": "vlm_coco_multitask",
    "structured_output": true,
    "thinking_with_image": true
  }
}
```

### Phase B: Marked Image Prompting

Implemented in `run_yolo_master_skill.py`.

1. Create a marked copy of the input image with YOLO boxes and numeric IDs.
2. Send the marked image as the primary VLM input for stable box references.
3. Parse `linked_yolo_indices` and `fusion_hints`.

Minimum viable overlay:

- class label
- confidence
- stable `index`
- thin high-contrast rectangle

### Phase C: Fusion-to-COCO Evaluation

Initial preview implemented via `multimodal.fusion` and `fusion-preview-coco-predictions.json`.

Convert VLM output into prediction deltas:

- keep confirmed YOLO boxes
- suppress high-confidence false positives only when VLM confidence is high
- add VLM detections only when class is COCO-valid and geometry is usable
- relabel only when IoU with YOLO box is strong and semantic evidence is clear
- run COCO evaluation on the fused prediction JSON

Success metric:

- compare YOLO-only vs fused predictions on the same split
- report Precision/Recall/mAP50/mAP50-95 deltas
- keep qualitative verdicts separate from metric deltas

Current preview limits:

- It is conservative and auditable, not yet a claimed mAP improvement.
- VLM-added boxes must pass `fusion_add_confidence_min`.
- Suppress/adjust proposals are retained as actions with thresholds so the agent can inspect what changed.
- Suppression and relabeling are blocked for high-confidence YOLO boxes by default, and adjusted boxes must retain enough IoU with the original detector box.
- The default fusion policy is now `add_only`, because our real runs have not shown stable gains from default suppress/adjust/relabel behavior.
- A stronger opt-in path now exists: `fusion_policy=open_world_assist` plus `prompt_template=vlm_open_world_detection`. This keeps unmapped novel objects in a separate preview lane instead of discarding them.
- For `qwen-vl-plus`-style providers, long open-world JSON often truncates before the outer object closes. The dispatcher now auto-switches `vlm_open_world_detection` to a compact task profile for those providers unless the caller overrides the template.
- The compact profile is now split into `detect_classify` and `caption_misses` lanes so the agent can spend its token budget where the request actually needs it.
- Even inside `add_only`, proposals are filtered toward likely misses: low overlap with strong YOLO boxes, acceptable `bbox_quality`, and enough VLM confidence.
- In `open_world_assist`, the agent can accept novel-object additions with a separate `fusion_open_world_confidence_min` threshold, while still keeping COCO metric guardrails limited to mappable predictions.
- `open_world_assist` now behaves as add-first by default: keep novel additions, but do not assume suppress/adjust/relabel authority unless you escalate to broader fusion policies.
- Open-world taxonomy anchoring now has a conservative quality gate. Weak lexical matches are retained as candidates for analysis, but only become `taxonomy.best` when they clear `open_world_taxonomy_min_score`; generic labels additionally require exact matches by default.
- The agent now also supports `open_world_assist_profile` so it can shift between evaluation-safe and exploration-heavy behavior without rewriting every threshold:
  - `strict` for metric-facing validation
  - `balanced` for mixed discovery plus cleaner stats
  - `exploratory` for broad open-world harvesting
- The helper stack is now split out of `run_yolo_master_skill.py`, and the ensemble layer can merge strict/balanced/exploratory outputs into a verified open-world list for downstream use.
- Full COCO multitask JSON needs enough output budget; `vlm_coco_multitask` defaults to `3500` output tokens and keeps lists capped to avoid truncation.
- `evaluation.metric_preview` now compares YOLO-only vs fused predictions against available YOLO labels on the exact sampled images, giving a fast precision/recall/mAP delta before running official validation.
- `metric_guardrail` automatically falls back to YOLO-only prediction records when fused preview makes no material difference, or does not produce a positive same-sample `map50_95` gain, or hurts recall.
- Final acceptance still requires running COCO evaluation on `fusion-preview-coco-predictions.json`.

## Sources To Revisit

- OpenAI, "Thinking with images" - https://openai.com/index/thinking-with-images/
- Set-of-Mark Prompting - https://som-gpt4v.github.io/
- V* Guided Visual Search - https://vstar-seal.github.io/
- Multimodal Chain-of-Thought reasoning - https://arxiv.org/abs/2302.00923
- LLaVA-CoT - https://arxiv.org/abs/2411.10440
- Whiteboard-of-Thought - https://arxiv.org/abs/2406.14562
