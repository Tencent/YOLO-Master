from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from runtime.cli.contract import ensure_manifest_dir, json_safe, plan_response, response, write_manifest
from runtime.cli.dataset import collect_multimodal_evaluation_images, read_ground_truth_summary
from runtime.cli.device import apply_runtime_defaults, collect_environment_report, resolve_device_selection
from runtime.cli.executor import replace_cli_device
from runtime.cli.normalize import (
    default_request_id,
    is_dry_run,
    normalize_request,
    parse_bool,
    resolved_path,
    slugify,
    split_yolo_and_multimodal_params,
    split_yolo_multimodal_evaluate_params,
)
from runtime.evaluation.metrics import (
    aggregate_metric_preview,
    aggregate_multimodal_evaluation,
    build_item_metric_preview,
    build_metric_guardrail,
    detection_label_counts,
    overall_multimodal_evaluation_status,
)
from runtime.multimodal.fusion import build_multimodal_fusion_preview as fusion_build_multimodal_fusion_preview, merge_verdicts
from runtime.multimodal.runtime import (
    attach_multimodal_verdict,
    build_multimodal_evaluation_prompt,
    build_thinking_with_image_prompt,
    default_llm_refine_developer_prompt,
    default_vlm_developer_prompt,
    multimodal_prompt_from_request,
    multimodal_overall_status,
    openai_config,
    run_visual_search_crop_passes as runtime_run_visual_search_crop_passes,
    summarize_results_for_reasoning,
)
from runtime.multimodal.visual import (
    encode_image_reference_for_openai as encode_image_reference_for_openai_raw,
    image_source_for_openai as visual_image_source_for_openai,
    normalize_detection_boxes,
    render_marked_image as visual_render_marked_image,
)
from runtime.open_world.taxonomy import (
    aggregate_open_world_comparison,
    apply_open_world_assist_profile_defaults,
    build_open_world_comparison_entry,
    default_multimodal_max_output_tokens,
    effective_prompt_template_name,
)

SKILL_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = SKILL_ROOT / "assets" / "prompts"


@dataclass(frozen=True)
class MultimodalDeps:
    build_model: Callable[[dict[str, Any]], Any]
    call_openai_compatible: Callable[..., dict[str, Any]]
    run_val: Callable[[dict[str, Any]], dict[str, Any]]


def run_multimodal_infer(request: dict[str, Any], deps: MultimodalDeps) -> dict[str, Any]:
    params = dict(request["params"])
    yolo_params, multimodal_params = split_yolo_and_multimodal_params(params)
    multimodal_params = apply_open_world_assist_profile_defaults(multimodal_params)
    source = request["inputs"].get("source") or yolo_params.pop("source", None) or multimodal_params.get("source")
    if source is None:
        raise ValueError("`inputs.source` is required for yolo.multimodal.infer.")
    prompt = multimodal_prompt_from_request(request, multimodal_params)
    provider_cfg = openai_config(multimodal_params)
    if provider_cfg.get("api_family") != "openai-compatible":
        raise ValueError(f"Unsupported multimodal provider: {provider_cfg['provider']}")

    max_items = int(yolo_params.pop("max_items", multimodal_params.get("max_reasoning_items", 3)))
    max_boxes = int(multimodal_params.get("max_reasoning_boxes", 20))
    thinking_with_image = parse_bool(multimodal_params.get("thinking_with_image"), True)
    structured_output = parse_bool(multimodal_params.get("structured_output"), True)
    prompt_template = multimodal_params.get("prompt_template")
    resolved_prompt_template = effective_prompt_template_name(provider_cfg, prompt_template, multimodal_params, prompt)
    default_max_output_tokens = default_multimodal_max_output_tokens(
        provider_cfg,
        prompt_template,
        multimodal_params,
        structured_output=structured_output,
        user_prompt=prompt,
    )
    max_output_tokens = int(multimodal_params.get("max_output_tokens", default_max_output_tokens))
    use_marked_image = parse_bool(multimodal_params.get("use_marked_image"), bool(resolved_prompt_template))
    visual_search_mode = str(multimodal_params.get("visual_search_mode") or "auto")
    fusion_mode = str(multimodal_params.get("fusion_mode") or "preview")
    device_selection = resolve_device_selection(request, yolo_params)
    yolo_params, chosen_device, auto_completed = apply_runtime_defaults(request, yolo_params, purpose="predict")
    effective_request = deepcopy(request)
    effective_request["params"] = yolo_params
    effective_request["inputs"]["source"] = source

    method = str(multimodal_params.get("method") or ("thinking-with-image" if thinking_with_image else "detector-text-reflection"))
    if is_dry_run(request):
        return plan_response(
            effective_request,
            "multimodal inference dry run prepared",
            "orchestrator",
            "yolo.multimodal.infer",
            params={
                "stages": [
                    {"name": "yolo_predict", "executor": "python_api", "target": "YOLO(...).predict", "params": yolo_params},
                    {
                        "name": "vlm_reasoning",
                        "executor": "openai.compatible",
                        "provider": provider_cfg["provider"],
                        "model": provider_cfg["vlm_model"],
                        "api_mode": provider_cfg["api_mode"],
                        "image": "input_image" if thinking_with_image else "text_only",
                        "method": method,
                        "structured_output": structured_output,
                        "prompt_template": resolved_prompt_template,
                        "marked_image": use_marked_image,
                    },
                    {
                        "name": "visual_search_crop_pass",
                        "executor": "openai.compatible",
                        "mode": visual_search_mode,
                        "enabled": visual_search_mode.lower() not in {"off", "none", "false", "0"},
                    },
                    {
                        "name": "fusion_preview",
                        "executor": "python",
                        "strategy": "metric_safe_v1",
                        "mode": fusion_mode,
                    },
                    {
                        "name": "llm_refine",
                        "executor": "openai.compatible",
                        "provider": provider_cfg["provider"],
                        "model": provider_cfg["llm_model"],
                        "api_mode": provider_cfg["api_mode"],
                        "enabled": bool(provider_cfg.get("llm_model") or multimodal_params.get("enable_llm_refine")),
                    },
                ],
                "prompt": prompt,
            },
            extra={
                "environment": collect_environment_report(effective_request, selected_device=chosen_device),
                "auto_completed": auto_completed,
                "multimodal": {
                    "provider": provider_cfg["provider"],
                    "vlm_model": provider_cfg["vlm_model"],
                    "llm_model": provider_cfg["llm_model"],
                    "api_mode": provider_cfg["api_mode"],
                    "api_key_env": provider_cfg["api_key_env"],
                    "api_key_present": provider_cfg["api_key_present"],
                    "method": method,
                    "thinking_with_image": thinking_with_image,
                    "structured_output": structured_output,
                    "prompt_template": resolved_prompt_template,
                    "open_world_assist_profile": multimodal_params.get("open_world_assist_profile"),
                    "open_world_filters": {
                        "taxonomy_min_score": multimodal_params.get("open_world_taxonomy_min_score"),
                        "taxonomy_require_exact_for_generic": multimodal_params.get("open_world_taxonomy_require_exact_for_generic"),
                        "filter_unmatched_taxonomy": multimodal_params.get("open_world_filter_unmatched_taxonomy"),
                        "filter_generic_labels": multimodal_params.get("open_world_filter_generic_labels"),
                    },
                    "use_marked_image": use_marked_image,
                    "visual_search_mode": visual_search_mode,
                    "fusion_mode": fusion_mode,
                },
            },
        )

    detections: list[dict[str, Any]]
    save_dir = None
    yolo_error = None
    if bool(multimodal_params.get("skip_yolo", False)):
        detections = json_safe(request["inputs"].get("detections") or multimodal_params.get("detections") or [])
    else:
        try:
            model = deps.build_model(effective_request)
            results = model.predict(source=source, **yolo_params)
        except Exception as exc:
            if device_selection["source"] == "auto" and chosen_device not in (None, "cpu"):
                retry_params = replace_cli_device(yolo_params, "cpu")
                effective_request["params"] = retry_params
                try:
                    model = deps.build_model(effective_request)
                    results = model.predict(source=source, **retry_params)
                    yolo_params = retry_params
                    chosen_device = "cpu"
                    auto_completed["device"] = "cpu"
                    auto_completed["device_source"] = "recovery"
                except Exception:
                    raise exc
            else:
                raise
        save_dir = getattr(model.predictor, "save_dir", None)
        detections = summarize_results_for_reasoning(results, max_items=max_items, max_boxes=max_boxes)

    image_ref = visual_image_source_for_openai(source, detections, resolved_path=resolved_path)
    vlm_result: dict[str, Any]
    llm_result: dict[str, Any] | None = None
    image_meta: dict[str, Any] = {
        "requested": image_ref,
        "thinking_with_image": thinking_with_image,
        "attached": False,
    }
    visual_artifacts: list[dict[str, Any]] = []
    fusion_artifacts: list[dict[str, Any]] = []
    visual_search_passes: list[dict[str, Any]] = []
    marked_meta: dict[str, Any] | None = None
    marked_error: dict[str, Any] | None = None
    if thinking_with_image and image_ref is None:
        vlm_result = {
            "status": "blocked",
            "provider": "openai",
            "summary": "No image source was available for multimodal reasoning.",
        }
    elif not provider_cfg["api_key_present"]:
        vlm_result = {
            "status": "blocked",
            "provider": "openai",
            "summary": "OPENAI_API_KEY is not set; multimodal reasoning was skipped.",
            "api_key_env": provider_cfg["api_key_env"],
        }
    else:
        try:
            encoded_image = None
            if thinking_with_image and image_ref is not None:
                reasoning_image_ref = image_ref
                if use_marked_image and not str(image_ref).startswith(("http://", "https://", "data:image/")):
                    try:
                        marked = visual_render_marked_image(
                            image_ref,
                            detections,
                            resolved_path=resolved_path,
                            output_dir=ensure_manifest_dir(request) / "visual-search",
                            prefix=slugify(Path(str(image_ref)).stem),
                        )
                        reasoning_image_ref = marked["path"]
                        marked_meta = marked
                        visual_artifacts.append({"kind": "marked_image", "path": marked["path"]})
                    except Exception as mark_exc:
                        marked_error = {"type": type(mark_exc).__name__, "message": str(mark_exc)}
                encoded_image = encode_image_reference_for_openai_raw(
                    reasoning_image_ref,
                    resolved_path=resolved_path,
                    max_bytes=int(multimodal_params.get("max_image_bytes", 20_000_000)),
                )
                image_meta = {k: v for k, v in encoded_image.items() if k != "image_url"}
                image_meta["requested"] = image_ref
                image_meta["reasoning_input"] = reasoning_image_ref
                if marked_meta:
                    image_meta["marked"] = marked_meta
                if marked_error:
                    image_meta["marked_error"] = marked_error
                image_meta["thinking_with_image"] = True
                image_meta["attached"] = True
            user_text = build_thinking_with_image_prompt(
                prompt,
                detections,
                method=method,
                thinking_with_image=thinking_with_image,
                structured_output=structured_output,
                prompt_template=str(resolved_prompt_template) if resolved_prompt_template is not None else None,
                prompt_dir=PROMPT_DIR,
            )
            vlm_result = deps.call_openai_compatible(
                model=str(provider_cfg["vlm_model"]),
                user_text=user_text,
                developer_text=str(
                    multimodal_params.get("developer_prompt")
                    or multimodal_params.get("system_prompt")
                    or default_vlm_developer_prompt(resolved_prompt_template)
                ),
                image_url=encoded_image["image_url"] if encoded_image else None,
                image_detail=str(multimodal_params.get("image_detail", "auto")),
                base_url=provider_cfg["base_url"],
                provider=str(provider_cfg.get("provider", "openai")),
                api_key_env=str(provider_cfg.get("api_key_env", "OPENAI_API_KEY")),
                api_mode=str(provider_cfg["api_mode"]),
                max_output_tokens=max_output_tokens,
                temperature=float(multimodal_params["temperature"]) if "temperature" in multimodal_params else None,
            )
            vlm_result = attach_multimodal_verdict(vlm_result)
            if vlm_result.get("status") == "ok" and image_ref is not None:
                visual_search_passes, search_artifacts = runtime_run_visual_search_crop_passes(
                    image_path=image_ref,
                    base_prompt=prompt,
                    detections=detections,
                    initial_verdict=vlm_result.get("verdict", {}),
                    provider_cfg=provider_cfg,
                    multimodal_params=multimodal_params,
                    output_dir=ensure_manifest_dir(request),
                    max_output_tokens=max_output_tokens,
                    method=method,
                    resolved_path_fn=resolved_path,
                    normalize_detection_boxes_fn=normalize_detection_boxes,
                    call_openai_compatible_fn=deps.call_openai_compatible,
                    attach_multimodal_verdict_fn=attach_multimodal_verdict,
                )
                visual_artifacts.extend(search_artifacts)
        except Exception as exc:
            vlm_result = {
                "status": "failed",
                "provider": provider_cfg.get("provider", "openai"),
                "summary": "Failed to prepare or call VLM reasoning.",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }

    llm_enabled = bool(provider_cfg.get("llm_model") or multimodal_params.get("enable_llm_refine"))
    if llm_enabled and vlm_result.get("status") == "ok":
        llm_model = str(provider_cfg.get("llm_model") or provider_cfg["vlm_model"])
        refine_prompt = (
            "Refine this YOLO + VLM inference into a compact final answer. Do not add unsupported visual claims. "
            "Return exactly one JSON object without Markdown fences. Use these keys: answer, visual_evidence, "
            "yolo_cross_check, uncertainty, recommended_next_actions. In yolo_cross_check, include arrays named "
            "confirmed, false_positives, possible_misses, duplicate_or_fragmented, and notes when applicable. "
            "If the VLM answer includes caption, global_classification, vlm_detections, vlm_segmentation, "
            "visual_search, or fusion_hints, preserve those keys and refine them conservatively.\n\n"
            f"User task:\n{prompt}\n\n"
            f"YOLO detection summary:\n{json.dumps(json_safe(detections), ensure_ascii=False, indent=2)}\n\n"
            f"VLM answer:\n{vlm_result.get('text', '')}\n\n"
            f"Parsed VLM verdict:\n{json.dumps(json_safe(vlm_result.get('verdict', {})), ensure_ascii=False, indent=2)}"
            f"\n\nVisual search crop passes:\n{json.dumps(json_safe(visual_search_passes), ensure_ascii=False, indent=2)}"
        )
        llm_result = deps.call_openai_compatible(
            model=llm_model,
            user_text=refine_prompt,
            developer_text=default_llm_refine_developer_prompt(resolved_prompt_template),
            base_url=provider_cfg["base_url"],
            provider=str(provider_cfg.get("provider", "openai")),
            api_key_env=str(provider_cfg.get("api_key_env", "OPENAI_API_KEY")),
            api_mode=str(provider_cfg["api_mode"]),
            max_output_tokens=max_output_tokens,
            temperature=float(multimodal_params["temperature"]) if "temperature" in multimodal_params else None,
        )
        llm_result = attach_multimodal_verdict(llm_result)

    vlm_verdict = vlm_result.get("verdict", {}) if isinstance(vlm_result.get("verdict"), dict) else {}
    llm_verdict = llm_result.get("verdict", {}) if isinstance(llm_result, dict) and isinstance(llm_result.get("verdict"), dict) else {}
    fusion_verdict = merge_verdicts(vlm_verdict, llm_verdict)
    fusion_preview = fusion_build_multimodal_fusion_preview(
        detections=detections,
        verdict=fusion_verdict,
        multimodal_params=multimodal_params,
        image_path=image_ref,
        normalize_detection_boxes_fn=normalize_detection_boxes,
    )
    open_world_comparison = build_open_world_comparison_entry(
        image_path=image_ref,
        detections=detections,
        fusion_preview=fusion_preview,
        verdict=fusion_verdict,
        multimodal_params=multimodal_params,
        effective_prompt_template=str(resolved_prompt_template) if resolved_prompt_template is not None else None,
    )
    if fusion_preview.get("enabled"):
        fusion_path = ensure_manifest_dir(request) / "fusion-preview.json"
        fusion_path.write_text(json.dumps(json_safe({"image": image_ref, "fusion": fusion_preview}), ensure_ascii=False, indent=2), encoding="utf-8")
        fusion_preview["artifact"] = str(fusion_path.resolve())
        fusion_artifacts.append({"kind": "fusion_preview", "path": str(fusion_path.resolve())})

    overall_status = multimodal_overall_status(vlm_result, llm_result)
    summary_map = {
        "ok": "multimodal inference finished",
        "blocked": "YOLO inference finished, but multimodal reasoning was blocked",
        "partial": "YOLO inference finished; multimodal reasoning was incomplete",
    }
    summary = summary_map[overall_status]
    environment = collect_environment_report(effective_request, selected_device=chosen_device)
    payload = response(
        request["skill"],
        overall_status,
        summary,
        job={"mode": "sync", "save_dir": json_safe(save_dir), "executor": "python_api+openai", "device": chosen_device},
        results=detections,
        environment=environment,
        auto_completed=auto_completed,
        multimodal={
            "method": method,
            "thinking_with_image": thinking_with_image,
            "structured_output": structured_output,
            "prompt_template": prompt_template,
            "effective_prompt_template": resolved_prompt_template,
            "open_world_assist_profile": multimodal_params.get("open_world_assist_profile"),
            "open_world_filters": {
                "taxonomy_min_score": multimodal_params.get("open_world_taxonomy_min_score"),
                "taxonomy_require_exact_for_generic": multimodal_params.get("open_world_taxonomy_require_exact_for_generic"),
                "filter_unmatched_taxonomy": multimodal_params.get("open_world_filter_unmatched_taxonomy"),
                "filter_generic_labels": multimodal_params.get("open_world_filter_generic_labels"),
            },
            "use_marked_image": use_marked_image,
            "visual_search": {"mode": visual_search_mode, "passes": visual_search_passes, "artifacts": visual_artifacts},
            "provider": {
                "name": provider_cfg["provider"],
                "base_url": provider_cfg["base_url"],
                "api_mode": provider_cfg["api_mode"],
                "api_key_env": provider_cfg["api_key_env"],
                "api_key_present": provider_cfg["api_key_present"],
            },
            "image": image_meta,
            "prompt": prompt,
            "vlm": vlm_result,
            "llm_refine": llm_result or {"status": "skipped"},
            "fusion": fusion_preview,
            "open_world_comparison": open_world_comparison,
        },
        next_actions=["yolo.predict", "yolo.val"],
    )
    if yolo_error:
        payload["yolo_error"] = yolo_error
    if save_dir:
        payload["artifacts"] = [{"kind": "directory", "path": str(Path(save_dir).resolve())}]
    if visual_artifacts or fusion_artifacts:
        payload.setdefault("artifacts", [])
        payload["artifacts"].extend(visual_artifacts + fusion_artifacts)
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_multimodal_evaluate(request: dict[str, Any], deps: MultimodalDeps) -> dict[str, Any]:
    params = dict(request["params"])
    yolo_params, multimodal_params, evaluate_params = split_yolo_multimodal_evaluate_params(params)
    multimodal_params = apply_open_world_assist_profile_defaults(multimodal_params)
    prompt = multimodal_prompt_from_request(request, multimodal_params)
    provider_cfg = openai_config(multimodal_params)
    if provider_cfg.get("api_family") != "openai-compatible":
        raise ValueError(f"Unsupported multimodal provider: {provider_cfg['provider']}")

    thinking_with_image = parse_bool(multimodal_params.get("thinking_with_image"), True)
    structured_output = parse_bool(multimodal_params.get("structured_output"), True)
    prompt_template = multimodal_params.get("prompt_template")
    resolved_prompt_template = effective_prompt_template_name(provider_cfg, prompt_template, multimodal_params, prompt)
    default_max_output_tokens = default_multimodal_max_output_tokens(
        provider_cfg,
        prompt_template,
        multimodal_params,
        structured_output=structured_output,
        user_prompt=prompt,
    )
    max_output_tokens = int(multimodal_params.get("max_output_tokens", default_max_output_tokens))
    use_marked_image = parse_bool(multimodal_params.get("use_marked_image"), bool(resolved_prompt_template))
    visual_search_mode = str(multimodal_params.get("visual_search_mode") or "auto")
    fusion_mode = str(multimodal_params.get("fusion_mode") or "preview")
    include_ground_truth = parse_bool(
        evaluate_params.get("include_ground_truth_in_prompt", evaluate_params.get("include_ground_truth")),
        False,
    )
    run_yolo_val = parse_bool(evaluate_params.get("run_yolo_val"), False)
    continue_on_error = parse_bool(evaluate_params.get("continue_on_error"), True)
    report_name = str(evaluate_params.get("report_name") or "multimodal-evaluation.json")
    data_ref_for_baseline = request["inputs"].get("data") or evaluate_params.get("data") or yolo_params.get("data")
    device_selection = resolve_device_selection(request, yolo_params)
    yolo_params, chosen_device, auto_completed = apply_runtime_defaults(request, yolo_params, purpose="predict")
    if "verbose" not in yolo_params:
        yolo_params["verbose"] = False
        auto_completed["verbose"] = False

    effective_request = deepcopy(request)
    effective_request["params"] = yolo_params
    if request["inputs"].get("data") is not None:
        effective_request["inputs"]["data"] = request["inputs"]["data"]
    if request["inputs"].get("source") is not None:
        effective_request["inputs"]["source"] = request["inputs"]["source"]

    images, dataset_info, names = collect_multimodal_evaluation_images(request, evaluate_params, yolo_params, multimodal_params)
    effective_request["inputs"]["source"] = str(images[0].parent)
    environment = collect_environment_report(effective_request, selected_device=chosen_device)

    if is_dry_run(request):
        return plan_response(
            effective_request,
            "multimodal evaluation dry run prepared",
            "orchestrator",
            "yolo.multimodal.evaluate",
            params={
                "stages": [
                    {"name": "collect_images", "executor": "dataset_or_source_resolver"},
                    {"name": "yolo_predict_batch", "executor": "python_api", "target": "YOLO(...).predict"},
                    {"name": "marked_image_batch", "executor": "pillow", "enabled": use_marked_image},
                    {"name": "vlm_reasoning_batch", "executor": "openai.compatible"},
                    {"name": "visual_search_crop_pass", "executor": "openai.compatible", "mode": visual_search_mode},
                    {"name": "fusion_preview", "executor": "python", "strategy": "metric_safe_v1", "mode": fusion_mode},
                    {"name": "aggregate_report", "executor": "python"},
                ],
                "sample_count": len(images),
                "sample_images": [str(path) for path in images[:10]],
                "split": dataset_info.get("split"),
                "run_yolo_val": run_yolo_val,
                "include_ground_truth_in_prompt": include_ground_truth,
                "prompt": prompt,
            },
            extra={
                "environment": environment,
                "auto_completed": auto_completed,
                "multimodal": {
                    "provider": provider_cfg["provider"],
                    "vlm_model": provider_cfg["vlm_model"],
                    "llm_model": provider_cfg["llm_model"],
                    "api_mode": provider_cfg["api_mode"],
                    "api_key_env": provider_cfg["api_key_env"],
                    "api_key_present": provider_cfg["api_key_present"],
                    "method": str(multimodal_params.get("method") or ("thinking-with-image" if thinking_with_image else "detector-text-reflection")),
                    "thinking_with_image": thinking_with_image,
                    "structured_output": structured_output,
                    "prompt_template": prompt_template,
                    "effective_prompt_template": resolved_prompt_template,
                    "open_world_assist_profile": multimodal_params.get("open_world_assist_profile"),
                    "use_marked_image": use_marked_image,
                    "visual_search_mode": visual_search_mode,
                    "fusion_mode": fusion_mode,
                    "open_world_filters": {
                        "taxonomy_min_score": multimodal_params.get("open_world_taxonomy_min_score"),
                        "taxonomy_require_exact_for_generic": multimodal_params.get("open_world_taxonomy_require_exact_for_generic"),
                        "filter_unmatched_taxonomy": multimodal_params.get("open_world_filter_unmatched_taxonomy"),
                        "filter_generic_labels": multimodal_params.get("open_world_filter_generic_labels"),
                    },
                    "dataset": dataset_info,
                },
            },
        )

    model = deps.build_model(effective_request)
    selected_device = chosen_device
    items: list[dict[str, Any]] = []
    save_dir = None
    method = str(multimodal_params.get("method") or ("thinking-with-image" if thinking_with_image else "detector-text-reflection"))
    llm_enabled = bool(provider_cfg.get("llm_model") or multimodal_params.get("enable_llm_refine"))

    for index, image_path in enumerate(images):
        image_item: dict[str, Any] = {
            "path": str(image_path),
            "index": index,
            "ground_truth": read_ground_truth_summary(image_path, names),
        }
        image_prompt = build_multimodal_evaluation_prompt(
            prompt,
            image_path,
            index,
            len(images),
            image_item["ground_truth"],
            include_ground_truth=include_ground_truth,
        )
        yolo_prediction = None
        try:
            yolo_prediction = model.predict(source=str(image_path), **yolo_params)
        except Exception as exc:
            if device_selection["source"] == "auto" and selected_device not in (None, "cpu") and request.get("runtime", {}).get("allow_device_fallback", True):
                retry_params = replace_cli_device(yolo_params, "cpu")
                effective_request["params"] = retry_params
                try:
                    model = deps.build_model(effective_request)
                    yolo_prediction = model.predict(source=str(image_path), **retry_params)
                    yolo_params = retry_params
                    selected_device = "cpu"
                    auto_completed["device"] = "cpu"
                    auto_completed["device_source"] = "recovery"
                except Exception as retry_exc:
                    if not continue_on_error:
                        raise retry_exc
                    image_item["status"] = "failed"
                    image_item["error"] = {"type": type(retry_exc).__name__, "message": str(retry_exc)}
                    items.append(image_item)
                    continue
            elif continue_on_error:
                image_item["status"] = "failed"
                image_item["error"] = {"type": type(exc).__name__, "message": str(exc)}
                items.append(image_item)
                continue
            else:
                raise

        detections = summarize_results_for_reasoning(yolo_prediction, max_items=1, max_boxes=int(multimodal_params.get("max_reasoning_boxes", 20)))
        detection_summary = detections[0] if detections else {"path": str(image_path), "speed": {}, "detections": []}
        image_ref = visual_image_source_for_openai(str(image_path), detections, resolved_path=resolved_path)
        image_meta: dict[str, Any] = {
            "requested": image_ref,
            "thinking_with_image": thinking_with_image,
            "attached": False,
        }
        item_visual_artifacts: list[dict[str, Any]] = []
        visual_search_passes: list[dict[str, Any]] = []
        marked_meta: dict[str, Any] | None = None
        marked_error: dict[str, Any] | None = None
        vlm_result: dict[str, Any]
        llm_result: dict[str, Any] | None = None
        if thinking_with_image and image_ref is None:
            vlm_result = {
                "status": "blocked",
                "provider": provider_cfg.get("provider", "openai"),
                "summary": "No image source was available for multimodal reasoning.",
            }
        elif not provider_cfg["api_key_present"]:
            vlm_result = {
                "status": "blocked",
                "provider": provider_cfg.get("provider", "openai"),
                "summary": f"{provider_cfg['api_key_env']} is not set; multimodal reasoning was skipped.",
                "api_key_env": provider_cfg["api_key_env"],
            }
        else:
            try:
                encoded_image = None
                if thinking_with_image and image_ref is not None:
                    reasoning_image_ref = image_ref
                    if use_marked_image and not str(image_ref).startswith(("http://", "https://", "data:image/")):
                        try:
                            marked = visual_render_marked_image(
                                image_ref,
                                detections,
                                resolved_path=resolved_path,
                                output_dir=ensure_manifest_dir(request) / "visual-search",
                                prefix=f"{index:04d}-{slugify(Path(str(image_ref)).stem)}",
                            )
                            reasoning_image_ref = marked["path"]
                            marked_meta = marked
                            item_visual_artifacts.append({"kind": "marked_image", "path": marked["path"]})
                        except Exception as mark_exc:
                            marked_error = {"type": type(mark_exc).__name__, "message": str(mark_exc)}
                    encoded_image = encode_image_reference_for_openai_raw(
                        reasoning_image_ref,
                        resolved_path=resolved_path,
                        max_bytes=int(multimodal_params.get("max_image_bytes", 20_000_000)),
                    )
                    image_meta = {k: v for k, v in encoded_image.items() if k != "image_url"}
                    image_meta["requested"] = image_ref
                    image_meta["reasoning_input"] = reasoning_image_ref
                    if marked_meta:
                        image_meta["marked"] = marked_meta
                    if marked_error:
                        image_meta["marked_error"] = marked_error
                    image_meta["thinking_with_image"] = True
                    image_meta["attached"] = True
                user_text = build_thinking_with_image_prompt(
                    image_prompt,
                    detections,
                    method=method,
                    thinking_with_image=thinking_with_image,
                    structured_output=structured_output,
                    prompt_template=str(resolved_prompt_template) if resolved_prompt_template is not None else None,
                    prompt_dir=PROMPT_DIR,
                )
                vlm_result = deps.call_openai_compatible(
                    model=str(provider_cfg["vlm_model"]),
                    user_text=user_text,
                    developer_text=str(
                        multimodal_params.get("developer_prompt")
                        or multimodal_params.get("system_prompt")
                        or default_vlm_developer_prompt(resolved_prompt_template)
                    ),
                    image_url=encoded_image["image_url"] if encoded_image else None,
                    image_detail=str(multimodal_params.get("image_detail", "auto")),
                    base_url=provider_cfg["base_url"],
                    provider=str(provider_cfg.get("provider", "openai")),
                    api_key_env=str(provider_cfg.get("api_key_env", "OPENAI_API_KEY")),
                    api_mode=str(provider_cfg["api_mode"]),
                    max_output_tokens=max_output_tokens,
                    temperature=float(multimodal_params["temperature"]) if "temperature" in multimodal_params else None,
                )
                vlm_result = attach_multimodal_verdict(vlm_result)
                if vlm_result.get("status") == "ok" and image_ref is not None:
                    visual_search_passes, search_artifacts = runtime_run_visual_search_crop_passes(
                        image_path=image_ref,
                        base_prompt=image_prompt,
                        detections=detections,
                        initial_verdict=vlm_result.get("verdict", {}),
                        provider_cfg=provider_cfg,
                        multimodal_params=multimodal_params,
                        output_dir=ensure_manifest_dir(request),
                        max_output_tokens=max_output_tokens,
                        method=method,
                        resolved_path_fn=resolved_path,
                        normalize_detection_boxes_fn=normalize_detection_boxes,
                        call_openai_compatible_fn=deps.call_openai_compatible,
                        attach_multimodal_verdict_fn=attach_multimodal_verdict,
                    )
                    item_visual_artifacts.extend(search_artifacts)
            except Exception as exc:
                vlm_result = {
                    "status": "failed",
                    "provider": provider_cfg.get("provider", "openai"),
                    "summary": "Failed to prepare or call VLM reasoning.",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }

        if llm_enabled and vlm_result.get("status") == "ok":
            llm_model = str(provider_cfg.get("llm_model") or provider_cfg["vlm_model"])
            refine_prompt = (
                "Refine this YOLO + VLM evaluation into a compact final answer. Do not add unsupported visual claims. "
                "Return exactly one JSON object without Markdown fences. Use these keys: answer, visual_evidence, "
                "yolo_cross_check, uncertainty, recommended_next_actions. In yolo_cross_check, include arrays named "
                "confirmed, false_positives, possible_misses, duplicate_or_fragmented, and notes when applicable. "
                "If the VLM answer includes caption, global_classification, vlm_detections, vlm_segmentation, "
                "visual_search, or fusion_hints, preserve those keys and refine them conservatively.\n\n"
                f"User task:\n{image_prompt}\n\n"
                f"YOLO detection summary:\n{json.dumps(json_safe(detections), ensure_ascii=False, indent=2)}\n\n"
                f"VLM answer:\n{vlm_result.get('text', '')}\n\n"
                f"Parsed VLM verdict:\n{json.dumps(json_safe(vlm_result.get('verdict', {})), ensure_ascii=False, indent=2)}"
                f"\n\nVisual search crop passes:\n{json.dumps(json_safe(visual_search_passes), ensure_ascii=False, indent=2)}"
            )
            llm_result = deps.call_openai_compatible(
                model=llm_model,
                user_text=refine_prompt,
                developer_text=default_llm_refine_developer_prompt(resolved_prompt_template),
                base_url=provider_cfg["base_url"],
                provider=str(provider_cfg.get("provider", "openai")),
                api_key_env=str(provider_cfg.get("api_key_env", "OPENAI_API_KEY")),
                api_mode=str(provider_cfg["api_mode"]),
                max_output_tokens=max_output_tokens,
                temperature=float(multimodal_params["temperature"]) if "temperature" in multimodal_params else None,
            )
            llm_result = attach_multimodal_verdict(llm_result)

        vlm_verdict = vlm_result.get("verdict", {}) if isinstance(vlm_result.get("verdict"), dict) else {}
        llm_verdict = llm_result.get("verdict", {}) if isinstance(llm_result, dict) and isinstance(llm_result.get("verdict"), dict) else {}
        merged_verdict = merge_verdicts(vlm_verdict, llm_verdict)
        fusion_preview = fusion_build_multimodal_fusion_preview(
            detections=detections,
            verdict=merged_verdict,
            multimodal_params=multimodal_params,
            image_path=image_ref,
            normalize_detection_boxes_fn=normalize_detection_boxes,
        )
        metric_preview = build_item_metric_preview(
            image_path=image_path,
            names=names,
            detections=detections,
            fusion_preview=fusion_preview,
            verdict=merged_verdict,
        )
        open_world_comparison = build_open_world_comparison_entry(
            image_path=image_path,
            detections=detections,
            fusion_preview=fusion_preview,
            verdict=merged_verdict,
            multimodal_params=multimodal_params,
            effective_prompt_template=str(resolved_prompt_template) if resolved_prompt_template is not None else None,
        )

        status = multimodal_overall_status(vlm_result, llm_result)
        image_item.update(
            {
                "status": status,
                "detector": {
                    "boxes": len(detection_summary.get("detections", []) or []),
                    "summary": detection_summary,
                    "label_counts": detection_label_counts(detections),
                },
                "prompt": image_prompt,
                "multimodal": {
                    "method": method,
                    "thinking_with_image": thinking_with_image,
                    "structured_output": structured_output,
                    "prompt_template": prompt_template,
                    "effective_prompt_template": resolved_prompt_template,
                    "open_world_assist_profile": multimodal_params.get("open_world_assist_profile"),
                    "open_world_filters": {
                        "taxonomy_min_score": multimodal_params.get("open_world_taxonomy_min_score"),
                        "taxonomy_require_exact_for_generic": multimodal_params.get("open_world_taxonomy_require_exact_for_generic"),
                        "filter_unmatched_taxonomy": multimodal_params.get("open_world_filter_unmatched_taxonomy"),
                        "filter_generic_labels": multimodal_params.get("open_world_filter_generic_labels"),
                    },
                    "use_marked_image": use_marked_image,
                    "visual_search_mode": visual_search_mode,
                    "fusion_mode": fusion_mode,
                    "provider": {
                        "name": provider_cfg["provider"],
                        "base_url": provider_cfg["base_url"],
                        "api_mode": provider_cfg["api_mode"],
                        "api_key_env": provider_cfg["api_key_env"],
                        "api_key_present": provider_cfg["api_key_present"],
                    },
                    "image": image_meta,
                    "visual_search": {"mode": visual_search_mode, "passes": visual_search_passes, "artifacts": item_visual_artifacts},
                    "prompt": image_prompt,
                    "vlm": vlm_result,
                    "llm_refine": llm_result or {"status": "skipped"},
                    "fusion": fusion_preview,
                    "open_world_comparison": open_world_comparison,
                },
                "metric_preview": metric_preview,
            }
        )
        if item_visual_artifacts:
            image_item["artifacts"] = item_visual_artifacts
        if status in {"blocked", "partial", "failed"}:
            image_item["notes"] = ["See multimodal cross-check and detection summary for details."]
        items.append(image_item)
        save_dir = getattr(getattr(model, "predictor", None), "save_dir", save_dir)

    aggregate = aggregate_multimodal_evaluation(items)
    metric_preview = aggregate_metric_preview(items, names)
    aggregate["metric_preview"] = metric_preview
    overall_status = overall_multimodal_evaluation_status(aggregate)
    if run_yolo_val and data_ref_for_baseline not in (None, ""):
        baseline_request = normalize_request(
            {
                "skill": "yolo.val",
                "runtime": request.get("runtime", {}),
                "inputs": {"model": request["inputs"]["model"], "data": data_ref_for_baseline},
                "params": {k: v for k, v in yolo_params.items() if k not in {"source"}},
                "artifacts": request.get("artifacts", {}),
                "policy": request.get("policy", {}),
                "request_id": f"{request.get('request_id', default_request_id('yolo.multimodal.evaluate'))}-baseline",
            }
        )
        baseline = deps.run_val(baseline_request)
    else:
        baseline = {"status": "skipped"}

    final_selection_source = "recovery" if selected_device != chosen_device and selected_device == "cpu" else device_selection["source"]
    environment = collect_environment_report(
        effective_request,
        selected_device=selected_device,
        requested_device=chosen_device,
        selection_source=final_selection_source,
    )

    report_dir = ensure_manifest_dir(request)
    report_path = report_dir / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "skill": request.get("skill"),
        "request_id": request.get("request_id"),
        "dataset": dataset_info,
        "aggregate": aggregate,
        "metric_preview": metric_preview,
        "baseline": baseline,
        "items": items,
        "environment": environment,
        "auto_completed": auto_completed,
        "multimodal": {
            "method": method,
            "thinking_with_image": thinking_with_image,
            "structured_output": structured_output,
            "prompt_template": prompt_template,
            "effective_prompt_template": resolved_prompt_template,
            "open_world_assist_profile": multimodal_params.get("open_world_assist_profile"),
            "open_world_filters": {
                "taxonomy_min_score": multimodal_params.get("open_world_taxonomy_min_score"),
                "taxonomy_require_exact_for_generic": multimodal_params.get("open_world_taxonomy_require_exact_for_generic"),
                "filter_unmatched_taxonomy": multimodal_params.get("open_world_filter_unmatched_taxonomy"),
                "filter_generic_labels": multimodal_params.get("open_world_filter_generic_labels"),
            },
            "use_marked_image": use_marked_image,
            "visual_search_mode": visual_search_mode,
            "fusion_mode": fusion_mode,
            "provider": {
                "name": provider_cfg["provider"],
                "base_url": provider_cfg["base_url"],
                "api_mode": provider_cfg["api_mode"],
                "api_key_env": provider_cfg["api_key_env"],
                "api_key_present": provider_cfg["api_key_present"],
            },
            "dataset": dataset_info,
            "prompt": prompt,
        },
    }
    open_world_report = {
        "items": [
            item.get("multimodal", {}).get("open_world_comparison", {})
            for item in items
            if isinstance(item.get("multimodal", {}).get("open_world_comparison", {}), dict)
        ]
    }
    open_world_report["aggregate"] = aggregate_open_world_comparison(open_world_report["items"])
    report["open_world_comparison_report"] = open_world_report
    fusion_coco_records = [
        record
        for item in items
        for record in (
            item.get("multimodal", {})
            .get("fusion", {})
            .get("coco_predictions_preview", [])
            if isinstance(item.get("multimodal", {}).get("fusion", {}), dict)
            else []
        )
    ]
    metric_guardrail = build_metric_guardrail(
        items=items,
        metric_preview=metric_preview,
        fused_coco_records=fusion_coco_records,
        multimodal_params=multimodal_params,
    )
    if fusion_coco_records:
        report["fusion_preview"] = {
            "strategy": "metric_safe_v1",
            "coco_prediction_records": len(fusion_coco_records),
            "note": "Preview records are VLM-assisted candidates; run COCO evaluation before treating them as metric gains.",
        }
    report["metric_guardrail"] = {k: v for k, v in metric_guardrail.items() if k != "records"}
    report_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts = [{"kind": "json", "path": str(report_path.resolve())}]
    open_world_report_path = report_dir / "open-world-comparison-report.json"
    open_world_report_path.write_text(json.dumps(json_safe(open_world_report), ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts.append({"kind": "open_world_comparison_report", "path": str(open_world_report_path.resolve())})
    if fusion_coco_records:
        fusion_path = report_dir / "fusion-preview-coco-predictions.json"
        fusion_path.write_text(json.dumps(json_safe(fusion_coco_records), ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts.append({"kind": "fusion_coco_predictions_preview", "path": str(fusion_path.resolve())})
    if metric_preview.get("status") == "ok":
        metric_path = report_dir / "fusion-metric-preview.json"
        metric_path.write_text(json.dumps(json_safe(metric_preview), ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts.append({"kind": "fusion_metric_preview", "path": str(metric_path.resolve())})
    if metric_guardrail.get("records"):
        guarded_path = report_dir / "metric-guarded-coco-predictions.json"
        guarded_path.write_text(json.dumps(json_safe(metric_guardrail["records"]), ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts.append({"kind": "metric_guarded_coco_predictions", "path": str(guarded_path.resolve())})
    if save_dir:
        artifacts.append({"kind": "directory", "path": str(Path(save_dir).resolve())})

    payload = response(
        request["skill"],
        overall_status,
        f"multimodal evaluation finished on {len(items)} images",
        job={"mode": "sync", "save_dir": json_safe(save_dir), "executor": "python_api+openai", "device": selected_device},
        results=items,
        evaluation=aggregate,
        environment=environment,
        auto_completed=auto_completed,
        artifacts=artifacts,
        multimodal=report["multimodal"],
        metric_guardrail={k: v for k, v in metric_guardrail.items() if k != "records"},
        baseline=baseline,
        next_actions=["yolo.val", "yolo.multimodal.infer", "yolo.predict"],
    )
    payload["manifest"] = str(write_manifest(request, payload))
    return payload
