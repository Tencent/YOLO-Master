from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .visual import clamp_box_xyxy, encode_image_reference_for_openai, load_pillow_image

PROVIDER_CONFIG_DIR = Path(__file__).resolve().parent / "providers"


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "results_dict"):
        return json_safe(value.results_dict)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def prompt_template_name(prompt_template: Any) -> str:
    if prompt_template in (None, ""):
        return ""
    value = str(prompt_template)
    return Path(value).stem if value.endswith(".md") else Path(value).name


def load_provider_config(provider: str) -> dict[str, Any]:
    path = PROVIDER_CONFIG_DIR / f"{provider.lower()}.yaml"
    if not path.exists():
        return {}
    try:
        import yaml
    except Exception:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _env_present(primary: str) -> bool:
    return bool(os.environ.get(primary) or (primary != "OPENAI_API_KEY" and os.environ.get("OPENAI_API_KEY")))


def openai_config(params: dict[str, Any]) -> dict[str, Any]:
    provider = params.get("vlm_provider") or params.get("provider") or "openai"
    provider_cfg = load_provider_config(str(provider))
    defaults = provider_cfg.get("defaults") if isinstance(provider_cfg.get("defaults"), dict) else {}
    api_key_env = str(params.get("openai_api_key_env") or provider_cfg.get("api_key_env") or "OPENAI_API_KEY")
    base_url = str(
        params.get("openai_base_url")
        or os.environ.get("OPENAI_BASE_URL")
        or provider_cfg.get("base_url")
        or "https://api.openai.com/v1"
    ).rstrip("/")
    return {
        "provider": provider,
        "api_family": provider_cfg.get("api_family", "openai-compatible"),
        "api_key_env": api_key_env,
        "api_key_present": _env_present(api_key_env),
        "base_url": base_url,
        "api_mode": params.get("openai_api_mode") or os.environ.get("OPENAI_API_MODE") or defaults.get("api_mode") or "auto",
        "vlm_model": params.get("vlm_model") or os.environ.get("OPENAI_VLM_MODEL") or os.environ.get("OPENAI_MODEL") or defaults.get("vlm_model") or "gpt-4.1-mini",
        "llm_model": params.get("llm_model") or os.environ.get("OPENAI_LLM_MODEL") or os.environ.get("OPENAI_MODEL") or defaults.get("llm_model"),
    }


def load_prompt_template(prompt_template: str | None, *, prompt_dir: Path) -> str | None:
    if not prompt_template:
        return None
    candidate = Path(str(prompt_template))
    if candidate.exists() and candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    template_path = prompt_dir / (candidate.name if candidate.suffix else f"{candidate.name}.md")
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return None


def render_prompt_template(template: str, context: dict[str, str]) -> str:
    rendered = template
    for key, value in context.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def default_vlm_developer_prompt(prompt_template: Any = None) -> str:
    if prompt_template:
        if prompt_template_name(prompt_template) == "vlm_open_world_detection":
            return (
                "You are a careful open-world multimodal perception assistant for YOLO-Master. "
                "Return exactly one JSON object that follows the user's schema and includes every top-level key. "
                "Use the image and detector evidence, preserve novel categories when clearly visible, and do not expose hidden chain-of-thought."
            )
        return (
            "You are a careful COCO-oriented visual reasoning assistant for YOLO-Master. "
            "Return exactly one JSON object that follows the user's schema and includes every top-level key. "
            "Use the image and detector evidence, but do not expose hidden chain-of-thought."
        )
    return (
        "You are a careful visual reasoning assistant for YOLO-Master. Use the image and detector evidence, "
        "but only return concise evidence and uncertainty, not hidden chain-of-thought."
    )


def default_llm_refine_developer_prompt(prompt_template: Any = None) -> str:
    if prompt_template:
        if prompt_template_name(prompt_template) == "vlm_open_world_detection":
            return (
                "You are a concise verifier. Return exactly one JSON object. Preserve the open-world schema keys "
                "when present, including caption, global_classification, vlm_detections, vlm_segmentation, visual_search, and fusion_hints."
            )
        return (
            "You are a concise verifier. Return exactly one JSON object. Preserve the COCO multitask schema keys "
            "when present, including caption, global_classification, vlm_detections, vlm_segmentation, visual_search, and fusion_hints."
        )
    return "You are a concise verifier. Return answer, evidence, uncertainty, and next actions."


def build_thinking_with_image_prompt(
    user_prompt: str,
    detections: list[dict[str, Any]],
    *,
    method: str = "thinking-with-image",
    thinking_with_image: bool = True,
    structured_output: bool = True,
    prompt_template: str | None = None,
    prompt_dir: Path,
) -> str:
    detection_text = json.dumps(json_safe(detections), ensure_ascii=False, indent=2)
    image_instruction = (
        "Privately inspect the image, compare it with the YOLO detection summary, and resolve disagreements."
        if thinking_with_image
        else "Use the user prompt and YOLO detection summary as the evidence surface, and do not assume image access."
    )
    output_instruction = (
        "Return exactly one JSON object without Markdown fences. Use these keys: answer, visual_evidence, "
        "yolo_cross_check, uncertainty, recommended_next_actions. In yolo_cross_check, include arrays named "
        "confirmed, false_positives, possible_misses, duplicate_or_fragmented, and notes when applicable."
        if structured_output
        else "Return these sections: answer, visual_evidence, yolo_cross_check, uncertainty, recommended_next_actions."
    )
    template = load_prompt_template(prompt_template, prompt_dir=prompt_dir)
    if template:
        template_output_instruction = (
            "Return the complete schema above as one JSON object. Include every top-level key even when a list is empty, "
            "especially caption, global_classification, vlm_detections, vlm_segmentation, visual_search, yolo_cross_check, fusion_hints, uncertainty, and recommended_next_actions."
        )
        if prompt_template_name(prompt_template) == "vlm_open_world_detection":
            template_output_instruction += " Keep open-world labels when visible, and add COCO mapping fields only when grounded."
        return render_prompt_template(
            template,
            {
                "METHOD": method,
                "IMAGE_INSTRUCTION": image_instruction,
                "OUTPUT_INSTRUCTION": template_output_instruction,
                "USER_PROMPT": user_prompt,
                "DETECTION_SUMMARY": detection_text,
            },
        )
    return (
        "You are helping a YOLO-Master agent perform multimodal visual inference.\n"
        f"Method: {method}. {image_instruction} Do not reveal hidden chain-of-thought. "
        "Return a concise, evidence-based answer.\n\n"
        f"User task:\n{user_prompt}\n\n"
        f"YOLO detection summary:\n{detection_text}\n\n{output_instruction}"
    )


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    def balanced_fragment(source: str) -> str | None:
        start = None
        depth = 0
        in_string = False
        escape = False
        opener = None
        for idx, ch in enumerate(source):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in "{[":
                if start is None:
                    start = idx
                    opener = ch
                    depth = 1
                    continue
                depth += 1
            elif ch in "}]":
                if start is None:
                    continue
                depth -= 1
                if depth == 0 and opener is not None:
                    return source[start : idx + 1]
        return None

    fragment = balanced_fragment(cleaned)
    if fragment is not None:
        try:
            value = json.loads(fragment)
            return value if isinstance(value, dict) else None
        except Exception:
            pass
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[{]", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start() :])
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return None


def attach_multimodal_verdict(result: dict[str, Any]) -> dict[str, Any]:
    result = dict(result)
    if result.get("status") == "ok" and "verdict" not in result:
        verdict = extract_json_object(str(result.get("text") or ""))
        expected_keys = {"answer", "visual_evidence", "yolo_cross_check", "uncertainty", "recommended_next_actions"}
        if verdict is not None and expected_keys.intersection(verdict):
            result["verdict"] = json_safe(verdict)
            result["verdict_parse_status"] = "parsed"
        else:
            result["verdict_parse_status"] = "unparsed"
    return result


def extract_openai_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    chunks: list[str] = []
    for output in payload.get("output", []) or []:
        for content in output.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks).strip()


def extract_openai_chat_text(payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    for choice in payload.get("choices", []) or []:
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
    return "\n".join(chunks).strip()


def classify_openai_http_status(detail: str) -> str:
    text = detail.lower()
    blocked_markers = ("access denied", "arrearage", "insufficient_quota", "quota", "billing", "permission", "unauthorized", "forbidden")
    return "blocked" if any(marker in text for marker in blocked_markers) else "failed"


def call_openai_responses(
    *,
    model: str,
    user_text: str,
    developer_text: str | None = None,
    image_url: str | None = None,
    image_detail: str = "auto",
    base_url: str | None = None,
    provider: str = "openai",
    api_key_env: str = "OPENAI_API_KEY",
    max_output_tokens: int = 800,
    temperature: float | None = None,
) -> dict[str, Any]:
    api_key = os.environ.get(api_key_env) or (os.environ.get("OPENAI_API_KEY") if api_key_env != "OPENAI_API_KEY" else None)
    if not api_key:
        return {"status": "blocked", "provider": provider, "summary": f"{api_key_env} is not set; multimodal reasoning was skipped.", "api_key_env": api_key_env}
    base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    content: list[dict[str, Any]] = [{"type": "input_text", "text": user_text}]
    if image_url:
        content.append({"type": "input_image", "image_url": image_url, "detail": image_detail})
    input_items = []
    if developer_text:
        input_items.append({"role": "developer", "content": [{"type": "input_text", "text": developer_text}]})
    input_items.append({"role": "user", "content": content})
    body: dict[str, Any] = {"model": model, "input": input_items, "max_output_tokens": max_output_tokens}
    if temperature is not None:
        body["temperature"] = temperature
    request = urllib.request.Request(
        f"{base_url}/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response_handle:
            payload = json.loads(response_handle.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {
            "status": classify_openai_http_status(detail),
            "provider": provider,
            "summary": f"OpenAI Responses API returned HTTP {exc.code}",
            "error": {"type": "HTTPError", "code": exc.code, "body": detail},
        }
    except Exception as exc:
        return {"status": "failed", "provider": provider, "summary": "OpenAI Responses API request failed", "error": {"type": type(exc).__name__, "message": str(exc)}}
    return {
        "status": "ok",
        "provider": provider,
        "api_mode": "responses",
        "model": model,
        "text": extract_openai_text(payload),
        "response_id": payload.get("id"),
        "usage": json_safe(payload.get("usage", {})),
    }


def call_openai_chat_completions(
    *,
    model: str,
    user_text: str,
    developer_text: str | None = None,
    image_url: str | None = None,
    base_url: str | None = None,
    provider: str = "openai",
    api_key_env: str = "OPENAI_API_KEY",
    max_output_tokens: int = 800,
    temperature: float | None = None,
) -> dict[str, Any]:
    api_key = os.environ.get(api_key_env) or (os.environ.get("OPENAI_API_KEY") if api_key_env != "OPENAI_API_KEY" else None)
    if not api_key:
        return {
            "status": "blocked",
            "provider": provider,
            "api_mode": "chat.completions",
            "summary": f"{api_key_env} is not set; multimodal reasoning was skipped.",
            "api_key_env": api_key_env,
        }
    base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    messages = []
    if developer_text:
        messages.append({"role": "system", "content": developer_text})
    messages.append({"role": "user", "content": content})
    body: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_output_tokens}
    if temperature is not None:
        body["temperature"] = temperature
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response_handle:
            payload = json.loads(response_handle.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {
            "status": classify_openai_http_status(detail),
            "provider": provider,
            "api_mode": "chat.completions",
            "summary": f"OpenAI-compatible Chat Completions API returned HTTP {exc.code}",
            "error": {"type": "HTTPError", "code": exc.code, "body": detail},
        }
    except Exception as exc:
        return {"status": "failed", "provider": provider, "api_mode": "chat.completions", "summary": "OpenAI-compatible Chat Completions API request failed", "error": {"type": type(exc).__name__, "message": str(exc)}}
    return {
        "status": "ok",
        "provider": provider,
        "api_mode": "chat.completions",
        "model": model,
        "text": extract_openai_chat_text(payload),
        "response_id": payload.get("id"),
        "usage": json_safe(payload.get("usage", {})),
    }


def call_openai_compatible(
    *,
    model: str,
    user_text: str,
    developer_text: str | None = None,
    image_url: str | None = None,
    image_detail: str = "auto",
    base_url: str | None = None,
    provider: str = "openai",
    api_key_env: str = "OPENAI_API_KEY",
    api_mode: str = "auto",
    max_output_tokens: int = 800,
    temperature: float | None = None,
) -> dict[str, Any]:
    normalized_mode = api_mode.replace("_", ".").lower()
    if normalized_mode in {"chat", "chat.completion", "chat.completions"}:
        return call_openai_chat_completions(
            model=model,
            user_text=user_text,
            developer_text=developer_text,
            image_url=image_url,
            base_url=base_url,
            provider=provider,
            api_key_env=api_key_env,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    if normalized_mode == "responses":
        return call_openai_responses(
            model=model,
            user_text=user_text,
            developer_text=developer_text,
            image_url=image_url,
            image_detail=image_detail,
            base_url=base_url,
            provider=provider,
            api_key_env=api_key_env,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    responses_result = call_openai_responses(
        model=model,
        user_text=user_text,
        developer_text=developer_text,
        image_url=image_url,
        image_detail=image_detail,
        base_url=base_url,
        provider=provider,
        api_key_env=api_key_env,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    if responses_result.get("status") in {"ok", "blocked"}:
        return responses_result
    chat_result = call_openai_chat_completions(
        model=model,
        user_text=user_text,
        developer_text=developer_text,
        image_url=image_url,
        base_url=base_url,
        provider=provider,
        api_key_env=api_key_env,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    if chat_result.get("status") == "ok":
        chat_result["fallback_from"] = responses_result
    return chat_result


def multimodal_overall_status(vlm_result: dict[str, Any], llm_result: dict[str, Any] | None) -> str:
    vlm_status = vlm_result.get("status")
    if vlm_status == "blocked" or (llm_result and llm_result.get("status") == "blocked"):
        return "blocked"
    if vlm_status != "ok":
        return "partial"
    if llm_result and llm_result.get("status") == "failed":
        return "partial"
    return "ok"


def build_visual_search_crop_prompt(base_prompt: str, region: dict[str, Any], detections: list[dict[str, Any]]) -> str:
    return (
        "You are inspecting a zoomed crop extracted from the original image.\n"
        "Focus on local object presence, object boundaries, and whether the YOLO box should be kept, adjusted, or suppressed.\n"
        "Return exactly one JSON object without markdown fences. Preserve any useful keys from the schema, especially answer, visual_evidence, "
        "yolo_cross_check, uncertainty, recommended_next_actions, visual_search, and fusion_hints.\n\n"
        f"Crop region:\n{json.dumps(json_safe(region), ensure_ascii=False, indent=2)}\n\n"
        f"Base task:\n{base_prompt}\n\n"
        f"YOLO detection summary:\n{json.dumps(json_safe(detections), ensure_ascii=False, indent=2)}\n"
    )


def default_visual_search_regions(
    detections: list[dict[str, Any]],
    *,
    normalize_detection_boxes_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    max_regions: int = 2,
) -> list[dict[str, Any]]:
    normalized = normalize_detection_boxes_fn(detections)
    ranked = sorted(
        normalized,
        key=lambda item: (
            float(item.get("confidence") or 0.0),
            (item["bbox_xyxy"][2] - item["bbox_xyxy"][0]) * (item["bbox_xyxy"][3] - item["bbox_xyxy"][1]),
        ),
    )
    regions: list[dict[str, Any]] = []
    for item in ranked[: max(0, max_regions)]:
        regions.append(
            {
                "region_id": f"auto-{item['index']}",
                "bbox_xyxy": item["bbox_xyxy"],
                "reason": f"low_confidence_{item.get('label')}",
                "priority": "high" if float(item.get("confidence") or 0.0) < 0.35 else "medium",
                "linked_yolo_indices": [item["index"]],
            }
        )
    return regions


def extract_visual_search_regions(verdict: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(verdict, dict):
        return []
    visual_search = verdict.get("visual_search", {})
    if not isinstance(visual_search, dict):
        return []
    regions = visual_search.get("search_regions", [])
    if not isinstance(regions, list):
        return []
    normalized: list[dict[str, Any]] = []
    for idx, region in enumerate(regions):
        if not isinstance(region, dict):
            continue
        bbox = region.get("bbox_xyxy") or region.get("bbox") or region.get("xyxy")
        if not isinstance(bbox, (list, tuple)):
            continue
        normalized.append(
            {
                "region_id": str(region.get("region_id") or f"r{idx + 1}"),
                "bbox_xyxy": [float(v) for v in bbox[:4]],
                "purpose": str(region.get("purpose") or region.get("reason") or "inspect uncertain region"),
                "priority": str(region.get("priority") or "medium"),
                "linked_yolo_indices": [int(v) for v in region.get("linked_yolo_indices", []) if isinstance(v, (int, float, str)) and str(v).isdigit()],
                "raw": region,
            }
        )
    return normalized


def run_visual_search_crop_passes(
    *,
    image_path: str | Path,
    base_prompt: str,
    detections: list[dict[str, Any]],
    initial_verdict: dict[str, Any],
    provider_cfg: dict[str, Any],
    multimodal_params: dict[str, Any],
    output_dir: Path,
    max_output_tokens: int,
    method: str,
    resolved_path_fn: Callable[[str | Path], Path],
    normalize_detection_boxes_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    call_openai_compatible_fn: Callable[..., dict[str, Any]],
    attach_multimodal_verdict_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mode = str(multimodal_params.get("visual_search_mode") or "auto").lower()
    if mode in {"off", "none", "false", "0"}:
        return [], []
    if mode in {"auto", "smart"}:
        visual_search = initial_verdict.get("visual_search", {}) if isinstance(initial_verdict, dict) else {}
        if not (isinstance(visual_search, dict) and parse_bool(visual_search.get("needs_zoom"), False)):
            return [], []
    regions = extract_visual_search_regions(initial_verdict)
    if not regions and mode in {"always", "low-confidence", "low_confidence"}:
        regions = default_visual_search_regions(
            detections,
            normalize_detection_boxes_fn=normalize_detection_boxes_fn,
            max_regions=int(multimodal_params.get("visual_search_max_regions", 2)),
        )
    if not regions:
        return [], []
    source_path = resolved_path_fn(image_path)
    image = load_pillow_image(source_path).convert("RGB")
    width, height = image.size
    crop_margin = float(multimodal_params.get("visual_search_crop_margin", 0.18))
    max_regions = max(0, int(multimodal_params.get("visual_search_max_regions", 2)))
    region_results: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    crop_dir = output_dir / "visual-search"
    crop_dir.mkdir(parents=True, exist_ok=True)
    for idx, region in enumerate(regions[:max_regions]):
        bbox = clamp_box_xyxy(region.get("bbox_xyxy", []), width, height, margin=crop_margin)
        if bbox is None:
            continue
        crop = image.crop(tuple(bbox))
        crop_path = crop_dir / f"{Path(source_path).stem}-region-{idx + 1}.jpg"
        crop.save(crop_path, quality=92)
        artifacts.append({"kind": "crop_image", "path": str(crop_path.resolve()), "region_id": region.get("region_id")})
        crop_prompt = build_visual_search_crop_prompt(base_prompt, region, detections)
        crop_result = call_openai_compatible_fn(
            model=str(provider_cfg["vlm_model"]),
            user_text=crop_prompt,
            developer_text=str(
                multimodal_params.get("developer_prompt")
                or multimodal_params.get("system_prompt")
                or "You are a careful visual search assistant. Focus on the crop and return concise structured JSON."
            ),
            image_url=encode_image_reference_for_openai(str(crop_path), resolved_path=resolved_path_fn, max_bytes=int(multimodal_params.get("max_image_bytes", 20_000_000)))["image_url"],
            image_detail=str(multimodal_params.get("image_detail", "auto")),
            base_url=provider_cfg["base_url"],
            provider=str(provider_cfg.get("provider", "openai")),
            api_key_env=str(provider_cfg.get("api_key_env", "OPENAI_API_KEY")),
            api_mode=str(provider_cfg["api_mode"]),
            max_output_tokens=max_output_tokens,
            temperature=float(multimodal_params["temperature"]) if "temperature" in multimodal_params else None,
        )
        crop_result = attach_multimodal_verdict_fn(crop_result)
        region_results.append({"region": {**region, "bbox_xyxy": bbox}, "crop": {"path": str(crop_path.resolve()), "bbox_xyxy": bbox}, "vlm": crop_result})
    return region_results, artifacts
