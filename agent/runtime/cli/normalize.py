from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_CFG_DIR = REPO_ROOT / "ultralytics" / "cfg" / "datasets"

MULTIMODAL_PARAM_KEYS = {
    "prompt",
    "question",
    "system_prompt",
    "developer_prompt",
    "thinking_with_image",
    "method",
    "prompt_template",
    "compact_open_world_profile",
    "open_world_profile",
    "open_world_assist_profile",
    "open_world_label_normalizer",
    "open_world_label_aliases",
    "open_world_label_aliases_path",
    "open_world_taxonomy_datasets",
    "open_world_taxonomy_topk",
    "open_world_taxonomy_min_score",
    "open_world_taxonomy_require_exact_for_generic",
    "open_world_taxonomy_hypernym_fallback",
    "open_world_filter_unmatched_taxonomy",
    "open_world_filter_generic_labels",
    "open_world_iou_relabel_enabled",
    "open_world_iou_relabel_threshold",
    "open_world_iou_relabel_max_yolo_confidence",
    "provider",
    "vlm_provider",
    "vlm_model",
    "llm_model",
    "openai_base_url",
    "openai_api_mode",
    "image_detail",
    "max_output_tokens",
    "temperature",
    "max_reasoning_items",
    "max_reasoning_boxes",
    "max_image_bytes",
    "structured_output",
    "enable_llm_refine",
    "skip_yolo",
    "detections",
    "use_marked_image",
    "visual_search_mode",
    "visual_search_max_regions",
    "visual_search_crop_margin",
    "visual_search_prompt",
    "fusion_mode",
    "fusion_policy",
    "fusion_enabled",
    "fusion_open_world_confidence_min",
    "fusion_add_confidence_min",
    "fusion_add_require_unlinked",
    "fusion_add_max_linked_yolo_confidence",
    "fusion_add_allowed_bbox_quality",
    "fusion_suppress_confidence_min",
    "fusion_adjust_confidence_min",
    "fusion_suppress_max_yolo_confidence",
    "fusion_relabel_max_yolo_confidence",
    "fusion_adjust_min_iou",
    "fusion_metric_guardrail",
    "fusion_guardrail_min_map50_95_delta",
    "fusion_guardrail_require_recall_nonnegative",
}
MULTIMODAL_EVALUATE_PARAM_KEYS = {
    "data",
    "split",
    "limit",
    "max_images",
    "offset",
    "stride",
    "shuffle",
    "seed",
    "include_ground_truth",
    "include_ground_truth_in_prompt",
    "run_yolo_val",
    "continue_on_error",
    "report_name",
}

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
def path_like(value: str) -> bool:
    return any(token in value for token in ("/", "\\", ".")) and not value.startswith(("http://", "https://"))
def normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: normalize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_value(v) for v in value]
    if isinstance(value, str) and path_like(value):
        p = Path(value)
        if not p.is_absolute():
            candidate = (REPO_ROOT / p).resolve()
            if candidate.exists() or value.startswith((".", "/")) or "/" in value:
                return str(candidate)
    if isinstance(value, str):
        builtin_dataset = (DATASET_CFG_DIR / value).resolve()
        if builtin_dataset.exists():
            return str(builtin_dataset)
    return value
def normalize_request(request: dict[str, Any]) -> dict[str, Any]:
    request = deepcopy(request)
    request.setdefault("workspace_root", str(REPO_ROOT))
    request.setdefault("request_id", default_request_id(request.get("skill", "skill")))
    request.setdefault("runtime", {})
    request.setdefault("inputs", {})
    request.setdefault("params", {})
    request.setdefault("artifacts", {})
    request.setdefault("policy", {})
    request["inputs"] = normalize_value(request["inputs"])
    request["params"] = normalize_value(request["params"])
    request["artifacts"] = normalize_value(request["artifacts"])
    return request
def is_dry_run(request: dict[str, Any]) -> bool:
    return bool(request.get("policy", {}).get("dry_run", False))
def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "skill"
def default_request_id(skill: str) -> str:
    return f"{slugify(skill)}-{uuid4().hex[:8]}"
def prefer_cli(request: dict[str, Any]) -> bool:
    runtime = request.get("runtime", {})
    if runtime.get("prefer_python_api"):
        return False
    return runtime.get("prefer_cli", True)
def reference_state(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {"requested": value, "resolved": value, "exists": False}
    resolved = normalize_value(value)
    state = {"requested": value, "resolved": resolved}
    if isinstance(resolved, str) and path_like(resolved):
        state["exists"] = Path(resolved).exists()
    else:
        state["exists"] = bool(resolved)
    return state
def resolved_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()
def coerce_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if re.fullmatch(r"-?\d+", text):
        try:
            return int(text)
        except Exception:
            return text
    try:
        return float(text)
    except Exception:
        return text
def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]
def coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        if isinstance(value, str):
            text = value.strip()
            if not re.fullmatch(r"[-+]?\d+(?:\.0+)?", text):
                return None
            return int(float(text))
        return int(value)
    except Exception:
        return None
def coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None
def split_yolo_and_multimodal_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    yolo_params = {}
    multimodal_params = {}
    for key, value in params.items():
        if key in MULTIMODAL_PARAM_KEYS:
            multimodal_params[key] = value
        else:
            yolo_params[key] = value
    return yolo_params, multimodal_params
def split_yolo_multimodal_evaluate_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    yolo_params: dict[str, Any] = {}
    multimodal_params: dict[str, Any] = {}
    evaluate_params: dict[str, Any] = {}
    for key, value in params.items():
        if key in MULTIMODAL_PARAM_KEYS:
            multimodal_params[key] = value
        elif key in MULTIMODAL_EVALUATE_PARAM_KEYS:
            evaluate_params[key] = value
        else:
            yolo_params[key] = value
    return yolo_params, multimodal_params, evaluate_params
