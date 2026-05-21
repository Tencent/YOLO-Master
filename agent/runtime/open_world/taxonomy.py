from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


SKILL_ROOT = Path(__file__).resolve().parents[2]
OPEN_WORLD_TAXONOMY_DIR = SKILL_ROOT / "assets" / "open-world-taxonomy"
DEFAULT_OPEN_WORLD_LABEL_ALIASES = {
    "meal box": "bento box",
    "lunch box": "bento box",
    "lunchbox": "bento box",
    "bento meal": "bento box",
    "bento box meal": "bento box",
    "dried apricot": "dried fruit",
    "dried mango": "dried fruit",
    "apricot": "dried fruit",
    "pineapple chunks": "pineapple",
    "pineapple piece": "pineapple",
    "meat ball": "meatball",
    "meatball in sauce": "meatball",
    "bouquet": "flower arrangement",
    "flower bouquet": "flower arrangement",
    "flowers": "flower arrangement",
    "floral arrangement": "flower arrangement",
    "tree trunk": "log",
    "fallen tree": "log",
    "grass patch": "grass",
}
OPEN_WORLD_GENERIC_LABELS = {
    "food",
    "meal",
    "dish",
    "object",
    "objects",
    "container",
    "scene",
    "outdoor scene",
    "indoor scene",
}
DEFAULT_OPEN_WORLD_TAXONOMY_FILES = {
    "lvis": OPEN_WORLD_TAXONOMY_DIR / "lvis_1203_classes.json",
    "v3det": OPEN_WORLD_TAXONOMY_DIR / "v3det_13204_classes.json",
}
STATIC_HYPERNYM_FALLBACKS = {
    "grass": ["gramineous plant"],
}
_CACHE: dict[str, Any] = {}


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def cached(key: str, loader):
    if key not in _CACHE:
        _CACHE[key] = loader()
    return _CACHE[key]


def normalize_open_world_label_text(label: str) -> str:
    return re.sub(r"\s+", " ", str(label).strip().lower())


def normalize_detection_boxes(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in detections:
        image_path = item.get("path")
        for idx, detection in enumerate(item.get("detections", []) or []):
            bbox = detection.get("xyxy") or detection.get("bbox_xyxy") or detection.get("bbox")
            if bbox is None:
                continue
            normalized.append(
                {
                    "image_path": image_path,
                    "index": int(detection.get("index", idx)),
                    "class_id": detection.get("class_id"),
                    "label": detection.get("label"),
                    "confidence": detection.get("confidence"),
                    "bbox_xyxy": [float(v) for v in bbox[:4]],
                }
            )
    return normalized


def taxonomy_token_set(label: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", normalize_open_world_label_text(label)) if token}


def resolved_open_world_aliases(multimodal_params: dict[str, Any]) -> dict[str, str]:
    aliases = dict(DEFAULT_OPEN_WORLD_LABEL_ALIASES)
    alias_file = multimodal_params.get("open_world_label_aliases_path")
    candidates = [Path(alias_file)] if alias_file else [SKILL_ROOT / "assets" / "open_world_label_aliases.json"]
    custom = multimodal_params.get("open_world_label_aliases")
    if isinstance(custom, dict):
        for key, value in custom.items():
            if key in (None, "") or value in (None, ""):
                continue
            aliases[normalize_open_world_label_text(str(key))] = str(value).strip()
    for candidate in candidates:
        try:
            if candidate.exists():
                loaded = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    for key, value in loaded.items():
                        if key in (None, "") or value in (None, ""):
                            continue
                        aliases[normalize_open_world_label_text(str(key))] = str(value).strip()
        except Exception:
            continue
    return aliases


def open_world_template_enabled(multimodal_params: dict[str, Any]) -> bool:
    template = str(multimodal_params.get("prompt_template") or "")
    return template in {"vlm_open_world_detection", "vlm_open_world_detection_compact"}


def open_world_policy_enabled(multimodal_params: dict[str, Any]) -> bool:
    policy = str(multimodal_params.get("fusion_policy") or "").lower()
    return policy in {"open_world", "open_world_assist", "open-world", "open-world-assist"}


def compact_open_world_profile(multimodal_params: dict[str, Any], user_prompt: str = "") -> str:
    explicit = str(multimodal_params.get("compact_open_world_profile") or multimodal_params.get("open_world_profile") or "").strip().lower()
    if explicit in {"detect_classify", "detect-classify", "classification"}:
        return "detect_classify"
    if explicit in {"caption_misses", "caption-misses", "misses"}:
        return "caption_misses"
    prompt_text = user_prompt.lower()
    if any(token in prompt_text for token in ("missed", "misses", "caption", "summarize the scene", "scene summary")):
        return "caption_misses"
    return "detect_classify"


def resolved_open_world_assist_profile(multimodal_params: dict[str, Any]) -> str:
    explicit = str(multimodal_params.get("open_world_assist_profile") or "").strip().lower()
    if explicit in {"strict", "balanced", "exploratory"}:
        return explicit
    return "strict"


def apply_open_world_assist_profile_defaults(multimodal_params: dict[str, Any]) -> dict[str, Any]:
    updated = dict(multimodal_params)
    if not open_world_policy_enabled(updated):
        return updated
    profile = resolved_open_world_assist_profile(updated)
    updated["open_world_assist_profile"] = profile
    defaults_by_profile = {
        "strict": {
            "open_world_taxonomy_min_score": 40,
            "open_world_taxonomy_require_exact_for_generic": True,
            "open_world_filter_unmatched_taxonomy": True,
            "open_world_filter_generic_labels": True,
            "fusion_open_world_confidence_min": 0.55,
        },
        "balanced": {
            "open_world_taxonomy_min_score": 30,
            "open_world_taxonomy_require_exact_for_generic": True,
            "open_world_filter_unmatched_taxonomy": False,
            "open_world_filter_generic_labels": True,
            "fusion_open_world_confidence_min": 0.5,
        },
        "exploratory": {
            "open_world_taxonomy_min_score": 20,
            "open_world_taxonomy_require_exact_for_generic": False,
            "open_world_filter_unmatched_taxonomy": False,
            "open_world_filter_generic_labels": False,
            "fusion_open_world_confidence_min": 0.45,
        },
    }
    for key, value in defaults_by_profile[profile].items():
        if key not in updated:
            updated[key] = value
    return updated


def provider_prefers_compact_open_world_schema(provider_cfg: dict[str, Any], prompt_template: Any, multimodal_params: dict[str, Any]) -> bool:
    template_name = str(prompt_template or "")
    if template_name != "vlm_open_world_detection":
        return False
    if not open_world_policy_enabled(multimodal_params):
        return False
    model_name = str(provider_cfg.get("vlm_model") or "").lower()
    return any(token in model_name for token in ("qwen-vl", "qwen_vl", "qwenvl"))


def effective_prompt_template_name(provider_cfg: dict[str, Any], prompt_template: Any, multimodal_params: dict[str, Any], user_prompt: str = "") -> str | None:
    template_name = str(prompt_template or "")
    if not template_name:
        return None
    if provider_prefers_compact_open_world_schema(provider_cfg, prompt_template, multimodal_params):
        profile = compact_open_world_profile(multimodal_params, user_prompt)
        return "vlm_open_world_caption_misses_compact" if profile == "caption_misses" else "vlm_open_world_detect_classify_compact"
    return str(prompt_template)


def default_multimodal_max_output_tokens(
    provider_cfg: dict[str, Any],
    prompt_template: Any,
    multimodal_params: dict[str, Any],
    *,
    structured_output: bool,
    user_prompt: str = "",
) -> int:
    effective_template = str(effective_prompt_template_name(provider_cfg, prompt_template, multimodal_params, user_prompt) or "")
    if effective_template in {
        "vlm_open_world_detection_compact",
        "vlm_open_world_detect_classify_compact",
        "vlm_open_world_caption_misses_compact",
    }:
        return 1200 if structured_output else 800
    if prompt_template and structured_output:
        return 3500
    return 1000 if structured_output else 800


def load_open_world_taxonomy(dataset: str) -> list[dict[str, Any]]:
    cache_key = f"open_world_taxonomy:{dataset}"

    def _loader():
        path = DEFAULT_OPEN_WORLD_TAXONOMY_FILES.get(dataset)
        if path is None or not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        results = []
        for item in payload.get("classes", []):
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            results.append(
                {
                    "dataset": dataset,
                    "id": item.get("id"),
                    "name": name,
                    "normalized_name": normalize_open_world_label_text(name),
                    "tokens": taxonomy_token_set(name),
                }
            )
        return results

    return cached(cache_key, _loader)


def taxonomy_match_score(label_info: dict[str, Any], candidate: dict[str, Any]) -> float:
    canonical = str(label_info.get("canonical_label") or "")
    normalized = str(label_info.get("normalized_label") or "")
    candidate_name = str(candidate.get("normalized_name") or "")
    candidate_tokens = set(candidate.get("tokens") or set())
    label_tokens = taxonomy_token_set(canonical or normalized)
    if not candidate_name or not label_tokens:
        return 0.0
    if candidate_name == canonical or candidate_name == normalized:
        return 100.0
    score = 0.0
    if canonical and canonical in candidate_name:
        score += 20.0
    if candidate_name in {canonical, normalized}:
        score += 15.0
    overlap = len(label_tokens & candidate_tokens)
    score += overlap * 5.0
    union = len(label_tokens | candidate_tokens) or 1
    score += (overlap / union) * 10.0
    if candidate_name.startswith(canonical) or candidate_name.endswith(canonical):
        score += 3.0
    return round(score, 4)


def _wordnet_hypernym_candidates(label: str) -> list[dict[str, Any]]:
    def static_candidates() -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for name in STATIC_HYPERNYM_FALLBACKS.get(normalize_open_world_label_text(label), []):
            normalized = normalize_open_world_label_text(name)
            candidates.append(
                {
                    "dataset": "wordnet",
                    "id": None,
                    "name": normalized,
                    "normalized_name": normalized,
                    "tokens": taxonomy_token_set(normalized),
                    "source": "static_hypernym_fallback",
                }
            )
        return candidates

    try:
        from nltk.corpus import wordnet as wn  # type: ignore
    except Exception:
        return static_candidates()
    text = normalize_open_world_label_text(label).replace("-", " ")
    tokens = [token for token in re.split(r"[^a-z0-9]+", text) if token]
    try:
        synsets = wn.synsets(text) or (wn.synsets(tokens[0]) if tokens else [])
    except Exception:
        return static_candidates()
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for syn in synsets[:3]:
        for hyper in syn.hypernyms()[:4]:
            for lemma in hyper.lemma_names()[:3]:
                candidate = normalize_open_world_label_text(lemma.replace("_", " "))
                if not candidate or candidate in seen:
                    continue
                seen.add(candidate)
                candidates.append(
                    {
                        "dataset": "wordnet",
                        "id": None,
                        "name": candidate,
                        "normalized_name": candidate,
                        "tokens": taxonomy_token_set(candidate),
                        "source": "wordnet_hypernym",
                    }
                )
    return candidates or static_candidates()


def resolve_open_world_taxonomy_candidates(label_info: dict[str, Any], multimodal_params: dict[str, Any]) -> dict[str, Any]:
    datasets = multimodal_params.get("open_world_taxonomy_datasets") or ["lvis", "v3det"]
    dataset_list = [str(item).strip().lower() for item in as_list(datasets) if str(item).strip()]
    topk = max(1, int(multimodal_params.get("open_world_taxonomy_topk", 5) or 5))
    min_score = float(multimodal_params.get("open_world_taxonomy_min_score", 40.0) or 40.0)
    require_exact_for_generic = parse_bool(multimodal_params.get("open_world_taxonomy_require_exact_for_generic"), True)
    enable_hypernym_fallback = parse_bool(multimodal_params.get("open_world_taxonomy_hypernym_fallback"), True)
    scored: list[dict[str, Any]] = []
    for dataset in dataset_list:
        for candidate in load_open_world_taxonomy(dataset):
            score = taxonomy_match_score(label_info, candidate)
            if score <= 0:
                continue
            scored.append(
                {
                    "dataset": dataset,
                    "id": candidate.get("id"),
                    "name": candidate.get("name"),
                    "normalized_name": candidate.get("normalized_name"),
                    "score": score,
                }
            )
    scored.sort(key=lambda item: (-float(item.get("score") or 0.0), str(item.get("dataset")), str(item.get("name"))))
    top = scored[:topk]
    raw_best = top[0] if top else None
    best = raw_best
    match_status = "matched" if best else "unmatched"
    canonical = str(label_info.get("canonical_label") or "")
    if best and float(best.get("score") or 0.0) < min_score:
        best = None
        match_status = "below_min_score"
    if best and require_exact_for_generic and canonical in OPEN_WORLD_GENERIC_LABELS:
        if str(best.get("normalized_name") or "") != canonical:
            best = None
            match_status = "generic_requires_exact"

    if not best and enable_hypernym_fallback:
        hypernym_candidates = _wordnet_hypernym_candidates(canonical or str(label_info.get("normalized_label") or ""))
        if hypernym_candidates:
            fallback = hypernym_candidates[0]
            best = {
                "dataset": fallback.get("dataset"),
                "id": fallback.get("id"),
                "name": fallback.get("name"),
                "normalized_name": fallback.get("normalized_name"),
                "score": max(25.0, min_score),
                "source": fallback.get("source"),
            }
            if not raw_best:
                raw_best = best
            top = scored[:topk] + [
                {
                    "dataset": fallback.get("dataset"),
                    "id": fallback.get("id"),
                    "name": fallback.get("name"),
                    "normalized_name": fallback.get("normalized_name"),
                    "score": best["score"],
                    "source": fallback.get("source"),
                }
            ]
            match_status = "hypernym_fallback"

    return {
        "datasets": dataset_list,
        "topk": topk,
        "min_score": min_score,
        "require_exact_for_generic": require_exact_for_generic,
        "matched": bool(best),
        "best": best,
        "raw_best": raw_best,
        "match_status": match_status,
        "candidates": top,
    }


def canonicalize_open_world_label(label: Any, multimodal_params: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(label, str) or not label.strip():
        return {"raw_label": label, "canonical_label": None, "status": "empty"}
    raw = str(label).strip()
    normalized = normalize_open_world_label_text(raw)
    aliases = resolved_open_world_aliases(multimodal_params)
    canonical = aliases.get(normalized, normalized)
    if canonical.endswith("es") and canonical[:-2] in aliases.values():
        canonical = canonical[:-2]
    elif canonical.endswith("s") and canonical[:-1] not in {"grass"} and len(canonical) > 4:
        singular = canonical[:-1]
        if singular in aliases.values() or singular in {"meatball", "pineapple", "giraffe", "tree", "log", "broccoli", "almond", "flower arrangement", "dried fruit", "bento box"}:
            canonical = singular
    category = "generic" if canonical in OPEN_WORLD_GENERIC_LABELS else "object"
    status = "aliased" if canonical != normalized else "normalized"
    result = {
        "raw_label": raw,
        "normalized_label": normalized,
        "canonical_label": canonical,
        "category": category,
        "status": status,
    }
    if parse_bool(multimodal_params.get("open_world_label_normalizer"), True):
        result["taxonomy"] = resolve_open_world_taxonomy_candidates(result, multimodal_params)
    return result


def resolve_open_world_stats_policy(label_info: dict[str, Any], multimodal_params: dict[str, Any]) -> dict[str, Any]:
    canonical = str(label_info.get("canonical_label") or "")
    category = str(label_info.get("category") or "")
    taxonomy = label_info.get("taxonomy") if isinstance(label_info.get("taxonomy"), dict) else {}
    filter_unmatched_taxonomy = parse_bool(multimodal_params.get("open_world_filter_unmatched_taxonomy"), True)
    filter_generic_labels = parse_bool(multimodal_params.get("open_world_filter_generic_labels"), True)
    include_in_stats = True
    reasons: list[str] = []
    if filter_generic_labels and (category == "generic" or canonical in OPEN_WORLD_GENERIC_LABELS):
        include_in_stats = False
        reasons.append("generic_label")
    if filter_unmatched_taxonomy and taxonomy and not parse_bool(taxonomy.get("matched"), False):
        include_in_stats = False
        reasons.append(f"taxonomy_{taxonomy.get('match_status') or 'unmatched'}")
    report_bucket = "enhancement_stats" if include_in_stats else "reasoning_only"
    return {
        "include_in_stats": include_in_stats,
        "report_bucket": report_bucket,
        "filter_reasons": reasons,
        "filter_policy": {
            "filter_unmatched_taxonomy": filter_unmatched_taxonomy,
            "filter_generic_labels": filter_generic_labels,
        },
    }


def normalize_open_world_prediction_items(items: list[dict[str, Any]], multimodal_params: dict[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        label_info = canonicalize_open_world_label(item.get("open_label") or item.get("label"), multimodal_params)
        stats_policy = resolve_open_world_stats_policy(label_info, multimodal_params)
        enriched = dict(item)
        enriched.update(
            {
                "normalized_open_label": label_info.get("normalized_label"),
                "canonical_open_label": label_info.get("canonical_label"),
                "open_label_category": label_info.get("category"),
                "open_label_status": label_info.get("status"),
                "taxonomy": label_info.get("taxonomy"),
                "include_in_open_world_stats": stats_policy.get("include_in_stats"),
                "open_world_report_bucket": stats_policy.get("report_bucket"),
                "open_world_filter_reasons": stats_policy.get("filter_reasons"),
                "open_world_filter_policy": stats_policy.get("filter_policy"),
            }
        )
        normalized.append(enriched)
    return normalized


def normalize_possible_miss_items(items: Any, multimodal_params: dict[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in as_list(items):
        if isinstance(item, dict):
            label = item.get("label") or item.get("open_label") or item.get("name")
            info = canonicalize_open_world_label(label, multimodal_params)
            stats_policy = resolve_open_world_stats_policy(info, multimodal_params)
            enriched = dict(item)
            enriched.update(
                {
                    "normalized_label": info.get("normalized_label"),
                    "canonical_label": info.get("canonical_label"),
                    "label_category": info.get("category"),
                    "label_status": info.get("status"),
                    "taxonomy": info.get("taxonomy"),
                    "include_in_open_world_stats": stats_policy.get("include_in_stats"),
                    "open_world_report_bucket": stats_policy.get("report_bucket"),
                    "open_world_filter_reasons": stats_policy.get("filter_reasons"),
                    "open_world_filter_policy": stats_policy.get("filter_policy"),
                }
            )
            normalized.append(enriched)
        elif isinstance(item, str):
            info = canonicalize_open_world_label(item, multimodal_params)
            stats_policy = resolve_open_world_stats_policy(info, multimodal_params)
            normalized.append(
                {
                    "label": item,
                    "normalized_label": info.get("normalized_label"),
                    "canonical_label": info.get("canonical_label"),
                    "label_category": info.get("category"),
                    "label_status": info.get("status"),
                    "taxonomy": info.get("taxonomy"),
                    "include_in_open_world_stats": stats_policy.get("include_in_stats"),
                    "open_world_report_bucket": stats_policy.get("report_bucket"),
                    "open_world_filter_reasons": stats_policy.get("filter_reasons"),
                    "open_world_filter_policy": stats_policy.get("filter_policy"),
                }
            )
    return normalized


def build_open_world_comparison_entry(
    *,
    image_path: str | Path | None,
    detections: list[dict[str, Any]],
    fusion_preview: dict[str, Any],
    verdict: dict[str, Any],
    multimodal_params: dict[str, Any],
    effective_prompt_template: str | None = None,
) -> dict[str, Any]:
    yolo_labels = [item.get("label") for item in normalize_detection_boxes(detections) if item.get("label")]
    raw_open_world = fusion_preview.get("open_world_predictions_preview", []) if isinstance(fusion_preview, dict) else []
    normalized_open_world = normalize_open_world_prediction_items(raw_open_world if isinstance(raw_open_world, list) else [], multimodal_params)
    cross_check = verdict.get("yolo_cross_check", {}) if isinstance(verdict, dict) else {}
    possible_misses = normalize_possible_miss_items(cross_check.get("possible_misses"), multimodal_params) if isinstance(cross_check, dict) else []
    false_positives = normalize_possible_miss_items(cross_check.get("false_positives"), multimodal_params) if isinstance(cross_check, dict) else []
    fusion_summary = fusion_preview.get("summary", {}) if isinstance(fusion_preview, dict) else {}
    return {
        "image": str(image_path) if image_path is not None else None,
        "effective_prompt_template": effective_prompt_template,
        "yolo_labels": yolo_labels,
        "open_world_predictions": normalized_open_world,
        "possible_misses": possible_misses,
        "false_positives": false_positives,
        "perturbation": {
            "suppressed": int(fusion_summary.get("suppressed", 0) or 0),
            "adjusted": int(fusion_summary.get("adjusted", 0) or 0),
            "relabelled": int(fusion_summary.get("relabelled", 0) or 0),
        },
    }


def aggregate_open_world_comparison(entries: list[dict[str, Any]]) -> dict[str, Any]:
    open_world_counts: dict[str, int] = {}
    miss_counts: dict[str, int] = {}
    reasoning_only_counts = {"open_world_predictions": 0, "possible_misses": 0, "false_positives": 0}
    taxonomy_dataset_counts = {"open_world_predictions": {}, "possible_misses": {}, "false_positives": {}}
    taxonomy_label_counts: dict[str, int] = {}
    taxonomy_unmatched = {"open_world_predictions": 0, "possible_misses": 0, "false_positives": 0}
    filtered_reason_counts: dict[str, int] = {}
    perturbation = {"suppressed": 0, "adjusted": 0, "relabelled": 0}
    for entry in entries:
        for item in entry.get("open_world_predictions", []) or []:
            if not parse_bool(item.get("include_in_open_world_stats"), True):
                reasoning_only_counts["open_world_predictions"] += 1
                for reason in as_list(item.get("open_world_filter_reasons")):
                    if reason:
                        filtered_reason_counts[str(reason)] = filtered_reason_counts.get(str(reason), 0) + 1
                continue
            label = item.get("canonical_open_label") or item.get("open_label") or item.get("label")
            if label:
                open_world_counts[str(label)] = open_world_counts.get(str(label), 0) + 1
            taxonomy = item.get("taxonomy") if isinstance(item, dict) else None
            best = taxonomy.get("best") if isinstance(taxonomy, dict) else None
            if isinstance(best, dict) and best.get("dataset"):
                dataset = str(best["dataset"])
                taxonomy_dataset_counts["open_world_predictions"][dataset] = taxonomy_dataset_counts["open_world_predictions"].get(dataset, 0) + 1
                taxonomy_label = best.get("normalized_name") or best.get("name")
                if taxonomy_label:
                    taxonomy_label_counts[str(taxonomy_label)] = taxonomy_label_counts.get(str(taxonomy_label), 0) + 1
            else:
                taxonomy_unmatched["open_world_predictions"] += 1
        for item in entry.get("possible_misses", []) or []:
            if not parse_bool(item.get("include_in_open_world_stats"), True):
                reasoning_only_counts["possible_misses"] += 1
                for reason in as_list(item.get("open_world_filter_reasons")):
                    if reason:
                        filtered_reason_counts[str(reason)] = filtered_reason_counts.get(str(reason), 0) + 1
                continue
            label = item.get("canonical_label") or item.get("label")
            if label:
                miss_counts[str(label)] = miss_counts.get(str(label), 0) + 1
            taxonomy = item.get("taxonomy") if isinstance(item, dict) else None
            best = taxonomy.get("best") if isinstance(taxonomy, dict) else None
            if isinstance(best, dict) and best.get("dataset"):
                dataset = str(best["dataset"])
                taxonomy_dataset_counts["possible_misses"][dataset] = taxonomy_dataset_counts["possible_misses"].get(dataset, 0) + 1
            else:
                taxonomy_unmatched["possible_misses"] += 1
        for item in entry.get("false_positives", []) or []:
            if not parse_bool(item.get("include_in_open_world_stats"), True):
                reasoning_only_counts["false_positives"] += 1
                for reason in as_list(item.get("open_world_filter_reasons")):
                    if reason:
                        filtered_reason_counts[str(reason)] = filtered_reason_counts.get(str(reason), 0) + 1
                continue
            taxonomy = item.get("taxonomy") if isinstance(item, dict) else None
            best = taxonomy.get("best") if isinstance(taxonomy, dict) else None
            if isinstance(best, dict) and best.get("dataset"):
                dataset = str(best["dataset"])
                taxonomy_dataset_counts["false_positives"][dataset] = taxonomy_dataset_counts["false_positives"].get(dataset, 0) + 1
            else:
                taxonomy_unmatched["false_positives"] += 1
        perturb = entry.get("perturbation", {}) or {}
        for key in perturbation:
            perturbation[key] += int(perturb.get(key, 0) or 0)
    return {
        "images": len(entries),
        "open_world_label_counts": open_world_counts,
        "possible_miss_label_counts": miss_counts,
        "reasoning_only_counts": reasoning_only_counts,
        "filtered_reason_counts": filtered_reason_counts,
        "taxonomy_dataset_counts": taxonomy_dataset_counts,
        "taxonomy_label_counts": taxonomy_label_counts,
        "taxonomy_unmatched": taxonomy_unmatched,
        "perturbation_totals": perturbation,
    }


def box_iou_xyxy(box_a: list[float] | tuple[float, ...], box_b: list[float] | tuple[float, ...]) -> float:
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    return 0.0 if denom <= 0 else round(inter_area / denom, 4)


def merge_verified_open_world_candidates(entries: list[dict[str, Any]], *, iou_threshold: float = 0.7) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for entry in entries:
        for candidate in entry.get("open_world_predictions", []) or []:
            bbox = candidate.get("bbox_xyxy") or candidate.get("xyxy")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            match = None
            for existing in merged:
                existing_bbox = existing.get("bbox_xyxy") or existing.get("xyxy")
                if not isinstance(existing_bbox, (list, tuple)) or len(existing_bbox) != 4:
                    continue
                if box_iou_xyxy(existing_bbox, bbox) < iou_threshold:
                    continue
                match = existing
                break
            if match is None:
                merged.append(dict(candidate))
                continue
            candidate_score = float(candidate.get("confidence") or 0.0)
            existing_score = float(match.get("confidence") or 0.0)
            candidate_matched = parse_bool(candidate.get("taxonomy", {}).get("matched"), False) if isinstance(candidate.get("taxonomy"), dict) else False
            existing_matched = parse_bool(match.get("taxonomy", {}).get("matched"), False) if isinstance(match.get("taxonomy"), dict) else False
            if candidate_matched and not existing_matched:
                merged[merged.index(match)] = dict(candidate)
            elif candidate_score > existing_score:
                merged[merged.index(match)] = dict(candidate)
    return merged
