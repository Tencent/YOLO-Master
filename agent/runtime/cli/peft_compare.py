from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from .contract import json_safe


@dataclass(frozen=True)
class PeftCompareDeps:
    normalize_request: Callable[[dict[str, Any]], dict[str, Any]]
    is_dry_run: Callable[[dict[str, Any]], bool]
    response: Callable[..., dict[str, Any]]
    plan_response: Callable[..., dict[str, Any]]
    write_manifest: Callable[[dict[str, Any], dict[str, Any]], Any]
    best_checkpoint: Callable[[dict[str, Any]], str | None]
    run_train_like: Callable[[dict[str, Any], str], dict[str, Any]]
    run_val: Callable[[dict[str, Any]], dict[str, Any]]


DEFAULT_VARIANTS = [
    {"name": "full_sft", "train": {"lora_r": 0}},
    {"name": "lora_r16", "train": {"lora_type": "lora", "lora_r": 16, "lora_alpha": 32, "lora_use_rslora": True}},
    {"name": "dora_r16", "train": {"lora_type": "lora", "lora_r": 16, "lora_alpha": 32, "lora_use_dora": True, "lora_use_rslora": True}},
    {"name": "loha_r16", "train": {"lora_type": "loha", "lora_r": 16, "lora_alpha": 32}},
]


def variants_from_params(params: dict[str, Any]) -> list[dict[str, Any]]:
    variants = params.get("variants")
    if isinstance(variants, list) and variants:
        normalized = []
        for idx, item in enumerate(variants):
            if isinstance(item, str):
                normalized.append({"name": item, "train": {"lora_type": item}})
            elif isinstance(item, dict):
                normalized.append({"name": item.get("name") or f"variant_{idx + 1}", **item})
        return normalized
    types = params.get("lora_types")
    if isinstance(types, list) and types:
        rank = int(params.get("lora_r", 16))
        alpha = int(params.get("lora_alpha", rank * 2))
        return [
            {"name": str(lora_type), "train": {"lora_type": str(lora_type), "lora_r": rank, "lora_alpha": alpha}}
            for lora_type in types
        ]
    return deepcopy(DEFAULT_VARIANTS)


def _nested_get(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _score_payload(payload: dict[str, Any], metric_path: str) -> float | None:
    for root in ("evaluation", "metrics", "data"):
        value = _nested_get(payload.get(root, {}) if isinstance(payload.get(root), dict) else {}, metric_path)
        if value is None:
            value = (payload.get(root) or {}).get(metric_path) if isinstance(payload.get(root), dict) else None
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _variant_train_skill(train_params: dict[str, Any], explicit: str | None = None) -> str:
    if explicit:
        return explicit
    try:
        rank = int(train_params.get("lora_r", 0) or 0)
    except (TypeError, ValueError):
        rank = 0
    return "yolo.lora.train" if rank > 0 or train_params.get("lora_type") else "yolo.train"


def _preview_variant(base_train: dict[str, Any], base_val: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    train_params = {**base_train, **dict(variant.get("train") or variant.get("params") or {})}
    val_params = {**base_val, **dict(variant.get("val") or {})}
    skill = _variant_train_skill(train_params, variant.get("skill"))
    return {
        "name": variant.get("name"),
        "train_skill": skill,
        "train": json_safe(train_params),
        "val": json_safe(val_params),
    }


def _stage_artifacts(request: dict[str, Any], variant_name: str) -> dict[str, Any]:
    artifacts = dict(request.get("artifacts", {}))
    base_name = artifacts.get("name")
    if base_name:
        artifacts["name"] = f"{base_name}-{variant_name}"
    elif artifacts.get("project"):
        artifacts["name"] = variant_name
    return artifacts


def run_peft_compare(request: dict[str, Any], deps: PeftCompareDeps) -> dict[str, Any]:
    params = dict(request["params"])
    variants = variants_from_params(params)
    base_train = dict(params.get("train") or {})
    base_val = dict(params.get("val") or {})
    run_val = bool(params.get("run_val", bool(base_val)))
    rank_metric = str(params.get("rank_metric") or "map50_95")
    previews = [_preview_variant(base_train, base_val, variant) for variant in variants]

    if deps.is_dry_run(request):
        return deps.plan_response(
            request,
            "PEFT comparison dry run prepared",
            "orchestrator",
            "yolo.eval.peft_compare",
            params={
                "variants": previews,
                "run_val": run_val,
                "rank_metric": rank_metric,
            },
            next_actions=["yolo.lora.train", "yolo.val", "yolo.lora.diagnose"],
        )

    results = []
    for idx, variant in enumerate(variants):
        name = str(variant.get("name") or f"variant_{idx + 1}")
        train_params = {**base_train, **dict(variant.get("train") or variant.get("params") or {})}
        val_params = {**base_val, **dict(variant.get("val") or {})}
        train_skill = _variant_train_skill(train_params, variant.get("skill"))
        train_request = deps.normalize_request(
            {
                "skill": train_skill,
                "runtime": deepcopy(request.get("runtime", {})),
                "inputs": deepcopy(request.get("inputs", {})),
                "params": train_params,
                "artifacts": _stage_artifacts(request, name),
                "policy": deepcopy(request.get("policy", {})),
                "request_id": f"{request.get('request_id')}-{name}-train",
            }
        )
        train_payload = deps.run_train_like(train_request, train_skill)
        best = deps.best_checkpoint(train_payload)
        val_payload = None
        score = _score_payload(train_payload, rank_metric)
        if run_val and train_payload.get("status") == "ok":
            val_inputs = {**deepcopy(request.get("inputs", {})), "model": best or request.get("inputs", {}).get("model")}
            val_request = deps.normalize_request(
                {
                    "skill": "yolo.val",
                    "runtime": deepcopy(request.get("runtime", {})),
                    "inputs": val_inputs,
                    "params": val_params,
                    "artifacts": _stage_artifacts(request, f"{name}-val"),
                    "policy": deepcopy(request.get("policy", {})),
                    "request_id": f"{request.get('request_id')}-{name}-val",
                }
            )
            val_payload = deps.run_val(val_request)
            score = _score_payload(val_payload, rank_metric) if val_payload else score
        results.append(
            {
                "name": name,
                "train_skill": train_skill,
                "status": val_payload.get("status") if val_payload else train_payload.get("status"),
                "score": score,
                "best_checkpoint": best,
                "train": train_payload,
                "val": val_payload,
            }
        )

    ranked = sorted(
        [item for item in results if item.get("score") is not None],
        key=lambda item: float(item["score"]),
        reverse=True,
    )
    payload = deps.response(
        request["skill"],
        "ok" if all(item.get("status") == "ok" for item in results) else "partial",
        "PEFT comparison finished",
        peft_compare={
            "rank_metric": rank_metric,
            "best_variant": ranked[0]["name"] if ranked else None,
            "ranking": [{"name": item["name"], "score": item["score"]} for item in ranked],
        },
        results=results,
        next_actions=["yolo.lora.diagnose"],
    )
    payload["manifest"] = str(deps.write_manifest(request, payload))
    return payload
