from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


SKILL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_DIR = REPO_ROOT / "runs" / "agent"
PROVIDER_CONFIG_DIR = SKILL_ROOT / "runtime" / "multimodal" / "providers"


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


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _token_value(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = _number(usage.get(key))
        if value is not None:
            return int(value)
    tokens = usage.get("tokens")
    if isinstance(tokens, dict):
        for key in keys:
            value = _number(tokens.get(key))
            if value is not None:
                return int(value)
    return 0


def normalize_token_usage(raw_usage: dict[str, Any]) -> dict[str, int]:
    input_tokens = _token_value(raw_usage, "input", "input_tokens", "prompt_tokens")
    output_tokens = _token_value(raw_usage, "output", "output_tokens", "completion_tokens")
    total_tokens = _token_value(raw_usage, "total", "total_tokens")
    if total_tokens <= 0 and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens
    return {"input": input_tokens, "output": output_tokens, "total": total_tokens}


def _iter_usage_records(value: Any, *, path: str = "$") -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        usage = value.get("usage")
        if isinstance(usage, dict):
            tokens = normalize_token_usage(usage)
            if tokens["total"] or tokens["input"] or tokens["output"]:
                records.append(
                    {
                        "path": path,
                        "provider": value.get("provider"),
                        "model": value.get("model"),
                        "api_mode": value.get("api_mode"),
                        "tokens": tokens,
                        "raw": json_safe(usage),
                    }
                )
        for key, child in value.items():
            if key in {"usage", "cost_estimate"}:
                continue
            records.extend(_iter_usage_records(child, path=f"{path}.{key}"))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            records.extend(_iter_usage_records(child, path=f"{path}[{idx}]"))
    return records


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except Exception:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _pricing_for(provider: Any, model: Any) -> dict[str, float | None]:
    provider_name = str(provider or "openai").lower()
    config = _load_yaml(PROVIDER_CONFIG_DIR / f"{provider_name}.yaml")
    pricing = config.get("pricing") if isinstance(config.get("pricing"), dict) else {}
    models = pricing.get("models") if isinstance(pricing.get("models"), dict) else {}
    model_pricing = models.get(str(model)) if model is not None else None
    if not isinstance(model_pricing, dict):
        model_pricing = pricing.get("default") if isinstance(pricing.get("default"), dict) else {}
    return {
        "input_per_1k_usd": _number(model_pricing.get("input_per_1k_usd")),
        "output_per_1k_usd": _number(model_pricing.get("output_per_1k_usd")),
    }


def summarize_usage(payload: dict[str, Any]) -> dict[str, Any]:
    records = _iter_usage_records(payload)
    tokens = {
        "input": sum(record["tokens"]["input"] for record in records),
        "output": sum(record["tokens"]["output"] for record in records),
        "total": sum(record["tokens"]["total"] for record in records),
    }
    return {"tokens": tokens, "requests": len(records), "records": records}


def estimate_cost(usage: dict[str, Any]) -> dict[str, Any]:
    amount = 0.0
    priced = 0
    missing = 0
    for record in usage.get("records", []):
        pricing = _pricing_for(record.get("provider"), record.get("model"))
        in_rate = pricing.get("input_per_1k_usd")
        out_rate = pricing.get("output_per_1k_usd")
        if in_rate is None or out_rate is None:
            missing += 1
            continue
        amount += (record["tokens"]["input"] / 1000.0) * float(in_rate)
        amount += (record["tokens"]["output"] / 1000.0) * float(out_rate)
        priced += 1
    if not usage.get("requests"):
        return {"currency": "USD", "amount": 0.0, "basis": "no token usage"}
    if priced == 0:
        return {
            "currency": "USD",
            "amount": None,
            "basis": "provider pricing unavailable",
            "priced_requests": priced,
            "unpriced_requests": missing,
        }
    return {
        "currency": "USD",
        "amount": round(amount, 8),
        "basis": "configured provider token pricing",
        "priced_requests": priced,
        "unpriced_requests": missing,
    }


def enrich_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    if "usage" not in payload:
        payload["usage"] = summarize_usage(payload)
    if "cost_estimate" not in payload:
        payload["cost_estimate"] = estimate_cost(payload["usage"])
    return json_safe(payload)


def ensure_manifest_dir(request: dict[str, Any]) -> Path:
    project = request.get("artifacts", {}).get("project")
    name = request.get("artifacts", {}).get("name")
    base = (REPO_ROOT / project).resolve() if project else DEFAULT_MANIFEST_DIR / request["request_id"]
    target = base / name if name else base
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_manifest(request: dict[str, Any], payload: dict[str, Any]) -> Path:
    payload = enrich_envelope(payload)
    manifest_dir = ensure_manifest_dir(request)
    manifest_path = manifest_dir / "skill_manifest.json"
    manifest = {
        "skill": request.get("skill"),
        "request_id": request.get("request_id"),
        "status": payload.get("status"),
        "summary": payload.get("summary"),
        "artifacts": payload.get("artifacts", []),
        "metrics": payload.get("metrics", {}),
        "evaluation": payload.get("evaluation", {}),
        "environment": payload.get("environment", {}),
        "auto_completed": payload.get("auto_completed", {}),
        "attempts": payload.get("attempts", []),
        "recovery": payload.get("recovery", {}),
        "multimodal": payload.get("multimodal", {}),
        "usage": payload.get("usage", {}),
        "cost_estimate": payload.get("cost_estimate", {}),
        "progress": payload.get("progress", {}),
        "recommendations": payload.get("recommendations", []),
        "job": payload.get("job", {}),
        "dry_run": payload.get("dry_run", False),
    }
    manifest_path.write_text(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def finalize_payload(request: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    payload = enrich_envelope(payload)
    payload.setdefault("request_id", request.get("request_id"))
    if "manifest" not in payload:
        payload["manifest"] = str(write_manifest(request, payload))
    return json_safe(payload)


def response(skill: str, status: str, summary: str, **kwargs: Any) -> dict[str, Any]:
    payload = {"skill": skill, "status": status, "summary": summary}
    payload.update(kwargs)
    return enrich_envelope(payload)


def plan_response(
    request: dict[str, Any],
    summary: str,
    executor: str,
    target: str,
    params: dict[str, Any] | None = None,
    next_actions: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = response(
        request["skill"],
        "ok",
        summary,
        dry_run=True,
        plan={
            "executor": executor,
            "target": target,
            "inputs": json_safe(request.get("inputs", {})),
            "params": json_safe(params if params is not None else request.get("params", {})),
        },
        next_actions=next_actions or [],
    )
    if extra:
        payload.update(json_safe(extra))
        payload = enrich_envelope(payload)
    payload["manifest"] = str(write_manifest(request, payload))
    return payload
