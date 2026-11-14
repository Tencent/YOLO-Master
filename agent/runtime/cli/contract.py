from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.security import _known_secret_values

SKILL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_DIR = REPO_ROOT / "runs" / "agent"
PROVIDER_CONFIG_DIR = SKILL_ROOT / "runtime" / "multimodal" / "providers"
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_LOG_LIMIT = 20_000
REDACTED = "<redacted>"

_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "client_secret",
    "password",
    "refresh_token",
    "secret",
    "token",
}
_INLINE_SECRET_RE = re.compile(
    r"(?i)\b((?:[a-z0-9]+[_-])?(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|authorization))"
    r"(\s*[:=]\s*)([^\s,;&]+)"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[a-z0-9._~+/=-]+")
_URL_USERINFO_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)([^\s/]*@)")


def _sensitive_key(key: Any) -> bool:
    """Return whether a mapping key conventionally contains a credential."""
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
    return normalized in _SENSITIVE_KEYS or normalized.endswith(
        ("_api_key", "_access_token", "_refresh_token", "_password", "_secret")
    )


def _redact_inline_secrets(value: str) -> str:
    """Redact common credentials embedded in CLI strings or free text."""
    value = _BEARER_RE.sub(f"Bearer {REDACTED}", value)
    return _INLINE_SECRET_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}", value)


def redact_sensitive(value: Any) -> Any:
    """Recursively redact secret fields and inline credential assignments."""
    if isinstance(value, dict):
        return {str(key): REDACTED if _sensitive_key(key) else redact_sensitive(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [redact_sensitive(child) for child in value]
    if isinstance(value, str):
        return _redact_inline_secrets(value)
    return value


def request_secret_values(value: Any) -> set[str]:
    """Collect request credentials for scrubbing bare values echoed by child code."""
    secrets: set[str] = set()

    def visit(item: Any, sensitive: bool = False) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                name = str(key).lower().replace("-", "_")
                credential = _sensitive_key(key) or name in {"cookie", "passwd", "access_token"}
                credential |= name.endswith(("_token", "_access_key", "_private_key"))
                visit(child, sensitive or credential)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child, sensitive)
        elif isinstance(item, str):
            if sensitive and item:
                secrets.add(item)
            secrets.update(match.group(3).strip("\"'") for match in _INLINE_SECRET_RE.finditer(item))
            secrets.update(match.group().split(None, 1)[1] for match in _BEARER_RE.finditer(item))
            secrets.update(match.group(2)[:-1] for match in _URL_USERINFO_RE.finditer(item))

    visit(value)
    # JSON output escapes credentials; text output may split multiline credentials.
    for secret in tuple(secrets):
        secrets.update(part for part in secret.splitlines() if part)
        secrets.add(json.dumps(secret, ensure_ascii=False)[1:-1])
        secrets.add(json.dumps(secret, ensure_ascii=True)[1:-1])
    return {secret for secret in secrets if secret}


def redact_job_value(value: Any, secrets: set[str]) -> Any:
    """Sanitize public records and streams without changing execution inputs."""
    # Reuse Core's environment/.env secret discovery, without rescanning every
    # large log line with its general-purpose quoted-assignment regexes.
    known = set(secrets) | _known_secret_values()
    for secret in tuple(known):
        known.add(json.dumps(secret, ensure_ascii=False)[1:-1])
        known.add(json.dumps(secret, ensure_ascii=True)[1:-1])
    known = sorted(known, key=len, reverse=True)

    def scrub(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: scrub(child) for key, child in item.items()}
        if isinstance(item, list):
            return [scrub(child) for child in item]
        if isinstance(item, str):
            for secret in known:
                if len(secret) >= 4:
                    item = item.replace(secret, REDACTED)
                else:
                    item = re.sub(rf"(?<![A-Za-z0-9]){re.escape(secret)}(?![A-Za-z0-9])", REDACTED, item)
            item = _URL_USERINFO_RE.sub(lambda match: match.group(1) + REDACTED + "@", item)
            return item
        return item

    return scrub(redact_sensitive(value))


def _compact_manifest_logs(value: Any) -> Any:
    """Bound raw subprocess logs while preserving their most recent diagnostic tail."""
    if isinstance(value, dict):
        compact = {}
        for key, child in value.items():
            if key in {"stdout", "stderr"} and isinstance(child, str) and len(child) > MANIFEST_LOG_LIMIT:
                omitted = len(child) - MANIFEST_LOG_LIMIT
                compact[key] = f"[truncated {omitted} chars]\n{child[-MANIFEST_LOG_LIMIT:]}"
            else:
                compact[key] = _compact_manifest_logs(child)
        return compact
    if isinstance(value, list):
        return [_compact_manifest_logs(child) for child in value]
    return value


def _manifest_safe(value: Any) -> Any:
    """Convert arbitrary runtime values to bounded, redacted JSON data."""
    return _compact_manifest_logs(redact_sensitive(json_safe(value)))


def manifest_checksum(manifest: dict[str, Any]) -> str:
    """Return the canonical SHA-256 for a manifest excluding its checksum field."""
    payload = dict(_manifest_safe(manifest))
    payload.pop("manifest_sha256", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _request_snapshot(request: dict[str, Any]) -> dict[str, Any]:
    """Return the normalized experiment request without machine-local dispatcher state."""
    excluded = {"request_id", "workspace_root"}
    return _manifest_safe({key: value for key, value in request.items() if key not in excluded})


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON file from a unique temporary file in the same directory."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=".skill_manifest.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


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
    compatibility_fields = {
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
    represented_fields = set(compatibility_fields) | {"manifest"}
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **compatibility_fields,
        "request": _request_snapshot(request),
        "result": _manifest_safe({key: value for key, value in payload.items() if key not in represented_fields}),
    }
    manifest = redact_job_value(_manifest_safe(manifest), request_secret_values(request))
    manifest["manifest_sha256"] = manifest_checksum(manifest)
    _atomic_write_json(manifest_path, manifest)
    return manifest_path


def finalize_payload(request: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    payload = enrich_envelope(payload)
    payload.setdefault("request_id", request.get("request_id"))
    if "manifest" not in payload:
        payload["manifest"] = str(write_manifest(request, payload))
    return redact_job_value(json_safe(payload), request_secret_values(request))


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
