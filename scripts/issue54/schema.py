#!/usr/bin/env python3
"""Versioned schemas and validation for Issue #54 experiment evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable

from agent.runtime.cli.contract import manifest_checksum, redact_sensitive

EXPERIMENT_MANIFEST_SCHEMA_VERSION = 1
ROUTING_RECORD_SCHEMA_VERSION = 1
REGISTRY_SCHEMA_VERSION = 1
ANALYSIS_SCHEMA_VERSION = 1

VALID_STATUSES = frozenset({"passed", "failed", "diagnostic", "not_executed"})
EVIDENCE_STATUSES = frozenset({"passed", "diagnostic"})
FORMAL_STATUS = "passed"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+-]*$")
_DRIVE_RE = re.compile(r"^[A-Za-z]:")

EXPERIMENT_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "model_variant",
        "seed",
        "dataset",
        "dataset_version",
        "dataset_manifest_sha256",
        "split",
        "requested_epochs",
        "epochs",
        "requested_batch",
        "batch",
        "effective_batch",
        "imgsz",
        "optimizer",
        "precision_mode",
        "checkpoint_path",
        "checkpoint_sha256",
        "config_path",
        "config_sha256",
        "git_commit",
        "timestamp",
        "status",
        "failure_reason",
        "manifest_sha256",
    }
)

ROUTING_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "model_variant",
        "seed",
        "dataset",
        "dataset_version",
        "split",
        "checkpoint_sha256",
        "image_id",
        "image_path",
        "image_sha256",
        "scene_groups",
        "layer_name",
        "layer_index",
        "expert_names",
        "expert_probabilities",
        "selected_expert",
        "top_k",
        "token_top1_indices",
        "spatial_shape",
        "inference_repeat",
        "inference_batch_actual",
        "timestamp",
        "status",
        "failure_reason",
    }
)

EXPERIMENT_MANIFEST_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/Tencent/YOLO-Master/reports/issue54/experiment-manifest.schema.json",
    "title": "Issue54ExperimentManifest",
    "type": "object",
    "additionalProperties": False,
    "required": sorted(EXPERIMENT_MANIFEST_FIELDS - {"manifest_sha256"}),
    "properties": {
        "schema_version": {"const": EXPERIMENT_MANIFEST_SCHEMA_VERSION},
        "experiment_id": {"type": "string", "minLength": 1},
        "model_variant": {"type": "string", "minLength": 1},
        "seed": {"type": "integer", "minimum": 0},
        "dataset": {"type": "string", "minLength": 1},
        "dataset_version": {"type": "string", "minLength": 1},
        "dataset_manifest_sha256": {"type": ["string", "null"], "pattern": "^[0-9a-f]{64}$"},
        "split": {"type": "string", "minLength": 1},
        "requested_epochs": {"type": ["integer", "null"], "minimum": 1},
        "epochs": {"type": ["integer", "null"], "minimum": 0},
        "requested_batch": {"type": ["integer", "string", "null"]},
        "batch": {"type": ["integer", "null"], "minimum": 1},
        "effective_batch": {"type": ["integer", "null"], "minimum": 1},
        "imgsz": {"type": ["integer", "null"], "minimum": 1},
        "optimizer": {"type": ["string", "null"]},
        "precision_mode": {"type": ["string", "null"]},
        "checkpoint_path": {"type": ["string", "null"]},
        "checkpoint_sha256": {"type": ["string", "null"], "pattern": "^[0-9a-f]{64}$"},
        "config_path": {"type": ["string", "null"]},
        "config_sha256": {"type": ["string", "null"], "pattern": "^[0-9a-f]{64}$"},
        "git_commit": {"type": "string", "pattern": "^[0-9a-f]{7,40}$"},
        "timestamp": {"type": "string", "format": "date-time"},
        "status": {"enum": sorted(VALID_STATUSES)},
        "failure_reason": {"type": ["string", "null"]},
        "manifest_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    },
}

ROUTING_RECORD_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/Tencent/YOLO-Master/reports/issue54/routing-record.schema.json",
    "title": "Issue54RoutingRecord",
    "type": "object",
    "additionalProperties": False,
    "required": sorted(ROUTING_RECORD_FIELDS),
    "properties": {
        "schema_version": {"const": ROUTING_RECORD_SCHEMA_VERSION},
        "experiment_id": {"type": "string", "minLength": 1},
        "model_variant": {"type": "string", "minLength": 1},
        "seed": {"type": "integer", "minimum": 0},
        "dataset": {"type": "string", "minLength": 1},
        "dataset_version": {"type": "string", "minLength": 1},
        "split": {"type": "string", "minLength": 1},
        "checkpoint_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "image_id": {"type": "string", "minLength": 1},
        "image_path": {"type": "string", "minLength": 1},
        "image_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "scene_groups": {"type": "object", "additionalProperties": {"type": "string"}},
        "layer_name": {"type": "string", "minLength": 1},
        "layer_index": {"type": "integer", "minimum": 0},
        "expert_names": {"type": "array", "items": {"type": "string"}, "minItems": 1, "uniqueItems": True},
        "expert_probabilities": {"type": "array", "items": {"type": "number"}},
        "selected_expert": {"type": ["string", "null"]},
        "top_k": {"type": "integer", "minimum": 0},
        "token_top1_indices": {"type": "array", "items": {"type": "integer", "minimum": 0}},
        "spatial_shape": {"type": "array", "items": {"type": "integer", "minimum": 1}},
        "inference_repeat": {"type": "integer", "minimum": 0},
        "inference_batch_actual": {"type": "integer", "minimum": 1},
        "timestamp": {"type": "string", "format": "date-time"},
        "status": {"enum": sorted(VALID_STATUSES)},
        "failure_reason": {"type": ["string", "null"]},
    },
}


class SchemaValidationError(ValueError):
    """Raised when experiment or routing evidence violates the versioned contract."""


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest for a local file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_payload_sha256(payload: dict[str, Any], *, exclude: tuple[str, ...] = ()) -> str:
    """Return a deterministic SHA-256 for a JSON-compatible payload."""
    normalized = dict(redact_sensitive(payload))
    for field in exclude:
        normalized.pop(field, None)
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def with_manifest_checksum(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a redacted experiment manifest with the repository-standard checksum."""
    normalized = dict(redact_sensitive(payload))
    normalized["manifest_sha256"] = manifest_checksum(normalized)
    return normalized


def _require_exact_fields(
    payload: dict[str, Any],
    allowed: frozenset[str],
    *,
    kind: str,
    optional: frozenset[str] = frozenset(),
) -> None:
    missing = (allowed - optional) - payload.keys()
    unknown = payload.keys() - allowed
    if missing:
        raise SchemaValidationError(f"{kind} missing required fields: {sorted(missing)}")
    if unknown:
        raise SchemaValidationError(f"{kind} contains unknown fields: {sorted(unknown)}")


def _require_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or not _IDENTIFIER_RE.fullmatch(value):
        raise SchemaValidationError(f"{field} must be a non-empty portable identifier")
    return value


def _require_integer(value: Any, field: str, *, minimum: int = 0, nullable: bool = False) -> int | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        suffix = " or null" if nullable else ""
        raise SchemaValidationError(f"{field} must be an integer >= {minimum}{suffix}")
    return value


def _require_sha256(value: Any, field: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        suffix = " or null" if nullable else ""
        raise SchemaValidationError(f"{field} must be a lowercase SHA-256 hex digest{suffix}")
    return value


def _require_timestamp(value: Any, field: str = "timestamp") -> str:
    if not isinstance(value, str) or not value:
        raise SchemaValidationError(f"{field} must be a non-empty ISO-8601 timestamp")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise SchemaValidationError(f"{field} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise SchemaValidationError(f"{field} must include a timezone")
    return value


def _require_public_relative_path(value: Any, field: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip():
        suffix = " or null" if nullable else ""
        raise SchemaValidationError(f"{field} must be a non-empty sanitized relative path{suffix}")
    raw = value.strip()
    windows = PureWindowsPath(raw)
    posix = PurePosixPath(raw.replace("\\", "/"))
    if (
        windows.is_absolute()
        or posix.is_absolute()
        or _DRIVE_RE.match(raw)
        or ".." in windows.parts
        or ".." in posix.parts
        or "~" in windows.parts
    ):
        raise SchemaValidationError(f"{field} must not contain an absolute or parent-traversal path")
    return posix.as_posix()


def _require_optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SchemaValidationError(f"{field} must be a string or null")
    return value


def validate_experiment_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one experiment manifest without guessing from directory names."""
    if not isinstance(payload, dict):
        raise SchemaValidationError("experiment manifest must be a JSON object")
    _require_exact_fields(
        payload,
        EXPERIMENT_MANIFEST_FIELDS,
        kind="experiment manifest",
        optional=frozenset({"manifest_sha256"}),
    )
    if payload["schema_version"] != EXPERIMENT_MANIFEST_SCHEMA_VERSION:
        raise SchemaValidationError(f"unsupported experiment manifest schema_version={payload['schema_version']!r}")

    normalized = dict(payload)
    for field in ("experiment_id", "model_variant", "dataset", "dataset_version", "split"):
        normalized[field] = _require_identifier(payload[field], field)
    normalized["seed"] = _require_integer(payload["seed"], "seed")
    normalized["dataset_manifest_sha256"] = _require_sha256(
        payload["dataset_manifest_sha256"], "dataset_manifest_sha256", nullable=True
    )
    normalized["requested_epochs"] = _require_integer(
        payload["requested_epochs"], "requested_epochs", minimum=1, nullable=True
    )
    normalized["epochs"] = _require_integer(payload["epochs"], "epochs", nullable=True)
    requested_batch = payload["requested_batch"]
    if not (
        requested_batch is None
        or (isinstance(requested_batch, int) and not isinstance(requested_batch, bool) and requested_batch >= 1)
        or requested_batch == "auto"
    ):
        raise SchemaValidationError("requested_batch must be an integer >= 1, 'auto', or null")
    normalized["batch"] = _require_integer(payload["batch"], "batch", minimum=1, nullable=True)
    normalized["effective_batch"] = _require_integer(
        payload["effective_batch"], "effective_batch", minimum=1, nullable=True
    )
    normalized["imgsz"] = _require_integer(payload["imgsz"], "imgsz", minimum=1, nullable=True)
    for field in ("optimizer", "precision_mode", "failure_reason"):
        normalized[field] = _require_optional_text(payload[field], field)
    normalized["checkpoint_path"] = _require_public_relative_path(
        payload["checkpoint_path"], "checkpoint_path", nullable=True
    )
    normalized["config_path"] = _require_public_relative_path(payload["config_path"], "config_path", nullable=True)
    normalized["checkpoint_sha256"] = _require_sha256(payload["checkpoint_sha256"], "checkpoint_sha256", nullable=True)
    normalized["config_sha256"] = _require_sha256(payload["config_sha256"], "config_sha256", nullable=True)
    if not isinstance(payload["git_commit"], str) or not _GIT_COMMIT_RE.fullmatch(payload["git_commit"]):
        raise SchemaValidationError("git_commit must be a 7-40 character lowercase Git commit")
    normalized["timestamp"] = _require_timestamp(payload["timestamp"])
    status = payload["status"]
    if status not in VALID_STATUSES:
        raise SchemaValidationError(f"status must be one of {sorted(VALID_STATUSES)}, got {status!r}")
    if status == "failed" and not normalized["failure_reason"]:
        raise SchemaValidationError("failed experiments require failure_reason")
    if status != "failed" and normalized["failure_reason"]:
        raise SchemaValidationError("failure_reason must be null unless status is failed")
    if status in EVIDENCE_STATUSES:
        for field in (
            "dataset_manifest_sha256",
            "checkpoint_path",
            "checkpoint_sha256",
            "config_path",
            "config_sha256",
        ):
            if normalized[field] is None:
                raise SchemaValidationError(f"{status} experiments require {field}")
        for field in ("requested_epochs", "epochs", "requested_batch", "batch", "effective_batch", "imgsz"):
            if normalized[field] is None:
                raise SchemaValidationError(f"{status} experiments require {field}")
        for field in ("optimizer", "precision_mode"):
            if normalized[field] is None or not normalized[field].strip():
                raise SchemaValidationError(f"{status} experiments require a non-empty {field}")
    if status == "not_executed" and any(
        normalized[field] is not None
        for field in ("checkpoint_path", "checkpoint_sha256", "epochs", "batch", "effective_batch")
    ):
        raise SchemaValidationError(
            "not_executed experiments cannot claim checkpoint or actual/effective training values"
        )
    checksum = payload.get("manifest_sha256")
    if checksum is not None:
        _require_sha256(checksum, "manifest_sha256")
        if manifest_checksum(normalized) != checksum:
            raise SchemaValidationError("manifest_sha256 does not match the canonical redacted manifest")
    return normalized


def validate_routing_record(payload: dict[str, Any], *, probability_tolerance: float = 1e-6) -> dict[str, Any]:
    """Validate and normalize one per-image, per-layer MoT routing record."""
    if not isinstance(payload, dict):
        raise SchemaValidationError("routing record must be a JSON object")
    _require_exact_fields(payload, ROUTING_RECORD_FIELDS, kind="routing record")
    if payload["schema_version"] != ROUTING_RECORD_SCHEMA_VERSION:
        raise SchemaValidationError(f"unsupported routing record schema_version={payload['schema_version']!r}")

    normalized = dict(payload)
    for field in (
        "experiment_id",
        "model_variant",
        "dataset",
        "dataset_version",
        "split",
        "image_id",
        "layer_name",
    ):
        normalized[field] = _require_identifier(payload[field], field)
    normalized["seed"] = _require_integer(payload["seed"], "seed")
    normalized["checkpoint_sha256"] = _require_sha256(payload["checkpoint_sha256"], "checkpoint_sha256")
    normalized["image_path"] = _require_public_relative_path(payload["image_path"], "image_path")
    normalized["image_sha256"] = _require_sha256(payload["image_sha256"], "image_sha256")
    normalized["layer_index"] = _require_integer(payload["layer_index"], "layer_index")
    normalized["inference_repeat"] = _require_integer(payload["inference_repeat"], "inference_repeat")
    normalized["inference_batch_actual"] = _require_integer(
        payload["inference_batch_actual"], "inference_batch_actual", minimum=1
    )
    normalized["timestamp"] = _require_timestamp(payload["timestamp"])

    status = payload["status"]
    if status not in VALID_STATUSES:
        raise SchemaValidationError(f"status must be one of {sorted(VALID_STATUSES)}, got {status!r}")
    failure_reason = _require_optional_text(payload["failure_reason"], "failure_reason")
    if status == "failed" and not failure_reason:
        raise SchemaValidationError("failed routing records require failure_reason")
    if status != "failed" and failure_reason:
        raise SchemaValidationError("failure_reason must be null unless status is failed")
    normalized["failure_reason"] = failure_reason

    scene_groups = payload["scene_groups"]
    if not isinstance(scene_groups, dict) or any(
        not isinstance(key, str) or not key or not isinstance(value, str) or not value
        for key, value in scene_groups.items()
    ):
        raise SchemaValidationError("scene_groups must map non-empty strings to non-empty strings")
    normalized["scene_groups"] = dict(sorted(scene_groups.items()))

    expert_names = payload["expert_names"]
    if not isinstance(expert_names, list) or not expert_names:
        raise SchemaValidationError("expert_names must be a non-empty list")
    if any(not isinstance(name, str) or not name for name in expert_names) or len(set(expert_names)) != len(
        expert_names
    ):
        raise SchemaValidationError("expert_names must contain unique non-empty strings")
    probabilities = payload["expert_probabilities"]
    if not isinstance(probabilities, list) or len(probabilities) != len(expert_names):
        raise SchemaValidationError("expert_probabilities length must match expert_names")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in probabilities):
        raise SchemaValidationError("expert_probabilities must contain numeric values")
    probabilities = [float(value) for value in probabilities]
    if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
        raise SchemaValidationError("expert_probabilities must be finite and non-negative")
    if status in EVIDENCE_STATUSES and not math.isclose(
        math.fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=probability_tolerance
    ):
        raise SchemaValidationError("expert_probabilities must sum to 1")
    normalized["expert_probabilities"] = probabilities

    selected = payload["selected_expert"]
    if status in EVIDENCE_STATUSES:
        if selected not in expert_names:
            raise SchemaValidationError("selected_expert must be one of expert_names")
        expected = expert_names[max(range(len(probabilities)), key=probabilities.__getitem__)]
        if selected != expected:
            raise SchemaValidationError("selected_expert must equal argmax(expert_probabilities)")
    elif selected is not None and selected not in expert_names:
        raise SchemaValidationError("selected_expert must be null or one of expert_names")

    top_k = _require_integer(payload["top_k"], "top_k")
    if status in EVIDENCE_STATUSES and not 1 <= top_k <= len(expert_names):
        raise SchemaValidationError("top_k must be within [1, num_experts] for evidence records")
    token_indices = payload["token_top1_indices"]
    if not isinstance(token_indices, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= len(expert_names)
        for value in token_indices
    ):
        raise SchemaValidationError("token_top1_indices must contain valid expert indices")
    spatial_shape = payload["spatial_shape"]
    if not isinstance(spatial_shape, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in spatial_shape
    ):
        raise SchemaValidationError("spatial_shape must contain positive integers")
    expected_tokens = math.prod(spatial_shape) if spatial_shape else 1
    if status in EVIDENCE_STATUSES and len(token_indices) != expected_tokens:
        raise SchemaValidationError("token_top1_indices length must equal product(spatial_shape)")

    normalized["expert_names"] = list(expert_names)
    normalized["selected_expert"] = selected
    normalized["top_k"] = top_k
    normalized["token_top1_indices"] = list(token_indices)
    normalized["spatial_shape"] = list(spatial_shape)
    return normalized


def load_json(path: str | Path) -> dict[str, Any]:
    """Load one UTF-8 JSON object."""
    target = Path(path)
    with target.open(encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(f"{target.name}: invalid JSON") from exc
    if not isinstance(payload, dict):
        raise SchemaValidationError(f"{path} must contain a JSON object")
    return payload


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load non-empty UTF-8 JSONL rows."""
    target = Path(path)
    rows = []
    with target.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaValidationError(f"{target.name}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise SchemaValidationError(f"{target.name}:{line_number}: record must be a JSON object")
            rows.append(row)
    return rows


def ensure_outputs_available(paths: Iterable[str | Path], *, overwrite: bool = False) -> None:
    """Reject duplicate or existing output paths unless overwrite is explicitly enabled."""
    targets = [Path(path) for path in paths]
    canonical = [str(target.resolve()) for target in targets]
    if len(set(canonical)) != len(canonical):
        raise SchemaValidationError("output paths must be distinct")
    existing = [target.name for target in targets if target.exists()]
    if existing and not overwrite:
        raise SchemaValidationError(f"output already exists; pass --overwrite to replace it: {sorted(existing)}")


def cli_error_message(error: Exception) -> str:
    """Return a concise CLI error that does not disclose private absolute paths."""
    if isinstance(error, FileNotFoundError):
        name = Path(error.filename).name if error.filename else "input"
        return f"input file not found: {name}"
    if isinstance(error, OSError):
        name = Path(error.filename).name if error.filename else "input/output"
        return f"filesystem error for {name}: {error.strerror or type(error).__name__}"
    return str(error)


def write_json(path: str | Path, payload: dict[str, Any], *, overwrite: bool = False) -> None:
    """Write deterministic, redacted UTF-8 JSON without silent replacement."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    safe = redact_sensitive(payload)
    mode = "w" if overwrite else "x"
    try:
        with target.open(mode, encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(safe, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    except FileExistsError as exc:
        raise SchemaValidationError(f"output already exists; pass --overwrite to replace it: {target.name}") from exc


def write_jsonl(path: str | Path, rows: list[dict[str, Any]], *, overwrite: bool = False) -> None:
    """Write deterministic, redacted UTF-8 JSONL without silent replacement."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(redact_sensitive(row), ensure_ascii=False, sort_keys=True) for row in rows]
    mode = "w" if overwrite else "x"
    try:
        with target.open(mode, encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines) + ("\n" if lines else ""))
    except FileExistsError as exc:
        raise SchemaValidationError(f"output already exists; pass --overwrite to replace it: {target.name}") from exc


__all__ = (
    "ANALYSIS_SCHEMA_VERSION",
    "EVIDENCE_STATUSES",
    "EXPERIMENT_MANIFEST_JSON_SCHEMA",
    "EXPERIMENT_MANIFEST_SCHEMA_VERSION",
    "FORMAL_STATUS",
    "REGISTRY_SCHEMA_VERSION",
    "ROUTING_RECORD_JSON_SCHEMA",
    "ROUTING_RECORD_SCHEMA_VERSION",
    "SchemaValidationError",
    "VALID_STATUSES",
    "canonical_payload_sha256",
    "cli_error_message",
    "ensure_outputs_available",
    "load_json",
    "load_jsonl",
    "sha256_file",
    "validate_experiment_manifest",
    "validate_routing_record",
    "with_manifest_checksum",
    "write_json",
    "write_jsonl",
)
