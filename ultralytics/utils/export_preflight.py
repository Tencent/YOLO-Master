"""Preflight checks for dynamic mixture and adapter export."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch.nn as nn

from ultralytics.nn.modules.routing_protocol import export_capabilities as default_export_capabilities
from ultralytics.utils.export_capabilities import (
    classify_routed_module,
    load_export_capability_matrix,
    normalize_export_format,
    resolve_export_capability,
    validate_export_capability_matrix,
)


@dataclass
class ExportDecision:
    module: str
    module_type: str
    module_family: str
    backend: str
    supported: bool
    strategy: str
    dense_fallback: bool
    requires_merge: bool
    known_error: str | None = None
    matrix_known_error: str | None = None


def _load_matrix(
    matrix: Mapping[str, Any] | None,
    matrix_path: str | Path | None,
) -> tuple[dict[str, Any], str]:
    """Load one validated matrix and return its reportable source."""
    if matrix is not None and matrix_path is not None:
        raise ValueError("matrix and matrix_path are mutually exclusive")
    if matrix is not None:
        validated = validate_export_capability_matrix(matrix)
        return validated, str(matrix.get("source", "<in-memory>"))
    loaded = load_export_capability_matrix(matrix_path)
    return loaded, str(loaded.pop("source"))


def _runtime_capabilities(module: nn.Module) -> dict[str, Any]:
    """Read a module declaration, falling back to the routing protocol defaults."""
    declaration = getattr(module, "export_capabilities", None)
    capabilities = declaration() if callable(declaration) else default_export_capabilities(module)
    if not isinstance(capabilities, Mapping):
        raise TypeError(f"{type(module).__name__}.export_capabilities() must return a mapping")
    return dict(capabilities)


def _join_reasons(*reasons: Any) -> str | None:
    """Join non-empty limitations without duplicating identical messages."""
    unique: list[str] = []
    for reason in reasons:
        if reason is None:
            continue
        text = str(reason).strip()
        if text and text not in unique:
            unique.append(text)
    return "; ".join(unique) or None


def _is_merged(module: nn.Module) -> bool:
    """Return whether a routed adapter has been materialized into its base layer."""
    merged = getattr(module, "merged", False)
    return bool(merged() if callable(merged) else merged)


def export_preflight(
    model: nn.Module,
    fmt: str,
    *,
    strict: bool = True,
    matrix: Mapping[str, Any] | None = None,
    matrix_path: str | Path | None = None,
) -> dict[str, Any]:
    """Scan routed modules and select dynamic, dense fallback, merged, or refusal."""
    capability_matrix, matrix_source = _load_matrix(matrix, matrix_path)
    fmt = normalize_export_format(fmt)
    decisions: list[ExportDecision] = []
    errors: list[str] = []

    format_capability = capability_matrix["formats"].get(fmt)
    if format_capability is None:
        format_error = f"export format {fmt!r} is not declared in the capability matrix"
    elif not format_capability["supported"]:
        format_error = _join_reasons(
            format_capability.get("known_error"),
            f"export format {fmt!r} is unsupported by matrix policy",
        )
    else:
        format_error = None

    routed_modules = [
        (name or "<root>", module, family)
        for name, module in model.named_modules()
        if (family := classify_routed_module(module)) is not None
    ]
    if format_capability is None and not routed_modules:
        errors.append(format_error)

    for name, module, family in routed_modules:
        if format_capability is None:
            policy = {
                "format": fmt,
                "family": family,
                "format_supported": False,
                "default_strategy": "refuse",
                "format_known_error": format_error,
                "supported": False,
                "dense_fallback": False,
                "requires_merge": False,
                "known_error": None,
            }
        else:
            policy = resolve_export_capability(capability_matrix, family, fmt)

        runtime = _runtime_capabilities(module)
        matrix_error = _join_reasons(policy.get("format_known_error"), policy.get("known_error"))
        requires_merge = bool(policy["requires_merge"] or runtime.get("requires_merge", False))
        runtime_supported = bool(runtime.get("supported", True))
        runtime_fallback = bool(runtime.get("export_safe_dense_fallback", False))
        dense_fallback = bool(policy["dense_fallback"] and runtime_fallback)
        merged = _is_merged(module)

        failure: str | None = None
        strategy = policy["default_strategy"]
        if format_error:
            failure = format_error
        elif not policy["supported"]:
            failure = _join_reasons(matrix_error, f"{family} is unsupported by matrix policy for {fmt}")
        elif not runtime_supported:
            failure = _join_reasons(runtime.get("known_error"), f"{type(module).__name__} refuses export")
        elif requires_merge and not merged:
            failure = _join_reasons(
                matrix_error,
                runtime.get("known_error"),
                f"{family} must be merged before {fmt} export",
            )
        elif merged:
            strategy = "merged"
        elif strategy == "dense_fallback" and not dense_fallback:
            failure = _join_reasons(
                matrix_error,
                runtime.get("known_error"),
                "dense fallback is not allowed by both matrix and runtime declarations",
            )
        elif strategy == "refuse":
            failure = _join_reasons(matrix_error, f"matrix policy refuses {family} export to {fmt}")

        if failure:
            strategy = "refuse"
            errors.append(f"{name}: {failure}")
        decision = ExportDecision(
            module=name,
            module_type=type(module).__name__,
            module_family=family,
            backend=fmt,
            supported=failure is None,
            strategy=strategy,
            dense_fallback=failure is None and strategy == "dense_fallback",
            requires_merge=requires_merge,
            known_error=failure or _join_reasons(matrix_error, runtime.get("known_error")),
            matrix_known_error=matrix_error,
        )
        decisions.append(decision)

    report = {
        "format": fmt,
        "supported": not errors,
        "matrix_schema_version": capability_matrix["schema_version"],
        "matrix_source": matrix_source,
        "decisions": [asdict(item) for item in decisions],
        "errors": errors,
    }
    if errors and strict:
        raise RuntimeError("Export preflight failed: " + "; ".join(errors))
    return report


__all__ = ["ExportDecision", "export_preflight"]
