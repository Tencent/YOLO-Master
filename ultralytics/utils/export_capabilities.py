"""Canonical export capability matrix loading, validation, and rendering."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch.nn as nn
import yaml

from ultralytics.utils import ROOT


DEFAULT_EXPORT_CAPABILITY_MATRIX = ROOT / "cfg/export-capability-matrix.yaml"
REQUIRED_MODULES = frozenset({"MoE", "MoA", "MoT", "MoLoRA"})
REQUIRED_FORMAT_FIELDS = frozenset({"supported", "default", "known_error"})
REQUIRED_MODULE_FIELDS = frozenset({"supported", "dense_fallback", "requires_merge", "known_error"})
VALID_STRATEGIES = frozenset({"dynamic", "dense_fallback", "refuse"})


def normalize_export_format(fmt: str) -> str:
    """Normalize public exporter aliases to matrix keys."""
    value = str(fmt).strip().lower()
    aliases = {
        "-": "pytorch",
        "pt": "pytorch",
        "tensorrt": "engine",
        "trt": "engine",
        "mlmodel": "coreml",
        "mlpackage": "coreml",
        "mlprogram": "coreml",
        "apple": "coreml",
        "ios": "coreml",
        "tflite": "litert",
    }
    return aliases.get(value, value)


def validate_export_capability_matrix(matrix: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a detached matrix dictionary."""
    if not isinstance(matrix, Mapping):
        raise ValueError("export capability matrix must be a mapping")
    if matrix.get("schema_version") != 1:
        raise ValueError("export capability matrix schema_version must be 1")
    formats = matrix.get("formats")
    modules = matrix.get("modules")
    if not isinstance(formats, Mapping) or not formats:
        raise ValueError("export capability matrix formats must be a non-empty mapping")
    if not isinstance(modules, Mapping):
        raise ValueError("export capability matrix modules must be a mapping")

    missing_modules = sorted(REQUIRED_MODULES - set(modules))
    if missing_modules:
        raise ValueError(f"export capability matrix missing module families: {missing_modules}")

    for name, capability in formats.items():
        if not isinstance(capability, Mapping):
            raise ValueError(f"format {name!r} capability must be a mapping")
        missing = sorted(REQUIRED_FORMAT_FIELDS - set(capability))
        if missing:
            raise ValueError(f"format {name!r} missing required fields: {missing}")
        if not isinstance(capability["supported"], bool):
            raise ValueError(f"format {name!r} supported must be bool")
        if capability["default"] not in VALID_STRATEGIES:
            raise ValueError(f"format {name!r} has invalid default strategy {capability['default']!r}")
        if not capability["supported"] and capability["default"] != "refuse":
            raise ValueError(f"unsupported format {name!r} must use the refuse strategy")
        if capability["known_error"] is not None and not isinstance(capability["known_error"], str):
            raise ValueError(f"format {name!r} known_error must be a string or null")

    for family in REQUIRED_MODULES:
        capability = modules[family]
        if not isinstance(capability, Mapping):
            raise ValueError(f"module {family!r} capability must be a mapping")
        missing = sorted(REQUIRED_MODULE_FIELDS - set(capability))
        if missing:
            raise ValueError(f"module {family!r} missing required fields: {missing}")
        for field in ("supported", "dense_fallback", "requires_merge"):
            if not isinstance(capability[field], bool):
                raise ValueError(f"module {family!r} {field} must be bool")
        if capability["known_error"] is not None and not isinstance(capability["known_error"], str):
            raise ValueError(f"module {family!r} known_error must be a string or null")
        overrides = capability.get("formats", {})
        if not isinstance(overrides, Mapping):
            raise ValueError(f"module {family!r} formats override must be a mapping")
        unknown = sorted(set(overrides) - set(formats))
        if unknown:
            raise ValueError(f"module {family!r} has overrides for unknown formats: {unknown}")
        for fmt, override in overrides.items():
            if not isinstance(override, Mapping):
                raise ValueError(f"module {family!r} format {fmt!r} override must be a mapping")
            invalid = sorted(set(override) - REQUIRED_MODULE_FIELDS)
            if invalid:
                raise ValueError(f"module {family!r} format {fmt!r} has invalid fields: {invalid}")
            for field in ("supported", "dense_fallback", "requires_merge"):
                if field in override and not isinstance(override[field], bool):
                    raise ValueError(f"module {family!r} format {fmt!r} {field} must be bool")
            known_error = override.get("known_error")
            if known_error is not None and not isinstance(known_error, str):
                raise ValueError(f"module {family!r} format {fmt!r} known_error must be a string or null")
    return deepcopy(dict(matrix))


def load_export_capability_matrix(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the canonical export capability matrix."""
    source = Path(path) if path is not None else DEFAULT_EXPORT_CAPABILITY_MATRIX
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    matrix = validate_export_capability_matrix(data)
    matrix["source"] = str(source)
    return matrix


def classify_routed_module(module: nn.Module) -> str | None:
    """Return the canonical routed module family for a model module."""
    name = module.__class__.__name__
    if name in {"MoABlock", "C2fMoA", "NeckMoAFusion"}:
        return "MoA"
    if name in {"MoTBlock", "C2fMoT"}:
        return "MoT"
    if name in {"MoLoRALayer", "MoLoRAMoEAwareLayer"}:
        return "MoLoRA"
    if name in {"DyMoEBlock", "DyC2f"}:
        return "MoE"
    try:
        from ultralytics.nn.modules.moe import EXPERIMENTAL_MOE_CLASSES, LEGACY_MOE_CLASSES, STABLE_MOE_CLASSES
        from ultralytics.nn.modules.moe.utils import is_core_moe_block

        if name in STABLE_MOE_CLASSES | EXPERIMENTAL_MOE_CLASSES | LEGACY_MOE_CLASSES or is_core_moe_block(module):
            return "MoE"
    except (ImportError, AttributeError):
        pass
    return None


def resolve_export_capability(matrix: Mapping[str, Any], family: str, fmt: str) -> dict[str, Any]:
    """Resolve effective format and module policy for one routed family."""
    fmt = normalize_export_format(fmt)
    formats = matrix["formats"]
    if fmt not in formats:
        raise ValueError(f"export format {fmt!r} is not declared in the capability matrix")
    if family not in matrix["modules"]:
        raise ValueError(f"routed module family {family!r} is not declared in the capability matrix")

    format_capability = dict(formats[fmt])
    module_capability = dict(matrix["modules"][family])
    override = module_capability.pop("formats", {}).get(fmt, {})
    module_capability.update(override)
    return {
        "format": fmt,
        "family": family,
        "format_supported": format_capability["supported"],
        "default_strategy": format_capability["default"],
        "format_known_error": format_capability.get("known_error"),
        **module_capability,
    }


def render_export_capability_markdown(matrix: Mapping[str, Any]) -> str:
    """Render deterministic governance documentation from the matrix."""
    matrix = validate_export_capability_matrix(matrix)
    lines = [
        "# Export Capability Matrix",
        "",
        "> Generated from `ultralytics/cfg/export-capability-matrix.yaml`. Do not edit manually.",
        "",
        "## Formats",
        "",
        "| Format | Supported | Default strategy | Known limitation |",
        "|---|---:|---|---|",
    ]
    for fmt, capability in matrix["formats"].items():
        limitation = capability.get("known_error") or "-"
        lines.append(
            f"| `{fmt}` | {'yes' if capability['supported'] else 'no'} | `{capability['default']}` | {limitation} |"
        )

    lines.extend(
        [
            "",
            "## Routed Modules",
            "",
            "| Module family | Supported | Dense fallback | Requires merge | Known limitation |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for family in sorted(matrix["modules"]):
        capability = matrix["modules"][family]
        limitation = capability.get("known_error") or "-"
        lines.append(
            f"| `{family}` | {'yes' if capability['supported'] else 'no'} | "
            f"{'yes' if capability['dense_fallback'] else 'no'} | "
            f"{'yes' if capability['requires_merge'] else 'no'} | {limitation} |"
        )

    lines.extend(
        [
            "",
            "## Effective Policies",
            "",
            "The effective policy intersects each format default with the module policy. Runtime preflight may refuse a "
            "declared dense fallback when a concrete module does not advertise a safe implementation.",
            "",
            "| Module family | Format | Effective strategy | Dense fallback | Requires merge | Known limitation |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for family in sorted(matrix["modules"]):
        for fmt in matrix["formats"]:
            capability = resolve_export_capability(matrix, family, fmt)
            supported = capability["format_supported"] and capability["supported"]
            strategy = capability["default_strategy"] if supported else "refuse"
            limitations = [capability.get("format_known_error"), capability.get("known_error")]
            limitation = "; ".join(dict.fromkeys(item for item in limitations if item)) or "-"
            lines.append(
                f"| `{family}` | `{fmt}` | `{strategy}` | "
                f"{'yes' if supported and capability['dense_fallback'] else 'no'} | "
                f"{'yes' if capability['requires_merge'] else 'no'} | {limitation} |"
            )

    lines.extend(
        [
            "",
            "## Evidence Boundary",
            "",
            "A `supported` entry means preflight has a declared execution strategy. It does not imply full-model or hardware-specific numerical verification. Consult `model-registry.yaml` for executable evidence status.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_EXPORT_CAPABILITY_MATRIX",
    "classify_routed_module",
    "load_export_capability_matrix",
    "normalize_export_format",
    "render_export_capability_markdown",
    "resolve_export_capability",
    "validate_export_capability_matrix",
]
