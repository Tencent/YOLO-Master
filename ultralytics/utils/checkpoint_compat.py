"""Read-only compatibility inspection and conversion for legacy YOLO-Master artifacts."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ultralytics import __version__
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import unwrap_model

ADAPTER_TOKENS = ("lora_", "hada_", "lokr_", "ia3_", "boft_", "oft_")


@dataclass
class CheckpointCompatibilityReport:
    """Machine-readable audit emitted for every inspected or converted artifact."""

    source: str
    artifact_type: str
    source_version: str | None = None
    target_version: str = __version__
    source_model_class: str | None = None
    target_model_class: str | None = None
    source_graph: dict[str, Any] = field(default_factory=dict)
    target_graph: dict[str, Any] = field(default_factory=dict)
    adapter_metadata: dict[str, Any] = field(default_factory=dict)
    key_remap: dict[str, str] = field(default_factory=dict)
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    shape_mismatches: list[str] = field(default_factory=list)
    semantic_risks: list[str] = field(default_factory=list)
    state_reports: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _qualified_name(value: Any) -> str | None:
    return None if value is None else f"{value.__class__.__module__}.{value.__class__.__name__}"


def _model_head(model: nn.Module) -> nn.Module | None:
    sequence = getattr(unwrap_model(model), "model", None)
    if isinstance(sequence, (nn.Sequential, nn.ModuleList, list, tuple)) and sequence:
        return sequence[-1]
    return None


def graph_metadata(model: nn.Module | None) -> dict[str, Any]:
    """Describe graph properties that must never be changed silently during conversion."""
    if not isinstance(model, nn.Module):
        return {}
    model = unwrap_model(model)
    head = _model_head(model)
    modules = {module.__class__.__name__ for module in model.modules()}
    return {
        "model_class": _qualified_name(model),
        "head_class": _qualified_name(head),
        "task": getattr(model, "task", None),
        "reg_max": getattr(head, "reg_max", None),
        "end2end": getattr(head, "end2end", getattr(model, "end2end", None)),
        "one2many": hasattr(head, "one2many"),
        "one2one": hasattr(head, "one2one"),
        "routed_families": sorted(
            family
            for family, names in {
                "moe": {"ES_MOE", "DyMoEBlock", "DyC2f", "A2C2fMoE", "ABlockMoE"},
                "moa": {"MoABlock", "C2fMoA", "NeckMoAFusion"},
                "mot": {"MoTBlock", "C2fMoT"},
                "molora": {"MoLoRALayer", "MoLoRAModel"},
            }.items()
            if modules & names
        ),
    }


def checkpoint_runtime_metadata(model: nn.Module | None) -> dict[str, Any]:
    """Build additive metadata for newly saved checkpoints."""
    metadata = {"schema_version": 1, "graph": graph_metadata(model)}
    if isinstance(model, nn.Module):
        from ultralytics.utils.lora import adapter_metadata

        adapter = adapter_metadata(unwrap_model(model))
        if adapter:
            metadata["adapter"] = adapter
    return metadata


def _read_adapter_metadata(path: Path) -> dict[str, Any]:
    candidates = (
        path / "runtime_metadata.json",
        path / "fallback_meta.json",
        path / "adapter_config.json",
    )
    for candidate in candidates:
        if candidate.exists():
            try:
                payload = json.loads(candidate.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                return payload
    return {}


def _is_molora_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("format") == "molora_adapter" and "state_dict" in payload


def _artifact_type(path: Path, payload: Any = None) -> str:
    if path.is_dir():
        metadata = _read_adapter_metadata(path)
        if metadata.get("backend") == "molora" or (path / "molora_adapter.pt").exists():
            return "molora_adapter"
        return "adapter_directory"
    if _is_molora_payload(payload):
        return "molora_adapter"
    return "full_checkpoint"


def inspect_checkpoint_artifact(source: str | Path) -> CheckpointCompatibilityReport:
    """Inspect a full checkpoint, adapter directory, or MoLoRA checkpoint without modifying it."""
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint artifact not found: {path}")
    if path.is_dir():
        metadata = _read_adapter_metadata(path)
        return CheckpointCompatibilityReport(
            source=str(path),
            artifact_type=_artifact_type(path),
            source_version=str(metadata.get("source_version")) if metadata.get("source_version") else None,
            adapter_metadata=metadata,
        )

    try:
        payload = torch_load(path, map_location="cpu", weights_only=False)
    except (AttributeError, ImportError, ModuleNotFoundError):
        payload = None
    if _is_molora_payload(payload):
        return CheckpointCompatibilityReport(
            source=str(path),
            artifact_type="molora_adapter",
            source_version=str(payload.get("source_version")) if payload.get("source_version") else None,
            adapter_metadata={
                "backend": "molora",
                "schema_version": payload.get("schema_version"),
                "config": payload.get("config", {}),
                "structure": payload.get("structure", []),
            },
        )

    from ultralytics.nn.tasks import torch_safe_load

    checkpoint, resolved = torch_safe_load(path)
    source_model = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    selected = checkpoint.get("ema") or source_model if isinstance(checkpoint, dict) else None
    runtime = checkpoint.get("mixture_checkpoint", {}) if isinstance(checkpoint, dict) else {}
    adapter = runtime.get("adapter", {}) if isinstance(runtime, dict) else {}
    if isinstance(selected, nn.Module) and not adapter:
        from ultralytics.utils.lora import adapter_metadata

        adapter = adapter_metadata(unwrap_model(selected))
    return CheckpointCompatibilityReport(
        source=str(resolved),
        artifact_type="full_checkpoint",
        source_version=str(checkpoint.get("version")) if checkpoint.get("version") else None,
        source_model_class=_qualified_name(selected),
        source_graph=graph_metadata(selected),
        adapter_metadata=adapter,
    )


def _key_candidates(key: str) -> list[str]:
    candidates = [key]
    prefixes = ("module.", "_orig_mod.", "model.module.")
    changed = True
    while changed:
        changed = False
        for candidate in tuple(candidates):
            for prefix in prefixes:
                if candidate.startswith(prefix):
                    stripped = candidate[len(prefix) :]
                    if stripped not in candidates:
                        candidates.append(stripped)
                        changed = True
    return candidates


def remap_checkpoint_state(
    source_state: dict[str, torch.Tensor], target_state: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, str], list[str], list[str], list[str]]:
    """Map known wrapper-prefix changes while rejecting shape changes."""
    mapped: dict[str, torch.Tensor] = {}
    key_remap: dict[str, str] = {}
    shape_mismatches: list[str] = []
    unexpected: list[str] = []
    for source_key, value in source_state.items():
        target_key = next((key for key in _key_candidates(source_key) if key in target_state), None)
        if target_key is None:
            unexpected.append(source_key)
            continue
        if tuple(value.shape) != tuple(target_state[target_key].shape):
            shape_mismatches.append(
                f"{source_key} -> {target_key}: {tuple(value.shape)} != {tuple(target_state[target_key].shape)}"
            )
            continue
        mapped[target_key] = value
        if target_key != source_key:
            key_remap[source_key] = target_key
    missing = sorted(set(target_state) - set(mapped))
    return mapped, key_remap, missing, sorted(unexpected), sorted(shape_mismatches)


def _head_risks(source_graph: dict[str, Any], target_graph: dict[str, Any]) -> list[str]:
    risks = []
    for key in ("head_class", "reg_max", "end2end", "one2many", "one2one"):
        source_value, target_value = source_graph.get(key), target_graph.get(key)
        if source_value != target_value and source_value is not None and target_value is not None:
            risks.append(f"{key} differs: source={source_value!r}, target={target_value!r}")
    return risks


def load_compatible_checkpoint(
    model: nn.Module,
    source: str | Path,
    *,
    use_ema: bool = True,
    allow_head_mismatch: bool = False,
) -> CheckpointCompatibilityReport:
    """Load an artifact into a caller-provided graph and return a complete compatibility audit."""
    path = Path(source).expanduser().resolve()
    report = inspect_checkpoint_artifact(path)
    target = unwrap_model(model)
    report.target_model_class = _qualified_name(target)
    report.target_graph = graph_metadata(target)

    if report.artifact_type in {"adapter_directory", "molora_adapter"}:
        from ultralytics.utils.lora import load_adapters

        if not load_adapters(target, path):
            raise RuntimeError(f"Adapter load failed for {path}")
        report.adapter_metadata = checkpoint_runtime_metadata(target).get("adapter", report.adapter_metadata)
        return report

    from ultralytics.nn.tasks import torch_safe_load

    checkpoint, _ = torch_safe_load(path)
    source_model = checkpoint.get("ema") if use_ema and isinstance(checkpoint.get("ema"), nn.Module) else checkpoint.get("model")
    if not isinstance(source_model, nn.Module):
        raise TypeError(f"Full checkpoint {path} has no loadable model or EMA module")
    report.source_model_class = _qualified_name(source_model)
    report.source_graph = graph_metadata(source_model)
    report.semantic_risks.extend(_head_risks(report.source_graph, report.target_graph))
    if report.semantic_risks and not allow_head_mismatch:
        raise ValueError("Checkpoint graph mismatch requires allow_head_mismatch=True: " + "; ".join(report.semantic_risks))

    source_state, target_state = source_model.float().state_dict(), target.state_dict()
    mapped, remap, missing, unexpected, mismatches = remap_checkpoint_state(source_state, target_state)
    source_has_adapters = any(token in key for key in source_state for token in ADAPTER_TOKENS)
    target_has_adapters = any(token in key for key in target_state for token in ADAPTER_TOKENS)
    if source_has_adapters != target_has_adapters:
        raise ValueError("Adapter topology differs between source checkpoint and target graph; merge or attach adapters first")
    target.load_state_dict(mapped, strict=False)
    report.key_remap = remap
    report.missing_keys = missing
    report.unexpected_keys = unexpected
    report.shape_mismatches = mismatches
    if not mapped:
        raise ValueError("Checkpoint conversion found zero compatible tensors")
    return report


def convert_checkpoint_artifact(
    source: str | Path,
    destination: str | Path,
    *,
    target_model: nn.Module | None = None,
    allow_head_mismatch: bool = False,
) -> CheckpointCompatibilityReport:
    """Write a converted copy and never modify the source artifact in place."""
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if source_path == destination_path:
        raise ValueError("Checkpoint conversion is read-only; destination must differ from source")
    report = inspect_checkpoint_artifact(source_path)

    if source_path.is_dir():
        if destination_path.exists():
            raise FileExistsError(f"Conversion destination already exists: {destination_path}")
        shutil.copytree(source_path, destination_path)
        report_path = destination_path / "checkpoint_compatibility.json"
        report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
        return report

    try:
        payload = torch_load(source_path, map_location="cpu", weights_only=False)
    except (AttributeError, ImportError, ModuleNotFoundError):
        payload = None
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if _is_molora_payload(payload):
        converted = deepcopy(payload)
        converted["source_version"] = report.source_version
        converted["target_version"] = __version__
        converted["checkpoint_compat"] = report.to_dict()
        torch.save(converted, destination_path)
        return report

    from ultralytics.nn.tasks import torch_safe_load

    checkpoint, _ = torch_safe_load(source_path)
    converted = deepcopy(checkpoint)
    if target_model is not None:
        online = deepcopy(unwrap_model(target_model))
        online_report = load_compatible_checkpoint(
            online, source_path, use_ema=False, allow_head_mismatch=allow_head_mismatch
        )
        report = online_report
        report.state_reports["model"] = {
            "key_remap": online_report.key_remap,
            "missing_keys": online_report.missing_keys,
            "unexpected_keys": online_report.unexpected_keys,
            "shape_mismatches": online_report.shape_mismatches,
        }
        converted["model"] = online
        if isinstance(checkpoint.get("ema"), nn.Module):
            ema = deepcopy(unwrap_model(target_model))
            ema_report = load_compatible_checkpoint(
                ema, source_path, use_ema=True, allow_head_mismatch=allow_head_mismatch
            )
            report.state_reports["ema"] = {
                "key_remap": ema_report.key_remap,
                "missing_keys": ema_report.missing_keys,
                "unexpected_keys": ema_report.unexpected_keys,
                "shape_mismatches": ema_report.shape_mismatches,
            }
            converted["ema"] = ema
    converted["version"] = __version__
    converted["mixture_checkpoint"] = checkpoint_runtime_metadata(converted.get("model"))
    converted["checkpoint_compat"] = report.to_dict()
    torch.save(converted, destination_path)
    return report


__all__ = (
    "CheckpointCompatibilityReport",
    "checkpoint_runtime_metadata",
    "convert_checkpoint_artifact",
    "graph_metadata",
    "inspect_checkpoint_artifact",
    "load_compatible_checkpoint",
    "remap_checkpoint_state",
)
