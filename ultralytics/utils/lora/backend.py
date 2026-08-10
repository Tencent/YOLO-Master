"""Unified adapter backend discovery and orchestration.

Backends intentionally stay thin: standard LoRA delegates to the existing
PEFT/fallback IO, while MoLoRA delegates to its versioned wrapper checkpoint.
The orchestration layer provides one stable API for the model and trainer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class AdapterBackend(Protocol):
    """Contract implemented by adapter serialization backends."""

    name: str

    def can_handle(self, model: nn.Module) -> bool: ...

    def save(self, model: nn.Module, path: str | Path) -> bool: ...

    def load(self, model: nn.Module, path: str | Path) -> bool: ...

    def merge(self, model: nn.Module, **kwargs: Any) -> bool: ...

    def metadata(self, model: nn.Module) -> dict[str, Any]: ...


def _unwrap(model: nn.Module) -> nn.Module:
    while hasattr(model, "module") and isinstance(getattr(model, "module"), nn.Module):
        model = model.module
    return model


class StandardLoRABackend:
    name = "lora"

    def can_handle(self, model: nn.Module) -> bool:
        model = _unwrap(model)
        return bool(getattr(model, "lora_enabled", False)) and not bool(getattr(model, "molora_enabled", False))

    def save(self, model: nn.Module, path: str | Path) -> bool:
        from .io import save_lora_adapters

        return save_lora_adapters(_unwrap(model), path)

    def load(self, model: nn.Module, path: str | Path) -> bool:
        from .io import load_lora_adapters

        return load_lora_adapters(_unwrap(model), path)

    def merge(self, model: nn.Module, **kwargs: Any) -> bool:
        from .io import merge_lora_weights

        return merge_lora_weights(_unwrap(model))

    def metadata(self, model: nn.Module) -> dict[str, Any]:
        model = _unwrap(model)
        return {
            "backend": self.name,
            "variant": getattr(model, "lora_variant", "lora"),
            "schema_version": 1,
            "target_modules": list(getattr(model, "lora_target_modules", [])),
            "runtime_metadata": getattr(model, "lora_runtime_metadata", {}),
            "merge_mode": "exact",
        }


class MoLoRABackend:
    name = "molora"

    def can_handle(self, model: nn.Module) -> bool:
        model = _unwrap(model)
        return bool(getattr(model, "molora_enabled", False)) or any(
            module.__class__.__name__ == "MoLoRALayer" for module in model.modules()
        )

    def save(self, model: nn.Module, path: str | Path) -> bool:
        model = _unwrap(model)
        from ultralytics.nn.peft.molora.model import MoLoRAModel, _molora_structure

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if isinstance(model, MoLoRAModel):
            model.save_checkpoint(str(path / "molora_adapter.pt"))
            wrapped = model.model
            config = model.config
        else:
            config = getattr(model, "molora_config", {})
            state = {
                "schema_version": 1,
                "format": "molora_adapter",
                "config": getattr(config, "__dict__", dict(config) if isinstance(config, dict) else {}),
                "structure": _molora_structure(model),
                "model_fingerprint": _model_fingerprint(model),
                "merge_metadata": {
                    name: dict(getattr(layer, "_merge_metadata", {}))
                    for name, layer in model.named_modules()
                    if layer.__class__.__name__ == "MoLoRALayer"
                },
                "state_dict": {
                    k: v
                    for k, v in model.state_dict().items()
                    if any(
                        p in k
                        for p in ("lora_A", "lora_B", "router", "_step_count", "_usage_ema", "_domain_active_mask")
                    )
                },
            }
            torch.save(state, path / "molora_adapter.pt")
            wrapped = model
        payload = self.metadata(wrapped)
        payload["config"] = getattr(config, "__dict__", dict(config) if isinstance(config, dict) else {})
        (path / "runtime_metadata.json").write_text(json.dumps(payload, indent=2, default=str))
        return True

    def load(self, model: nn.Module, path: str | Path) -> bool:
        model = _unwrap(model)
        from ultralytics.nn.peft.molora.model import MoLoRAModel

        checkpoint = Path(path)
        if checkpoint.is_dir():
            checkpoint = checkpoint / "molora_adapter.pt"
        if isinstance(model, MoLoRAModel):
            model.load_checkpoint(str(checkpoint))
            return True
        raise ValueError("MoLoRA backend load requires a MoLoRAModel wrapper for structural validation")

    def merge(self, model: nn.Module, **kwargs: Any) -> bool:
        mode = kwargs.get("mode", "ema")
        if mode not in {"ema", "uniform", "calibrated"}:
            raise ValueError("MoLoRA merge mode must be 'ema', 'uniform', or 'calibrated'")
        model = _unwrap(model)
        from ultralytics.nn.peft.molora.model import MoLoRAModel, merge_molora_layers

        if isinstance(model, MoLoRAModel):
            model.merge(
                mode=mode,
                sync_ema=kwargs.get("sync_ema", False),
                merge_authority=kwargs.get("merge_authority"),
                calibration_data=kwargs.get("calibration_data"),
                calibration=kwargs.get("calibration"),
                max_batches=kwargs.get("max_batches"),
                forward_fn=kwargs.get("forward_fn"),
                verify_fidelity=kwargs.get("verify_fidelity", False),
                fidelity_tolerance=kwargs.get("fidelity_tolerance"),
            )
            return True
        merge_molora_layers(
            model,
            mode=mode,
            sync_ema=kwargs.get("sync_ema", False),
            merge_authority=kwargs.get("merge_authority"),
            calibration_data=kwargs.get("calibration_data"),
            calibration=kwargs.get("calibration"),
            max_batches=kwargs.get("max_batches"),
            forward_fn=kwargs.get("forward_fn"),
            verify_fidelity=kwargs.get("verify_fidelity", False),
            fidelity_tolerance=kwargs.get("fidelity_tolerance"),
        )
        return True

    def metadata(self, model: nn.Module) -> dict[str, Any]:
        model = _unwrap(model)
        config = getattr(model, "molora_config", None)
        merge_records = [
            {"name": name, **dict(getattr(module, "_merge_metadata", {}))}
            for name, module in model.named_modules()
            if module.__class__.__name__ == "MoLoRALayer"
        ]
        return {
            "backend": self.name,
            "variant": "molora",
            "schema_version": 1,
            "target_modules": list(getattr(config, "target_modules", None) or []) if config is not None else [],
            "num_experts": getattr(config, "num_experts", None),
            "top_k": getattr(config, "top_k", None),
            "router_type": getattr(config, "router_type", None),
            "model_fingerprint": _model_fingerprint(model),
            "placement_plan": getattr(model, "molora_placement_plan", None),
            "merge_mode": "dynamic",
            "exact_merge": False,
            "merge_records": merge_records,
            "publishable_merge": bool(merge_records)
            and all(
                record.get("mode") == "calibrated" and bool(record.get("calibration_fingerprint"))
                for record in merge_records
            ),
        }


def _model_fingerprint(model: nn.Module) -> str:
    """Build a stable graph/parameter-shape binding for adapter artifacts."""

    import hashlib

    entries = []
    for name, module in model.named_modules():
        if name:
            entries.append((name, module.__class__.__qualname__))
    for name, parameter in model.named_parameters():
        entries.append((f"param:{name}", tuple(parameter.shape), str(parameter.dtype)))
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_BACKENDS = (MoLoRABackend(), StandardLoRABackend())


def discover_adapter_backend(model: nn.Module, *, required: bool = False) -> AdapterBackend | None:
    """Discover the most specific active adapter backend."""

    for backend in _BACKENDS:
        if backend.can_handle(model):
            return backend
    if required:
        raise ValueError("No active adapter backend found")
    return None


def save_adapters(model: nn.Module, path: str | Path) -> bool:
    backend = discover_adapter_backend(model)
    return False if backend is None else backend.save(model, path)


def load_adapters(model: nn.Module, path: str | Path) -> bool:
    path = Path(path)
    metadata_paths = (
        (path / "runtime_metadata.json", path / "fallback_meta.json", path / "adapter_config.json")
        if path.is_dir()
        else (path.with_name("runtime_metadata.json"),)
    )
    payload = {}
    for metadata_path in metadata_paths:
        if metadata_path.exists():
            payload = json.loads(metadata_path.read_text())
            if payload.get("backend"):
                break
    backend_name = {"fallback": "lora", "peft": "lora"}.get(payload.get("backend"), payload.get("backend"))
    if backend_name is None and (path / "molora_adapter.pt" if path.is_dir() else path).name == "molora_adapter.pt":
        backend_name = "molora"
    backend = next((item for item in _BACKENDS if item.name == backend_name), discover_adapter_backend(model))
    if backend is None:
        raise ValueError(f"Cannot determine adapter backend for {path}")
    return backend.load(model, path)


def merge_adapters(model: nn.Module, **kwargs: Any) -> bool:
    backend = discover_adapter_backend(model, required=True)
    return backend.merge(model, **kwargs)


def adapter_metadata(model: nn.Module) -> dict[str, Any]:
    backend = discover_adapter_backend(model)
    return {} if backend is None else backend.metadata(model)


__all__ = [
    "AdapterBackend",
    "adapter_metadata",
    "discover_adapter_backend",
    "load_adapters",
    "merge_adapters",
    "save_adapters",
]
