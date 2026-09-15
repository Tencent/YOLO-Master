"""Non-invasive capture of true spatial router tensors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .geometry import validate_probability_grid


def routing_metric_fields(weights: np.ndarray) -> dict[str, np.ndarray]:
    """Derive bounded token diagnostics from a truthful ``[E,H,W]`` probability grid."""

    values = np.asarray(weights, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"expected [E,H,W], got shape={list(values.shape)}")
    validate_probability_grid(values[None, ...])
    expert_count = values.shape[0]
    safe = np.clip(values, np.finfo(np.float32).tiny, 1.0)
    entropy = -np.sum(values * np.log(safe), axis=0) / np.log(float(expert_count))
    ordered = np.sort(values, axis=0)
    margin = ordered[-1] - ordered[-2] if expert_count > 1 else np.ones_like(ordered[-1])
    return {
        "normalized_entropy": np.clip(entropy, 0.0, 1.0).astype(np.float32),
        "top1_margin": np.clip(margin, 0.0, 1.0).astype(np.float32),
    }


def routing_diagnostics(weights: np.ndarray) -> dict[str, Any]:
    """Summarize spatial routing without discarding the raw grid evidence."""

    values = np.asarray(weights, dtype=np.float32)
    fields = routing_metric_fields(values)
    dominant = np.argmax(values, axis=0)
    token_count = int(dominant.size)
    horizontal = np.abs(np.diff(values, axis=2)).reshape(-1)
    vertical = np.abs(np.diff(values, axis=1)).reshape(-1)
    variation_values = np.concatenate([horizontal, vertical])

    def stats(field: np.ndarray) -> dict[str, float]:
        return {
            "min": float(field.min()),
            "max": float(field.max()),
            "mean": float(field.mean()),
        }

    return {
        "token_count": token_count,
        "mean_expert_probability": [float(item) for item in values.mean(axis=(1, 2))],
        "dominant_token_count": [int(np.sum(dominant == expert)) for expert in range(values.shape[0])],
        "dominant_token_fraction": [
            float(np.sum(dominant == expert) / token_count) for expert in range(values.shape[0])
        ],
        "active_dominant_experts": int(np.unique(dominant).size),
        "normalized_entropy": stats(fields["normalized_entropy"]),
        "top1_margin": stats(fields["top1_margin"]),
        "neighbor_probability_l1_mean": float(variation_values.mean()) if variation_values.size else 0.0,
    }


def tensor_shapes(value: Any) -> list[list[int]]:
    shapes: list[list[int]] = []
    if hasattr(value, "shape"):
        shapes.append([int(item) for item in value.shape])
    elif isinstance(value, dict):
        for child in value.values():
            shapes.extend(tensor_shapes(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            shapes.extend(tensor_shapes(child))
    return shapes


def flatten_tensors(value: Any) -> list[Any]:
    tensors: list[Any] = []
    if hasattr(value, "detach") and hasattr(value, "shape"):
        tensors.append(value)
    elif isinstance(value, dict):
        for key in sorted(value):
            tensors.extend(flatten_tensors(value[key]))
    elif isinstance(value, (list, tuple)):
        for child in value:
            tensors.extend(flatten_tensors(child))
    return tensors


def max_output_delta(reference: Any, observed: Any) -> float:
    """Return max tensor delta while enforcing identical nested tensor structure."""

    left = flatten_tensors(reference)
    right = flatten_tensors(observed)
    if len(left) != len(right):
        raise ValueError(f"output tensor count changed: {len(left)} != {len(right)}")
    maximum = 0.0
    for index, (a, b) in enumerate(zip(left, right)):
        if tuple(a.shape) != tuple(b.shape):
            raise ValueError(f"output tensor {index} shape changed: {tuple(a.shape)} != {tuple(b.shape)}")
        if a.numel():
            maximum = max(maximum, float((a.detach().float() - b.detach().float()).abs().max().cpu()))
    return maximum


@dataclass
class SpatialRecord:
    family: str
    module_name: str
    module_type: str
    weights: np.ndarray
    logits: np.ndarray
    indices: np.ndarray | None
    validation: dict[str, Any]


@dataclass
class SpatialRouterCollector:
    family: str
    router_class: str
    records: list[SpatialRecord] = field(default_factory=list)
    handles: list[Any] = field(default_factory=list)
    registered_names: list[str] = field(default_factory=list)

    def register(self, model: Any) -> list[str]:
        if self.handles:
            raise RuntimeError("collector is already registered")
        for name, module in model.named_modules():
            if module.__class__.__name__ != self.router_class:
                continue

            def capture(current_module: Any, inputs: Any, output: Any, *, module_name: str = name) -> None:
                del inputs
                if not isinstance(output, (list, tuple)):
                    raise TypeError(f"{module_name} did not expose the expected router tuple")
                if self.family == "mot" and len(output) >= 3:
                    weights, indices, logits = output[:3]
                elif self.family == "moa" and len(output) >= 2:
                    weights, logits = output[:2]
                    indices = None
                else:
                    raise RuntimeError(f"unsupported router output for family={self.family}: length={len(output)}")
                weight_array = weights.detach().float().cpu().numpy()
                logit_array = logits.detach().float().cpu().numpy()
                index_array = indices.detach().cpu().numpy() if indices is not None else None
                validation = validate_probability_grid(weight_array)
                if list(logit_array.shape) != list(weight_array.shape):
                    raise RuntimeError(
                        f"logit/probability shape mismatch at {module_name}: {logit_array.shape} != {weight_array.shape}"
                    )
                if index_array is not None and tuple(index_array.shape[0:1] + index_array.shape[2:]) != (
                    weight_array.shape[0],
                    weight_array.shape[2],
                    weight_array.shape[3],
                ):
                    raise RuntimeError(f"Top-K index grid does not align with weights at {module_name}")
                if index_array is not None:
                    if int(index_array.min()) < 0 or int(index_array.max()) >= weight_array.shape[1]:
                        raise RuntimeError(f"Top-K expert index is out of range at {module_name}")
                    selected = np.take_along_axis(weight_array, index_array.astype(np.int64), axis=1)
                    selected_mass_error = float(np.max(np.abs(selected.sum(axis=1) - 1.0)))
                    selected_mask = np.zeros_like(weight_array, dtype=bool)
                    np.put_along_axis(selected_mask, index_array.astype(np.int64), True, axis=1)
                    max_unselected_probability = float(np.max(np.where(selected_mask, 0.0, weight_array)))
                    if selected_mass_error > 1e-5 or max_unselected_probability > 1e-5:
                        raise RuntimeError(
                            f"Top-K indices and sparse probabilities disagree at {module_name}: "
                            f"mass_error={selected_mass_error}, unselected={max_unselected_probability}"
                        )
                    validation.update(
                        {
                            "top_k": int(index_array.shape[1]),
                            "selected_mass_error": selected_mass_error,
                            "max_unselected_probability": max_unselected_probability,
                            "indices_in_range": True,
                        }
                    )
                self.records.append(
                    SpatialRecord(
                        family=self.family,
                        module_name=module_name,
                        module_type=current_module.__class__.__name__,
                        weights=weight_array,
                        logits=logit_array,
                        indices=index_array,
                        validation=validation,
                    )
                )

            self.handles.append(module.register_forward_hook(capture))
            self.registered_names.append(name)
        if not self.registered_names:
            raise RuntimeError(f"no {self.router_class} modules found for family={self.family}")
        return list(self.registered_names)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
