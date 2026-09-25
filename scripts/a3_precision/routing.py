"""Cross-family route capture, drift metrics, and fake-precision probes."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class RouteTensor:
    """Normalized output of one routing layer for a single forward pass."""

    probabilities: torch.Tensor
    topk_indices: torch.Tensor
    top_k: int
    num_experts: int


def quantize_dequantize(tensor: torch.Tensor, precision: str, *, per_channel: bool = True) -> torch.Tensor:
    """Simulate storage quantization while retaining FP32 kernel execution."""
    source = tensor.detach().float()
    if precision == "fp32":
        return source.clone()
    if precision == "fp16":
        return source.half().float()
    if precision != "int8":
        raise ValueError(f"unsupported precision {precision!r}")
    if not source.numel():
        return source.clone()
    if per_channel and source.ndim >= 2:
        reduce_dims = tuple(range(1, source.ndim))
        scale = source.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-12) / 127.0
    else:
        scale = source.abs().max().clamp_min(1e-12) / 127.0
    return torch.round(source / scale).clamp(-127, 127) * scale


def _module_router(module: nn.Module) -> nn.Module | None:
    for attribute in ("router", "routing", "gate", "gating"):
        candidate = getattr(module, attribute, None)
        if isinstance(candidate, nn.Module):
            return candidate
    return None


def routed_leaf_modules(model: nn.Module, *, family: str | None = None) -> dict[str, nn.Module]:
    """Return interpretable leaf routing modules, optionally restricted to one family."""
    family = family.lower() if family else None
    candidates: dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        num_experts = getattr(module, "num_experts", 0)
        top_k = getattr(module, "top_k", 0)
        if not isinstance(num_experts, int) or num_experts <= 1 or _module_router(module) is None:
            continue
        if not isinstance(top_k, int) or top_k <= 0:
            top_k = num_experts
        module_name = type(module).__name__.lower()
        routing_kind = str(getattr(module, "routing_kind", "")).lower()
        if family and family not in module_name and family not in routing_kind:
            # MoE has many class names without the exact family token. Keep it as
            # the broad default, while explicit families remain strict.
            if family != "moe" or any(token in module_name for token in ("moa", "mot", "molora", "latent")):
                continue
        candidates[name or "<root>"] = module
    result = {}
    for name, module in candidates.items():
        nested = any(child is not module and child in candidates.values() for child in module.modules())
        if not nested:
            result[name] = module
    return result


def _expert_axis(tensor: torch.Tensor, num_experts: int) -> int | None:
    if tensor.ndim > 1 and tensor.shape[1] == num_experts:
        return 1
    if tensor.ndim and tensor.shape[-1] == num_experts:
        return tensor.ndim - 1
    if tensor.ndim == 1 and tensor.shape[0] == num_experts:
        return 0
    return None


def _first_float_tensor(output: Any, num_experts: int) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output if output.is_floating_point() and _expert_axis(output, num_experts) is not None else None
    if isinstance(output, (tuple, list)):
        # Router contracts normally place weights first and logits last. Prefer
        # the first probability-like tensor for uniform cross-family behavior.
        for candidate in output:
            found = _first_float_tensor(candidate, num_experts)
            if found is not None:
                return found
    if isinstance(output, dict):
        for key in ("probabilities", "weights", "router_probs", "logits"):
            if key in output:
                found = _first_float_tensor(output[key], num_experts)
                if found is not None:
                    return found
    return None


def _sparse_topk_probabilities(output: Any, *, num_experts: int, top_k: int) -> torch.Tensor | None:
    """Reconstruct dense expert probabilities from ``(topk_weights, indices, ...)`` outputs."""
    if isinstance(output, dict):
        items = list(output.values())
    elif isinstance(output, (tuple, list)):
        items = list(output)
    else:
        return None
    values = [item for item in items if isinstance(item, torch.Tensor) and item.is_floating_point()]
    indices = [item for item in items if isinstance(item, torch.Tensor) and not item.is_floating_point()]
    for weights in values:
        for index in indices:
            if weights.shape != index.shape or weights.numel() == 0:
                continue
            candidate_axes = [axis for axis, size in enumerate(weights.shape) if size == top_k]
            axis = (
                1 if weights.ndim > 1 and weights.shape[1] == top_k else candidate_axes[-1] if candidate_axes else None
            )
            if axis is None or int(index.min()) < 0 or int(index.max()) >= num_experts:
                continue
            shape = list(weights.shape)
            shape[axis] = num_experts
            dense = torch.zeros(shape, dtype=weights.dtype, device=weights.device)
            dense.scatter_(axis, index.long(), weights)
            dense = dense.clamp_min(0.0)
            return dense / dense.sum(dim=axis, keepdim=True).clamp_min(1e-12)
    return None


def normalize_router_output(output: Any, *, num_experts: int, top_k: int) -> RouteTensor:
    """Convert heterogeneous router outputs to ``[B,E,...]`` probabilities."""
    tensor = _first_float_tensor(output, num_experts)
    if tensor is None:
        tensor = _sparse_topk_probabilities(output, num_experts=num_experts, top_k=top_k)
    if tensor is None:
        raise TypeError(f"router output has no floating tensor with {num_experts} experts")
    axis = _expert_axis(tensor, num_experts)
    if axis is None:
        raise ValueError("router expert axis could not be located")
    value = tensor.detach().float()
    sums = value.sum(dim=axis, keepdim=True)
    looks_probability = bool(
        torch.isfinite(value).all()
        and (value >= -1e-6).all()
        and torch.allclose(sums, torch.ones_like(sums), atol=2e-3, rtol=2e-3)
    )
    probabilities = value.clamp_min(0.0) if looks_probability else value.softmax(dim=axis)
    if axis != 1:
        probabilities = probabilities.movedim(axis, 1)
    if probabilities.ndim == 1:
        probabilities = probabilities.unsqueeze(0)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)
    top_k = max(1, min(int(top_k), num_experts))
    indices = probabilities.topk(top_k, dim=1).indices
    return RouteTensor(
        probabilities=probabilities.cpu(), topk_indices=indices.cpu(), top_k=top_k, num_experts=num_experts
    )


class RouteCapture:
    """Temporary forward-hook collector for all leaf routers in a model."""

    def __init__(self, model: nn.Module, *, family: str | None = None):
        self.layers = routed_leaf_modules(model, family=family)
        if not self.layers:
            raise ValueError(f"no routed leaf modules found for family={family!r}")
        self.current: dict[str, RouteTensor] = {}
        self._handles = []

    def __enter__(self) -> "RouteCapture":
        for name, module in self.layers.items():
            router = _module_router(module)
            assert router is not None

            def hook(_module, _inputs, output, *, layer_name=name, owner=module):
                self.current[layer_name] = normalize_router_output(
                    output,
                    num_experts=int(getattr(owner, "num_experts")),
                    top_k=int(getattr(owner, "top_k", getattr(owner, "num_experts"))),
                )

            self._handles.append(router.register_forward_hook(hook))
        return self

    def clear(self) -> None:
        self.current.clear()

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _parameter_modules(model: nn.Module, scope: str, routed: dict[str, nn.Module]) -> list[nn.Module]:
    if scope == "full_model_weight":
        return [model]
    if scope == "router_only":
        return [router for module in routed.values() if (router := _module_router(module)) is not None]
    raise ValueError("scope must be 'router_only' or 'full_model_weight'")


@contextmanager
def simulated_precision(
    model: nn.Module,
    precision: str,
    *,
    scope: str,
    routed: dict[str, nn.Module],
    per_channel: bool = True,
) -> Iterator[None]:
    """Temporarily fake-quantize selected parameters and restore them exactly."""
    modules = _parameter_modules(model, scope, routed)
    selected: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen and parameter.is_floating_point():
                seen.add(id(parameter))
                selected.append(parameter)
    originals = [parameter.detach().clone() for parameter in selected]
    try:
        for parameter in selected:
            parameter.data.copy_(quantize_dequantize(parameter.data, precision, per_channel=per_channel))
        yield
    finally:
        for parameter, original in zip(selected, originals):
            parameter.data.copy_(original)


def compare_route_tensors(reference: RouteTensor, candidate: RouteTensor) -> dict[str, float | int | bool]:
    """Compute exact Top-K, Jaccard, flip, margin, and distribution drift metrics."""
    if reference.num_experts != candidate.num_experts or reference.top_k != candidate.top_k:
        raise ValueError("route contracts differ between reference and candidate")
    if reference.probabilities.shape != candidate.probabilities.shape:
        raise ValueError(
            f"route probability shapes differ: {reference.probabilities.shape} != {candidate.probabilities.shape}"
        )
    ref_indices = reference.topk_indices.movedim(1, -1).reshape(-1, reference.top_k).long()
    new_indices = candidate.topk_indices.movedim(1, -1).reshape(-1, candidate.top_k).long()
    ordered_exact = (ref_indices == new_indices).all(dim=1)
    unordered_exact = (ref_indices.sort(dim=1).values == new_indices.sort(dim=1).values).all(dim=1)
    intersection = (ref_indices[:, :, None] == new_indices[:, None, :]).any(dim=2).sum(dim=1).float()
    union = (2 * reference.top_k - intersection).clamp_min(1.0)
    jaccard = intersection / union
    route_flip = ~unordered_exact
    top1_flip = ref_indices[:, 0] != new_indices[:, 0]

    ref_probs = reference.probabilities.movedim(1, -1).reshape(-1, reference.num_experts).double()
    new_probs = candidate.probabilities.movedim(1, -1).reshape(-1, candidate.num_experts).double()
    ref_top2 = ref_probs.topk(min(2, reference.num_experts), dim=1).values
    new_top2 = new_probs.topk(min(2, candidate.num_experts), dim=1).values
    ref_margin = ref_top2[:, 0] - ref_top2[:, 1] if reference.num_experts > 1 else ref_top2[:, 0]
    new_margin = new_top2[:, 0] - new_top2[:, 1] if candidate.num_experts > 1 else new_top2[:, 0]
    mean_probs = 0.5 * (ref_probs + new_probs)
    ref_safe = ref_probs.clamp_min(1e-12)
    new_safe = new_probs.clamp_min(1e-12)
    mean_safe = mean_probs.clamp_min(1e-12)
    js = 0.5 * (
        (ref_safe * (ref_safe.log() - mean_safe.log())).sum(dim=1)
        + (new_safe * (new_safe.log() - mean_safe.log())).sum(dim=1)
    )
    return {
        "tokens": int(ref_indices.shape[0]),
        "topk_ordered_exact_rate": float(ordered_exact.double().mean()),
        "topk_exact_rate": float(unordered_exact.double().mean()),
        "jaccard_rate": float(jaccard.double().mean()),
        "token_flip_rate": float(route_flip.double().mean()),
        "sample_flipped": bool(route_flip.any()),
        "top1_flip_rate": float(top1_flip.double().mean()),
        "sample_top1_flipped": bool(top1_flip.any()),
        "reference_margin_mean": float(ref_margin.mean()),
        "reference_margin_p01": float(torch.quantile(ref_margin, 0.01)),
        "reference_margin_p05": float(torch.quantile(ref_margin, 0.05)),
        "candidate_margin_mean": float(new_margin.mean()),
        "candidate_margin_p01": float(torch.quantile(new_margin, 0.01)),
        "candidate_margin_p05": float(torch.quantile(new_margin, 0.05)),
        "margin_delta_mean": float((new_margin - ref_margin).mean()),
        "probability_mae": float((ref_probs - new_probs).abs().mean()),
        "total_variation": float((0.5 * (ref_probs - new_probs).abs().sum(dim=1)).mean()),
        "jensen_shannon": float(js.mean()),
    }


def weighted_layer_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-sample route records while respecting each layer's token count."""
    if not records:
        return {}
    weighted_fields = (
        "topk_ordered_exact_rate",
        "topk_exact_rate",
        "jaccard_rate",
        "token_flip_rate",
        "top1_flip_rate",
        "reference_margin_mean",
        "candidate_margin_mean",
        "margin_delta_mean",
        "probability_mae",
        "total_variation",
        "jensen_shannon",
    )
    total_tokens = sum(int(record["tokens"]) for record in records)
    result: dict[str, Any] = {
        "samples": len(records),
        "tokens": total_tokens,
        "sample_flip_rate": float(np.mean([bool(record["sample_flipped"]) for record in records])),
        "sample_top1_flip_rate": float(np.mean([bool(record["sample_top1_flipped"]) for record in records])),
        "reference_margin_p01": float(min(float(record["reference_margin_p01"]) for record in records)),
        "reference_margin_p05": float(np.mean([float(record["reference_margin_p05"]) for record in records])),
        "candidate_margin_p01": float(min(float(record["candidate_margin_p01"]) for record in records)),
        "candidate_margin_p05": float(np.mean([float(record["candidate_margin_p05"]) for record in records])),
    }
    for field in weighted_fields:
        result[field] = float(
            sum(float(record[field]) * int(record["tokens"]) for record in records) / max(total_tokens, 1)
        )
    return result
