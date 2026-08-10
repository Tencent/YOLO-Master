"""Unified routed-module protocol, snapshot adapters, and derived metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

import torch
import torch.nn as nn

from ..routing_protocol import RoutingAuxPublisher


@runtime_checkable
class RoutedModule(Protocol):
    """Structural protocol shared by MoE, MoA, MoT, and MoLoRA modules."""

    num_experts: int
    top_k: int

    @property
    def aux_loss(self) -> torch.Tensor:
        """Return the scalar routing auxiliary loss."""
        ...

    @property
    def last_routing_snapshot(self) -> dict[str, Any]:
        """Return detached routing diagnostics from the latest forward."""
        ...


class RoutedModuleMixin:
    """Compatibility mixin providing generic FLOPs and deepcopy helpers."""

    def get_gflops(self, input_shape: tuple[int, int, int, int]) -> dict[str, float]:
        """Estimate Conv2d/Linear work for a routed module."""
        batch, _, height, width = input_shape
        total_macs = 0.0
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                macs = (
                    batch
                    * module.in_channels
                    * module.out_channels
                    * (height // module.stride[0])
                    * (width // module.stride[1])
                )
                total_macs += macs * (module.kernel_size[0] * module.kernel_size[1]) / max(module.groups, 1)
            elif isinstance(module, nn.Linear):
                total_macs += batch * module.in_features * module.out_features
        gflops = total_macs / 1e9
        return {"total_gflops": gflops, "conv_linear_gflops": gflops}

    def __deepcopy__(self, memo):
        """Delegate to the canonical routed-module deepcopy helper."""
        from ._helpers import _robust_deepcopy

        return _robust_deepcopy(self, memo)


def is_routed_module(module: nn.Module) -> bool:
    """Return whether ``module`` satisfies the legacy routed-module contract."""
    return all(hasattr(module, name) for name in ("num_experts", "top_k", "aux_loss", "last_routing_snapshot"))


def collect_routed_children(module: nn.Module) -> list[nn.Module]:
    """Return routed descendants in module traversal order."""
    return [child for child in module.modules() if child is not module and is_routed_module(child)]

from ..routing_protocol import RoutedModule, collect_routed_children, global_routing_metrics, is_routed_module


@dataclass(frozen=True)
class RoutingMetrics:
    """JSON-safe metrics derived from one normalized routing snapshot."""

    expert_usage: list[float]
    topk_counts: list[float]
    num_experts: int
    top_k: int
    aux_loss: float
    gini: float
    dominant_expert: int
    dominant_share: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _float_list(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().reshape(-1).cpu().tolist()
    if value is None:
        return []
    return [float(item) for item in value]


def normalize_routing_snapshot(
    snapshot: Mapping[str, Any] | None, *, num_experts: int = 0, top_k: int = 0
) -> dict[str, Any]:
    """Normalize legacy/current snapshot keys without mutating the producer payload."""
    source = snapshot or {}
    usage = _float_list(source.get("expert_usage", source.get("usage")))
    counts = _float_list(source.get("topk_counts", source.get("counts")))
    size = int(source.get("num_experts", num_experts or len(usage)))
    if size > 0:
        usage = (usage + [0.0] * size)[:size]
        counts = (counts + [0.0] * size)[:size]
    return {
        **source,
        "expert_usage": usage,
        "topk_counts": counts,
        "num_experts": size,
        "top_k": int(source.get("top_k", top_k)),
        "aux_loss": float(source.get("aux_loss", 0.0)),
        "usage_scope": str(source.get("usage_scope", "rank_local")),
        "global_usage_available": bool(source.get("global_usage_available", False)),
        "global_expert_usage": _float_list(source.get("global_expert_usage")),
        "global_entropy": float(source.get("global_entropy", 0.0)),
        "global_gini": float(source.get("global_gini", 0.0)),
    }


def usage_gini(usage: Iterable[float] | torch.Tensor) -> float:
    """Return one canonical bounded Gini coefficient for expert usage."""
    values = torch.as_tensor(_float_list(usage), dtype=torch.float32).clamp_min(0.0)
    if values.numel() == 0:
        return 0.0
    total = values.sum()
    # Sorted cumulative form is O(n log n) instead of the pairwise O(n^2).
    sorted_values = torch.sort(values).values
    index = torch.arange(1, sorted_values.numel() + 1, dtype=sorted_values.dtype)
    gini = (2 * torch.sum(index * sorted_values) / (sorted_values.numel() * total.clamp_min(1e-12))) - (
        (sorted_values.numel() + 1) / sorted_values.numel()
    )
    gini = torch.where(total > 0, gini, torch.zeros_like(gini))
    value = float(gini.clamp(0.0, 1.0).item())
    return 0.0 if value != value else value


def routing_metrics(snapshot: Mapping[str, Any] | None, *, num_experts: int = 0, top_k: int = 0) -> RoutingMetrics:
    """Adapt a routing snapshot to the shared scheduler/diagnostic metric vocabulary."""
    normalized = normalize_routing_snapshot(snapshot, num_experts=num_experts, top_k=top_k)
    usage = normalized["expert_usage"]
    dominant = max(range(len(usage)), key=usage.__getitem__) if usage else -1
    share = usage[dominant] if dominant >= 0 else 0.0
    return RoutingMetrics(
        expert_usage=usage,
        topk_counts=normalized["topk_counts"],
        num_experts=normalized["num_experts"],
        top_k=normalized["top_k"],
        aux_loss=normalized["aux_loss"],
        gini=usage_gini(usage),
        dominant_expert=dominant,
        dominant_share=share,
    )


__all__ = [
    "RoutedModule",
    "RoutedModuleMixin",
    "RoutingAuxPublisher",
    "RoutingMetrics",
    "collect_routed_children",
    "RoutingMetrics",
    "RoutedModule",
    "collect_routed_children",
    "global_routing_metrics",
    "is_routed_module",
    "normalize_routing_snapshot",
    "routing_metrics",
    "usage_gini",
]
