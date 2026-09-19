"""Forward-hook collector for routed modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .adapters import adapt_snapshot, infer_family


def _shapes(value: Any) -> list[list[int]]:
    shapes: list[list[int]] = []
    if hasattr(value, "shape"):
        shapes.append([int(item) for item in value.shape])
    elif isinstance(value, dict):
        for child in value.values():
            shapes.extend(_shapes(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            shapes.extend(_shapes(child))
    return shapes


def _default_is_routed(module: Any) -> bool:
    try:
        num_experts = int(module.num_experts)
        top_k = int(module.top_k)
    except (AttributeError, TypeError, ValueError):
        return False
    return num_experts > 0 and 0 < top_k <= num_experts and isinstance(
        getattr(module, "last_routing_snapshot", None), dict
    )


def _snapshot_for_hook(module: Any, family: str, route_cache: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Return the official snapshot or an explicitly labelled eval-safe MoE buffer view."""

    snapshot = getattr(module, "last_routing_snapshot", {})
    if isinstance(snapshot, dict) and snapshot:
        return dict(snapshot)
    cached = route_cache.pop(id(module), None)
    if cached:
        return cached
    # OptimizedMOEImproved deliberately publishes last_routing_snapshot only in
    # training mode. Its eval forward still updates these detached diagnostic
    # buffers, so use them without changing model mode or core forward code.
    if family == "moe":
        usage = getattr(module, "expert_usage_counts", None)
        if hasattr(usage, "numel") and usage.numel() > 0 and float(usage.detach().sum()) > 0.0:
            balance = getattr(module, "load_balancing_loss", None)
            return {
                "num_experts": int(module.num_experts),
                "top_k": int(module.top_k),
                "expert_usage": usage.detach(),
                "mean_router_probs": usage.detach(),
                "aux_loss": balance.detach() if hasattr(balance, "detach") else balance,
                "diagnostic_transport": "official_eval_runtime_buffers",
                "diagnostic_transport_fields": ["expert_usage_counts", "load_balancing_loss"],
            }
    return {}


def _moe_route_snapshot(module: Any, router_output: Any) -> dict[str, Any]:
    """Normalize an OptimizedMOEImproved eval-router return without retaining tensors."""

    if not isinstance(router_output, (list, tuple)) or len(router_output) < 2:
        return {}
    weights, indices = router_output[:2]
    if not all(hasattr(item, "detach") for item in (weights, indices)):
        return {}
    flat_indices = indices.detach().reshape(-1).to(dtype=getattr(indices, "dtype", None))
    try:
        import torch

        flat_indices = flat_indices.to(torch.long)
        counts = torch.bincount(flat_indices, minlength=int(module.num_experts)).float()
        usage = counts / counts.sum().clamp_min(1.0)
    except (ImportError, RuntimeError, TypeError, ValueError):
        return {}
    flat_weights = weights.detach().float().reshape(-1, int(module.top_k))
    return {
        "num_experts": int(module.num_experts),
        "top_k": int(module.top_k),
        "expert_usage": usage.cpu(),
        "mean_topk_weight": flat_weights.mean(dim=0).cpu(),
        "diagnostic_transport": "nested_router_forward_hook",
        "diagnostic_transport_fields": ["routing_indices", "routing_weights"],
    }


def discover_routed_modules(
    model: Any,
    family: str,
    *,
    predicate: Callable[[Any], bool] | None = None,
    leaf_only: bool = True,
) -> list[tuple[str, Any]]:
    """Return deterministic family-matched routed modules, optionally only leaf producers."""

    predicate = predicate or _default_is_routed
    candidates = [(name, module) for name, module in model.named_modules() if predicate(module) and infer_family(module) == family]
    if not leaf_only:
        return candidates
    candidate_ids = {id(module) for _, module in candidates}
    selected: list[tuple[str, Any]] = []
    for name, module in candidates:
        has_routed_child = any(id(child) in candidate_ids for child in module.modules() if child is not module)
        if not has_routed_child:
            selected.append((name, module))
    return selected


@dataclass
class RoutingCollector:
    """Collect normalized events without changing forward arguments or return values."""

    family: str
    run_id: str
    events: list[dict[str, Any]] = field(default_factory=list)
    handles: list[Any] = field(default_factory=list)
    registered_names: list[str] = field(default_factory=list)
    route_cache: dict[int, dict[str, Any]] = field(default_factory=dict)
    runtime_context: dict[str, Any] = field(default_factory=dict)
    event_callback: Callable[[dict[str, Any]], None] | None = None
    max_events: int | None = None

    def set_context(self, **context: Any) -> None:
        """Attach explicit batch/sample provenance to subsequently captured events."""

        self.runtime_context = dict(context)

    def register(self, model: Any, *, predicate: Callable[[Any], bool] | None = None) -> list[str]:
        if self.handles:
            raise RuntimeError("Collector is already registered")
        modules = discover_routed_modules(model, self.family, predicate=predicate, leaf_only=True)
        for module_name, module in modules:
            if self.family == "moe" and hasattr(module, "routing"):

                def capture_route(
                    current_router: Any,
                    inputs: Any,
                    output: Any,
                    *,
                    owner: Any = module,
                ) -> None:
                    del current_router, inputs
                    snapshot = _moe_route_snapshot(owner, output)
                    if snapshot:
                        self.route_cache[id(owner)] = snapshot

                self.handles.append(module.routing.register_forward_hook(capture_route))

            def capture(current_module: Any, inputs: Any, output: Any, *, name: str = module_name) -> None:
                snapshot = _snapshot_for_hook(current_module, self.family, self.route_cache)
                if not snapshot:
                    return
                input_shapes = _shapes(inputs)
                input_shape = input_shapes[0] if input_shapes else None
                event = adapt_snapshot(
                    family=self.family,
                    run_id=self.run_id,
                    sequence=len(self.events),
                    module_name=name,
                    module=current_module,
                    snapshot=snapshot,
                    input_shape=input_shape,
                    output_shapes=_shapes(output),
                    runtime_context=self.runtime_context,
                )
                self.events.append(event)
                if self.max_events is not None and len(self.events) > self.max_events:
                    del self.events[: len(self.events) - self.max_events]
                if self.event_callback is not None:
                    self.event_callback(event)

            self.handles.append(module.register_forward_hook(capture))
            self.registered_names.append(module_name)
        return list(self.registered_names)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.route_cache.clear()
