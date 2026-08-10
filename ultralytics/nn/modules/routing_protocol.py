"""Canonical routing auxiliary-loss publication and collection.

The project historically carried three runtime channels for routing losses:
the MoE registry, module attributes, and wrapper-specific collectors.  This
module provides one small, weakly-referenced state channel while keeping the
old registry available as a compatibility transport.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable
import weakref

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AuxLossRecord:
    """One canonical publication for a routed module and forward step."""

    value: torch.Tensor
    step: int
    training: bool
    kind: str
    covered_modules: frozenset[int] = frozenset()


@runtime_checkable
class RoutingAuxPublisher(Protocol):
    """Protocol exposed by modules that publish routing regularisation."""

    def publish_aux_loss(self, *, step: int, training: bool) -> torch.Tensor: ...

    def routing_snapshot(self) -> dict[str, Any]: ...

    def export_capabilities(self) -> dict[str, Any]: ...


@runtime_checkable
class RoutedModule(Protocol):
    """Structural protocol shared by every expert-routing module."""

    num_experts: int
    top_k: int

    @property
    def aux_loss(self) -> torch.Tensor: ...

    @property
    def last_routing_snapshot(self) -> dict[str, Any]: ...


_RECORDS: weakref.WeakKeyDictionary[nn.Module, AuxLossRecord] = weakref.WeakKeyDictionary()
_RECORDS_LOCK = Lock()
_CURRENT_STEP: ContextVar[int] = ContextVar("routing_aux_step", default=0)


def current_aux_step() -> int:
    """Return the active forward step used by implicit publications."""

    return int(_CURRENT_STEP.get())


def is_export_or_tracing() -> bool:
    """Return whether the current forward is being captured for export/tracing."""

    return bool(torch.jit.is_tracing() or torch.onnx.is_in_onnx_export())


def begin_aux_step(step: int | None = None) -> int:
    """Set and return the canonical step for the current forward context."""

    if step is None:
        step = current_aux_step() + 1
    step = int(step)
    _CURRENT_STEP.set(step)
    return step


def clear_aux_records(*, step: int | None = None) -> int:
    """Clear canonical publications and advance the implicit step."""

    next_step = begin_aux_step(step)
    with _RECORDS_LOCK:
        _RECORDS.clear()
    return next_step


def reset_routing_runtime_state(model: nn.Module | None = None, *, step: int | None = None) -> int:
    """Clear canonical and module-local forward state after recovery/eval boundaries."""

    next_step = clear_aux_records(step=step)
    if model is None:
        return next_step
    for module in model.modules():
        for name in ("last_aux_loss", "_last_aux_loss"):
            value = getattr(module, name, None)
            if isinstance(value, torch.Tensor) and value.grad_fn is not None:
                setattr(module, name, value.detach().new_zeros(()))
        if hasattr(module, "last_routing_snapshot"):
            module.last_routing_snapshot = {}
        if hasattr(module, "last_routing_diagnostics"):
            module.last_routing_diagnostics = {}
        if hasattr(module, "_last_routing_stats"):
            module._last_routing_stats = None
    return next_step


def anneal_mixture_temperatures(
    model: nn.Module,
    *,
    factor: float = 0.97,
    min_temp: float = 0.3,
    families: Iterable[str] = ("moe", "moa", "mot", "latent"),
) -> int:
    """Anneal every enabled mixture router through one protocol-level entry point.

    Router implementations historically stored temperature as either a Python
    float or a persistent tensor, and some families nested the router under a
    ``routing`` attribute. This helper handles both representations and filters
    by module package, so the trainer cannot accidentally anneal unrelated model
    temperatures. Returns the number of updated router modules.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model)!r}")
    factor = float(factor)
    min_temp = float(min_temp)
    if not 0.0 < factor:
        raise ValueError(f"temperature anneal factor must be > 0, got {factor}")
    if not 0.0 < min_temp:
        raise ValueError(f"minimum temperature must be > 0, got {min_temp}")
    family_tokens = tuple(str(item).lower() for item in families)
    updated = 0
    seen: set[int] = set()
    for module in model.modules():
        module_path = module.__class__.__module__.lower()
        if not any(token in module_path for token in family_tokens):
            continue
        if not hasattr(module, "temperature"):
            continue
        module._external_temperature_schedule = True
        temperature = getattr(module, "temperature")
        marker = id(temperature) if isinstance(temperature, torch.Tensor) else id(module)
        if marker in seen:
            continue
        seen.add(marker)
        if isinstance(temperature, torch.Tensor):
            with torch.no_grad():
                temperature.fill_(max(float(temperature.detach()), min_temp) * factor)
                temperature.clamp_(min=min_temp)
        elif isinstance(temperature, (float, int)):
            setattr(module, "temperature", max(float(temperature) * factor, min_temp))
        else:
            continue
        updated += 1
    return updated


def configure_mixture_temperature_schedule(
    model: nn.Module,
    *,
    external: bool = True,
    families: Iterable[str] = ("moe", "moa", "mot", "latent"),
) -> int:
    """Select trainer-level scheduling and disable conflicting per-forward annealing."""
    family_tokens = tuple(str(item).lower() for item in families)
    configured = 0
    for module in model.modules():
        module_path = module.__class__.__module__.lower()
        if any(token in module_path for token in family_tokens) and hasattr(module, "temperature"):
            module._external_temperature_schedule = bool(external)
            configured += 1
    return configured


@contextmanager
def aux_step_scope(step: int | None = None, *, clear: bool = False):
    """Temporarily establish a canonical step for standalone forward calls."""

    previous = current_aux_step()
    active = clear_aux_records(step=step) if clear else begin_aux_step(step)
    try:
        yield active
    finally:
        _CURRENT_STEP.set(previous)


def publish_aux_loss(
    module: nn.Module,
    value: torch.Tensor,
    *,
    step: int | None = None,
    kind: str = "moe",
    training: bool | None = None,
    covered_modules: Iterable[nn.Module | int] = (),
) -> torch.Tensor:
    """Publish one canonical scalar loss for ``module``.

    Evaluation publications are intentionally detached zeros.  This keeps
    diagnostics available without retaining an autograd graph between eval
    batches.  Re-publishing a module replaces its previous record, so repeated
    reads never accumulate state.
    """

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"auxiliary loss must be a Tensor, got {type(value)!r}")
    if value.numel() != 1:
        raise ValueError(f"auxiliary loss must be scalar, got shape {tuple(value.shape)}")
    training = bool(module.training if training is None else training)
    step = current_aux_step() if step is None else int(step)
    canonical = value.reshape(())
    if not training:
        canonical = canonical.detach().new_zeros(())
    covered = frozenset(id(item) if isinstance(item, nn.Module) else int(item) for item in covered_modules)
    record = AuxLossRecord(canonical, step, training, str(kind), covered)
    with _RECORDS_LOCK:
        _RECORDS[module] = record
    return canonical


def get_aux_record(module: nn.Module) -> AuxLossRecord | None:
    """Return the latest canonical record for one module."""

    with _RECORDS_LOCK:
        return _RECORDS.get(module)


def iter_aux_records(
    model: nn.Module | None,
    modules: Iterable[nn.Module] | None = None,
) -> list[tuple[nn.Module, AuxLossRecord]]:
    """Return records in model traversal order, or all records when model is None."""

    with _RECORDS_LOCK:
        if model is None and modules is None:
            return list(_RECORDS.items())
        candidates = model.modules() if modules is None else modules
        return [(module, _RECORDS[module]) for module in candidates if module in _RECORDS]


def export_capabilities(module: nn.Module) -> dict[str, Any]:
    """Describe conservative export capabilities for a routed module."""

    eager_sparse = bool(
        getattr(module, "_routing_sparse_dispatch", False)
        or getattr(module, "sparse_train", False)
        or getattr(module, "use_sparse_inference", False)
    )
    training_sparse = bool(getattr(module, "sparse_train", False))
    return {
        "routing_kind": getattr(module, "_routing_aux_kind", "unknown"),
        "supported": True,
        "dynamic_routing": True,
        "sparse_dispatch": eager_sparse,
        "eager_sparse_dispatch": eager_sparse,
        "training_sparse_dispatch": training_sparse,
        "onnx_sparse_dispatch": False,
        "torchscript_trace_sparse_dispatch": False,
        "exact_sparse_export": False,
        "export_safe_dense_fallback": True,
        "sparse_export_limitation": (
            "Data-dependent expert dispatch is supported only in eager execution; "
            "ONNX and TorchScript tracing use the dense fallback."
            if eager_sparse
            else "This module uses dense routed execution in eager and exported graphs."
        ),
        "aux_loss_training_only": True,
    }


def graph_connected_finite_zero(*values: torch.Tensor) -> torch.Tensor:
    """Return a finite scalar zero while preserving a safe autograd connection."""

    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    if not tensors:
        return torch.tensor(0.0)
    source = next((value for value in tensors if value.requires_grad), tensors[0])
    safe = torch.nan_to_num(source.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return safe.sum() * 0.0


def routing_finite_diagnostics(
    *,
    logits: torch.Tensor | None = None,
    probabilities: torch.Tensor | None = None,
    aux_loss: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Summarize the first non-finite routing boundary without retaining graphs."""

    if is_export_or_tracing():
        return {
            "first_nonfinite_boundary": None,
            "logits_finite": None,
            "logits_nonfinite_count": None,
            "probabilities_finite": None,
            "probabilities_nonfinite_count": None,
            "aux_loss_finite": None,
            "aux_loss_nonfinite_count": None,
            "all_finite": None,
        }
    values = (
        ("router_logits", "logits", logits),
        ("router_probabilities", "probabilities", probabilities),
        ("aux_loss", "aux_loss", aux_loss),
    )
    diagnostics: dict[str, Any] = {"first_nonfinite_boundary": None}
    for boundary, key, value in values:
        if not isinstance(value, torch.Tensor):
            diagnostics[f"{key}_finite"] = None
            diagnostics[f"{key}_nonfinite_count"] = None
            continue
        detached = value.detach()
        finite = torch.isfinite(detached)
        all_finite = bool(finite.all().item())
        diagnostics[f"{key}_finite"] = all_finite
        diagnostics[f"{key}_nonfinite_count"] = int((~finite).sum().item())
        if not all_finite and diagnostics["first_nonfinite_boundary"] is None:
            diagnostics["first_nonfinite_boundary"] = boundary
    diagnostics["all_finite"] = diagnostics["first_nonfinite_boundary"] is None
    return diagnostics


def routing_snapshot(module: nn.Module) -> dict[str, Any]:
    """Return a detached snapshot without exposing mutable canonical state."""

    snapshot = getattr(module, "last_routing_snapshot", {})
    if not isinstance(snapshot, dict):
        return {}
    result = dict(snapshot)
    # Usage EMA and capacity counters are rank-local unless the producer
    # explicitly publishes a reduced global value.
    result.setdefault("usage_scope", "rank_local")
    result.setdefault("global_usage_available", False)
    return result


def _routing_gini(usage: torch.Tensor) -> float:
    """Compute a bounded Gini coefficient from detached expert usage."""

    values = usage.detach().float().reshape(-1).clamp_min(0.0)
    if values.numel() == 0 or not bool(torch.isfinite(values).all().item()):
        return 0.0
    total = values.sum()
    if float(total) <= 0.0:
        return 0.0
    ordered = values.sort().values
    index = torch.arange(1, ordered.numel() + 1, device=ordered.device, dtype=ordered.dtype)
    gini = 2.0 * (index * ordered).sum() / (ordered.numel() * total) - (ordered.numel() + 1.0) / ordered.numel()
    return float(gini.clamp(0.0, 1.0).item())


def global_routing_metrics(
    snapshot: Mapping[str, Any] | None,
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Add optional all-reduced usage diagnostics without retaining gradients.

    The producer's ``expert_usage`` remains rank-local. When a process group is
    active, the detached usage vector is reduced independently and exposed as
    ``global_expert_usage``, ``global_entropy``, and ``global_gini``. If no
    process group exists, the local vector is reused and marked unavailable.
    """

    result = dict(snapshot or {})
    usage = result.get("expert_usage", result.get("usage"))
    if isinstance(usage, torch.Tensor):
        local = usage.detach().float().reshape(-1)
    elif usage is None:
        local = torch.empty(0, dtype=torch.float32)
    else:
        local = torch.as_tensor(usage, dtype=torch.float32).reshape(-1)
    local = torch.nan_to_num(local, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    global_usage = local.clone()
    reduced = False
    if local.numel() and torch.distributed.is_available() and torch.distributed.is_initialized():
        world = int(torch.distributed.get_world_size())
        if world > 1:
            reduce_device = device
            backend = str(torch.distributed.get_backend()).lower()
            if backend == "nccl":
                if reduce_device is None or reduce_device.type != "cuda":
                    reduce_device = torch.device("cuda", torch.cuda.current_device())
            elif reduce_device is not None and reduce_device.type != "cpu":
                reduce_device = torch.device("cpu")
            elif reduce_device is None and local.is_cuda:
                reduce_device = torch.device("cpu")
            if reduce_device is not None:
                global_usage = local.to(reduce_device)
            torch.distributed.all_reduce(global_usage, op=torch.distributed.ReduceOp.SUM)
            global_usage = global_usage / world
            reduced = True
            if global_usage.device != local.device:
                global_usage = global_usage.to(local.device)
    total = global_usage.sum().clamp_min(1e-12)
    probs = global_usage / total
    entropy = float((-(probs * probs.clamp_min(1e-12).log()).sum()).item()) if probs.numel() else 0.0
    result.update(
        usage_scope=str(result.get("usage_scope", "rank_local")),
        global_usage_available=bool(reduced),
        global_expert_usage=global_usage.cpu(),
        global_entropy=entropy,
        global_gini=_routing_gini(global_usage),
    )
    result["diagnostics_schema_version"] = 1
    return result


def is_routed_module(module: nn.Module) -> bool:
    """Return whether ``module`` exposes the common routed-module contract."""

    if not isinstance(module, nn.Module):
        return False
    try:
        num_experts = int(getattr(module, "num_experts"))
        top_k = int(getattr(module, "top_k"))
    except (AttributeError, TypeError, ValueError):
        return False
    return (
        num_experts > 0
        and 0 < top_k <= num_experts
        and isinstance(getattr(module, "aux_loss", None), torch.Tensor)
        and isinstance(getattr(module, "last_routing_snapshot", None), dict)
    )


def collect_routed_children(module: nn.Module) -> list[nn.Module]:
    """Return nested routed modules, excluding the supplied root module."""

    if not isinstance(module, nn.Module):
        raise TypeError(f"module must be an nn.Module, got {type(module)!r}")
    return [child for child in module.modules() if child is not module and is_routed_module(child)]


def collect_aux_loss(
    model: nn.Module | None,
    *,
    step: int | None = None,
    device: torch.device | None = None,
    include_kinds: Iterable[str] = ("moe", "moa", "mot", "molora"),
    require_training: bool = True,
    ddp_sync: bool = False,
    return_diagnostics: bool = False,
    return_tensor_values: bool = False,
    return_value_scalars: bool = True,
    modules: Iterable[nn.Module] | None = None,
):
    """Collect canonical losses once, rejecting stale and eval publications."""

    kinds = frozenset(include_kinds)
    target_step = current_aux_step() if step is None else int(step)
    diagnostics: dict[str, Any] = {
        "step": target_step,
        "counts_by_kind": {kind: 0 for kind in kinds},
        "values_by_kind": {},
        "modules": [],
        "stale_skipped": 0,
        "eval_skipped": 0,
        "duplicate_skipped": 0,
    }
    if model is None:
        zero = torch.tensor(0.0, device=device)
        return (zero, diagnostics) if return_diagnostics else zero

    selected: list[torch.Tensor] = []
    covered: set[int] = set()
    for module, record in iter_aux_records(model, modules=modules):
        if record.kind not in kinds:
            continue
        if record.step != target_step:
            diagnostics["stale_skipped"] += 1
            continue
        if require_training and not record.training:
            diagnostics["eval_skipped"] += 1
            continue
        if not isinstance(record.value, torch.Tensor) or not record.value.requires_grad:
            continue
        if id(module) in covered:
            diagnostics["duplicate_skipped"] += 1
            continue
        selected.append(record.value)
        covered.add(id(module))
        covered.update(record.covered_modules)
        diagnostics["counts_by_kind"][record.kind] = diagnostics["counts_by_kind"].get(record.kind, 0) + 1
        if return_value_scalars:
            diagnostics["values_by_kind"].setdefault(record.kind, []).append(float(record.value.detach()))
        if return_tensor_values:
            diagnostics.setdefault("_tensor_values_by_kind", {}).setdefault(record.kind, []).append(record.value)
        diagnostics["modules"].append(module.__class__.__name__)

    if selected:
        total = selected[0]
        for value in selected[1:]:
            total = total + value.to(total.device, dtype=total.dtype)
        if device is not None and total.device != device:
            total = total.to(device)
    else:
        parameter = next(model.parameters(), None)
        target = parameter.device if parameter is not None else device or torch.device("cpu")
        total = torch.zeros((), device=target)

    if ddp_sync and selected and torch.distributed.is_available() and torch.distributed.is_initialized():
        world = torch.distributed.get_world_size()
        if world > 1:
            synced = total.detach().float().clone()
            torch.distributed.all_reduce(synced, op=torch.distributed.ReduceOp.SUM)
            total = total + (synced.to(dtype=total.dtype) / world - total.detach())
    return (total, diagnostics) if return_diagnostics else total


__all__ = [
    "AuxLossRecord",
    "RoutingAuxPublisher",
    "RoutedModule",
    "aux_step_scope",
    "begin_aux_step",
    "clear_aux_records",
    "collect_aux_loss",
    "anneal_mixture_temperatures",
    "configure_mixture_temperature_schedule",
    "current_aux_step",
    "export_capabilities",
    "graph_connected_finite_zero",
    "global_routing_metrics",
    "get_aux_record",
    "is_routed_module",
    "collect_routed_children",
    "iter_aux_records",
    "publish_aux_loss",
    "reset_routing_runtime_state",
    "routing_finite_diagnostics",
    "routing_snapshot",
]
