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
from typing import Any, Iterable, Protocol, runtime_checkable
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

    def publish_aux_loss(self, *, step: int, training: bool) -> torch.Tensor:
        ...

    def routing_snapshot(self) -> dict[str, Any]:
        ...

    def export_capabilities(self) -> dict[str, Any]:
        ...


_RECORDS: weakref.WeakKeyDictionary[nn.Module, AuxLossRecord] = weakref.WeakKeyDictionary()
_RECORDS_LOCK = Lock()
_CURRENT_STEP: ContextVar[int] = ContextVar("routing_aux_step", default=0)


def current_aux_step() -> int:
    """Return the active forward step used by implicit publications."""

    return int(_CURRENT_STEP.get())


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
        if hasattr(module, "_last_routing_stats"):
            module._last_routing_stats = None
    return next_step


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


def iter_aux_records(model: nn.Module) -> list[tuple[nn.Module, AuxLossRecord]]:
    """Return records in model traversal order for diagnostics and collectors."""

    with _RECORDS_LOCK:
        return [(module, _RECORDS[module]) for module in model.modules() if module in _RECORDS]


def export_capabilities(module: nn.Module) -> dict[str, Any]:
    """Describe conservative export capabilities for a routed module."""

    return {
        "routing_kind": getattr(module, "_routing_aux_kind", "unknown"),
        "dynamic_routing": True,
        "sparse_dispatch": bool(getattr(module, "sparse_train", False)),
        "export_safe_dense_fallback": True,
        "aux_loss_training_only": True,
    }


def routing_snapshot(module: nn.Module) -> dict[str, Any]:
    """Return a detached snapshot without exposing mutable canonical state."""

    snapshot = getattr(module, "last_routing_snapshot", {})
    if not isinstance(snapshot, dict):
        return {}
    return dict(snapshot)


def collect_aux_loss(
    model: nn.Module | None,
    *,
    step: int | None = None,
    device: torch.device | None = None,
    include_kinds: Iterable[str] = ("moe", "moa", "mot", "molora"),
    require_training: bool = True,
    ddp_sync: bool = False,
    return_diagnostics: bool = False,
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
    for module, record in iter_aux_records(model):
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
        diagnostics["values_by_kind"].setdefault(record.kind, []).append(float(record.value.detach()))
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
    "aux_step_scope",
    "begin_aux_step",
    "clear_aux_records",
    "collect_aux_loss",
    "current_aux_step",
    "export_capabilities",
    "get_aux_record",
    "iter_aux_records",
    "publish_aux_loss",
    "reset_routing_runtime_state",
    "routing_snapshot",
]
