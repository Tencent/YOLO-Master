# 🐧Please note that this file has been modified by Tencent on 2026/02/13. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Shared utilities, registry, and imports for MoE submodules.

This module centralizes all common infrastructure used by base.py, advanced.py,
hybrid.py, and integration.py: device-agnostic autocast, the global auxiliary-loss
registry, snapshot recording, robust deepcopy, and the consolidated import block.
"""
import os
import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import weakref
from typing import Tuple, Dict, Optional, Union
from .utils import FlopsUtils, get_safe_groups, BatchedExpertComputation
from .experts import (
    OptimizedSimpleExpert, FusedGhostExpert, SimpleExpert, GhostExpert,
    InvertedResidualExpert, EfficientExpertGroup, SpatialExpert, SharedInvertedExpertGroup
)
from .routers import (
    UltraEfficientRouter, EfficientSpatialRouter, LocalRoutingLayer,
    AdaptiveRoutingLayer, DynamicRoutingLayer, AdvancedRoutingLayer
)
from ultralytics.nn.modules.block import ABlock, A2C2f, C3k

try:
    from torch.amp import autocast as _device_autocast
except ImportError:  # torch<1.10
    _device_autocast = None
from torch.cuda.amp import autocast as _cuda_autocast


def autocast(enabled=True, **kwargs):
    """Device-agnostic autocast wrapper. Falls back gracefully on non-CUDA devices."""
    if torch.cuda.is_available():
        if _device_autocast is not None:
            return _device_autocast("cuda", enabled=enabled, **kwargs)
        return _cuda_autocast(enabled=enabled, **kwargs)
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and _device_autocast is not None:
        return _device_autocast("mps", enabled=enabled, **kwargs)
    # On CPU, autocast is not fully supported; disable to avoid warnings/errors
    return nullcontext()
from .loss import MoELoss, gshard_balance_loss, weighted_gshard_balance_loss, differentiable_balance_loss, all_reduce_mean
from .scheduler import MoEDynamicScheduler, MoEDynamicSchedulerConfig
from ..routing_protocol import publish_aux_loss
from ultralytics.nn.modules.utils import robust_deepcopy as _robust_deepcopy

# Global registry to store auxiliary losses for MoE modules
# This prevents storing non-leaf tensors in the module instance, avoiding deepcopy errors.
# Guarded by a lock: WeakKeyDictionary mutation is not atomic, and concurrent
# forward passes (e.g. multi-threaded eval / hook callbacks) could otherwise
# corrupt its internal weakref bookkeeping.
import threading as _threading

MOE_LOSS_REGISTRY = weakref.WeakKeyDictionary()
_MOE_LOSS_REGISTRY_LOCK = _threading.Lock()


def _registry_set(module: nn.Module, value: torch.Tensor) -> None:
    """Thread-safe write to the MoE aux-loss registry."""
    with _MOE_LOSS_REGISTRY_LOCK:
        MOE_LOSS_REGISTRY[module] = value
    # Keep the legacy registry as a transport adapter while canonical readers
    # use step-aware records and can reject stale autograd graphs.
    publish_aux_loss(module, value, kind="moe", training=module.training)


def _registry_get(module: nn.Module):
    """Thread-safe read from the MoE aux-loss registry."""
    with _MOE_LOSS_REGISTRY_LOCK:
        return MOE_LOSS_REGISTRY.get(module)

# Diagnostic snapshot sampling: only every Nth forward per module records the
# latest routing summary. Tensors stay on their current device; diagnostic
# consumers move them to CPU only when they actually format/export them. Set
# MOE_SNAPSHOT_INTERVAL=1 to restore per-step recording.
MOE_SNAPSHOT_INTERVAL = max(int(os.environ.get("MOE_SNAPSHOT_INTERVAL", "10")), 1)


def _should_record_snapshot(module: nn.Module) -> bool:
    """Per-module forward gate so snapshots are taken every Nth step only."""
    if getattr(module, "_moe_force_snapshot", False):
        return True
    if MOE_SNAPSHOT_INTERVAL <= 1:
        return True
    c = getattr(module, "_moe_snap_counter", 0) + 1
    module._moe_snap_counter = c
    return (c % MOE_SNAPSHOT_INTERVAL) == 0


def _zero_aux_loss_like(module: nn.Module) -> torch.Tensor:
    """Return a scalar zero on the same device/dtype as the module parameters."""
    try:
        param = next(module.parameters())
        return param.new_zeros(())
    except StopIteration:
        return torch.tensor(0.0)


def _detached_zero_like(value) -> torch.Tensor:
    """Return a detached scalar zero on the same device/dtype as a tensor when possible."""
    if isinstance(value, torch.Tensor):
        return value.detach().new_zeros(())
    return torch.tensor(0.0)


def _get_moe_aux_loss(module: nn.Module) -> torch.Tensor:
    """Read the registered MoE aux loss, defaulting to a device-safe zero."""
    loss = _registry_get(module)
    return loss if isinstance(loss, torch.Tensor) else _zero_aux_loss_like(module)


def _flatten_moe_topk(topk_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Normalize Top-K tensors to `[N, K]` for lightweight diagnostics.

    Supports shapes:
      - 2D: `[N, K]` — already flat, return as-is
      - 4D: `[B, K, H, W]` — spatial top-k (permute + reshape)
      - 4D: `[B, H, W, K]` — NHWC top-k (reshape only)
      - Other: flatten first dim, treat second dim as K
    """
    if topk_tensor is None:
        return None
    if topk_tensor.dim() == 2:
        return topk_tensor
    if topk_tensor.dim() == 4:
        # Heuristic: if dim 1 is small (<= top_k max, usually <=8), assume [B, K, H, W]
        # Otherwise assume [B, H, W, K]
        if topk_tensor.shape[1] <= 8:
            return topk_tensor.permute(0, 2, 3, 1).reshape(-1, topk_tensor.shape[1])
        else:
            return topk_tensor.reshape(-1, topk_tensor.shape[3])
    return topk_tensor.reshape(topk_tensor.shape[0], -1)


def _compute_usage_from_topk(topk_indices: Optional[torch.Tensor], num_experts: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return normalized usage share and raw hit counts from Top-K indices."""
    if topk_indices is None or num_experts <= 0:
        zero = torch.zeros(max(num_experts, 0), dtype=torch.float32)
        return zero, zero

    flat_indices = _flatten_moe_topk(topk_indices)
    if flat_indices is None or flat_indices.numel() == 0:
        zero = torch.zeros(num_experts, dtype=torch.float32, device=topk_indices.device)
        return zero, zero

    flat = flat_indices.reshape(-1).to(torch.long)
    try:
        counts = torch.bincount(flat, minlength=num_experts).to(torch.float32)
    except RuntimeError:
        # Some accelerator backends do not implement bincount; fall back without
        # forcing CUDA users through a per-snapshot D2H sync.
        counts = torch.bincount(flat.cpu(), minlength=num_experts).to(device=topk_indices.device, dtype=torch.float32)
    total = counts.sum().clamp_min(1.0)
    return counts / total, counts


def _record_moe_snapshot(
    module: nn.Module,
    *,
    expert_usage: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    router_probs: Optional[torch.Tensor] = None,
    aux_loss: Optional[torch.Tensor] = None,
    finite_diagnostics: Optional[dict] = None,
) -> None:
    """Store a compact, detached routing snapshot for later diagnostics.

    If both `expert_usage` and `topk_indices` are provided, `expert_usage` takes
    precedence because it reflects the router's actual computed usage frequencies.
    `topk_indices` is only used as a fallback to derive usage counts.

    Sampled every ``MOE_SNAPSHOT_INTERVAL`` forwards per module; existing
    snapshot is kept between samples so consumers always see the most recent
    recorded value.
    """
    if not _should_record_snapshot(module):
        return
    # Prefer expert_usage when available; fallback to topk_indices-derived counts
    if isinstance(expert_usage, torch.Tensor):
        usage_tensor = expert_usage.detach().float()
        counts_tensor = None
    elif topk_indices is not None:
        usage_tensor, counts_tensor = _compute_usage_from_topk(topk_indices, getattr(module, "num_experts", 0))
    else:
        usage_tensor = None
        counts_tensor = None

    mean_probs = None
    if isinstance(router_probs, torch.Tensor):
        probs = router_probs.detach().float()
        if probs.dim() == 4:
            mean_probs = probs.mean(dim=(0, 2, 3))
        elif probs.dim() == 2:
            mean_probs = probs.mean(dim=0)
        else:
            mean_probs = probs.reshape(probs.shape[0], -1).mean(dim=0)

    snapshot = {
        "num_experts": int(getattr(module, "num_experts", 0)),
        "top_k": int(_flatten_moe_topk(topk_indices).shape[1]) if isinstance(topk_indices, torch.Tensor) else int(getattr(module, "top_k", 0)),
        "expert_usage": usage_tensor,
        "topk_counts": counts_tensor,
        "mean_router_probs": mean_probs,
        "aux_loss": aux_loss.detach().float() if isinstance(aux_loss, torch.Tensor) else float(aux_loss or 0.0),
    }
    if finite_diagnostics is not None:
        snapshot["finite_diagnostics"] = dict(finite_diagnostics)

    if isinstance(topk_weights, torch.Tensor):
        weights = _flatten_moe_topk(topk_weights.detach().float())
        if weights is not None and weights.numel():
            snapshot["mean_topk_weight"] = weights.mean(dim=0)

    module.last_routing_snapshot = snapshot

# ==========================================
# Ultra-optimized MoE module
