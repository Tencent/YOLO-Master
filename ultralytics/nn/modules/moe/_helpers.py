# 🐧Please note that this file has been modified by Tencent on 2026/02/13. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Shared helper functions, registry, and utilities for MoE modules.

Extracted from ``modules.py`` to reduce file size and improve maintainability.
All symbols here are re-exported by ``modules.py`` for backward compatibility.
"""
import os
import copy
import weakref
import threading as _threading

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, Union
from torch.amp import autocast as _autocast


# ---------------------------------------------------------------------------
# Device-agnostic autocast wrapper
# ---------------------------------------------------------------------------
def autocast(enabled=True, **kwargs):
    """Device-agnostic autocast wrapper. Falls back gracefully on non-CUDA devices."""
    if torch.cuda.is_available():
        return _autocast('cuda', enabled=enabled, **kwargs)
    if torch.backends.mps.is_available():
        return _autocast('mps', enabled=enabled, **kwargs)
    # On CPU, autocast is not fully supported; disable to avoid warnings/errors
    from contextlib import nullcontext
    return nullcontext() if not enabled else nullcontext()


# ---------------------------------------------------------------------------
# Global auxiliary-loss registry (WeakKeyDictionary, thread-safe)
# ---------------------------------------------------------------------------
# This prevents storing non-leaf tensors in the module instance, avoiding
# deepcopy errors.  Guarded by a lock: WeakKeyDictionary mutation is not
# atomic, and concurrent forward passes (e.g. multi-threaded eval / hook
# callbacks) could otherwise corrupt its internal weakref bookkeeping.

# Re-export the single canonical registry from _common to avoid dual-dict bugs.
from ._common import MOE_LOSS_REGISTRY, _MOE_LOSS_REGISTRY_LOCK, _registry_set, _registry_get  # noqa: E402


# ---------------------------------------------------------------------------
# Diagnostic snapshot sampling
# ---------------------------------------------------------------------------
# Only every Nth forward per module records the latest routing summary.
# Tensors stay on their current device; diagnostic consumers move them to CPU
# only when they actually format/export them. Set MOE_SNAPSHOT_INTERVAL=1 to
# restore per-step recording.

MOE_SNAPSHOT_INTERVAL = max(int(os.environ.get("MOE_SNAPSHOT_INTERVAL", "10")), 1)


def _should_record_snapshot(module: nn.Module) -> bool:
    """Per-module forward gate so snapshots are taken every Nth step only."""
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


# ---------------------------------------------------------------------------
# Top-K tensor normalisation and usage computation
# ---------------------------------------------------------------------------
def _flatten_moe_topk(topk_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Normalize Top-K tensors to ``[N, K]`` for lightweight diagnostics.

    Supports shapes:
      - 2D: ``[N, K]`` — already flat, return as-is
      - 4D: ``[B, K, H, W]`` — spatial top-k (permute + reshape)
      - 4D: ``[B, H, W, K]`` — NHWC top-k (reshape only)
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
) -> None:
    """Store a compact, detached routing snapshot for later diagnostics.

    If both ``expert_usage`` and ``topk_indices`` are provided, ``expert_usage``
    takes precedence because it reflects the router's actual computed usage
    frequencies.  ``topk_indices`` is only used as a fallback to derive usage
    counts.

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

    if isinstance(topk_weights, torch.Tensor):
        weights = _flatten_moe_topk(topk_weights.detach().float())
        if weights is not None and weights.numel():
            snapshot["mean_topk_weight"] = weights.mean(dim=0)

    module.last_routing_snapshot = snapshot


# ---------------------------------------------------------------------------
# Robust deepcopy
# ---------------------------------------------------------------------------
def _robust_deepcopy(obj, memo):
    """
    Robust deepcopy helper that sanitizes the object's __dict__ to remove
    any non-leaf tensors (which cause RuntimeError in deepcopy) before copying.
    """
    cls = obj.__class__
    new_obj = cls.__new__(cls)
    memo[id(obj)] = new_obj

    for k, v in obj.__dict__.items():
        # Check for non-leaf tensor (has grad_fn)
        if isinstance(v, torch.Tensor) and v.grad_fn is not None:
            # Replace with a safe scalar zero on the same device/dtype.
            setattr(new_obj, k, _detached_zero_like(v))
        else:
            try:
                setattr(new_obj, k, copy.deepcopy(v, memo))
            except RuntimeError as e:
                # Fallback: if deepcopy fails on a specific attribute, try to skip or reset it
                if "Only Tensors created explicitly" in str(e):
                    print(f"WARNING: Skipped deepcopy for attribute '{k}' in {cls.__name__} due to non-leaf tensor error.")
                    setattr(new_obj, k, _detached_zero_like(v))
                else:
                    raise e
            except Exception:
                # Best effort copy for other errors (e.g. pickling issues)
                # If it fails, we assume it's transient state and ignore it or shallow copy
                setattr(new_obj, k, v)

    return new_obj


__all__ = [
    "autocast",
    "MOE_LOSS_REGISTRY",
    "MOE_SNAPSHOT_INTERVAL",
    "_registry_set",
    "_registry_get",
    "_should_record_snapshot",
    "_zero_aux_loss_like",
    "_detached_zero_like",
    "_get_moe_aux_loss",
    "_flatten_moe_topk",
    "_compute_usage_from_topk",
    "_record_moe_snapshot",
    "_robust_deepcopy",
]
