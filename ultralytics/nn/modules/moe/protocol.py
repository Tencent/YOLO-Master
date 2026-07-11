# 🐧Please note that this file has been modified by Tencent on 2026/02/13. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Unified protocol for all routed (mixture) modules: MoE, MoA, MoT, MoLoRA.

Defines a common interface so that downstream code (trainers, loss collectors,
exporters, diagnostic tools) can treat all mixture modules uniformly without
``hasattr`` probes or module-specific branching.

## RoutedModule Protocol

Any nn.Module that routes input across multiple experts/heads SHOULD satisfy
this protocol.  Existing MoE classes already comply; MoA, MoT, and MoLoRA
are patched to comply via ``@property`` shims in their respective files.

Required attributes/properties:
  - ``num_experts`` (int): Total number of expert branches.
  - ``top_k`` (int): Number of active experts per forward (``== num_experts`` if dense).
  - ``aux_loss`` (Tensor): Scalar auxiliary loss (balance + z-loss); zero if eval.
  - ``last_routing_snapshot`` (dict): Detached routing diagnostics from last forward.

Optional (recommended) methods:
  - ``get_gflops(input_shape) -> dict``: Per-component FLOPs estimate.
  - ``__deepcopy__(memo)``: Safe deepcopy that strips non-leaf tensors.
  - ``set_top_k(k)``: Dynamically adjust Top-K (MoE only; MoA/MoT use temperature).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable, Tuple

import torch
import torch.nn as nn


@runtime_checkable
class RoutedModule(Protocol):
    """Protocol that all mixture-routing modules (MoE/MoA/MoT/MoLoRA) implement.

    Use ``isinstance(module, RoutedModule)`` for duck-typing checks, or simply
    access the attributes directly — the protocol is non-binding; each module
    type already provides all required attributes.
    """

    # ── Required attributes ──────────────────────────────────────────
    num_experts: int
    top_k: int

    @property
    def aux_loss(self) -> torch.Tensor:
        """Scalar auxiliary loss (balance + z-loss). Zero outside training."""
        ...

    @property
    def last_routing_snapshot(self) -> Dict[str, Any]:
        """Detached routing diagnostics from the most recent forward pass.

        Keys (all optional, at least one present):
          - ``expert_usage`` (Tensor [E]): Normalized per-expert usage share.
          - ``mean_router_probs`` (Tensor [E]): Mean router probabilities.
          - ``aux_loss`` (float): Detached scalar aux loss value.
          - ``num_experts`` (int), ``top_k`` (int).
        """
        ...


# ── Mixin for modules that want a zero-cost default implementation ──────

class RoutedModuleMixin:
    """Mixin providing default RoutedModule protocol shims.

    Subclasses should override ``aux_loss`` and populate
    ``last_routing_snapshot`` during forward.  This mixin ensures that
    ``get_gflops`` and ``__deepcopy__`` are always available, even on modules
    that don't define their own.
    """

    def get_gflops(self, input_shape: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Default GFLOPs estimate: sum over all Conv2d/Linear submodules."""
        B, C, H, W = input_shape
        total_macs = 0.0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                macs = B * m.in_channels * m.out_channels * (H // m.stride[0]) * (W // m.stride[1])
                macs *= (m.kernel_size[0] * m.kernel_size[1]) / max(m.groups, 1)
                total_macs += macs
            elif isinstance(m, nn.Linear):
                total_macs += B * m.in_features * m.out_features
        gf = total_macs / 1e9
        return {"total_gflops": gf, "conv_linear_gflops": gf}

    def __deepcopy__(self, memo):
        """Default safe deepcopy — delegates to ``_robust_deepcopy``."""
        from ._helpers import _robust_deepcopy
        return _robust_deepcopy(self, memo)


def is_routed_module(module: nn.Module) -> bool:
    """Check whether a module satisfies the RoutedModule protocol.

    This is a structural check: the module must have ``num_experts``,
    ``top_k``, ``aux_loss``, and ``last_routing_snapshot``.
    """
    return (
        hasattr(module, "num_experts")
        and hasattr(module, "top_k")
        and hasattr(module, "aux_loss")
        and hasattr(module, "last_routing_snapshot")
    )


def collect_routed_children(module: nn.Module) -> list:
    """Return all immediate+nested RoutedModule children of ``module``.

    Useful for trainers and loss collectors that need to iterate over
    all mixture modules in a model.
    """
    results = []
    for child in module.modules():
        if child is module:
            continue
        if is_routed_module(child):
            results.append(child)
    return results


__all__ = [
    "RoutedModule",
    "RoutedModuleMixin",
    "is_routed_module",
    "collect_routed_children",
]
