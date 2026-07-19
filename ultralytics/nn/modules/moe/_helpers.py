"""Compatibility re-exports for canonical MoE helper implementations.

Historically this module copied the registry, snapshot, and deepcopy helpers
from ``_common.py``. Re-exporting the canonical objects keeps old import paths
working while ensuring fixes have one implementation source.
"""

from ._common import (
    MOE_LOSS_REGISTRY,
    MOE_SNAPSHOT_INTERVAL,
    _MOE_LOSS_REGISTRY_LOCK,
    _compute_usage_from_topk,
    _detached_zero_like,
    _flatten_moe_topk,
    _get_moe_aux_loss,
    _record_moe_snapshot,
    _registry_get,
    _registry_set,
    _robust_deepcopy,
    _should_record_snapshot,
    _zero_aux_loss_like,
    autocast,
)

__all__ = [
    "autocast",
    "MOE_LOSS_REGISTRY",
    "MOE_SNAPSHOT_INTERVAL",
    "_MOE_LOSS_REGISTRY_LOCK",
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
