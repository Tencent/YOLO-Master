"""Runtime-neutral conditional expert dispatcher with auditable execution semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np

from ultralytics.nn.modules.topk_contract import (
    DEADBAND_LOWEST_ID_TOPK,
    DEFAULT_DETERMINISTIC_TOPK,
    LEGACY_PRIORITY_BIAS_TOPK,
    SUPPORTED_DETERMINISTIC_TOPK,
)


class DynamicDispatchContractError(RuntimeError):
    """Raised when routing tensors violate the conditional-execution contract."""


@dataclass(frozen=True)
class DynamicDispatchAudit:
    """Evidence describing which experts were actually invoked for one runtime call."""

    routing_granularity: str
    batch_size: int
    num_experts: int
    top_k: int
    executed_expert_ids: tuple[int, ...]
    executed_expert_calls: int
    dense_expert_calls: int
    selected_sample_expert_pairs: int
    dense_sample_expert_pairs: int
    min_selected_experts_per_sample: int
    max_selected_experts_per_sample: int
    sample_pair_reduction_ratio: float
    batch_union_reduction_ratio: float
    conditional_execution_observed: bool
    batch_union_reduction_observed: bool
    dense_fallback_detected: bool

    def to_dict(self) -> dict:
        """Return a JSON-serializable audit record."""
        return asdict(self)


def _validate_routing_weights(
    routing_weights: np.ndarray,
    *,
    batch_size: int,
    num_experts: int,
    top_k: int,
    zero_tolerance: float,
) -> np.ndarray:
    weights = np.asarray(routing_weights)
    if weights.ndim != 4:
        raise DynamicDispatchContractError(
            f"routing_weights must be [B,E,H,W], got shape {tuple(weights.shape)}"
        )
    if weights.shape[0] != batch_size or weights.shape[1] != num_experts:
        raise DynamicDispatchContractError(
            f"routing_weights shape {tuple(weights.shape)} does not match B={batch_size}, E={num_experts}"
        )
    if not np.isfinite(weights).all():
        raise DynamicDispatchContractError("routing_weights contains NaN or Inf")
    if (weights < -zero_tolerance).any():
        raise DynamicDispatchContractError("routing_weights contains negative probabilities")

    active_per_location = (weights > zero_tolerance).sum(axis=1)
    if (active_per_location == 0).any():
        raise DynamicDispatchContractError("at least one routing location has no selected expert")
    if (active_per_location > top_k).any():
        maximum = int(active_per_location.max())
        raise DynamicDispatchContractError(
            f"routing tensor is not sparse Top-K: observed {maximum} active experts at one location, top_k={top_k}. "
            "Refusing a masked-dense/dense-router fallback."
        )

    probability_sum = weights.sum(axis=1)
    if not np.allclose(probability_sum, 1.0, rtol=1e-4, atol=1e-5):
        maximum_error = float(np.max(np.abs(probability_sum - 1.0)))
        raise DynamicDispatchContractError(
            f"routing weights must sum to one across experts; maximum error={maximum_error:.6g}"
        )
    return weights


def sparsify_topk_probabilities(
    routing_probabilities: np.ndarray,
    *,
    top_k: int,
    zero_tolerance: float = 1e-8,
    tie_tolerance: float = 1e-6,
    tie_break: str = DEFAULT_DETERMINISTIC_TOPK,
) -> np.ndarray:
    """Apply deterministic host Top-K to dense router probabilities.

    The default deadband policy repeatedly selects the largest remaining
    probability and chooses the lowest expert id among values within
    ``tie_tolerance`` of that maximum. Original probabilities are retained as
    mixture weights. Keeping Top-K outside ONNX avoids backend-specific ``TopK``
    behavior while experts remain conditionally executed.
    """
    probabilities = np.asarray(routing_probabilities)
    if probabilities.ndim != 4:
        raise DynamicDispatchContractError(
            f"routing_probabilities must be [B,E,H,W], got shape {tuple(probabilities.shape)}"
        )
    num_experts = int(probabilities.shape[1])
    if not 1 <= int(top_k) < num_experts:
        raise DynamicDispatchContractError(f"top_k must be in [1, {num_experts - 1}], got {top_k}")
    if float(tie_tolerance) < 0.0:
        raise DynamicDispatchContractError(f"tie_tolerance must be nonnegative, got {tie_tolerance}")
    if tie_break not in SUPPORTED_DETERMINISTIC_TOPK:
        raise DynamicDispatchContractError(f"unsupported deterministic Top-K policy: {tie_break!r}")
    if not np.isfinite(probabilities).all():
        raise DynamicDispatchContractError("routing_probabilities contains NaN or Inf")
    if (probabilities < -zero_tolerance).any():
        raise DynamicDispatchContractError("routing_probabilities contains negative values")
    probability_sum = probabilities.sum(axis=1)
    if not np.allclose(probability_sum, 1.0, rtol=1e-4, atol=1e-5):
        maximum_error = float(np.max(np.abs(probability_sum - 1.0)))
        raise DynamicDispatchContractError(
            f"routing probabilities must sum to one across experts; maximum error={maximum_error:.6g}"
        )

    if tie_break == LEGACY_PRIORITY_BIAS_TOPK:
        ranking_probabilities = probabilities
        if tie_tolerance > 0.0:
            expert_priority = np.arange(num_experts, dtype=probabilities.dtype).reshape(1, -1, 1, 1)
            ranking_probabilities = probabilities - expert_priority * tie_tolerance
        top_indices = np.argsort(-ranking_probabilities, axis=1, kind="stable")[:, : int(top_k)]
    elif tie_break == DEADBAND_LOWEST_ID_TOPK:
        available = np.ones_like(probabilities, dtype=bool)
        expert_ids = np.arange(num_experts, dtype=np.int64).reshape(1, -1, 1, 1)
        expert_ids = np.broadcast_to(expert_ids, probabilities.shape)
        selections = []
        for _ in range(int(top_k)):
            remaining = np.where(available, probabilities, -np.inf)
            maximum = remaining.max(axis=1, keepdims=True)
            candidates = available & (remaining >= maximum - float(tie_tolerance))
            candidate_ids = np.where(candidates, expert_ids, num_experts)
            selected = candidate_ids.min(axis=1, keepdims=True)
            selections.append(selected)
            np.put_along_axis(available, selected, False, axis=1)
        top_indices = np.concatenate(selections, axis=1)
    else:  # pragma: no cover - guarded above, retained for type narrowing.
        raise DynamicDispatchContractError(f"unsupported deterministic Top-K policy: {tie_break!r}")
    top_values = np.take_along_axis(probabilities, top_indices, axis=1)
    sparse = np.zeros_like(probabilities)
    np.put_along_axis(sparse, top_indices, top_values, axis=1)
    normalizer = sparse.sum(axis=1, keepdims=True)
    if (normalizer <= zero_tolerance).any():
        raise DynamicDispatchContractError("host Top-K retained zero routing mass")
    return sparse / normalizer


def _select_sample_experts(
    weights: np.ndarray,
    *,
    routing_granularity: str,
    top_k: int,
    dynamic_threshold: float,
    zero_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size, num_experts = weights.shape[:2]
    if routing_granularity == "sample":
        importance = weights.mean(axis=(2, 3))
        order = np.argsort(-importance, axis=1, kind="stable")[:, :top_k]
        selected = np.zeros((batch_size, num_experts), dtype=bool)
        for batch_index in range(batch_size):
            for rank, expert_index in enumerate(order[batch_index]):
                importance_value = importance[batch_index, expert_index]
                if rank == 0 or (importance_value > zero_tolerance and importance_value >= dynamic_threshold):
                    selected[batch_index, expert_index] = True
    elif routing_granularity == "spatial_union":
        selected = (weights > zero_tolerance).any(axis=(2, 3))
    else:
        raise DynamicDispatchContractError(
            f"unsupported routing_granularity={routing_granularity!r}; expected 'sample' or 'spatial_union'"
        )

    if not selected.any(axis=1).all():
        raise DynamicDispatchContractError("every sample must retain at least one expert")

    retained_weights = weights * selected[:, :, None, None].astype(weights.dtype, copy=False)
    normalizer = retained_weights.sum(axis=1, keepdims=True)
    if (normalizer <= zero_tolerance).any():
        raise DynamicDispatchContractError("retained routing mass is zero for at least one location")
    retained_weights = retained_weights / normalizer
    return selected, retained_weights


def _make_dispatch_audit(
    selected: np.ndarray,
    *,
    routing_granularity: str,
    num_experts: int,
    top_k: int,
) -> DynamicDispatchAudit:
    """Build execution evidence from the sample/expert set a sparse dispatcher will invoke."""
    batch_size = int(selected.shape[0])
    selected_per_sample = selected.sum(axis=1)
    selected_pairs = int(selected_per_sample.sum())
    dense_pairs = batch_size * num_experts
    executed_ids = tuple(int(index) for index in np.flatnonzero(selected.any(axis=0)))
    executed_calls = len(executed_ids)
    conditional_observed = selected_pairs < dense_pairs
    union_reduction_observed = executed_calls < num_experts
    return DynamicDispatchAudit(
        routing_granularity=routing_granularity,
        batch_size=batch_size,
        num_experts=num_experts,
        top_k=top_k,
        executed_expert_ids=executed_ids,
        executed_expert_calls=executed_calls,
        dense_expert_calls=num_experts,
        selected_sample_expert_pairs=selected_pairs,
        dense_sample_expert_pairs=dense_pairs,
        min_selected_experts_per_sample=int(selected_per_sample.min()),
        max_selected_experts_per_sample=int(selected_per_sample.max()),
        sample_pair_reduction_ratio=1.0 - selected_pairs / dense_pairs,
        batch_union_reduction_ratio=1.0 - executed_calls / num_experts,
        conditional_execution_observed=conditional_observed,
        batch_union_reduction_observed=union_reduction_observed,
        dense_fallback_detected=not conditional_observed,
    )


def audit_sparse_routing(
    routing_weights: np.ndarray,
    *,
    batch_size: int,
    num_experts: int,
    top_k: int,
    routing_granularity: str,
    dynamic_threshold: float = 0.0,
    zero_tolerance: float = 1e-8,
    require_reduction: bool = False,
) -> DynamicDispatchAudit:
    """Validate sparse routing and describe the expert calls it requires.

    This is used by non-NumPy expert backends (for example PyTorch CUDA) to
    publish the same execution contract as :func:`dispatch_numpy_experts`.
    It does not execute an expert and therefore must be paired with a dispatcher
    that invokes exactly the returned ``executed_expert_ids``.
    """
    if int(num_experts) < 1:
        raise DynamicDispatchContractError("num_experts must be positive")
    if not 1 <= int(top_k) <= int(num_experts):
        raise DynamicDispatchContractError(f"top_k must be in [1, {num_experts}], got {top_k}")
    if not 0.0 <= float(dynamic_threshold) <= 1.0:
        raise DynamicDispatchContractError(f"dynamic_threshold must be in [0,1], got {dynamic_threshold}")
    weights = _validate_routing_weights(
        routing_weights,
        batch_size=int(batch_size),
        num_experts=int(num_experts),
        top_k=int(top_k),
        zero_tolerance=float(zero_tolerance),
    )
    selected, _ = _select_sample_experts(
        weights,
        routing_granularity=routing_granularity,
        top_k=int(top_k),
        dynamic_threshold=float(dynamic_threshold),
        zero_tolerance=float(zero_tolerance),
    )
    audit = _make_dispatch_audit(
        selected,
        routing_granularity=routing_granularity,
        num_experts=int(num_experts),
        top_k=int(top_k),
    )
    if require_reduction and not audit.conditional_execution_observed:
        raise DynamicDispatchContractError(
            "this input selected the dense sample/expert set; true conditional execution was not observed"
        )
    return audit


def dispatch_numpy_experts(
    x: np.ndarray,
    routing_weights: np.ndarray,
    expert_runners: Sequence[Callable[[np.ndarray], np.ndarray]],
    *,
    top_k: int,
    routing_granularity: str,
    dynamic_threshold: float = 0.0,
    zero_tolerance: float = 1e-8,
    require_reduction: bool = False,
) -> tuple[np.ndarray, DynamicDispatchAudit]:
    """Run only selected expert callables and combine their outputs.

    Args:
        x: Input tensor in NCHW layout.
        routing_weights: Sparse, normalized routing tensor in ``[B,E,H,W]`` layout.
        expert_runners: One callable per expert. Unselected callables are never invoked.
        top_k: Maximum number of nonzero experts at any routing location.
        routing_granularity: ``sample`` for global routing or ``spatial_union`` for token/spatial routing.
        dynamic_threshold: Optional ES-MoE sample-level pruning threshold; rank zero is always retained.
        zero_tolerance: Values at or below this threshold are treated as exact routing zeros.
        require_reduction: Raise when this concrete call executes the dense sample/expert set.

    Returns:
        Combined expert tensor and an execution audit.
    """
    inputs = np.asarray(x)
    if inputs.ndim != 4:
        raise DynamicDispatchContractError(f"x must be [B,C,H,W], got shape {tuple(inputs.shape)}")
    if not np.isfinite(inputs).all():
        raise DynamicDispatchContractError("x contains NaN or Inf")

    batch_size = int(inputs.shape[0])
    num_experts = len(expert_runners)
    if num_experts < 1:
        raise DynamicDispatchContractError("at least one expert runner is required")
    if not 1 <= int(top_k) <= num_experts:
        raise DynamicDispatchContractError(f"top_k must be in [1, {num_experts}], got {top_k}")
    if not 0.0 <= float(dynamic_threshold) <= 1.0:
        raise DynamicDispatchContractError(f"dynamic_threshold must be in [0,1], got {dynamic_threshold}")

    weights = _validate_routing_weights(
        routing_weights,
        batch_size=batch_size,
        num_experts=num_experts,
        top_k=int(top_k),
        zero_tolerance=float(zero_tolerance),
    )
    selected, retained_weights = _select_sample_experts(
        weights,
        routing_granularity=routing_granularity,
        top_k=int(top_k),
        dynamic_threshold=float(dynamic_threshold),
        zero_tolerance=float(zero_tolerance),
    )

    mixture = None
    executed_ids: list[int] = []
    for expert_index, runner in enumerate(expert_runners):
        batch_indices = np.flatnonzero(selected[:, expert_index])
        if batch_indices.size == 0:
            continue

        expert_output = np.asarray(runner(inputs[batch_indices]))
        if expert_output.ndim != 4 or expert_output.shape[0] != batch_indices.size:
            raise DynamicDispatchContractError(
                f"expert {expert_index} returned shape {tuple(expert_output.shape)} for "
                f"selected batch size {batch_indices.size}"
            )
        if expert_output.shape[2:] != inputs.shape[2:]:
            raise DynamicDispatchContractError(
                f"expert {expert_index} changed spatial shape from {tuple(inputs.shape[2:])} "
                f"to {tuple(expert_output.shape[2:])}"
            )

        if mixture is None:
            mixture = np.zeros(
                (batch_size, expert_output.shape[1], *expert_output.shape[2:]),
                dtype=np.result_type(inputs.dtype, expert_output.dtype, retained_weights.dtype),
            )
        elif expert_output.shape[1:] != mixture.shape[1:]:
            raise DynamicDispatchContractError(
                f"expert {expert_index} output shape {tuple(expert_output.shape[1:])} does not match "
                f"the established expert shape {tuple(mixture.shape[1:])}"
            )

        expert_weight = retained_weights[batch_indices, expert_index : expert_index + 1]
        mixture[batch_indices] += expert_output * expert_weight
        executed_ids.append(expert_index)

    if mixture is None:
        raise DynamicDispatchContractError("no expert was executed")

    audit = _make_dispatch_audit(
        selected,
        routing_granularity=routing_granularity,
        num_experts=num_experts,
        top_k=int(top_k),
    )
    if audit.executed_expert_ids != tuple(executed_ids):
        raise DynamicDispatchContractError(
            "executed experts do not match the sparse routing selection; refusing inconsistent audit evidence"
        )
    if require_reduction and not audit.conditional_execution_observed:
        raise DynamicDispatchContractError(
            "this input selected the dense sample/expert set; true conditional execution was not observed"
        )
    return mixture, audit


__all__ = (
    "DynamicDispatchAudit",
    "DynamicDispatchContractError",
    "audit_sparse_routing",
    "dispatch_numpy_experts",
    "sparsify_topk_probabilities",
)
