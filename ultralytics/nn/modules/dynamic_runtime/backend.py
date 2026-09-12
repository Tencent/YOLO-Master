"""Backend contract for executing conditionally selected expert branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence, runtime_checkable

import numpy as np

from .dispatch import DynamicDispatchAudit, DynamicDispatchContractError, dispatch_numpy_experts


@dataclass(frozen=True)
class DispatchBackendCapabilities:
    """Machine-readable guarantees exposed by a conditional dispatch backend."""

    name: str
    conditional_execution: bool
    routing_granularities: tuple[str, ...]
    devices: tuple[str, ...]
    zero_copy: bool
    emits_execution_audit: bool


@runtime_checkable
class ConditionalDispatchBackend(Protocol):
    """Interface required by the dynamic DAG executor.

    Implementations must invoke only the expert/sample pairs selected by the
    sparse routing tensor. A backend that evaluates every expert and masks the
    results must advertise ``conditional_execution=False`` and is rejected by
    :class:`DynamicDAGExecutor`.
    """

    @property
    def capabilities(self) -> DispatchBackendCapabilities:
        """Return backend execution guarantees."""

    def dispatch(
        self,
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
        """Conditionally execute selected experts and return auditable output."""


class NumpyConditionalDispatchBackend:
    """CPU reference backend for the custom conditional-dispatch interface."""

    _CAPABILITIES = DispatchBackendCapabilities(
        name="numpy_reference_conditional_dispatch",
        conditional_execution=True,
        routing_granularities=("sample", "spatial_union"),
        devices=("cpu",),
        zero_copy=False,
        emits_execution_audit=True,
    )

    @property
    def capabilities(self) -> DispatchBackendCapabilities:
        """Return the immutable backend capability record."""
        return self._CAPABILITIES

    def dispatch(
        self,
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
        """Delegate to the audited NumPy conditional dispatcher."""
        if routing_granularity not in self.capabilities.routing_granularities:
            raise DynamicDispatchContractError(
                f"backend {self.capabilities.name!r} does not support routing granularity "
                f"{routing_granularity!r}"
            )
        return dispatch_numpy_experts(
            x,
            routing_weights,
            expert_runners,
            top_k=top_k,
            routing_granularity=routing_granularity,
            dynamic_threshold=dynamic_threshold,
            zero_tolerance=zero_tolerance,
            require_reduction=require_reduction,
        )


__all__ = (
    "ConditionalDispatchBackend",
    "DispatchBackendCapabilities",
    "NumpyConditionalDispatchBackend",
)
