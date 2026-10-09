"""Executable dynamic DAG with explicit conditional expert-dispatch nodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from .backend import ConditionalDispatchBackend, NumpyConditionalDispatchBackend
from .dispatch import DynamicDispatchAudit, DynamicDispatchContractError


DYNAMIC_DAG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DAGCallableNode:
    """Ordinary DAG node resolved through a named callable registry."""

    name: str
    runner: str
    inputs: tuple[str, ...]
    output: str
    kind: str = "callable"

    def to_dict(self) -> dict:
        """Return a JSON-serializable node description."""
        return asdict(self)


@dataclass(frozen=True)
class DAGDispatchNode:
    """DAG node whose expert branches are selected at runtime."""

    name: str
    input: str
    routing_weights: str
    output: str
    expert_runners: tuple[str, ...]
    top_k: int
    routing_granularity: str
    dynamic_threshold: float = 0.0
    zero_tolerance: float = 1e-8
    require_reduction: bool = False
    kind: str = "conditional_dispatch"

    def to_dict(self) -> dict:
        """Return a JSON-serializable node description."""
        return asdict(self)


DynamicDAGNode = DAGCallableNode | DAGDispatchNode


@dataclass(frozen=True)
class DynamicDAG:
    """Topologically ordered dynamic graph with no masked-dense fallback."""

    inputs: tuple[str, ...]
    nodes: tuple[DynamicDAGNode, ...]
    outputs: tuple[str, ...]
    name: str = "dynamic_expert_dag"
    schema_version: int = DYNAMIC_DAG_SCHEMA_VERSION
    masked_dense_allowed: bool = False

    def __post_init__(self) -> None:
        """Validate dependencies and the conditional-execution contract."""
        if self.schema_version != DYNAMIC_DAG_SCHEMA_VERSION:
            raise DynamicDispatchContractError(
                f"unsupported dynamic DAG schema {self.schema_version}; expected {DYNAMIC_DAG_SCHEMA_VERSION}"
            )
        if self.masked_dense_allowed:
            raise DynamicDispatchContractError("dynamic DAG forbids masked-dense fallback")
        if not self.inputs:
            raise DynamicDispatchContractError("dynamic DAG requires at least one input")
        if len(set(self.inputs)) != len(self.inputs):
            raise DynamicDispatchContractError("dynamic DAG input names must be unique")
        if not self.outputs:
            raise DynamicDispatchContractError("dynamic DAG requires at least one output")

        available = set(self.inputs)
        node_names: set[str] = set()
        for node in self.nodes:
            if not isinstance(node, (DAGCallableNode, DAGDispatchNode)):
                raise DynamicDispatchContractError(f"unsupported dynamic DAG node: {type(node).__name__}")
            if node.name in node_names:
                raise DynamicDispatchContractError(f"duplicate dynamic DAG node name: {node.name!r}")
            node_names.add(node.name)
            dependencies = node.inputs if isinstance(node, DAGCallableNode) else (node.input, node.routing_weights)
            missing = [name for name in dependencies if name not in available]
            if missing:
                raise DynamicDispatchContractError(
                    f"node {node.name!r} uses values before they are produced: {missing}"
                )
            if node.output in available:
                raise DynamicDispatchContractError(f"dynamic DAG value {node.output!r} is produced more than once")
            if isinstance(node, DAGDispatchNode):
                if not node.expert_runners:
                    raise DynamicDispatchContractError(f"dispatch node {node.name!r} has no experts")
                if not 1 <= node.top_k <= len(node.expert_runners):
                    raise DynamicDispatchContractError(
                        f"dispatch node {node.name!r} top_k={node.top_k} is invalid for "
                        f"{len(node.expert_runners)} experts"
                    )
            available.add(node.output)

        missing_outputs = [name for name in self.outputs if name not in available]
        if missing_outputs:
            raise DynamicDispatchContractError(f"dynamic DAG outputs are not produced: {missing_outputs}")

    def to_dict(self) -> dict:
        """Return a portable, JSON-serializable execution plan."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "execution_semantics": "runtime_conditional_dag",
            "masked_dense_allowed": self.masked_dense_allowed,
            "inputs": list(self.inputs),
            "nodes": [node.to_dict() for node in self.nodes],
            "outputs": list(self.outputs),
        }


class DynamicDAGExecutor:
    """Execute a :class:`DynamicDAG` through a custom dispatch backend."""

    def __init__(
        self,
        dag: DynamicDAG,
        runners: Mapping[str, Callable[..., np.ndarray]],
        *,
        dispatch_backend: ConditionalDispatchBackend | None = None,
    ) -> None:
        self.dag = dag
        self.runners = dict(runners)
        self.dispatch_backend = dispatch_backend or NumpyConditionalDispatchBackend()
        capabilities = self.dispatch_backend.capabilities
        if not capabilities.conditional_execution:
            raise DynamicDispatchContractError(
                f"backend {capabilities.name!r} does not guarantee conditional expert execution"
            )
        if not capabilities.emits_execution_audit:
            raise DynamicDispatchContractError(f"backend {capabilities.name!r} does not emit execution audits")
        required_runners = {
            node.runner
            for node in dag.nodes
            if isinstance(node, DAGCallableNode)
        }
        required_runners.update(
            runner
            for node in dag.nodes
            if isinstance(node, DAGDispatchNode)
            for runner in node.expert_runners
        )
        missing = sorted(required_runners.difference(self.runners))
        if missing:
            raise DynamicDispatchContractError(f"dynamic DAG runner registry is missing: {missing}")
        self.last_dispatch_audits: dict[str, DynamicDispatchAudit] = {}

    def run(self, feeds: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Execute the graph and return its named outputs."""
        missing_inputs = [name for name in self.dag.inputs if name not in feeds]
        if missing_inputs:
            raise DynamicDispatchContractError(f"dynamic DAG feeds are missing: {missing_inputs}")
        state = {name: np.asarray(feeds[name]) for name in self.dag.inputs}
        self.last_dispatch_audits = {}
        for node in self.dag.nodes:
            if isinstance(node, DAGCallableNode):
                state[node.output] = np.asarray(self.runners[node.runner](*(state[name] for name in node.inputs)))
                continue
            experts: Sequence[Callable[[np.ndarray], np.ndarray]] = tuple(
                self.runners[name] for name in node.expert_runners
            )
            output, audit = self.dispatch_backend.dispatch(
                state[node.input],
                state[node.routing_weights],
                experts,
                top_k=node.top_k,
                routing_granularity=node.routing_granularity,
                dynamic_threshold=node.dynamic_threshold,
                zero_tolerance=node.zero_tolerance,
                require_reduction=node.require_reduction,
            )
            state[node.output] = output
            self.last_dispatch_audits[node.name] = audit
        return {name: state[name] for name in self.dag.outputs}

    def execution_summary(self) -> dict:
        """Return backend capabilities and audits from the most recent call."""
        return {
            "dag": self.dag.to_dict(),
            "backend": asdict(self.dispatch_backend.capabilities),
            "dispatch_audits": {name: audit.to_dict() for name, audit in self.last_dispatch_audits.items()},
        }


__all__ = (
    "DAGCallableNode",
    "DAGDispatchNode",
    "DYNAMIC_DAG_SCHEMA_VERSION",
    "DynamicDAG",
    "DynamicDAGExecutor",
    "DynamicDAGNode",
)
