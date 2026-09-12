"""True conditional expert export and host-runtime primitives."""

from .backend import ConditionalDispatchBackend, DispatchBackendCapabilities, NumpyConditionalDispatchBackend
from .bundle import BUNDLE_SCHEMA_VERSION, export_dynamic_expert_bundle
from .bridge import ORTDynamicBlockAdapter, ORTRouterTorchExpertAdapter
from .dag import DAGCallableNode, DAGDispatchNode, DynamicDAG, DynamicDAGExecutor
from .dispatch import (
    DynamicDispatchAudit,
    DynamicDispatchContractError,
    audit_sparse_routing,
    dispatch_numpy_experts,
    sparsify_topk_probabilities,
)
from .ort import ORTDynamicExpertRuntime

__all__ = (
    "BUNDLE_SCHEMA_VERSION",
    "ConditionalDispatchBackend",
    "DAGCallableNode",
    "DAGDispatchNode",
    "DispatchBackendCapabilities",
    "DynamicDispatchAudit",
    "DynamicDispatchContractError",
    "DynamicDAG",
    "DynamicDAGExecutor",
    "NumpyConditionalDispatchBackend",
    "ORTDynamicExpertRuntime",
    "ORTDynamicBlockAdapter",
    "ORTRouterTorchExpertAdapter",
    "audit_sparse_routing",
    "dispatch_numpy_experts",
    "export_dynamic_expert_bundle",
    "sparsify_topk_probabilities",
)
