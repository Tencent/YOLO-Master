"""True conditional expert export and host-runtime primitives."""

from .authority import (
    EAGER_CHECKPOINT_ROUTE_DIAGNOSTIC,
    EXPORTED_ROUTER_HOST_TOPK_AUTHORITY,
    ROUTE_GATE_EXACT_REFERENCE,
    ROUTE_GATE_EXPORTED_AUTHORITATIVE,
    ROUTE_GATE_REPORT_ONLY,
    SUPPORTED_ROUTE_GATE_MODES,
    evaluate_route_gate,
)
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
    "EAGER_CHECKPOINT_ROUTE_DIAGNOSTIC",
    "EXPORTED_ROUTER_HOST_TOPK_AUTHORITY",
    "NumpyConditionalDispatchBackend",
    "ORTDynamicExpertRuntime",
    "ORTDynamicBlockAdapter",
    "ORTRouterTorchExpertAdapter",
    "ROUTE_GATE_EXACT_REFERENCE",
    "ROUTE_GATE_EXPORTED_AUTHORITATIVE",
    "ROUTE_GATE_REPORT_ONLY",
    "SUPPORTED_ROUTE_GATE_MODES",
    "audit_sparse_routing",
    "dispatch_numpy_experts",
    "export_dynamic_expert_bundle",
    "evaluate_route_gate",
    "sparsify_topk_probabilities",
)
