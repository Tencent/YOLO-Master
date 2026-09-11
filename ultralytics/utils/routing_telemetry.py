"""Cross-family routing telemetry normalization for diagnostics and loggers."""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

import torch

E3_ROUTING_SCHEMA_VERSION = "e3.routing_snapshot.v1"
SUPPORTED_ROUTING_FAMILIES = frozenset({"latent", "moe", "mot"})
AUX_STATUS_CODES = {
    "not_configured": 0.0,
    "configured_inactive_eval": 1.0,
    "configured_zero_observed": 2.0,
    "active_training": 3.0,
    "available": 4.0,
}


class RoutingSnapshotError(ValueError):
    """Raised when a routed module publishes a malformed E3 snapshot."""


def routing_module_family(module: torch.nn.Module, snapshot: dict[str, Any] | None = None) -> str:
    """Return the family declared by a routed module, or ``unknown`` when unsupported."""
    snapshot = snapshot if isinstance(snapshot, dict) else getattr(module, "last_routing_snapshot", {})
    if isinstance(snapshot, dict) and snapshot.get("family"):
        return str(snapshot["family"]).strip().lower()
    kind = getattr(module, "_routing_aux_kind", None)
    if kind:
        return str(kind).strip().lower()
    module_path = f"{type(module).__module__}.{type(module).__name__}".lower()
    return next((family for family in ("latent", "mot", "moa", "molora", "moe") if family in module_path), "unknown")


def _finite_scalar(value: Any, *, field: str) -> float:
    """Return one finite scalar or raise a field-specific validation error."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RoutingSnapshotError(f"{field} must be scalar, got shape={tuple(value.shape)}")
        value = value.detach().float().item()
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RoutingSnapshotError(f"{field} must be numeric, got {type(value).__name__}") from exc
    if not math.isfinite(result):
        raise RoutingSnapshotError(f"{field} must be finite, got {result}")
    return result


def _probability_vector(value: Any, *, size: int, field: str) -> list[float]:
    """Validate and normalize one exact-length non-negative probability vector."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().float().reshape(-1).cpu()
    else:
        try:
            tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise RoutingSnapshotError(f"{field} must be a numeric vector") from exc
    if tensor.numel() != size:
        raise RoutingSnapshotError(f"{field} length={tensor.numel()} does not match num_experts={size}")
    if not bool(torch.isfinite(tensor).all().item()):
        raise RoutingSnapshotError(f"{field} contains non-finite values")
    if bool((tensor < 0).any().item()):
        raise RoutingSnapshotError(f"{field} contains negative values")
    total = float(tensor.sum().item())
    if total <= 0.0:
        raise RoutingSnapshotError(f"{field} must have a positive sum")
    return [float(item) for item in (tensor / total).tolist()]


def _mean_probabilities(probabilities: torch.Tensor, *, num_experts: int) -> tuple[list[float], list[int]]:
    """Reduce a routing tensor whose expert axis is dimension one."""
    if not isinstance(probabilities, torch.Tensor) or probabilities.ndim < 2:
        raise RoutingSnapshotError("probabilities must be a tensor with expert axis at dimension 1")
    if probabilities.shape[1] != num_experts:
        raise RoutingSnapshotError(
            f"probabilities expert axis={probabilities.shape[1]} does not match num_experts={num_experts}"
        )
    detached = probabilities.detach().float()
    if not bool(torch.isfinite(detached).all().item()):
        raise RoutingSnapshotError("probabilities contains non-finite values")
    if bool((detached < 0).any().item()):
        raise RoutingSnapshotError("probabilities contains negative values")
    reduce_dims = tuple(index for index in range(detached.ndim) if index != 1)
    mean = detached.mean(dim=reduce_dims) if reduce_dims else detached
    return _probability_vector(mean, size=num_experts, field="mean_router_probs"), list(detached.shape)


def routing_aux_status(module: torch.nn.Module, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Describe auxiliary-loss configuration without presenting an eval zero as training evidence."""
    balance = _finite_scalar(getattr(module, "balance_loss_coeff", 0.0) or 0.0, field="balance_loss_coeff")
    z_loss = _finite_scalar(getattr(module, "router_z_loss_coeff", 0.0) or 0.0, field="router_z_loss_coeff")
    observed = _finite_scalar(snapshot.get("aux_loss", 0.0) or 0.0, field="aux_loss")
    configured = balance > 0.0 or z_loss > 0.0
    capabilities = module.export_capabilities() if callable(getattr(module, "export_capabilities", None)) else {}
    training_only = bool(capabilities.get("aux_loss_training_only", True))
    if not configured:
        status = "not_configured"
    elif module.training:
        status = "active_training" if observed != 0.0 else "configured_zero_observed"
    elif training_only:
        status = "configured_inactive_eval"
    else:
        status = "available"
    return {
        "status": status,
        "status_code": AUX_STATUS_CODES[status],
        "observed": observed,
        "configured": configured,
        "balance_loss_coeff": balance,
        "router_z_loss_coeff": z_loss,
        "training_only": training_only,
    }


def normalize_routing_layer(
    name: str,
    module: torch.nn.Module,
    *,
    probabilities: torch.Tensor | None = None,
    supported_families: Iterable[str] = SUPPORTED_ROUTING_FAMILIES,
) -> dict[str, Any]:
    """Normalize one routed module to the E3 layer schema, raising on invalid evidence."""
    snapshot = getattr(module, "last_routing_snapshot", None)
    if probabilities is None and (not isinstance(snapshot, dict) or not snapshot):
        raise RoutingSnapshotError("last_routing_snapshot is missing or empty")
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    try:
        num_experts = int(snapshot.get("num_experts", getattr(module, "num_experts", 0)))
        top_k = int(snapshot.get("top_k", getattr(module, "top_k", 0)))
    except (TypeError, ValueError) as exc:
        raise RoutingSnapshotError("num_experts and top_k must be integers") from exc
    if num_experts <= 0:
        raise RoutingSnapshotError(f"num_experts must be positive, got {num_experts}")
    if not 0 < top_k <= num_experts:
        raise RoutingSnapshotError(f"top_k={top_k} must be in [1, {num_experts}]")

    family = routing_module_family(module, snapshot)
    supported = {str(item).lower() for item in supported_families}
    if family not in supported:
        raise RoutingSnapshotError(f"unsupported routing family={family!r}")

    if probabilities is not None:
        mean_probs, probability_shape = _mean_probabilities(probabilities, num_experts=num_experts)
    else:
        raw_mean = snapshot.get("mean_router_probs")
        if raw_mean is None:
            raw_mean = snapshot.get("expert_usage")
        mean_probs = _probability_vector(raw_mean, size=num_experts, field="mean_router_probs")
        probability_shape = list(snapshot.get("probability_shape", [num_experts]))
    raw_usage = snapshot.get("expert_usage")
    usage = _probability_vector(
        mean_probs if raw_usage is None else raw_usage,
        size=num_experts,
        field="expert_usage",
    )
    probability_tensor = torch.tensor(mean_probs, dtype=torch.float32)
    entropy = float((-(probability_tensor * probability_tensor.clamp_min(1e-12).log()).sum()).item())
    normalized_entropy = min(max(entropy / math.log(num_experts), 0.0), 1.0) if num_experts > 1 else 0.0
    dispatch = getattr(module, "_last_dispatch_stats", {}) or {}
    dispatch_policy = snapshot.get("dispatch_policy", snapshot.get("dispatch_mode", dispatch.get("mode", "unknown")))
    return {
        "layer_name": str(name),
        "module_type": type(module).__name__,
        "family": family,
        "num_experts": num_experts,
        "top_k": top_k,
        "routing_axis": str(snapshot.get("routing_axis", "spatial" if len(probability_shape) > 2 else "image")),
        "probability_shape": probability_shape,
        "expert_usage": usage,
        "mean_router_probs": mean_probs,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "dispatch_policy": str(dispatch_policy),
        "aux_loss": routing_aux_status(module, snapshot),
    }


def collect_routing_snapshot(
    model: torch.nn.Module,
    *,
    supported_families: Iterable[str] = SUPPORTED_ROUTING_FAMILIES,
    include_wrappers: bool = False,
) -> dict[str, Any]:
    """Collect valid layers and explicit issues without interrupting training."""
    layers: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, str]] = []
    candidates = [
        (name or "<root>", module)
        for name, module in model.named_modules()
        if isinstance(getattr(module, "last_routing_snapshot", None), dict) and module.last_routing_snapshot
    ]
    candidate_ids = {id(module) for _, module in candidates}
    if not include_wrappers:
        candidates = [
            (name, module)
            for name, module in candidates
            if not any(child is not module and id(child) in candidate_ids for child in module.modules())
        ]
    for layer_name, module in candidates:
        snapshot = getattr(module, "last_routing_snapshot", None)
        try:
            layers[layer_name] = normalize_routing_layer(layer_name, module, supported_families=supported_families)
        except RoutingSnapshotError as exc:
            issues.append(
                {
                    "layer_name": layer_name,
                    "module_type": type(module).__name__,
                    "family": routing_module_family(module, snapshot),
                    "status": "unsupported" if str(exc).startswith("unsupported routing family") else "invalid",
                    "reason": str(exc),
                }
            )
    values = list(layers.values())
    return {
        "schema_version": E3_ROUTING_SCHEMA_VERSION,
        "layers": layers,
        "issues": issues,
        "routed_layers": len(values),
        "invalid_layers": sum(item["status"] == "invalid" for item in issues),
        "unsupported_layers": sum(item["status"] == "unsupported" for item in issues),
        "mean_normalized_entropy": (
            sum(float(item["normalized_entropy"]) for item in values) / len(values) if values else None
        ),
        "configured_aux_layers": sum(bool(item["aux_loss"]["configured"]) for item in values),
        "active_aux_layers": sum(item["aux_loss"]["status"] == "active_training" for item in values),
    }


def _logger_component(value: str) -> str:
    """Return a stable TensorBoard path component."""
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_") or "root"


def routing_snapshot_scalars(payload: dict[str, Any]) -> dict[str, float]:
    """Flatten an E3 snapshot to stable TensorBoard scalar keys."""
    scalars = {
        "routing/global/routed_layers": float(payload.get("routed_layers", 0)),
        "routing/global/invalid_layers": float(payload.get("invalid_layers", 0)),
        "routing/global/unsupported_layers": float(payload.get("unsupported_layers", 0)),
        "routing/global/configured_aux_layers": float(payload.get("configured_aux_layers", 0)),
        "routing/global/active_aux_layers": float(payload.get("active_aux_layers", 0)),
    }
    mean_entropy = payload.get("mean_normalized_entropy")
    if mean_entropy is not None:
        scalars["routing/global/mean_normalized_entropy"] = float(mean_entropy)
    for layer in payload.get("layers", {}).values():
        prefix = f"routing/{_logger_component(layer['family'])}/{_logger_component(layer['layer_name'])}"
        scalars[f"{prefix}/normalized_entropy"] = float(layer["normalized_entropy"])
        scalars[f"{prefix}/aux_observed"] = float(layer["aux_loss"]["observed"])
        scalars[f"{prefix}/aux_status_code"] = float(layer["aux_loss"]["status_code"])
        for index, value in enumerate(layer["expert_usage"]):
            scalars[f"{prefix}/expert_{index}_usage"] = float(value)
        for index, value in enumerate(layer["mean_router_probs"]):
            scalars[f"{prefix}/expert_{index}_mean_probability"] = float(value)
    return scalars


__all__ = (
    "AUX_STATUS_CODES",
    "E3_ROUTING_SCHEMA_VERSION",
    "SUPPORTED_ROUTING_FAMILIES",
    "RoutingSnapshotError",
    "collect_routing_snapshot",
    "normalize_routing_layer",
    "routing_aux_status",
    "routing_module_family",
    "routing_snapshot_scalars",
)
