"""Normalize heterogeneous YOLO-Master routing snapshots into one schema."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np

SCHEMA_VERSION = "e3.routing/v1.0.0"
ADAPTER_VERSION = "0.1.0"
SUPPORTED_FAMILIES = frozenset({"moe", "mot", "latent"})


def to_jsonable(value: Any) -> Any:
    """Detach tensor-like values and recursively convert them to JSON values."""

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
        if getattr(value, "ndim", 0) == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _finite_vector(value: Any) -> np.ndarray:
    vector = np.asarray(to_jsonable(value), dtype=np.float64).reshape(-1)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def routing_metrics(expert_usage: Any) -> dict[str, Any]:
    """Compute normalized load, entropy and concentration from expert usage."""

    usage = np.clip(_finite_vector(expert_usage), 0.0, None)
    total = float(usage.sum())
    load = usage / total if total > 0.0 else np.zeros_like(usage)
    positive = load[load > 0.0]
    entropy = float(-(positive * np.log(positive)).sum()) if positive.size else 0.0
    max_entropy = math.log(load.size) if load.size > 1 else 0.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0.0 else 0.0
    if total > 0.0 and usage.size:
        ordered = np.sort(usage)
        index = np.arange(1, ordered.size + 1, dtype=np.float64)
        gini = 2.0 * np.sum(index * ordered) / (ordered.size * ordered.sum())
        gini -= (ordered.size + 1.0) / ordered.size
        gini = float(np.clip(gini, 0.0, 1.0))
    else:
        gini = 0.0
    return {
        "expert_load": load.tolist(),
        "expert_load_sum": float(load.sum()),
        "entropy_nats": entropy,
        "entropy_normalized": normalized_entropy,
        "load_gini": gini,
        "dominant_expert_share": float(load.max()) if load.size else 0.0,
    }


def _field_state(snapshot: dict[str, Any], names: tuple[str, ...], *, fallback: Any = None) -> dict[str, Any]:
    for name in names:
        if name in snapshot and snapshot[name] is not None:
            return {"state": "observed", "source": name, "value": to_jsonable(snapshot[name])}
    if fallback is not None:
        return {"state": "derived", "source": "expert_load", "value": to_jsonable(fallback)}
    return {"state": "unavailable", "source": None, "value": None}


def infer_family(module: Any) -> str | None:
    """Infer one supported family from module metadata without class-name allowlists."""

    declared = str(getattr(module, "_routing_aux_kind", "")).lower()
    module_path = module.__class__.__module__.lower()
    class_name = module.__class__.__name__.lower()
    if declared in SUPPORTED_FAMILIES:
        return declared
    if "latent_mixture" in module_path or "latent" in class_name:
        return "latent"
    if ".mot." in module_path or class_name.endswith("mot") or "motblock" in class_name:
        return "mot"
    if ".moe." in module_path or "moe" in class_name:
        return "moe"
    return None


def adapt_snapshot(
    *,
    family: str,
    run_id: str,
    sequence: int,
    module_name: str,
    module: Any,
    snapshot: dict[str, Any],
    input_shape: list[int] | None,
    output_shapes: list[list[int]],
    runtime_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one schema-valid event from a detached source snapshot."""

    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"Unsupported routing family: {family}")
    usage_source = "expert_usage" if snapshot.get("expert_usage") is not None else "mean_router_probs"
    usage = snapshot.get(usage_source)
    if usage is None:
        raise ValueError(f"{module_name} snapshot has no expert usage vector")
    metrics = routing_metrics(usage)
    num_experts = int(snapshot.get("num_experts", getattr(module, "num_experts", len(metrics["expert_load"]))))
    top_k = int(snapshot.get("top_k", getattr(module, "top_k", 0)))
    if num_experts != len(metrics["expert_load"]):
        raise ValueError(
            f"{module_name} num_experts={num_experts} differs from load length={len(metrics['expert_load'])}"
        )
    mixing = _field_state(snapshot, ("mean_topk_weight", "mean_router_probs", "value_fusion_weights"), fallback=metrics["expert_load"])
    aux = _field_state(snapshot, ("aux_loss",))
    aux_value = aux["value"]
    aux_number = float(aux_value) if isinstance(aux_value, (int, float)) else None
    aux.update(finite=math.isfinite(aux_number) if aux_number is not None else None, mode="eval_observation")
    granularity = {
        "moe": "module-defined token/spatial routing",
        "mot": "spatial-token routing",
        "latent": str(snapshot.get("routing_axis", "image/scale routing")),
    }[family]
    runtime = {
        "training": bool(getattr(module, "training", False)),
        "input_shape": input_shape,
        "output_shapes": output_shapes,
    }
    if runtime_context:
        runtime.update(to_jsonable(runtime_context))
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "sequence": int(sequence),
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "family": family,
        "module": {
            "name": module_name,
            "type": module.__class__.__name__,
            "python_module": module.__class__.__module__,
        },
        "routing": {
            "num_experts": num_experts,
            "top_k": top_k,
            "granularity": granularity,
            **metrics,
            "mixing_weights": mixing,
        },
        "aux_loss": aux,
        "runtime": runtime,
        "provenance": {
            "adapter_version": ADAPTER_VERSION,
            "selection_policy": "leaf_routed_modules",
            "usage_source": usage_source,
            "source_snapshot_keys": sorted(snapshot),
        },
        "source_snapshot": to_jsonable(snapshot),
    }
