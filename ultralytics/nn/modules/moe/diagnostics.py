"""Utilities for lightweight MoE routing diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .protocol import routing_metrics, usage_gini


@dataclass
class MoELayerDiagnostic:
    """Structured routing summary for a single MoE layer."""

    name: str
    module_type: str
    num_experts: int
    top_k: int
    aux_loss: float
    usage: list[float]
    counts: list[float]
    dominant_expert: int
    dominant_share: float
    mean_router_probs: list[float] | None
    mean_topk_weight: list[float] | None
    collapse_flag: bool


def _tensor_to_list(value: Any) -> list[float] | None:
    if not isinstance(value, torch.Tensor):
        return None
    return [float(x) for x in value.detach().cpu().reshape(-1).tolist()]


def collect_moe_diagnostics(model: torch.nn.Module, collapse_threshold: float = 0.8) -> list[MoELayerDiagnostic]:
    """Collect diagnostics from MoE layers that expose `last_routing_snapshot`."""
    diagnostics: list[MoELayerDiagnostic] = []

    for name, module in model.named_modules():
        snapshot = getattr(module, "last_routing_snapshot", None)
        num_experts = int(getattr(module, "num_experts", 0))
        if not snapshot or num_experts <= 0:
            continue

        metrics = routing_metrics(snapshot, num_experts=num_experts, top_k=int(getattr(module, "top_k", 0)))
        usage = metrics.expert_usage
        counts = metrics.topk_counts
        dominant_share = metrics.dominant_share
        dominant_expert = metrics.dominant_expert

        diagnostics.append(
            MoELayerDiagnostic(
                name=name,
                module_type=type(module).__name__,
                num_experts=num_experts,
                top_k=metrics.top_k,
                aux_loss=metrics.aux_loss,
                usage=usage,
                counts=counts,
                dominant_expert=dominant_expert,
                dominant_share=dominant_share,
                mean_router_probs=_tensor_to_list(snapshot.get("mean_router_probs")),
                mean_topk_weight=_tensor_to_list(snapshot.get("mean_topk_weight")),
                collapse_flag=dominant_share >= collapse_threshold,
            )
        )

    return diagnostics


def diagnostics_to_dict(diagnostics: list[MoELayerDiagnostic]) -> list[dict[str, Any]]:
    """Convert diagnostics to JSON-serializable dictionaries."""
    return [diag.__dict__.copy() for diag in diagnostics]


def routing_runtime_metrics(model: torch.nn.Module, collapse_threshold: float = 0.8) -> dict[str, Any]:
    """Return JSON-safe routing health and dispatch metrics after a forward."""
    layers: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        snapshot = getattr(module, "last_routing_snapshot", None)
        if not isinstance(snapshot, dict) or not snapshot:
            continue
        usage = _tensor_to_list(snapshot.get("expert_usage"))
        if not usage:
            continue
        usage_tensor = torch.tensor(usage, dtype=torch.float32).clamp_min(1e-12)
        dispatch = getattr(module, "_last_dispatch_stats", {}) or {}
        if not dispatch:
            sparse = bool(
                not module.training
                and getattr(module, "use_sparse_inference", False)
                and int(snapshot.get("top_k", 0)) < len(usage)
            )
            dispatch = {
                "mode": "sparse" if sparse else "dense",
                "expert_calls": int(snapshot.get("top_k", len(usage))) if sparse else len(usage),
            }
        layers[name] = {
            "module_type": type(module).__name__,
            "num_experts": int(snapshot.get("num_experts", len(usage))),
            "top_k": int(snapshot.get("top_k", getattr(module, "top_k", 0))),
            "expert_usage": usage,
            "gini": usage_gini(usage),
            "entropy": float((-usage_tensor * torch.log(usage_tensor)).sum()),
            "dominant_share": max(usage),
            "collapse_flag": max(usage) >= float(collapse_threshold),
            "dispatch_mode": dispatch.get("mode"),
            "expert_calls": dispatch.get("expert_calls"),
        }
    values = list(layers.values())
    return {
        "layers": layers,
        "routed_layers": len(values),
        "collapsed_layers": sum(bool(item["collapse_flag"]) for item in values),
        "mean_gini": sum(float(item["gini"]) for item in values) / len(values) if values else 0.0,
        "mean_dominant_share": sum(float(item["dominant_share"]) for item in values) / len(values) if values else 0.0,
        "expert_calls": sum(int(item["expert_calls"] or 0) for item in values),
    }


def format_moe_diagnostics(diagnostics: list[MoELayerDiagnostic], title: str = "MoE Routing Diagnostics") -> str:
    """Render a compact text summary shared by CLI and training callbacks."""
    lines = [f"[MoE] {title}"]
    if not diagnostics:
        lines.append("[MoE] No routing snapshots collected yet.")
        return "\n".join(lines)

    for diag in diagnostics:
        usage_str = ", ".join(f"E{i}:{share:.3f}" for i, share in enumerate(diag.usage))
        counts_str = ", ".join(f"E{i}:{int(count)}" for i, count in enumerate(diag.counts))
        line = (
            f"[MoE] {diag.name} | aux={diag.aux_loss:.6f} | top_k={diag.top_k}/{diag.num_experts} | "
            f"dominant=E{diag.dominant_expert}({diag.dominant_share:.3f}) | collapse={diag.collapse_flag}"
        )
        lines.append(line)
        lines.append(f"[MoE] usage  {usage_str}")
        lines.append(f"[MoE] counts {counts_str}")
        if diag.mean_topk_weight:
            topk_str = ", ".join(f"k{i}:{weight:.3f}" for i, weight in enumerate(diag.mean_topk_weight))
            lines.append(f"[MoE] topk   {topk_str}")

    return "\n".join(lines)
