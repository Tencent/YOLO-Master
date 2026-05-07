"""Utilities for lightweight MoE routing diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


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

        usage = _tensor_to_list(snapshot.get("expert_usage")) or [0.0] * num_experts
        counts = _tensor_to_list(snapshot.get("topk_counts")) or [0.0] * num_experts
        dominant_share = max(usage) if usage else 0.0
        dominant_expert = int(max(range(len(usage)), key=usage.__getitem__)) if usage else -1

        diagnostics.append(
            MoELayerDiagnostic(
                name=name,
                module_type=type(module).__name__,
                num_experts=num_experts,
                top_k=int(snapshot.get("top_k", getattr(module, "top_k", 0))),
                aux_loss=float(snapshot.get("aux_loss", 0.0)),
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
