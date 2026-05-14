from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .contract import ensure_manifest_dir, json_safe


@dataclass(frozen=True)
class LoraDiagnoseDeps:
    build_model: Callable[[dict[str, Any]], Any]
    is_dry_run: Callable[[dict[str, Any]], bool]
    response: Callable[..., dict[str, Any]]
    plan_response: Callable[..., dict[str, Any]]
    write_manifest: Callable[[dict[str, Any], dict[str, Any]], Any]


def _unwrap_model(model: Any) -> Any:
    try:
        from ultralytics.utils.torch_utils import unwrap_model

        return unwrap_model(model)
    except Exception:
        return model


def _weights(attr: Any) -> list[tuple[str, Any]]:
    if attr is None:
        return []
    try:
        import torch.nn as nn
    except Exception:
        nn = None
    if nn is not None and isinstance(attr, nn.ModuleDict):
        return [(str(name), child.weight) for name, child in attr.items() if hasattr(child, "weight")]
    if isinstance(attr, dict):
        return [(str(name), child.weight) for name, child in attr.items() if hasattr(child, "weight")]
    if hasattr(attr, "weight"):
        return [("default", attr.weight)]
    return []


def _adapter_value(value: Any, adapter: str, default: float | None = None) -> float | None:
    if isinstance(value, dict):
        value = value.get(adapter, value.get("default", default))
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _scaling(module: Any, adapter: str, rank: int) -> float:
    scaling = _adapter_value(getattr(module, "scaling", None), adapter)
    if scaling is not None:
        return scaling
    alpha = _adapter_value(getattr(module, "lora_alpha", None), adapter)
    if alpha is None:
        return 1.0
    return alpha / max(rank, 1)


def _delta_matrix(a_weight: Any, b_weight: Any, scaling: float) -> Any | None:
    try:
        import torch
    except Exception:
        return None
    try:
        a = a_weight.detach().float().cpu()
        b = b_weight.detach().float().cpu()
        if a.dim() < 2 or b.dim() < 2:
            return None
        a2 = a.reshape(a.shape[0], -1)
        if b.dim() > 2:
            b2 = b.reshape(b.shape[0], b.shape[1], -1).mean(dim=-1)
        else:
            b2 = b.reshape(b.shape[0], b.shape[1])
        if b2.shape[1] != a2.shape[0]:
            return None
        return torch.matmul(b2, a2) * float(scaling)
    except Exception:
        return None


def _spectrum_for_delta(delta: Any, *, topk: int) -> dict[str, Any] | None:
    try:
        import torch

        singular = torch.linalg.svdvals(delta)
        if singular.numel() == 0:
            return None
        singular = singular.float()
        energy = singular.square()
        energy_sum = energy.sum()
        if energy_sum.item() <= 0:
            return None
        normalized = energy / energy_sum
        entropy = -(normalized * torch.log(normalized.clamp_min(1e-12))).sum()
        effective_rank = float(torch.exp(entropy).item())
        threshold_rank = int((singular > singular[0] * 0.01).sum().item()) if singular[0].item() > 0 else 0
        return {
            "shape": list(delta.shape),
            "rank_1pct": threshold_rank,
            "effective_rank_entropy": round(effective_rank, 6),
            "singular_values_topk": [round(float(v), 6) for v in singular[:topk].tolist()],
            "energy_topk": [round(float(v), 6) for v in normalized[:topk].tolist()],
        }
    except Exception:
        return None


def delta_weight_spectrum(model: Any, *, max_layers: int = 12, topk: int = 8) -> dict[str, Any]:
    layers = []
    skipped = 0
    for name, module in model.named_modules():
        a_weights = _weights(getattr(module, "lora_A", None))
        b_weights = dict(_weights(getattr(module, "lora_B", None)))
        for adapter, a_weight in a_weights:
            b_weight = b_weights.get(adapter)
            if b_weight is None:
                b_weight = b_weights.get("default")
            if b_weight is None:
                skipped += 1
                continue
            rank = int(getattr(a_weight, "shape", [0])[0] or 0)
            delta = _delta_matrix(a_weight, b_weight, _scaling(module, adapter, rank))
            if delta is None:
                skipped += 1
                continue
            spectrum = _spectrum_for_delta(delta, topk=topk)
            if spectrum is None:
                skipped += 1
                continue
            layers.append({"module": name, "adapter": adapter, **spectrum})
            if len(layers) >= max_layers:
                break
        if len(layers) >= max_layers:
            break
    if not layers:
        return {"layers": [], "summary": {"sampled_layers": 0, "skipped_layers": skipped}}
    return {
        "layers": layers,
        "summary": {
            "sampled_layers": len(layers),
            "skipped_layers": skipped,
            "effective_rank_entropy_avg": round(
                sum(layer["effective_rank_entropy"] for layer in layers) / len(layers),
                6,
            ),
            "rank_1pct_avg": round(sum(layer["rank_1pct"] for layer in layers) / len(layers), 6),
        },
    }


def run_lora_diagnose(request: dict[str, Any], deps: LoraDiagnoseDeps) -> dict[str, Any]:
    params = dict(request["params"])
    adapter_path = request["inputs"].get("path") or request["inputs"].get("adapter") or params.get("path") or params.get("adapter_path")
    svd_sample_ratio = float(params.get("svd_sample_ratio", 0.2))
    svd_max_layers = int(params.get("svd_max_layers", 20))
    spectrum_max_layers = int(params.get("spectrum_max_layers", 12))
    spectrum_topk = int(params.get("spectrum_topk", 8))

    if deps.is_dry_run(request):
        return deps.plan_response(
            request,
            "LoRA diagnose dry run prepared",
            "python_api",
            "ultralytics.utils.lora.get_lora_training_stats + delta-W spectrum",
            params={
                "adapter_path": adapter_path,
                "svd_sample_ratio": svd_sample_ratio,
                "svd_max_layers": svd_max_layers,
                "spectrum_max_layers": spectrum_max_layers,
                "spectrum_topk": spectrum_topk,
            },
            next_actions=["yolo.eval.peft_compare", "yolo.lora.train"],
        )

    model = deps.build_model(request)
    adapter_loaded = None
    if adapter_path:
        adapter_loaded = bool(model.load_lora(adapter_path, merge=bool(params.get("merge", False))))
    base_model = _unwrap_model(getattr(model, "model", model))

    from ultralytics.utils.lora import get_lora_training_stats

    stats = get_lora_training_stats(base_model, svd_sample_ratio=svd_sample_ratio, svd_max_layers=svd_max_layers)
    spectrum = delta_weight_spectrum(base_model, max_layers=spectrum_max_layers, topk=spectrum_topk)
    output_dir = ensure_manifest_dir(request) / "lora_diagnose"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "lora_diagnose.json"
    report = {
        "adapter_path": str(Path(adapter_path).resolve()) if adapter_path else None,
        "adapter_loaded": adapter_loaded,
        "stats": json_safe(stats),
        "delta_weight_spectrum": spectrum,
    }
    report_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    payload = deps.response(
        request["skill"],
        "ok",
        "LoRA diagnosis finished",
        data=report,
        metrics={"effective_rank_avg": stats.get("effective_rank_avg"), "lora_modules": stats.get("lora_modules")},
        artifacts=[{"kind": "json", "path": str(report_path.resolve())}],
        next_actions=["yolo.eval.peft_compare", "yolo.lora.train"],
    )
    payload["manifest"] = str(deps.write_manifest(request, payload))
    return payload
