"""Internal visual-routing helpers for the gated MoE family."""

from typing import Tuple

import torch
import torch.nn.functional as F

from ._helpers import _record_moe_snapshot, _registry_set


def pool_to_size_mps_safe(x: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
    """Pool to a target spatial size without hitting MPS adaptive-pool limits."""
    h, w = output_size
    height, width = x.shape[-2:]
    if (height, width) == (h, w):
        return x
    if x.device.type != "mps":
        return F.adaptive_avg_pool2d(x, (h, w))

    if height % h == 0 and width % w == 0:
        kernel = (height // h, width // w)
        return F.avg_pool2d(x, kernel_size=kernel, stride=kernel)

    pad_h = ((height + h - 1) // h) * h - height
    pad_w = ((width + w - 1) // w) * w - width
    pooled_source = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate") if pad_h or pad_w else x
    padded_height, padded_width = pooled_source.shape[-2:]
    kernel = (padded_height // h, padded_width // w)
    return F.avg_pool2d(pooled_source, kernel_size=kernel, stride=kernel)


def run_visual_hybrid_moe_forward(module, x, detail_gate=None, context_mixer=None, refine_features=False):
    """Run the shared forward path used by visual gated-MoE variants."""
    if module.training:
        module._update_temperature()
        module.training_step += 1
        module._training_step_value += 1

    gate_weights = module.se_gate(x)
    gate_static = gate_weights[:, : module.static_channels].unsqueeze(-1).unsqueeze(-1)
    gate_dynamic = gate_weights[:, module.static_channels :].unsqueeze(-1).unsqueeze(-1)

    x_static = x[:, : module.static_channels, :, :] * gate_static
    x_dynamic = x[:, module.static_channels :, :, :] * gate_dynamic
    configured_hooks = bool(getattr(module, "router_hooks", ()))
    if configured_hooks:
        x_dynamic = module._apply_router_hooks("pre_route", x_dynamic)
    elif detail_gate is not None:
        x_dynamic = detail_gate(x_dynamic)

    out_static = module.static_net(x_static)
    complexity = module._safe_complexity(x_dynamic)

    routing_weights, routing_indices, routing_stats = module.routing(x_dynamic)
    routing_weights, routing_indices, routing_stats, adaptive_top_k = module._apply_complexity_gate(
        routing_weights, routing_indices, routing_stats, complexity
    )
    out_dynamic = module.fused_experts(x_dynamic, routing_weights, routing_indices, adaptive_top_k)

    out_concat = module._channel_shuffle(torch.cat([out_static, out_dynamic], dim=1))
    if configured_hooks:
        out_concat = module._apply_router_hooks("post_fusion", out_concat)
    elif context_mixer is not None:
        out_concat = context_mixer(out_concat)
    if not configured_hooks and refine_features and hasattr(module, "_refine_features"):
        out_concat = module._refine_features(out_concat)

    out = module.bn(module.proj(out_concat)) + x

    if module.training:
        router_probs = routing_stats.get("router_probs")
        router_logits = routing_stats.get("router_logits")
        topk_indices = routing_stats.get("topk_indices")
        if isinstance(router_probs, torch.Tensor) and isinstance(router_logits, torch.Tensor):
            aux_loss = module.moe_loss_fn(router_probs, router_logits, topk_indices)
            _registry_set(module, aux_loss)
            _record_moe_snapshot(
                module,
                expert_usage=routing_stats.get("expert_usage"),
                topk_indices=topk_indices,
                topk_weights=routing_weights,
                router_probs=router_probs,
                aux_loss=aux_loss,
            )

    return out


__all__ = ("pool_to_size_mps_safe", "run_visual_hybrid_moe_forward")
