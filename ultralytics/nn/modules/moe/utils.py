# 🐧Please note that this file has been modified by Tencent on 2026/01/18. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Utility functions for Mixture-of-Experts models"""
import threading
import weakref
from typing import Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# Namespace for full MoE *block* classes (excludes routers/experts/loss helpers).
_CORE_MOE_MODULES = {
    "ultralytics.nn.modules.moe.modules",
    "ultralytics.nn.modules.moe.base",
    "ultralytics.nn.modules.moe.advanced",
    "ultralytics.nn.modules.moe.hybrid",
    "ultralytics.nn.modules.moe.integration",
}
_CORE_MOE_MODULE_SUFFIXES = tuple(name[len("ultralytics") :] for name in _CORE_MOE_MODULES)
_MOE_USAGE_ACCUMULATORS = weakref.WeakKeyDictionary()
_MOE_USAGE_ACCUMULATORS_LOCK = threading.Lock()


def reset_moe_usage_accumulator(module: nn.Module) -> None:
    """Enable and reset process-local epoch usage accumulation for a module."""
    with _MOE_USAGE_ACCUMULATORS_LOCK:
        _MOE_USAGE_ACCUMULATORS[module] = {"sum": None, "observations": 0}


def moe_usage_accumulator_enabled(module: nn.Module) -> bool:
    """Return whether epoch usage accumulation is enabled for a module."""
    with _MOE_USAGE_ACCUMULATORS_LOCK:
        return module in _MOE_USAGE_ACCUMULATORS


def accumulate_moe_usage(module: nn.Module, usage: torch.Tensor) -> None:
    """Add one detached usage vector to a process-local epoch accumulator."""
    value = usage.detach().float()
    with _MOE_USAGE_ACCUMULATORS_LOCK:
        state = _MOE_USAGE_ACCUMULATORS.get(module)
        if state is None:
            return
        current = state["sum"]
        if not isinstance(current, torch.Tensor) or current.shape != value.shape:
            current = torch.zeros_like(value)
        state["sum"] = current + value
        state["observations"] += 1


def get_moe_usage_accumulator(module: nn.Module) -> Tuple[Optional[torch.Tensor], int]:
    """Return the current detached usage sum and observation count."""
    with _MOE_USAGE_ACCUMULATORS_LOCK:
        state = _MOE_USAGE_ACCUMULATORS.get(module)
        if state is None:
            return None, 0
        return state["sum"], int(state["observations"])


def is_core_moe_block(module: nn.Module) -> bool:
    """Return True for MoE blocks from the facade or its split implementation modules."""
    mod = getattr(module.__class__, "__module__", "") or ""
    return mod in _CORE_MOE_MODULES or mod.endswith(_CORE_MOE_MODULE_SUFFIXES)


def model_has_core_moe(model: nn.Module) -> bool:
    """Return True when the model contains at least one core MoE block."""
    return any(is_core_moe_block(m) for m in model.modules())


def set_core_moe_balance_loss_coeff(module: nn.Module, coeff: float) -> bool:
    """Set balance-loss state on a core MoE block, including legacy unpickled checkpoints."""
    if not is_core_moe_block(module):
        return False
    module.balance_loss_coeff = float(coeff)
    moe_loss_fn = getattr(module, "moe_loss_fn", None)
    if moe_loss_fn is not None and hasattr(moe_loss_fn, "balance_loss_coeff"):
        moe_loss_fn.balance_loss_coeff = float(coeff)
    return True


def iter_core_moe_expert_params(model: nn.Module) -> Iterator[torch.nn.Parameter]:
    """Yield expert-weight parameters that belong to core MoE blocks only.

    Excludes MoT ``experts.*``, MoLoRA ``experts.*``, routers, and shared paths so
    expert-warmup does not freeze unrelated mixture modules.
    """
    for prefix, m in model.named_modules():
        if not is_core_moe_block(m):
            continue
        for name, param in m.named_parameters():
            full = f"{prefix}.{name}" if prefix else name
            if "routing" in full or "router" in full:
                continue
            if "shared" in full:
                continue
            if "expert" in full:
                yield param


def get_safe_groups(channels: int, desired_groups: int = 8) -> int:
    """Ensure num_groups divides channels"""
    if channels <= 0:
        return 1
    groups = min(desired_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return max(1, groups)


def last_conv_out_channels(module: nn.Module) -> int:
    """Return out_channels of the last nn.Conv2d in `module` (layout-agnostic)."""
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Conv2d):
            return layer.out_channels
    raise ValueError(f"No nn.Conv2d found in expert module {type(module).__name__}")


# ==========================================
# Utility: FLOPs calculator (optimized)
# ==========================================
class FlopsUtils:
    @staticmethod
    def count_conv2d(layer: Union[nn.Conv2d, nn.Sequential], input_shape: Tuple[int, int, int, int]) -> float:
        B, C, H, W = input_shape
        if isinstance(layer, nn.Sequential):
            total = 0
            curr_shape = input_shape
            for m in layer:
                if isinstance(m, nn.Conv2d):
                    total += FlopsUtils.count_conv2d(m, curr_shape)
                    # Simple shape derivation
                    curr_h = int((curr_shape[2] + 2 * m.padding[0] - m.kernel_size[0]) / m.stride[0] + 1)
                    curr_w = int((curr_shape[3] + 2 * m.padding[1] - m.kernel_size[1]) / m.stride[1] + 1)
                    curr_shape = (B, m.out_channels, curr_h, curr_w)
            return total

        # Single Conv2d compute
        out_h = (H + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1
        out_w = (W + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) // layer.stride[1] + 1
        ops = (layer.in_channels // layer.groups) * layer.kernel_size[0] * layer.kernel_size[1]
        ops = (ops + (1 if layer.bias is not None else 0)) * layer.out_channels * out_h * out_w
        return ops * 2.0 * B


# ==========================================
# Batched expert computation (key optimization)
# ==========================================
class BatchedExpertComputation:
    """
    Strategy: batch expert computations to eliminate for-loops.
    Performance: ~3–5x inference speedup observed.
    """

    @staticmethod
    def compute_sparse_experts_batched(
            x: torch.Tensor,
            experts: nn.ModuleList,
            routing_weights: torch.Tensor,
            routing_indices: torch.Tensor,
            top_k: int,
            num_experts: int
    ) -> torch.Tensor:
        """
        Batched expert computation:
        1) Pre-allocate outputs for all experts
        2) Compute all activated experts in parallel
        3) Aggregate using efficient scatter/index_add

        ONNX export note: ``torch.onnx.export`` uses tracing, which cannot
        capture data-dependent control flow (``if not mask.any(): continue``).
        When exporting, fall back to a dense path that computes *all* experts
        and selects outputs via ``torch.gather`` — fully traceable, no dynamic
        skips.  The sparse path remains for normal training/eval.
        """
        B, C, H, W = x.shape
        out_channels = last_conv_out_channels(experts[0])

        # Flatten indices/weights to [B, top_k]. Use reshape (handles
        # non-contiguous tensors from gather/permute) and slice the leading
        # top_k cols — robust to [B, k] and [B, k, 1, 1] alike, no squeeze juggling.
        current_top_k = routing_indices.shape[1]
        indices_flat = routing_indices.reshape(B, -1)[:, :current_top_k]  # [B, top_k]
        weights_flat = routing_weights.reshape(B, -1)[:, :current_top_k]  # [B, top_k]

        # ── ONNX-safe dense path ──────────────────────────────────────────
        # Compute every expert for the full batch (static Python loop over
        # ``num_experts`` → traceable), then gather the Top-K selected outputs
        # and weight-sum them.  No ``if mask.any()`` / ``continue`` guards.
        if torch.onnx.is_in_onnx_export():
            all_outs = torch.stack(
                [experts[i](x) for i in range(num_experts)], dim=1
            )  # [B, E, out_C, H, W]

            expert_output = torch.zeros(
                B, out_channels, H, W, device=x.device, dtype=x.dtype
            )
            for k in range(current_top_k):
                idx_k = indices_flat[:, k]                                   # [B]
                w_k = weights_flat[:, k]                                     # [B]
                idx_exp = idx_k.view(B, 1, 1, 1, 1).expand(B, 1, out_channels, H, W)
                selected = torch.gather(all_outs, 1, idx_exp).squeeze(1)     # [B, out_C, H, W]
                expert_output = expert_output + selected * w_k.view(B, 1, 1, 1)

            return expert_output.clamp_(-1e4, 1e4)

        # ── Sparse path (training / normal eval) ──────────────────────────
        # Plan A: conditional computation (skip low-weight experts). Keep all
        # selected experts during training so low-weight routes can still learn;
        # thresholding is an inference-only speed trade-off.
        weight_threshold = 0.0 if experts.training else 0.01
        valid_mask = weights_flat > weight_threshold

        # Initialize outputs
        expert_output = torch.zeros(B, out_channels, H, W, device=x.device, dtype=x.dtype)

        # Plan B: parallel batching (recommended)
        # Collect all samples per expert
        for expert_idx in range(num_experts):
            # Find all (batch, k) positions that selected this expert
            expert_mask = (indices_flat == expert_idx) & valid_mask

            if not expert_mask.any():
                continue

            # Get batch indices and corresponding weights
            where_res = torch.where(expert_mask)
            if len(where_res) == 1:
                batch_indices = where_res[0]
                k_indices = torch.zeros_like(batch_indices)
            else:
                batch_indices, k_indices = where_res

            # Batched forward pass
            expert_input = x[batch_indices]
            expert_out = experts[expert_idx](expert_input)

            # weights_flat is always [B, top_k] now; index by (batch, k).
            weights = weights_flat[batch_indices, k_indices].view(-1, 1, 1, 1)
            weighted_out = expert_out * weights

            # Accumulate outputs (efficient index_add_)
            expert_output.index_add_(0, batch_indices, weighted_out.to(expert_output.dtype))

        # Guard against activation explosion if routing collapses (all tokens to
        # one expert) so downstream norm layers don't produce NaN.
        expert_output = expert_output.clamp_(-1e4, 1e4)

        return expert_output
