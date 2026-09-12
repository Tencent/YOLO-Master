"""Shared numerical stability helpers for routed neural-network modules."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn

from ultralytics.nn.modules.topk_contract import (
    DEFAULT_DETERMINISTIC_TOPK,
    LEGACY_PRIORITY_BIAS_TOPK,
    SUPPORTED_DETERMINISTIC_TOPK,
)


def _autocast_is_available(device_type: str) -> bool:
    """Return whether this PyTorch build exposes autocast for ``device_type``."""
    checker = getattr(getattr(torch, "amp", None), "autocast_mode", None)
    checker = getattr(checker, "is_autocast_available", None)
    if checker is not None:
        try:
            return bool(checker(device_type))
        except (RuntimeError, TypeError):
            return False
    # PyTorch 2.2 does not expose the capability query and has no MPS
    # autocast implementation. CPU and CUDA autocast are supported there.
    return device_type in {"cpu", "cuda"}


def disabled_autocast(device_type: str):
    """Disable autocast when supported, otherwise return a no-op context."""
    if not _autocast_is_available(device_type):
        return nullcontext()

    # ``torch.autocast`` was introduced after the oldest supported PyTorch
    # release.  Keep the legacy CUDA context available while treating CPU
    # autocast as a no-op on those builds.
    autocast = getattr(torch, "autocast", None)
    if callable(autocast):
        return autocast(device_type=device_type, enabled=False)
    legacy_autocast = getattr(getattr(torch.cuda, "amp", None), "autocast", None)
    if device_type == "cuda" and callable(legacy_autocast):
        return legacy_autocast(enabled=False)
    return nullcontext()


class FP32RouterMixin:
    """Keep router parameters in FP32 across model-wide dtype conversions."""

    def _apply(self, fn):
        super()._apply(fn)
        for parameter in self.parameters(recurse=True):
            parameter.data = parameter.data.float()
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.float()
        for buffer in self.buffers(recurse=True):
            if buffer.is_floating_point():
                buffer.data = buffer.data.float()
        return self


def should_reduce_ddp(module: nn.Module | None = None, *, training: bool | None = None) -> bool:
    """Return whether this explicitly identified training forward may enter a DDP collective.

    A missing module/training flag is deliberately treated as local-only. This
    prevents eval-with-grad, export, profiling, or rank-0-only diagnostics from
    joining the training collective sequence merely because gradients happen to
    be globally enabled.
    """
    if training is None:
        if module is None:
            return False
        training = module.training
    return bool(
        training
        and torch.is_grad_enabled()
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )


def fp_clamp_floor(value: float, dtype: torch.dtype) -> float:
    """Return a practical normalization floor for the requested floating dtype."""
    if dtype == torch.float16:
        return max(value, 1e-4)
    if dtype == torch.bfloat16:
        return max(value, 1e-3)
    return value


def clamp_min_for_dtype(tensor: torch.Tensor, value: float = 1e-6) -> torch.Tensor:
    """Clamp with a floor that remains effective under fp16 and bf16 AMP."""
    dtype = tensor.dtype
    work = tensor.float() if tensor.device.type == "cpu" and dtype in {torch.float16, torch.bfloat16} else tensor
    return work.clamp_min(fp_clamp_floor(value, dtype)).to(dtype)


def stable_normalize(tensor: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    """Normalize along ``dim`` without allowing a low-precision zero denominator."""
    dtype = tensor.dtype
    work = tensor.float() if tensor.device.type == "cpu" and dtype in {torch.float16, torch.bfloat16} else tensor
    denominator = work.sum(dim=dim, keepdim=True).clamp_min(fp_clamp_floor(eps, dtype))
    return (work / denominator).to(dtype)


def deterministic_topk_indices(
    probabilities: torch.Tensor,
    top_k: int,
    *,
    tie_tolerance: float = 1e-6,
    tie_break: str = DEFAULT_DETERMINISTIC_TOPK,
) -> torch.Tensor:
    """Select Top-K with a versioned cross-runtime tie policy.

    ``max_deadband_then_lowest_expert_id`` repeatedly finds the largest
    remaining probability and, among experts within ``tie_tolerance`` of that
    maximum, chooses the lowest expert id. The original probabilities remain
    unchanged and are still used as mixture weights.

    The legacy expert-id priority-bias rule remains available so old exported
    bundle manifests can be replayed without silently changing their contract.
    """
    if probabilities.ndim < 2:
        raise ValueError(f"probabilities must have an expert dimension, got {tuple(probabilities.shape)}")
    num_experts = int(probabilities.shape[1])
    if not 1 <= int(top_k) <= num_experts:
        raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
    if float(tie_tolerance) < 0.0:
        raise ValueError(f"tie_tolerance must be nonnegative, got {tie_tolerance}")
    if tie_break not in SUPPORTED_DETERMINISTIC_TOPK:
        raise ValueError(f"unsupported deterministic Top-K policy: {tie_break!r}")

    if tie_break == LEGACY_PRIORITY_BIAS_TOPK:
        ranking_probabilities = probabilities
        if tie_tolerance > 0.0:
            expert_priority = torch.arange(
                num_experts,
                device=probabilities.device,
                dtype=probabilities.dtype,
            )
            priority_shape = [1, num_experts] + [1] * (probabilities.ndim - 2)
            ranking_probabilities = probabilities - expert_priority.view(*priority_shape) * tie_tolerance
        return torch.argsort(ranking_probabilities, dim=1, descending=True, stable=True)[:, : int(top_k)]

    # This path is used only for eager sparse inference, not for ONNX tracing.
    available = torch.ones_like(probabilities, dtype=torch.bool)
    expert_ids = torch.arange(num_experts, device=probabilities.device, dtype=torch.long)
    expert_shape = [1, num_experts] + [1] * (probabilities.ndim - 2)
    expert_ids = expert_ids.view(*expert_shape).expand_as(probabilities)
    selections = []
    for _ in range(int(top_k)):
        remaining = probabilities.masked_fill(~available, float("-inf"))
        maximum = remaining.amax(dim=1, keepdim=True)
        candidates = available & (remaining >= maximum - float(tie_tolerance))
        candidate_ids = torch.where(candidates, expert_ids, num_experts)
        selected = candidate_ids.amin(dim=1, keepdim=True)
        selections.append(selected)
        available = available.scatter(1, selected, False)
    return torch.cat(selections, dim=1)


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Return the DDP mean with a global value and a local autograd Jacobian."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    world = dist.get_world_size()
    if world <= 1:
        return tensor

    original_dtype = tensor.dtype
    if tensor.device.type == "cpu" and dist.get_backend() == "nccl":
        tensor = tensor.cuda()
    local = tensor.float()
    global_value = local.detach().clone()
    dist.all_reduce(global_value, op=dist.ReduceOp.SUM)
    global_value = global_value / world
    return (local + (global_value - local.detach())).to(original_dtype)
