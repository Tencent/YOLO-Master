"""Shared numerical stability helpers for routed neural-network modules."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn


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
    if _autocast_is_available(device_type):
        return torch.autocast(device_type=device_type, enabled=False)
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
