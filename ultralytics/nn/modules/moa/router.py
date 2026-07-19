"""Router and auxiliary-loss utilities for Mixture-of-Attention."""
from __future__ import annotations
import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules._numeric import all_reduce_mean, fp_clamp_floor
from ultralytics.nn.modules.moa._constants import DEFAULT_MIN_TEMPERATURE, DEFAULT_TEMPERATURE_ANNEAL_FACTOR, ROUTER_ENTROPY_FLOOR, ROUTER_LOGIT_LIMIT, ROUTER_Z_LOSS_LIMIT
from ultralytics.nn.modules.utils import get_safe_groups as _safe_groups

_all_reduce_mean = all_reduce_mean
_fp_min = fp_clamp_floor

class _MoARouter(nn.Module):
    """Lightweight soft-router: assigns each spatial token a weight over M head-groups.

    Complexity: O(H·W·C_in / reduction).
    Output: [B, M, H, W] soft gate probabilities (sum-to-one over M).
    """

    def __init__(self, dim: int, num_groups: int, reduction: int = 8,
                 temperature: float = 1.0):
        super().__init__()
        self.temperature = max(temperature, 0.1)
        hidden = max(dim // reduction, num_groups * 2)
        self.router = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.GroupNorm(_safe_groups(hidden, 4), hidden),
            nn.SiLU(inplace=False),
            nn.Conv2d(hidden, num_groups, 1, bias=True),
        )
        # init: near-uniform routing
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        # Use the (possibly annealed) temperature in both train and eval so
        # that routing entropy stays consistent across modes.  Previously eval
        # hardcoded temp=1.0, which could shift router distributions after
        # annealing and destabilise MoA (no Top-K stable set).
        temp = self.temperature
        logits = self.router(x) / temp           # [B, M, H, W]
        probs = F.softmax(logits, dim=1)
        if return_logits:
            return probs, logits
        return probs

def _moa_router_aux_loss(
    weights: torch.Tensor,
    logits: torch.Tensor,
    coeff: float,
    *,
    reduce_ddp: bool = False,
) -> torch.Tensor:
    """GShard-scale MoA regularization with exact DDP global-value/local-gradient semantics.

    Uses the shared numerical all-reduce helper (including NCCL CPU-tensor safety) for the
    cross-rank synchronization of detached statistics, so this function never crashes
    when router outputs land on CPU under a NCCL-backed DDP group.
    """
    num_groups = weights.shape[1]
    local_sum = weights.float().sum(dim=(0, 2, 3))
    local_count = weights.new_tensor(float(weights.shape[0] * weights.shape[2] * weights.shape[3])).float()
    if reduce_ddp:
        global_sum = _all_reduce_mean(local_sum.detach().clone())
        global_count = _all_reduce_mean(local_count.detach().clone())
        # DDP averages parameter gradients by world size. Scale the local Jacobian
        # by R/N while exposing the exact detached global value S/N.
        importance = global_sum / global_count.clamp_min(1.0)
        local_grad = (local_sum - local_sum.detach()) * (dist.get_world_size() / global_count.clamp_min(1.0))
        importance = importance + local_grad
    else:
        importance = local_sum / local_count.clamp_min(1.0)
    importance_sum = importance.sum().clamp_min(_fp_min(1e-6, weights.dtype))
    # Prevent Inf propagation from importance division (all-reduce can produce Inf)
    if not torch.isfinite(importance_sum).any():
        importance_sum = importance_sum.new_tensor(1.0)
    importance = importance / importance_sum
    balance_loss = num_groups * torch.sum(importance * importance)
    # Stabilize z-loss: clip logits before logsumexp to prevent overflow (logsumexp of >88 -> inf in float32)
    safe_logits = logits.float().clamp(min=-ROUTER_LOGIT_LIMIT, max=ROUTER_LOGIT_LIMIT)
    log_z = torch.logsumexp(safe_logits, dim=1)
    z_loss = (log_z ** 2).clamp(max=ROUTER_Z_LOSS_LIMIT).mean()
    # Guard: clamp importance to finite before entropy computation
    importance_safe = importance.clamp(min=0.0, max=1.0)
    entropy = -(importance_safe * torch.log(importance_safe.clamp_min(ROUTER_ENTROPY_FLOOR))).sum()
    max_entropy = math.log(max(num_groups, 2))
    entropy_deficit = (max_entropy - entropy).clamp_min(0.0) / max_entropy
    # Lower entropy weight (0.01) avoids over-constraining the router toward
    # uniform mixing when balance_loss already encourages load balance.
    result = coeff * (balance_loss + 0.1 * z_loss + 0.01 * entropy_deficit)
    # Final safety: prevent Inf/NaN aux_loss from poisoning the total loss
    if not torch.isfinite(result):
        return weights.new_zeros(())
    return result

def anneal_moa_temperature(
    model: nn.Module,
    factor: float = DEFAULT_TEMPERATURE_ANNEAL_FACTOR,
    min_temp: float = DEFAULT_MIN_TEMPERATURE,
) -> None:
    """Multiplicatively anneal router temperatures in all MoA modules.

    Call at the end of each epoch:
        anneal_moa_temperature(model, factor=0.97, min_temp=0.3)
    """
    for m in model.modules():
        if isinstance(m, _MoARouter):
            m.temperature = max(m.temperature * factor, min_temp)

__all__ = ("_MoARouter", "_moa_router_aux_loss", "anneal_moa_temperature")
