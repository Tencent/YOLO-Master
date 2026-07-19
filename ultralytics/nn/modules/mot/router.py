"""Router and auxiliary-loss utilities for Mixture-of-Transformer."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules._numeric import stable_normalize
from ultralytics.nn.modules.moe import loss as _moe_loss
from ultralytics.nn.modules.utils import get_safe_groups as _safe_groups
from ultralytics.nn.modules.mot._constants import (
    DEFAULT_MIN_TEMPERATURE,
    DEFAULT_TEMPERATURE_ANNEAL_FACTOR,
    ROUTER_LOGIT_LIMIT,
    ROUTER_Z_LOSS_LIMIT,
)


def differentiable_balance_loss(
    router_probs: torch.Tensor,
    expert_usage: torch.Tensor,
    num_experts: int,
    target_usage: Optional[torch.Tensor] = None,
    reduce_ddp: bool = True,  # P1-2 fix: default True for DDP safety (was False)
) -> torch.Tensor:
    """GShard loss with local router gradients and globally averaged detached usage.

    MoT intentionally keeps ``importance`` local so DDP never touches a graph-connected
    router tensor. The detached discrete usage uses MoE's canonical reduction helper.

    Args:
        reduce_ddp: If True, synchronize expert usage across ranks before computing
            balance loss. Defaults to True (P1-2 fix). Set False only for debugging.
    """
    probs = (
        router_probs.reshape(router_probs.shape[0], router_probs.shape[1], -1).mean(-1)
        if router_probs.dim() == 4
        else router_probs.reshape(-1, num_experts)
    )
    importance = probs.mean(dim=0)
    importance = importance / importance.sum().clamp_min(torch.finfo(importance.dtype).tiny)

    usage = expert_usage.reshape(-1).float().detach()
    usage = usage / usage.sum().clamp_min(torch.finfo(usage.dtype).tiny)
    if reduce_ddp:
        usage = _moe_loss.all_reduce_mean(usage)

    if target_usage is not None:
        weights = target_usage.reshape(-1).float()
        weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
        usage = usage * weights * num_experts
    return num_experts * torch.sum(importance * usage)


class _MoTRouter(nn.Module):
    """Content-aware router for MoT: assigns each token to Top-K experts.

    Architecture: global average pool → linear → softmax (soft) or
    top-k + renormalize (hard).

    For spatial routing (token-level), uses a lightweight 1×1 conv.
    For image-level routing (same assignment for all tokens), uses GAP.
    Default: token-level soft routing for stability; hard routing optional.

    Args:
        dim (int): Input channels.
        num_experts (int): Number of transformer experts (default 3).
        top_k (int): Number of active experts per token (soft Top-K mask).
        use_spatial (bool): Token-level vs. image-level routing.
        temperature (float): Softmax temperature for soft routing.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 3,
        top_k: int = 2,
        use_spatial: bool = True,
        temperature: float = 1.0,
        exploration_eps: float = 0.02,
        scene_aware: bool = False,
        scene_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_spatial = use_spatial
        # Register temperature as a buffer so checkpoint save/restore
        # preserves annealing progress (Python float would be lost).
        self.register_buffer("temperature", torch.tensor(max(temperature, 0.1)), persistent=True)
        self.exploration_eps = exploration_eps

        hidden = max(dim // 8, num_experts * 4)
        if use_spatial:
            self.router = nn.Sequential(
                nn.Conv2d(dim, hidden, 1, bias=False),
                nn.GroupNorm(_safe_groups(hidden, 4), hidden),
                nn.SiLU(inplace=False),
                nn.Conv2d(hidden, num_experts, 1, bias=True),
            )
        else:
            self.router = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(dim, hidden, bias=False),
                nn.SiLU(inplace=False),
                nn.Linear(hidden, num_experts, bias=True),
            )
        # init: near-uniform
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

        self.scene_aware = False
        self.scene_hidden_dim = scene_hidden_dim
        self.scene_projector: Optional[nn.Sequential] = None
        self.last_scene_stats: Optional[torch.Tensor] = None
        self.last_scene_bias: Optional[torch.Tensor] = None
        self._last_scene_stats_for_loss: Optional[torch.Tensor] = None
        if scene_aware:
            self.enable_scene_aware(scene_hidden_dim)

    def enable_scene_aware(self, hidden_dim: Optional[int] = None) -> None:
        """Enable the zero-initialized scene residual, creating parameters once."""
        if self.scene_projector is None:
            hidden = int(hidden_dim or self.scene_hidden_dim or 3)
            if hidden <= 0:
                raise ValueError("scene_hidden_dim must be positive")
            self.scene_projector = nn.Sequential(
                nn.Linear(3, hidden),
                nn.SiLU(inplace=False),
                nn.Linear(hidden, self.num_experts),
            )
            nn.init.zeros_(self.scene_projector[-1].weight)
            nn.init.zeros_(self.scene_projector[-1].bias)
            self.scene_hidden_dim = hidden
        self.scene_aware = True

    @staticmethod
    def compute_scene_stats(x: torch.Tensor) -> torch.Tensor:
        """Compute differentiable high-frequency, heterogeneity, and scale statistics."""
        feature = x.float()
        eps = torch.finfo(feature.dtype).eps
        rms = feature.square().mean(dim=(1, 2, 3)).sqrt().clamp_min(eps)

        dx = (feature[..., 1:] - feature[..., :-1]).abs().mean(dim=(1, 2, 3)) if feature.shape[-1] > 1 else rms * 0
        dy = (
            (feature[..., 1:, :] - feature[..., :-1, :]).abs().mean(dim=(1, 2, 3)) if feature.shape[-2] > 1 else rms * 0
        )
        high_frequency = 0.5 * (dx + dy) / rms

        spatial_energy = feature.square().mean(dim=1)
        heterogeneity = spatial_energy.flatten(1).std(dim=1, unbiased=False) / spatial_energy.flatten(1).mean(
            dim=1
        ).clamp_min(eps)

        pooled2 = F.adaptive_avg_pool2d(feature, (min(2, feature.shape[-2]), min(2, feature.shape[-1])))
        pooled4 = F.adaptive_avg_pool2d(feature, (min(4, feature.shape[-2]), min(4, feature.shape[-1])))
        scale2 = pooled2.var(dim=(1, 2, 3), unbiased=False)
        scale4 = pooled4.var(dim=(1, 2, 3), unbiased=False)
        multi_scale = (scale4 - scale2).abs() / feature.var(dim=(1, 2, 3), unbiased=False).clamp_min(eps)

        return torch.stack((high_frequency, heterogeneity, multi_scale), dim=1)

    def scene_consistency_loss(
        self,
        weights: torch.Tensor,
        scene_stats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Align Local, Window, and Deformable probabilities with scene statistics."""
        if self.num_experts != 3:
            return weights.new_zeros(())
        stats = self._last_scene_stats_for_loss if scene_stats is None else scene_stats
        if stats is None:
            return weights.new_zeros(())
        target_scores = torch.stack((stats[:, 0], stats[:, 2], stats[:, 1]), dim=1)
        target = F.softmax(target_scores.detach(), dim=1)
        probs = weights.float().mean(dim=(2, 3)).clamp_min(1e-8)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return F.kl_div(probs.log(), target, reduction="batchmean")

    def _compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x)
        if not self.use_spatial:
            logits = logits.unsqueeze(-1).unsqueeze(-1)
        if self.scene_aware:
            if self.scene_projector is None:
                raise RuntimeError("scene-aware MoT router is enabled without a scene projector")
            scene_stats = self.compute_scene_stats(x)
            scene_bias = self.scene_projector(scene_stats).to(dtype=logits.dtype)
            logits = logits + scene_bias.unsqueeze(-1).unsqueeze(-1)
            self._last_scene_stats_for_loss = scene_stats
            self.last_scene_stats = scene_stats.detach()
            self.last_scene_bias = scene_bias.detach()
        else:
            self._last_scene_stats_for_loss = None
            self.last_scene_stats = None
            self.last_scene_bias = None
        return logits

    @staticmethod
    def z_loss_from_logits(logits: torch.Tensor) -> torch.Tensor:
        # Clamp logits before logsumexp to prevent overflow (logsumexp of >88 overflows in float32)
        safe_logits = logits.float().clamp(min=-ROUTER_LOGIT_LIMIT, max=ROUTER_LOGIT_LIMIT)
        log_z = torch.logsumexp(safe_logits, dim=1)
        # Also clamp the z_loss result itself to prevent inf propagation
        return ((log_z**2).clamp(max=ROUTER_Z_LOSS_LIMIT)).mean()

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            weights : [B, num_experts, H, W] or [B, num_experts, 1, 1]  (soft, sum-to-1)
            indices : [B, top_k, H, W] or [B, top_k, 1, 1]  (top-k expert ids)
        """
        logits = self._compute_logits(x)  # [B, E, H, W] or [B, E, 1, 1] after GAP

        # Use the (possibly annealed) temperature in both train and eval so
        # that routing entropy stays consistent across modes.  Previously eval
        # hardcoded temp=1.0, which could shift router distributions and
        # expert combinations after annealing.
        temp = self.temperature

        # Soft weights (always computed for gradient flow)
        weights = F.softmax(logits / temp, dim=1)  # [B, E, H, W]
        dense_weights = weights

        # Top-K mask
        if self.top_k < self.num_experts:
            # get top-k indices [B, K, H, W]
            topk_vals, topk_idx = weights.topk(self.top_k, dim=1)
            # renormalize selected weights
            topk_weights = stable_normalize(topk_vals, dim=1)
            # scatter back to [B, E, H, W] sparse
            sparse_w = torch.zeros_like(weights)
            sparse_w.scatter_(1, topk_idx, topk_weights)
            weights = sparse_w
            if self.training and self.exploration_eps > 0:
                eps = min(max(self.exploration_eps, 0.0), 0.2)
                weights = weights * (1.0 - eps) + dense_weights * eps
            indices = topk_idx
        else:
            indices = (
                torch.arange(self.num_experts, device=x.device)
                .view(1, -1, 1, 1)
                .expand(x.shape[0], -1, x.shape[2], x.shape[3])
            )

        if return_logits:
            return weights, indices, logits
        return weights, indices

    @staticmethod
    def expert_usage_from_indices(indices: torch.Tensor, num_experts: int) -> torch.Tensor:
        """Discrete expert usage share from Top-K index tensor (any rank ≥ 2)."""
        flat = indices.reshape(-1).to(torch.long)
        counts = torch.bincount(flat, minlength=num_experts).float()
        return counts / max(flat.numel(), 1)

    def router_z_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Z-loss for load balance: encourages router logits to be small."""
        return self.z_loss_from_logits(self._compute_logits(x))


def _mot_router_aux_loss(
    weights: torch.Tensor,
    logits: torch.Tensor,
    indices: torch.Tensor,
    num_experts: int,
    balance_coeff: float,
    z_coeff: float,
    *,
    reduce_ddp: bool = False,
) -> torch.Tensor:
    """GShard balance + router z-loss for MoT (matches MoE/MoLoRA formulation)."""
    if balance_coeff <= 0 and z_coeff <= 0:
        return weights.new_zeros(())

    probs = weights
    if probs.dim() == 4:
        probs = probs.reshape(probs.shape[0], num_experts, -1).mean(-1)
    probs = probs.reshape(-1, num_experts)

    usage = _MoTRouter.expert_usage_from_indices(indices, num_experts)
    balance = differentiable_balance_loss(probs, usage, num_experts, reduce_ddp=reduce_ddp)
    z_loss = _MoTRouter.z_loss_from_logits(logits)

    total = weights.new_zeros(())
    if balance_coeff > 0:
        total = total + balance_coeff * balance
    if z_coeff > 0:
        total = total + z_coeff * z_loss
    # Guard against non-finite aux_loss propagating to total loss
    if not torch.isfinite(total):
        return weights.new_zeros(())
    return total


def anneal_mot_temperature(
    model: nn.Module,
    factor: float = DEFAULT_TEMPERATURE_ANNEAL_FACTOR,
    min_temp: float = DEFAULT_MIN_TEMPERATURE,
) -> None:
    """Multiplicatively anneal MoT router temperatures each epoch."""
    for m in model.modules():
        if isinstance(m, _MoTRouter):
            # temperature is now a persistent buffer tensor
            new_temp = max(float(m.temperature) * factor, min_temp)
            m.temperature.fill_(new_temp)


__all__ = ("_MoTRouter", "_mot_router_aux_loss", "anneal_mot_temperature", "differentiable_balance_loss")
