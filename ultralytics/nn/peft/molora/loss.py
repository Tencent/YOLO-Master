"""MoLoRA auxiliary losses.

Reimplements GShard load-balancing, z-loss, and diversity loss so that
MoLoRA can publish to the shared MOE_LOSS_REGISTRY with the same formula
as the core MoE modules.
"""
import math
from typing import Optional, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a tensor across DDP ranks (no-op on single GPU)."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    world = dist.get_world_size()
    if world <= 1:
        return tensor
    orig_dtype = tensor.dtype
    out = tensor.float().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out = out / world
    return out.to(orig_dtype)


class MoLoRALoss(nn.Module):
    """Auxiliary losses for MoLoRA routing.

    Components:
      - balance_loss: GShard-style N * sum(f_i * P_i) to encourage uniform usage
      - z_loss: penalizes large router logits for stability
      - diversity_loss: penalizes cosine similarity between expert outputs
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        balance_loss_coef: float = 0.01,
        z_loss_coef: float = 0.001,
        diversity_loss_coef: float = 0.0,
        reduce_ddp: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_loss_coef = balance_loss_coef
        self.z_loss_coef = z_loss_coef
        self.diversity_loss_coef = diversity_loss_coef
        self.reduce_ddp = reduce_ddp

    def _balance_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """GShard balance loss using discrete top-k counts."""
        # router_probs: [B, E] or [B, E, 1, 1]
        probs = router_probs.reshape(router_probs.shape[0], -1)
        if probs.shape[1] != self.num_experts:
            # Handle [B, E, 1, 1] -> [B, E]
            probs = probs.reshape(router_probs.shape[0], self.num_experts)

        importance = probs.mean(dim=0)  # [E], grad-preserving
        importance = importance / importance.sum().clamp_min(1e-6)

        # Usage from discrete expert selections
        flat_indices = expert_indices.reshape(-1).to(torch.long)
        counts = torch.bincount(flat_indices, minlength=self.num_experts).float()
        total = flat_indices.numel()
        usage = counts / max(total, 1)
        usage = usage.detach()  # non-differentiable usage

        if self.reduce_ddp:
            importance = all_reduce_mean(importance)
            usage = all_reduce_mean(usage)

        return self.num_experts * torch.sum(importance * usage)

    def _z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Router z-loss: log(sum(exp(x)))^2 averaged over batch."""
        logits = router_logits.reshape(router_logits.shape[0], -1)
        if logits.shape[1] != self.num_experts:
            logits = logits.reshape(router_logits.shape[0], self.num_experts)
        log_z = torch.logsumexp(logits, dim=1)
        return torch.mean(log_z ** 2)

    def _diversity_loss(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        """Penalize pairwise cosine similarity between expert outputs.

        expert_outputs: [B, num_experts, D]
        """
        B, E, D = expert_outputs.shape
        if E < 2:
            return torch.tensor(0.0, device=expert_outputs.device, dtype=expert_outputs.dtype)
        normed = F.normalize(expert_outputs, dim=-1)  # [B, E, D]
        sim = torch.bmm(normed, normed.transpose(1, 2))  # [B, E, E]
        mask = 1.0 - torch.eye(E, device=sim.device)
        masked = sim * mask.unsqueeze(0)
        num_pairs = E * (E - 1)
        return (masked ** 2).sum() / (B * num_pairs + 1e-8)

    def forward(
        self,
        router_probs: torch.Tensor,
        router_logits: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_outputs: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            router_probs:  [B, E] or [B, E, 1, 1]
            router_logits: [B, E] or [B, E, 1, 1]
            expert_indices: [B, K] or [B, K, 1, 1]
            expert_outputs: optional [B, E, D] for diversity loss
            return_dict: if True return dict of components
        """
        balance = self._balance_loss(router_probs, expert_indices)
        z = self._z_loss(router_logits)

        total = self.balance_loss_coef * balance + self.z_loss_coef * z

        div = torch.tensor(0.0, device=router_probs.device, dtype=router_probs.dtype)
        if self.diversity_loss_coef > 0 and expert_outputs is not None:
            div = self._diversity_loss(expert_outputs)
            total = total + self.diversity_loss_coef * div

        if not torch.isfinite(total).all():
            total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)

        if return_dict:
            return {
                "loss": total,
                "balance_loss": balance.detach(),
                "z_loss": z.detach(),
                "diversity_loss": div.detach() if self.diversity_loss_coef > 0 else 0.0,
            }
        return total


# ---------------------------------------------------------------------------
# Standalone helpers (for direct use in layer / model)
# ---------------------------------------------------------------------------

def compute_expert_usage(expert_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Return normalized expert usage histogram from discrete top-k indices."""
    flat = expert_indices.reshape(-1).to(torch.long)
    counts = torch.bincount(flat, minlength=num_experts).float()
    return counts / counts.sum().clamp_min(1.0)
