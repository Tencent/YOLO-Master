# 🐧Please note that this file has been modified by Tencent on 2026/02/13. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Auxiliary losses for Mixture-of-Experts models (Production Grade)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Dict, Union, Tuple


def gshard_balance_loss(expert_usage: torch.Tensor, num_experts: int) -> torch.Tensor:
    """GShard-style balance loss: N * sum(usage^2). Equals 1.0 at uniform usage."""
    usage = expert_usage.reshape(-1).float()
    usage = usage / usage.sum().clamp_min(1e-6)
    return num_experts * torch.sum(usage * usage)


class MoELoss(nn.Module):
    """
    Advanced Auxiliary losses for MoE models.
    Features:
    - Distributed-aware calculation
    - Support for both Hard (GShard-style) and Soft (Differentiable) load balancing
    - Entropy regularization to prevent router indecisiveness
    - Detailed diagnostic outputs
    """

    def __init__(
        self,
        balance_loss_coeff: float = 1.0,
        z_loss_coeff: float = 1.0,
        entropy_loss_coeff: float = 0.0,
        diversity_loss_coeff: float = 0.0,  # New: penalize similar expert outputs
        variance_loss_coeff: float = 0.0,   # New: direct variance penalty on usage
        num_experts: int = 8,
        top_k: int = 2,
        use_soft_balancing: bool = False,
        coeff_floor: float = 0.0,
    ):
        super().__init__()
        self.balance_loss_coeff = balance_loss_coeff
        self.z_loss_coeff = z_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.diversity_loss_coeff = diversity_loss_coeff
        self.variance_loss_coeff = variance_loss_coeff
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_soft_balancing = use_soft_balancing
        self.coeff_floor = coeff_floor  # 0 = no floor (trainer already guards stale configs)

    @staticmethod
    def _flatten_router_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Normalize router tensors to `[N, num_experts]`.

        Several MoE blocks emit global router tensors as `[B, E, 1, 1]` while
        others emit `[B, E]`. Keeping the loss code shape-normalized prevents
        accidental broadcasting between `[E, 1, 1]` and `[E]`.
        """
        if tensor.dim() == 2:
            return tensor
        if tensor.dim() == 4:
            return tensor.permute(0, 2, 3, 1).reshape(-1, tensor.shape[1])
        return tensor.reshape(-1, tensor.shape[-1])

    @staticmethod
    def _flatten_expert_indices(indices: torch.Tensor) -> torch.Tensor:
        """Normalize Top-K index tensors to `[N, top_k]`."""
        if indices.dim() == 2:
            return indices
        if indices.dim() == 4:
            if indices.shape[1] <= 8:  # [B, K, H, W]
                return indices.permute(0, 2, 3, 1).reshape(-1, indices.shape[1])
            return indices.reshape(-1, indices.shape[-1])  # [B, H, W, K]
        return indices.reshape(indices.shape[0], -1)

    def _get_global_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        """Computes the mean of a tensor across all distributed processes."""
        if not (dist.is_available() and dist.is_initialized()):
            return tensor.mean(dim=0)
            
        # Sum locally first
        local_sum = tensor.sum(dim=0)
        # We need the global batch size count
        local_count = torch.tensor(tensor.size(0), device=tensor.device, dtype=tensor.dtype)
        
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
        
        return local_sum / local_count.clamp(min=1.0)

    def forward(
        self,
        router_probs: torch.Tensor,
        router_logits: torch.Tensor,
        expert_indices: Optional[torch.Tensor] = None,
        expert_outputs: Optional[torch.Tensor] = None,
        return_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            router_probs: [B, num_experts] Full probability distribution
            router_logits: [B, num_experts] Raw logits
            expert_indices: [B, k] Selected expert indices (required if use_soft_balancing=False)
            expert_outputs: [B, num_experts, D] Expert output features for diversity loss
            return_dict: If True, returns a dict with loss components for logging.
        """
        router_probs = self._flatten_router_tensor(router_probs)
        router_logits = self._flatten_router_tensor(router_logits)
        if expert_indices is not None:
            expert_indices = self._flatten_expert_indices(expert_indices)

        # 1. Load Balancing Loss
        importance = self._get_global_mean(router_probs)

        if self.use_soft_balancing:
            # === Soft Balancing (Fully Differentiable) ===
            # Usage is defined by the sum of probabilities allocated to each expert.
            # This allows gradients to flow through the "usage" term back to the router.
            usage = importance # In soft mode, usage approximates importance
        else:
            # === Hard Balancing (GShard / Switch Style) ===
            # Usage is defined by the discrete selection count.
            # Requires expert_indices.
            if expert_indices is None:
                raise ValueError("expert_indices is required for hard load balancing.")
                
            B = expert_indices.shape[0]
            flat_indices = expert_indices.view(-1)
            
            # Vectorized count using one_hot
            local_expert_counts = F.one_hot(flat_indices, num_classes=self.num_experts).float().sum(dim=0)
            
            # Sync counts across GPUs
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(local_expert_counts, op=dist.ReduceOp.SUM)
                total_samples = B * self.top_k * dist.get_world_size()
            else:
                total_samples = B * self.top_k
            
            usage = local_expert_counts / max(total_samples, 1.0)
            # Detach usage because discrete selection is non-differentiable here
            usage = usage.detach()

        # Balance Loss: N * sum(importance * usage)
        balance_loss = self.num_experts * torch.sum(importance * usage)

        # 2. Z-Loss (Router Stability)
        # ------------------------------------------------------------------
        # log(sum(exp(x)))^2
        log_z = torch.logsumexp(router_logits, dim=1)
        z_loss = torch.mean(log_z ** 2)

        # 3. Entropy Loss (Certainty Regularization) - Optional
        entropy_loss = torch.tensor(0.0, device=router_probs.device)
        if self.entropy_loss_coeff > 0:
            entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-8), dim=1).mean()
            entropy_loss = entropy

        # 4. Diversity Loss (Penalize similar expert outputs) - Optional
        # Targets orthogonal experts: cosine similarity -> 0, not -1
        diversity_loss = torch.tensor(0.0, device=router_probs.device)
        # Require E >= 2 (E == 1 makes num_pairs == 0 and blows up the divisor).
        if self.diversity_loss_coeff > 0 and expert_outputs is not None and expert_outputs.shape[1] >= 2:
            # expert_outputs: [B, num_experts, D]
            B, E, D = expert_outputs.shape
            # Normalize each expert output
            outputs_norm = F.normalize(expert_outputs, dim=-1)  # [B, E, D]
            # Compute pairwise cosine similarity: [B, E, E]
            similarity = torch.bmm(outputs_norm, outputs_norm.transpose(1, 2))  # [B, E, E]
            # Zero out diagonal (self-similarity)
            mask = 1.0 - torch.eye(E, device=similarity.device)
            masked_sim = similarity * mask.unsqueeze(0)  # [B, E, E]
            # Target: similarity -> 0 (orthogonal), penalize deviation from 0
            num_pairs = E * (E - 1)
            diversity_loss = (masked_sim ** 2).sum() / (B * num_pairs + 1e-8)

        # 5. Variance Loss (Direct usage variance penalty) - Optional
        # Penalizes high variance in expert usage, encouraging uniform distribution
        variance_loss = torch.tensor(0.0, device=router_probs.device)
        if self.variance_loss_coeff > 0:
            # Target: uniform distribution -> variance = 0
            # usage here is importance (soft) or counts (hard)
            target_usage = 1.0 / self.num_experts
            variance = torch.mean((usage - target_usage) ** 2)
            variance_loss = variance

        floor = float(getattr(self, "coeff_floor", 0.0))
        bl_coeff = max(self.balance_loss_coeff, floor)
        zl_coeff = max(self.z_loss_coeff, floor)

        # 6. Total Loss
        total_loss = (bl_coeff * balance_loss) + \
                     (zl_coeff * z_loss) + \
                     (self.entropy_loss_coeff * entropy_loss) + \
                     (self.diversity_loss_coeff * diversity_loss) + \
                     (self.variance_loss_coeff * variance_loss)
        
        # NaN Guard (Graph Safe)
        if not torch.isfinite(total_loss).all():
            total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)

        if return_dict:
            return {
                "loss": total_loss,
                "balance_loss": balance_loss.detach(),
                "z_loss": z_loss.detach(),
                "entropy_loss": entropy_loss.detach() if self.entropy_loss_coeff > 0 else 0.0,
                "diversity_loss": diversity_loss.detach() if self.diversity_loss_coeff > 0 else 0.0,
                "variance_loss": variance_loss.detach() if self.variance_loss_coeff > 0 else 0.0
            }
            
        return total_loss
