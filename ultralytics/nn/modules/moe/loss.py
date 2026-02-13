# 🐧Please note that this file has been modified by Tencent on 2026/02/13. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Auxiliary losses for Mixture-of-Experts models (Production Grade)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Dict, Union, Tuple

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
        balance_loss_coeff: float = 0.01,
        z_loss_coeff: float = 1e-3,
        entropy_loss_coeff: float = 0.0, # New: Penalize uncertainty
        num_experts: int = 8,
        top_k: int = 2,
        use_soft_balancing: bool = False # New: Use probs instead of indices for usage
    ):
        super().__init__()
        self.balance_loss_coeff = balance_loss_coeff
        self.z_loss_coeff = z_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_soft_balancing = use_soft_balancing

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
        return_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            router_probs: [B, num_experts] Full probability distribution
            router_logits: [B, num_experts] Raw logits
            expert_indices: [B, k] Selected expert indices (required if use_soft_balancing=False)
            return_dict: If True, returns a dict with loss components for logging.
        """
        # 1. Load Balancing Loss
        # ------------------------------------------------------------------
        # Importance: Global average probability for each expert
        # [num_experts]
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
        # ------------------------------------------------------------------
        # We want the router to be "sure" about its choice.
        # Minimize Entropy: sum(-p * log(p))
        # Note: Use a small epsilon for log stability
        entropy_loss = torch.tensor(0.0, device=router_probs.device)
        if self.entropy_loss_coeff > 0:
            entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-8), dim=1).mean()
            entropy_loss = entropy

        # 4. Total Loss
        # ------------------------------------------------------------------
        total_loss = (self.balance_loss_coeff * balance_loss) + \
                     (self.z_loss_coeff * z_loss) + \
                     (self.entropy_loss_coeff * entropy_loss)
        
        # NaN Guard (Graph Safe)
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = total_loss * 0.0

        if return_dict:
            return {
                "loss": total_loss,
                "balance_loss": balance_loss.detach(),
                "z_loss": z_loss.detach(),
                "entropy_loss": entropy_loss.detach() if self.entropy_loss_coeff > 0 else 0.0
            }
            
        return total_loss