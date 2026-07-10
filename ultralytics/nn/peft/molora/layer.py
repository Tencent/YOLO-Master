"""MoLoRA layer implementation.

Contains:
  - MoLoRAExpert: a single LoRA expert (Conv2d or Linear low-rank pair)
  - MoLoRALayer: a wrapper that replaces a base layer with top-k sparse experts
"""
import math
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER
from .router import build_router
from .loss import MoLoRALoss
from .utils import (
    _molora_scales,
    init_lora_expert_a,
    init_lora_expert_b,
    _merge_conv_delta,
    _merge_linear_delta,
    _unmerge_conv_delta,
    _unmerge_linear_delta,
)


class MoLoRAExpert(nn.Module):
    """A single LoRA expert: low-rank A + B pair.

    For Conv2d:
      lora_A: 1x1 conv (C_in -> r)
      lora_B: KxK conv (r -> C_out)  — same kernel size as base conv
    For Linear:
      lora_A: Linear (in_features -> r)
      lora_B: Linear (r -> out_features)
    """

    def __init__(
        self,
        base_layer: nn.Module,
        r: int,
        alpha: int,
        dropout: float = 0.0,
        use_rslora: bool = True,
        init_type: str = "default",
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = _molora_scales(r, alpha, use_rslora)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.init_type = init_type

        if isinstance(base_layer, nn.Conv2d):
            self.is_conv = True
            in_c = base_layer.in_channels
            out_c = base_layer.out_channels
            k = base_layer.kernel_size
            if isinstance(k, int):
                k = (k, k)
            stride = base_layer.stride
            padding = base_layer.padding
            dilation = base_layer.dilation
            groups = base_layer.groups

            # Conv2d LoRA convention: A is 1x1, B is KxK with same stride/pad as base
            # groups=1 for both A and B: the low-rank delta can have cross-channel
            # interactions even when the base conv is grouped (e.g. DWConv).
            self.base_groups = groups
            self.lora_A = nn.Conv2d(
                in_c, r, kernel_size=1, bias=False,
            )
            self.lora_B = nn.Conv2d(
                r, out_c, kernel_size=k,
                stride=stride, padding=padding,
                dilation=dilation, groups=1, bias=False,
            )
        elif isinstance(base_layer, nn.Linear):
            self.is_conv = False
            in_f = base_layer.in_features
            out_f = base_layer.out_features
            self.lora_A = nn.Linear(in_f, r, bias=False)
            self.lora_B = nn.Linear(r, out_f, bias=False)
        else:
            raise TypeError(f"MoLoRAExpert only supports Conv2d and Linear, got {type(base_layer)}")

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_lora_expert_a(self.lora_A.weight, self.init_type)
        init_lora_expert_b(self.lora_B.weight, self.init_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute delta = B @ A(x) for this expert."""
        x = self.dropout(x)
        return self.lora_B(self.lora_A(x)) * self.scaling

    def delta_weight(self) -> torch.Tensor:
        """Return the equivalent delta weight (for inspection / merge).

        For grouped base conv, the full-rank delta [out_c, in_c, kH, kW] is
        folded into the grouped shape [out_c, in_c//g, kH, kW] by summing
        over the group dimension.
        """
        if self.is_conv:
            a = self.lora_A.weight.squeeze(-1).squeeze(-1)  # [r, in_c]
            b = self.lora_B.weight  # [out_c, r, kH, kW]
            full_delta = torch.einsum("orkw,ri->oikw", b, a) * self.scaling  # [out_c, in_c, kH, kW]
            g = getattr(self, "base_groups", 1)
            if g > 1:
                in_c = full_delta.shape[1]
                # Reshape: [out_c, in_c, kH, kW] -> [out_c, g, in_c//g, kH, kW] -> sum over g
                full_delta = full_delta.view(
                    full_delta.shape[0], g, in_c // g, *full_delta.shape[2:]
                ).sum(dim=1)
            return full_delta
        else:
            return (self.lora_B.weight @ self.lora_A.weight) * self.scaling


class MoLoRALayer(nn.Module):
    """Wrapper that replaces a base Conv2d/Linear with a sparse mixture of LoRA experts.

    Forward:
      1. Compute base layer output
      2. Route input -> top-k experts
      3. Sum g_k * expert_k(x)
      4. If share_moe_registry: write aux loss to MOE_LOSS_REGISTRY
    """

    def __init__(
        self,
        base_layer: nn.Module,
        r: int = 8,
        alpha: int = 16,
        num_experts: int = 4,
        top_k: int = 2,
        router_type: str = "linear",
        dropout: float = 0.0,
        use_rslora: bool = True,
        balance_loss_coef: float = 0.01,
        z_loss_coef: float = 0.001,
        diversity_loss_coef: float = 0.0,
        expert_init: str = "default",
        share_moe_registry: bool = True,
        router_hidden_dim: Optional[int] = None,
        capacity_factor: float = 1.0,
        expert_dropout: float = 0.0,
        top_k_warmup: Optional[int] = None,
        warmup_steps: int = 0,
        domain_experts: Optional[Dict[str, List[int]]] = None,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.num_experts = num_experts
        self.top_k = top_k
        self.scaling = _molora_scales(r, alpha, use_rslora)
        self.share_moe_registry = share_moe_registry
        self.merged = False
        self.capacity_factor = capacity_factor
        self.expert_dropout = expert_dropout
        self.top_k_warmup = top_k_warmup
        self.warmup_steps = warmup_steps
        self.domain_experts = domain_experts
        self.register_buffer("_step_count", torch.tensor(0, dtype=torch.long), persistent=True)
        # CPU-side step counter to avoid GPU sync (.item()) on every forward
        self._step_count_cpu: int = 0
        # Active mask for domain pre-allocation
        self._domain_active_mask: Optional[torch.Tensor] = None
        # Expert frozen mask
        self._expert_frozen_mask: Optional[torch.Tensor] = None
        # P1 fix (merge_weights weighting): EMA of per-expert routing usage,
        # updated every training forward. `merge_weights()` uses this instead
        # of a uniform 1/num_experts prior, so the merged single-path weight
        # better approximates the actual mixture the model was trained with —
        # a uniform average silently misrepresents experts whose usage share
        # deviates from uniform (the common case once the router specializes).
        self.register_buffer(
            "_usage_ema", torch.full((num_experts,), 1.0 / num_experts), persistent=True
        )
        self._usage_ema_momentum = 0.99

        # Freeze base layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        # Build experts
        self.experts = nn.ModuleList(
            MoLoRAExpert(
                base_layer,
                r=r,
                alpha=alpha,
                dropout=dropout,
                use_rslora=use_rslora,
                init_type=expert_init,
            )
            for _ in range(num_experts)
        )

        # Build router
        if isinstance(base_layer, nn.Conv2d):
            in_channels = base_layer.in_channels
        elif isinstance(base_layer, nn.Linear):
            in_channels = base_layer.in_features
        else:
            raise TypeError(f"Unsupported base layer: {type(base_layer)}")

        self.router = build_router(
            router_type=router_type,
            in_channels=in_channels,
            num_experts=num_experts,
            hidden_dim=router_hidden_dim,
        )

        # Auxiliary loss module
        self.loss_fn = MoLoRALoss(
            num_experts=num_experts,
            top_k=top_k,
            balance_loss_coef=balance_loss_coef,
            z_loss_coef=z_loss_coef,
            diversity_loss_coef=diversity_loss_coef,
            reduce_ddp=True,  # P1 fix: follow DDP (no-op on single GPU)
        )

        # Routing stats for diagnostics (not persistent)
        self._last_routing_stats: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Dynamic top-k
    # ------------------------------------------------------------------

    def _current_top_k(self) -> int:
        """Return current effective top_k considering warmup schedule.

        During warmup, gradually increase from 1 to self.top_k over
        self.warmup_steps steps. If top_k_warmup is set, use that as
        the number of steps instead.
        """
        if not self.training:
            return self.top_k
        if self.top_k_warmup is None or self.top_k_warmup <= 0:
            return self.top_k
        if self.warmup_steps <= 0:
            return self.top_k
        steps = min(self._step_count_cpu, self.warmup_steps)
        return max(1, int(1 + (self.top_k - 1) * steps / self.warmup_steps))

    # ------------------------------------------------------------------
    # Expert dropout
    # ------------------------------------------------------------------

    def _apply_expert_dropout(self, router_probs: torch.Tensor) -> torch.Tensor:
        """During training, randomly disable experts with probability expert_dropout.

        Returns modified router_probs with disabled experts zeroed out and renormalized.
        """
        if not self.training or self.expert_dropout <= 0.0:
            return router_probs
        mask = torch.bernoulli(
            torch.full((self.num_experts,), 1.0 - self.expert_dropout, device=router_probs.device)
        ).to(router_probs.dtype)
        any_active = mask.sum() > 0
        mask = torch.where(any_active, mask, torch.ones_like(mask))
        probs = router_probs * mask.unsqueeze(0)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    # ------------------------------------------------------------------
    # Capacity factor
    # ------------------------------------------------------------------

    def _apply_capacity_limit(
        self,
        top_k_weights: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a global soft capacity penalty on Top-K expert selections.

        When an expert is selected by more than ``capacity_factor * B * K / E`` slots
        in the batch, its Top-K weights are scaled down and renormalized.

        P1 fix — ``capacity_factor`` semantics clarified: valid *limiting*
        range is the open interval ``0 < capacity_factor < 1``. Both
        boundary cases are treated as "no limit" but for different reasons,
        which was previously undocumented and easy to misread as a bug:
          - ``capacity_factor <= 0``: invalid/disabled — no limit is applied.
          - ``capacity_factor >= 1.0``: mathematically already >= uniform
            capacity (``B*K/E``), so the penalty could never trigger anyway;
            short-circuiting just skips the redundant per-expert loop.
        Values are validated at config time (``MoLoRAConfig.__post_init__``);
        this method only guards against a directly-constructed layer.
        """
        if self.capacity_factor <= 0 or self.capacity_factor >= 1.0:
            return top_k_weights
        B, K = top_k_weights.shape
        max_slots = max(1, int(math.ceil(self.capacity_factor * B * K / self.num_experts)))
        weights = top_k_weights.clone()
        for e in range(self.num_experts):
            slots = top_k_indices == e
            usage = slots.sum()
            cond = usage > max_slots
            ratio = torch.tensor(float(max_slots), dtype=weights.dtype, device=weights.device) / usage.clamp_min(1)
            scale = torch.where(cond, ratio, torch.ones((), dtype=weights.dtype, device=weights.device))
            weights = torch.where(slots, weights * scale, weights)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    # ------------------------------------------------------------------
    # Domain pre-allocation
    # ------------------------------------------------------------------

    def set_domain(self, domain: str) -> None:
        """Restrict routing to a subset of experts for domain-specific inference.

        Only experts assigned to the given domain will be active.
        """
        if self.domain_experts is None or domain not in self.domain_experts:
            self._domain_active_mask = None
            return
        active = self.domain_experts[domain]
        # P1 fix: validate expert indices before building the mask. An
        # out-of-range index previously reached `mask[active] = True`
        # unguarded and raised an opaque `IndexError` deep inside routing;
        # an empty `active` list would silently produce an all-False mask
        # that later divides-by-zero-guards to uniform routing (misleading).
        invalid = [i for i in active if not (0 <= i < self.num_experts)]
        if invalid:
            raise IndexError(
                f"[MoLoRA] set_domain('{domain}'): expert indices {invalid} out of range "
                f"[0, {self.num_experts - 1}]."
            )
        if not active:
            raise ValueError(f"[MoLoRA] set_domain('{domain}'): domain_experts['{domain}'] is empty.")
        mask = torch.zeros(self.num_experts, dtype=torch.bool)
        mask[active] = True
        self._domain_active_mask = mask

    def clear_domain(self) -> None:
        """Clear domain restriction, restore all experts."""
        self._domain_active_mask = None

    # ------------------------------------------------------------------
    # Expert freezing (continual learning)
    # ------------------------------------------------------------------

    def freeze_experts(self, expert_indices: List[int]) -> None:
        """Freeze specific experts so their weights are not updated during training."""
        # P1 fix: validate expert indices before use.
        if not expert_indices:
            raise ValueError("[MoLoRA] freeze_experts: expert_indices is empty.")
        invalid = [i for i in expert_indices if not (0 <= i < self.num_experts)]
        if invalid:
            raise IndexError(
                f"[MoLoRA] freeze_experts: expert indices {invalid} out of range "
                f"[0, {self.num_experts - 1}]."
            )
        mask = torch.zeros(self.num_experts, dtype=torch.bool)
        mask[expert_indices] = True
        self._expert_frozen_mask = mask
        for idx in expert_indices:
            for p in self.experts[idx].parameters():
                p.requires_grad = False
        LOGGER.info(f"[MoLoRA] Frozen experts {expert_indices}")

    def unfreeze_experts(self, expert_indices: Optional[List[int]] = None) -> None:
        """Unfreeze specific or all experts."""
        if expert_indices is None:
            expert_indices = list(range(self.num_experts))
        # P1 fix: validate indices for the same reasons as freeze_experts.
        invalid = [i for i in expert_indices if not (0 <= i < self.num_experts)]
        if invalid:
            raise IndexError(
                f"[MoLoRA] unfreeze_experts: expert indices {invalid} out of range "
                f"[0, {self.num_experts - 1}]."
            )
        for idx in expert_indices:
            for p in self.experts[idx].parameters():
                p.requires_grad = True
        self._expert_frozen_mask = None
        LOGGER.info(f"[MoLoRA] Unfrozen experts {expert_indices}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] for Conv2d, [B, C] or [B, C, *] for Linear
        Returns:
            y: same shape as x, adapted by top-k experts
        """
        # Base output (always frozen)
        base_out = self.base_layer(x)

        if self.merged:
            return base_out

        # Increment step counter during training (CPU-side, no GPU sync)
        if self.training:
            self._step_count.add_(1)
            self._step_count_cpu += 1

        # Router
        router_logits = self.router(x)  # [B, E]

        # Apply domain restriction if set
        if self._domain_active_mask is not None:
            device = router_logits.device
            mask = self._domain_active_mask.to(device).to(router_logits.dtype)
            # Mask out inactive experts with large negative value
            router_logits = router_logits + (1.0 - mask) * -1e9

        router_probs = F.softmax(router_logits, dim=-1)  # [B, E]

        # Apply expert dropout during training
        router_probs = self._apply_expert_dropout(router_probs)

        # Dynamic top-k (warmup support)
        effective_k = self._current_top_k()

        # Top-K gating
        top_k_weights, top_k_indices = torch.topk(router_probs, effective_k, dim=-1)  # [B, K]
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        # Apply capacity limit (1.0 = no limit per config convention)
        if 0 < self.capacity_factor < 1.0:
            top_k_weights = self._apply_capacity_limit(top_k_weights, top_k_indices)

        # Compute sparse expert outputs
        adapted = self._compute_sparse_experts(x, top_k_weights, top_k_indices, base_out)

        # Auxiliary loss (graph-connected to router_logits)
        aux_loss = self.loss_fn(
            router_probs=router_probs,
            router_logits=router_logits,
            expert_indices=top_k_indices,
            expert_outputs=None,  # diversity loss disabled by default for efficiency
        )

        # Write to MOE_LOSS_REGISTRY if enabled
        if self.share_moe_registry and self.training:
            try:
                from ultralytics.nn.modules.moe.modules import _registry_set
                _registry_set(self, aux_loss)
            except Exception:
                pass

        # Store diagnostics
        usage_now = self._expert_usage(top_k_indices)
        self._last_routing_stats = {
            "top_k_indices": top_k_indices.detach(),
            "top_k_weights": top_k_weights.detach(),
            "expert_usage": usage_now,
            "effective_k": effective_k,
            "domain_mask": self._domain_active_mask,
        }

        # P1 fix (merge_weights weighting): track an EMA of expert usage
        # during training so `merge_weights()` can weight by actual routing
        # frequency instead of a uniform 1/E prior.
        if self.training:
            with torch.no_grad():
                m = self._usage_ema_momentum
                self._usage_ema.mul_(m).add_(usage_now.detach().to(self._usage_ema.device), alpha=1.0 - m)

        return base_out + adapted

    def _compute_sparse_experts(
        self,
        x: torch.Tensor,
        top_k_weights: torch.Tensor,
        top_k_indices: torch.Tensor,
        out_template: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate top-k expert outputs using batched expert indexing.

        Optimized path: groups samples by expert to avoid per-sample loops.
        """
        B = x.shape[0]
        K = top_k_indices.shape[1]
        expert_out = torch.zeros_like(out_template)

        for k in range(K):
            expert_idx = top_k_indices[:, k]  # [B]
            weights = top_k_weights[:, k]      # [B]

            # Batched: group all samples selecting the same expert
            for e in range(self.num_experts):
                mask = expert_idx == e
                if not mask.any():
                    continue
                # Check if expert is frozen (skip gradient for frozen experts)
                if self._expert_frozen_mask is not None and self._expert_frozen_mask[e]:
                    with torch.no_grad():
                        x_e = x[mask]
                        out_e = self.experts[e](x_e)
                else:
                    x_e = x[mask]
                    out_e = self.experts[e](x_e)
                w = weights[mask].view(-1, *([1] * (out_e.dim() - 1)))
                expert_out[mask] += out_e * w

        return expert_out

    def _expert_usage(self, expert_indices: torch.Tensor) -> torch.Tensor:
        """Normalized expert usage histogram."""
        flat = expert_indices.reshape(-1).to(torch.long)
        counts = torch.bincount(flat, minlength=self.num_experts).float()
        return counts / counts.sum().clamp_min(1.0)

    # ------------------------------------------------------------------
    # Merge / Unmerge
    # ------------------------------------------------------------------

    def merge_weights(self) -> None:
        """Merge all expert deltas into the base layer weight.

        After merge, forward skips the LoRA path.  This is useful for
        ONNX export / inference where you want zero adapter overhead.

        P1 fix: previously merged with a uniform ``1/num_experts`` prior
        regardless of how the router actually routed traffic during
        training. That silently misrepresents the trained mixture once the
        router specializes (the common case) — e.g. an expert used 80% of
        the time and one used 5% of the time both contributed equally to
        the merged weight. Now weights by the EMA of per-expert routing
        usage tracked during training (``self._usage_ema``), falling back
        to uniform only if the layer was never run in training mode (EMA
        still at its uniform initial value).
        """
        if self.merged:
            return

        with torch.no_grad():
            usage = self._usage_ema.to(dtype=torch.float32)
            usage = usage / usage.sum().clamp_min(1e-6)
            # Cache the per-expert weights actually applied so `unmerge_weights`
            # can exactly reverse this merge even if usage keeps updating
            # afterwards (merge is meant to be a terminal, inference-only op,
            # but unmerge should still be safe to call for debugging/tests).
            self._merged_expert_weights = usage.clone()

            if self.experts[0].is_conv:
                for e, w in zip(self.experts, usage):
                    _merge_conv_delta(self.base_layer.weight, e.lora_A, e.lora_B, e.scaling * w.item(), groups=getattr(e, 'base_groups', 1))
            else:
                for e, w in zip(self.experts, usage):
                    _merge_linear_delta(self.base_layer.weight, e.lora_A, e.lora_B, e.scaling * w.item())

        self.merged = True
        LOGGER.debug(
            f"[MoLoRA] Merged {self.num_experts} experts into base layer "
            f"(usage-weighted: {usage.tolist()})."
        )

    def unmerge_weights(self) -> None:
        """Restore the original base layer weight."""
        if not self.merged:
            return
        # P1 fix: reverse with the *same* per-expert weights used at merge
        # time (not a fresh/uniform recompute), so unmerge is an exact
        # inverse regardless of what happened to `_usage_ema` since.
        usage = getattr(self, "_merged_expert_weights", None)
        if usage is None:
            usage = torch.full((self.num_experts,), 1.0 / self.num_experts)
        with torch.no_grad():
            if self.experts[0].is_conv:
                for e, w in zip(self.experts, usage):
                    _unmerge_conv_delta(self.base_layer.weight, e.lora_A, e.lora_B, e.scaling * w.item(), groups=getattr(e, 'base_groups', 1))
            else:
                for e, w in zip(self.experts, usage):
                    _unmerge_linear_delta(self.base_layer.weight, e.lora_A, e.lora_B, e.scaling * w.item())
        self.merged = False
        self._merged_expert_weights = None
        LOGGER.debug("[MoLoRA] Unmerged experts from base layer.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def weight(self):
        """Expose base layer weight for compatibility with PEFT / merge utils."""
        return self.base_layer.weight

    def extra_repr(self) -> str:
        return (
            f"r={self.r}, alpha={self.alpha}, num_experts={self.num_experts}, "
            f"top_k={self.top_k}, merged={self.merged}"
        )
