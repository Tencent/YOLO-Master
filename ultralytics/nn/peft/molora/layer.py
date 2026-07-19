"""MoLoRA layer implementation.

Contains:
  - MoLoRAExpert: a single LoRA expert (Conv2d or Linear low-rank pair)
  - MoLoRALayer: a wrapper that replaces a base layer with top-k sparse experts
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

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
from ultralytics.nn.modules.routing_protocol import (
    export_capabilities as _export_routing_capabilities,
    publish_aux_loss,
    routing_snapshot as _routing_snapshot,
)
from ultralytics.nn.modules.utils import robust_deepcopy


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
            self.lora_A = nn.Conv2d(
                in_c, r, kernel_size=1, bias=False,
            )
            self.lora_B = nn.Conv2d(
                r, out_c, kernel_size=k,
                stride=stride, padding=padding,
                dilation=dilation, groups=groups, bias=False,
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
        """Compute delta = B @ A(x) for this expert.

        Keep low-rank execution in parameter dtype so half parameters work on CPU.
        """
        x = self.dropout(x)
        input_dtype = x.dtype
        parameter_dtype = self.lora_A.weight.dtype
        out = self.lora_B(self.lora_A(x.to(parameter_dtype))) * self.scaling
        return out.to(input_dtype) if input_dtype != parameter_dtype else out

    def delta_weight(self) -> torch.Tensor:
        """Return the equivalent full-rank delta weight (for inspection / merge)."""
        if self.is_conv:
            a = self.lora_A.weight.squeeze(-1).squeeze(-1)  # [r, in_c]
            b = self.lora_B.weight  # [out_c, r, kH, kW]
            return torch.einsum("orkw,ri->oikw", b, a) * self.scaling
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
        self.register_buffer(
            "_usage_ema", torch.full((num_experts,), 1.0 / num_experts, dtype=torch.float32), persistent=True
        )
        self.usage_ema_decay = 0.99
        # Active mask for domain pre-allocation
        self._domain_active_mask: Optional[torch.Tensor] = None
        # Expert frozen mask
        self._expert_frozen_mask: Optional[torch.Tensor] = None

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
            reduce_ddp=True,
        )

        # Routing stats for diagnostics (not persistent)
        self._last_routing_stats: Optional[Dict[str, Any]] = None
        self.last_routing_snapshot: dict = {}
        self._last_dispatch_stats: dict = {}
        self._merge_metadata: dict = {"mode": "dynamic", "approximate": True, "expert_weights": []}
        self._merge_calibration_usage: Optional[torch.Tensor] = None
        self._merge_calibration_batches = 0

    def publish_aux_loss(self, *, step: int, training: bool) -> torch.Tensor:
        return publish_aux_loss(self, getattr(self, "_last_aux_loss", torch.zeros(())), step=step, kind="molora", training=training)

    def routing_snapshot(self) -> dict:
        return _routing_snapshot(self)

    def export_capabilities(self) -> dict:
        return _export_routing_capabilities(self)

    def __deepcopy__(self, memo):
        return robust_deepcopy(self, memo)

    # ── RoutedModule protocol ───────────────────────────────────────────

    @property
    def aux_loss(self) -> torch.Tensor:
        """Auxiliary loss from the most recent forward pass.

        Reads from ``MOE_LOSS_REGISTRY`` when ``share_moe_registry`` is True,
        otherwise returns the internally stored ``_last_aux_loss``.
        """
        if self.share_moe_registry:
            from ultralytics.nn.modules.moe.modules import MOE_LOSS_REGISTRY
            val = MOE_LOSS_REGISTRY.get(self)
            if val is not None:
                return val
        return getattr(self, "_last_aux_loss", torch.zeros(()))

    # ------------------------------------------------------------------
    # Property proxy to base layer (needed for ultralytics fuse/AutoBackend)
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Proxy geometric attributes to base_layer (Conv2d/Linear)."""
        proxy_names = (
            "out_channels", "in_channels", "kernel_size",
            "stride", "padding", "dilation", "groups", "bias",
            "out_features", "in_features",
        )
        if name in proxy_names:
            base_layer = self.__dict__.get("_modules", {}).get("base_layer")
            if base_layer is not None and hasattr(base_layer, name):
                return getattr(base_layer, name)
        return super().__getattr__(name)

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
        steps = min(self._step_count.item(), self.warmup_steps)
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
        # Ensure at least one expert is active without GPU→CPU sync:
        # use torch.where on a scalar tensor instead of mask.sum() > 0.
        any_active = mask.max() > 0
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
        """
        if self.capacity_factor <= 0 or self.capacity_factor >= 1.0:
            return top_k_weights
        B, K = top_k_weights.shape
        max_slots = max(1, int(math.ceil(self.capacity_factor * B * K / self.num_experts)))
        # Vectorised: compute per-expert usage and scale in one pass,
        # eliminating the Python for-loop over experts.
        one_hot = F.one_hot(top_k_indices.reshape(-1).long(), self.num_experts).float()  # [B*K, E]
        usage = one_hot.sum(dim=0)  # [E]
        max_slots_t = torch.tensor(float(max_slots), dtype=usage.dtype, device=usage.device)
        scale = torch.where(
            usage > max_slots_t,
            max_slots_t / usage.clamp_min(1.0),
            torch.ones_like(usage),
        )  # [E]
        # Gather scale per selected expert: scale[top_k_indices] → [B, K]
        weights = top_k_weights * scale[top_k_indices]
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
        mask = torch.zeros(self.num_experts, dtype=torch.bool)
        mask[active] = True
        self._domain_active_mask = mask

    def clear_domain(self) -> None:
        """Clear domain restriction, restore all experts."""
        self._domain_active_mask = None

    # ------------------------------------------------------------------
    # Routing-aware merge statistics
    # ------------------------------------------------------------------

    def _routing_contribution(
        self,
        top_k_weights: torch.Tensor,
        top_k_indices: torch.Tensor,
        *,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Return weighted expert usage from the final sparse routing decision."""
        contribution = torch.zeros(self.num_experts, device=top_k_weights.device, dtype=torch.float32)
        contribution.scatter_add_(
            0,
            top_k_indices.reshape(-1).long(),
            top_k_weights.detach().float().reshape(-1),
        )
        if normalize:
            contribution = contribution / contribution.sum().clamp_min(1e-12)
        return contribution

    def _record_routing_contribution(
        self,
        top_k_weights: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> None:
        """Update training EMA and any active calibration accumulator."""
        contribution = self._routing_contribution(top_k_weights, top_k_indices, normalize=False)
        normalized = contribution / contribution.sum().clamp_min(1e-12)
        if self.training:
            with torch.no_grad():
                self._usage_ema.mul_(self.usage_ema_decay).add_(
                    normalized.to(self._usage_ema.device), alpha=1.0 - self.usage_ema_decay
                )
        if self._merge_calibration_usage is not None:
            self._merge_calibration_usage.add_(contribution.to(self._merge_calibration_usage.device))
            self._merge_calibration_batches += 1

    def start_merge_calibration(self) -> None:
        """Start an ephemeral routing-usage accumulator for calibrated merge."""
        if self.merged:
            raise RuntimeError("Cannot calibrate an already merged MoLoRA layer")
        self._merge_calibration_usage = torch.zeros_like(self._usage_ema, dtype=torch.float32)
        self._merge_calibration_batches = 0

    def finish_merge_calibration(self) -> tuple[list[float], int]:
        """Return normalized calibration weights and clear the accumulator."""
        usage = self._merge_calibration_usage
        batches = self._merge_calibration_batches
        self.cancel_merge_calibration()
        if usage is None or batches <= 0 or not torch.isfinite(usage).all() or float(usage.sum()) <= 0:
            raise RuntimeError("MoLoRA layer received no valid routing observations during calibration")
        weights = usage.clamp_min(0) / usage.clamp_min(0).sum()
        return weights.detach().cpu().tolist(), batches

    def cancel_merge_calibration(self) -> None:
        """Discard an in-progress calibration accumulator."""
        self._merge_calibration_usage = None
        self._merge_calibration_batches = 0

    # ------------------------------------------------------------------
    # Expert freezing (continual learning)
    # ------------------------------------------------------------------

    def freeze_experts(self, expert_indices: List[int]) -> None:
        """Freeze specific experts so their weights are not updated during training."""
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

        # Increment step counter during training
        if self.training:
            self._step_count.add_(1)

        # Router
        router_logits = self.router(x)  # [B, E]

        # Apply domain restriction if set
        if self._domain_active_mask is not None:
            device = router_logits.device
            mask = self._domain_active_mask.to(device).to(router_logits.dtype)
            # Mask out inactive experts with large negative value.
            # Use torch.finfo to avoid fp16 underflow (-1e9 → -inf in fp16).
            neg_mask_val = torch.finfo(router_logits.dtype).min
            router_logits = router_logits + (1.0 - mask) * neg_mask_val

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

        self._record_routing_contribution(top_k_weights, top_k_indices)

        # Compute sparse expert outputs
        adapted = self._compute_sparse_experts(x, top_k_weights, top_k_indices, base_out)

        # Auxiliary loss (graph-connected to router_logits)
        aux_loss = (
            self.loss_fn(
                router_probs=router_probs,
                router_logits=router_logits,
                expert_indices=top_k_indices,
                expert_outputs=None,  # diversity loss disabled by default for efficiency
            )
            if self.training
            else base_out.new_zeros(())
        )

        # Write to MOE_LOSS_REGISTRY if enabled
        if self.share_moe_registry and self.training:
            try:
                from ultralytics.nn.modules.moe._common import _registry_set
                _registry_set(self, aux_loss)
            except (ImportError, AttributeError, RuntimeError) as exc:
                import logging
                logging.getLogger("molora").warning(
                    "Failed to register MoLoRA aux loss to MOE_LOSS_REGISTRY: %s", exc
                )

        # Store diagnostics
        self._last_routing_stats = {
            "top_k_indices": top_k_indices.detach(),
            "top_k_weights": top_k_weights.detach(),
            "expert_usage": self._expert_usage(top_k_indices),
            "effective_k": effective_k,
            "domain_mask": self._domain_active_mask,
        }

        # ── RoutedModule protocol: routing snapshot ────────────────────
        with torch.no_grad():
            self.last_routing_snapshot = {
                "num_experts": self.num_experts,
                "top_k": effective_k,
                "expert_usage": self._last_routing_stats["expert_usage"].float(),
                "mean_router_probs": router_probs.detach().float().mean(dim=0),
                "aux_loss": float(aux_loss.detach()),
            }
        self._last_aux_loss = aux_loss
        publish_aux_loss(self, aux_loss, kind="molora", training=self.training)

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
        if torch.jit.is_tracing() or torch.onnx.is_in_onnx_export():
            assignments = F.one_hot(top_k_indices, num_classes=self.num_experts).to(top_k_weights.dtype)
            dense_weights = (assignments * top_k_weights.unsqueeze(-1)).sum(dim=1)
            for expert_idx, expert in enumerate(self.experts):
                out_e = expert(x)
                shape = (-1,) + (1,) * (out_e.dim() - 1)
                expert_out = expert_out + out_e * dense_weights[:, expert_idx].view(shape)
            self._last_dispatch_stats = {
                "mode": "dense_export",
                "expert_calls": self.num_experts,
                "selected_samples": B,
                "top_k": K,
            }
            return expert_out
        grouped = B >= 4
        calls = 0
        for e in range(self.num_experts):
            mask = top_k_indices == e
            selected = mask.any(dim=1)
            batch_idx = torch.nonzero(selected, as_tuple=True)[0]
            if batch_idx.numel() == 0:
                continue
            calls += 1
            x_e = x[batch_idx]
            out_e = self.experts[e](x_e)
            weights = (top_k_weights[batch_idx] * mask[batch_idx].to(top_k_weights.dtype)).sum(dim=1)
            shape = (-1,) + (1,) * (out_e.dim() - 1)
            expert_out = expert_out.index_add(0, batch_idx, out_e * weights.view(shape))
        self._last_dispatch_stats = {
            "mode": "grouped_sparse" if grouped else "dense_small_batch",
            "expert_calls": calls,
            "selected_samples": B,
            "top_k": K,
        }
        return expert_out

    def _expert_usage(self, expert_indices: torch.Tensor) -> torch.Tensor:
        """Normalized expert usage histogram."""
        flat = expert_indices.reshape(-1).to(torch.long)
        counts = torch.bincount(flat, minlength=self.num_experts).float()
        return counts / counts.sum().clamp_min(1.0)

    # ------------------------------------------------------------------
    # Merge / Unmerge
    # ------------------------------------------------------------------

    def merge_weights(
        self,
        mode: str = "ema",
        calibration: Optional[List[float]] = None,
        calibration_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Merge all expert deltas into the base layer weight.

        After merge, forward skips the LoRA path.  This is useful for
        ONNX export / inference where you want zero adapter overhead.
        """
        if mode not in {"ema", "uniform", "calibrated"}:
            raise ValueError("MoLoRA merge mode must be 'ema', 'uniform', or 'calibrated'")
        if self.merged:
            return
        if mode == "ema":
            weights = self._usage_ema.detach().float().cpu()
            if not torch.isfinite(weights).all() or float(weights.sum()) <= 0:
                weights = torch.full((self.num_experts,), 1.0 / self.num_experts)
            else:
                weights = weights.clamp_min(0) / weights.clamp_min(0).sum()
        elif mode == "calibrated":
            if calibration is None or len(calibration) != self.num_experts:
                raise ValueError("calibration must provide one weight per expert")
            weights = torch.as_tensor(calibration, dtype=torch.float32)
            if not torch.isfinite(weights).all() or (weights < 0).any() or float(weights.sum()) <= 0:
                raise ValueError("calibration weights must be finite, non-negative, and non-zero")
            weights = weights / weights.sum()
        else:
            weights = torch.full((self.num_experts,), 1.0 / self.num_experts)
        with torch.no_grad():
            if self.experts[0].is_conv:
                for weight, e in zip(weights.tolist(), self.experts):
                    _merge_conv_delta(self.base_layer.weight, e.lora_A, e.lora_B, e.scaling * weight)
            else:
                for weight, e in zip(weights.tolist(), self.experts):
                    _merge_linear_delta(self.base_layer.weight, e.lora_A, e.lora_B, e.scaling * weight)

        self.merged = True
        self._merge_metadata = {"mode": mode, "approximate": True, "expert_weights": weights.tolist()}
        if calibration_metadata:
            self._merge_metadata.update(calibration_metadata)
        LOGGER.debug(f"[MoLoRA] Merged {self.num_experts} experts into base layer.")

    def unmerge_weights(self) -> None:
        """Restore the original base layer weight."""
        if not self.merged:
            return
        with torch.no_grad():
            weights = self._merge_metadata.get("expert_weights") or [1.0 / self.num_experts] * self.num_experts
            if self.experts[0].is_conv:
                for weight, e in zip(weights, self.experts):
                    _unmerge_conv_delta(self.base_layer.weight, e.lora_A, e.lora_B, e.scaling * weight)
            else:
                for weight, e in zip(weights, self.experts):
                    _unmerge_linear_delta(self.base_layer.weight, e.lora_A, e.lora_B, e.scaling * weight)
        self.merged = False
        LOGGER.debug("[MoLoRA] Unmerged experts from base layer.")

    def fuse_batchnorm(self, bn: nn.BatchNorm2d) -> None:
        """Fold a following BatchNorm into the base conv and every expert output."""
        if not isinstance(self.base_layer, nn.Conv2d):
            raise TypeError("MoLoRA BatchNorm fusion requires a Conv2d base layer")
        scale = bn.weight.div(torch.sqrt(bn.running_var + bn.eps)).detach()
        from ultralytics.utils.torch_utils import fuse_conv_and_bn

        self.base_layer = fuse_conv_and_bn(self.base_layer, bn)
        with torch.no_grad():
            for expert in self.experts:
                expert.lora_B.weight.mul_(scale[:, None, None, None])

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
