"""MoE-aware PEFT extensions for MoLoRA.

Provides:
  - PerExpertRankAllocator: heuristic rank allocation based on expert activation frequency
  - RouterCalibration: learnable low-rank correction on frozen router logits
  - MoLoRAMoEAwareLayer: extends MoLoRALayer with per-expert ranks + router calibration
  - MoLoRAMoEAwareConfig: configuration dataclass
  - build_moe_aware_layer: factory function

Reference: AAAI 2026 MoE-aware PEFT strategy (Sec. 3.7 extension).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER
from .config import MoLoRAConfig
from .layer import MoLoRALayer, MoLoRAExpert
from .router import build_router
from .loss import MoLoRALoss
from .utils import _molora_scales


# ---------------------------------------------------------------------------
# Per-Expert Rank Allocator
# ---------------------------------------------------------------------------

class PerExpertRankAllocator:
    """Allocate per-expert LoRA rank budget based on activation frequency.

    Modes:
      - "uniform": divide budget equally among experts.
      - "frequency": allocate proportionally to historical usage frequency,
        with a minimum rank floor to prevent degenerate experts.

    The allocation is stateless: ``allocate()`` is called with fresh usage
    histograms every time (e.g. from a running average updated each batch).
    """

    def __init__(
        self,
        num_experts: int,
        total_budget: int,
        min_rank: int = 2,
        mode: str = "frequency",
    ):
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        if total_budget < num_experts * min_rank:
            raise ValueError(
                f"total_budget ({total_budget}) must be >= num_experts * min_rank "
                f"({num_experts * min_rank})"
            )
        if mode not in ("uniform", "frequency"):
            raise ValueError(f"mode must be 'uniform' or 'frequency', got '{mode}'")

        self.num_experts = num_experts
        self.total_budget = total_budget
        self.min_rank = min_rank
        self.mode = mode

    def allocate(self, usage_history: torch.Tensor) -> List[int]:
        """Compute per-expert rank list from a usage histogram.

        Args:
            usage_history: [num_experts] normalized frequency (sum ≈ 1.0).
                           If empty / all-zero, falls back to uniform.

        Returns:
            List[int] of length num_experts, sum == total_budget.
        """
        usage = usage_history.float().cpu()
        if usage.numel() != self.num_experts:
            raise ValueError(
                f"usage_history size ({usage.numel()}) != num_experts ({self.num_experts})"
            )

        if self.mode == "uniform":
            base = self.total_budget // self.num_experts
            remainder = self.total_budget - base * self.num_experts
            ranks = [base] * self.num_experts
            # Distribute remainder to first experts
            for i in range(remainder):
                ranks[i] += 1
            return ranks

        # frequency mode
        # Step 1: allocate min_rank to everyone
        ranks = [self.min_rank] * self.num_experts
        remaining = self.total_budget - self.num_experts * self.min_rank

        # Guard against zero / negative usage
        usage = usage.clamp_min(0.0)
        usage_sum = usage.sum().item()
        if usage_sum < 1e-8:
            usage = torch.ones(self.num_experts) / self.num_experts
            usage_sum = 1.0

        # Step 2: distribute remaining budget proportionally
        props = (usage / usage_sum).tolist()
        # Integer allocation via largest-remainder method
        frac_ranks = [remaining * p for p in props]
        int_parts = [int(math.floor(v)) for v in frac_ranks]
        remainders = [v - int_parts[i] for i, v in enumerate(frac_ranks)]

        for i in range(self.num_experts):
            ranks[i] += int_parts[i]

        # Distribute leftover budget by largest remainder
        leftover = remaining - sum(int_parts)
        if leftover > 0:
            sorted_idx = sorted(range(self.num_experts), key=lambda j: remainders[j], reverse=True)
            for i in range(leftover):
                ranks[sorted_idx[i]] += 1

        # Sanity check
        assert sum(ranks) == self.total_budget, f"Rank sum {sum(ranks)} != budget {self.total_budget}"
        return ranks


# ---------------------------------------------------------------------------
# Router Calibration ΔW_r
# ---------------------------------------------------------------------------

class RouterCalibration(nn.Module):
    """Low-rank calibration term applied to frozen router logits.

    Given input x and router_logits from a frozen pretrained router:
        calibrated_logits = router_logits + B_r @ A_r(x)

    where:
        A_r: Linear/Conv  (in_channels -> r_r)
        B_r: Linear       (r_r -> num_experts)

    B_r is initialized to zero so training starts from the frozen router
    distribution (no disruption at step 0).
    """

    def __init__(self, in_channels: int, num_experts: int, r_r: int = 4):
        super().__init__()
        if r_r < 1:
            raise ValueError(f"r_r must be >= 1, got {r_r}")

        self.in_channels = in_channels
        self.num_experts = num_experts
        self.r_r = r_r

        # A_r: pool -> linear bottleneck
        self.lora_A = nn.Linear(in_channels, r_r, bias=False)
        # B_r: project to num_experts, zero-init
        self.lora_B = nn.Linear(r_r, num_experts, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] for Conv2d input, or [B, C] for Linear input.
            router_logits: [B, num_experts] from the frozen/base router.

        Returns:
            calibrated_logits: [B, num_experts]
        """
        # Global average pool for spatial inputs
        if x.dim() == 4:
            pooled = x.mean(dim=[2, 3])  # [B, C]
        elif x.dim() == 2:
            pooled = x
        else:
            raise ValueError(f"RouterCalibration expects 2D or 4D input, got {x.dim()}D")

        delta = self.lora_B(self.lora_A(pooled))  # [B, E]
        return router_logits + delta


# ---------------------------------------------------------------------------
# MoE-aware Config
# ---------------------------------------------------------------------------

@dataclass
class MoLoRAMoEAwareConfig(MoLoRAConfig):
    """Extended MoLoRA configuration with MoE-aware PEFT options.

    Adds:
      - router_calibration: whether to apply learnable ΔW_r on router logits
      - router_calib_rank: rank r_r of the router calibration bottleneck
      - per_expert_rank: whether to use per-expert rank allocation
      - rank_allocator_mode: "uniform" or "frequency"
      - rank_budget_total: total LoRA rank budget across all experts
      - rank_min: minimum rank per expert (floor)
    """

    router_calibration: bool = False
    router_calib_rank: int = 4

    per_expert_rank: bool = False
    rank_allocator_mode: str = "frequency"  # "uniform" | "frequency"
    rank_budget_total: int = 32
    rank_min: int = 2

    def __post_init__(self):
        super().__post_init__()
        if self.router_calib_rank < 1:
            raise ValueError(f"router_calib_rank must be >= 1, got {self.router_calib_rank}")
        if self.rank_allocator_mode not in ("uniform", "frequency"):
            raise ValueError(
                f"rank_allocator_mode must be 'uniform' or 'frequency', "
                f"got '{self.rank_allocator_mode}'"
            )
        if self.rank_budget_total < self.num_experts * self.rank_min:
            raise ValueError(
                f"rank_budget_total ({self.rank_budget_total}) must be >= "
                f"num_experts * rank_min ({self.num_experts * self.rank_min})"
            )


# ---------------------------------------------------------------------------
# MoE-aware MoLoRA Layer
# ---------------------------------------------------------------------------

class MoLoRAMoEAwareLayer(MoLoRALayer):
    """Extends MoLoRALayer with per-expert rank allocation and router calibration.

    New behaviour:
      1. Per-expert ranks: each expert can have its own LoRA rank r_e, allocated
         by a PerExpertRankAllocator from running usage statistics.
      2. Router calibration: a learnable low-rank term ΔW_r = B_r @ A_r(x) is
         added to the frozen router logits, allowing the routing distribution to
         adapt to the target task without modifying the pretrained router weights.

    The layer remains backward-compatible with standard MoLoRALayer when both
    ``per_expert_rank`` and ``router_calibration`` are disabled.
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
        # MoE-aware extensions
        router_calibration: Optional[RouterCalibration] = None,
        expert_ranks: Optional[List[int]] = None,
    ):
        # If per-expert ranks are provided, we cannot use the parent __init__
        # directly because it builds experts with uniform rank.  We manually
        # replicate the parent init logic here with per-expert rank support.
        nn.Module.__init__(self)
        self.base_layer = base_layer
        self.r = r  # base / default rank (used when expert_ranks is None)
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
        self._step_count_cpu: int = 0
        self._domain_active_mask: Optional[torch.Tensor] = None
        self._expert_frozen_mask: Optional[torch.Tensor] = None

        # P1 fix (merge_weights weighting): EMA of per-expert routing usage.
        # Must mirror MoLoRALayer.__init__ since we skip super().__init__.
        self.register_buffer(
            "_usage_ema", torch.full((num_experts,), 1.0 / num_experts), persistent=True
        )
        self._usage_ema_momentum = 0.99

        # MoE-aware additions
        self.router_calibration = router_calibration
        self._expert_ranks = expert_ranks  # cached for inspection

        # Freeze base layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        # Build experts — uniform or per-expert rank
        if expert_ranks is not None:
            if len(expert_ranks) != num_experts:
                raise ValueError(
                    f"expert_ranks length ({len(expert_ranks)}) != num_experts ({num_experts})"
                )
            self.experts = nn.ModuleList(
                MoLoRAExpert(
                    base_layer,
                    r=expert_ranks[e],
                    alpha=alpha,
                    dropout=dropout,
                    use_rslora=use_rslora,
                    init_type=expert_init,
                )
                for e in range(num_experts)
            )
        else:
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

        # Build router (same as parent)
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
            reduce_ddp=False,
        )

        self._last_routing_stats: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Forward override with calibration
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)

        if self.merged:
            return base_out

        if self.training:
            self._step_count.add_(1)

        # Router logits
        router_logits = self.router(x)  # [B, E]

        # Apply router calibration if present
        calibration_applied = False
        if self.router_calibration is not None:
            router_logits = self.router_calibration(x, router_logits)
            calibration_applied = True

        # Domain restriction
        if self._domain_active_mask is not None:
            device = router_logits.device
            mask = self._domain_active_mask.to(device).to(router_logits.dtype)
            router_logits = router_logits + (1.0 - mask) * -1e9

        router_probs = F.softmax(router_logits, dim=-1)
        router_probs = self._apply_expert_dropout(router_probs)
        effective_k = self._current_top_k()

        top_k_weights, top_k_indices = torch.topk(router_probs, effective_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        if 0 < self.capacity_factor < 1.0:
            top_k_weights = self._apply_capacity_limit(top_k_weights, top_k_indices)

        adapted = self._compute_sparse_experts(x, top_k_weights, top_k_indices, base_out)

        aux_loss = self.loss_fn(
            router_probs=router_probs,
            router_logits=router_logits,
            expert_indices=top_k_indices,
            expert_outputs=None,
        )

        if self.share_moe_registry and self.training:
            try:
                from ultralytics.nn.modules.moe.modules import _registry_set
                _registry_set(self, aux_loss)
            except Exception:
                pass

        # Enhanced diagnostics for MoE-aware mode
        self._last_routing_stats = {
            "top_k_indices": top_k_indices.detach(),
            "top_k_weights": top_k_weights.detach(),
            "expert_usage": self._expert_usage(top_k_indices),
            "effective_k": effective_k,
            "domain_mask": self._domain_active_mask,
            "calibration_applied": calibration_applied,
            "expert_ranks": self._expert_ranks,
        }

        return base_out + adapted

    def extra_repr(self) -> str:
        base = super().extra_repr()
        calib = f"calib={self.router_calibration is not None}"
        per_rank = f"per_rank={self._expert_ranks is not None}"
        return f"{base}, moe_aware=True, {calib}, {per_rank}"

    def __getattr__(self, name: str):
        """Proxy unknown attributes to the wrapped base layer.

        When a subclass defines ``__getattr__``, PyTorch's C-level ``tp_getattro``
        skips the default ``_modules`` / ``_buffers`` / ``_parameters`` lookup.
        We must manually replicate ``nn.Module``'s default behaviour before
        proxying to ``base_layer``.
        """
        # 1. Replicate nn.Module's default __getattr__ behaviour
        _parameters = self.__dict__.get('_parameters')
        if _parameters is not None and name in _parameters:
            return _parameters[name]

        _buffers = self.__dict__.get('_buffers')
        if _buffers is not None and name in _buffers:
            return _buffers[name]

        _modules = self.__dict__.get('_modules')
        if _modules is not None and name in _modules:
            return _modules[name]

        # 2. Guard against recursion if base_layer itself is missing
        if name == 'base_layer':
            raise AttributeError(name)

        # 3. Proxy to base_layer
        if _modules is not None and 'base_layer' in _modules:
            return getattr(_modules['base_layer'], name)

        raise AttributeError(name)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_moe_aware_layer(
    base_layer: nn.Module,
    config: Union[MoLoRAMoEAwareConfig, Dict[str, Any]],
    usage_history: Optional[torch.Tensor] = None,
) -> MoLoRAMoEAwareLayer:
    """Build a MoLoRAMoEAwareLayer from config, handling optional features.

    Args:
        base_layer: Conv2d or Linear layer to wrap.
        config: MoLoRAMoEAwareConfig or compatible dict.
        usage_history: [num_experts] frequency histogram for rank allocation.
                       Required when ``config.per_expert_rank=True`` and
                       ``config.rank_allocator_mode='frequency'``.

    Returns:
        MoLoRAMoEAwareLayer instance.
    """
    if isinstance(config, dict):
        cfg = MoLoRAMoEAwareConfig(**{k: v for k, v in config.items() if k in MoLoRAMoEAwareConfig.__dataclass_fields__})
    else:
        cfg = config

    # Router calibration
    router_calib = None
    if cfg.router_calibration:
        if isinstance(base_layer, nn.Conv2d):
            in_c = base_layer.in_channels
        else:
            in_c = base_layer.in_features
        router_calib = RouterCalibration(
            in_channels=in_c,
            num_experts=cfg.num_experts,
            r_r=cfg.router_calib_rank,
        )

    # Per-expert rank allocation
    expert_ranks = None
    if cfg.per_expert_rank:
        allocator = PerExpertRankAllocator(
            num_experts=cfg.num_experts,
            total_budget=cfg.rank_budget_total,
            min_rank=cfg.rank_min,
            mode=cfg.rank_allocator_mode,
        )
        if usage_history is None:
            # Fall back to uniform if no history provided
            if cfg.rank_allocator_mode == "frequency":
                # Provide a skewed default distribution so frequency mode behaves
                # differently from uniform; this enables meaningful ablation even
                # when no prior usage stats are available.
                if cfg.num_experts == 4:
                    usage_history = torch.tensor([0.5, 0.2, 0.2, 0.1])
                else:
                    # Generalise: linearly interpolate to any num_experts
                    x = torch.linspace(0, 1, cfg.num_experts)
                    usage_history = torch.exp(-3 * x)
                    usage_history = usage_history / usage_history.sum()
            else:
                usage_history = torch.ones(cfg.num_experts) / cfg.num_experts
        expert_ranks = allocator.allocate(usage_history)

    return MoLoRAMoEAwareLayer(
        base_layer=base_layer,
        r=cfg.r,
        alpha=cfg.alpha,
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        router_type=cfg.router_type,
        dropout=cfg.dropout,
        use_rslora=getattr(cfg, "use_rslora", True),
        balance_loss_coef=cfg.balance_loss_coef,
        z_loss_coef=cfg.z_loss_coef,
        diversity_loss_coef=cfg.diversity_loss_coef,
        expert_init=cfg.expert_init,
        share_moe_registry=cfg.share_moe_registry,
        router_hidden_dim=getattr(cfg, "router_hidden_dim", None),
        capacity_factor=cfg.capacity_factor,
        expert_dropout=cfg.expert_dropout,
        top_k_warmup=cfg.top_k_warmup,
        warmup_steps=cfg.warmup_steps,
        domain_experts=getattr(cfg, "domain_experts", None),
        router_calibration=router_calib,
        expert_ranks=expert_ranks,
    )
