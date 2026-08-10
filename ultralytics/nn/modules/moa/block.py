"""Core Mixture-of-Attention block."""

from __future__ import annotations
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules._numeric import should_reduce_ddp, stable_normalize
from ultralytics.nn.modules.utils import robust_deepcopy
from ultralytics.nn.modules.routing_protocol import (
    export_capabilities as _export_routing_capabilities,
    is_export_or_tracing,
    publish_aux_loss,
    routing_finite_diagnostics,
    routing_snapshot as _routing_snapshot,
)
from .heads import _GlobalAttnHead, _LocalAttnHead, _RegionalAttnHead, _init_conv_weights
from .router import _MoARouter, _moa_router_aux_loss


def _resolve_sparse_inference_threshold(
    sparse_inference_threshold: float | None,
    inference_sparse_threshold: float | None,
) -> float:
    """Resolve the canonical threshold and its legacy constructor alias."""
    if sparse_inference_threshold is not None and inference_sparse_threshold is not None:
        if float(sparse_inference_threshold) != float(inference_sparse_threshold):
            raise ValueError(
                "sparse_inference_threshold and inference_sparse_threshold must match when both are provided"
            )
    threshold = (
        inference_sparse_threshold
        if sparse_inference_threshold is None and inference_sparse_threshold is not None
        else sparse_inference_threshold
    )
    return 0.02 if threshold is None else float(threshold)


class MoABlock(nn.Module):
    """Mixture-of-Attention Block.

    Routes each spatial token softly over three attention head-groups:
      - local  (DW-biased, captures fine-grained detail)
      - regional (stride-2 pooled KV, mid-range context)
      - global (linear attention, scene semantics)

    A lightweight FFN follows the attention mixture.

    Args:
        dim (int): Channel dimension (input == output).
        num_heads (int): Total attention heads, split equally over groups.
        mlp_ratio (float): FFN expansion ratio.
        temperature (float): Router softmax temperature. Starts at 1.0 and
            is automatically annealed each epoch by the trainer's
            ``_anneal_moa_mot_temperature`` callback (factor=0.97,
            min_temp=0.3 by default). Override via CLI args
            ``moa_mot_temperature_factor`` / ``moa_mot_min_temperature``.
        attn_drop (float): Dropout on attention output.
        shortcut (bool): Residual connection around the block.

    Shape:
        - Input:  [B, dim, H, W]
        - Output: [B, dim, H, W]
    """

    NUM_GROUPS: int = 3  # local / regional / global

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        temperature: float = 1.0,
        attn_drop: float = 0.0,
        shortcut: bool = True,
        aux_loss_coeff: float = 0.01,
        block_index: int = 0,
        local_window_size: int = 7,
        sequential_heads: bool = True,
        regional_max_kv_tokens: int | None = 4096,
        sparse_inference: bool = False,
        sparse_inference_threshold: float = 0.02,
        inference_sparse_threshold: float | None = None,
    ):
        super().__init__()
        if inference_sparse_threshold is not None:
            if sparse_inference_threshold != 0.02 and sparse_inference_threshold != inference_sparse_threshold:
                raise ValueError(
                    "Specify only one sparse inference threshold: "
                    "sparse_inference_threshold or inference_sparse_threshold."
                )
            sparse_inference_threshold = inference_sparse_threshold
            sparse_inference = True
        self.sequential_heads = sequential_heads
        self.sparse_inference = bool(sparse_inference or inference_sparse_threshold is not None)
        self.sparse_inference_threshold = _resolve_sparse_inference_threshold(
            sparse_inference_threshold, inference_sparse_threshold
        )
        if not 0.0 <= self.sparse_inference_threshold < 1.0:
            raise ValueError("sparse_inference_threshold must be in [0, 1)")
        if num_heads <= 0 or num_heads % self.NUM_GROUPS != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be positive and divisible by NUM_GROUPS ({self.NUM_GROUPS})"
            )
        self.shortcut = shortcut
        self.aux_loss_coeff = aux_loss_coeff
        head_dim = max(dim // num_heads, 16)
        heads_per_group = num_heads // self.NUM_GROUPS

        # Three attention head-groups (global head uses a per-block RF seed).
        global_rf_seed = block_index * 7919 + 2 * 65537
        self.local_head = _LocalAttnHead(dim, heads_per_group, head_dim, window_size=local_window_size)
        self.region_head = _RegionalAttnHead(dim, heads_per_group, head_dim, max_kv_tokens=regional_max_kv_tokens)
        self.global_head = _GlobalAttnHead(dim, heads_per_group, head_dim, rf_seed=global_rf_seed)

        # Router
        self.router = _MoARouter(dim, self.NUM_GROUPS, temperature=temperature)

        # Fusion conv (combine weighted head outputs)
        self.fusion = Conv(dim, dim, 1, act=False)
        self.attn_drop = nn.Dropout2d(attn_drop) if attn_drop > 0 else nn.Identity()

        # Layer-scale (initialised at 0.1: small residual contribution for
        # stable early training, à la CaiT LayerScale).
        # P2-3 fix: when shortcut=False, the block has no residual path so LayerScale=0.1
        # would cause gradient vanishing in deep networks. Initialise to 1.0 instead.
        ls_init = torch.ones(dim, 1, 1) if not shortcut else torch.ones(dim, 1, 1) * 0.1
        self.ls_attn = nn.Parameter(ls_init)

        # FFN
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            Conv(dim, hidden, 1),
            Conv(hidden, dim, 1, act=False),
        )
        ls_ffn_init = torch.ones(dim, 1, 1) if not shortcut else torch.ones(dim, 1, 1) * 0.1
        self.ls_ffn = nn.Parameter(ls_ffn_init)
        self.last_aux_loss: torch.Tensor = torch.zeros((), requires_grad=False)
        self.last_routing_snapshot: dict = {}

        self._init_weights()

    def _init_weights(self):
        _init_conv_weights(self)

    # ── RoutedModule protocol ───────────────────────────────────────────
    @property
    def num_experts(self) -> int:
        """Number of attention head-groups (local/regional/global)."""
        return self.NUM_GROUPS

    @property
    def top_k(self) -> int:
        """All groups always active (soft routing, dense mixture)."""
        return self.NUM_GROUPS

    @property
    def aux_loss(self) -> torch.Tensor:
        """Router auxiliary loss (balance regularizer). Zero outside training."""
        return self.last_aux_loss

    def publish_aux_loss(self, *, step: int, training: bool) -> torch.Tensor:
        return publish_aux_loss(self, self.last_aux_loss, step=step, kind="moa", training=training)

    def routing_snapshot(self) -> dict:
        return _routing_snapshot(self)

    def export_capabilities(self) -> dict:
        capabilities = _export_routing_capabilities(self)
        eager_sparse = bool(self.sparse_inference)
        capabilities.update(
            routing_kind="moa",
            sparse_dispatch=eager_sparse,
            eager_sparse_dispatch=eager_sparse,
            training_sparse_dispatch=False,
            sparse_export_limitation=(
                "MoA optional eager inference skips low-weight head groups; ONNX and TorchScript tracing use "
                "the dense fallback."
                if eager_sparse
                else "MoA uses dense soft routing in eager and exported graphs."
            ),
        )
        return capabilities

    def __deepcopy__(self, memo):
        return robust_deepcopy(self, memo)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # ── Routing weights ──────────────────────────────────────────────
        weights, router_logits = self.router(x, return_logits=True)  # [B, 3, H, W]
        exporting = is_export_or_tracing()
        if not exporting and self.training and self.aux_loss_coeff > 0:
            self.last_aux_loss, finite_diagnostics = _moa_router_aux_loss(
                weights,
                router_logits,
                self.aux_loss_coeff,
                reduce_ddp=should_reduce_ddp(self),
                return_diagnostics=True,
            )
        elif exporting:
            self.last_aux_loss = x.new_zeros(())
            finite_diagnostics = {}
        else:
            self.last_aux_loss = x.new_zeros(())
            finite_diagnostics = routing_finite_diagnostics(
                logits=router_logits, probabilities=weights, aux_loss=self.last_aux_loss
            )
        if not exporting:
            publish_aux_loss(self, self.last_aux_loss, kind="moa", training=self.training)

        # ── Routing snapshot (detached diagnostics) ──────────────────────
        if not exporting:
            with torch.no_grad():
                mean_w = weights.detach().float().mean(dim=(0, 2, 3))  # [3]
                self.last_routing_snapshot = {
                    "num_experts": self.NUM_GROUPS,
                    "top_k": self.NUM_GROUPS,
                    "expert_usage": mean_w,
                    "mean_router_probs": mean_w,
                    "aux_loss": float(self.last_aux_loss.detach()),
                    "finite_diagnostics": finite_diagnostics,
                }
        else:
            self.last_routing_snapshot = {}

        # ── Attention head outputs ────────────────────────────────────────
        exporting = torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()
        sparse_eval = not self.training and self.sparse_inference and not exporting
        active = None
        executed_groups = self.NUM_GROUPS
        dropped_routing_mass = 0.0
        if sparse_eval:
            # Skip a group only when it contributes less than the configured
            # threshold for every token in this batch.  The decision is batch-
            # level, so no per-token Python control flow enters exported graphs.
            active = weights.detach().amax(dim=(0, 2, 3)) > self.sparse_inference_threshold
            if not bool(active.any()):
                dominant = weights.detach().mean(dim=(0, 2, 3)).argmax()
                active = torch.arange(self.NUM_GROUPS, device=weights.device) == dominant
            if bool(active.all()):
                active = None
            else:
                active_weights = weights * active.view(1, -1, 1, 1).to(dtype=weights.dtype)
                approximation_error = float((weights - active_weights).sum(dim=1).mean().detach())
                weights = stable_normalize(active_weights, dim=1)
                executed_groups = int(active.sum())

        if not exporting:
            self.last_routing_snapshot.update(
                executed_groups=executed_groups,
                approximation_error=approximation_error,
            )

        w_l = weights[:, 0:1]  # [B, 1, H, W]
        w_r = weights[:, 1:2]
        w_g = weights[:, 2:3]

        if active is not None:
            mixed = x.new_zeros(x.shape)
            if bool(active[0]):
                mixed = mixed + w_l * self.local_head(x)
            if bool(active[1]):
                mixed = mixed + w_r * self.region_head(x)
            if bool(active[2]):
                mixed = mixed + w_g * self.global_head(x)
        elif self.sequential_heads:
            # Sequential path: compute and accumulate one head at a time.
            # Mathematically identical to the default path; useful for
            # memory-constrained environments and ONNX export validation.
            mixed = w_l * self.local_head(x)
            mixed = mixed + w_r * self.region_head(x)
            mixed = mixed + w_g * self.global_head(x)
        else:
            out_l = self.local_head(x)
            out_r = self.region_head(x)
            out_g = self.global_head(x)
            mixed = w_l * out_l + w_r * out_r + w_g * out_g  # [B, C, H, W]
        mixed = self.attn_drop(self.fusion(mixed))

        # ── Routing snapshot (detached diagnostics) ──────────────────────
        if not exporting:
            with torch.no_grad():
                mean_w = weights.detach().float().mean(dim=(0, 2, 3))  # [3]
                self.last_routing_snapshot = {
                    "num_experts": self.NUM_GROUPS,
                    "top_k": self.NUM_GROUPS,
                    "expert_usage": mean_w,
                    "mean_router_probs": mean_w,
                    "aux_loss": float(self.last_aux_loss.detach()),
                    "finite_diagnostics": finite_diagnostics,
                    "executed_groups": executed_groups,
                    "dropped_routing_mass": dropped_routing_mass,
                    # Legacy diagnostics exposed this value as an approximation
                    # error. It is a routing-mass proxy, not a tensor error norm.
                    "approximation_error": dropped_routing_mass,
                }
        else:
            self.last_routing_snapshot = {}

        # ── Residual + layer-scale ────────────────────────────────────────
        # `shortcut` controls *all* block-level residual paths consistently:
        #   True  → pre-activation residual around both attention and FFN
        #   False → pure feed-forward transform (no residual anywhere), so the
        #           block fully replaces its input rather than refining it.
        if self.shortcut:
            x = x + self.ls_attn * mixed
            x = x + self.ls_ffn * self.ffn(x)
        else:
            x = self.ls_attn * mixed
            x = self.ls_ffn * self.ffn(x)

        return x


__all__ = ("MoABlock",)
