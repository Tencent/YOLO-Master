"""Core Mixture-of-Attention block."""
from __future__ import annotations
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules._numeric import should_reduce_ddp
from ultralytics.nn.modules.utils import robust_deepcopy
from ultralytics.nn.modules.routing_protocol import export_capabilities as _export_routing_capabilities, publish_aux_loss, routing_snapshot as _routing_snapshot
from .heads import _GlobalAttnHead, _LocalAttnHead, _RegionalAttnHead, _init_conv_weights
from .router import _MoARouter, _moa_router_aux_loss

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
        sequential_heads: bool = False,
    ):
        super().__init__()
        self.sequential_heads = sequential_heads
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
        self.local_head   = _LocalAttnHead(dim, heads_per_group, head_dim, window_size=local_window_size)
        self.region_head  = _RegionalAttnHead(dim, heads_per_group, head_dim)
        self.global_head  = _GlobalAttnHead(dim, heads_per_group, head_dim, rf_seed=global_rf_seed)

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
        return _export_routing_capabilities(self)

    def __deepcopy__(self, memo):
        return robust_deepcopy(self, memo)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # ── Routing weights ──────────────────────────────────────────────
        weights, router_logits = self.router(x, return_logits=True)   # [B, 3, H, W]
        if self.training and self.aux_loss_coeff > 0:
            self.last_aux_loss = _moa_router_aux_loss(
                weights, router_logits, self.aux_loss_coeff, reduce_ddp=should_reduce_ddp(self)
            )
        else:
            self.last_aux_loss = x.new_zeros(())
        publish_aux_loss(self, self.last_aux_loss, kind="moa", training=self.training)

        # ── Routing snapshot (detached diagnostics) ──────────────────────
        with torch.no_grad():
            mean_w = weights.detach().float().mean(dim=(0, 2, 3))  # [3]
            self.last_routing_snapshot = {
                "num_experts": self.NUM_GROUPS,
                "top_k": self.NUM_GROUPS,
                "expert_usage": mean_w,
                "mean_router_probs": mean_w,
                "aux_loss": float(self.last_aux_loss.detach()),
            }

        w_l = weights[:, 0:1]     # [B, 1, H, W]
        w_r = weights[:, 1:2]
        w_g = weights[:, 2:3]

        # ── Attention head outputs ────────────────────────────────────────
        if self.sequential_heads:
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
            mixed = w_l * out_l + w_r * out_r + w_g * out_g   # [B, C, H, W]
        mixed = self.attn_drop(self.fusion(mixed))

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
