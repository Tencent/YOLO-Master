"""Core Mixture-of-Transformer block."""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple
from ultralytics.nn.modules._numeric import should_reduce_ddp
from ultralytics.nn.modules.utils import get_safe_groups as _safe_groups, robust_deepcopy
from ultralytics.nn.modules.routing_protocol import export_capabilities as _export_routing_capabilities, publish_aux_loss, routing_snapshot as _routing_snapshot
from .experts import _DeformableTransformerExpert, _LocalConvTransformerExpert, _WindowTransformerExpert
from .router import _MoTRouter, _mot_router_aux_loss
from ultralytics.utils import LOGGER

class MoTBlock(nn.Module):
    """Mixture-of-Transformers Block.

    Routes each spatial token (or image) across K of E complete Transformer
    experts, then softly combines their outputs via learned routing weights.

    Experts:
      0: LocalConvTransformer  — Conv-biased attention + Gated FFN
      1: WindowTransformer     — Swin-style shifted window attention + FFN
      2: DeformableTransformer — Deformable sparse sampling attention + FFN

    Args:
        dim (int): Channel dimension.
        num_heads (int): Attention heads per expert.
        top_k (int): Active experts per forward (1 or 2 recommended).
        window_size (int): Window size for WindowTransformer expert.
        n_points (int): Deformable sampling points per head.
        mlp_ratio (float): FFN expansion ratio.
        temperature (float): Router softmax temperature.
        use_spatial_router (bool): Token-level vs image-level routing.
        balance_loss_coeff (float): Weight of router z-loss (0 to disable).
        dropout (float): Dropout for attention/FFN.
        exploration_eps (float): Training-only dense routing floor that keeps all experts trainable.

    Shape:
        Input:  [B, dim, H, W]
        Output: [B, dim, H, W]
    """

    NUM_EXPERTS: int = 3

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        top_k: int = 2,
        window_size: int = 7,
        n_points: int = 4,
        mlp_ratio: float = 2.0,
        temperature: float = 1.0,
        use_spatial_router: bool = True,
        balance_loss_coeff: float = 0.01,
        router_z_loss_coeff: float | None = None,
        dropout: float = 0.0,
        exploration_eps: float = 0.02,
        window_shift: bool = False,
        grid_align_corners: bool = True,
        sparse_train: bool = False,
        scene_aware_router: bool = False,
        scene_hidden_dim: Optional[int] = None,
        scene_consistency_coeff: float = 0.0,
    ):
        super().__init__()
        if not 1 <= top_k <= self.NUM_EXPERTS:
            raise ValueError(f"top_k must be in [1, {self.NUM_EXPERTS}], got {top_k}")
        self.top_k = top_k
        self.balance_loss_coeff = balance_loss_coeff
        self.sparse_train = sparse_train
        self.scene_consistency_coeff = max(float(scene_consistency_coeff), 0.0)
        # Legacy YAML/checkpoints used balance_loss_coeff for z-loss only; when
        # router_z_loss_coeff is omitted, keep that behaviour for the z term.
        self.router_z_loss_coeff = balance_loss_coeff if router_z_loss_coeff is None else router_z_loss_coeff

        # Each expert uses the *full* dim with its own num_heads.
        # Find largest num_heads ≤ requested that divides dim.
        expert_heads = num_heads
        while dim % expert_heads != 0 and expert_heads > 1:
            expert_heads -= 1
        expert_heads = max(1, expert_heads)
        if expert_heads != num_heads:
            LOGGER.warning(
                f"MoTBlock(dim={dim}): num_heads {num_heads} reduced to {expert_heads} "
                f"(must divide dim and yield head_dim ≥ 1)"
            )

        # Three Transformer experts
        self.experts = nn.ModuleList([
            _LocalConvTransformerExpert(dim, expert_heads, mlp_ratio, dropout),
            _WindowTransformerExpert(dim, expert_heads, window_size, mlp_ratio, dropout,
                                     shift_size=window_size // 2 if window_shift else 0),
            _DeformableTransformerExpert(
                dim, expert_heads, n_points, mlp_ratio, dropout,
                align_corners=grid_align_corners,
            ),
        ])

        # Router
        self.router = _MoTRouter(
            dim, self.NUM_EXPERTS, top_k,
            use_spatial=use_spatial_router,
            temperature=temperature,
            exploration_eps=exploration_eps,
            scene_aware=scene_aware_router,
            scene_hidden_dim=scene_hidden_dim,
        )

        # Final output norm & projection
        self.out_norm = nn.GroupNorm(_safe_groups(dim, 8), dim)
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=False)

        self._init_weights()
        self.last_aux_loss: torch.Tensor | None = None
        self.last_routing_snapshot: dict = {}
        self._last_dispatch_stats: dict = {}

    # ── RoutedModule protocol ───────────────────────────────────────────
    @property
    def num_experts(self) -> int:
        """Number of Transformer expert branches."""
        return self.NUM_EXPERTS

    # top_k is already stored as self.top_k (instance attribute).

    @property
    def aux_loss(self) -> torch.Tensor:
        """Router auxiliary loss (GShard balance + z-loss). Zero outside training."""
        return self.last_aux_loss if self.last_aux_loss is not None else torch.zeros(())

    def publish_aux_loss(self, *, step: int, training: bool) -> torch.Tensor:
        value = self.last_aux_loss if self.last_aux_loss is not None else torch.zeros(())
        return publish_aux_loss(self, value, step=step, kind="mot", training=training)

    def routing_snapshot(self) -> dict:
        return _routing_snapshot(self)

    def export_capabilities(self) -> dict:
        return _export_routing_capabilities(self)

    def __deepcopy__(self, memo):
        return robust_deepcopy(self, memo)

    def _init_weights(self):
        nn.init.trunc_normal_(self.out_proj.weight, std=0.02)

    def _blend_experts(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Blend expert outputs; sparse when eval or ``sparse_train``.

        Sparse dispatch keys off discrete Top-K ``indices`` so training-time
        ``exploration_eps`` blending does not force every expert to run.

        .. warning::
            The sparse path uses ``torch.nonzero`` which produces **data-dependent
            control flow**.  This is safe for eager inference and ``torch.compile``
            (which handles dynamic shapes), but **not** for ONNX/TorchScript
            ``trace`` — tracing will unroll only the batch seen at trace time.
            ``torch.onnx.is_in_onnx_export()`` is checked to fall back to dense
            blending during export.  For TorchScript, use ``torch.jit.script``
            (not ``trace``) or set ``sparse_train=False`` and eval before export.
        """
        out = x.new_zeros(x.shape)
        # Export tracing always uses dense blending because nonzero/any control flow is input-dependent.
        exporting = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()
        use_sparse = (not self.training or self.sparse_train) and not exporting
        B = x.shape[0]
        if use_sparse:
            expert_calls = 0
            for e_idx, expert in enumerate(self.experts):
                if indices is not None:
                    active = (indices == e_idx).reshape(B, -1).any(dim=1)
                else:
                    active = weights[:, e_idx].reshape(B, -1).sum(dim=1) > 0
                batch_idx = torch.nonzero(active, as_tuple=True)[0]
                if batch_idx.numel() == 0:
                    continue
                expert_calls += 1
                w = weights[batch_idx, e_idx:e_idx + 1]
                expert_out = expert(x[batch_idx])
                if expert_out.shape != x[batch_idx].shape:
                    raise RuntimeError(
                        f"Expert {e_idx} is not shape-preserving: input {tuple(x[batch_idx].shape)} "
                        f"→ output {tuple(expert_out.shape)}. All experts must preserve "
                        f"the input tensor shape."
                    )
                out[batch_idx] = out[batch_idx] + expert_out * w
            self._last_dispatch_stats = {"mode": "sample_sparse", "expert_calls": expert_calls, "selected_samples": B}
        else:
            for e_idx, expert in enumerate(self.experts):
                w = weights[:, e_idx:e_idx + 1]
                expert_out = expert(x)
                if expert_out.shape != x.shape:
                    raise RuntimeError(
                        f"Expert {e_idx} is not shape-preserving: input {tuple(x.shape)} "
                        f"→ output {tuple(expert_out.shape)}. All experts must preserve "
                        f"the input tensor shape."
                    )
                out = out + expert_out * w
            self._last_dispatch_stats = {"mode": "dense", "expert_calls": len(self.experts), "selected_samples": B}
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            out           : [B, C, H, W]
            aux_loss      : scalar (GShard balance + router z-loss, 0 if both coeffs==0)
        """
        # ── Routing weights ──────────────────────────────────────────────
        weights, indices, router_logits = self.router(x, return_logits=True)   # [B, E, H, W]

        # ── Expert computation ───────────────────────────────────────────
        out = self._blend_experts(x, weights, indices)
        out = self.out_norm(self.out_proj(out))

        # Residual (block-level shortcut)
        out = out + x

        # ── Auxiliary loss ───────────────────────────────────────────────
        if self.training and (
            self.balance_loss_coeff > 0 or self.router_z_loss_coeff > 0 or self.scene_consistency_coeff > 0
        ):
            aux = _mot_router_aux_loss(
                weights,
                router_logits,
                indices,
                self.NUM_EXPERTS,
                self.balance_loss_coeff,
                self.router_z_loss_coeff,
                reduce_ddp=should_reduce_ddp(self),
            )
            scene_consistency = self.router.scene_consistency_loss(weights)
            if self.scene_consistency_coeff > 0:
                aux = aux + self.scene_consistency_coeff * scene_consistency
        else:
            aux = x.new_zeros(())
            scene_consistency = x.new_zeros(())

        self.last_aux_loss = aux
        publish_aux_loss(self, aux, kind="mot", training=self.training)

        # ── Routing snapshot (detached diagnostics) ──────────────────────
        with torch.no_grad():
            mean_w = weights.detach().float().mean(dim=(0, 2, 3))  # [E]
            self.last_routing_snapshot = {
                "num_experts": self.NUM_EXPERTS,
                "top_k": self.top_k,
                "expert_usage": mean_w,
                "mean_router_probs": mean_w,
                "aux_loss": float(aux.detach()),
                "scene_aware": self.router.scene_aware,
                "scene_stats": self.router.last_scene_stats,
                "scene_bias": self.router.last_scene_bias,
                "scene_consistency_loss": float(scene_consistency.detach()),
            }

        return out, aux

__all__ = ("MoTBlock",)
