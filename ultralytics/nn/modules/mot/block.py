"""Core Mixture-of-Transformer block."""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple
from ultralytics.nn.modules._numeric import should_reduce_ddp
from ultralytics.nn.modules.utils import get_safe_groups as _safe_groups, robust_deepcopy
from ultralytics.nn.modules.routing_protocol import (
    export_capabilities as _export_routing_capabilities,
    graph_connected_finite_zero,
    publish_aux_loss,
    routing_finite_diagnostics,
    routing_snapshot as _routing_snapshot,
)
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
        sparse_train_warmup_steps (int): Dense training forwards before enabled sparse dispatch begins.
        local_attn_window (int): LocalConv attention window; 0 keeps global attention.
        export_masked (bool): When True (default), traced/exported routers rebuild
            sparse-equivalent Top-K masked weights. Pass False for legacy dense softmax.

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
        sparse_train_warmup_steps: int = 0,
        scene_inference_mode: str = "dynamic",
        local_attn_window: int = 0,
        export_masked: bool = True,
    ):
        super().__init__()
        if not 1 <= top_k <= self.NUM_EXPERTS:
            raise ValueError(f"top_k must be in [1, {self.NUM_EXPERTS}], got {top_k}")
        self._top_k = int(top_k)
        self.balance_loss_coeff = balance_loss_coeff
        self.sparse_train = sparse_train
        if int(sparse_train_warmup_steps) < 0:
            raise ValueError("sparse_train_warmup_steps must be non-negative")
        self.sparse_train_warmup_steps = int(sparse_train_warmup_steps)
        self.register_buffer("_sparse_train_step", torch.tensor(0, dtype=torch.long), persistent=True)
        self._ddp_find_unused_parameters: Optional[bool] = None
        self._ddp_contract_source = "unconfigured"
        self._ddp_sparse_state_cache: Optional[Tuple[tuple, Tuple[bool, bool, Optional[str]]]] = None
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
        self.experts = nn.ModuleList(
            [
                _LocalConvTransformerExpert(dim, expert_heads, mlp_ratio, dropout, local_window_size=local_attn_window),
                _WindowTransformerExpert(
                    dim,
                    expert_heads,
                    window_size,
                    mlp_ratio,
                    dropout,
                    shift_size=window_size // 2 if window_shift else 0,
                ),
                _DeformableTransformerExpert(
                    dim,
                    expert_heads,
                    n_points,
                    mlp_ratio,
                    dropout,
                    align_corners=grid_align_corners,
                ),
            ]
        )

        # Router
        self.router = _MoTRouter(
            dim,
            self.NUM_EXPERTS,
            top_k,
            use_spatial=use_spatial_router,
            temperature=temperature,
            exploration_eps=exploration_eps,
            scene_aware=scene_aware_router,
            scene_hidden_dim=scene_hidden_dim,
            scene_inference_mode=scene_inference_mode,
            export_masked=export_masked,
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

    @property
    def top_k(self) -> int:
        """Number of Transformer experts active per forward."""
        return self._top_k

    @top_k.setter
    def top_k(self, value: int) -> None:
        value = int(value)
        if not 1 <= value <= self.NUM_EXPERTS:
            raise ValueError(f"top_k must be in [1, {self.NUM_EXPERTS}], got {value}")
        self._top_k = value
        if hasattr(self, "router"):
            self.router.top_k = value

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
        capabilities = _export_routing_capabilities(self)
        eager_sparse = self.top_k < self.NUM_EXPERTS
        sparse_train_ready = self._sparse_training_ready()
        ddp_active, ddp_sparse_safe, ddp_fallback_reason = self._ddp_sparse_training_state()
        capabilities.update(
            routing_kind="mot",
            sparse_dispatch=eager_sparse,
            eager_sparse_dispatch=eager_sparse,
            training_sparse_dispatch=bool(self.sparse_train),
            sparse_train=bool(self.sparse_train),
            sparse_train_warmup_steps=self.sparse_train_warmup_steps,
            sparse_train_step=int(self._sparse_train_step.item()),
            sparse_train_ready=sparse_train_ready,
            dispatch_policy=self._training_dispatch_policy(
                sparse_train_ready=sparse_train_ready,
                ddp_active=ddp_active,
                ddp_sparse_safe=ddp_sparse_safe,
            ),
            ddp_active=ddp_active,
            ddp_find_unused_parameters=self._ddp_find_unused_parameters,
            ddp_sparse_train_safe=ddp_sparse_safe,
            ddp_contract_source=self._ddp_contract_source,
            ddp_fallback_reason=ddp_fallback_reason,
            export_router_weights="masked_topk" if self.router.export_masked else "dense_softmax",
            sparse_export_limitation=(
                "MoT eager execution supports Top-K sparse dispatch; ONNX and TorchScript tracing rebuild "
                "sparse-equivalent masked Top-K router weights (bit-exact with eager dispatch)."
                if self.router.export_masked
                else "MoT eager execution supports Top-K sparse dispatch; ONNX and TorchScript tracing use dense blending "
                "because expert selection is data-dependent."
            ),
        )
        return capabilities

    def __deepcopy__(self, memo):
        return robust_deepcopy(self, memo)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Migrate checkpoints created before sparse-training warmup state existed."""
        step_key = f"{prefix}_sparse_train_step"
        if step_key not in state_dict:
            state_dict[step_key] = self._sparse_train_step.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _init_weights(self):
        nn.init.trunc_normal_(self.out_proj.weight, std=0.02)

    def _sparse_training_ready(self) -> bool:
        """Return whether training has completed the configured dense warmup."""
        return bool(self.sparse_train and int(self._sparse_train_step.item()) >= self.sparse_train_warmup_steps)

    def configure_ddp_sparse_training(self, *, find_unused_parameters: bool, source: str = "external") -> None:
        """Record the DDP unused-parameter contract required by sparse expert dispatch."""
        self._ddp_find_unused_parameters = bool(find_unused_parameters)
        self._ddp_contract_source = str(source)
        self._ddp_sparse_state_cache = None

    def _ddp_sparse_training_state(self) -> Tuple[bool, bool, Optional[str]]:
        """Return whether active DDP can safely tolerate locally unused experts."""
        dist_available = torch.distributed.is_available()
        dist_initialized = dist_available and torch.distributed.is_initialized()
        world_size = torch.distributed.get_world_size() if dist_initialized else 1
        cache_key = (self.training, dist_available, dist_initialized, world_size, self._ddp_find_unused_parameters)
        if self._ddp_sparse_state_cache is not None and self._ddp_sparse_state_cache[0] == cache_key:
            return self._ddp_sparse_state_cache[1]

        ddp_active = bool(self.training and dist_initialized and world_size > 1)
        if not ddp_active:
            state = (False, True, None)
        elif self._ddp_find_unused_parameters is True:
            state = (True, True, None)
        else:
            reason = (
                "find_unused_parameters_disabled"
                if self._ddp_find_unused_parameters is False
                else "find_unused_parameters_not_confirmed"
            )
            state = (True, False, reason)
        self._ddp_sparse_state_cache = (cache_key, state)
        return state

    def _training_dispatch_policy(
        self,
        *,
        sparse_train_ready: bool,
        ddp_active: bool,
        ddp_sparse_safe: bool,
    ) -> str:
        """Resolve the auditable dense/sparse dispatch policy for the current state."""
        if not self.training:
            return "eager_sparse"
        if not self.sparse_train:
            return "dense_train"
        if not sparse_train_ready:
            return "dense_warmup"
        if ddp_active and not ddp_sparse_safe:
            return "dense_ddp_fallback"
        return "sparse_train"

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
        # Export tracing always uses dense blending because nonzero/any control flow is input-dependent.
        exporting = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()
        out = x.new_zeros(x.shape)
        if exporting:
            # Keep the exported graph free of routing diagnostics and their
            # data-dependent Top-K mask. The router supplies dense softmax
            # weights in this mode, so every expert remains in the graph.
            for expert, weight in zip(self.experts, weights.unbind(dim=1)):
                out = out + expert(x) * weight.unsqueeze(1)
            return out

        sparse_train_ready = False if exporting else self._sparse_training_ready()
        ddp_active, ddp_sparse_safe, ddp_fallback_reason = self._ddp_sparse_training_state()
        dispatch_policy = self._training_dispatch_policy(
            sparse_train_ready=sparse_train_ready,
            ddp_active=ddp_active,
            ddp_sparse_safe=ddp_sparse_safe,
        )
        use_sparse = (not self.training or (sparse_train_ready and ddp_sparse_safe)) and not exporting
        warmup_step = int(self._sparse_train_step.item())
        B = x.shape[0]
        route_ids = indices if indices is not None else weights.argmax(dim=1, keepdim=True)
        route_mask = torch.zeros_like(weights, dtype=torch.bool)
        route_mask.scatter_(1, route_ids, True)
        token_mask_sparsity = 1.0 - float(route_mask.float().mean())
        experts_per_sample = route_mask.reshape(B, self.NUM_EXPERTS, -1).any(dim=2).sum(dim=1)
        # Torch 1.8 does not accept a tuple for ``Tensor.any(dim=...)``.
        # Flatten batch and spatial axes while retaining one expert axis.
        batch_expert_union = int(route_mask.permute(1, 0, 2, 3).reshape(self.NUM_EXPERTS, -1).any(dim=1).sum())
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
                w = weights[batch_idx, e_idx : e_idx + 1]
                expert_out = expert(x[batch_idx])
                if expert_out.shape != x[batch_idx].shape:
                    raise RuntimeError(
                        f"Expert {e_idx} is not shape-preserving: input {tuple(x[batch_idx].shape)} "
                        f"→ output {tuple(expert_out.shape)}. All experts must preserve "
                        f"the input tensor shape."
                    )
                out[batch_idx] = out[batch_idx] + (expert_out * w).to(out.dtype)
            mode, expert_calls_total = "sample_sparse", expert_calls
        else:
            for e_idx, expert in enumerate(self.experts):
                w = weights[:, e_idx : e_idx + 1]
                expert_out = expert(x)
                if expert_out.shape != x.shape:
                    raise RuntimeError(
                        f"Expert {e_idx} is not shape-preserving: input {tuple(x.shape)} "
                        f"→ output {tuple(expert_out.shape)}. All experts must preserve "
                        f"the input tensor shape."
                    )
                out = out + (expert_out * w).to(out.dtype)
            mode, expert_calls_total = "dense", len(self.experts)
        self._last_dispatch_stats = {
            "mode": mode,
            "policy": dispatch_policy,
            "expert_calls": expert_calls_total,
            "actual_expert_calls": expert_calls_total,
            "selected_samples": B,
            "selected_experts": batch_expert_union,
            "sparsity_ratio": token_mask_sparsity,
            "token_mask_sparsity": token_mask_sparsity,
            "experts_per_sample": experts_per_sample.detach().cpu(),
            "mean_experts_per_sample": float(experts_per_sample.float().mean()),
            "batch_expert_union": batch_expert_union,
            "warmup_step": warmup_step,
            "ddp_active": ddp_active,
            "ddp_find_unused_parameters": self._ddp_find_unused_parameters,
            "ddp_contract_source": self._ddp_contract_source,
            "ddp_fallback_reason": ddp_fallback_reason,
        }
        if self.training:
            # Count each training forward so the dense-warmup schedule advances.
            self._sparse_train_step.add_(1)
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            out           : [B, C, H, W]
            aux_loss      : scalar (GShard balance + router z-loss, 0 if both coeffs==0)
        """
        # ── Routing weights ──────────────────────────────────────────────
        weights, indices, router_logits = self.router(x, return_logits=True)  # [B, E, H, W]

        # ── Expert computation ───────────────────────────────────────────
        out = self._blend_experts(x, weights, indices)
        out = self.out_norm(self.out_proj(out))

        # Residual (block-level shortcut)
        out = out + x

        # ── Auxiliary loss ───────────────────────────────────────────────
        if self.training and (
            self.balance_loss_coeff > 0 or self.router_z_loss_coeff > 0 or self.scene_consistency_coeff > 0
        ):
            aux, finite_diagnostics = _mot_router_aux_loss(
                weights,
                router_logits,
                indices,
                self.NUM_EXPERTS,
                self.balance_loss_coeff,
                self.router_z_loss_coeff,
                reduce_ddp=should_reduce_ddp(self),
                return_diagnostics=True,
            )
            scene_consistency = self.router.scene_consistency_loss(weights)
            if self.scene_consistency_coeff > 0:
                aux = aux + self.scene_consistency_coeff * scene_consistency
                finite_diagnostics = routing_finite_diagnostics(
                    logits=router_logits, probabilities=weights, aux_loss=aux
                )
        else:
            aux = x.new_zeros(())
            scene_consistency = x.new_zeros(())
            finite_diagnostics = routing_finite_diagnostics(logits=router_logits, probabilities=weights, aux_loss=aux)

        exporting = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()
        if exporting:
            # Exported graphs intentionally carry only the inference tensor.
            # Aux loss publication and routing snapshots are eager diagnostics.
            return out, x.new_zeros(())
        if self.training and not torch.isfinite(aux):
            aux = graph_connected_finite_zero(weights, router_logits, aux)

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
                "scene_inference_mode": self.router.scene_inference_mode,
                "scene_aware_applied": self.router.last_scene_applied,
                "scene_bypass_reason": self.router.last_scene_bypass_reason,
                "scene_consistency_loss": float(scene_consistency.detach()),
                "finite_diagnostics": finite_diagnostics,
                "dispatch": dict(self._last_dispatch_stats),
            }

        return out, aux


__all__ = ("MoTBlock",)
