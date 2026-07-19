"""YAML-facing wrappers and collection helpers for Mixture-of-Transformer."""
from __future__ import annotations
import torch
import torch.distributed as dist
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.utils import robust_deepcopy
from ultralytics.nn.modules.routing_protocol import collect_aux_loss, export_capabilities as _export_routing_capabilities, publish_aux_loss, routing_snapshot as _routing_snapshot
from ultralytics.utils import LOGGER
from .block import MoTBlock

class C2fMoT(nn.Module):
    """C2f-style feature-flow wrapper around MoTBlock.

    Provides the same [c1, c2, n, ...] interface as C3k2 / A2C2f / C2fMoA,
    enabling direct YAML substitution.

    The aux_loss from each MoTBlock is accumulated and accessible via
    ``self.last_aux_loss`` after each forward call.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of stacked MoTBlock layers.
        num_heads (int): Attention heads per expert.
        top_k (int): Active experts per token (1 or 2).
        window_size (int): Window size for WindowTransformer.
        n_points (int): Deformable sampling points.
        mlp_ratio (float): FFN expansion ratio.
        temperature (float): Router temperature.
        balance_loss_coeff (float): Router balance loss weight.
        e (float): Internal channel expansion ratio.

    Shape:
        Input:  [B, c1, H, W]
        Output: [B, c2, H, W]
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        num_heads: int = 6,
        top_k: int = 2,
        window_size: int = 7,
        n_points: int = 4,
        mlp_ratio: float = 2.0,
        temperature: float = 1.0,
        balance_loss_coeff: float = 0.01,
        e: float = 0.5,
        sparse_train: bool = False,
        scene_aware_router: bool = False,
        scene_hidden_dim: int | None = None,
        scene_consistency_coeff: float = 0.0,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # Clamp num_heads to valid range
        dim = self.c
        # Find largest num_heads ≤ requested that divides dim and gives head_dim ≥ 8
        eff_heads = num_heads
        while eff_heads > 1 and (dim % eff_heads != 0 or dim // eff_heads < 8):
            eff_heads -= 1
        eff_heads = max(1, eff_heads)
        if eff_heads != num_heads:
            LOGGER.warning(
                f"C2fMoT(c={dim}): num_heads {num_heads} reduced to {eff_heads} "
                f"(must divide channels and yield head_dim ≥ 8)"
            )

        # Alternate window shift by block index (Swin-style): even blocks use
        # regular windows, odd blocks use shifted windows. Fixed at build time
        # → deterministic inference and trace-stable export.
        self.m = nn.ModuleList(
            MoTBlock(
                dim=dim,
                num_heads=eff_heads,
                top_k=top_k,
                window_size=window_size,
                n_points=n_points,
                mlp_ratio=mlp_ratio,
                temperature=temperature,
                balance_loss_coeff=balance_loss_coeff,
                window_shift=bool(i % 2),
                sparse_train=sparse_train,
                scene_aware_router=scene_aware_router,
                scene_hidden_dim=scene_hidden_dim,
                scene_consistency_coeff=scene_consistency_coeff,
            )
            for i in range(n)
        )
        # Lazily set on first forward (always overwritten there). Kept as a
        # device-agnostic scalar so a pre-forward read never injects a stale,
        # wrong-device tensor; requires_grad=False ⇒ filtered by the collector.
        self.last_aux_loss: torch.Tensor = torch.zeros((), requires_grad=False)
        self.last_routing_snapshot: dict = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        aux_total = x.new_zeros(())
        for m in self.m:
            out, aux = m(y[-1])
            y.append(out)
            aux_total = aux_total + aux
        self.last_aux_loss = aux_total
        publish_aux_loss(self, aux_total, kind="mot", training=self.training, covered_modules=self.m)

        # ── Routing snapshot (aggregated from child MoTBlocks) ──────────
        with torch.no_grad():
            child_snaps = [getattr(m, "last_routing_snapshot", {}) for m in self.m]
            if child_snaps:
                usages = [s["expert_usage"] for s in child_snaps if "expert_usage" in s]
                mean_usage = sum(usages) / len(usages) if usages else x.new_zeros(3)
                self.last_routing_snapshot = {
                    "num_experts": MoTBlock.NUM_EXPERTS,
                    "top_k": self.m[0].top_k if self.m else 0,
                    "expert_usage": mean_usage,
                    "mean_router_probs": mean_usage,
                    "aux_loss": float(aux_total.detach()),
                }
            else:
                self.last_routing_snapshot = {}

        return self.cv2(torch.cat(y, dim=1))

    # ── RoutedModule protocol ───────────────────────────────────────────
    @property
    def num_experts(self) -> int:
        return MoTBlock.NUM_EXPERTS

    @property
    def top_k(self) -> int:
        """Active experts per forward (from first child MoTBlock)."""
        return self.m[0].top_k if self.m else 0

    @property
    def aux_loss(self) -> torch.Tensor:
        return self.last_aux_loss

    def publish_aux_loss(self, *, step: int, training: bool) -> torch.Tensor:
        return publish_aux_loss(
            self, self.last_aux_loss, step=step, kind="mot", training=training, covered_modules=self.m
        )

    def routing_snapshot(self) -> dict:
        return _routing_snapshot(self)

    def export_capabilities(self) -> dict:
        return _export_routing_capabilities(self)

    def __deepcopy__(self, memo):
        return robust_deepcopy(self, memo)

def _aux_loss_device(model: nn.Module) -> torch.device:
    """Best-effort device lookup for zero aux-loss fallbacks."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def collect_mot_aux_loss(model: nn.Module, ddp_sync: bool = True) -> torch.Tensor:
    """Sum all MoT router aux losses in the model and optionally DDP-sync across ranks.

    P1-2 fix: added ``ddp_sync`` parameter so that multi-GPU training gets correct
    global balance statistics rather than per-rank targets.

    Scans both C2fMoT wrappers and standalone MoTBlock instances.
    Uses a unified ``set[int]`` deduplication mechanism so blocks nested
    inside C2fMoT are not double-counted, regardless of traversal order.

    Args:
        model: YOLO model containing MoT modules.
        ddp_sync: If True (default), average the total aux-loss across DDP ranks
            in float32 before returning.

    Call in the training loss function:
        loss = det_loss + collect_mot_aux_loss(model)  # ddp_sync=True by default
    """
    total = collect_aux_loss(model, include_kinds=("mot",), device=_aux_loss_device(model))
    if not total.requires_grad:
        return total

    # P1-2 fix: DDP synchronize the aux-loss scalar so all ranks optimise a shared global target
    if ddp_sync and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        # Collective operations must not mutate a graph-connected tensor.
        with torch.no_grad():
            sync_tensor = total.detach().float().to(total.device).clone()
            dist.all_reduce(sync_tensor, op=dist.ReduceOp.SUM)
            global_value = sync_tensor / dist.get_world_size()
        # Keep the local autograd path while using the globally averaged value.
        total = total + (global_value.to(dtype=total.dtype) - total.detach())

    return total

__all__ = ("C2fMoT", "collect_mot_aux_loss")
