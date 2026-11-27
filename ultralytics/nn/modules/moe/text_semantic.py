"""Text-conditioned, objectness/semantic-decoupled MoE adapter.

Two changes relative to the shipped ``C3k2MoE``:

1. **Text-conditioned routing.** The original router only sees pooled visual
   features, so it cannot know *what* the current query asks for; in full-parameter
   training it drove its own gate down to 0.10-0.14 and stopped contributing. The
   semantic expert group here is routed by ``visual summary + projected text
   context``, so experts can specialise by semantic cluster instead of image
   statistics alone.

2. **Decoupled objectness / semantic experts.** Localisation and category
   matching have different requirements. The expert bank is split in two groups:
   objectness experts (vision-only router) and semantic experts (text-conditioned
   router), with independent learnable gates. Both gates start at zero, so the
   block is an exact no-op on top of a pretrained C3k2 and stays checkpoint
   compatible.

The bank is also far cheaper than ``A2C2fMoE``: one 1x1 down-projection, Top-K
depthwise 3x3 experts on the reduced width, one 1x1 up-projection. No area
attention, no per-token gather/scatter.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.routing_protocol import publish_aux_loss

from .loss import gshard_balance_loss

from ultralytics.nn.modules.block import C3k2

__all__ = ["C3k2TSMoE", "TextSemanticExpertBank", "set_route_context", "get_route_context", "clear_route_context"]

TEXT_DIM = 512  # YOLOE text embedding width

_route_context: torch.Tensor | None = None

# Optional query-agnostic routing ablation. Default off => byte-for-byte the
# original behaviour. "zero" feeds a zero context; "random" feeds one fixed
# random vector (same for every image, layer and step) so the semantic router
# sees a constant bias instead of the query text. No module or parameter count
# changes: the existing self.txt projection still receives the same shape.
_TEXT_ABLATION = os.getenv("YOLO_TSMOE_TEXT_ABLATION", "").strip().lower()
_ABLATION_SEED = int(os.getenv("YOLO_TSMOE_TEXT_ABLATION_SEED", "20261126"))
_fixed_random_ctx: torch.Tensor | None = None


def _ablated_context(vec: torch.Tensor) -> torch.Tensor:
    global _fixed_random_ctx
    if _TEXT_ABLATION == "zero":
        return torch.zeros_like(vec)
    if _TEXT_ABLATION == "random":
        if _fixed_random_ctx is None:
            g = torch.Generator(device="cpu").manual_seed(_ABLATION_SEED)
            _fixed_random_ctx = torch.randn(vec.shape[-1], generator=g)
        return _fixed_random_ctx.to(device=vec.device, dtype=vec.dtype).expand_as(vec)
    return vec


def set_route_context(text_feats: torch.Tensor | None) -> None:
    """Publish the current batch's text embeddings for routers to read.

    ``YOLOEModel`` holds ``txt_feats`` of shape (1, num_prompts, 512); we collapse the
    prompt axis to one context vector per image. Called once per forward, before the
    backbone runs.
    """
    global _route_context
    if text_feats is None:
        _route_context = None
        return
    with torch.no_grad():
        vec = text_feats.detach().float().mean(dim=1) if text_feats.dim() == 3 else text_feats.detach().float()
        vec = _ablated_context(vec)
    _route_context = vec


def clear_route_context() -> None:
    global _route_context
    _route_context = None


def get_route_context(batch: int, device, dtype) -> torch.Tensor | None:
    """Return a per-image context of shape (batch, 512), or None when unavailable."""
    if _route_context is None:
        return None
    vec = _route_context.to(device=device, dtype=dtype)
    if vec.shape[0] == 1 and batch > 1:
        vec = vec.expand(batch, -1)
    elif vec.shape[0] != batch:
        vec = vec[:batch]
    return vec


class TextSemanticRouter(nn.Module):
    """Top-K router over either visual-only or visual+text context."""

    def __init__(self, channels: int, num_experts: int, use_text: bool, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.vis = nn.Conv2d(channels, hidden, 1)
        self.txt = nn.Linear(TEXT_DIM, hidden) if use_text else None
        self.out = nn.Conv2d(hidden, num_experts, 1)
        # Start with a near-uniform router so no expert is starved before training.
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        v = self.vis(self.pool(x))
        if self.txt is not None and ctx is not None:
            if ctx.dim() == 3:  # (1, num_prompts, D) -> one context per image
                ctx = ctx.mean(dim=1)
            elif ctx.dim() == 2 and ctx.shape[0] != v.shape[0]:
                ctx = ctx.mean(dim=0, keepdim=True)
                ctx = ctx.expand(v.shape[0], -1)
            v = v + self.txt(ctx).view(-1, v.shape[1], 1, 1)
        return self.out(v).flatten(1)


class _DepthwiseExpert(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class TextSemanticExpertBank(nn.Module):
    """Split expert bank: objectness experts (vision route) + semantic experts (text route)."""

    def __init__(
        self,
        channels: int,
        e: float = 0.5,
        num_objectness: int = 2,
        num_semantic: int = 2,
        top_k: int = 1,
    ):
        super().__init__()
        hidden = max(int(channels * e), 8)
        self.hidden = hidden
        self.top_k = max(1, min(top_k, num_objectness, num_semantic))
        self.down = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.SiLU(inplace=True)
        )
        self.up = nn.Sequential(nn.Conv2d(hidden, channels, 1, bias=False), nn.BatchNorm2d(channels))
        self.objectness_experts = nn.ModuleList([_DepthwiseExpert(hidden) for _ in range(num_objectness)])
        self.semantic_experts = nn.ModuleList([_DepthwiseExpert(hidden) for _ in range(num_semantic)])
        self.objectness_router = TextSemanticRouter(channels, num_objectness, use_text=False)
        self.semantic_router = TextSemanticRouter(channels, num_semantic, use_text=True)
        # Independent gates, zero-initialised => exact no-op on top of the pretrained base.
        self.scale_objectness = nn.Parameter(torch.zeros(()))
        self.scale_semantic = nn.Parameter(torch.zeros(()))
        # GShard balance weight. Without it, top_k=1 lets a single expert absorb the
        # batch and the rest receive no gradient at all.
        self.balance_coeff = float(os.getenv("YOLO_TSMOE_BALANCE", "0.01"))
        self.aux_loss: torch.Tensor | None = None

    def _mix(self, h: torch.Tensor, experts: nn.ModuleList, logits: torch.Tensor):
        w = F.softmax(logits, dim=-1)
        top_w, top_i = w.topk(self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        out = h.new_zeros(h.shape)
        for k in range(self.top_k):
            idx = top_i[:, k]
            weight = top_w[:, k].view(-1, 1, 1, 1)
            for e_i, expert in enumerate(experts):
                mask = (idx == e_i).view(-1, 1, 1, 1)
                if mask.any():
                    out = out + expert(h) * weight * mask
        # Soft usage (mean routing probability per expert) is differentiable, so the
        # balance term trains the router rather than just measuring it.
        return out, w.mean(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down(x)
        ctx = get_route_context(x.shape[0], x.device, x.dtype)
        logits_obj = self.objectness_router(x)
        logits_sem = self.semantic_router(x, ctx)
        c_obj, usage_obj = self._mix(h, self.objectness_experts, logits_obj)
        c_sem, usage_sem = self._mix(h, self.semantic_experts, logits_sem)

        if self.training and self.balance_coeff > 0:
            balance = 0.5 * (
                gshard_balance_loss(usage_obj, len(self.objectness_experts))
                + gshard_balance_loss(usage_sem, len(self.semantic_experts))
            )
            publish_aux_loss(self, self.balance_coeff * balance, kind="moe", training=True)
            self.aux_loss = balance.detach()

        mixed = torch.tanh(self.scale_objectness) * c_obj + torch.tanh(self.scale_semantic) * c_sem
        return self.up(mixed)


class C3k2TSMoE(C3k2):
    """C3k2 with a zero-gated text/semantic MoE adapter.

    Inherits ``C3k2`` unchanged so the base branch keeps the exact parameter names of
    the released checkpoint; only the new adapter keys are absent, and they are
    zero-gated, so loading a C3k2 checkpoint reproduces the original function exactly.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        num_objectness: int = 2,
        num_semantic: int = 2,
        top_k: int = 1,
        adapter_e: float = 0.5,
    ):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.ts_adapter = TextSemanticExpertBank(
            c2,
            e=adapter_e,
            num_objectness=num_objectness,
            num_semantic=num_semantic,
            top_k=top_k,
        )
        self.gate_cap = float(__import__("os").getenv("YOLO_TSMOE_GATE_CAP", "0.02"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = super().forward(x)
        if not self.training and not torch.is_grad_enabled():
            if self.ts_adapter.scale_objectness.item() == 0.0 and self.ts_adapter.scale_semantic.item() == 0.0:
                return base
        return base + self.gate_cap * self.ts_adapter(base)
