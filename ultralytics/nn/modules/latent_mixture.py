"""Dense latent mixture modules for YOLO feature routing."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.routing_protocol import (
    export_capabilities as _export_routing_capabilities,
    graph_connected_finite_zero,
    publish_aux_loss as _publish_aux_loss,
    routing_finite_diagnostics,
)
from ultralytics.utils.ops import make_divisible


_ROUTER_LOGIT_LIMIT = 30.0


@dataclass(frozen=True)
class LatentRoutingContext:
    latent: torch.Tensor
    scale_tokens: torch.Tensor
    logits: torch.Tensor
    probs: torch.Tensor


def _positive_int(value: int | float, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _non_negative_float(value: float, name: str) -> float:
    value = float(value)
    if value < 0.0 or not torch.isfinite(torch.tensor(value)):
        raise ValueError(f"{name} must be finite and non-negative, got {value}")
    return value


def _disabled_autocast(device_type: str):
    if device_type in {"cpu", "cuda", "mps"}:
        return torch.autocast(device_type=device_type, enabled=False)
    return nullcontext()


def _conv1x1(c1: int, c2: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(c1, c2, kernel_size=1, bias=False),
        nn.GroupNorm(1, c2),
        nn.SiLU(inplace=True),
    )


def _validate_inputs(
    xs: Sequence[torch.Tensor],
    channels: Sequence[int],
    *,
    require_same_spatial: bool,
) -> list[torch.Tensor]:
    if not isinstance(xs, (list, tuple)):
        raise TypeError(f"expected a list/tuple of tensors, got {type(xs)!r}")
    if len(xs) != len(channels):
        raise ValueError(f"expected {len(channels)} input tensors, got {len(xs)}")
    if not xs:
        raise ValueError("latent mixture requires at least one input tensor")

    first = xs[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError(f"input 0 must be a Tensor, got {type(first)!r}")
    if first.ndim != 4:
        raise ValueError(f"input 0 must be BCHW, got shape {tuple(first.shape)}")
    if not first.is_floating_point():
        raise TypeError(f"input 0 must be floating point, got dtype {first.dtype}")
    if first.shape[0] <= 0 or first.shape[2] <= 0 or first.shape[3] <= 0:
        raise ValueError(f"input 0 has invalid shape {tuple(first.shape)}")

    batch, _, height, width = first.shape
    device, dtype = first.device, first.dtype
    checked: list[torch.Tensor] = []
    for i, (x, expected_channels) in enumerate(zip(xs, channels)):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"input {i} must be a Tensor, got {type(x)!r}")
        if x.ndim != 4:
            raise ValueError(f"input {i} must be BCHW, got shape {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError(f"input {i} must be floating point, got dtype {x.dtype}")
        if x.shape[0] != batch:
            raise ValueError(f"input {i} batch {x.shape[0]} does not match input 0 batch {batch}")
        if x.device != device:
            raise ValueError(f"input {i} device {x.device} does not match input 0 device {device}")
        if x.dtype != dtype:
            raise ValueError(f"input {i} dtype {x.dtype} does not match input 0 dtype {dtype}")
        if int(x.shape[1]) != int(expected_channels):
            raise ValueError(f"input {i} channels {x.shape[1]} do not match expected {expected_channels}")
        if x.shape[2] <= 0 or x.shape[3] <= 0:
            raise ValueError(f"input {i} has invalid spatial size {tuple(x.shape[2:])}")
        if require_same_spatial and tuple(x.shape[2:]) != (height, width):
            raise ValueError(f"input {i} spatial size {tuple(x.shape[2:])} does not match input 0 {(height, width)}")
        if not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export():
            if not bool(torch.isfinite(x.detach()).all().item()):
                raise FloatingPointError(f"input {i} contains non-finite values")
        checked.append(x)
    return checked


class DenseChannelExpert(nn.Module):
    """Lightweight shape-preserving expert."""

    def __init__(self, channels: int, expert_ratio: float = 0.25):
        super().__init__()
        channels = _positive_int(channels, "channels")
        expert_ratio = float(expert_ratio)
        if not 0.0 < expert_ratio <= 1.0:
            raise ValueError(f"expert_ratio must be in (0, 1], got {expert_ratio}")
        hidden = make_divisible(max(8, int(round(channels * expert_ratio))), 8)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(1, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentRouter(nn.Module):
    """FP32 router with persistent temperature and train-only logit noise."""

    def __init__(
        self,
        latent_dim: int,
        num_experts: int,
        router_hidden_dim: int | None = None,
        temperature: float = 1.0,
        noise_std: float = 0.0,
        router_init_std: float = 0.0,
        num_tokens: int | None = None,
        per_token: bool = False,
    ):
        super().__init__()
        self.latent_dim = _positive_int(latent_dim, "latent_dim")
        self.num_experts = _positive_int(num_experts, "num_experts")
        hidden = self.latent_dim if router_hidden_dim is None else _positive_int(router_hidden_dim, "router_hidden_dim")
        temperature = float(temperature)
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.router_init_std = _non_negative_float(router_init_std, "router_init_std")
        self.num_tokens = None if num_tokens is None else _positive_int(num_tokens, "num_tokens")
        self.per_token = bool(per_token)
        self.norm = nn.LayerNorm(self.latent_dim)
        self.trunk = nn.Sequential(
            nn.Linear(self.latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.latent_dim),
            nn.SiLU(),
        )
        self.expert_head = nn.Linear(self.latent_dim, self.num_experts)
        if self.num_tokens is None:
            self.register_parameter("scale_embedding", None)
        else:
            self.scale_embedding = nn.Parameter(torch.empty(self.num_tokens, self.latent_dim, dtype=torch.float32))
            nn.init.normal_(self.scale_embedding, mean=0.0, std=0.02)
        self.register_buffer("_temperature", torch.tensor(float(temperature), dtype=torch.float32), persistent=True)
        self.register_buffer("_noise_std", torch.tensor(float(noise_std), dtype=torch.float32), persistent=True)
        if self.router_init_std > 0.0:
            nn.init.normal_(self.expert_head.weight, mean=0.0, std=self.router_init_std)
            nn.init.normal_(self.expert_head.bias, mean=0.0, std=self.router_init_std)
        else:
            nn.init.zeros_(self.expert_head.weight)
            nn.init.zeros_(self.expert_head.bias)
        self._cast_fp32()

    @property
    def temperature(self) -> torch.Tensor:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float | torch.Tensor) -> None:
        value_f = float(value.detach().reshape(())) if isinstance(value, torch.Tensor) else float(value)
        if value_f <= 0.0:
            raise ValueError(f"temperature must be positive, got {value_f}")
        with torch.no_grad():
            self._temperature.fill_(value_f)

    @property
    def noise_std(self) -> float:
        return float(self._noise_std.detach())

    @noise_std.setter
    def noise_std(self, value: float | torch.Tensor) -> None:
        self.set_noise_std(value)

    def set_noise_std(self, value: float | torch.Tensor) -> None:
        value_f = float(value.detach().reshape(())) if isinstance(value, torch.Tensor) else float(value)
        value_f = _non_negative_float(value_f, "noise_std")
        with torch.no_grad():
            self._noise_std.fill_(value_f)

    def _cast_fp32(self) -> None:
        for parameter in self.parameters(recurse=True):
            parameter.data = parameter.data.float()
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.float()
        for buffer in self.buffers(recurse=True):
            buffer.data = buffer.data.float()

    def _apply(self, fn):  # noqa: D401
        """Keep router math in FP32 after device/dtype transforms."""
        super()._apply(fn)
        self._cast_fp32()
        return self

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim not in {2, 3}:
            raise ValueError(f"LatentRouter expects [B,D] or [B,T,D], got shape {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.latent_dim:
            raise ValueError(f"token dim {tokens.shape[-1]} does not match latent_dim {self.latent_dim}")
        with _disabled_autocast(tokens.device.type):
            x = tokens.float()
            if x.ndim == 3:
                if self.num_tokens is not None and x.shape[1] != self.num_tokens:
                    raise ValueError(f"token count {x.shape[1]} does not match configured {self.num_tokens}")
                if self.scale_embedding is not None:
                    x = x + self.scale_embedding.to(device=x.device, dtype=torch.float32).unsqueeze(0)
                routed = x if self.per_token else x.mean(dim=1)
            else:
                routed = x
            hidden = self.trunk(self.norm(routed))
            logits = self.expert_head(hidden)
            if self.training and float(self._noise_std.detach()) > 0.0:
                logits = logits + torch.randn_like(logits) * float(self._noise_std.detach())
            safe_logits = torch.nan_to_num(logits, nan=0.0, posinf=_ROUTER_LOGIT_LIMIT, neginf=-_ROUTER_LOGIT_LIMIT)
            safe_logits = safe_logits.clamp(min=-_ROUTER_LOGIT_LIMIT, max=_ROUTER_LOGIT_LIMIT)
            probs = F.softmax(safe_logits / self._temperature.clamp_min(0.1), dim=-1)
        return safe_logits, probs


class _LatentAuxMixin:
    _routing_aux_kind = "latent"

    def _init_runtime_state(self) -> None:
        self._last_aux_loss = torch.zeros((), dtype=torch.float32)
        self.last_routing_snapshot: dict[str, Any] = {}
        self.last_routing_diagnostics: dict[str, Any] = {}

    @property
    def aux_loss(self) -> torch.Tensor:
        return self._last_aux_loss

    @property
    def last_aux_loss(self) -> torch.Tensor:
        return self._last_aux_loss

    @last_aux_loss.setter
    def last_aux_loss(self, value: torch.Tensor) -> None:
        self._last_aux_loss = value

    @property
    def routing(self) -> LatentRouter:
        return self.router

    @property
    def temperature(self) -> torch.Tensor:
        return self.router.temperature

    @temperature.setter
    def temperature(self, value: float | torch.Tensor) -> None:
        self.router.temperature = value

    @property
    def noise_std(self) -> float:
        return self.router.noise_std

    @noise_std.setter
    def noise_std(self, value: float | torch.Tensor) -> None:
        self.router.noise_std = value

    def set_noise_std(self, value: float | torch.Tensor) -> None:
        self.router.set_noise_std(value)

    def _compute_aux(
        self, logits: torch.Tensor, probs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training or (float(self.balance_loss_coeff) == 0.0 and float(self.router_z_loss_coeff) == 0.0):
            zero = graph_connected_finite_zero(logits, probs)
            return zero, zero.detach(), zero.detach()
        p = probs.float()
        importance = p.reshape(-1, p.shape[-1]).mean(dim=0)
        balance = self.num_experts * torch.sum(importance.square()) - 1.0
        balance = balance.clamp_min(0.0)
        z_loss = torch.logsumexp(logits.float(), dim=-1).square().mean()
        aux = float(self.balance_loss_coeff) * balance + float(self.router_z_loss_coeff) * z_loss
        return aux.reshape(()), balance.detach().reshape(()), z_loss.detach().reshape(())

    def _record_routing(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        aux: torch.Tensor,
        balance: torch.Tensor,
        z_loss: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            p = probs.detach().float()
            mean_probs = p.reshape(-1, p.shape[-1]).mean(dim=0)
            entropy = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(dim=-1).mean()
            snapshot: dict[str, Any] = {
                "family": "latent",
                "num_experts": int(self.num_experts),
                "top_k": int(self.top_k),
                "mean_router_probs": mean_probs.cpu(),
                "expert_usage": mean_probs.cpu(),
                "entropy": float(entropy.cpu()),
                "balance_loss": float(balance.cpu()),
                "z_loss": float(z_loss.cpu()),
                "aux_loss": float(aux.cpu()),
                "temperature": float(self.router.temperature.detach().cpu()),
                "noise_std": float(self.router._noise_std.detach().cpu()),
                "router_init_std": float(getattr(self.router, "router_init_std", 0.0)),
                "finite": bool(
                    routing_finite_diagnostics(logits=logits, probabilities=probs, aux_loss=aux).get("all_finite", True)
                ),
            }
            if probs.ndim == 3:
                snapshot["routing_axis"] = "scale_expert"
                snapshot["num_scales"] = int(probs.shape[1])
                snapshot["scale_mean_probs"] = p.mean(dim=0).cpu()
            else:
                snapshot["routing_axis"] = "expert"
            if hasattr(self, "residual_gain"):
                snapshot["residual_gain"] = self.residual_gain.detach().float().cpu()
            self.last_routing_snapshot = snapshot
            self.last_routing_diagnostics = routing_finite_diagnostics(logits=logits, probabilities=probs, aux_loss=aux)

    def routing_snapshot(self) -> dict[str, Any]:
        return dict(self.last_routing_snapshot)

    def publish_aux_loss(self, *, step: int, training: bool) -> torch.Tensor:
        return _publish_aux_loss(self, self.aux_loss, step=step, kind="latent", training=training)

    def export_capabilities(self) -> dict[str, Any]:
        capabilities = _export_routing_capabilities(self)
        capabilities.update(
            routing_kind="latent",
            sparse_dispatch=False,
            eager_sparse_dispatch=False,
            sparse_export_limitation="Latent mixture uses dense expert execution only.",
        )
        return capabilities


class LatentMixture(_LatentAuxMixin, nn.Module):
    """Single-scale latent mixture: multiple aligned features in, one feature out."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_experts: int = 4,
        expert_ratio: float = 0.25,
        router_hidden_dim: int | None = None,
        temperature: float = 1.0,
        balance_loss_coeff: float = 1e-2,
        router_z_loss_coeff: float = 1e-3,
        residual_init: float = 0.0,
        noise_std: float = 0.0,
        router_init_std: float = 0.0,
    ):
        super().__init__()
        if isinstance(in_channels, int):
            in_channels = [in_channels]
        self.in_channels = tuple(_positive_int(c, "in_channels") for c in in_channels)
        if not self.in_channels:
            raise ValueError("LatentMixture requires at least one input channel")
        self.out_channels = _positive_int(out_channels, "out_channels")
        self.num_inputs = len(self.in_channels)
        self.num_experts = _positive_int(num_experts, "num_experts")
        self.top_k = self.num_experts
        self.balance_loss_coeff = _non_negative_float(balance_loss_coeff, "balance_loss_coeff")
        self.router_z_loss_coeff = _non_negative_float(router_z_loss_coeff, "router_z_loss_coeff")
        self.base_proj = (
            nn.Identity()
            if self.in_channels[0] == self.out_channels
            else _conv1x1(self.in_channels[0], self.out_channels)
        )
        self.token_projs = nn.ModuleList(
            [nn.Identity() if c == self.out_channels else _conv1x1(c, self.out_channels) for c in self.in_channels]
        )
        self.router = LatentRouter(
            self.out_channels,
            self.num_experts,
            router_hidden_dim=router_hidden_dim,
            temperature=temperature,
            noise_std=noise_std,
            router_init_std=router_init_std,
            num_tokens=self.num_inputs,
            per_token=False,
        )
        self.experts = nn.ModuleList(
            DenseChannelExpert(self.out_channels, expert_ratio) for _ in range(self.num_experts)
        )
        self.residual_gain = nn.Parameter(torch.tensor(float(residual_init), dtype=torch.float32))
        self._init_runtime_state()

    def _build_context(self, xs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, LatentRoutingContext]:
        checked = _validate_inputs(xs, self.in_channels, require_same_spatial=True)
        base = self.base_proj(checked[0])
        tokens = []
        for x, proj in zip(checked, self.token_projs):
            token = proj(x)
            tokens.append(F.adaptive_avg_pool2d(token, 1).flatten(1).float())
        scale_tokens = torch.stack(tokens, dim=1)
        logits, probs = self.router(scale_tokens)
        return base, LatentRoutingContext(
            latent=scale_tokens.mean(dim=1), scale_tokens=scale_tokens, logits=logits, probs=probs
        )

    def forward(self, xs: Sequence[torch.Tensor]) -> torch.Tensor:
        base, context = self._build_context(xs)
        mixed = torch.zeros_like(base)
        for e, expert in enumerate(self.experts):
            gate = context.probs[:, e].to(device=base.device, dtype=base.dtype).view(-1, 1, 1, 1)
            mixed = mixed + expert(base) * gate
        output = base + self.residual_gain.to(device=base.device, dtype=base.dtype) * mixed
        aux, balance, z_loss = self._compute_aux(context.logits, context.probs)
        published = _publish_aux_loss(self, aux, kind="latent", training=self.training)
        self._last_aux_loss = published.detach()
        self._record_routing(context.logits, context.probs, self._last_aux_loss, balance, z_loss)
        return output


class MultiScaleLatentMixture(_LatentAuxMixin, nn.Module):
    """Multi-scale list-to-list latent mixture kept for compatibility/tests."""

    def __init__(
        self,
        channels: Sequence[int],
        latent_dim: int = 128,
        num_experts: int = 4,
        expert_ratio: float = 0.25,
        router_hidden_dim: int | None = None,
        temperature: float = 1.0,
        balance_loss_coeff: float = 1e-2,
        router_z_loss_coeff: float = 1e-3,
        residual_init: float = 0.0,
        noise_std: float = 0.0,
        router_init_std: float = 0.0,
    ):
        super().__init__()
        self.channels = tuple(_positive_int(c, "channels") for c in channels)
        if not self.channels:
            raise ValueError("MultiScaleLatentMixture requires at least one scale")
        self.latent_dim = _positive_int(latent_dim, "latent_dim")
        self.num_scales = len(self.channels)
        self.num_experts = _positive_int(num_experts, "num_experts")
        self.top_k = self.num_experts
        self.balance_loss_coeff = _non_negative_float(balance_loss_coeff, "balance_loss_coeff")
        self.router_z_loss_coeff = _non_negative_float(router_z_loss_coeff, "router_z_loss_coeff")
        self.input_projs = nn.ModuleList(
            [nn.Identity() if c == self.latent_dim else _conv1x1(c, self.latent_dim) for c in self.channels]
        )
        self.router = LatentRouter(
            self.latent_dim,
            self.num_experts,
            router_hidden_dim=router_hidden_dim,
            temperature=temperature,
            noise_std=noise_std,
            router_init_std=router_init_std,
            num_tokens=self.num_scales,
            per_token=True,
        )
        self.experts = nn.ModuleList(
            nn.ModuleList(DenseChannelExpert(c, expert_ratio) for _ in range(self.num_experts)) for c in self.channels
        )
        self.residual_gain = nn.Parameter(torch.full((self.num_scales,), float(residual_init), dtype=torch.float32))
        self._init_runtime_state()

    def _build_context(self, xs: Sequence[torch.Tensor]) -> LatentRoutingContext:
        checked = _validate_inputs(xs, self.channels, require_same_spatial=False)
        tokens = []
        for x, proj in zip(checked, self.input_projs):
            token = proj(x)
            tokens.append(F.adaptive_avg_pool2d(token, 1).flatten(1).float())
        scale_tokens = torch.stack(tokens, dim=1)
        logits, probs = self.router(scale_tokens)
        return LatentRoutingContext(
            latent=scale_tokens.mean(dim=1), scale_tokens=scale_tokens, logits=logits, probs=probs
        )

    def forward(self, xs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        checked = _validate_inputs(xs, self.channels, require_same_spatial=False)
        context = self._build_context(checked)
        outputs: list[torch.Tensor] = []
        for s, x in enumerate(checked):
            mixed = torch.zeros_like(x)
            for e, expert in enumerate(self.experts[s]):
                gate = context.probs[:, s, e].to(device=x.device, dtype=x.dtype).view(-1, 1, 1, 1)
                mixed = mixed + expert(x) * gate
            gain = self.residual_gain[s].to(device=x.device, dtype=x.dtype)
            outputs.append(x + gain * mixed)
        aux, balance, z_loss = self._compute_aux(context.logits, context.probs)
        published = _publish_aux_loss(self, aux, kind="latent", training=self.training)
        self._last_aux_loss = published.detach()
        self._record_routing(context.logits, context.probs, self._last_aux_loss, balance, z_loss)
        return outputs


__all__ = [
    "DenseChannelExpert",
    "LatentMixture",
    "LatentRouter",
    "LatentRoutingContext",
    "MultiScaleLatentMixture",
]
