"""Deployment helpers for detection-utility-supervised MoT routing."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from types import TracebackType
from typing import Any, Self

import torch

from ultralytics.nn.modules._numeric import stable_normalize
from ultralytics.nn.modules.mot.router import _MoTRouter


class UtilityRouterDeployment:
    """Temporarily blend a baseline MoT router with utility-trained weights.

    The baseline module remains attached to the detector. A detached utility
    copy computes a second distribution, and a forward hook blends the two
    distributions before fixed or adaptive Top-K dispatch. Closing the context
    removes the hook and restores the baseline adaptive-K configuration.
    """

    def __init__(
        self,
        router: _MoTRouter,
        utility_state_dict: Mapping[str, torch.Tensor],
        *,
        alpha: float,
        adaptive_k: bool = False,
        adaptive_k_threshold: float = 0.5,
        max_batch_router_kl: float | None = None,
    ):
        if not isinstance(router, _MoTRouter):
            raise TypeError("router must be a _MoTRouter")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0,1]")
        if not 0 < adaptive_k_threshold <= 1:
            raise ValueError("adaptive_k_threshold must be in (0,1]")
        if max_batch_router_kl is not None and max_batch_router_kl <= 0:
            raise ValueError("max_batch_router_kl must be positive")
        self.router = router
        self.alpha = float(alpha)
        self.adaptive_k = bool(adaptive_k)
        self.adaptive_k_threshold = float(adaptive_k_threshold)
        self.max_batch_router_kl = max_batch_router_kl
        self.last_mean_router_kl = 0.0
        self.drift_guard_triggered = False
        self.utility_router = copy.deepcopy(router)
        scene_weight = utility_state_dict.get("scene_projector.0.weight")
        if scene_weight is not None and self.utility_router.scene_projector is None:
            self.utility_router.enable_scene_aware(int(scene_weight.shape[0]))
        utility_weight = utility_state_dict.get("utility_projector.0.weight")
        if utility_weight is not None and getattr(self.utility_router, "utility_projector", None) is None:
            self.utility_router.enable_global_utility_head(int(utility_weight.shape[0]))
        self.utility_router.load_state_dict(utility_state_dict, strict=True)
        self.utility_router.configure_adaptive_k(False)
        self.utility_router.requires_grad_(False).eval()
        self._handle = None
        self._original_adaptive_k = bool(getattr(router, "adaptive_k", False))
        self._original_threshold = float(getattr(router, "adaptive_k_threshold", 0.5))
        self._original_selected_k = getattr(router, "last_selected_k", None)

    @staticmethod
    def _fixed_top_k(
        module: _MoTRouter,
        probabilities: torch.Tensor,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the router's existing fixed-K inference semantics."""
        if module.top_k < module.num_experts:
            top_values, indices = probabilities.topk(module.top_k, dim=1)
            top_weights = stable_normalize(top_values, dim=1)
            weights = torch.zeros_like(probabilities)
            weights.scatter_(1, indices, top_weights)
        else:
            weights = probabilities
            indices = (
                torch.arange(module.num_experts, device=x.device)
                .view(1, -1, 1, 1)
                .expand(x.shape[0], -1, x.shape[2], x.shape[3])
            )
        module.last_selected_k = torch.full(
            (x.shape[0],),
            module.top_k,
            dtype=torch.long,
            device=x.device,
        )
        return weights, indices

    def _blend_hook(self, module: _MoTRouter, inputs: tuple[Any, ...], output: Any) -> Any:
        if module.training:
            raise RuntimeError("UtilityRouterDeployment is evaluation-only")
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("MoT router did not receive a tensor input")
        if not isinstance(output, tuple) or len(output) not in {2, 3}:
            raise RuntimeError("MoT router output must be (weights, indices[, logits])")
        x = inputs[0]
        baseline_logits = module._compute_logits(x) if len(output) == 2 else output[2]
        utility_logits = self.utility_router._compute_logits(x)
        temperature = torch.as_tensor(module.temperature, device=x.device, dtype=torch.float32).clamp_min(1e-6)
        baseline_probabilities = torch.softmax(baseline_logits.float() / temperature, dim=1)
        utility_probabilities = torch.softmax(utility_logits.float() / temperature, dim=1)
        if self.max_batch_router_kl is None:
            self.last_mean_router_kl = 0.0
            self.drift_guard_triggered = False
        else:
            router_kl = utility_probabilities * (
                utility_probabilities.clamp_min(1e-8).log() - baseline_probabilities.clamp_min(1e-8).log()
            )
            self.last_mean_router_kl = float(router_kl.sum(dim=1).mean().detach())
            self.drift_guard_triggered = self.last_mean_router_kl > self.max_batch_router_kl
        effective_alpha = 0.0 if self.drift_guard_triggered else self.alpha
        probabilities = (1.0 - effective_alpha) * baseline_probabilities + effective_alpha * utility_probabilities
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)
        if self.adaptive_k:
            weights, indices = module._adaptive_top_k(probabilities)
        else:
            weights, indices = self._fixed_top_k(module, probabilities, x)
        weights = weights.to(dtype=x.dtype)
        blended_logits = temperature * probabilities.clamp_min(1e-8).log()
        return (weights, indices, blended_logits) if len(output) == 3 else (weights, indices)

    def __enter__(self) -> Self:
        if self._handle is not None:
            raise RuntimeError("UtilityRouterDeployment is already active")
        if self.router.training:
            raise RuntimeError("UtilityRouterDeployment is evaluation-only")
        self.router.configure_adaptive_k(self.adaptive_k, self.adaptive_k_threshold)
        self._handle = self.router.register_forward_hook(self._blend_hook)
        return self

    def close(self) -> None:
        """Remove the deployment hook and restore baseline routing settings."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.router.configure_adaptive_k(self._original_adaptive_k, self._original_threshold)
        self.router.last_selected_k = self._original_selected_k

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.close()
        return False


__all__ = ("UtilityRouterDeployment",)
