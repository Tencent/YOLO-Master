"""Text-conditioned sparse two-expert routing for detector feature maps."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.routing_protocol import (
    current_aux_step,
    export_capabilities as _export_routing_capabilities,
    publish_aux_loss,
    routing_snapshot as _routing_snapshot,
)

from .router import _MoTRouter


class TextConditionedMoT(nn.Module):
    """Route a detector feature map through one of two text-conditioned experts.

    The condition is caller-owned and detached at the module boundary.  The
    module is deliberately separate from the YOLOE classifier prompt tensor:
    callers may change routing conditions without changing the OVD head's
    ``tpe``/``txt_feats`` input.
    """

    NUM_EXPERTS = 2
    publishes_aux_loss = True

    def __init__(
        self,
        c1: int,
        c2: int,
        text_dim: int = 512,
        hidden_dim: int = 64,
        balance_loss_coeff: float = 0.01,
    ):
        super().__init__()
        if c1 <= 0 or c2 <= 0:
            raise ValueError(f"feature channels must be positive, got c1={c1}, c2={c2}")
        if text_dim <= 0 or hidden_dim <= 0:
            raise ValueError(f"text_dim and hidden_dim must be positive, got {text_dim}, {hidden_dim}")
        if balance_loss_coeff < 0:
            raise ValueError(f"balance_loss_coeff must be non-negative, got {balance_loss_coeff}")

        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)
        self.balance_loss_coeff = float(balance_loss_coeff)
        self._top_k = 1
        # Hard sample-level Top-1 is the frozen attempt-001 behavior.  The
        # explicit soft mode is reserved for the pre-registered starvation
        # repair (10 dense soft steps followed by 10 hard steps); callers must
        # opt into it, so normal inference/training semantics are unchanged.
        self._routing_mode = "hard"

        self.input_projection = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)
        self.condition_projection = nn.Linear(self.text_dim, self.hidden_dim)
        self.visual_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c2, self.hidden_dim),
            nn.SiLU(inplace=False),
        )
        self.router = nn.Linear(self.hidden_dim * 2, self.NUM_EXPERTS)
        self.experts = nn.ModuleList([self._make_expert(c2) for _ in range(self.NUM_EXPERTS)])
        self.output_projection = nn.Conv2d(c2, c2, 1, bias=False)

        # A persistent fallback makes device movement deterministic without
        # registering caller-owned prompt tensors in the checkpoint.
        self.register_buffer("zero_condition", torch.zeros(1, self.text_dim), persistent=True)
        self.last_aux_loss: torch.Tensor | None = None
        self.last_routing_snapshot: dict[str, Any] = {}
        self.last_routing_logits: torch.Tensor | None = None
        self._routing_duplicate_publication_count = 0

    @staticmethod
    def _make_expert(channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=False),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

    @property
    def num_experts(self) -> int:
        return self.NUM_EXPERTS

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def routing_mode(self) -> str:
        """Return the explicit dispatch mode (``hard`` by default)."""

        return self._routing_mode

    def set_routing_mode(self, mode: str) -> None:
        """Select frozen hard Top-1 or the bounded dense soft repair mode."""

        normalized = str(mode).strip().lower()
        if normalized not in {"hard", "soft"}:
            raise ValueError(f"routing mode must be 'hard' or 'soft', got {mode!r}")
        self._routing_mode = normalized

    def _condition_for_batch(
        self,
        condition: torch.Tensor | None,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Validate and detach a caller-owned ``[D]`` or ``[B,D]`` tensor."""

        parameter = next(self.condition_projection.parameters())
        if condition is None:
            return self.zero_condition.to(device=device, dtype=parameter.dtype).expand(batch, -1)
        if not isinstance(condition, torch.Tensor):
            raise TypeError(f"condition must be a Tensor or None, got {type(condition)!r}")
        if condition.ndim == 1:
            condition = condition.unsqueeze(0)
        if condition.ndim != 2 or condition.shape[-1] != self.text_dim:
            raise ValueError(f"condition must have shape [D] or [B, {self.text_dim}], got {tuple(condition.shape)}")
        if not torch.isfinite(condition).all():
            raise ValueError("condition must contain only finite values")
        if condition.shape[0] == 1:
            condition = condition.expand(batch, -1)
        elif condition.shape[0] != batch:
            raise ValueError(f"condition batch {condition.shape[0]} does not match feature batch {batch}")
        return condition.detach().to(device=device, dtype=parameter.dtype)

    @contextmanager
    def _measure_successful_expert_forwards(self):
        """Measure only successful expert returns for this forward scope.

        The hooks are deliberately scoped to one dispatch call.  PyTorch runs a
        forward hook only after the wrapped module returns, so an expert that
        raises never contributes to either the invocation or output-batch
        counters.  No hook or counter is retained on an expert after the
        context exits.
        """

        counts = {
            "module_invocations": 0,
            "output_batch_samples": 0,
            "module_invocations_by_expert": [0] * len(self.experts),
            "output_batch_samples_by_expert": [0] * len(self.experts),
        }

        def make_success_counter(expert_index: int):
            def count_successful_return(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                if not isinstance(output, torch.Tensor) or output.ndim == 0:
                    raise TypeError("text-router experts must return a batched Tensor")
                counts["module_invocations"] += 1
                counts["output_batch_samples"] += int(output.shape[0])
                counts["module_invocations_by_expert"][expert_index] += 1
                counts["output_batch_samples_by_expert"][expert_index] += int(output.shape[0])

            return count_successful_return

        handles = []
        try:
            for index, expert in enumerate(self.experts):
                handles.append(expert.register_forward_hook(make_success_counter(index)))
            yield counts
        finally:
            for handle in handles:
                handle.remove()

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        """Run image-level hard Top-1 dispatch while preserving detector connectivity."""

        if x.ndim != 4:
            raise ValueError(f"expected feature map [B,C,H,W], got shape {tuple(x.shape)}")
        features = self.input_projection(x)
        text = self._condition_for_batch(condition, features.shape[0], features.device)
        condition_features = self.condition_projection(text)
        visual_features = self.visual_projection(features)
        logits = self.router(torch.cat((visual_features, condition_features), dim=1)).float()
        probabilities = F.softmax(logits, dim=-1)
        assignments = probabilities.argmax(dim=-1)
        selected_probability = probabilities.gather(1, assignments.unsqueeze(1)).squeeze(1)

        routed = torch.zeros_like(features)
        with self._measure_successful_expert_forwards() as expert_forward_counts:
            if self.routing_mode == "soft":
                # The repair is deliberately dense: both experts see every
                # sample, and the differentiable probabilities mix their
                # outputs.  It is never selected implicitly.
                expert_outputs = [expert(features) for expert in self.experts]
                routed = sum(
                    output * probabilities[:, expert_index].to(output.dtype).view(-1, 1, 1, 1)
                    for expert_index, output in enumerate(expert_outputs)
                )
            else:
                # Compute only selected samples for each expert.
                # ``index_copy`` keeps the selected expert output connected to
                # the detector loss graph.
                for expert_index, expert in enumerate(self.experts):
                    sample_indices = (assignments == expert_index).nonzero(as_tuple=True)[0]
                    if sample_indices.numel() == 0:
                        continue
                    selected = expert(features.index_select(0, sample_indices))
                    routed = routed.index_copy(0, sample_indices, selected)
        actual_expert_module_invocations = int(expert_forward_counts["module_invocations"])
        actual_expert_forward_sample_calls = int(expert_forward_counts["output_batch_samples"])
        actual_expert_module_invocations_by_expert = list(expert_forward_counts["module_invocations_by_expert"])
        actual_expert_forward_sample_calls_by_expert = list(expert_forward_counts["output_batch_samples_by_expert"])
        if self.routing_mode == "hard":
            routed = routed * selected_probability.to(dtype=routed.dtype).view(-1, 1, 1, 1)
        output = self.output_projection(routed) + features

        usage = F.one_hot(assignments, num_classes=self.NUM_EXPERTS).float().mean(dim=0)
        importance = probabilities.float().mean(dim=0)
        balance = self.NUM_EXPERTS * torch.sum(importance * usage)
        z_loss = _MoTRouter.z_loss_from_logits(logits)
        raw_aux_loss = self.balance_loss_coeff * (balance + z_loss)
        self.last_aux_loss = raw_aux_loss if self.training else raw_aux_loss.detach().new_zeros(())
        if self.training:
            publish_aux_loss(
                self,
                self.last_aux_loss,
                step=current_aux_step(),
                kind="mot",
                training=True,
            )

        self.last_routing_logits = logits.detach()
        with torch.no_grad():
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
            top2 = probabilities.topk(k=min(2, self.NUM_EXPERTS), dim=-1).values
            margin = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else 0.0)
            # Opportunity is the detector-connected batch entering this
            # adapter.  Actual expert calls come only from the independent
            # successful-forward hooks above, never from assignments.
            opportunity_count = int(features.shape[0])
            actual_expert_calls = actual_expert_forward_sample_calls
            dense_reference_calls = opportunity_count * self.NUM_EXPERTS
            skipped_expert_calls = dense_reference_calls - actual_expert_calls
            self.last_routing_snapshot = {
                "num_experts": self.NUM_EXPERTS,
                "top_k": self.top_k,
                "routing_mode": self.routing_mode,
                "expert_usage": usage.detach(),
                "mean_router_probs": importance.detach(),
                "route_indices": assignments.detach(),
                "route": assignments.detach(),
                "executed_expert": assignments.detach().cpu().tolist(),
                "routing_probability_entropy": entropy.detach(),
                "mean_router_entropy": float(entropy.mean()),
                "top1_margin": margin.detach(),
                "mean_top1_margin": float(margin.mean()),
                "opportunity_count": opportunity_count,
                "nontrivial_action_count": skipped_expert_calls,
                "active_expert_count": actual_expert_module_invocations,
                "dense_reference_calls": dense_reference_calls,
                "actual_forward_calls": actual_expert_calls,
                "actual_forward_calls_by_expert": actual_expert_forward_sample_calls_by_expert,
                "actual_expert_module_invocations": actual_expert_module_invocations,
                "actual_expert_module_invocations_by_expert": actual_expert_module_invocations_by_expert,
                "actual_expert_calls": actual_expert_calls,
                "actual_expert_calls_by_expert": actual_expert_forward_sample_calls_by_expert,
                "skipped_expert_calls": skipped_expert_calls,
                "skipped_expert_count": skipped_expert_calls,
                "aux_loss": float(self.last_aux_loss.detach()),
                "dispatch": {
                    "policy": "sample_top1_sparse" if self.routing_mode == "hard" else "dense_soft",
                    "selected_samples": actual_expert_calls,
                    "opportunity_count": opportunity_count,
                    "active_experts": actual_expert_module_invocations,
                    "skipped_experts": self.NUM_EXPERTS - actual_expert_module_invocations,
                    "dense_reference_calls": dense_reference_calls,
                    "actual_forward_calls": actual_expert_calls,
                    "actual_forward_calls_by_expert": actual_expert_forward_sample_calls_by_expert,
                    "actual_expert_module_invocations": actual_expert_module_invocations,
                    "actual_expert_module_invocations_by_expert": actual_expert_module_invocations_by_expert,
                    "actual_expert_calls": actual_expert_calls,
                    "actual_expert_calls_by_expert": actual_expert_forward_sample_calls_by_expert,
                    "skipped_expert_calls": skipped_expert_calls,
                    "nontrivial_action_count": skipped_expert_calls,
                    "dense_top1_action_count": 0,
                },
            }
        return output

    @property
    def aux_loss(self) -> torch.Tensor:
        """Return the current graph-connected routing auxiliary scalar."""

        if self.last_aux_loss is not None:
            return self.last_aux_loss
        return self.zero_condition.new_zeros(())

    def publish_aux_loss(self, *, step: int, training: bool) -> torch.Tensor:
        if not training:
            return self.aux_loss.detach().new_zeros(())
        return publish_aux_loss(self, self.aux_loss, step=step, kind="mot", training=training)

    def routing_snapshot(self) -> dict[str, Any]:
        return _routing_snapshot(self)

    def export_capabilities(self) -> dict[str, Any]:
        capabilities = _export_routing_capabilities(self)
        capabilities.update(
            routing_kind="mot",
            sparse_dispatch=True,
            eager_sparse_dispatch=True,
            training_sparse_dispatch=True,
            sparse_train=True,
            dispatch_policy="sample_top1_sparse",
            sparse_export_limitation="Data-dependent sample top-1 dispatch is eager-only.",
        )
        return capabilities


__all__ = ("TextConditionedMoT",)
