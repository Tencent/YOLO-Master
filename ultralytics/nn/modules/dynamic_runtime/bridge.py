"""PyTorch bridge for inserting split-ORT dynamic blocks into a complete eager model."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
from torch import nn

from ultralytics.nn.modules.topk_contract import (
    DEADBAND_LOWEST_ID_TOPK,
    LEGACY_PRIORITY_BIAS_TOPK,
)

from .dispatch import DynamicDispatchContractError, audit_sparse_routing
from .ort import ORTDynamicExpertRuntime


_ROUTE_MARGIN_THRESHOLDS = (1e-7, 1e-6, 1e-5, 1e-4)
_MAX_MISMATCH_MARGIN_SAMPLES = 4096
_MAX_ROUTE_MISMATCH_EXAMPLES = 32


def _host_topk_boundary_margin(
    dense_probabilities: np.ndarray,
    *,
    top_k: int,
    tie_tolerance: float,
    tie_break: str,
) -> np.ndarray:
    """Return the selected/unselected boundary margin used by host Top-K.

    Legacy bundles rank an expert-id-adjusted score, while new bundles use a
    raw-probability deadband followed by the lowest expert id. For the latter,
    the raw K/K+1 gap is reported; the policy and tolerance are recorded next
    to it so the value is not misrepresented as a standalone decision rule.
    """
    probabilities = np.asarray(dense_probabilities)
    if probabilities.ndim != 4:
        raise DynamicDispatchContractError(
            f"dense router probabilities must be [B,E,H,W], got {tuple(probabilities.shape)}"
        )
    num_experts = int(probabilities.shape[1])
    if not 1 <= top_k < num_experts:
        raise DynamicDispatchContractError(
            f"route-margin audit requires Top-K in [1, {num_experts - 1}], got {top_k}"
        )
    if tie_break == LEGACY_PRIORITY_BIAS_TOPK:
        priority = np.arange(num_experts, dtype=probabilities.dtype).reshape(1, -1, 1, 1)
        ranking_scores = probabilities - priority * float(tie_tolerance)
    elif tie_break == DEADBAND_LOWEST_ID_TOPK:
        ranking_scores = probabilities
    else:
        raise DynamicDispatchContractError(f"unsupported route-margin Top-K policy: {tie_break!r}")
    ordered = np.sort(ranking_scores, axis=1)[:, ::-1]
    return ordered[:, top_k - 1] - ordered[:, top_k]


class ORTDynamicBlockAdapter(nn.Module):
    """Replace one eager routed block with a split-ORT conditional runtime.

    This bridge is an integration PoC: the outer YOLO graph remains PyTorch and
    tensors cross the PyTorch/NumPy boundary on CPU. It is not a TensorRT or
    zero-copy deployment implementation.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        providers: Sequence[str] | None = None,
        require_reduction: bool = False,
    ):
        super().__init__()
        self.runtime = ORTDynamicExpertRuntime(manifest_path, providers=providers)
        self.family = str(self.runtime.manifest["family"])
        self.top_k = int(self.runtime.manifest["top_k"])
        self.num_experts = int(self.runtime.manifest["num_experts"])
        self.require_reduction = bool(require_reduction)
        self.router = SimpleNamespace(scene_inference_mode="dynamic")
        self.last_audit = None
        self.last_aux_loss = torch.zeros(())
        self.last_routing_snapshot: dict = {}

    def forward(self, x: torch.Tensor):
        if self.training:
            raise RuntimeError("ORTDynamicBlockAdapter is inference-only; call model.eval()")
        if x.requires_grad:
            raise RuntimeError("ORTDynamicBlockAdapter does not support autograd")
        input_array = x.detach().cpu().numpy()
        output_array, audit = self.runtime.run(input_array, require_reduction=self.require_reduction)
        output = torch.from_numpy(output_array).to(device=x.device, dtype=x.dtype)
        self.last_audit = audit
        self.last_aux_loss = x.new_zeros(())

        routing_weights = self.runtime.last_routing_weights
        if routing_weights is None:
            expert_usage = x.new_zeros(self.num_experts)
        else:
            active = routing_weights > float(self.runtime.manifest["zero_tolerance"])
            expert_usage = torch.from_numpy(active.mean(axis=(0, 2, 3)).astype(np.float32)).to(x.device)
        self.last_routing_snapshot = {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "expert_usage": expert_usage,
            "mean_router_probs": expert_usage,
            "aux_loss": 0.0,
            "finite_diagnostics": {},
            "dispatch": audit.to_dict(),
            "scene_inference_mode": "dynamic",
            "scene_aware_applied": False,
            "scene_bypass_reason": "split_ort_bridge",
        }
        if self.family == "MoT":
            return output, self.last_aux_loss
        return output

    @property
    def loaded_expert_ids(self) -> tuple[int, ...]:
        """Return expert sessions actually loaded by this adapter."""
        return self.runtime.loaded_expert_ids

    @property
    def aux_loss(self) -> torch.Tensor:
        """Return a zero auxiliary loss because the bridge is inference-only."""
        return self.last_aux_loss

    def routing_snapshot(self) -> dict:
        """Return the latest runtime-backed routing snapshot."""
        return self.last_routing_snapshot


class ORTRouterTorchExpertAdapter(nn.Module):
    """Run an exported ORT router and conditionally execute checkpoint MoT experts in PyTorch.

    This adapter exists for full-model accuracy validation on a CUDA host. The
    router crosses the CPU/NumPy boundary, but the original checkpoint experts,
    projection, normalization, and residual remain on the PyTorch device. It is
    therefore evidence for dynamic routing and full-model mAP, not a latency or
    TensorRT deployment implementation.
    """

    def __init__(
        self,
        block: nn.Module,
        manifest_path: str | Path,
        *,
        providers: Sequence[str] | None = None,
        require_reduction: bool = False,
        compare_eager_routes: bool = True,
    ):
        super().__init__()
        from ultralytics.nn.modules.mot.block import MoTBlock

        if not isinstance(block, MoTBlock):
            raise TypeError(f"ORTRouterTorchExpertAdapter requires MoTBlock, got {type(block).__name__}")
        if block.training:
            raise DynamicDispatchContractError("checkpoint MoTBlock must be in eval mode before adapter creation")

        self.block = block
        self.runtime = ORTDynamicExpertRuntime(manifest_path, providers=providers)
        manifest = self.runtime.manifest
        if manifest["family"] != "MoT":
            raise DynamicDispatchContractError("router-to-PyTorch validation adapter currently supports MoT only")

        self.top_k = int(manifest["top_k"])
        self.num_experts = int(manifest["num_experts"])
        if self.top_k != int(block.top_k) or self.num_experts != int(block.NUM_EXPERTS):
            raise DynamicDispatchContractError(
                "bundle routing contract does not match the checkpoint block: "
                f"bundle Top-{self.top_k}/{self.num_experts}, "
                f"checkpoint Top-{block.top_k}/{block.NUM_EXPERTS}"
            )
        # Bind the eager reference to the versioned bundle contract. This is
        # essential when replaying a legacy manifest with a checkpoint whose
        # serialized router predates the explicit policy attribute.
        block.router.route_tie_break = str(manifest.get("host_topk_tie_break", LEGACY_PRIORITY_BIAS_TOPK))
        block.router.route_tie_tolerance = float(manifest.get("host_topk_tie_tolerance", 0.0))
        self.require_reduction = bool(require_reduction)
        self.compare_eager_routes = bool(compare_eager_routes)
        self.last_audit = None
        self.last_aux_loss = torch.zeros(())
        self.last_routing_snapshot: dict = {}
        self.last_route_drift: dict = {}
        self.reset_execution_summary()

    def reset_execution_summary(self) -> None:
        """Clear aggregate evidence, for example after framework warmup calls."""
        self.last_audit = None
        self.last_routing_snapshot = {}
        self.last_route_drift = {}
        self._total_calls = 0
        self._total_samples = 0
        self._total_selected_pairs = 0
        self._total_dense_pairs = 0
        self._total_expert_calls = 0
        self._total_dense_expert_calls = 0
        self._total_route_locations = 0
        self._total_route_location_mismatches = 0
        self._total_route_mask_entries = 0
        self._total_route_mask_mismatches = 0
        self._max_sparse_weight_abs_error = 0.0
        self._max_dense_probability_abs_error = 0.0
        self._total_route_margin_locations = 0
        self._route_margin_sum = 0.0
        self._route_margin_min = float("inf")
        self._route_margin_max = 0.0
        self._route_margin_below_counts = {threshold: 0 for threshold in _ROUTE_MARGIN_THRESHOLDS}
        self._mismatch_route_margins: list[float] = []
        self._mismatch_eager_route_margins: list[float] = []
        self._mismatch_route_margin_total = 0
        self._route_mismatch_examples: list[dict] = []

    @property
    def router(self) -> nn.Module:
        """Expose the original router protocol expected by ``C2fMoT`` diagnostics."""
        return self.block.router

    @property
    def loaded_onnx_expert_ids(self) -> tuple[int, ...]:
        """Return ONNX expert sessions loaded by this adapter (expected to remain empty)."""
        return self.runtime.loaded_expert_ids

    def _compare_routes(self, x: torch.Tensor, ort_weights: np.ndarray) -> dict:
        if not self.compare_eager_routes:
            return {"enabled": False}
        with torch.no_grad():
            eager_weights, _, eager_logits = self.block.router(x, return_logits=True)
            eager_dense_weights = torch.softmax(eager_logits / self.block.router.temperature.float(), dim=1)
        eager = eager_weights.detach().float().cpu().numpy()
        eager_dense = eager_dense_weights.detach().float().cpu().numpy()
        zero_tolerance = float(self.runtime.manifest["zero_tolerance"])
        route_mask_mismatch = (eager > zero_tolerance) != (ort_weights > zero_tolerance)
        route_location_mismatch = route_mask_mismatch.any(axis=1)
        sparse_weight_error = float(np.max(np.abs(eager - ort_weights)))
        dense_probabilities = self.runtime.last_dense_routing_probabilities
        result = {
            "enabled": True,
            "route_mask_mismatch_count": int(route_mask_mismatch.sum()),
            "route_mask_total": int(route_mask_mismatch.size),
            "route_mask_mismatch_ratio": float(route_mask_mismatch.mean()),
            "route_location_mismatch_count": int(route_location_mismatch.sum()),
            "route_location_total": int(route_location_mismatch.size),
            "route_location_mismatch_ratio": float(route_location_mismatch.mean()),
            "sparse_weight_max_abs_error": sparse_weight_error,
        }
        if dense_probabilities is not None and self.top_k < self.num_experts:
            tie_tolerance = float(self.runtime.manifest.get("host_topk_tie_tolerance", 0.0))
            tie_break = str(
                self.runtime.manifest.get(
                    "host_topk_tie_break",
                    LEGACY_PRIORITY_BIAS_TOPK,
                )
            )
            margins = _host_topk_boundary_margin(
                dense_probabilities,
                top_k=self.top_k,
                tie_tolerance=tie_tolerance,
                tie_break=tie_break,
            )
            eager_margins = _host_topk_boundary_margin(
                eager_dense,
                top_k=self.top_k,
                tie_tolerance=tie_tolerance,
                tie_break=tie_break,
            )
            margin_values = margins.reshape(-1).astype(np.float64, copy=False)
            mismatch_margins = margins[route_location_mismatch].reshape(-1).astype(np.float64, copy=False)
            mismatch_eager_margins = eager_margins[route_location_mismatch].reshape(-1).astype(
                np.float64,
                copy=False,
            )
            dense_probability_error = np.abs(eager_dense - dense_probabilities)
            dense_probability_max_abs_error = float(dense_probability_error.max())
            threshold_counts = {
                f"{threshold:.0e}": int(np.count_nonzero(margin_values <= threshold))
                for threshold in _ROUTE_MARGIN_THRESHOLDS
            }
            result["host_topk_boundary_margin"] = {
                "available": True,
                "definition": (
                    "kth_minus_k_plus_1_adjusted_ranking_score"
                    if tie_break == LEGACY_PRIORITY_BIAS_TOPK
                    else "kth_minus_k_plus_1_raw_probability"
                ),
                "tie_break": tie_break,
                "tie_tolerance": tie_tolerance,
                "locations": int(margin_values.size),
                "min": float(margin_values.min()),
                "mean": float(margin_values.mean()),
                "max": float(margin_values.max()),
                "below_or_equal_counts": threshold_counts,
                "mismatch_locations": int(mismatch_margins.size),
                "ort_mismatch_min": float(mismatch_margins.min()) if mismatch_margins.size else None,
                "ort_mismatch_median": float(np.median(mismatch_margins)) if mismatch_margins.size else None,
                "ort_mismatch_max": float(mismatch_margins.max()) if mismatch_margins.size else None,
                "eager_mismatch_min": (
                    float(mismatch_eager_margins.min()) if mismatch_eager_margins.size else None
                ),
                "eager_mismatch_median": (
                    float(np.median(mismatch_eager_margins)) if mismatch_eager_margins.size else None
                ),
                "eager_mismatch_max": (
                    float(mismatch_eager_margins.max()) if mismatch_eager_margins.size else None
                ),
                "dense_probability_max_abs_error": dense_probability_max_abs_error,
            }
            self._total_route_margin_locations += int(margin_values.size)
            self._route_margin_sum += float(margin_values.sum())
            self._route_margin_min = min(self._route_margin_min, float(margin_values.min()))
            self._route_margin_max = max(self._route_margin_max, float(margin_values.max()))
            for threshold in _ROUTE_MARGIN_THRESHOLDS:
                self._route_margin_below_counts[threshold] += int(np.count_nonzero(margin_values <= threshold))
            self._mismatch_route_margin_total += int(mismatch_margins.size)
            remaining = _MAX_MISMATCH_MARGIN_SAMPLES - len(self._mismatch_route_margins)
            if remaining > 0 and mismatch_margins.size:
                self._mismatch_route_margins.extend(mismatch_margins[:remaining].tolist())
                self._mismatch_eager_route_margins.extend(mismatch_eager_margins[:remaining].tolist())
            self._max_dense_probability_abs_error = max(
                self._max_dense_probability_abs_error,
                dense_probability_max_abs_error,
            )
            example_slots = _MAX_ROUTE_MISMATCH_EXAMPLES - len(self._route_mismatch_examples)
            if example_slots > 0 and mismatch_margins.size:
                mismatch_coordinates = np.argwhere(route_location_mismatch)
                dense_error_by_location = dense_probability_error.max(axis=1)
                for coordinate in mismatch_coordinates[:example_slots]:
                    index = tuple(int(value) for value in coordinate)
                    self._route_mismatch_examples.append(
                        {
                            "call_index": self._total_calls,
                            "location_index": list(index),
                            "ort_host_topk_boundary_margin": float(margins[index]),
                            "eager_host_topk_boundary_margin": float(eager_margins[index]),
                            "dense_probability_max_abs_error": float(dense_error_by_location[index]),
                        }
                    )
        else:
            result["host_topk_boundary_margin"] = {
                "available": False,
                "reason": (
                    "bundle_router_output_is_already_sparse"
                    if dense_probabilities is None
                    else "top_k_selects_all_experts"
                ),
            }
        self._total_route_locations += result["route_location_total"]
        self._total_route_location_mismatches += result["route_location_mismatch_count"]
        self._total_route_mask_entries += result["route_mask_total"]
        self._total_route_mask_mismatches += result["route_mask_mismatch_count"]
        self._max_sparse_weight_abs_error = max(self._max_sparse_weight_abs_error, sparse_weight_error)
        return result

    def forward(self, x: torch.Tensor):
        if self.training:
            raise RuntimeError("ORTRouterTorchExpertAdapter is inference-only; call model.eval()")
        if x.requires_grad:
            raise RuntimeError("ORTRouterTorchExpertAdapter does not support autograd")

        # Router ONNX artifacts are exported in FP32. Experts stay on x.device.
        input_array = x.detach().float().cpu().numpy()
        routing_weights = self.runtime.route(input_array)
        audit = audit_sparse_routing(
            routing_weights,
            batch_size=int(x.shape[0]),
            num_experts=self.num_experts,
            top_k=self.top_k,
            routing_granularity=self.runtime.manifest["routing_granularity"],
            dynamic_threshold=float(self.runtime.manifest["dynamic_threshold"]),
            zero_tolerance=float(self.runtime.manifest["zero_tolerance"]),
            require_reduction=self.require_reduction,
        )
        route_drift = self._compare_routes(x, routing_weights)

        weights = torch.from_numpy(routing_weights).to(device=x.device, dtype=x.dtype)
        index_array = np.argsort(-routing_weights, axis=1, kind="stable")[:, : self.top_k].copy()
        indices = torch.from_numpy(index_array).to(device=x.device, dtype=torch.long)
        mixture = self.block._blend_experts(x, weights, indices)
        observed_calls = int(self.block._last_dispatch_stats.get("actual_expert_calls", -1))
        if observed_calls != audit.executed_expert_calls:
            raise DynamicDispatchContractError(
                f"PyTorch executed {observed_calls} experts but routing audit requires "
                f"{audit.executed_expert_calls}"
            )
        output = self.block.out_norm(self.block.out_proj(mixture)) + x

        self.last_audit = audit
        self.last_route_drift = route_drift
        self.last_aux_loss = x.new_zeros(())
        mean_weights = weights.detach().float().mean(dim=(0, 2, 3))
        self.last_routing_snapshot = {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "expert_usage": mean_weights,
            "mean_router_probs": mean_weights,
            "aux_loss": 0.0,
            "finite_diagnostics": {"routing_weights_finite": bool(torch.isfinite(weights).all())},
            "dispatch": audit.to_dict(),
            "route_drift": route_drift,
            "scene_inference_mode": self.block.router.scene_inference_mode,
            "scene_aware_applied": bool(self.runtime.manifest.get("scene_aware", False)),
            "scene_bypass_reason": "onnx_router_host_execution",
        }

        self._total_calls += 1
        self._total_samples += audit.batch_size
        self._total_selected_pairs += audit.selected_sample_expert_pairs
        self._total_dense_pairs += audit.dense_sample_expert_pairs
        self._total_expert_calls += audit.executed_expert_calls
        self._total_dense_expert_calls += audit.dense_expert_calls
        return output, self.last_aux_loss

    @property
    def aux_loss(self) -> torch.Tensor:
        """Return zero because this adapter is validation-only."""
        return self.last_aux_loss

    def routing_snapshot(self) -> dict:
        """Return the latest ORT-router/PyTorch-expert snapshot."""
        return self.last_routing_snapshot

    def execution_summary(self) -> dict:
        """Return aggregate conditional-execution and route-drift evidence."""
        sample_reduction = (
            1.0 - self._total_selected_pairs / self._total_dense_pairs if self._total_dense_pairs else 0.0
        )
        union_reduction = (
            1.0 - self._total_expert_calls / self._total_dense_expert_calls
            if self._total_dense_expert_calls
            else 0.0
        )
        summary = {
            "calls": self._total_calls,
            "samples": self._total_samples,
            "selected_sample_expert_pairs": self._total_selected_pairs,
            "dense_sample_expert_pairs": self._total_dense_pairs,
            "sample_pair_reduction_ratio": sample_reduction,
            "executed_expert_calls": self._total_expert_calls,
            "dense_expert_calls": self._total_dense_expert_calls,
            "batch_union_reduction_ratio": union_reduction,
            "onnx_expert_sessions_loaded": list(self.loaded_onnx_expert_ids),
            "checkpoint_pytorch_experts_executed": self._total_expert_calls > 0,
            "route_drift_audit_enabled": self.compare_eager_routes,
        }
        if self.compare_eager_routes:
            summary.update(
                route_location_mismatch_count=self._total_route_location_mismatches,
                route_location_total=self._total_route_locations,
                route_location_mismatch_ratio=(
                    self._total_route_location_mismatches / self._total_route_locations
                    if self._total_route_locations
                    else 0.0
                ),
                route_mask_mismatch_count=self._total_route_mask_mismatches,
                route_mask_total=self._total_route_mask_entries,
                route_mask_mismatch_ratio=(
                    self._total_route_mask_mismatches / self._total_route_mask_entries
                    if self._total_route_mask_entries
                    else 0.0
                ),
                sparse_weight_max_abs_error=self._max_sparse_weight_abs_error,
            )
            if self._total_route_margin_locations:
                tie_break = str(
                    self.runtime.manifest.get(
                        "host_topk_tie_break",
                        LEGACY_PRIORITY_BIAS_TOPK,
                    )
                )
                sampled_mismatch_margins = np.asarray(self._mismatch_route_margins, dtype=np.float64)
                sampled_eager_mismatch_margins = np.asarray(
                    self._mismatch_eager_route_margins,
                    dtype=np.float64,
                )
                summary["route_margin_audit"] = {
                    "available": True,
                    "definition": (
                        "kth_minus_k_plus_1_adjusted_ranking_score"
                        if tie_break == LEGACY_PRIORITY_BIAS_TOPK
                        else "kth_minus_k_plus_1_raw_probability"
                    ),
                    "tie_break": tie_break,
                    "tie_tolerance": float(self.runtime.manifest.get("host_topk_tie_tolerance", 0.0)),
                    "locations": self._total_route_margin_locations,
                    "min": self._route_margin_min,
                    "mean": self._route_margin_sum / self._total_route_margin_locations,
                    "max": self._route_margin_max,
                    "below_or_equal_counts": {
                        f"{threshold:.0e}": self._route_margin_below_counts[threshold]
                        for threshold in _ROUTE_MARGIN_THRESHOLDS
                    },
                    "mismatch_locations": self._mismatch_route_margin_total,
                    "mismatch_margin_sample_count": int(sampled_mismatch_margins.size),
                    "mismatch_margin_sample_complete": (
                        int(sampled_mismatch_margins.size) == self._mismatch_route_margin_total
                    ),
                    "ort_mismatch_min": (
                        float(sampled_mismatch_margins.min()) if sampled_mismatch_margins.size else None
                    ),
                    "ort_mismatch_median": (
                        float(np.median(sampled_mismatch_margins)) if sampled_mismatch_margins.size else None
                    ),
                    "ort_mismatch_p95": (
                        float(np.percentile(sampled_mismatch_margins, 95))
                        if sampled_mismatch_margins.size
                        else None
                    ),
                    "ort_mismatch_max": (
                        float(sampled_mismatch_margins.max()) if sampled_mismatch_margins.size else None
                    ),
                    "eager_mismatch_min": (
                        float(sampled_eager_mismatch_margins.min())
                        if sampled_eager_mismatch_margins.size
                        else None
                    ),
                    "eager_mismatch_median": (
                        float(np.median(sampled_eager_mismatch_margins))
                        if sampled_eager_mismatch_margins.size
                        else None
                    ),
                    "eager_mismatch_p95": (
                        float(np.percentile(sampled_eager_mismatch_margins, 95))
                        if sampled_eager_mismatch_margins.size
                        else None
                    ),
                    "eager_mismatch_max": (
                        float(sampled_eager_mismatch_margins.max())
                        if sampled_eager_mismatch_margins.size
                        else None
                    ),
                    "dense_probability_max_abs_error": self._max_dense_probability_abs_error,
                    "mismatch_examples": self._route_mismatch_examples,
                }
            else:
                summary["route_margin_audit"] = {
                    "available": False,
                    "reason": (
                        "bundle_router_output_is_already_sparse"
                        if self.runtime.last_dense_routing_probabilities is None
                        else "top_k_selects_all_experts"
                    ),
                }
        return summary


__all__ = ("ORTDynamicBlockAdapter", "ORTRouterTorchExpertAdapter")
