"""PyTorch bridge for inserting split-ORT dynamic blocks into a complete eager model."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .dispatch import DynamicDispatchContractError, audit_sparse_routing
from .ort import ORTDynamicExpertRuntime


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
            eager_weights, _ = self.block.router(x)
        eager = eager_weights.detach().float().cpu().numpy()
        zero_tolerance = float(self.runtime.manifest["zero_tolerance"])
        route_mask_mismatch = (eager > zero_tolerance) != (ort_weights > zero_tolerance)
        route_location_mismatch = route_mask_mismatch.any(axis=1)
        sparse_weight_error = float(np.max(np.abs(eager - ort_weights)))
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
        return summary


__all__ = ("ORTDynamicBlockAdapter", "ORTRouterTorchExpertAdapter")
