"""Cross-family routing interpretation for MoE, MoA, MoT, and MoLoRA."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ultralytics.nn.modules.moe.protocol import is_routed_module


@dataclass(frozen=True)
class RoutingLayerSummary:
    """Normalized routing state from one routed layer."""

    layer_name: str
    module_type: str
    num_experts: int
    top_k: int
    expert_usage: tuple[float, ...]
    mean_router_probs: tuple[float, ...] | None
    aux_loss: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoutingCollapseReport:
    """Collapse indicators for one routed layer."""

    layer_name: str
    expert_usage: tuple[float, ...]
    dominant_expert: int
    dominant_share: float
    normalized_gini: float
    normalized_entropy: float
    dead_experts: tuple[int, ...]
    collapsed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExpertSpecializationReport:
    """Dataset-level expert usage and input signatures for one routed layer."""

    layer_name: str
    module_type: str
    num_experts: int
    num_samples: int
    mean_usage: tuple[float, ...]
    dominant_samples: tuple[int, ...]
    feature_signatures: tuple[dict[str, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoutingHeatmap:
    """Raw routing probabilities and the corresponding top-1 assignment map."""

    layer_name: str
    module_type: str
    probabilities: torch.Tensor
    assignments: torch.Tensor

    def to_dict(self) -> dict[str, Any]:
        reduce_dims = tuple(dim for dim in range(self.probabilities.ndim) if dim != 1)
        mean_usage = self.probabilities.mean(dim=reduce_dims) if reduce_dims else self.probabilities
        visualization_type = "global_distribution" if self.probabilities.ndim == 2 else "spatial_heatmap"
        return {
            "layer_name": self.layer_name,
            "module_type": self.module_type,
            "probability_shape": list(self.probabilities.shape),
            "assignment_shape": list(self.assignments.shape),
            "mean_usage": [float(value) for value in mean_usage.reshape(-1).tolist()],
            "visualization_type": visualization_type,
        }


@dataclass(frozen=True)
class RoutingCausalReport:
    """Output change produced by forcing a routed layer to one expert."""

    layer_name: str
    expert_idx: int
    tensor_count: int
    element_count: int
    mean_absolute_difference: float
    root_mean_square_difference: float
    max_absolute_difference: float
    cosine_similarity: float
    natural_l2_norm: float
    forced_l2_norm: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RoutingInterpreter:
    """Interpret routed models without changing their persistent runtime state."""

    def __init__(self, model: nn.Module) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model)!r}")
        self.model = model

    def collect_layer_summaries(
        self,
        *,
        include_wrappers: bool = False,
        heatmaps: Mapping[str, RoutingHeatmap] | None = None,
    ) -> list[RoutingLayerSummary]:
        """Collect normalized snapshots from routed layers after a forward pass."""
        summaries: list[RoutingLayerSummary] = []
        if heatmaps is not None:
            all_modules = self._routed_modules(leaf_only=False)
            modules = {name: all_modules[name] for name in heatmaps if name in all_modules}
        else:
            modules = self._routed_modules(leaf_only=not include_wrappers)
        for name, module in modules.items():
            snapshot = getattr(module, "last_routing_snapshot", None)
            snapshot = snapshot if isinstance(snapshot, dict) else {}
            heatmap = heatmaps.get(name) if heatmaps is not None else None
            num_experts = int(getattr(module, "num_experts", 0) or snapshot.get("num_experts", 0))
            if heatmap is not None:
                reduce_dims = tuple(dim for dim in range(heatmap.probabilities.ndim) if dim != 1)
                usage_source = heatmap.probabilities.mean(dim=reduce_dims)
                mean_probs_source = usage_source
            else:
                usage_source = snapshot.get("expert_usage", snapshot.get("mean_router_probs"))
                mean_probs_source = snapshot.get("mean_router_probs", usage_source)
            usage = self._normalized_vector(usage_source, num_experts)
            if usage is None:
                continue
            mean_probs = self._normalized_vector(mean_probs_source, num_experts)
            summaries.append(
                RoutingLayerSummary(
                    layer_name=name,
                    module_type=type(module).__name__,
                    num_experts=num_experts,
                    top_k=int(snapshot.get("top_k", getattr(module, "top_k", 0))),
                    expert_usage=tuple(float(value) for value in usage.tolist()),
                    mean_router_probs=(
                        tuple(float(value) for value in mean_probs.tolist()) if mean_probs is not None else None
                    ),
                    aux_loss=self._scalar_float(snapshot.get("aux_loss", 0.0)),
                )
            )
        return summaries

    def detect_routing_collapse(
        self,
        *,
        dominant_threshold: float = 0.8,
        gini_threshold: float = 0.8,
        entropy_threshold: float = 0.5,
        dead_threshold: float = 0.01,
        heatmaps: Mapping[str, RoutingHeatmap] | None = None,
    ) -> dict[str, RoutingCollapseReport]:
        """Detect dominant, unequal, low-entropy, and dead-expert routing."""
        reports: dict[str, RoutingCollapseReport] = {}
        for summary in self.collect_layer_summaries(heatmaps=heatmaps):
            usage = torch.tensor(summary.expert_usage, dtype=torch.float64)
            dominant_share, dominant_expert = torch.max(usage, dim=0)
            normalized_gini = self._normalized_gini(usage)
            normalized_entropy = self._normalized_entropy(usage)
            dead_experts = tuple(index for index, value in enumerate(usage.tolist()) if value <= dead_threshold)
            collapsed = bool(
                float(dominant_share) >= dominant_threshold
                or normalized_gini >= gini_threshold
                or normalized_entropy <= entropy_threshold
                or dead_experts
            )
            reports[summary.layer_name] = RoutingCollapseReport(
                layer_name=summary.layer_name,
                expert_usage=summary.expert_usage,
                dominant_expert=int(dominant_expert),
                dominant_share=float(dominant_share),
                normalized_gini=normalized_gini,
                normalized_entropy=normalized_entropy,
                dead_experts=dead_experts,
                collapsed=collapsed,
            )
        return reports

    def capture_routing(
        self,
        batch: Any,
        *,
        layer_name: str | None = None,
        forward_fn: Callable[[nn.Module, Any], Any] | None = None,
    ) -> dict[str, RoutingHeatmap]:
        """Run one forward and capture router probabilities before expert dispatch."""
        modules = self._routed_modules(layer_name=layer_name, leaf_only=layer_name is None)
        if not modules:
            target = f" named {layer_name!r}" if layer_name else ""
            raise ValueError(f"no routed modules{target} were found")

        captured: dict[str, RoutingHeatmap] = {}
        handles = []
        for name, module in modules.items():
            router = self._router_for(module)
            if router is None:
                continue

            def capture_hook(_router, _inputs, output, *, current_name=name, current_module=module):
                probabilities = self._router_probabilities(output, int(getattr(current_module, "num_experts", 0)))
                if probabilities is None:
                    return None
                probabilities = probabilities.detach().float().cpu()
                captured[current_name] = RoutingHeatmap(
                    layer_name=current_name,
                    module_type=type(current_module).__name__,
                    probabilities=probabilities,
                    assignments=probabilities.argmax(dim=1),
                )
                return None

            handles.append(router.register_forward_hook(capture_hook))

        if not handles:
            raise ValueError("routed modules were found, but none expose a supported router or routing submodule")

        training_flags = self._training_flags()
        try:
            self.model.eval()
            with torch.no_grad():
                self._forward(batch, forward_fn)
        finally:
            for handle in handles:
                handle.remove()
            self._restore_training_flags(training_flags)

        if layer_name is not None and layer_name not in captured:
            raise RuntimeError(f"router for layer {layer_name!r} did not produce a supported probability tensor")
        return captured

    def visualize_routing(
        self,
        batch: Any,
        *,
        layer_name: str | None = None,
        output_dir: str | Path | None = None,
        input_image: str | Path | torch.Tensor | None = None,
        forward_fn: Callable[[nn.Module, Any], Any] | None = None,
    ) -> dict[str, RoutingHeatmap]:
        """Capture routing and optionally render probability panels to PNG files."""
        heatmaps = self.capture_routing(batch, layer_name=layer_name, forward_fn=forward_fn)
        if output_dir is not None:
            self.save_routing_visualizations(heatmaps, output_dir, input_image=input_image)
        return heatmaps

    def save_routing_visualizations(
        self,
        heatmaps: Mapping[str, RoutingHeatmap],
        output_dir: str | Path,
        *,
        input_image: str | Path | torch.Tensor | None = None,
    ) -> dict[str, dict[str, Path]]:
        """Write spatial heatmap overlays or global routing distributions."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        background = self._load_image(input_image)
        written: dict[str, dict[str, Path]] = {}

        for name, heatmap in heatmaps.items():
            probabilities = heatmap.probabilities
            safe_name = "root" if name == "<root>" else re.sub(r"[^A-Za-z0-9._-]+", "_", name).replace(".", "_") or "root"

            if probabilities.ndim == 2:
                values = probabilities[0]
                figure, axis = plt.subplots(figsize=(max(5.0, values.numel() * 0.8), 3.5))
                axis.bar(range(values.numel()), values.tolist(), color="#3498db")
                axis.set_xticks(range(values.numel()), [f"E{i}" for i in range(values.numel())])
                axis.set_ylim(0.0, 1.0)
                axis.set_ylabel("routing probability")
                axis.set_title(f"{name} ({heatmap.module_type}) - global distribution")
                axis.grid(axis="y", alpha=0.25)
                path = output_path / f"{safe_name}_routing_distribution.png"
                figure.tight_layout()
                figure.savefig(path, dpi=160, bbox_inches="tight")
                plt.close(figure)
                written[name] = {"distribution": path}
                continue

            spatial = probabilities[0]
            while spatial.ndim > 3:
                spatial = spatial.mean(dim=-1)
            if spatial.ndim != 3:
                continue
            confidence, assignments = spatial.max(dim=0)
            layer_written: dict[str, Path] = {}
            confidence_min, confidence_max = self._display_range(confidence)
            for artifact_name, values, categorical, cmap, vmin, vmax in (
                ("confidence_heatmap", confidence, False, "magma", confidence_min, confidence_max),
                ("assignment_map", assignments.float(), True, "tab20", None, None),
            ):
                path = output_path / f"{safe_name}_{artifact_name}.png"
                self._save_overlay(
                    values,
                    background,
                    path,
                    title=f"{name} - {artifact_name.replace('_', ' ')}",
                    categorical=categorical,
                    cmap=cmap,
                    vmin=vmin if vmin is not None else self._display_range(values)[0],
                    vmax=vmax if vmax is not None else self._display_range(values)[1],
                )
                layer_written[artifact_name] = path
            for expert_idx in range(spatial.shape[0]):
                artifact_name = f"expert_{expert_idx}_heatmap"
                path = output_path / f"{safe_name}_{artifact_name}.png"
                self._save_overlay(
                    spatial[expert_idx],
                    background,
                    path,
                    title=f"{name} - expert {expert_idx} activation",
                    categorical=False,
                    cmap="inferno",
                    vmin=self._display_range(spatial[expert_idx])[0],
                    vmax=self._display_range(spatial[expert_idx])[1],
                )
                layer_written[artifact_name] = path

            dashboard_path = output_path / f"{safe_name}_routing_dashboard.png"
            columns = min(spatial.shape[0], 4) + 2
            rows = math.ceil((spatial.shape[0] + 2) / columns)
            figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.8 * rows), squeeze=False)
            panels = [(confidence, "confidence", False, "magma"), (assignments.float(), "top-1 assignment", True, "tab20")]
            panels.extend((spatial[idx], f"expert {idx}", False, "inferno") for idx in range(spatial.shape[0]))
            for panel_idx, (values, title, categorical, cmap) in enumerate(panels):
                low, high = (self._display_range(values) if not categorical else (None, None))
                self._plot_map(
                    axes.flat[panel_idx],
                    values,
                    background,
                    title=title,
                    categorical=categorical,
                    cmap=cmap,
                    vmin=low,
                    vmax=high,
                )
            for axis in axes.flat[len(panels) :]:
                axis.axis("off")
            figure.suptitle(f"{name} ({heatmap.module_type})")
            figure.tight_layout()
            figure.savefig(dashboard_path, dpi=160, bbox_inches="tight")
            plt.close(figure)
            layer_written["dashboard"] = dashboard_path
            written[name] = layer_written
        return written

    def save_routing_heatmaps(
        self,
        heatmaps: Mapping[str, RoutingHeatmap],
        output_dir: str | Path,
        *,
        input_image: str | Path | torch.Tensor | None = None,
    ) -> dict[str, dict[str, Path]]:
        """Compatibility alias for :meth:`save_routing_visualizations`."""
        return self.save_routing_visualizations(heatmaps, output_dir, input_image=input_image)

    def analyze_expert_specialization(
        self,
        dataset: Iterable[Any],
        *,
        num_samples: int = 1000,
        max_batches: int | None = None,
        layer_name: str | None = None,
        feature_fn: Callable[[Any], Mapping[str, Any]] | None = None,
        forward_fn: Callable[[nn.Module, Any], Any] | None = None,
    ) -> dict[str, ExpertSpecializationReport]:
        """Aggregate per-expert input signatures over a dataset."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        accumulators: dict[str, dict[str, Any]] = {}
        for batch_index, batch in enumerate(dataset):
            if max_batches is not None and batch_index >= max_batches:
                break
            heatmaps = self.capture_routing(batch, layer_name=layer_name, forward_fn=forward_fn)
            descriptors = self._feature_descriptors(batch, feature_fn)
            if not descriptors:
                raise ValueError("could not derive numeric per-sample features from the dataset batch")

            batch_size = next(iter(descriptors.values())).numel()
            remaining = num_samples - max((state["num_samples"] for state in accumulators.values()), default=0)
            if remaining <= 0:
                break
            take = min(batch_size, remaining)

            for name, heatmap in heatmaps.items():
                probabilities = heatmap.probabilities
                sample_usage = probabilities.flatten(2).mean(dim=2) if probabilities.ndim > 2 else probabilities
                take_for_layer = min(take, sample_usage.shape[0])
                sample_usage = sample_usage[:take_for_layer]
                sample_usage = sample_usage / sample_usage.sum(dim=1, keepdim=True).clamp_min(1e-12)
                num_experts = sample_usage.shape[1]
                state = accumulators.setdefault(
                    name,
                    {
                        "module_type": heatmap.module_type,
                        "num_samples": 0,
                        "usage_sum": torch.zeros(num_experts, dtype=torch.float64),
                        "dominant": torch.zeros(num_experts, dtype=torch.long),
                        "feature_sum": [dict() for _ in range(num_experts)],
                        "feature_weight": torch.zeros(num_experts, dtype=torch.float64),
                    },
                )
                state["num_samples"] += take_for_layer
                state["usage_sum"] += sample_usage.double().sum(dim=0)
                state["dominant"] += torch.bincount(sample_usage.argmax(dim=1), minlength=num_experts)
                state["feature_weight"] += sample_usage.double().sum(dim=0)

                for feature_name, values in descriptors.items():
                    values = values[:take_for_layer].double()
                    if values.numel() != take_for_layer:
                        raise ValueError(f"feature {feature_name!r} does not match batch size {take_for_layer}")
                    for expert_idx in range(num_experts):
                        weighted_sum = float((sample_usage[:, expert_idx].double() * values).sum())
                        current = state["feature_sum"][expert_idx].get(feature_name, 0.0)
                        state["feature_sum"][expert_idx][feature_name] = current + weighted_sum

            if accumulators and min(state["num_samples"] for state in accumulators.values()) >= num_samples:
                break

        reports: dict[str, ExpertSpecializationReport] = {}
        for name, state in accumulators.items():
            mean_usage = state["usage_sum"] / max(state["num_samples"], 1)
            mean_usage = mean_usage / mean_usage.sum().clamp_min(1e-12)
            signatures = []
            for expert_idx, feature_sums in enumerate(state["feature_sum"]):
                denominator = max(float(state["feature_weight"][expert_idx]), 1e-12)
                signatures.append({key: float(value / denominator) for key, value in sorted(feature_sums.items())})
            reports[name] = ExpertSpecializationReport(
                layer_name=name,
                module_type=state["module_type"],
                num_experts=mean_usage.numel(),
                num_samples=state["num_samples"],
                mean_usage=tuple(float(value) for value in mean_usage.tolist()),
                dominant_samples=tuple(int(value) for value in state["dominant"].tolist()),
                feature_signatures=tuple(signatures),
            )
        return reports

    def routing_causal_analysis(
        self,
        batch: Any,
        layer_name: str,
        expert_idx: int,
        *,
        forward_fn: Callable[[nn.Module, Any], Any] | None = None,
    ) -> RoutingCausalReport:
        """Compare natural routing with a temporary forced-expert counterfactual."""
        modules = self._routed_modules(layer_name=layer_name, leaf_only=False)
        if layer_name not in modules:
            raise ValueError(f"routed layer {layer_name!r} was not found")
        module = modules[layer_name]
        num_experts = int(getattr(module, "num_experts", 0))
        if not 0 <= expert_idx < num_experts:
            raise ValueError(f"expert_idx must be in [0, {num_experts - 1}], got {expert_idx}")
        router = self._router_for(module)
        if router is None:
            raise ValueError(f"routed layer {layer_name!r} has no supported router or routing submodule")

        training_flags = self._training_flags()
        snapshots = {
            routed: dict(getattr(routed, "last_routing_snapshot", {}))
            for routed in self.model.modules()
            if hasattr(routed, "last_routing_snapshot")
        }
        handle = None
        try:
            self.model.eval()
            with torch.no_grad():
                natural = self._forward(batch, forward_fn)

            def force_hook(_router, _inputs, output):
                return self._force_router_output(output, num_experts, expert_idx)

            handle = router.register_forward_hook(force_hook)
            with torch.no_grad():
                forced = self._forward(batch, forward_fn)
            return self._compare_outputs(layer_name, expert_idx, natural, forced)
        finally:
            if handle is not None:
                handle.remove()
            for routed, snapshot in snapshots.items():
                routed.last_routing_snapshot = snapshot
            self._restore_training_flags(training_flags)

    def _routed_modules(self, *, layer_name: str | None = None, leaf_only: bool) -> dict[str, nn.Module]:
        routed = {
            name or "<root>": module
            for name, module in self.model.named_modules()
            if self._is_interpretable_routed_module(module)
        }
        if layer_name is not None:
            return {layer_name: routed[layer_name]} if layer_name in routed else {}
        if not leaf_only:
            return routed
        return {
            name: module
            for name, module in routed.items()
            if not any(child is not module and self._is_interpretable_routed_module(child) for child in module.modules())
        }

    @classmethod
    def _is_interpretable_routed_module(cls, module: nn.Module) -> bool:
        """Accept the unified protocol and pre-protocol routed checkpoints."""
        if is_routed_module(module):
            return True
        num_experts = getattr(module, "num_experts", 0)
        return isinstance(num_experts, int) and num_experts > 0 and cls._router_for(module) is not None

    @staticmethod
    def _router_for(module: nn.Module) -> nn.Module | None:
        for attribute in ("router", "routing", "gate", "gating"):
            candidate = getattr(module, attribute, None)
            if isinstance(candidate, nn.Module):
                return candidate
        return None

    @staticmethod
    def _normalized_vector(value: Any, num_experts: int) -> torch.Tensor | None:
        if value is None or num_experts <= 0:
            return None
        try:
            tensor = value.detach().float().cpu().reshape(-1) if isinstance(value, torch.Tensor) else torch.tensor(value).float().reshape(-1)
        except (TypeError, ValueError):
            return None
        if tensor.numel() != num_experts:
            return None
        tensor = torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor)).clamp_min(0.0)
        total = tensor.sum()
        if float(total) <= 0:
            return torch.full((num_experts,), 1.0 / num_experts)
        return tensor / total

    @staticmethod
    def _scalar_float(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().reshape(-1)[0]) if value.numel() else 0.0
        try:
            result = float(value)
        except (TypeError, ValueError):
            return 0.0
        return result if math.isfinite(result) else 0.0

    @staticmethod
    def _normalized_gini(usage: torch.Tensor) -> float:
        values = usage.double().clamp_min(0.0)
        count = values.numel()
        if count <= 1 or float(values.sum()) <= 0:
            return 0.0
        values = torch.sort(values / values.sum()).values
        index = torch.arange(1, count + 1, dtype=values.dtype)
        gini = (2.0 * torch.sum(index * values) / count) - ((count + 1.0) / count)
        return float((gini * count / (count - 1)).clamp(0.0, 1.0))

    @staticmethod
    def _normalized_entropy(usage: torch.Tensor) -> float:
        values = usage.double().clamp_min(0.0)
        count = values.numel()
        if count <= 1 or float(values.sum()) <= 0:
            return 0.0
        values = values / values.sum()
        entropy = -(values * values.clamp_min(1e-12).log()).sum()
        return float((entropy / math.log(count)).clamp(0.0, 1.0))

    @classmethod
    def _router_probabilities(cls, output: Any, num_experts: int) -> torch.Tensor | None:
        sparse_pair = cls._sparse_router_pair(output, num_experts)
        if sparse_pair is not None:
            weights, indices, axis = sparse_pair
            return cls._dense_sparse_probabilities(weights, indices, axis, num_experts)

        tensor = cls._first_router_tensor(output, num_experts)
        if tensor is None:
            return None
        axis = cls._expert_axis(tensor, num_experts)
        if axis is None:
            return None
        probabilities = tensor if cls._looks_like_probabilities(tensor, axis) else tensor.float().softmax(dim=axis)
        if axis != 1:
            probabilities = probabilities.movedim(axis, 1)
        if probabilities.ndim == 1:
            probabilities = probabilities.unsqueeze(0)
        probabilities = probabilities.float().clamp_min(0.0)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return cls._squeeze_image_level_spatial_dims(probabilities)

    @classmethod
    def _sparse_router_pair(
        cls, output: Any, num_experts: int
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        """Find matching top-k weights and indices inside a router output."""
        tensors = cls._router_tensors(output)
        weights = [tensor for tensor in tensors if tensor.is_floating_point()]
        indices = [tensor for tensor in tensors if not tensor.is_floating_point() and tensor.dtype != torch.bool]
        for weight_tensor in weights:
            for index_tensor in indices:
                axis = cls._sparse_expert_axis(weight_tensor, index_tensor, num_experts)
                if axis is not None:
                    return weight_tensor, index_tensor, axis
        return None

    @classmethod
    def _router_tensors(cls, output: Any) -> list[torch.Tensor]:
        if isinstance(output, torch.Tensor):
            return [output]
        if isinstance(output, Mapping):
            values = output.values()
        elif isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
            values = output
        else:
            return []
        tensors: list[torch.Tensor] = []
        for value in values:
            tensors.extend(cls._router_tensors(value))
        return tensors

    @staticmethod
    def _sparse_expert_axis(weights: torch.Tensor, indices: torch.Tensor, num_experts: int) -> int | None:
        if weights.shape != indices.shape or weights.ndim == 0 or weights.numel() == 0 or num_experts <= 0:
            return None
        if int(indices.min()) < 0 or int(indices.max()) >= num_experts:
            return None
        if weights.ndim == 1:
            return 0 if weights.shape[0] <= num_experts else None
        if weights.shape[1] <= num_experts:
            return 1
        if weights.shape[-1] <= num_experts:
            return weights.ndim - 1
        return None

    @classmethod
    def _dense_sparse_probabilities(
        cls,
        weights: torch.Tensor,
        indices: torch.Tensor,
        axis: int,
        num_experts: int,
    ) -> torch.Tensor:
        sparse_weights = weights.detach().float()
        sparse_weights = torch.where(
            torch.isfinite(sparse_weights), sparse_weights, torch.zeros_like(sparse_weights)
        ).clamp_min(0.0)
        sparse_indices = indices.detach().long()
        if axis != 1:
            sparse_weights = sparse_weights.movedim(axis, 1)
            sparse_indices = sparse_indices.movedim(axis, 1)
        if sparse_weights.ndim == 1:
            sparse_weights = sparse_weights.unsqueeze(0)
            sparse_indices = sparse_indices.unsqueeze(0)

        dense_shape = list(sparse_weights.shape)
        dense_shape[1] = num_experts
        probabilities = sparse_weights.new_zeros(dense_shape)
        probabilities.scatter_add_(1, sparse_indices, sparse_weights)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return cls._squeeze_image_level_spatial_dims(probabilities)

    @staticmethod
    def _squeeze_image_level_spatial_dims(probabilities: torch.Tensor) -> torch.Tensor:
        if probabilities.ndim > 2 and all(size == 1 for size in probabilities.shape[2:]):
            probabilities = probabilities.reshape(probabilities.shape[0], probabilities.shape[1])
        return probabilities

    @classmethod
    def _first_router_tensor(cls, output: Any, num_experts: int) -> torch.Tensor | None:
        if isinstance(output, torch.Tensor):
            return output if output.is_floating_point() and cls._expert_axis(output, num_experts) is not None else None
        if isinstance(output, Mapping):
            candidates = output.values()
        elif isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
            candidates = output
        else:
            return None
        for candidate in candidates:
            tensor = cls._first_router_tensor(candidate, num_experts)
            if tensor is not None:
                return tensor
        return None

    @staticmethod
    def _expert_axis(tensor: torch.Tensor, num_experts: int) -> int | None:
        if tensor.ndim == 0 or num_experts <= 0:
            return None
        if tensor.ndim > 1 and tensor.shape[1] == num_experts:
            return 1
        if tensor.shape[-1] == num_experts:
            return tensor.ndim - 1
        if tensor.ndim == 1 and tensor.shape[0] == num_experts:
            return 0
        return None

    @staticmethod
    def _looks_like_probabilities(tensor: torch.Tensor, axis: int) -> bool:
        detached = tensor.detach().float()
        if detached.numel() == 0 or not bool(torch.isfinite(detached).all()) or not bool((detached >= -1e-6).all()):
            return False
        sums = detached.sum(dim=axis)
        return bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4))

    def _training_flags(self) -> dict[nn.Module, bool]:
        return {module: bool(module.training) for module in self.model.modules()}

    @staticmethod
    def _restore_training_flags(flags: Mapping[nn.Module, bool]) -> None:
        for module, training in flags.items():
            module.training = training

    def _forward(self, batch: Any, forward_fn: Callable[[nn.Module, Any], Any] | None) -> Any:
        if forward_fn is not None:
            return forward_fn(self.model, batch)
        if isinstance(batch, torch.Tensor):
            return self.model(batch)
        if isinstance(batch, Mapping):
            return self.model(**batch)
        if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
            return self.model(*batch)
        return self.model(batch)

    @classmethod
    def _feature_descriptors(
        cls, batch: Any, feature_fn: Callable[[Any], Mapping[str, Any]] | None
    ) -> dict[str, torch.Tensor]:
        if feature_fn is not None:
            raw = feature_fn(batch)
            if not isinstance(raw, Mapping):
                raise TypeError("feature_fn must return a mapping of feature names to per-sample values")
            return cls._normalize_descriptors(raw)

        tensor = cls._first_input_tensor(batch)
        if tensor is None or tensor.ndim == 0:
            return {}
        feature = tensor.detach().float().cpu()
        if feature.ndim == 1:
            feature = feature.unsqueeze(0)
        flat = feature.flatten(1)
        descriptors = {
            "activation_mean": flat.mean(dim=1),
            "activation_std": flat.std(dim=1, unbiased=False),
            "activation_rms": flat.square().mean(dim=1).sqrt(),
        }
        if feature.ndim >= 4:
            descriptors["spatial_height"] = torch.full((feature.shape[0],), float(feature.shape[-2]))
            descriptors["spatial_width"] = torch.full((feature.shape[0],), float(feature.shape[-1]))
            dx = (feature[..., 1:] - feature[..., :-1]).abs().flatten(1).mean(dim=1) if feature.shape[-1] > 1 else torch.zeros(feature.shape[0])
            dy = (feature[..., 1:, :] - feature[..., :-1, :]).abs().flatten(1).mean(dim=1) if feature.shape[-2] > 1 else torch.zeros(feature.shape[0])
            descriptors["high_frequency"] = 0.5 * (dx + dy)
        return descriptors

    @classmethod
    def _first_input_tensor(cls, value: Any) -> torch.Tensor | None:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, Mapping):
            preferred = value.get("img")
            if isinstance(preferred, torch.Tensor):
                return preferred
            candidates = value.values()
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            candidates = value
        else:
            return None
        for candidate in candidates:
            tensor = cls._first_input_tensor(candidate)
            if tensor is not None:
                return tensor
        return None

    @staticmethod
    def _normalize_descriptors(raw: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        descriptors: dict[str, torch.Tensor] = {}
        for name, value in raw.items():
            try:
                tensor = value.detach().float().cpu().reshape(-1) if isinstance(value, torch.Tensor) else torch.tensor(value).float().reshape(-1)
            except (TypeError, ValueError):
                continue
            if tensor.numel() and bool(torch.isfinite(tensor).all()):
                descriptors[str(name)] = tensor
        return descriptors

    @classmethod
    def _force_router_output(cls, output: Any, num_experts: int, expert_idx: int) -> Any:
        sparse_pair = cls._sparse_router_pair(output, num_experts)
        if sparse_pair is not None:
            weights, indices, axis = sparse_pair
            forced_weights = torch.zeros_like(weights)
            selection = [slice(None)] * weights.ndim
            selection[axis] = 0
            forced_weights[tuple(selection)] = 1
            replacements = {
                id(weights): forced_weights,
                id(indices): torch.full_like(indices, expert_idx),
            }
            return cls._replace_router_tensors(output, replacements)

        if isinstance(output, torch.Tensor):
            return cls._forced_tensor(output, num_experts, expert_idx)
        if isinstance(output, tuple):
            values = list(output)
            replaced = False
            for index, value in enumerate(values):
                if not replaced and isinstance(value, torch.Tensor) and value.is_floating_point() and cls._expert_axis(value, num_experts) is not None:
                    values[index] = cls._forced_tensor(value, num_experts, expert_idx)
                    replaced = True
                elif replaced and isinstance(value, torch.Tensor) and not value.is_floating_point():
                    values[index] = torch.full_like(value, expert_idx)
            return tuple(values)
        if isinstance(output, list):
            values = cls._force_router_output(tuple(output), num_experts, expert_idx)
            return list(values)
        if isinstance(output, Mapping):
            values = dict(output)
            for key, value in values.items():
                if isinstance(value, torch.Tensor) and value.is_floating_point() and cls._expert_axis(value, num_experts) is not None:
                    values[key] = cls._forced_tensor(value, num_experts, expert_idx)
                    break
            return type(output)(values)
        raise TypeError(f"unsupported router output type {type(output)!r}")

    @classmethod
    def _replace_router_tensors(cls, output: Any, replacements: Mapping[int, torch.Tensor]) -> Any:
        if isinstance(output, torch.Tensor):
            return replacements.get(id(output), output)
        if isinstance(output, tuple):
            return tuple(cls._replace_router_tensors(value, replacements) for value in output)
        if isinstance(output, list):
            return [cls._replace_router_tensors(value, replacements) for value in output]
        if isinstance(output, Mapping):
            values = {
                key: cls._replace_router_tensors(value, replacements)
                for key, value in output.items()
            }
            try:
                return type(output)(values)
            except TypeError:
                return values
        return output

    @classmethod
    def _forced_tensor(cls, tensor: torch.Tensor, num_experts: int, expert_idx: int) -> torch.Tensor:
        axis = cls._expert_axis(tensor, num_experts)
        if axis is None:
            return tensor
        forced = torch.zeros_like(tensor)
        selection = [slice(None)] * tensor.ndim
        selection[axis] = expert_idx
        if cls._looks_like_probabilities(tensor, axis):
            forced[tuple(selection)] = 1
        else:
            forced.fill_(-50.0)
            forced[tuple(selection)] = 50.0
        return forced

    @classmethod
    def _compare_outputs(cls, layer_name: str, expert_idx: int, natural: Any, forced: Any) -> RoutingCausalReport:
        natural_tensors = cls._output_tensors(natural)
        forced_tensors = cls._output_tensors(forced)
        pairs = [
            (left.detach().float().reshape(-1), right.detach().float().reshape(-1))
            for left, right in zip(natural_tensors, forced_tensors)
            if left.shape == right.shape and left.is_floating_point() and right.is_floating_point()
        ]
        if not pairs:
            raise ValueError("model outputs do not contain matching floating-point tensors for causal comparison")

        element_count = sum(left.numel() for left, _ in pairs)
        absolute_sum = sum(float((left - right).abs().sum()) for left, right in pairs)
        square_sum = sum(float((left - right).square().sum()) for left, right in pairs)
        max_difference = max(float((left - right).abs().max()) for left, right in pairs)
        dot = sum(float(torch.dot(left, right)) for left, right in pairs)
        natural_square = sum(float(torch.dot(left, left)) for left, _ in pairs)
        forced_square = sum(float(torch.dot(right, right)) for _, right in pairs)
        denominator = math.sqrt(natural_square) * math.sqrt(forced_square)
        cosine = dot / denominator if denominator > 0 else 1.0
        return RoutingCausalReport(
            layer_name=layer_name,
            expert_idx=expert_idx,
            tensor_count=len(pairs),
            element_count=element_count,
            mean_absolute_difference=absolute_sum / max(element_count, 1),
            root_mean_square_difference=math.sqrt(square_sum / max(element_count, 1)),
            max_absolute_difference=max_difference,
            cosine_similarity=max(min(cosine, 1.0), -1.0),
            natural_l2_norm=math.sqrt(natural_square),
            forced_l2_norm=math.sqrt(forced_square),
        )

    @classmethod
    def _output_tensors(cls, output: Any) -> list[torch.Tensor]:
        if isinstance(output, torch.Tensor):
            return [output]
        if isinstance(output, Mapping):
            values = output.values()
        elif isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
            values = output
        else:
            return []
        tensors: list[torch.Tensor] = []
        for value in values:
            tensors.extend(cls._output_tensors(value))
        return tensors

    @staticmethod
    def _load_image(image: str | Path | torch.Tensor | None) -> torch.Tensor | None:
        if image is None:
            return None
        if isinstance(image, torch.Tensor):
            tensor = image.detach().float().cpu()
            if tensor.ndim == 4:
                tensor = tensor[0]
            if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
                tensor = tensor.permute(1, 2, 0)
            if tensor.ndim == 3 and tensor.shape[-1] == 1:
                tensor = tensor[..., 0]
            if tensor.numel():
                tensor = tensor - tensor.min()
                tensor = tensor / tensor.max().clamp_min(1e-12)
            return tensor
        import matplotlib.pyplot as plt

        tensor = torch.tensor(plt.imread(Path(image))).float()
        if tensor.numel() and float(tensor.max()) > 1.0:
            tensor = tensor / 255.0
        return tensor.clamp(0.0, 1.0)

    def _save_overlay(
        self,
        values: torch.Tensor,
        background: torch.Tensor | None,
        path: Path,
        *,
        title: str,
        categorical: bool,
        cmap: str,
        vmin: float | None,
        vmax: float | None,
    ) -> None:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(7.0, 5.0))
        self._plot_map(axis, values, background, title=title, categorical=categorical, cmap=cmap, vmin=vmin, vmax=vmax)
        if not categorical:
            figure.colorbar(axis.images[-1], ax=axis, fraction=0.046, pad=0.04, label="activation")
        figure.tight_layout()
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)

    @staticmethod
    def _display_range(values: torch.Tensor) -> tuple[float, float]:
        """Use robust per-map limits so low-contrast activations remain visible."""
        flat = values.detach().float().reshape(-1)
        low, high = torch.quantile(flat, torch.tensor([0.02, 0.98])).tolist()
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            low, high = float(flat.min()), float(flat.max())
        if high <= low:
            high = low + 1e-6
        return low, high

    @staticmethod
    def _plot_map(
        axis,
        values: torch.Tensor,
        background: torch.Tensor | None,
        *,
        title: str,
        categorical: bool = False,
        cmap: str = "magma",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        values = values.detach().float().cpu()
        if background is not None and background.ndim in (2, 3):
            height, width = int(background.shape[0]), int(background.shape[1])
            axis.imshow(background.numpy(), cmap="gray" if background.ndim == 2 else None)
            resized = torch.nn.functional.interpolate(
                values[None, None], size=(height, width), mode="nearest" if categorical else "bilinear", align_corners=False if not categorical else None
            )[0, 0]
            axis.imshow(resized.numpy(), alpha=0.58, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            axis.imshow(values.numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")


__all__ = [
    "ExpertSpecializationReport",
    "RoutingCausalReport",
    "RoutingCollapseReport",
    "RoutingHeatmap",
    "RoutingInterpreter",
    "RoutingLayerSummary",
]
