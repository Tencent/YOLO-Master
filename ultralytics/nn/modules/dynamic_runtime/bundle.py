"""Split-ONNX exporter for host-orchestrated conditional expert execution."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .dispatch import DynamicDispatchContractError


BUNDLE_SCHEMA_VERSION = 1


class _ESRouterWrapper(nn.Module):
    def __init__(self, routing: nn.Module):
        super().__init__()
        self.routing = copy.deepcopy(routing).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.routing.global_pool(x)
        logits = self.routing.routing_network(pooled)
        probabilities = F.softmax(logits.float().clamp(-30.0, 30.0), dim=1).type_as(x)
        return probabilities.repeat(1, 1, x.size(2), x.size(3))


class _ESPostprocessWrapper(nn.Module):
    def __init__(self, norm: nn.Module):
        super().__init__()
        self.norm = copy.deepcopy(norm).eval()

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        return self.norm(mixture)


class _MoTRouterWrapper(nn.Module):
    def __init__(self, router: nn.Module):
        super().__init__()
        self.router = copy.deepcopy(router).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router._compute_logits(x)
        return F.softmax(logits / self.router.temperature.float(), dim=1).to(dtype=x.dtype)


class _MoTPostprocessWrapper(nn.Module):
    def __init__(self, out_proj: nn.Module, out_norm: nn.Module):
        super().__init__()
        self.out_proj = copy.deepcopy(out_proj).eval()
        self.out_norm = copy.deepcopy(out_norm).eval()

    def forward(self, x: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        return self.out_norm(self.out_proj(mixture)) + x


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path, root: Path, *, inputs: Sequence[str], outputs: Sequence[str]) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "inputs": list(inputs),
        "outputs": list(outputs),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _export_onnx(
    module: nn.Module,
    args: torch.Tensor | tuple[torch.Tensor, ...],
    path: Path,
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
    opset: int,
) -> None:
    dynamic_axes = {name: {0: "batch"} for name in (*input_names, *output_names)}
    with torch.no_grad():
        torch.onnx.export(
            module.eval(),
            args,
            str(path),
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_axes=dynamic_axes,
            opset_version=int(opset),
            do_constant_folding=True,
            dynamo=False,
        )


def _prepare_output_directory(output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    occupied = list(output_dir.iterdir())
    if occupied and not overwrite:
        names = ", ".join(path.name for path in occupied[:5])
        raise FileExistsError(
            f"dynamic bundle directory is not empty: {output_dir} ({names}). "
            "Use a new directory or pass overwrite=True."
        )


def export_dynamic_expert_bundle(
    module: nn.Module,
    sample: torch.Tensor,
    output_dir: str | Path,
    *,
    opset: int = 17,
    overwrite: bool = False,
) -> Path:
    """Export one routed block as router/expert/postprocess ONNX artifacts.

    The resulting bundle intentionally contains no dense expert-combine graph.
    A host runtime reads router output, launches only selected expert sessions,
    combines their outputs, and then runs the postprocess graph.

    Args:
        module: An evaluated ``ES_MOE`` or ``MoTBlock`` instance.
        sample: Representative NCHW input. Spatial dimensions remain static;
            only batch is marked dynamic in the first PoC contract.
        output_dir: Empty output directory for the bundle.
        opset: ONNX opset version.
        overwrite: Permit replacing files in an existing output directory.

    Returns:
        Path to ``dynamic_bundle.json``.
    """
    from ultralytics.nn.modules.moe.modules import ES_MOE
    from ultralytics.nn.modules.mot.block import MoTBlock

    if module.training:
        raise DynamicDispatchContractError("dynamic expert bundles must be exported from module.eval()")
    if sample.ndim != 4 or sample.shape[0] < 1:
        raise DynamicDispatchContractError(f"sample must be non-empty NCHW, got {tuple(sample.shape)}")
    if not torch.isfinite(sample).all():
        raise DynamicDispatchContractError("sample contains NaN or Inf")

    output_path = Path(output_dir).resolve()
    _prepare_output_directory(output_path, overwrite=overwrite)

    if isinstance(module, ES_MOE):
        family = "ES-MoE"
        num_experts = int(module.num_experts)
        top_k = int(module.top_k)
        if not module._eager_sparse_enabled():
            raise DynamicDispatchContractError("ES_MOE does not have eager Top-K sparse inference enabled")
        router_wrapper = _ESRouterWrapper(module.routing)
        postprocess_wrapper = _ESPostprocessWrapper(module.norm)
        postprocess_inputs = ("mixture",)
        routing_granularity = "sample"
        dynamic_threshold = float(module.dynamic_threshold)
        host_topk_tie_tolerance = float(getattr(module.routing, "route_tie_tolerance", 1e-6))
        postprocess_kind = "norm"
    elif isinstance(module, MoTBlock):
        family = "MoT"
        num_experts = int(module.NUM_EXPERTS)
        top_k = int(module.top_k)
        if top_k >= num_experts:
            raise DynamicDispatchContractError(
                "MoTBlock top_k selects every expert; no conditional execution is possible"
            )
        router_wrapper = _MoTRouterWrapper(module.router)
        postprocess_wrapper = _MoTPostprocessWrapper(module.out_proj, module.out_norm)
        postprocess_inputs = ("x", "mixture")
        routing_granularity = "spatial_union" if bool(module.router.use_spatial) else "sample"
        dynamic_threshold = 0.0
        host_topk_tie_tolerance = float(getattr(module.router, "route_tie_tolerance", 1e-6))
        postprocess_kind = "projection_norm_residual"
    else:
        raise TypeError(f"unsupported dynamic expert module: {type(module).__module__}.{type(module).__name__}")

    if top_k >= num_experts:
        raise DynamicDispatchContractError(
            f"top_k={top_k} selects all {num_experts} experts; refusing to label the bundle as conditional"
        )

    router_path = output_path / "router.onnx"
    _export_onnx(
        router_wrapper,
        sample,
        router_path,
        input_names=("x",),
        output_names=("routing_weights",),
        opset=opset,
    )

    expert_records: list[dict] = []
    with torch.no_grad():
        mixture_sample = module.experts[0](sample)
    for expert_index, expert in enumerate(module.experts):
        expert_path = output_path / f"expert_{expert_index}.onnx"
        _export_onnx(
            copy.deepcopy(expert).eval(),
            sample,
            expert_path,
            input_names=("x",),
            output_names=("expert_output",),
            opset=opset,
        )
        expert_records.append(
            _artifact_record(expert_path, output_path, inputs=("x",), outputs=("expert_output",))
        )

    postprocess_path = output_path / "postprocess.onnx"
    postprocess_args = mixture_sample if family == "ES-MoE" else (sample, mixture_sample)
    _export_onnx(
        postprocess_wrapper,
        postprocess_args,
        postprocess_path,
        input_names=postprocess_inputs,
        output_names=("y",),
        opset=opset,
    )

    compute_reduction_guaranteed = routing_granularity == "sample"
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "format": "yolo-master-split-onnx-dynamic-experts",
        "family": family,
        "module_type": f"{type(module).__module__}.{type(module).__name__}",
        "execution_semantics": "host_conditional_expert_dispatch",
        "masked_dense_allowed": False,
        "reference_semantics": "eager_sparse_portable_tie_break",
        "num_experts": num_experts,
        "top_k": top_k,
        "router_output_semantics": "dense_probabilities_host_topk",
        "host_topk_tie_break": "probability_minus_expert_id_times_tolerance",
        "host_topk_tie_tolerance": host_topk_tie_tolerance,
        "routing_granularity": routing_granularity,
        "dynamic_threshold": dynamic_threshold,
        "zero_tolerance": 1e-8,
        "dynamic_batch": True,
        "dynamic_spatial": False,
        "sample_shape": [int(value) for value in sample.shape],
        "sample_dtype": str(sample.detach().cpu().numpy().dtype),
        "compute_reduction_guaranteed_per_sample": compute_reduction_guaranteed,
        "claim_boundary": (
            "Only selected expert sessions are executed. Spatial routing may select the union of all experts for one "
            "sample, so each call must be accepted or rejected from its runtime audit."
            if routing_granularity == "spatial_union"
            else "Top-K is global per sample; each sample executes fewer than all experts."
        ),
        "artifacts": {
            "router": _artifact_record(router_path, output_path, inputs=("x",), outputs=("routing_weights",)),
            "experts": expert_records,
            "postprocess": _artifact_record(
                postprocess_path,
                output_path,
                inputs=postprocess_inputs,
                outputs=("y",),
            ),
        },
        "postprocess_kind": postprocess_kind,
    }
    manifest_path = output_path / "dynamic_bundle.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


__all__ = ("BUNDLE_SCHEMA_VERSION", "export_dynamic_expert_bundle")
