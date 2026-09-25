"""Validate a complete eager YOLO model with routed blocks replaced by split-ORT adapters."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_dynamic_blocks_from_checkpoint import load_model, make_capture_sample, sha256
from ultralytics.nn.modules.dynamic_runtime import ORTDynamicBlockAdapter, ORTRouterTorchExpertAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("model_manifest", type=Path)
    parser.add_argument("input_image", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-reduction", action="store_true")
    parser.add_argument(
        "--bridge-mode",
        choices=("split-ort", "ort-router-torch-experts"),
        default="split-ort",
        help="Use all split ORT components or only the ORT router with checkpoint PyTorch experts",
    )
    parser.add_argument(
        "--skip-route-drift-audit",
        action="store_true",
        help="Do not run the eager checkpoint router alongside the ORT router",
    )
    return parser.parse_args()


def first_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for value in output.values():
            try:
                return first_tensor(value)
            except TypeError:
                continue
    if isinstance(output, (tuple, list)):
        for value in output:
            try:
                return first_tensor(value)
            except TypeError:
                continue
    raise TypeError(f"model output contains no tensor: {type(output)}")


def replace_submodule(model: torch.nn.Module, name: str, replacement: torch.nn.Module) -> None:
    parent_name, child_name = name.rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    if child_name.isdigit() and isinstance(parent, (torch.nn.Sequential, torch.nn.ModuleList)):
        parent[int(child_name)] = replacement
    else:
        setattr(parent, child_name, replacement)


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cpu" and args.bridge_mode == "split-ort":
        raise ValueError("the current PyTorch/NumPy/ORT bridge is CPU-only; GPU requires the future zero-copy plugin")

    checkpoint_path = args.checkpoint.resolve()
    manifest_path = args.model_manifest.resolve()
    input_image = args.input_image.resolve()
    model_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if model_manifest["scope"] != "routed_blocks_only":
        raise ValueError("model manifest does not describe routed block bundles")
    if sha256(checkpoint_path) != model_manifest["checkpoint"]["sha256"]:
        raise ValueError("checkpoint SHA256 does not match the model manifest")
    source = model_manifest.get("capture_source", {})
    if source.get("kind") != "letterboxed_image" or sha256(input_image) != source.get("sha256"):
        raise ValueError("input image does not match the image-backed model manifest")

    model = load_model(checkpoint_path, device)
    sample, capture_source = make_capture_sample(
        input_image=input_image,
        batch=1,
        imgsz=args.imgsz,
        device=device,
    )
    with torch.no_grad():
        eager_started = time.perf_counter()
        eager_output = first_tensor(model(sample)).detach().cpu().numpy()
        eager_latency_ms = (time.perf_counter() - eager_started) * 1000.0

    adapters: list[tuple[str, ORTDynamicBlockAdapter | ORTRouterTorchExpertAdapter]] = []
    for block in model_manifest["blocks"]:
        block_manifest = manifest_path.parent / block["bundle_manifest"]
        if args.bridge_mode == "split-ort":
            adapter = ORTDynamicBlockAdapter(
                block_manifest,
                providers=["CPUExecutionProvider"],
                require_reduction=args.require_reduction,
            ).eval()
        else:
            original_block = model.get_submodule(block["module_name"])
            adapter = ORTRouterTorchExpertAdapter(
                original_block,
                block_manifest,
                providers=["CPUExecutionProvider"],
                require_reduction=args.require_reduction,
                compare_eager_routes=not args.skip_route_drift_audit,
            ).eval()
        replace_submodule(model, block["module_name"], adapter)
        adapters.append((block["module_name"], adapter))

    with torch.no_grad():
        hybrid_started = time.perf_counter()
        hybrid_output = first_tensor(model(sample)).detach().cpu().numpy()
        hybrid_latency_ms = (time.perf_counter() - hybrid_started) * 1000.0

    if eager_output.shape != hybrid_output.shape:
        raise RuntimeError(f"output shape mismatch: eager={eager_output.shape}, hybrid={hybrid_output.shape}")
    if not np.isfinite(hybrid_output).all():
        raise RuntimeError("hybrid model output contains NaN or Inf")

    block_audits = []
    for name, adapter in adapters:
        if adapter.last_audit is None:
            raise RuntimeError(f"dynamic adapter was not executed: {name}")
        block_audits.append(
            {
                "module_name": name,
                "loaded_onnx_expert_ids": list(
                    adapter.loaded_expert_ids
                    if isinstance(adapter, ORTDynamicBlockAdapter)
                    else adapter.loaded_onnx_expert_ids
                ),
                "audit": adapter.last_audit.to_dict(),
                "route_drift": getattr(adapter, "last_route_drift", {}),
                "aggregate_execution": (
                    adapter.execution_summary()
                    if isinstance(adapter, ORTRouterTorchExpertAdapter)
                    else None
                ),
            }
        )

    result = {
        "schema_version": 1,
        "status": (
            "EXECUTION_PASS_WITH_ROUTE_DRIFT"
            if model_manifest.get("status") == "VALIDATED_WITH_ROUTE_DRIFT"
            else "EXECUTION_PASS"
        ),
        "scope": (
            "full_yolo_pytorch_outer_graph_split_ort_routed_blocks"
            if args.bridge_mode == "split-ort"
            else "full_yolo_pytorch_outer_graph_ort_router_pytorch_checkpoint_experts"
        ),
        "bridge_mode": args.bridge_mode,
        "not_a_full_export": True,
        "not_tensorrt": True,
        "accuracy_gate_passed": False,
        "accuracy_gate_reason": "Full VisDrone mAP has not been evaluated.",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": model_manifest["checkpoint"]["sha256"],
        },
        "model_manifest": str(manifest_path),
        "capture_source": capture_source,
        "output_shape": list(hybrid_output.shape),
        "eager_vs_hybrid_max_abs_error": float(np.max(np.abs(eager_output - hybrid_output))),
        "eager_latency_ms_single_unwarmed": eager_latency_ms,
        "hybrid_latency_ms_single_unwarmed": hybrid_latency_ms,
        "latency_is_benchmark": False,
        "dynamic_block_count": len(block_audits),
        "blocks": block_audits,
        "limitations": [
            "Outer YOLO graph is PyTorch, not exported.",
            (
                "All routed-block components cross CPU NumPy and run in ORT."
                if args.bridge_mode == "split-ort"
                else "ORT router crosses CPU NumPy; checkpoint experts execute conditionally in PyTorch."
            ),
            "Bridge is not zero-copy and its latency is not a deployment benchmark.",
            "Single unwarmed latency values are diagnostics, not performance claims.",
            "Full VisDrone mAP and TensorRT GPU measurements are not included.",
        ],
    }
    args.output_json.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_json.resolve().write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
