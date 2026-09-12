"""Run an honest full-VisDrone mAP gate for ORT routing with checkpoint MoT experts.

The complete YOLO graph and experts stay in PyTorch (CUDA when requested).
Only exported router subgraphs run in ONNX Runtime on the host. This validates
full-model accuracy and conditional execution, but it is not TensorRT and its
latency must not be presented as an end-to-end deployment benchmark.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_dynamic_blocks_from_checkpoint import load_model, sha256
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.modules.dynamic_runtime import ORTRouterTorchExpertAdapter
from validate_full_yolo_hybrid_dynamic_runtime import replace_submodule


def disable_thop_flop_profiling() -> bool:
    """Disable optional THOP FLOPs accounting for this accuracy-only validator.

    THOP registers temporary forward hooks and ``total_ops`` buffers. Some
    PyTorch/THOP combinations can leave a counting hook behind after removing
    the buffer, which later crashes conditional expert execution when that
    expert is selected. FLOPs are not part of this validation gate, so avoiding
    THOP is both safer and semantically neutral for predictions and mAP.
    """
    try:
        import thop
    except ImportError:
        return False

    def disabled_profile(*_args: Any, **kwargs: Any):
        return (0.0, 0.0, {}) if kwargs.get("ret_layer_info") else (0.0, 0.0)

    thop.profile = disabled_profile
    return True


def strip_thop_runtime_state(model: torch.nn.Module) -> int:
    """Remove only THOP-owned hooks and temporary buffers from a model."""
    removed_hooks = 0
    for module in model.modules():
        for hook_store_name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks"):
            hook_store = getattr(module, hook_store_name, None)
            if hook_store is None:
                continue
            for hook_id, hook in list(hook_store.items()):
                hook_function = getattr(hook, "func", hook)
                hook_module = str(getattr(hook_function, "__module__", ""))
                if hook_module == "thop" or hook_module.startswith("thop."):
                    del hook_store[hook_id]
                    removed_hooks += 1
        module._buffers.pop("total_ops", None)
        module._buffers.pop("total_params", None)
    return removed_hooks


def load_validation_model(checkpoint: Path) -> tuple[torch.nn.Module, int]:
    """Load a checkpoint model and sanitize optional THOP instrumentation."""
    model = load_model(checkpoint, torch.device("cpu"))
    return model, strip_thop_runtime_state(model)


def preflight_onnxruntime() -> dict[str, Any]:
    """Fail before expensive eager validation if the ORT router cannot run."""
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise ImportError(
            "onnxruntime is required before validation starts. Install it into the same interpreter with: "
            f"{sys.executable} -m pip install onnxruntime==1.23.2"
        ) from error

    providers = list(ort.get_available_providers())
    if "CPUExecutionProvider" not in providers:
        raise RuntimeError(
            "onnxruntime is installed but CPUExecutionProvider is unavailable; "
            f"available providers: {providers}"
        )
    return {
        "python_executable": sys.executable,
        "onnxruntime_version": ort.__version__,
        "available_providers": providers,
        "selected_router_provider": "CPUExecutionProvider",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Real trained MoT .pt checkpoint")
    parser.add_argument("model_manifest", type=Path, help="dynamic_model_blocks.json from the same checkpoint")
    parser.add_argument("data", type=Path, help="VisDrone data.yaml")
    parser.add_argument("output_json", type=Path, help="Final auditable result JSON")
    parser.add_argument("--device", default="0", help="CUDA device for full YOLO and checkpoint experts")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--map-tolerance-pct-points",
        type=float,
        default=0.5,
        help="Maximum absolute mAP50-95 difference in percentage points",
    )
    parser.add_argument(
        "--require-exact-route",
        action="store_true",
        help="Fail if any ORT-router Top-K location differs from the eager checkpoint router",
    )
    parser.add_argument(
        "--skip-route-drift-audit",
        action="store_true",
        help="Skip duplicate eager-router calls; disables the exact-route gate",
    )
    parser.add_argument("--project", type=Path, help="Optional Ultralytics validation output directory")
    return parser.parse_args()


def _json_floats(values: dict) -> dict[str, float]:
    return {str(key): float(value) for key, value in values.items()}


def run_validation(
    model: torch.nn.Module,
    *,
    checkpoint: Path,
    data: Path,
    device: str,
    imgsz: int,
    batch: int,
    workers: int,
    project: Path,
    name: str,
    adapters: list[tuple[str, ORTRouterTorchExpertAdapter]] | None = None,
) -> dict:
    args = {
        "model": str(checkpoint),
        "data": str(data),
        "task": "detect",
        "mode": "val",
        "split": "val",
        "device": device,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        # Static router artifacts were captured at square 640x640. Rectangular
        # validation would change intermediate H/W and violate their manifest.
        "rect": False,
        "quantize": None,
        "plots": False,
        "save_json": False,
        "save_txt": False,
        "verbose": True,
        "project": str(project),
        "name": name,
        "exist_ok": True,
    }
    validator = DetectionValidator(args=args)
    if adapters:
        # AutoBackend warms up the model before on_val_start. Exclude those
        # synthetic zero-input calls from the 548-image execution evidence.
        def reset_adapter_evidence(_validator) -> None:
            for _, adapter in adapters:
                adapter.reset_execution_summary()

        validator.add_callback("on_val_start", reset_adapter_evidence)
    validator(model=model)
    return {
        "images": int(validator.seen),
        "metrics": _json_floats(validator.metrics.results_dict),
        "speed_ms_per_image": _json_floats(validator.speed),
        "save_dir": str(validator.save_dir),
        "rect": False,
        "imgsz": imgsz,
        "batch": batch,
        "device": str(validator.device),
    }


def install_adapters(
    model: torch.nn.Module,
    *,
    model_manifest: dict,
    model_manifest_path: Path,
    compare_eager_routes: bool,
) -> list[tuple[str, ORTRouterTorchExpertAdapter]]:
    adapters = []
    for record in model_manifest["blocks"]:
        name = record["module_name"]
        original_block = model.get_submodule(name)
        bundle_manifest = model_manifest_path.parent / record["bundle_manifest"]
        adapter = ORTRouterTorchExpertAdapter(
            original_block,
            bundle_manifest,
            providers=["CPUExecutionProvider"],
            compare_eager_routes=compare_eager_routes,
        ).eval()
        replace_submodule(model, name, adapter)
        adapters.append((name, adapter))
    return adapters


def main() -> int:
    args = parse_args()
    thop_flop_profiling_disabled = disable_thop_flop_profiling()
    checkpoint = args.checkpoint.resolve()
    model_manifest_path = args.model_manifest.resolve()
    data = args.data.resolve()
    output_json = args.output_json.resolve()
    for required in (checkpoint, model_manifest_path, data):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.imgsz < 1 or args.batch < 1 or args.workers < 0:
        raise ValueError("--imgsz/--batch must be positive and --workers must be nonnegative")
    if args.map_tolerance_pct_points < 0:
        raise ValueError("--map-tolerance-pct-points must be nonnegative")
    if args.require_exact_route and args.skip_route_drift_audit:
        raise ValueError("--require-exact-route cannot be combined with --skip-route-drift-audit")

    # The eager checkpoint does not need ORT, but phase 2 does. Check ORT now
    # so a missing package fails immediately instead of after all 548 eager images.
    ort_preflight = preflight_onnxruntime()
    print(f"[preflight] Python: {ort_preflight['python_executable']}")
    print(f"[preflight] ONNX Runtime: {ort_preflight['onnxruntime_version']}")
    print(f"[preflight] providers: {ort_preflight['available_providers']}")

    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if model_manifest.get("scope") != "routed_blocks_only":
        raise ValueError("model manifest does not describe routed checkpoint blocks")
    if model_manifest.get("status") == "FAILED":
        raise ValueError("refusing a failed dynamic-block manifest")
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != model_manifest["checkpoint"]["sha256"]:
        raise ValueError("checkpoint SHA256 does not match dynamic model manifest")
    capture_shape = model_manifest.get("capture_input_shape", [])
    if len(capture_shape) != 4 or capture_shape[-2:] != [args.imgsz, args.imgsz]:
        raise ValueError(
            f"manifest was captured at {capture_shape[-2:]}; requested validation is {args.imgsz}x{args.imgsz}"
        )

    project = (args.project or output_json.parent / "dynamic_mot_val_runs").resolve()
    project.mkdir(parents=True, exist_ok=True)

    print("===== 1/2 EAGER CHECKPOINT FP32 VALIDATION =====")
    eager_model, eager_thop_hooks_removed = load_validation_model(checkpoint)
    if eager_thop_hooks_removed:
        print(f"[safety] removed {eager_thop_hooks_removed} stale THOP hooks from eager model")
    eager = run_validation(
        eager_model,
        checkpoint=checkpoint,
        data=data,
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=project,
        name="eager_checkpoint_fp32",
    )
    del eager_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("===== 2/2 ORT ROUTER + PYTORCH CHECKPOINT EXPERTS VALIDATION =====")
    hybrid_model, hybrid_thop_hooks_removed = load_validation_model(checkpoint)
    if hybrid_thop_hooks_removed:
        print(f"[safety] removed {hybrid_thop_hooks_removed} stale THOP hooks from hybrid model")
    adapters = install_adapters(
        hybrid_model,
        model_manifest=model_manifest,
        model_manifest_path=model_manifest_path,
        compare_eager_routes=not args.skip_route_drift_audit,
    )
    hybrid = run_validation(
        hybrid_model,
        checkpoint=checkpoint,
        data=data,
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=project,
        name="ort_router_pytorch_checkpoint_experts",
        adapters=adapters,
    )

    block_summaries = []
    for name, adapter in adapters:
        summary = adapter.execution_summary()
        if summary["calls"] <= 0 or not summary["checkpoint_pytorch_experts_executed"]:
            raise RuntimeError(f"dynamic adapter did not execute checkpoint experts: {name}")
        if summary["onnx_expert_sessions_loaded"]:
            raise RuntimeError(f"router-only adapter unexpectedly loaded ONNX experts: {name}")
        block_summaries.append({"module_name": name, **summary})

    eager_map = eager["metrics"]["metrics/mAP50-95(B)"]
    hybrid_map = hybrid["metrics"]["metrics/mAP50-95(B)"]
    delta_pct_points = (hybrid_map - eager_map) * 100.0
    precision_gate = abs(delta_pct_points) <= args.map_tolerance_pct_points
    total_route_mismatches = sum(item.get("route_location_mismatch_count", 0) for item in block_summaries)
    total_route_locations = sum(item.get("route_location_total", 0) for item in block_summaries)
    route_drift_observed = total_route_mismatches > 0
    route_gate = not args.require_exact_route or not route_drift_observed
    conditional_observed = any(item["sample_pair_reduction_ratio"] > 0.0 for item in block_summaries)
    dynamic_gate = conditional_observed and all(item["calls"] > 0 for item in block_summaries)
    passed = precision_gate and route_gate and dynamic_gate and eager["images"] >= 500 and hybrid["images"] >= 500
    if not passed:
        status = "FAIL"
    elif route_drift_observed:
        status = "PASS_WITH_ROUTE_DRIFT"
    else:
        status = "PASS"

    result = {
        "schema_version": 1,
        "status": status,
        "scope": "full_yolo_visdrone_ort_router_pytorch_checkpoint_experts",
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_hash},
        "model_manifest": {"path": str(model_manifest_path), "status": model_manifest.get("status")},
        "data": str(data),
        "execution_semantics": "host_ort_router_conditional_pytorch_checkpoint_experts",
        "runtime_preflight": ort_preflight,
        "masked_dense_allowed": False,
        "not_a_full_export": True,
        "not_tensorrt": True,
        "latency_is_deployment_benchmark": False,
        "validation_safety": {
            "thop_flop_profiling_disabled": thop_flop_profiling_disabled,
            "stale_thop_hooks_removed": {
                "eager_model": eager_thop_hooks_removed,
                "hybrid_model": hybrid_thop_hooks_removed,
            },
            "prediction_or_metric_semantics_changed": False,
        },
        "eager_fp32": eager,
        "dynamic_hybrid_fp32": hybrid,
        "accuracy_gate": {
            "eager_mAP50_95": eager_map,
            "dynamic_mAP50_95": hybrid_map,
            "dynamic_minus_eager_percentage_points": delta_pct_points,
            "absolute_difference_percentage_points": abs(delta_pct_points),
            "tolerance_percentage_points": args.map_tolerance_pct_points,
            "passed": precision_gate,
        },
        "route_gate": {
            "audit_enabled": not args.skip_route_drift_audit,
            "exact_route_required": args.require_exact_route,
            "mismatched_locations": total_route_mismatches,
            "total_locations": total_route_locations,
            "mismatch_ratio": total_route_mismatches / total_route_locations if total_route_locations else 0.0,
            "passed": route_gate,
        },
        "dynamic_execution_gate": {
            "all_blocks_executed": all(item["calls"] > 0 for item in block_summaries),
            "at_least_one_block_observed_sample_pair_reduction": conditional_observed,
            "onnx_expert_sessions_loaded": 0,
            "checkpoint_pytorch_experts_executed": True,
            "passed": dynamic_gate,
        },
        "blocks": block_summaries,
        "limitations": [
            "The outer YOLO graph and experts remain PyTorch; this is not a complete exported model.",
            "ORT router tensors cross CPU/NumPy and synchronize with CUDA experts; timing is diagnostic only.",
            "This run does not provide TensorRT plugin correctness or end-to-end acceleration evidence.",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"result: {output_json}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
