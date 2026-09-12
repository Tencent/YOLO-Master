"""Capture routed-block inputs from a checkpoint and export split dynamic-expert ONNX bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.nn.modules.dynamic_runtime import (
    ORTDynamicExpertRuntime,
    dispatch_numpy_experts,
    export_dynamic_expert_bundle,
    sparsify_topk_probabilities,
)
from ultralytics.nn.modules.moe.modules import ES_MOE
from ultralytics.nn.modules.mot.block import MoTBlock
from ultralytics.nn.modules.topk_contract import LEGACY_PRIORITY_BIAS_TOPK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Ultralytics .pt checkpoint")
    parser.add_argument("output_dir", type=Path, help="Directory for model-level dynamic bundles")
    parser.add_argument("--imgsz", type=int, default=640, help="Square capture input size")
    parser.add_argument("--batch", type=int, default=1, help="Capture batch size")
    parser.add_argument("--input-image", type=Path, help="Optional real image for 640x640 letterbox capture")
    parser.add_argument("--device", default="cpu", help="PyTorch capture/export device")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset")
    parser.add_argument("--family", choices=("all", "esmoe", "mot"), default="all")
    parser.add_argument("--validate", action="store_true", help="Compare each bundle with eager sparse execution")
    parser.add_argument(
        "--require-exact-route",
        action="store_true",
        help="Fail validation when ORT and eager select different experts at any routing location",
    )
    parser.add_argument("--atol", type=float, default=1e-4, help="Maximum absolute error accepted by --validate")
    parser.add_argument("--overwrite", action="store_true", help="Allow existing bundle files to be replaced")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        model = checkpoint.get("ema") or checkpoint.get("model")
    else:
        model = None
    if not isinstance(model, torch.nn.Module):
        raise TypeError("checkpoint does not contain a PyTorch model under 'ema' or 'model'")
    return model.float().eval().to(device)


def select_dynamic_blocks(model: torch.nn.Module, family: str) -> list[tuple[str, torch.nn.Module]]:
    allowed_types: tuple[type[torch.nn.Module], ...]
    if family == "esmoe":
        allowed_types = (ES_MOE,)
    elif family == "mot":
        allowed_types = (MoTBlock,)
    else:
        allowed_types = (ES_MOE, MoTBlock)
    return [(name, module) for name, module in model.named_modules() if isinstance(module, allowed_types)]


def capture_block_inputs(
    model: torch.nn.Module,
    blocks: list[tuple[str, torch.nn.Module]],
    sample: torch.Tensor,
) -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    handles = []

    for name, module in blocks:

        def capture(_module, args, block_name=name):
            if block_name not in captured:
                if not args or not isinstance(args[0], torch.Tensor):
                    raise TypeError(f"dynamic block {block_name} did not receive a tensor as its first argument")
                captured[block_name] = args[0].detach().clone()

        handles.append(module.register_forward_pre_hook(capture))

    try:
        with torch.no_grad():
            model(sample)
    finally:
        for handle in handles:
            handle.remove()

    missing = [name for name, _ in blocks if name not in captured]
    if missing:
        raise RuntimeError(f"model forward did not execute dynamic blocks: {missing}")
    return captured


def safe_block_name(name: str) -> str:
    return name.replace(".", "__").replace("/", "__").replace("\\", "__")


def make_capture_sample(
    *,
    input_image: Path | None,
    batch: int,
    imgsz: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    if input_image is None:
        sample = torch.zeros(batch, 3, imgsz, imgsz, dtype=torch.float32, device=device)
        return sample, {"kind": "synthetic_zero"}

    from PIL import Image

    image_path = input_image.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as opened:
        rgb = opened.convert("RGB")
        original_width, original_height = rgb.size
        scale = min(imgsz / original_height, imgsz / original_width)
        resized_width = max(1, round(original_width * scale))
        resized_height = max(1, round(original_height * scale))
        resized = rgb.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        resized_array = np.asarray(resized, dtype=np.uint8).copy()

    pad_left = (imgsz - resized_width) // 2
    pad_top = (imgsz - resized_height) // 2
    canvas = torch.full((3, imgsz, imgsz), 114.0 / 255.0, dtype=torch.float32)
    resized_tensor = torch.from_numpy(resized_array).permute(2, 0, 1).float().div_(255.0)
    canvas[:, pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized_tensor
    sample = canvas.unsqueeze(0).repeat(batch, 1, 1, 1).to(device)
    source = {
        "kind": "letterboxed_image",
        "path": str(image_path),
        "bytes": image_path.stat().st_size,
        "sha256": sha256(image_path),
        "original_shape_hw": [original_height, original_width],
        "resized_shape_hw": [resized_height, resized_width],
        "padding_left_top": [pad_left, pad_top],
        "fill_value": 114,
        "rgb": True,
        "normalized_0_1": True,
    }
    return sample, source


def primary_tensor(output) -> torch.Tensor:
    """Extract the routed feature tensor from block APIs that may also return auxiliary loss."""
    feature = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(feature, torch.Tensor):
        raise TypeError(f"dynamic block returned unsupported output type {type(output)}")
    return feature


def component_parity_errors(
    module: torch.nn.Module,
    block_input: torch.Tensor,
    manifest_path: Path,
) -> tuple[dict, np.ndarray]:
    """Compare router, every expert, and postprocess subgraph against their eager modules."""
    import onnxruntime as ort

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    input_array = block_input.detach().cpu().numpy()

    def run(record: dict, feeds: dict) -> torch.Tensor:
        session = ort.InferenceSession(
            str(root / record["path"]),
            providers=["CPUExecutionProvider"],
        )
        result = session.run(record["outputs"], feeds)[0]
        return torch.from_numpy(result)

    artifacts = manifest["artifacts"]
    with torch.no_grad():
        if isinstance(module, ES_MOE):
            eager_router = module.routing(block_input)
            pooled = module.routing.global_pool(block_input)
            logits = module.routing.routing_network(pooled)
            eager_dense_router = torch.softmax(logits.float().clamp(-30.0, 30.0), dim=1).type_as(block_input)
            eager_dense_router = eager_dense_router.repeat(1, 1, block_input.shape[2], block_input.shape[3])
        else:
            eager_router, _, logits = module.router(block_input, return_logits=True)
            eager_dense_router = torch.softmax(logits / module.router.temperature.float(), dim=1).to(
                dtype=block_input.dtype
            )
        ort_dense_router = run(artifacts["router"], {"x": input_array})
        ort_router = ort_dense_router
        if manifest.get("router_output_semantics") == "dense_probabilities_host_topk":
            ort_router = torch.from_numpy(
                sparsify_topk_probabilities(
                    ort_router.numpy(),
                    top_k=int(manifest["top_k"]),
                    zero_tolerance=float(manifest["zero_tolerance"]),
                    tie_tolerance=float(manifest.get("host_topk_tie_tolerance", 0.0)),
                    tie_break=str(manifest.get("host_topk_tie_break", LEGACY_PRIORITY_BIAS_TOPK)),
                )
            )
        router_error = float((eager_router.cpu() - ort_router).abs().max())
        dense_router_error = float((eager_dense_router.cpu() - ort_dense_router).abs().max())
        eager_route_mask = eager_router.cpu().numpy() > float(manifest["zero_tolerance"])
        ort_route_mask = ort_router.numpy() > float(manifest["zero_tolerance"])
        route_mask_mismatch = eager_route_mask != ort_route_mask
        route_location_mismatch = route_mask_mismatch.any(axis=1)

        expert_errors = []
        eager_expert_outputs = []
        for expert, record in zip(module.experts, artifacts["experts"]):
            eager_expert = expert(block_input).cpu()
            eager_expert_outputs.append(eager_expert)
            ort_expert = run(record, {"x": input_array})
            expert_errors.append(float((eager_expert - ort_expert).abs().max()))

        mixture = eager_expert_outputs[0]
        if isinstance(module, ES_MOE):
            eager_postprocess = module.norm(mixture.to(block_input.device)).cpu()
            postprocess_feeds = {"mixture": mixture.numpy()}
        else:
            mixture_on_device = mixture.to(block_input.device)
            eager_postprocess = (module.out_norm(module.out_proj(mixture_on_device)) + block_input).cpu()
            postprocess_feeds = {"x": input_array, "mixture": mixture.numpy()}
        ort_postprocess = run(artifacts["postprocess"], postprocess_feeds)
        postprocess_error = float((eager_postprocess - ort_postprocess).abs().max())

        def make_expert_runner(expert: torch.nn.Module):
            def run_expert(selected_input: np.ndarray) -> np.ndarray:
                selected_tensor = torch.from_numpy(selected_input).to(block_input.device)
                return expert(selected_tensor).detach().cpu().numpy()

            return run_expert

        deployment_mixture, _ = dispatch_numpy_experts(
            input_array,
            ort_router.numpy(),
            [make_expert_runner(expert) for expert in module.experts],
            top_k=int(manifest["top_k"]),
            routing_granularity=manifest["routing_granularity"],
            dynamic_threshold=float(manifest["dynamic_threshold"]),
            zero_tolerance=float(manifest["zero_tolerance"]),
        )
        deployment_mixture_tensor = torch.from_numpy(deployment_mixture).to(block_input.device)
        if isinstance(module, ES_MOE):
            deployment_reference = module.norm(deployment_mixture_tensor)
        else:
            deployment_reference = module.out_norm(module.out_proj(deployment_mixture_tensor)) + block_input

    metrics = {
        "router_max_abs_error": router_error,
        "dense_router_max_abs_error": dense_router_error,
        "route_mask_mismatch_count": int(route_mask_mismatch.sum()),
        "route_mask_total": int(route_mask_mismatch.size),
        "route_mask_mismatch_ratio": float(route_mask_mismatch.mean()),
        "route_location_mismatch_count": int(route_location_mismatch.sum()),
        "route_location_total": int(route_location_mismatch.size),
        "route_location_mismatch_ratio": float(route_location_mismatch.mean()),
        "expert_max_abs_errors": expert_errors,
        "postprocess_max_abs_error": postprocess_error,
    }
    return metrics, deployment_reference.detach().cpu().numpy()


def main() -> None:
    args = parse_args()
    if args.batch < 1 or args.imgsz < 1:
        raise ValueError("--batch and --imgsz must be positive")

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_manifest_path = output_dir / "dynamic_model_blocks.json"
    if model_manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{model_manifest_path} already exists; pass --overwrite or choose another directory")

    device = torch.device(args.device)
    model = load_model(checkpoint_path, device)
    blocks = select_dynamic_blocks(model, args.family)
    if not blocks:
        raise RuntimeError(f"no {args.family} dynamic expert blocks found in {checkpoint_path}")

    sample, capture_source = make_capture_sample(
        input_image=args.input_image,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
    )
    captured = capture_block_inputs(model, blocks, sample)

    block_records = []
    validation_failed = False
    route_drift_observed = False
    for index, (name, module) in enumerate(blocks):
        block_dir = output_dir / f"{index:02d}_{safe_block_name(name)}"
        manifest_path = export_dynamic_expert_bundle(
            module,
            captured[name],
            block_dir,
            opset=args.opset,
            overwrite=args.overwrite,
        )
        bundle = json.loads(manifest_path.read_text(encoding="utf-8"))
        block_record = {
            "index": index,
            "module_name": name,
            "module_type": bundle["module_type"],
            "family": bundle["family"],
            "input_shape": bundle["sample_shape"],
            "routing_granularity": bundle["routing_granularity"],
            "num_experts": bundle["num_experts"],
            "top_k": bundle["top_k"],
            "host_topk_tie_break": bundle["host_topk_tie_break"],
            "host_topk_tie_tolerance": bundle["host_topk_tie_tolerance"],
            "compute_reduction_guaranteed_per_sample": bundle[
                "compute_reduction_guaranteed_per_sample"
            ],
            "bundle_manifest": manifest_path.relative_to(output_dir).as_posix(),
        }
        if args.validate:
            block_input = captured[name]
            with torch.no_grad():
                eager_output = primary_tensor(module(block_input)).detach().cpu().numpy()
            runtime = ORTDynamicExpertRuntime(manifest_path, providers=["CPUExecutionProvider"])
            ort_output, audit = runtime.run(block_input.detach().cpu().numpy())
            component_errors, deployment_reference = component_parity_errors(module, block_input, manifest_path)
            eager_max_abs_error = float(abs(eager_output - ort_output).max())
            deployment_max_abs_error = float(abs(deployment_reference - ort_output).max())
            route_drift = component_errors["route_location_mismatch_count"] > 0
            deployment_parity_passed = deployment_max_abs_error <= args.atol
            strict_route_passed = not route_drift or not args.require_exact_route
            block_passed = deployment_parity_passed and strict_route_passed
            validation_failed |= not block_passed
            route_drift_observed |= route_drift
            if not block_passed:
                validation_status = "FAIL"
            elif route_drift:
                validation_status = "PASS_WITH_ROUTE_DRIFT"
            else:
                validation_status = "PASS"
            block_record["validation"] = {
                "status": validation_status,
                "max_abs_error": deployment_max_abs_error,
                "deployment_max_abs_error": deployment_max_abs_error,
                "eager_max_abs_error": eager_max_abs_error,
                "atol": args.atol,
                "require_exact_route": args.require_exact_route,
                "loaded_expert_ids": list(runtime.loaded_expert_ids),
                "audit": audit.to_dict(),
                "component_errors": component_errors,
            }
        block_records.append(block_record)
        print(
            f"[{index + 1}/{len(blocks)}] {name}: {bundle['routing_granularity']} "
            f"Top-{bundle['top_k']}/{bundle['num_experts']} -> {manifest_path}"
        )

    model_manifest = {
        "schema_version": 1,
        "status": (
            "FAILED"
            if validation_failed
            else ("VALIDATED_WITH_ROUTE_DRIFT" if route_drift_observed else "VALIDATED")
            if args.validate
            else "EXPORTED"
        ),
        "execution_semantics": "host_conditional_expert_dispatch",
        "masked_dense_allowed": False,
        "scope": "routed_blocks_only",
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256(checkpoint_path),
        },
        "capture_input_shape": [args.batch, 3, args.imgsz, args.imgsz],
        "capture_source": capture_source,
        "device": str(device),
        "opset": args.opset,
        "dynamic_block_count": len(block_records),
        "blocks": block_records,
        "limitation": (
            "This manifest exports real checkpoint routed blocks only. Full-model DAG partitioning and host "
            "orchestration are the next implementation stage."
        ),
    }
    model_manifest_path.write_text(
        json.dumps(model_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"model manifest: {model_manifest_path}")
    if validation_failed:
        raise RuntimeError("one or more dynamic block bundles failed the selected validation policy")


if __name__ == "__main__":
    main()
