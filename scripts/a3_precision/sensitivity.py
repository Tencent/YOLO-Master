"""Per-group INT8 ablation, route drift correlation, and automatic fallback selection."""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .common import json_dump, letterbox_image, sha256_file
from .manifest import HarnessConfig, ModelSpec
from .onnx_backend import discover_sensitivity_groups, instrument_topk_outputs, quantize_static_onnx
from .routing import (
    RouteCapture,
    compare_route_tensors,
    normalize_router_output,
    simulated_precision,
    weighted_layer_summary,
)
from .validation import validate_accuracy


def _load_yolo_model(spec: ModelSpec, device: torch.device):
    from ultralytics import YOLO

    yolo = YOLO(str(spec.checkpoint))
    if spec.adapter is not None and not yolo.load_adapters(spec.adapter):
        raise RuntimeError(f"failed to load adapter {spec.adapter}")
    return yolo, yolo.model.eval().to(device).float()


def measure_route_drift(
    spec: ModelSpec,
    *,
    images: list[Path],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Fake-quantize one router at a time and compare it with FP32 routes."""
    requested = str(settings.get("route_device", settings.get("device", "cpu")))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("route sensitivity requested CUDA but torch.cuda.is_available() is false")
    device = torch.device(requested)
    _, model = _load_yolo_model(spec, device)
    imgsz = int(settings.get("imgsz", 640))
    baseline: dict[str, dict[str, Any]] = {}
    # Hybrid families can contain more than one router implementation (for
    # example MoT neck + MoE backbone). Capture every routed leaf so "per
    # layer" means the complete live checkpoint, not only name-matched layers.
    with RouteCapture(model) as capture:
        with torch.inference_mode():
            for path in images:
                capture.clear()
                model(torch.from_numpy(letterbox_image(path, imgsz)).to(device))
                baseline[str(path)] = dict(capture.current)
        layer_names = tuple(capture.layers)

    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for layer_name in layer_names:
        with RouteCapture(model) as capture:
            with simulated_precision(
                model,
                "int8",
                scope="router_only",
                routed={layer_name: capture.layers[layer_name]},
                per_channel=bool(settings.get("route_per_channel", True)),
            ):
                with torch.inference_mode():
                    for sample_index, path in enumerate(images):
                        capture.clear()
                        model(torch.from_numpy(letterbox_image(path, imgsz)).to(device))
                        if layer_name not in capture.current:
                            raise RuntimeError(f"route layer {layer_name} did not execute for {path}")
                        record = {
                            "sample_index": sample_index,
                            "image": str(path),
                            "layer": layer_name,
                            **compare_route_tensors(baseline[str(path)][layer_name], capture.current[layer_name]),
                        }
                        records[layer_name].append(record)
    return {
        "methodology": "one-router INT8 weight quantize-dequantize; FP32 kernels and activations",
        "images": len(images),
        "layers": {layer: weighted_layer_summary(rows) for layer, rows in records.items()},
        "per_sample_comparisons": [row for layer in sorted(records) for row in records[layer]],
    }


def measure_onnx_route_drift(
    fp32_path: Path,
    int8_path: Path,
    *,
    images: list[Path],
    settings: dict[str, Any],
    work_dir: Path,
) -> dict[str, Any]:
    """Compare actual routing TopK outputs from FP32 and full-INT8 ONNX graphs."""
    import onnxruntime as ort

    work_dir.mkdir(parents=True, exist_ok=True)
    fp32_probe = work_dir / "fp32.routing_outputs.onnx"
    int8_probe = work_dir / "full_int8.routing_outputs.onnx"
    try:
        fp32_specs = instrument_topk_outputs(fp32_path, fp32_probe)
        int8_specs = instrument_topk_outputs(int8_path, int8_probe)
        layer_names = sorted(set(fp32_specs) & set(int8_specs))
        if not layer_names:
            raise RuntimeError("FP32 and INT8 ONNX graphs have no common routing TopK nodes")
        configured = list(settings.get("ort_providers", ["CPUExecutionProvider"]))
        available = set(ort.get_available_providers())
        providers = [item for item in configured if (item[0] if isinstance(item, (list, tuple)) else item) in available]
        if not providers:
            raise RuntimeError(f"none of the requested ORT providers are available: {configured}; {sorted(available)}")
        fp32_session = ort.InferenceSession(str(fp32_probe), providers=providers)
        int8_session = ort.InferenceSession(str(int8_probe), providers=providers)
        fp32_input = fp32_session.get_inputs()[0].name
        int8_input = int8_session.get_inputs()[0].name
        records: dict[str, list[dict[str, Any]]] = defaultdict(list)
        imgsz = int(settings.get("imgsz", 640))
        fp32_output_names = list(
            dict.fromkeys(
                output
                for layer in layer_names
                for output in (fp32_specs[layer]["values_output"], fp32_specs[layer]["indices_output"])
                if output is not None
            )
        )
        int8_output_names = list(
            dict.fromkeys(
                output
                for layer in layer_names
                for output in (int8_specs[layer]["values_output"], int8_specs[layer]["indices_output"])
                if output is not None
            )
        )
        for sample_index, path in enumerate(images):
            tensor = letterbox_image(path, imgsz)
            fp32_outputs = dict(zip(fp32_output_names, fp32_session.run(fp32_output_names, {fp32_input: tensor})))
            int8_outputs = dict(zip(int8_output_names, int8_session.run(int8_output_names, {int8_input: tensor})))
            for layer_name in layer_names:
                ref_spec = fp32_specs[layer_name]
                new_spec = int8_specs[layer_name]
                ref_values = fp32_outputs[ref_spec["values_output"]]
                new_values = int8_outputs[new_spec["values_output"]]
                ref_indices_name = ref_spec["indices_output"]
                new_indices_name = new_spec["indices_output"]
                if (ref_indices_name is None) != (new_indices_name is None):
                    raise RuntimeError(f"routing representation changed after INT8 quantization: {layer_name}")
                reference_output = (
                    torch.from_numpy(ref_values)
                    if ref_indices_name is None
                    else (torch.from_numpy(ref_values), torch.from_numpy(fp32_outputs[ref_indices_name]))
                )
                candidate_output = (
                    torch.from_numpy(new_values)
                    if new_indices_name is None
                    else (torch.from_numpy(new_values), torch.from_numpy(int8_outputs[new_indices_name]))
                )
                reference = normalize_router_output(
                    reference_output,
                    num_experts=int(ref_spec["num_experts"]),
                    top_k=int(ref_spec["top_k"]),
                )
                candidate = normalize_router_output(
                    candidate_output,
                    num_experts=int(new_spec["num_experts"]),
                    top_k=int(new_spec["top_k"]),
                )
                records[layer_name].append(
                    {
                        "sample_index": sample_index,
                        "image": str(path),
                        "layer": layer_name,
                        **compare_route_tensors(reference, candidate),
                    }
                )
        return {
            "methodology": (
                "actual FP32 vs full-INT8 ONNX routing intermediates: native TopK for sparse graphs; "
                "standardized Top-2 ranking over Router Softmax probabilities for dense graphs"
            ),
            "images": len(images),
            "providers_requested": providers,
            "providers_fp32": fp32_session.get_providers(),
            "providers_int8": int8_session.get_providers(),
            "fp32_sha256": sha256_file(fp32_path),
            "int8_sha256": sha256_file(int8_path),
            "routing_contracts": {layer: fp32_specs[layer] for layer in layer_names},
            "layers": {layer: weighted_layer_summary(rows) for layer, rows in records.items()},
            "per_sample_comparisons": [row for layer in sorted(records) for row in records[layer]],
        }
    finally:
        fp32_probe.unlink(missing_ok=True)
        int8_probe.unlink(missing_ok=True)


def _onnx_layer_path(layer_name: str) -> str:
    segments = []
    for token in layer_name.split("."):
        if token.isdigit() and segments:
            segments[-1] += f".{token}"
        else:
            segments.append(token)
    return "/" + "/".join(segments)


def _route_for_group(group: dict[str, Any], route_drift: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    group_id = str(group["group"]).lower()
    matches = []
    for layer_name, metrics in route_drift.get("layers", {}).items():
        path = _onnx_layer_path(layer_name).lower()
        if path in group_id or group_id.split("::", 1)[-1] in path:
            matches.append((len(path), layer_name, metrics))
    if not matches:
        return None, None
    _, layer_name, metrics = max(matches)
    return layer_name, metrics


def _onnx_route_for_group(
    group: dict[str, Any], route_drift: dict[str, Any]
) -> tuple[str | None, dict[str, Any] | None]:
    group_path = str(group["group"]).split("::", 1)[-1].lower().rstrip("/")
    matches = []
    for layer_name, metrics in route_drift.get("layers", {}).items():
        layer_path = str(layer_name).lower()
        layer_path = layer_path[:-5] if layer_path.endswith("/topk") else layer_path
        layer_path = layer_path.rstrip("/")
        if group_path in layer_path or layer_path in group_path:
            matches.append((len(layer_path), layer_name, metrics))
    if not matches:
        return None, None
    _, layer_name, metrics = max(matches)
    return layer_name, metrics


def _route_component(metrics: dict[str, Any] | None) -> float:
    if not metrics:
        return 0.0
    values = (
        metrics.get("topk_exact_rate"),
        metrics.get("topk_ordered_exact_rate"),
        metrics.get("top1_flip_rate"),
    )
    if not all(isinstance(value, (int, float)) for value in values):
        return 0.0
    topk_exact, ordered_topk_exact, top1_flip = (float(value) for value in values)
    return 0.4 * (1.0 - topk_exact) + 0.3 * (1.0 - ordered_topk_exact) + 0.3 * top1_flip


def _safe_name(group_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", group_id).strip("_")[:80]
    digest = hashlib.sha256(group_id.encode()).hexdigest()[:10]
    return f"{stem}_{digest}" if stem else digest


def _correlation(x: list[float], y: list[float]) -> dict[str, Any]:
    finite = [(a, b) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)]
    if len(finite) < 2:
        return {"samples": len(finite), "pearson": None, "spearman": None, "reason": "fewer_than_two_layers"}
    xa, ya = (np.asarray(values, dtype=np.float64) for values in zip(*finite))
    if float(xa.std()) == 0.0 or float(ya.std()) == 0.0:
        return {"samples": len(finite), "pearson": None, "spearman": None, "reason": "constant_input"}
    pearson = float(np.corrcoef(xa, ya)[0, 1])
    try:
        from scipy.stats import spearmanr

        spearman = float(spearmanr(xa, ya).statistic)
    except ImportError:
        spearman = float(np.corrcoef(np.argsort(np.argsort(xa)), np.argsort(np.argsort(ya)))[0, 1])
    return {"samples": len(finite), "pearson": pearson, "spearman": spearman, "reason": None}


def _candidate_groups(groups: dict[str, dict[str, Any]], settings: dict[str, Any]) -> list[dict[str, Any]]:
    roles = set(settings.get("roles", ("router", "topk", "attention", "matmul")))
    rows = [row for row in groups.values() if row["role"] in roles and row["node_count"] > 0]
    priority = {"router": 0, "topk": 1, "attention": 2, "matmul": 3}
    rows.sort(key=lambda row: (priority.get(row["role"], 99), row["group"]))
    max_groups = int(settings.get("max_groups", 0))
    return rows[:max_groups] if max_groups > 0 else rows


def write_validation_subset(data_yaml: Path, images: list[Path], output_dir: Path) -> Path:
    """Create a deterministic Ultralytics data YAML for an exact locked image list."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_list = output_dir / "validation_images.txt"
    image_list.write_text("\n".join(str(path) for path in images) + "\n", encoding="utf-8")
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid dataset YAML: {data_yaml}")
    dataset_root = Path(str(data.get("path", data_yaml.parent)))
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()
    data["path"] = str(dataset_root)
    data["val"] = str(image_list.resolve())
    target = output_dir / "sensitivity_data.yaml"
    target.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return target


def analyze_sensitivity(
    config: HarnessConfig,
    spec: ModelSpec,
    lock: dict[str, Any],
    pre_auto_comparison: dict[str, Any],
) -> dict[str, Any]:
    """Quantize each structural group alone, validate mAP, correlate, and select fallbacks."""
    experiment = config.raw.get("experiment", {})
    quant = config.raw.get("quantization", {})
    sensitivity = config.raw.get("sensitivity", {})
    family_dir = config.output_dir / spec.family
    artifact_dir = family_dir / "artifacts"
    probe_dir = family_dir / "sensitivity_probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = artifact_dir / "fp32.onnx"
    groups = discover_sensitivity_groups(fp32_path)
    candidates = _candidate_groups(groups, sensitivity)
    validation_images = [Path(path) for path in lock["dataset"]["validation"]["images"]]
    route_count = min(int(sensitivity.get("route_samples", 32)), len(validation_images))
    onnx_route_drift = measure_onnx_route_drift(
        fp32_path,
        artifact_dir / "full_int8.onnx",
        images=validation_images[:route_count],
        settings={**experiment, **sensitivity},
        work_dir=probe_dir / "routing_outputs",
    )
    json_dump(family_dir / "route_drift.int8_per_layer.json", onnx_route_drift)
    route_drift = measure_route_drift(
        spec, images=validation_images[:route_count], settings={**experiment, **sensitivity}
    )
    json_dump(family_dir / "route_drift.one_router_int8_per_layer.json", route_drift)

    full_baseline_map = (
        pre_auto_comparison.get("fp32", {}).get("accuracy_result", {}).get("accuracy", {}).get("map50_95")
    )
    if not isinstance(full_baseline_map, (int, float)) or not math.isfinite(float(full_baseline_map)):
        raise RuntimeError(f"{spec.family}: FP32 validation must succeed before sensitivity analysis")
    sensitivity_count = min(int(sensitivity.get("validation_samples", 128)), len(validation_images))
    sensitivity_data = write_validation_subset(
        config.dataset.data_yaml, validation_images[:sensitivity_count], probe_dir
    )
    subset_baseline_result = validate_accuracy(
        fp32_path,
        data_yaml=sensitivity_data,
        settings={**experiment, **sensitivity.get("validation", {})},
    )
    baseline_map = subset_baseline_result.get("accuracy", {}).get("map50_95")
    if not isinstance(baseline_map, (int, float)) or not math.isfinite(float(baseline_map)):
        raise RuntimeError(f"{spec.family}: sensitivity-subset FP32 validation failed: {subset_baseline_result}")
    calibration_images = [Path(path) for path in lock["dataset"]["calibration"]["images"]]
    sensitivity_calib = int(sensitivity.get("calibration_samples", len(calibration_images)))
    calibration_images = calibration_images[:sensitivity_calib]
    rows = []
    map_threshold = float(sensitivity.get("map_loss_pp_threshold", 0.10))
    topk_min = float(sensitivity.get("topk_exact_min", 0.99))
    ordered_topk_min = float(sensitivity.get("topk_ordered_exact_min", topk_min))
    top1_flip_max = float(sensitivity.get("top1_flip_max", 1.0 - topk_min))
    score_threshold = float(sensitivity.get("score_threshold", 0.50))
    route_weight = float(sensitivity.get("route_weight", 0.40))
    map_weight = float(sensitivity.get("map_weight", 0.60))

    for group in candidates:
        group_id = group["group"]
        target = probe_dir / f"{_safe_name(group_id)}.onnx"
        quant_result = quantize_static_onnx(
            fp32_path,
            target,
            calibration_images=calibration_images,
            imgsz=int(experiment.get("imgsz", 640)),
            settings=quant,
            nodes_to_quantize=list(group["nodes"]),
            label=f"sensitivity::{group_id}",
        )
        route_layer, route_metrics = _route_for_group(group, route_drift)
        full_route_layer, full_route_metrics = _onnx_route_for_group(group, onnx_route_drift)
        if quant_result["status"] == "success":
            accuracy_result = validate_accuracy(
                target,
                data_yaml=sensitivity_data,
                settings={**experiment, **sensitivity.get("validation", {})},
            )
        else:
            accuracy_result = {"status": "skipped", "reason": "quantization_failed"}
        probe_map = accuracy_result.get("accuracy", {}).get("map50_95")
        map_loss_pp = (
            float((baseline_map - probe_map) * 100.0)
            if isinstance(probe_map, (int, float)) and math.isfinite(float(probe_map))
            else None
        )
        route_component = max(_route_component(route_metrics), _route_component(full_route_metrics))
        map_component = min(max(float(map_loss_pp or 0.0), 0.0) / max(map_threshold, 1e-12), 1.0)
        score = map_weight * map_component + route_weight * route_component
        failed = quant_result["status"] != "success" or accuracy_result["status"] not in {"success"}
        reasons = []
        if failed:
            reasons.append("probe_failed_conservative_fallback")
        if map_loss_pp is not None and map_loss_pp >= map_threshold:
            reasons.append("map_loss")
        for source, metrics in (("isolated", route_metrics), ("full_int8", full_route_metrics)):
            if not metrics:
                continue
            if metrics.get("topk_exact_rate", 1.0) < topk_min:
                reasons.append(f"{source}_route_topk_drift")
            if metrics.get("topk_ordered_exact_rate", 1.0) < ordered_topk_min:
                reasons.append(f"{source}_route_topk_order_drift")
            if metrics.get("top1_flip_rate", 0.0) > top1_flip_max:
                reasons.append(f"{source}_route_top1_flip")
        if score >= score_threshold:
            reasons.append("combined_score")
        row = {
            **group,
            "route_layer": route_layer,
            "route_metrics": route_metrics,
            "full_int8_route_layer": full_route_layer,
            "full_int8_route_metrics": full_route_metrics,
            "fp32_map50_95": float(baseline_map),
            "probe_map50_95": float(probe_map) if isinstance(probe_map, (int, float)) else None,
            "map_loss_percentage_points": map_loss_pp,
            "route_component": route_component,
            "map_component": map_component,
            "sensitivity_score": score,
            "sensitive": bool(reasons),
            "selection_reasons": reasons,
            "quantization": quant_result,
            "validation": accuracy_result,
            "probe_artifact_retained": bool(sensitivity.get("keep_probe_models", False)),
        }
        rows.append(row)
        if not bool(sensitivity.get("keep_probe_models", False)):
            target.unlink(missing_ok=True)

    correlation_rows = [
        row
        for row in rows
        if row["route_metrics"] is not None and isinstance(row["map_loss_percentage_points"], (int, float))
    ]
    map_losses = [float(row["map_loss_percentage_points"]) for row in correlation_rows]
    correlation_inputs = {
        "topk_set_drift": [1.0 - float(row["route_metrics"]["topk_exact_rate"]) for row in correlation_rows],
        "topk_ordered_drift": [
            1.0 - float(row["route_metrics"]["topk_ordered_exact_rate"]) for row in correlation_rows
        ],
        "token_flip_rate": [float(row["route_metrics"]["token_flip_rate"]) for row in correlation_rows],
        "top1_flip_rate": [float(row["route_metrics"]["top1_flip_rate"]) for row in correlation_rows],
        "total_variation": [float(row["route_metrics"]["total_variation"]) for row in correlation_rows],
    }
    correlations = {name: _correlation(values, map_losses) for name, values in correlation_inputs.items()}
    correlation = dict(correlations["topk_ordered_drift"])
    correlation["x"] = "1 - ordered Top-K exact agreement from one-router INT8 simulation"
    correlation["y"] = "mAP50-95 loss (percentage points) from one-group real ORT INT8 PTQ"
    selected = [row["group"] for row in rows if row["sensitive"]]
    result = {
        "schema_version": 1,
        "family": spec.family,
        "methodology": {
            "accuracy_ablation": "only the named ONNX group is statically INT8-quantized",
            "route_ablation": route_drift["methodology"],
            "full_int8_route_evidence": onnx_route_drift["methodology"],
            "fallback_precision": "FP32 for selected nodes; remaining quantizable nodes INT8",
            "failed_probe_policy": "conservative FP32 fallback",
            "probe_artifact_retention": bool(sensitivity.get("keep_probe_models", False)),
            "sensitivity_validation_images": sensitivity_count,
            "full_validation_images": len(validation_images),
        },
        "full_fp32_map50_95": float(full_baseline_map),
        "subset_fp32_validation": subset_baseline_result,
        "thresholds": {
            "map_loss_pp_threshold": map_threshold,
            "topk_exact_min": topk_min,
            "topk_ordered_exact_min": ordered_topk_min,
            "top1_flip_max": top1_flip_max,
            "score_threshold": score_threshold,
            "map_weight": map_weight,
            "route_weight": route_weight,
        },
        "candidate_groups": len(rows),
        "selected_groups": selected,
        "correlation": correlation,
        "correlations": correlations,
        "groups": rows,
    }
    json_dump(family_dir / "sensitivity.json", result)
    return result
