#!/usr/bin/env python3
"""Measure FP32 versus INT8 expert-selection consistency for five A3 families.

The script is intentionally route-only: it reuses locked ONNX artifacts and
does not train, export, quantize, validate mAP, or benchmark latency. Results
are written incrementally so completed families can be resumed safely.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any
from unittest.mock import patch


RATE_FIELDS = (
    "topk_ordered_exact_rate",
    "topk_exact_rate",
    "jaccard_rate",
    "token_flip_rate",
    "top1_flip_rate",
    "sample_flip_rate",
    "sample_top1_flip_rate",
    "probability_mae",
    "total_variation",
    "jensen_shannon",
)


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _onnx_constant_scalar(node: Any, numpy_helper: Any) -> int:
    """Read an integer scalar emitted either as a tensor or scalar Constant."""
    if node is None or node.op_type != "Constant":
        return 0
    for attribute in node.attribute:
        if attribute.name == "value":
            values = numpy_helper.to_array(attribute.t).reshape(-1)
            return int(values[0]) if values.size else 0
        if attribute.name == "value_int":
            return int(attribute.i)
        if attribute.name == "value_ints" and attribute.ints:
            return int(attribute.ints[0])
    return 0


def instrument_routing_outputs_with_structural_fallback(
    source: Path,
    target: Path,
    original_instrument,
) -> dict[str, dict[str, Any]]:
    """Use named routing nodes first, then a constrained Softmax->TopK probe.

    PyTorch names MoLoRA's functional ``softmax`` and ``topk`` operations after
    their parent layer rather than the ``router`` child. The shared named-node
    detector therefore cannot see them. The fallback only admits TopK nodes
    fed directly by Softmax with a small, statically known expert axis, which
    excludes raw detection-head candidate selection.
    """
    try:
        return original_instrument(source, target)
    except RuntimeError as exc:
        if "no routing TopK or dense router Softmax outputs found" not in str(exc):
            raise

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.shape_inference.infer_shapes(onnx.load(str(source), load_external_data=False))
    graph = model.graph
    value_info = {item.name: item for item in (*graph.input, *graph.output, *graph.value_info)}
    initializers = {item.name: item for item in graph.initializer}
    producer = {output: node for node in graph.node for output in node.output}
    existing_outputs = {item.name for item in graph.output}
    specs = {}
    for node in graph.node:
        if node.op_type != "TopK" or len(node.input) < 2 or len(node.output) < 2:
            continue
        parent = producer.get(node.input[0])
        if parent is None or parent.op_type != "Softmax":
            continue
        attributes = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
        input_info = value_info.get(node.input[0])
        dimensions = list(input_info.type.tensor_type.shape.dim) if input_info is not None else []
        axis = int(attributes.get("axis", -1))
        if dimensions and axis < 0:
            axis += len(dimensions)
        num_experts = int(dimensions[axis].dim_value) if dimensions and 0 <= axis < len(dimensions) else 0
        k_initializer = initializers.get(node.input[1])
        top_k = int(numpy_helper.to_array(k_initializer).reshape(-1)[0]) if k_initializer is not None else 0
        if top_k <= 0:
            top_k = _onnx_constant_scalar(producer.get(node.input[1]), numpy_helper)
        if top_k <= 0:
            output_info = value_info.get(node.output[0])
            output_dimensions = list(output_info.type.tensor_type.shape.dim) if output_info is not None else []
            if output_dimensions and 0 <= axis < len(output_dimensions):
                top_k = int(output_dimensions[axis].dim_value)
        if not (2 <= num_experts <= 64 and 1 <= top_k <= num_experts):
            continue
        for index, output_name in enumerate(node.output[:2]):
            if output_name in existing_outputs:
                continue
            info = value_info.get(output_name)
            if info is not None:
                graph.output.append(deepcopy(info))
            else:
                element_type = TensorProto.FLOAT if index == 0 else TensorProto.INT64
                graph.output.append(helper.make_tensor_value_info(output_name, element_type, None))
            existing_outputs.add(output_name)
        node_name = node.name or f"TopK::{node.output[0]}"
        specs[node_name] = {
            "node_name": node_name,
            "values_output": node.output[0],
            "indices_output": node.output[1],
            "axis": axis,
            "top_k": top_k,
            "num_experts": num_experts,
            "representation": "sparse_topk",
            "top_k_source": "onnx_topk",
            "discovery": "structural_softmax_to_topk_small_expert_axis",
        }
    if not specs:
        raise RuntimeError(f"no named or structural routing outputs found in {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, str(target))
    return dict(sorted(specs.items()))


def gate_layer(metrics: dict[str, Any], thresholds: dict[str, float]) -> tuple[bool, list[str]]:
    reasons = []
    if float(metrics["topk_exact_rate"]) < thresholds["topk_exact_min"]:
        reasons.append("topk_set_agreement_below_threshold")
    if float(metrics["topk_ordered_exact_rate"]) < thresholds["topk_ordered_exact_min"]:
        reasons.append("topk_ordered_agreement_below_threshold")
    if float(metrics["top1_flip_rate"]) > thresholds["top1_flip_max"]:
        reasons.append("top1_flip_above_threshold")
    return not reasons, reasons


def summarize_layers(
    layers: dict[str, dict[str, Any]],
    routing_contracts: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    if not layers:
        raise ValueError("route measurement did not produce any layers")
    layer_gates = {}
    for layer_name, metrics in layers.items():
        passed, reasons = gate_layer(metrics, thresholds)
        layer_gates[layer_name] = {"passed": passed, "reasons": reasons}

    total_tokens = sum(int(row.get("tokens", 0)) for row in layers.values())
    weighted = {}
    for field in RATE_FIELDS:
        rows = [row for row in layers.values() if finite_number(row.get(field))]
        if not rows:
            continue
        weight_key = "samples" if field.startswith("sample_") else "tokens"
        denominator = sum(int(row.get(weight_key, 0)) for row in rows)
        weighted[field] = float(
            sum(float(row[field]) * int(row.get(weight_key, 0)) for row in rows) / max(denominator, 1)
        )

    representations = {}
    for layer_name in layers:
        representation = routing_contracts.get(layer_name, {}).get("representation", "unknown")
        representations[representation] = representations.get(representation, 0) + 1

    failed_layers = [name for name, row in layer_gates.items() if not row["passed"]]
    return {
        "layers": len(layers),
        "tokens": total_tokens,
        "representations": representations,
        "weighted": weighted,
        "worst_layer_metrics": {
            "topk_exact_rate": min(float(row["topk_exact_rate"]) for row in layers.values()),
            "topk_ordered_exact_rate": min(
                float(row["topk_ordered_exact_rate"]) for row in layers.values()
            ),
            "jaccard_rate": min(float(row["jaccard_rate"]) for row in layers.values()),
            "token_flip_rate": max(float(row["token_flip_rate"]) for row in layers.values()),
            "top1_flip_rate": max(float(row["top1_flip_rate"]) for row in layers.values()),
            "sample_flip_rate": max(float(row["sample_flip_rate"]) for row in layers.values()),
            "sample_top1_flip_rate": max(
                float(row["sample_top1_flip_rate"]) for row in layers.values()
            ),
            "total_variation": max(float(row["total_variation"]) for row in layers.values()),
        },
        "gate_passed": not failed_layers,
        "failed_layers": failed_layers,
        "layer_gates": layer_gates,
    }


def write_csv(path: Path, families: dict[str, dict[str, Any]]) -> None:
    fields = (
        "family",
        "status",
        "gate_passed",
        "layer",
        "representation",
        "top_k",
        "num_experts",
        "samples",
        "tokens",
        "topk_exact_rate",
        "topk_ordered_exact_rate",
        "jaccard_rate",
        "token_flip_rate",
        "top1_flip_rate",
        "sample_flip_rate",
        "sample_top1_flip_rate",
        "probability_mae",
        "total_variation",
        "jensen_shannon",
        "gate_reasons",
        "error_type",
        "error",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for family, result in families.items():
            measurement = result.get("measurement", {})
            contracts = measurement.get("routing_contracts", {})
            layers = measurement.get("layers", {})
            gates = result.get("summary", {}).get("layer_gates", {})
            if not layers:
                writer.writerow(
                    {
                        "family": family,
                        "status": result.get("status"),
                        "gate_passed": result.get("summary", {}).get("gate_passed"),
                        "error_type": result.get("error_type"),
                        "error": result.get("error"),
                    }
                )
                continue
            for layer_name, metrics in layers.items():
                contract = contracts.get(layer_name, {})
                gate = gates.get(layer_name, {})
                writer.writerow(
                    {
                        "family": family,
                        "status": result.get("status"),
                        "gate_passed": gate.get("passed"),
                        "layer": layer_name,
                        "representation": contract.get("representation"),
                        "top_k": contract.get("top_k"),
                        "num_experts": contract.get("num_experts"),
                        "samples": metrics.get("samples"),
                        "tokens": metrics.get("tokens"),
                        **{field: metrics.get(field) for field in RATE_FIELDS},
                        "gate_reasons": ";".join(gate.get("reasons", [])),
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--family", action="append", help="repeat to select more than one family")
    parser.add_argument("--samples", type=int, help="locked validation images; default comes from config")
    parser.add_argument("--topk-exact-min", type=float)
    parser.add_argument("--topk-ordered-exact-min", type=float)
    parser.add_argument("--top1-flip-max", type=float)
    parser.add_argument("--allow-runtime-drift", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.repo = args.repo.resolve()
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(args.repo))
    os.chdir(args.repo)
    args.output.mkdir(parents=True, exist_ok=True)

    from scripts.a3_precision import manifest as manifest_module
    from scripts.a3_precision.common import file_set_digest, sha256_file
    from scripts.a3_precision import sensitivity as sensitivity_module
    from scripts.a3_precision.onnx_backend import instrument_topk_outputs as original_instrument

    def instrument_for_this_run(source, target):
        return instrument_routing_outputs_with_structural_fallback(
            Path(source), Path(target), original_instrument
        )

    sensitivity_module.instrument_topk_outputs = instrument_for_this_run
    measure_onnx_route_drift = sensitivity_module.measure_onnx_route_drift

    config = manifest_module.load_config(args.config)
    lock_path = config.output_dir / "experiment.lock.json"
    recorded_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    locked_runtime = recorded_lock["environment"]["contract"]
    current_runtime = manifest_module._runtime_contract()
    runtime_drift = current_runtime != locked_runtime
    if runtime_drift and not args.allow_runtime_drift:
        raise RuntimeError("Runtime differs from export lock; pass --allow-runtime-drift to record it")

    print("Verifying locked config, weights, datasets, and selections...", flush=True)
    if runtime_drift:
        with patch.object(manifest_module, "_runtime_contract", return_value=locked_runtime):
            lock = manifest_module.verify_lock(config)
    else:
        lock = manifest_module.verify_lock(config)

    requested_families = set(args.family or ())
    selected = [
        spec for spec in config.models if spec.enabled and (not requested_families or spec.family in requested_families)
    ]
    found = {spec.family for spec in selected}
    if not selected or requested_families - found:
        raise ValueError(f"Requested family is absent or disabled: {sorted(requested_families - found)}")

    sensitivity = config.raw.get("sensitivity", {})
    experiment = config.raw.get("experiment", {})
    sample_count = args.samples if args.samples is not None else int(sensitivity.get("route_samples", 64))
    validation_images = [Path(item) for item in lock["dataset"]["validation"]["images"]]
    if sample_count <= 0 or sample_count > len(validation_images):
        raise ValueError(f"samples must be in [1, {len(validation_images)}], got {sample_count}")
    images = validation_images[:sample_count]
    selection_digest = file_set_digest(images, Path(lock["dataset"]["validation"]["source"]))
    thresholds = {
        "topk_exact_min": float(
            args.topk_exact_min if args.topk_exact_min is not None else sensitivity.get("topk_exact_min", 0.99)
        ),
        "topk_ordered_exact_min": float(
            args.topk_ordered_exact_min
            if args.topk_ordered_exact_min is not None
            else sensitivity.get("topk_ordered_exact_min", sensitivity.get("topk_exact_min", 0.99))
        ),
        "top1_flip_max": float(
            args.top1_flip_max if args.top1_flip_max is not None else sensitivity.get("top1_flip_max", 0.01)
        ),
    }
    if any(not 0.0 <= value <= 1.0 for value in thresholds.values()):
        raise ValueError(f"route thresholds must be within [0, 1]: {thresholds}")

    base_contract = {
        "task": "fp32_vs_full_int8_route_consistency",
        "experiment_lock_sha256": sha256_file(lock_path),
        "locked_validation_selection_digest": lock["dataset"]["validation"]["selection_digest"],
        "route_image_count": sample_count,
        "route_image_selection": "first N images from the immutable validation lock",
        "route_image_selection_digest": selection_digest,
        "imgsz": int(experiment.get("imgsz", 640)),
        "providers": ["CPUExecutionProvider"],
        "thresholds": thresholds,
        "current_runtime": current_runtime,
    }
    summary_path = args.output / "route_consistency_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {"status": "running", "families": {}}
    summary.update(
        schema_version=1,
        claim_boundary=(
            "Sparse TopK rows measure actual exported expert selections. Dense-probability rows are a standardized "
            "Top-2 ranking proxy and do not prove conditional expert execution. This run measures routing only."
        ),
        source_experiment=str(config.output_dir),
        experiment_lock_sha256=sha256_file(lock_path),
        script_sha256=sha256_file(Path(__file__)),
        runtime_contract_match=not runtime_drift,
        runtime_drift_allowed=bool(runtime_drift and args.allow_runtime_drift),
        locked_export_runtime=locked_runtime,
        current_runtime=current_runtime,
        selected_families=[spec.family for spec in selected],
        thresholds=thresholds,
        run_contract=base_contract,
    )
    save_json(summary_path, summary)

    settings = {**experiment, **sensitivity, "ort_providers": ["CPUExecutionProvider"]}
    for spec in selected:
        family = spec.family
        artifact_dir = config.output_dir / family / "artifacts"
        manifest_path = artifact_dir / "primary_variants.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = {"fp32": artifact_dir / "fp32.onnx", "full_int8": artifact_dir / "full_int8.onnx"}
        artifact_hashes = {}
        for variant, artifact in artifacts.items():
            actual_sha256 = sha256_file(artifact)
            expected_sha256 = manifest.get(variant, {}).get("artifact", {}).get("sha256")
            if manifest.get(variant, {}).get("status") != "success" or actual_sha256 != expected_sha256:
                raise RuntimeError(f"Artifact manifest mismatch: {artifact}")
            artifact_hashes[variant] = actual_sha256

        family_contract = {**base_contract, "family": family, "artifacts": artifact_hashes}
        family_contract_sha256 = digest_json(family_contract)
        result_path = args.output / family / "route_consistency.json"
        prior = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
        reusable = (
            prior
            and prior.get("status") == "success"
            and prior.get("run_contract_sha256") == family_contract_sha256
        )
        same_failed = (
            prior
            and prior.get("status") == "failed"
            and prior.get("run_contract_sha256") == family_contract_sha256
        )
        print(f"\n========== {family} ==========", flush=True)
        if reusable and not args.force:
            print("  reuse completed route result", flush=True)
            summary["families"][family] = prior
            continue
        if same_failed and not (args.retry_failed or args.force):
            print("  retain failed result (use --retry-failed to retry)", flush=True)
            summary["families"][family] = prior
            continue

        print(f"  comparing FP32 vs full INT8 routes on {sample_count} locked images", flush=True)
        try:
            measurement = measure_onnx_route_drift(
                artifacts["fp32"],
                artifacts["full_int8"],
                images=images,
                settings=settings,
                work_dir=args.output / family / "routing_probes",
            )
            compact = summarize_layers(measurement["layers"], measurement["routing_contracts"], thresholds)
            result = {
                "status": "success",
                "family": family,
                "run_contract": family_contract,
                "run_contract_sha256": family_contract_sha256,
                "summary": compact,
                "measurement": measurement,
            }
        except Exception as exc:
            result = {
                "status": "failed",
                "family": family,
                "run_contract": family_contract,
                "run_contract_sha256": family_contract_sha256,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        save_json(result_path, result)
        summary["families"][family] = result
        save_json(summary_path, summary)
        write_csv(args.output / "route_consistency_summary.csv", summary["families"])
        if result["status"] == "success":
            worst = result["summary"]["worst_layer_metrics"]
            print(
                f"  layers={result['summary']['layers']} gate={result['summary']['gate_passed']} "
                f"worst_set={worst['topk_exact_rate']:.6f} "
                f"worst_ordered={worst['topk_ordered_exact_rate']:.6f} "
                f"worst_top1_flip={worst['top1_flip_rate']:.6f}",
                flush=True,
            )
        else:
            print(f"  failed: {result['error_type']}: {result['error']}", flush=True)

    selected_names = {spec.family for spec in selected}
    selected_results = [summary["families"].get(family, {}) for family in selected_names]
    measured = sum(row.get("status") == "success" for row in selected_results)
    gate_passed = sum(
        row.get("status") == "success" and row.get("summary", {}).get("gate_passed") is True
        for row in selected_results
    )
    summary["completed"] = measured
    summary["expected"] = len(selected_names)
    summary["status"] = "success" if measured == len(selected_names) else "failed"
    summary["acceptance_passed"] = measured == len(selected_names) and gate_passed == len(selected_names)
    summary["gates_passed"] = gate_passed
    save_json(summary_path, summary)
    write_csv(args.output / "route_consistency_summary.csv", summary["families"])

    print("\n========== route consistency summary ==========", flush=True)
    for family in [spec.family for spec in selected]:
        row = summary["families"].get(family, {})
        if row.get("status") != "success":
            print(f"{family:8s} measurement=failed error={row.get('error')}", flush=True)
            continue
        compact = row["summary"]
        worst = compact["worst_layer_metrics"]
        print(
            f"{family:8s} measurement=success gate={compact['gate_passed']} "
            f"layers={compact['layers']} representations={compact['representations']} "
            f"worst_set={worst['topk_exact_rate']:.6f} "
            f"worst_ordered={worst['topk_ordered_exact_rate']:.6f} "
            f"worst_top1_flip={worst['top1_flip_rate']:.6f}",
            flush=True,
        )
    print(
        f"Measurement status: {summary['status']} ({measured}/{len(selected_names)}); "
        f"acceptance_passed={summary['acceptance_passed']}",
        flush=True,
    )
    print(f"JSON: {summary_path}\nCSV: {args.output / 'route_consistency_summary.csv'}", flush=True)
    return 0 if summary["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
