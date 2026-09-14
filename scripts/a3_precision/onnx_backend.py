"""ONNX export, precision conversion, static PTQ, and operator auditing."""

from __future__ import annotations

import re
import shutil
import time
import warnings
from collections import Counter, deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from .common import json_dump, letterbox_image, sha256_file
from .manifest import HarnessConfig, ModelSpec


QUANTIZABLE_OPS = frozenset({"Conv", "Gemm", "MatMul"})
SENSITIVE_ROLES = ("router", "topk", "attention", "matmul")


def _require_onnx():
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("ONNX stages require `pip install onnx onnxruntime onnxruntime-gpu`") from exc
    return onnx


def graph_audit(path: str | Path) -> dict[str, Any]:
    """Return reproducible graph, operator, quantization, and artifact metadata."""
    onnx = _require_onnx()
    model = onnx.load(str(path), load_external_data=False)
    counts = Counter(node.op_type for node in model.graph.node)
    quantized = sum(
        count
        for op, count in counts.items()
        if op.startswith("QLinear") or op in {"MatMulInteger", "ConvInteger", "DynamicQuantizeLinear"}
    )
    qdq = counts.get("QuantizeLinear", 0) + counts.get("DequantizeLinear", 0)
    target = Path(path)
    return {
        "path": str(target.resolve()),
        "size_bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "ir_version": int(model.ir_version),
        "opset": {item.domain or "ai.onnx": int(item.version) for item in model.opset_import},
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "operator_counts": dict(sorted(counts.items())),
        "fused_quantized_nodes": quantized,
        "qdq_nodes": qdq,
        "inputs": [item.name for item in model.graph.input],
        "outputs": [item.name for item in model.graph.output],
    }


def _load_yolo(spec: ModelSpec):
    from ultralytics import YOLO

    model = YOLO(str(spec.checkpoint))
    if spec.adapter is not None and not model.load_adapters(spec.adapter):
        raise RuntimeError(f"failed to load {spec.family} adapter from {spec.adapter}")
    return model


def _enable_masked_export(model) -> int:
    changed = 0
    for module in model.modules():
        router = getattr(module, "router", None)
        if hasattr(module, "export_masked"):
            module.export_masked = True
            changed += 1
        if hasattr(router, "export_masked"):
            router.export_masked = True
            changed += 1
    return changed


def _onnx_tensor_state_hook(module, state_dict, prefix, local_metadata):
    """Hide dictionary extra-state from the legacy tracer, not from the live model."""
    import torch

    for key, value in list(state_dict.items()):
        if not isinstance(value, torch.Tensor):
            if key.endswith("_extra_state") and isinstance(value, dict):
                del state_dict[key]
            else:
                raise TypeError(f"Unexpected non-tensor ONNX state: {key}: {type(value).__name__}")


def export_fp32(spec: ModelSpec, output_dir: Path, settings: dict[str, Any]) -> dict[str, Any]:
    """Export one family with routing-preserving masked-dense semantics."""
    family_dir = output_dir / spec.family / "artifacts"
    family_dir.mkdir(parents=True, exist_ok=True)
    target = family_dir / "fp32.onnx"
    target.unlink(missing_ok=True)
    yolo = _load_yolo(spec)
    masked_modules = _enable_masked_export(yolo.model)
    extra_state = {
        key: value
        for key, value in yolo.model.state_dict().items()
        if key.endswith("_extra_state") and isinstance(value, dict)
    }
    if extra_state:
        json_dump(family_dir / "source_extra_state.json", extra_state)
        register_hook = getattr(yolo.model, "register_state_dict_post_hook", None)
        if register_hook is None:
            register_hook = yolo.model._register_state_dict_hook
        register_hook(_onnx_tensor_state_hook)
    # Exporter derives the destination from pt_path. Point the in-memory model
    # at our experiment directory without touching the locked checkpoint.
    yolo.model.pt_path = str(family_dir / "fp32.pt")
    started = time.perf_counter()
    exported = yolo.export(
        format="onnx",
        imgsz=int(settings.get("imgsz", 640)),
        batch=int(settings.get("batch", 1)),
        device=str(settings.get("export_device", settings.get("device", "cpu"))),
        opset=int(settings.get("opset", 17)),
        simplify=bool(settings.get("simplify", False)),
        dynamic=bool(settings.get("dynamic", False)),
        quantize=32,
        molora_export_mode="routing_preserved",
    )
    exported_path = Path(exported)
    if exported_path.resolve() != target.resolve():
        shutil.copy2(exported_path, target)
    payload = {
        "status": "success",
        "family": spec.family,
        "source_checkpoint_sha256": sha256_file(spec.checkpoint),
        "source_config_sha256": sha256_file(spec.config),
        "source_adapter": str(spec.adapter) if spec.adapter is not None else None,
        "masked_export_flags": masked_modules,
        "elapsed_seconds": time.perf_counter() - started,
        "artifact": graph_audit(target),
        "claim_boundary": "masked-dense numerical parity; not conditional expert execution",
    }
    json_dump(family_dir / "fp32.export.json", payload)
    return payload


def convert_fp16(fp32_path: str | Path, fp16_path: str | Path) -> dict[str, Any]:
    """Convert an exported FP32 graph to FP16 while keeping public I/O FP32."""
    onnx = _require_onnx()
    source, target = Path(fp32_path), Path(fp16_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    try:
        from onnxruntime.transformers import float16

        model = onnx.load(str(source))
        converted = float16.convert_float_to_float16(model, keep_io_types=True)
        _topologically_sort_model(converted)
        onnx.checker.check_model(converted)
        onnx.save(converted, str(target))
        result = {"status": "success", "artifact": graph_audit(target)}
    except Exception as exc:
        target.unlink(missing_ok=True)
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_operators": _operators_from_error(str(exc)),
        }
    json_dump(target.with_suffix(".conversion.json"), result)
    return result


def _topologically_sort_graph(graph) -> None:
    """Repair converters that append boundary Cast nodes after their consumers."""
    nodes = list(graph.node)
    produced = {output for node in nodes for output in node.output if output}
    available = {item.name for item in graph.input}
    available.update(item.name for item in graph.initializer)
    available.update(name for node in nodes for name in node.input if name and name not in produced)
    pending = nodes
    ordered = []
    while pending:
        ready = [node for node in pending if all(not name or name in available for name in node.input)]
        if not ready:
            unresolved = {node.name or node.op_type: list(node.input) for node in pending[:10]}
            raise RuntimeError(f"could not topologically sort converted ONNX graph: {unresolved}")
        ready_ids = {id(node) for node in ready}
        pending = [node for node in pending if id(node) not in ready_ids]
        for node in ready:
            ordered.append(node)
            available.update(name for name in node.output if name)
    del graph.node[:]
    graph.node.extend(ordered)

    # Control-flow subgraphs may also be converted. Unknown inputs there are
    # valid outer-scope captures and are treated as external by the sorter.
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == attribute.GRAPH:
                _topologically_sort_graph(attribute.g)
            elif attribute.type == attribute.GRAPHS:
                for nested in attribute.graphs:
                    _topologically_sort_graph(nested)


def _topologically_sort_model(model) -> None:
    _topologically_sort_graph(model.graph)


def _clean_name(name: str) -> str:
    return (name or "<unnamed>").replace("\\", "/")


def _prefix_through_token(name: str, tokens: Iterable[str]) -> str | None:
    lower = name.lower()
    positions = [(lower.find(f"/{token}"), token) for token in tokens if lower.find(f"/{token}") >= 0]
    if not positions:
        return None
    position, token = min(positions)
    return name[: position + len(token) + 1]


def discover_sensitivity_groups(path: str | Path, *, topk_ancestor_depth: int = 4) -> dict[str, dict[str, Any]]:
    """Group quantizable Router, TopK, Attention, and MatMul nodes for ablation."""
    onnx = _require_onnx()
    model = onnx.load(str(path), load_external_data=False)
    nodes = list(model.graph.node)
    producer = {output: index for index, node in enumerate(nodes) for output in node.output}
    groups: dict[str, dict[str, Any]] = {}

    def add(group_id: str, role: str, node_name: str) -> None:
        row = groups.setdefault(group_id, {"group": group_id, "role": role, "nodes": []})
        if node_name and node_name not in row["nodes"]:
            row["nodes"].append(node_name)

    for node in nodes:
        if node.op_type not in QUANTIZABLE_OPS or not node.name:
            continue
        name = _clean_name(node.name)
        router_prefix = _prefix_through_token(name, ("router", "routing", "gate", "gating"))
        attention_prefix = _prefix_through_token(name, ("attn", "attention"))
        if router_prefix:
            add(f"router::{router_prefix}", "router", node.name)
        elif attention_prefix:
            add(f"attention::{attention_prefix}", "attention", node.name)
        elif node.op_type in {"MatMul", "Gemm"}:
            parent = name.rsplit("/", 1)[0] or name
            add(f"matmul::{parent}", "matmul", node.name)

    for topk_index, topk_node in enumerate(nodes):
        if topk_node.op_type != "TopK":
            continue
        prefix = _clean_name(topk_node.name).rsplit("/", 1)[0]
        group_id = f"topk::{prefix}"
        queue = deque((tensor, 0) for tensor in topk_node.input)
        visited = set()
        while queue:
            tensor, depth = queue.popleft()
            node_index = producer.get(tensor)
            if node_index is None or node_index in visited or depth > topk_ancestor_depth:
                continue
            visited.add(node_index)
            ancestor = nodes[node_index]
            if ancestor.op_type in QUANTIZABLE_OPS and ancestor.name:
                add(group_id, "topk", ancestor.name)
            for parent_tensor in ancestor.input:
                queue.append((parent_tensor, depth + 1))

    for row in groups.values():
        row["nodes"].sort()
        row["node_count"] = len(row["nodes"])
    return dict(sorted(groups.items()))


def instrument_topk_outputs(source: str | Path, target: str | Path) -> dict[str, dict[str, Any]]:
    """Expose sparse TopK decisions or dense router probabilities as graph outputs.

    Sparse routing graphs retain their actual TopK value/index tensors. Dense
    routers (for example MoA and uncalibrated Latent Mixture) have no TopK node,
    so their Softmax probabilities are exposed and evaluated with a standardized
    Top-2 ranking probe. The returned contract records which representation was
    used; callers must not describe the dense probe as conditional execution.
    """
    onnx = _require_onnx()
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.shape_inference.infer_shapes(onnx.load(str(source), load_external_data=False))
    graph = model.graph
    value_info = {item.name: item for item in (*graph.input, *graph.output, *graph.value_info)}
    initializers = {item.name: item for item in graph.initializer}
    existing_outputs = {item.name for item in graph.output}
    routing_tokens = ("router", "routing", "gate", "gating")
    sparse_nodes = [
        node
        for node in graph.node
        if node.op_type == "TopK" and any(token in node.name.lower() for token in routing_tokens)
    ]
    route_nodes = sparse_nodes or [
        node
        for node in graph.node
        if node.op_type == "Softmax" and any(token in node.name.lower() for token in routing_tokens)
    ]
    specs: dict[str, dict[str, Any]] = {}
    for node in route_nodes:
        attributes = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
        input_info = value_info.get(node.input[0])
        output_info = value_info.get(node.output[0])
        contract_info = input_info if node.op_type == "TopK" else output_info
        dimensions = list(contract_info.type.tensor_type.shape.dim) if contract_info is not None else []
        axis = int(attributes.get("axis", -1))
        if dimensions and axis < 0:
            axis += len(dimensions)
        num_experts = int(dimensions[axis].dim_value) if dimensions and 0 <= axis < len(dimensions) else 0
        if node.op_type == "TopK":
            k_initializer = initializers.get(node.input[1]) if len(node.input) > 1 else None
            top_k = int(numpy_helper.to_array(k_initializer).reshape(-1)[0]) if k_initializer is not None else 0
            output_dimensions = list(output_info.type.tensor_type.shape.dim) if output_info is not None else []
            if top_k <= 0 and output_dimensions and 0 <= axis < len(output_dimensions):
                top_k = int(output_dimensions[axis].dim_value)
            representation = "sparse_topk"
            top_k_source = "onnx_topk"
        else:
            top_k = min(2, num_experts)
            representation = "dense_probabilities"
            top_k_source = "standardized_top2_ranking_probe"
        if num_experts <= 1 or top_k <= 0:
            raise RuntimeError(
                f"could not infer routing contract for {node.name}: num_experts={num_experts}, top_k={top_k}"
            )
        exposed_outputs = node.output[:2] if node.op_type == "TopK" else node.output[:1]
        for index, output_name in enumerate(exposed_outputs):
            if output_name in existing_outputs:
                continue
            info = value_info.get(output_name)
            if info is not None:
                graph.output.append(deepcopy(info))
            else:
                element_type = TensorProto.FLOAT if index == 0 else TensorProto.INT64
                graph.output.append(helper.make_tensor_value_info(output_name, element_type, None))
            existing_outputs.add(output_name)
        specs[node.name] = {
            "node_name": node.name,
            "values_output": node.output[0],
            "indices_output": node.output[1] if node.op_type == "TopK" else None,
            "axis": axis,
            "top_k": top_k,
            "num_experts": num_experts,
            "representation": representation,
            "top_k_source": top_k_source,
        }
    if not specs:
        raise RuntimeError(f"no routing TopK or dense router Softmax outputs found in {source}")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, str(target))
    return dict(sorted(specs.items()))


def nodes_for_roles(groups: dict[str, dict[str, Any]], roles: Iterable[str]) -> list[str]:
    """Return the unique node union for selected structural roles."""
    selected = set(roles)
    return sorted({name for row in groups.values() if row["role"] in selected for name in row["nodes"]})


def nodes_for_groups(groups: dict[str, dict[str, Any]], group_ids: Iterable[str]) -> list[str]:
    """Return the unique node union for selected sensitivity groups."""
    return sorted({name for group_id in group_ids for name in groups.get(group_id, {}).get("nodes", [])})


class ImageCalibrationReader:
    """Deterministic ONNX Runtime calibration reader."""

    def __init__(self, images: list[Path], input_name: str, imgsz: int):
        from onnxruntime.quantization import CalibrationDataReader

        if not issubclass(type(self), CalibrationDataReader):
            # CalibrationDataReader uses structural get_next behavior; this
            # branch exists only to document the runtime interface.
            pass
        self.images = tuple(images)
        self.input_name = input_name
        self.imgsz = imgsz
        self.rewind()

    def get_next(self):
        path = next(self._iterator, None)
        return None if path is None else {self.input_name: letterbox_image(path, self.imgsz)}

    def rewind(self) -> None:
        self._iterator = iter(self.images)


def _operators_from_error(message: str) -> list[str]:
    known = (
        "Attention",
        "Conv",
        "DequantizeLinear",
        "Gather",
        "Gemm",
        "GridSample",
        "MatMul",
        "QLinearConv",
        "QuantizeLinear",
        "ScatterND",
        "Softmax",
        "TopK",
    )
    return [operator for operator in known if re.search(rf"\b{re.escape(operator)}\b", message, re.IGNORECASE)]


def _restore_metadata(source: Path, target: Path) -> None:
    onnx = _require_onnx()
    source_model = onnx.load(str(source), load_external_data=False)
    target_model = onnx.load(str(target), load_external_data=False)
    del target_model.metadata_props[:]
    for item in source_model.metadata_props:
        prop = target_model.metadata_props.add()
        prop.key, prop.value = item.key, item.value
    onnx.save(target_model, str(target))


def _eligible_quantization_nodes(
    source: Path,
    *,
    op_types: set[str],
    nodes_to_exclude: set[str],
    nodes_to_quantize: set[str],
) -> list[str]:
    """Return source nodes that ORT was explicitly allowed to quantize."""
    onnx = _require_onnx()
    model = onnx.load(str(source), load_external_data=False)
    eligible = []
    for node in model.graph.node:
        if node.op_type not in op_types or not node.name or node.name in nodes_to_exclude:
            continue
        if nodes_to_quantize and node.name not in nodes_to_quantize:
            continue
        eligible.append(node.name)
    return sorted(eligible)


def quantize_static_onnx(
    source: str | Path,
    target: str | Path,
    *,
    calibration_images: list[Path],
    imgsz: int,
    settings: dict[str, Any],
    nodes_to_exclude: list[str] | None = None,
    nodes_to_quantize: list[str] | None = None,
    label: str,
) -> dict[str, Any]:
    """Run ORT static PTQ with full failure and operator evidence."""
    onnx = _require_onnx()
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    input_name = onnx.load(str(source), load_external_data=False).graph.input[0].name
    format_name = str(settings.get("format", "QDQ"))
    activation_name = str(settings.get("activation_type", "QInt8"))
    weight_name = str(settings.get("weight_type", "QInt8"))
    method_name = str(settings.get("calibration_method", "MinMax"))
    nodes_to_exclude = sorted(set(nodes_to_exclude or []))
    nodes_to_quantize = sorted(set(nodes_to_quantize or []))
    op_types = set(settings.get("op_types", sorted(QUANTIZABLE_OPS)))
    started = time.perf_counter()
    warning_rows = []
    eligible_nodes: list[str] = []
    try:
        eligible_nodes = _eligible_quantization_nodes(
            source,
            op_types=op_types,
            nodes_to_exclude=set(nodes_to_exclude),
            nodes_to_quantize=set(nodes_to_quantize),
        )
        if nodes_to_quantize and not eligible_nodes:
            raise ValueError(f"{label}: requested nodes contain no eligible quantizable ONNX operators")
        if not calibration_images:
            raise ValueError(f"{label}: calibration image list is empty")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            quantize_static(
                str(source),
                str(target),
                calibration_data_reader=ImageCalibrationReader(calibration_images, input_name, imgsz),
                quant_format=getattr(QuantFormat, format_name),
                per_channel=bool(settings.get("per_channel", True)),
                activation_type=getattr(QuantType, activation_name),
                weight_type=getattr(QuantType, weight_name),
                calibrate_method=getattr(CalibrationMethod, method_name),
                calibration_providers=settings.get("calibration_providers"),
                nodes_to_exclude=nodes_to_exclude,
                nodes_to_quantize=nodes_to_quantize,
                op_types_to_quantize=sorted(op_types),
                extra_options=dict(settings.get("extra_options", {})),
            )
            warning_rows = [str(item.message) for item in caught]
        _restore_metadata(source, target)
        onnx.checker.check_model(str(target))
        audit = graph_audit(target)
        if eligible_nodes and audit["qdq_nodes"] == 0 and audit["fused_quantized_nodes"] == 0:
            raise RuntimeError(
                f"{label}: ORT returned a valid graph but quantized none of the {len(eligible_nodes)} eligible nodes"
            )
        payload = {
            "status": "success",
            "label": label,
            "elapsed_seconds": time.perf_counter() - started,
            "calibration_images": len(calibration_images),
            "settings": {
                "format": format_name,
                "activation_type": activation_name,
                "weight_type": weight_name,
                "calibration_method": method_name,
                "calibration_providers": settings.get("calibration_providers"),
                "fallback_precision": "fp32",
            },
            "nodes_to_exclude": nodes_to_exclude,
            "nodes_to_quantize": nodes_to_quantize,
            "eligible_quantization_nodes": eligible_nodes,
            "warnings": warning_rows,
            "warning_operators": _operators_from_error("\n".join(warning_rows)),
            "failed_operators": [],
            "artifact": audit,
        }
    except Exception as exc:
        target.unlink(missing_ok=True)
        payload = {
            "status": "failed",
            "label": label,
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_operators": _operators_from_error(str(exc)),
            "nodes_to_exclude": nodes_to_exclude,
            "nodes_to_quantize": nodes_to_quantize,
            "eligible_quantization_nodes": eligible_nodes,
            "warnings": warning_rows,
        }
    json_dump(target.with_suffix(".quantization.json"), payload)
    return payload


def build_primary_variants(config: HarnessConfig, spec: ModelSpec, lock: dict[str, Any]) -> dict[str, Any]:
    """Create FP32, FP16, full INT8, and manual-fallback artifacts."""
    settings = config.raw.get("experiment", {})
    quant = config.raw.get("quantization", {})
    family_dir = config.output_dir / spec.family / "artifacts"
    fp32 = family_dir / "fp32.onnx"
    export_fp32(spec, config.output_dir, settings)
    results = {"fp32": {"status": "success", "artifact": graph_audit(fp32)}}
    results["fp16"] = convert_fp16(fp32, family_dir / "fp16.onnx")
    calibration_images = [Path(path) for path in lock["dataset"]["calibration"]["images"]]
    groups = discover_sensitivity_groups(fp32)
    json_dump(family_dir / "sensitivity_groups.json", groups)
    results["full_int8"] = quantize_static_onnx(
        fp32,
        family_dir / "full_int8.onnx",
        calibration_images=calibration_images,
        imgsz=int(settings.get("imgsz", 640)),
        settings=quant,
        label="full_int8",
    )
    manual_roles = quant.get("manual_fallback_roles", list(SENSITIVE_ROLES))
    manual_nodes = nodes_for_roles(groups, manual_roles)
    results["manual_fallback"] = quantize_static_onnx(
        fp32,
        family_dir / "manual_fallback.onnx",
        calibration_images=calibration_images,
        imgsz=int(settings.get("imgsz", 640)),
        settings=quant,
        nodes_to_exclude=manual_nodes,
        label="manual_fallback",
    )
    json_dump(family_dir / "primary_variants.json", results)
    return results


def build_auto_fallback(
    config: HarnessConfig,
    spec: ModelSpec,
    lock: dict[str, Any],
    sensitivity: dict[str, Any],
) -> dict[str, Any]:
    """Quantize while keeping automatically selected groups at FP32."""
    settings = config.raw.get("experiment", {})
    quant = config.raw.get("quantization", {})
    family_dir = config.output_dir / spec.family / "artifacts"
    fp32 = family_dir / "fp32.onnx"
    groups = discover_sensitivity_groups(fp32)
    selected_groups = sensitivity.get("selected_groups", [])
    excluded = nodes_for_groups(groups, selected_groups)
    calibration_images = [Path(path) for path in lock["dataset"]["calibration"]["images"]]
    result = quantize_static_onnx(
        fp32,
        family_dir / "auto_fallback.onnx",
        calibration_images=calibration_images,
        imgsz=int(settings.get("imgsz", 640)),
        settings=quant,
        nodes_to_exclude=excluded,
        label="auto_fallback",
    )
    result["selected_groups"] = selected_groups
    result["selection_sha256"] = sha256_file(json_dump(family_dir / "auto_fallback.selection.json", sensitivity))
    json_dump(family_dir / "auto_fallback.quantization.json", result)
    return result
