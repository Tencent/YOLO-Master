#!/usr/bin/env python3
"""Export and verify EsMoE-N models for ONNX, NCNN and optional MNN targets.

The default path is the Issue #51 acceptance path: a static ONNX graph is
checked, simplified with onnxsim, and used as the canonical artifact; NCNN is
exported with dense MoE routing (pnnx cannot lower topk/one_hot) and loaded
through an actual ncnn extractor smoke test; MNN conversion consumes the
simplified ONNX file.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import numbers
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXAMPLE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# These labels are consumed by the evidence manifest and metric evaluators.
# Keeping the vocabulary here avoids silently treating a dense export as a
# numerically equivalent sparse-eager baseline.
ROUTING_SEMANTICS = ("native_sparse", "dense_fallback", "dense_native", "not_applicable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="trained .pt checkpoint")
    parser.add_argument("--out-dir", type=Path, default=Path("exports"))
    parser.add_argument(
        "--formats", nargs="+", choices=("onnx", "ncnn", "mnn"),
        default=["onnx", "ncnn"],
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--half", action="store_true")
    simplify = parser.add_mutually_exclusive_group()
    simplify.add_argument(
        "--simplify", dest="simplify", action="store_true",
        help="explicitly simplify and validate with onnxsim (the default)",
    )
    simplify.add_argument(
        "--no-simplify", dest="simplify", action="store_false",
        help="debug-only raw export; requires --allow-unsimplified",
    )
    parser.set_defaults(simplify=True)
    parser.add_argument(
        "--allow-unsimplified", action="store_true",
        help="permit --no-simplify for debugging; the result is not acceptance-ready",
    )
    parser.add_argument("--mnn-convert", default="mnnconvert", help="mnnconvert executable")
    return parser.parse_args()


def _copy_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def _opset(graph) -> Optional[int]:
    return next(
        (item.version for item in graph.opset_import if item.domain in ("", "ai.onnx")),
        None,
    )


def _restore_metadata(source, target) -> None:
    """Keep Ultralytics names/imgsz metadata when an onnxsim release drops it."""
    props = [(prop.key, prop.value) for prop in source.metadata_props]
    del target.metadata_props[:]
    for key, value in props:
        entry = target.metadata_props.add()
        entry.key, entry.value = key, value


def _force_ncnn_dense(model):
    """Force export-safe dense routing and return the reversible state.

    EsMoE implementations have used both ``DynamicRoutingLayer.use_top_k`` and
    ``ES_MOE.use_sparse_inference`` across YOLO-Master revisions.  Detect both
    forms, and also retain a conservative attribute-based fallback so a renamed
    implementation cannot silently produce an unrecorded sparse graph.
    """
    state = []
    router_count = esmoe_count = 0
    try:
        from ultralytics.nn.modules.moe.routers import DynamicRoutingLayer
    except ImportError:
        DynamicRoutingLayer = None
    try:
        import ultralytics.nn.modules.moe.modules as moe_modules
        esmoe_cls = getattr(moe_modules, "ES_MOE", None)
    except ImportError:
        esmoe_cls = None
    modules = getattr(getattr(model, "model", None), "modules", None)
    if not callable(modules):
        return state, router_count, esmoe_count
    for module in modules():
        is_router = DynamicRoutingLayer is not None and isinstance(module, DynamicRoutingLayer)
        is_esmoe = esmoe_cls is not None and isinstance(module, esmoe_cls)
        # Attribute checks cover compatible forks without weakening the explicit
        # class checks above.  Do not mutate an unrelated module unless it has
        # a routing-specific name and flag.
        class_name = type(module).__name__.lower()
        has_router_flag = hasattr(module, "use_top_k") and (
            is_router or "routing" in class_name or "router" in class_name
        )
        has_esmoe_flag = hasattr(module, "use_sparse_inference") and (
            is_esmoe or "moe" in class_name or "mixture" in class_name
        )
        if has_router_flag:
            state.append((module, "use_top_k", module.use_top_k))
            module.use_top_k = False
            router_count += 1
        if has_esmoe_flag:
            state.append((module, "use_sparse_inference", module.use_sparse_inference))
            module.use_sparse_inference = False
            esmoe_count += 1
    return state, router_count, esmoe_count


def _routing_overlap(state) -> int:
    """Count modules represented by both compatibility routing flags."""
    flags = {}
    for module, attribute, _ in state:
        flags.setdefault(id(module), set()).add(attribute)
    return sum(
        "use_top_k" in attributes and "use_sparse_inference" in attributes
        for attributes in flags.values()
    )


def _routing_record(router_count: int, esmoe_count: int, overlap_count: int = 0) -> dict:
    """Return a stable routing provenance record for an export result.

    ``router_count`` and ``esmoe_count`` count compatibility flags.  A single
    module can expose both flags, so ``overlap_count`` removes that duplicate
    from the union-based ``total`` while retaining the detailed counts.
    """
    router_count = int(router_count)
    esmoe_count = int(esmoe_count)
    overlap_count = int(overlap_count)
    if min(router_count, esmoe_count, overlap_count) < 0 or overlap_count > min(router_count, esmoe_count):
        raise ValueError("invalid routing layer counts")
    layer_count = router_count + esmoe_count - overlap_count
    return {
        "routing_semantics": "dense_fallback" if layer_count else "not_applicable",
        "routing_layers": {
            "dynamic_router": router_count,
            "esmoe": esmoe_count,
            "overlap": overlap_count,
            "total": layer_count,
        },
    }


def _check_static_input(graph, imgsz: int) -> None:
    if not graph.graph.input:
        raise RuntimeError("ONNX graph has no input tensor")
    dims = graph.graph.input[0].type.tensor_type.shape.dim
    values = [dim.dim_value for dim in dims]
    if len(values) != 4 or values != [1, 3, imgsz, imgsz]:
        raise RuntimeError(
            f"ONNX input must be static [1,3,{imgsz},{imgsz}], got {values}"
        )


def export_onnx(model, args: argparse.Namespace, out_dir: Path) -> dict:
    import onnx

    # Use the same explicit dense fallback for ONNX that is required by pnnx.
    # Restoring the flags in ``finally`` leaves the caller's model unchanged
    # for a subsequent reference evaluation or a different export format.
    routing_state, router_count, esmoe_count = _force_ncnn_dense(model)
    overlap_count = _routing_overlap(routing_state)
    try:
        exported_value = model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=args.simplify,
            dynamic=False,
            half=args.half,
        )
    finally:
        for module, attribute, value in reversed(routing_state):
            setattr(module, attribute, value)
    if not exported_value:
        raise RuntimeError("ONNX exporter returned no path")
    exported = Path(exported_value)
    if not exported.is_file():
        raise FileNotFoundError(f"ONNX exporter returned no file: {exported}")

    raw_destination = out_dir / f"{exported.stem}_raw.onnx"
    _copy_file(exported, raw_destination)
    graph = onnx.load(str(exported))
    onnx.checker.check_model(graph)
    _check_static_input(graph, args.imgsz)
    raw_opset = _opset(graph)
    if raw_opset is None or raw_opset > args.opset:
        raise RuntimeError(
            f"exported ONNX opset {raw_opset} exceeds requested compatibility {args.opset}"
        )

    canonical = out_dir / exported.name
    result = {
        "format": "onnx",
        "path": str(canonical),
        "raw_path": str(raw_destination),
        "opset": raw_opset,
        "checked": True,
        "simplified": False,
        "acceptance_ready": False,
    }
    result.update(_routing_record(router_count, esmoe_count, overlap_count))
    if not args.simplify:
        if not args.allow_unsimplified:
            raise RuntimeError(
                "Issue #51 requires an onnxsim-validated ONNX artifact; "
                "use the default or pass --allow-unsimplified for a debug export"
            )
        _copy_file(exported, canonical)
        result["warning"] = "raw ONNX retained by explicit --allow-unsimplified"
        return result

    try:
        import onnxsim
    except ImportError as exc:
        raise RuntimeError("onnxsim is required for the default acceptance export") from exc

    input_names = [value.name for value in graph.graph.input]
    override = {input_names[0]: [1, 3, args.imgsz, args.imgsz]} if input_names else None
    try:
        simplified, valid = onnxsim.simplify(
            str(exported), overwrite_input_shapes=override
        )
    except (TypeError, ValueError):
        simplified, valid = onnxsim.simplify(
            graph, overwrite_input_shapes=override
        )
    if not valid:
        raise RuntimeError("onnxsim returned check=False")
    _restore_metadata(graph, simplified)
    _check_static_input(simplified, args.imgsz)
    onnx.checker.check_model(simplified)
    simplified_opset = _opset(simplified)
    if simplified_opset is None or simplified_opset > args.opset:
        raise RuntimeError(
            f"simplified ONNX opset {simplified_opset} exceeds requested compatibility {args.opset}"
        )
    onnx.save(simplified, str(canonical))
    result.update({
        "opset": simplified_opset,
        "simplified": True,
        "onnxsim_check": True,
        "acceptance_ready": True,
    })
    return result


def _find_ncnn_pair(exported: Path) -> Tuple[Path, Path]:
    # A not-yet-created ``*_ncnn_model`` path is still a directory target.
    # Falling back to its parent would accidentally select an unrelated stale
    # param/bin pair from a previous export.
    directory = (
        exported
        if exported.is_dir() or (not exported.exists() and not exported.suffix)
        else exported.parent
    )
    if exported.is_file() and exported.suffix.lower() == ".param":
        param = exported
    else:
        candidates = [directory / "model.ncnn.param", *sorted(directory.glob("*.param"))]
        param = next((candidate for candidate in candidates if candidate.is_file()), None)
    if param is None:
        raise FileNotFoundError(f"NCNN .param not found under {directory}")
    binary = param.with_suffix(".bin")
    if not binary.is_file() or binary.stat().st_size == 0:
        raise FileNotFoundError(f"NCNN .bin missing or empty: {binary}")
    return param, binary


def _ncnn_pair_fingerprint(directory: Path) -> Optional[Tuple[Tuple[int, int, str], Tuple[int, int, str]]]:
    """Return content/stat fingerprints for an existing NCNN pair.

    pnnx may raise after writing its files.  Fingerprinting the pair before and
    after export prevents a stale pair from a previous run being reported as
    the result of the current checkpoint.
    """
    try:
        param, binary = _find_ncnn_pair(directory)
    except (FileNotFoundError, OSError):
        return None

    def fingerprint(path: Path) -> Tuple[int, int, str]:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns, digest.hexdigest()

    return fingerprint(param), fingerprint(binary)


def _write_ncnn_metadata(
    out_dir: Path,
    model,
    imgsz: int,
    input_blob: Optional[str] = None,
    output_blob: Optional[str] = None,
    proto_blob: Optional[str] = None,
) -> Path:
    names = (
        getattr(model, "names", None)
        or getattr(getattr(model, "model", None), "names", None)
        or {}
    )
    if isinstance(names, (list, tuple)):
        items = list(enumerate(names))
    elif isinstance(names, dict):
        items = []
        for key, value in names.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                index = len(items)
            items.append((index, value))
    else:
        items = []
    items.sort(key=lambda item: item[0])
    metadata = out_dir / "metadata.yaml"
    with metadata.open("w", encoding="utf-8") as handle:
        handle.write(f"imgsz: [{imgsz}, {imgsz}]\n")
        if input_blob:
            handle.write(f"input_blob: {json.dumps(input_blob)}\n")
        if output_blob:
            handle.write(f"output_blob: {json.dumps(output_blob)}\n")
        if proto_blob:
            handle.write(f"proto_blob: {json.dumps(proto_blob)}\n")
        handle.write("names:\n")
        for index, name in items:
            # JSON quoting is valid YAML and preserves names containing ':'.
            handle.write(f"  {index}: {json.dumps(str(name), ensure_ascii=False)}\n")
    return metadata


def _param_io_names(param: Path) -> Tuple[Optional[str], Optional[str]]:
    tops, bottoms, inputs = [], set(), []
    for line in param.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[0].startswith("#"):
            continue
        try:
            bottom_count, top_count = int(fields[2]), int(fields[3])
        except ValueError:
            continue
        start = 4
        bottom_names = fields[start : start + bottom_count]
        top_names = fields[start + bottom_count : start + bottom_count + top_count]
        bottoms.update(bottom_names)
        tops.extend(top_names)
        if fields[0] == "Input" and top_names:
            inputs.append(top_names[0])
    output = next(
        (name for name in ("out0", "output0", "output", "out") if name in tops),
        None,
    )
    if output is None:
        output = next((name for name in reversed(tops) if name not in bottoms), None)
    return (inputs[0] if inputs else None), output


def _param_output_names(param: Path) -> List[str]:
    """Return terminal NCNN blobs in graph order.

    pnnx commonly names detection/prototype outputs ``out0``/``out1``, but
    those names are not part of the NCNN ABI.  Persisting the actual terminal
    names in the sidecar lets the C++ runtime consume renamed graphs as well.
    """
    tops: List[str] = []
    bottoms = set()
    for line in param.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[0].startswith("#"):
            continue
        try:
            bottom_count, top_count = int(fields[2]), int(fields[3])
        except ValueError:
            continue
        start = 4
        bottoms.update(fields[start : start + bottom_count])
        tops.extend(fields[start + bottom_count : start + bottom_count + top_count])
    # Preserve first appearance while removing duplicate intermediate tops.
    return list(dict.fromkeys(name for name in tops if name not in bottoms))


def _ncnn_code(value) -> int:
    # The reference ncnn wheel returns integer status codes, while a few
    # bindings expose the same success/failure result as a boolean.  In both
    # conventions success must normalize to zero for the checks below.
    if isinstance(value, bool):
        return 0 if value else 1
    return int(value) if isinstance(value, numbers.Integral) else 0


def _ncnn_smoke_check(param: Path, binary: Path, imgsz: int) -> dict:
    """Load the graph and execute one zero-input extractor pass."""
    import io

    import numpy as np
    import ncnn

    input_name, output_name = _param_io_names(param)
    if not input_name or not output_name:
        raise RuntimeError("NCNN graph must expose a discoverable Input and terminal output blob")
    net = ncnn.Net()
    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
        code_param = net.load_param(str(param))
        if _ncnn_code(code_param) != 0:
            raise RuntimeError(f"ncnn load_param failed with code {_ncnn_code(code_param)}")
        code_model = net.load_model(str(binary))
        if _ncnn_code(code_model) != 0:
            raise RuntimeError(f"ncnn load_model failed with code {_ncnn_code(code_model)}")
        extractor = net.create_extractor()
        input_mat = ncnn.Mat(np.zeros((3, imgsz, imgsz), dtype=np.float32))
        code_input = extractor.input(input_name, input_mat)
        if _ncnn_code(code_input) != 0:
            raise RuntimeError(f"ncnn input failed with code {_ncnn_code(code_input)}")
        try:
            extracted = extractor.extract(output_name)
        except TypeError:
            output_mat = ncnn.Mat()
            code_extract = extractor.extract(output_name, output_mat)
            extracted = (code_extract, output_mat)
        if isinstance(extracted, tuple) and len(extracted) == 2:
            code_extract, output_mat = extracted
        else:
            code_extract, output_mat = 0, extracted
        if _ncnn_code(code_extract) != 0:
            raise RuntimeError(f"ncnn extract failed with code {_ncnn_code(code_extract)}")
        empty_attr = getattr(output_mat, "empty", None) if output_mat is not None else True
        is_empty = empty_attr() if callable(empty_attr) else bool(empty_attr)
        if output_mat is None or is_empty:
            raise RuntimeError("ncnn extractor returned an empty output")
        # Newer Python wheels expose ``numpy()`` on Mat.  When available, use
        # it to reject a graph that technically loads but emits non-finite
        # values; older wheels are still covered by the non-empty extractor
        # check above.
        to_numpy = getattr(output_mat, "numpy", None)
        if callable(to_numpy):
            values = np.asarray(to_numpy())
            if values.size == 0 or not np.isfinite(values).all():
                raise RuntimeError("ncnn extractor returned non-finite or empty output values")
    log_lines = [line for line in log_buffer.getvalue().splitlines() if line.strip()]
    unsupported = [
        line for line in log_lines
        if "not exists or registered" in line.lower() or "unsupported" in line.lower()
    ]
    if unsupported:
        raise RuntimeError("NCNN rejected graph operators: " + " | ".join(unsupported[-5:]))
    noop_layers = sorted({
        line.split()[0]
        for line in param.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]
        if line.startswith("torch.")
    })
    if noop_layers:
        raise RuntimeError("NCNN param contains unsupported passthrough layers: " + ", ".join(noop_layers))
    return {
        "input_blob": input_name,
        "output_blob": output_name,
        "load_checked": True,
        "noop_layers": [],
        "log_tail": log_lines[-5:],
    }


def export_ncnn(model, args: argparse.Namespace, out_dir: Path) -> dict:
    # Ultralytics/pnnx can raise after writing param/bin while generating its
    # optional model_pnnx.py.  Keep the expected directory so we can still run
    # the real NCNN load check below.
    expected_dir = args.model.with_name(args.model.stem + "_ncnn_model")
    before_pair = _ncnn_pair_fingerprint(expected_dir)
    state, router_count, esmoe_count = _force_ncnn_dense(model)
    overlap_count = _routing_overlap(state)
    exported_value = None
    note = None
    export_error = None
    try:
        try:
            exported_value = model.export(
                format="ncnn", imgsz=args.imgsz, half=args.half
            )
        except SyntaxError as exc:
            export_error = exc
            note = f"pnnx reference-script SyntaxError (param/bin may still be valid): {exc}"
            exported_value = expected_dir
        except Exception as exc:
            export_error = exc
            # Some Ultralytics releases wrap the generated reference-script
            # SyntaxError in RuntimeError.  Continue only when a complete pair
            # was actually written; never mask a genuine exporter failure.
            try:
                _find_ncnn_pair(expected_dir)
            except (FileNotFoundError, OSError):
                raise
            note = f"pnnx post-codegen error (param/bin validated): {exc}"
            exported_value = expected_dir
    finally:
        for module, attribute, value in reversed(state):
            setattr(module, attribute, value)
    exported_location = Path(exported_value or expected_dir)
    pair_location = (
        exported_location
        if exported_location.is_dir() or (not exported_location.exists() and not exported_location.suffix)
        else exported_location.parent
    )
    after_pair = _ncnn_pair_fingerprint(pair_location)
    same_expected_location = pair_location.resolve() == expected_dir.resolve()
    if export_error is not None and before_pair is not None and same_expected_location and after_pair == before_pair:
        raise RuntimeError(
            "pnnx failed without changing the existing NCNN pair; refusing to reuse a stale export"
        ) from export_error
    if export_error is None and before_pair is not None and same_expected_location and after_pair == before_pair:
        raise RuntimeError(
            "pnnx returned successfully without changing the existing NCNN pair; refusing to reuse a stale export"
        )
    if router_count == 0 and esmoe_count == 0:
        raise RuntimeError(
            "NCNN export did not find an ES_MOE/DynamicRoutingLayer to switch to dense routing; "
            "refusing an unverifiable acceptance artifact"
        )
    exported = exported_location
    param, binary = _find_ncnn_pair(exported)
    lines = param.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines or not lines[0].split() or lines[0].split()[0] not in {"7767517", "7767518"}:
        raise RuntimeError(f"unexpected NCNN param header in {param}")
    destination = out_dir / param.stem.replace(".ncnn", "")
    destination.mkdir(parents=True, exist_ok=True)
    param_out = _copy_file(param, destination / param.name)
    bin_out = _copy_file(binary, destination / binary.name)
    input_blob, output_blob = _param_io_names(param_out)
    terminal_outputs = _param_output_names(param_out)
    proto_blob = next((name for name in terminal_outputs if name != output_blob), None)
    metadata = _write_ncnn_metadata(
        destination, model, args.imgsz, input_blob, output_blob, proto_blob
    )
    # Keep a per-model sidecar alongside the shared legacy filename.  This is
    # important when several NCNN graphs are unpacked into one release
    # directory: the C++ runtime gives the stem-specific file precedence and
    # cannot accidentally apply another graph's blob names.
    metadata_per_model = destination / f"{param_out.stem}.metadata.yaml"
    if metadata_per_model.resolve() != metadata.resolve():
        _copy_file(metadata, metadata_per_model)
    smoke = _ncnn_smoke_check(param_out, bin_out, args.imgsz)
    result = {
        "format": "ncnn",
        "directory": str(destination),
        "param": str(param_out),
        "bin": str(bin_out),
        "metadata": str(metadata),
        "metadata_per_model": str(metadata_per_model),
        **_routing_record(router_count, esmoe_count, overlap_count),
        # Retain the detailed legacy keys for consumers of earlier summaries.
        "routing": {"routers_dense": router_count, "esmoe_dense": esmoe_count},
        "checked": True,
        "acceptance_ready": True,
        **smoke,
    }
    if note:
        result["export_note"] = note
    return result


def export_mnn(
    model,
    args: argparse.Namespace,
    out_dir: Path,
    onnx_path: Optional[Path],
    onnx_result: Optional[dict] = None,
) -> dict:
    del model  # MNN conversion intentionally consumes the canonical ONNX graph.
    if onnx_path is None:
        raise ValueError("MNN conversion requires a successful ONNX export first")
    destination = out_dir / onnx_path.with_suffix(".mnn").name
    command = [
        args.mnn_convert, "-f", "ONNX", "--modelFile", str(onnx_path),
        "--MNNModel", str(destination), "--bizCode", "edge",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if (
        completed.returncode != 0
        or not destination.is_file()
        or destination.stat().st_size == 0
    ):
        raise RuntimeError(
            f"mnnconvert failed ({completed.returncode}): {completed.stderr[-1000:]}"
        )
    routing = {}
    if isinstance(onnx_result, dict):
        for key in ("routing_semantics", "routing_layers"):
            if key in onnx_result:
                routing[key] = onnx_result[key]
    return {
        "format": "mnn",
        "path": str(destination),
        # mnnconvert validates serialization only. Loading/executing the
        # artifact requires the optional MNN Python runtime and representative
        # input data, so do not label a file acceptance-ready here.
        "checked": True,
        "checked_scope": "converter_output",
        "runtime_smoke_checked": False,
        "acceptance_ready": False,
        "parity_required": True,
        "parity_command": "python scripts/mnn_parity.py --mnn <file> --onnx <file> --images <dir>",
        **routing,
    }


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.model}")
    if args.imgsz <= 0 or args.opset < 7:
        raise ValueError("imgsz must be positive and opset must be >= 7")
    if "mnn" in args.formats and "onnx" not in args.formats:
        raise ValueError("MNN conversion requires 'onnx' in --formats")
    if not args.simplify and not args.allow_unsimplified:
        raise ValueError(
            "the default submission workflow requires onnxsim; "
            "pair --no-simplify with --allow-unsimplified only for debugging"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    results, errors = [], []
    onnx_path: Optional[Path] = None
    onnx_result: Optional[dict] = None
    # A user may provide formats in any order; dependencies are still emitted
    # deterministically and MNN always sees the canonical ONNX artifact.
    ordered_formats = [fmt for fmt in ("onnx", "ncnn", "mnn") if fmt in args.formats]
    for fmt in ordered_formats:
        try:
            if fmt == "onnx":
                result = export_onnx(model, args, args.out_dir)
                onnx_path = Path(result["path"])
                onnx_result = result
            elif fmt == "ncnn":
                result = export_ncnn(model, args, args.out_dir)
            else:
                result = export_mnn(model, args, args.out_dir, onnx_path, onnx_result)
            results.append(result)
            print(f"[OK] {fmt}")
        except Exception as exc:
            error = {"format": fmt, "error": f"{type(exc).__name__}: {exc}"}
            errors.append(error)
            results.append(error)
            print(f"[FAIL] {fmt}: {error['error']}")
    summary = args.out_dir / "export_summary.json"
    summary.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary: {summary}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
