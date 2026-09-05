#!/usr/bin/env python3
"""Compare raw MNN and ONNX outputs on an identical image list.

This is a numerical parity check, not an mAP evaluator. It exits non-zero when
the normalized output tensors differ beyond ``--tolerance`` or contain a shape
mismatch, making it suitable for CI and deployment smoke tests.
"""

from __future__ import annotations

import argparse
import json
import math
import numbers
import time
from pathlib import Path


# Keep the parity image list identical to the portable C++ runner's stb decoder.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
LIST_EXTS = {".txt", ".list"}


def status_failed(code) -> bool:
    """Normalize MNN's int/bool status conventions (0/True means success)."""
    if isinstance(code, bool):
        return not code
    return isinstance(code, numbers.Integral) and int(code) != 0


def tensor_to_numpy(tensor, shape):
    """Read a host tensor across old and new MNN Python wheel APIs."""
    import numpy as np

    getter = getattr(tensor, "getNumpyData", None)
    if callable(getter):
        values = getter()
    else:
        getter = getattr(tensor, "getData", None)
        if not callable(getter):
            raise RuntimeError("MNN host tensor exposes neither getNumpyData() nor getData()")
        values = getter()
    return np.asarray(values, dtype=np.float32).reshape(tuple(shape))


def image_list(source: Path, limit: int) -> list[Path]:
    """Resolve a directory, image, or ordered UTF-8 image list."""
    source = source.expanduser()
    if not source.exists():
        raise FileNotFoundError(f"validation source not found: {source}")
    if source.is_file() and source.suffix.lower() in IMAGE_EXTS:
        paths = [source.resolve()]
    elif source.is_file() and source.suffix.lower() in LIST_EXTS:
        base = source.resolve().parent
        paths = []
        with source.open("r", encoding="utf-8-sig") as handle:
            for line_number, raw in enumerate(handle, 1):
                value = raw.strip()
                if not value or value.startswith("#"):
                    continue
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                    value = value[1:-1].strip()
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = base / path
                path = path.resolve()
                if not path.is_file():
                    raise FileNotFoundError(
                        f"image list line {line_number} does not name a file: {path}"
                    )
                if path.suffix.lower() not in IMAGE_EXTS:
                    raise ValueError(
                        f"unsupported image extension at list line {line_number}: {path}"
                    )
                paths.append(path)
    elif source.is_dir():
        paths = sorted(
            (path.resolve() for path in source.rglob("*")
             if path.is_file() and path.suffix.lower() in IMAGE_EXTS),
            # Keep directory discovery identical to mnn_val.py and the
            # publication-grade evaluators.
            key=lambda path: (path.as_posix().casefold(), path.as_posix()),
        )
    else:
        raise ValueError(f"unsupported validation source: {source}")
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise RuntimeError(f"no validation images found under {source}")
    stems = [path.stem.casefold() for path in paths]
    if len(stems) != len(set(stems)):
        raise RuntimeError("validation image stems are not unique; flatten/rename the split before parity checking")
    return paths


def letterbox(path: Path, size: int):
    import cv2
    import numpy as np

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise RuntimeError(f"unable to read image: {path}")
    h, w = img.shape[:2]
    ratio = min(size / h, size / w)
    nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
    canvas = np.full((size, size, 3), 114, np.uint8)
    # Match ultralytics.data.augment.LetterBox's left/top padding rule.
    px, py = round((size - nw) / 2 - 0.1), round((size - nh) / 2 - 0.1)
    canvas[py : py + nh, px : px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rgb = canvas[:, :, ::-1].astype("float32") / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def normalize_output(raw, expected_feat: int):
    """Return a detection tensor as ``(features, anchors)``.

    Exporters disagree on whether the feature dimension precedes anchors and
    whether a singleton batch/channel dimension is retained. Normalize those
    harmless layout differences before comparing values.
    """
    import numpy as np

    arr = np.asarray(raw, dtype=np.float32)
    while arr.ndim > 2 and 1 in (arr.shape[0], arr.shape[-1]):
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
    if arr.ndim == 2 and 1 in arr.shape and arr.size != expected_feat:
        arr = np.squeeze(arr)
    if arr.ndim == 1 and arr.size % expected_feat == 0:
        return np.ascontiguousarray(arr.reshape(expected_feat, -1))
    if arr.ndim == 2:
        if arr.shape[0] == expected_feat:
            return np.ascontiguousarray(arr)
        if arr.shape[1] == expected_feat:
            return np.ascontiguousarray(arr.T)
    if arr.ndim == 3 and arr.shape[0] == expected_feat:
        return np.ascontiguousarray(arr.reshape(expected_feat, -1))
    if arr.ndim == 3 and arr.shape[-1] == expected_feat:
        if arr.shape[0] > 1 and arr.shape[1] > 1:
            raise ValueError(f"unsupported detection output shape {arr.shape}; non-singleton batch dimension")
        return np.ascontiguousarray(arr.reshape(-1, expected_feat).T)
    raise ValueError(f"unsupported detection output shape {arr.shape}; expected feature width {expected_feat}")


def select_detection_output(outputs, expected_feat: int):
    """Select the first ONNX output that has a YOLO detection layout."""
    for raw in outputs:
        try:
            normalize_output(raw, expected_feat)
        except ValueError:
            continue
        return raw
    raise ValueError(f"ONNX model exposes no output with feature width {expected_feat}")


def mnn_run(interpreter, session, input_tensor, batch):
    import MNN
    import numpy as np

    shape = list(input_tensor.getShape())
    if len(shape) != 4 or 3 not in (shape[1], shape[-1]):
        raise RuntimeError(f"unsupported MNN input shape {shape}; expected NCHW or NHWC with 3 channels")
    data = batch
    dimension_type = MNN.Tensor_DimensionType_Caffe
    if len(shape) == 4 and shape[-1] == 3 and shape[1] != 3:
        data = np.transpose(batch, (0, 2, 3, 1))
        dimension_type = getattr(MNN, "Tensor_DimensionType_Tensorflow", MNN.Tensor_DimensionType_Caffe)
    host = MNN.Tensor(shape, MNN.Halide_Type_Float, np.ascontiguousarray(data), dimension_type)
    copy_input = getattr(input_tensor, "copyFromHostTensor", None)
    if not callable(copy_input):
        copy_input = getattr(input_tensor, "copyFrom", None)
    if not callable(copy_input):
        raise RuntimeError("MNN input tensor exposes neither copyFromHostTensor() nor copyFrom()")
    code = copy_input(host)
    if status_failed(code):
        raise RuntimeError(f"MNN copyFromHostTensor failed with code {code}")
    run_code = interpreter.runSession(session)
    if status_failed(run_code):
        raise RuntimeError(f"MNN runSession failed with code {run_code}")
    output = get_session_output(interpreter, session)
    if output is None:
        raise RuntimeError("MNN session exposes no output tensor")
    output_shape = list(output.getShape())
    host_output = MNN.Tensor(
        output_shape, MNN.Halide_Type_Float, np.zeros(output_shape, dtype=np.float32),
        MNN.Tensor_DimensionType_Caffe,
    )
    copy_output = getattr(output, "copyToHostTensor", None)
    if not callable(copy_output):
        copy_output = getattr(output, "copyTo", None)
    if not callable(copy_output):
        raise RuntimeError("MNN output tensor exposes neither copyToHostTensor() nor copyTo()")
    code = copy_output(host_output)
    if status_failed(code):
        raise RuntimeError(f"MNN copyToHostTensor failed with code {code}")
    return tensor_to_numpy(host_output, output_shape)


def get_session_output(interpreter, session):
    """Get the primary MNN output across binding API variants."""
    try:
        output = interpreter.getSessionOutput(session)
    except TypeError:
        output = None
    if hasattr(output, "getShape"):
        return output
    if isinstance(output, dict):
        candidates = output.values()
    elif isinstance(output, (tuple, list)):
        candidates = output
    else:
        candidates = ()
    for candidate in candidates:
        if hasattr(candidate, "getShape"):
            return candidate
    # Some older bindings require an explicit output name. Try the names used
    # by the exporter before falling back to the all-outputs API.
    for name in ("output0", "out0", "output", "out"):
        try:
            candidate = interpreter.getSessionOutput(session, name)
        except (AttributeError, KeyError, TypeError):
            continue
        if hasattr(candidate, "getShape"):
            return candidate
    get_all = getattr(interpreter, "getSessionOutputAll", None)
    if callable(get_all):
        outputs = get_all(session)
        candidates = outputs.values() if isinstance(outputs, dict) else outputs or ()
        for candidate in candidates:
            if hasattr(candidate, "getShape"):
                return candidate
    raise RuntimeError("MNN session exposes no tensor output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnn", type=Path, default=Path("models/esmoe_n_visdrone.mnn"))
    parser.add_argument("--onnx", type=Path, default=Path("models/esmoe_n_visdrone_sim.onnx"))
    parser.add_argument("--images", type=Path, default=Path("/data/datasets/VisDrone/images/val"))
    parser.add_argument(
        "--limit", "--n", dest="n", type=int, default=100,
        help="number of ordered images to compare (default: 100; --n is kept as a compatibility alias)",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--nc", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--json", type=Path, help="optional JSON report path")
    parser.add_argument(
        "--debug-dir", type=Path,
        help="optional directory for input/MNN/ONNX .npy snapshots on the first mismatch",
    )
    parser.add_argument("--debug-limit", type=int, default=1, help="number of successful samples to snapshot")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.mnn.is_file() or not args.onnx.is_file():
        raise FileNotFoundError(f"missing model: mnn={args.mnn}, onnx={args.onnx}")
    if args.n <= 0 or args.imgsz <= 0 or args.threads <= 0 or args.tolerance < 0 or args.debug_limit < 0:
        raise ValueError("n, imgsz, threads must be positive; tolerance/debug-limit non-negative")
    if not math.isfinite(float(args.tolerance)):
        raise ValueError("tolerance must be finite")
    paths = image_list(args.images, args.n)

    import MNN
    import numpy as np
    import onnxruntime as ort
    import cv2  # noqa: F401  # fail early with a useful dependency error

    interpreter = MNN.Interpreter(str(args.mnn))
    session = interpreter.createSession({"numThread": args.threads, "backend": "CPU"})
    if session is None:
        raise RuntimeError("MNN failed to create a CPU session")
    input_tensor = interpreter.getSessionInput(session)
    if input_tensor is None:
        raise RuntimeError("MNN failed to create a CPU session or expose its input tensor")
    mnn_input_shape = list(input_tensor.getShape())
    if len(mnn_input_shape) != 4 or 3 not in (mnn_input_shape[1], mnn_input_shape[-1]):
        raise RuntimeError(
            f"unsupported MNN input shape {mnn_input_shape}; expected NCHW or NHWC with 3 channels"
        )
    if mnn_input_shape[1] == 3:
        mnn_h, mnn_w = mnn_input_shape[2:4]
    else:
        mnn_h, mnn_w = mnn_input_shape[1:3]
    try:
        concrete_mnn_hw = (int(mnn_h), int(mnn_w))
    except (TypeError, ValueError):
        concrete_mnn_hw = None
    if concrete_mnn_hw and all(value > 0 for value in concrete_mnn_hw) and concrete_mnn_hw != (args.imgsz, args.imgsz):
        raise ValueError(
            f"--imgsz={args.imgsz} does not match static MNN input [{concrete_mnn_hw[0]}, {concrete_mnn_hw[1]}]"
        )
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    ort_session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=["CPUExecutionProvider"])
    ort_inputs = ort_session.get_inputs()
    if not ort_inputs:
        raise RuntimeError("ONNX session exposes no input tensor")
    input_name = ort_inputs[0].name
    input_type = getattr(ort_inputs[0], "type", "tensor(float)")
    if input_type == "tensor(float16)":
        ort_dtype = np.float16
    elif input_type == "tensor(float)":
        ort_dtype = np.float32
    else:
        raise ValueError(f"unsupported ONNX input type {input_type}; expected float or float16")
    ort_shape = getattr(ort_inputs[0], "shape", None)
    if ort_shape and len(ort_shape) == 4:
        spatial = ort_shape[2:4] if ort_shape[1] == 3 else ort_shape[1:3] if ort_shape[-1] == 3 else ()
        if spatial and all(isinstance(value, int) and value > 0 for value in spatial):
            if tuple(spatial) != (args.imgsz, args.imgsz):
                raise ValueError(
                    f"--imgsz={args.imgsz} does not match static ONNX input [{spatial[0]}, {spatial[1]}]"
                )

    max_error = 0.0
    mean_error = 0.0
    count = 0
    mnn_times, ort_times = [], []
    debug_saved = 0
    for path in paths:
        batch = letterbox(path, args.imgsz)
        start = time.perf_counter(); mnn_raw = mnn_run(interpreter, session, input_tensor, batch)
        mnn_times.append((time.perf_counter() - start) * 1000.0)
        ort_batch = batch.astype(ort_dtype, copy=False)
        start = time.perf_counter(); ort_outputs = ort_session.run(None, {input_name: ort_batch})
        ort_raw = select_detection_output(ort_outputs, 4 + args.nc)
        ort_times.append((time.perf_counter() - start) * 1000.0)
        mnn_out = normalize_output(mnn_raw, 4 + args.nc)
        ort_out = normalize_output(ort_raw, 4 + args.nc)
        if args.debug_dir is not None and debug_saved < args.debug_limit:
            args.debug_dir.mkdir(parents=True, exist_ok=True)
            np.save(args.debug_dir / f"{path.stem}_input.npy", batch)
            np.save(args.debug_dir / f"{path.stem}_mnn.npy", mnn_out)
            np.save(args.debug_dir / f"{path.stem}_onnx.npy", ort_out)
            debug_saved += 1
        if mnn_out.shape != ort_out.shape:
            if args.debug_dir is not None:
                args.debug_dir.mkdir(parents=True, exist_ok=True)
                np.save(args.debug_dir / f"{path.stem}_mnn_raw.npy", mnn_raw)
                np.save(args.debug_dir / f"{path.stem}_onnx_raw.npy", ort_raw)
            raise ValueError(f"shape mismatch for {path.name}: MNN {mnn_out.shape}, ONNX {ort_out.shape}")
        if not np.isfinite(mnn_out).all() or not np.isfinite(ort_out).all():
            raise ValueError(f"non-finite output tensor for {path.name}")
        delta = np.abs(mnn_out - ort_out)
        max_error = max(max_error, float(delta.max(initial=0.0)))
        mean_error += float(delta.mean())
        count += 1
        if float(delta.max(initial=0.0)) > args.tolerance and args.debug_dir is not None:
            args.debug_dir.mkdir(parents=True, exist_ok=True)
            np.save(args.debug_dir / f"{path.stem}_mismatch_delta.npy", delta)

    mean_error /= count
    report = {
        "images": count,
        "max_abs_error": max_error,
        "mean_abs_error": mean_error,
        "tolerance": args.tolerance,
        "passed": max_error <= args.tolerance,
        "mnn_mean_ms": float(np.mean(mnn_times)),
        "onnx_mean_ms": float(np.mean(ort_times)),
        "mnn_fps": float(1000.0 / np.mean(mnn_times)),
        "onnx_fps": float(1000.0 / np.mean(ort_times)),
    }
    if args.debug_dir is not None:
        report["debug_dir"] = str(args.debug_dir)
    print(json.dumps(report, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
