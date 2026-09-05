#!/usr/bin/env python3
"""Run MNN over a fixed validation set and dump C++-compatible predictions."""

from __future__ import annotations

import argparse
import math
import numbers
from pathlib import Path

# Keep the validation list identical to the portable C++ runner's stb decoder.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
LIST_EXTS = {".txt", ".list"}


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
            # Match the case-folded, separator-normalized order used by the
            # C++ runner, evaluator, and evidence manifest.  The original
            # spelling is a deterministic tie-break for case-sensitive hosts.
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
        raise RuntimeError(
            "validation image stems are not unique; flatten/rename the split before dumping predictions"
        )
    return paths


def status_failed(code) -> bool:
    """Normalize MNN's int/bool status conventions (0/True means success)."""
    if isinstance(code, bool):
        return not code
    return isinstance(code, numbers.Integral) and int(code) != 0


def tensor_to_numpy(tensor, shape):
    """Copy an MNN host tensor using the API exposed by the installed wheel.

    ``getNumpyData`` is the public method in current MNN wheels; older wheels
    expose only ``getData``.  Supporting both keeps validation useful across
    the Python bindings without changing the numerical contract.
    """
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


def normalize_output(raw, expected_feat: int):
    import numpy as np

    arr = np.asarray(raw, dtype=np.float32)
    while arr.ndim > 2 and 1 in (arr.shape[0], arr.shape[-1]):
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr[..., 0]
    if arr.ndim == 2 and 1 in arr.shape and arr.size != expected_feat:
        arr = np.squeeze(arr)
    if arr.ndim == 1 and arr.size % expected_feat == 0:
        return np.ascontiguousarray(arr.reshape(expected_feat, -1))
    if arr.ndim == 2 and arr.shape[0] == expected_feat:
        return np.ascontiguousarray(arr)
    if arr.ndim == 2 and arr.shape[1] == expected_feat:
        return np.ascontiguousarray(arr.T)
    if arr.ndim == 3 and arr.shape[0] == expected_feat:
        return np.ascontiguousarray(arr.reshape(expected_feat, -1))
    if arr.ndim == 3 and arr.shape[-1] == expected_feat:
        if arr.shape[0] > 1 and arr.shape[1] > 1:
            raise ValueError(f"unsupported output shape {arr.shape}; non-singleton batch dimension")
        return np.ascontiguousarray(arr.reshape(-1, expected_feat).T)
    raise ValueError(f"unsupported output shape {arr.shape}; expected feature width {expected_feat}")


def letterbox(path: Path, size: int):
    import cv2
    import numpy as np

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise RuntimeError(f"unable to read image: {path}")
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
    canvas = np.full((size, size, 3), 114, np.uint8)
    # Match ultralytics.data.augment.LetterBox's left/top padding rule.
    px, py = round((size - nw) / 2 - 0.1), round((size - nh) / 2 - 0.1)
    canvas[py : py + nh, px : px + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    batch = np.ascontiguousarray((canvas[:, :, ::-1].astype(np.float32) / 255).transpose(2, 0, 1)[None])
    return batch, ratio, px, py, w, h


def nms(boxes, scores, iou_thr: float, max_keep: int = 300):
    import numpy as np

    if len(boxes) == 0 or max_keep <= 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    # Stable score-descending order with an explicit original-index tie-break;
    # default quicksort ordering can vary between NumPy builds and change NMS
    # survivors when quantized outputs contain equal scores.
    order = np.lexsort((np.arange(scores.size, dtype=np.int64), -scores))
    keep = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if len(keep) >= max_keep:
            break
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest]); yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest]); yy2 = np.minimum(y2[current], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[current] + areas[rest] - inter
        overlap = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[overlap <= iou_thr]
    return keep


def class_nms_offset(width: int, height: int, boxes=None) -> float:
    """Return the class-stratification offset used by the C++ decoder.

    Boxes are shifted before a single greedy NMS pass. The inverse letterbox
    transform can produce coordinates outside the image, so the offset leaves
    a margin larger than any possible box extent.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    # Match the C++ runner: derive the separation from the actual unclipped
    # candidates so a finite out-of-frame prediction cannot make classes
    # overlap after stratification.  Keep the historical frame-based value
    # when no candidate array is supplied for compatibility with callers.
    extent = float(max(width, height))
    if boxes is not None:
        import numpy as np
        values = np.asarray(boxes, dtype=np.float64)
        if values.size:
            if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
                raise ValueError("boxes must be a finite Nx4 array")
            extent = max(extent, float(np.max(np.abs(values))))
    return 2.0 * extent + 1.0 if boxes is not None else 2.0 * extent + 8192.0


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
    parser.add_argument("--images", type=Path, default=Path("/data/datasets/VisDrone/images/val"))
    parser.add_argument("--out", type=Path, default=Path("preds_mnn"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--nc", type=int, default=10)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument(
        "--small-conf", type=float, default=-1.0,
        help="optional threshold for boxes below --small-area",
    )
    parser.add_argument("--small-area", type=float, default=32.0 * 32.0)
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.mnn.is_file():
        raise FileNotFoundError(f"MNN model not found: {args.mnn}")
    if args.imgsz <= 0 or args.threads <= 0 or args.nc <= 0 or args.max_det <= 0:
        raise ValueError("imgsz, threads, nc, max-det must be positive")
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    if any(
        not math.isfinite(float(value))
        for value in (args.conf, args.iou, args.small_conf, args.small_area)
    ):
        raise ValueError("conf/iou/small-conf/small-area must be finite")
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1 or not -1 <= args.small_conf <= 1 or args.small_area < 0:
        raise ValueError("conf/iou must be in [0,1], small-conf in [-1,1], and small-area non-negative")
    image_paths = image_list(args.images, args.limit)

    import MNN
    import numpy as np

    interpreter = MNN.Interpreter(str(args.mnn))
    session = interpreter.createSession({"numThread": args.threads, "backend": "CPU"})
    if session is None:
        raise RuntimeError("MNN failed to create a CPU session")
    input_tensor = interpreter.getSessionInput(session)
    if input_tensor is None:
        raise RuntimeError("MNN failed to create a CPU session or expose its input tensor")
    model_input_shape = list(input_tensor.getShape())
    if len(model_input_shape) != 4 or 3 not in (model_input_shape[1], model_input_shape[-1]):
        raise RuntimeError(
            f"unsupported MNN input shape {model_input_shape}; expected NCHW or NHWC with 3 channels"
        )
    if model_input_shape[1] == 3:
        model_h, model_w = model_input_shape[2:4]
    else:
        model_h, model_w = model_input_shape[1:3]
    try:
        concrete_hw = (int(model_h), int(model_w))
    except (TypeError, ValueError):
        concrete_hw = None
    if concrete_hw and all(value > 0 for value in concrete_hw) and concrete_hw != (args.imgsz, args.imgsz):
        raise ValueError(
            f"--imgsz={args.imgsz} does not match static MNN input [{concrete_hw[0]}, {concrete_hw[1]}]"
        )

    def run(batch):
        shape = model_input_shape
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
            raise RuntimeError(f"MNN input failed with code {code}")
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
            raise RuntimeError(f"MNN output copy failed with code {code}")
        values = tensor_to_numpy(host_output, output_shape)
        if not np.isfinite(values).all():
            raise RuntimeError("MNN output contains NaN or Inf")
        return normalize_output(values, 4 + args.nc)

    args.out.mkdir(parents=True, exist_ok=True)
    for path in image_paths:
        batch, ratio, px, py, width, height = letterbox(path, args.imgsz)
        output = run(batch)
        boxes, classes = output[:4], output[4:]
        # Decode boxes first so the optional area-adaptive threshold is measured
        # in original-image pixels, exactly like the C++ runner.
        cx_all, cy_all = boxes[0], boxes[1]
        bw_raw, bh_raw = boxes[2], boxes[3]
        valid_geometry = (
            np.isfinite(np.stack((cx_all, cy_all, bw_raw, bh_raw), axis=0)).all(axis=0)
            & (bw_raw > 0.0)
            & (bh_raw > 0.0)
        )
        # Keep invalid/degenerate anchors out of thresholding and NMS, matching
        # the C++ decoder's finite and positive width/height guards.
        bw_all, bh_all = bw_raw, bh_raw
        x1_all = (cx_all - 0.5 * bw_all - px) / ratio
        y1_all = (cy_all - 0.5 * bh_all - py) / ratio
        x2_all = (cx_all + 0.5 * bw_all - px) / ratio
        y2_all = (cy_all + 0.5 * bh_all - py) / ratio
        candidate_threshold = np.full(classes.shape, args.conf, dtype=np.float32)
        if args.small_conf >= 0:
            area_all = np.maximum(0.0, x2_all - x1_all) * np.maximum(0.0, y2_all - y1_all)
            candidate_threshold[:, :] = np.where(
                area_all[None, :] < args.small_area,
                min(args.conf, args.small_conf),
                args.conf,
            )
        # The C++ runner and Ultralytics validation both retain candidates at
        # the declared threshold, so use an inclusive ``>= conf`` filter.
        candidate_mask = (classes >= candidate_threshold) & valid_geometry[None, :]
        class_ids, anchors = np.where(candidate_mask)
        if class_ids.size:
            scores = classes[class_ids, anchors]
            x1, y1 = x1_all[anchors], y1_all[anchors]
            x2, y2 = x2_all[anchors], y2_all[anchors]
            xyxy = np.stack([x1, y1, x2, y2], axis=1)
            finite = np.isfinite(xyxy).all(axis=1) & np.isfinite(scores)
            class_ids, scores, xyxy = class_ids[finite], scores[finite], xyxy[finite]
            if scores.size > 30000:
                # Match Ultralytics' max_nms guard before the quadratic
                # suppression loop, keeping every parallel array aligned.
                order = np.lexsort((np.arange(scores.size, dtype=np.int64), -scores))[:30000]
                class_ids, scores, xyxy = class_ids[order], scores[order], xyxy[order]
            shifted = xyxy + class_ids[:, None] * class_nms_offset(width, height, xyxy)
            keep = nms(shifted, scores, args.iou, args.max_det)
        else:
            keep, scores, xyxy = [], np.empty(0), np.empty((0, 4))
            class_ids = np.empty(0, dtype=np.int64)
        target = args.out / f"{path.stem}.txt"
        with target.open("w", encoding="utf-8") as handle:
            for index in keep:
                bx1 = max(0.0, min(float(xyxy[index, 0]), width))
                by1 = max(0.0, min(float(xyxy[index, 1]), height))
                bx2 = max(0.0, min(float(xyxy[index, 2]), width))
                by2 = max(0.0, min(float(xyxy[index, 3]), height))
                if bx2 > bx1 and by2 > by1:
                    handle.write(
                        f"{int(class_ids[index])} {float(scores[index]):.9g} "
                        f"{bx1:.9g} {by1:.9g} {bx2:.9g} {by2:.9g}\n"
                    )
    print(f"dumped predictions for {len(image_paths)} images -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
