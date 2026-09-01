#!/usr/bin/env python3
"""Static INT8 post-training quantization for the EsMoE-N ONNX export.

The calibration path deliberately shares the C++ runner's preprocessing:
letterbox (114 padding), BGR-to-RGB conversion, /255 normalization and NCHW
layout. A minimum of 300 *training* images is enforced so the validation set
is not leaked into calibration and the Issue #51 acceptance gate is explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Union

# Calibration must use formats the portable C++ runner can decode as well.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# The EsMoE head, attention softmaxes, and routing softmax are precision
# sensitive.  Keeping them FP32 is the documented recipe that stays within the
# Issue #51 <1% mAP50-95 INT8 budget; callers can override it deliberately.
DEFAULT_EXCLUDE = ("/model.25/", "/attn/", "routing")


def _image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.as_posix().casefold(),
    )


def _validate_unique_stems(paths: Iterable[Path], label: str) -> None:
    """Reject ambiguous image names before writing an ordered manifest."""
    seen: dict[str, Path] = {}
    for path in paths:
        stem = path.stem.casefold()
        if stem in seen:
            raise ValueError(
                f"{label} image stems are not unique: {stem} ({seen[stem]} and {path})"
            )
        seen[stem] = path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_name(path: Path, root: Path) -> str:
    """Return a mount-independent, POSIX-formatted calibration path."""
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError:
        # Symlinked files may resolve outside the nominal training root. Keep
        # the relationship to that root rather than embedding its mount point.
        try:
            relative = Path(os.path.relpath(resolved_path, resolved_root))
        except ValueError:
            # Different Windows drives have no common relative spelling. The
            # selected image list is already ordered and stem-unique, so the
            # basename is the only portable fallback available here.
            relative = Path(resolved_path.name)
    return relative.as_posix()


def select_calibration_images(directory: Path, count: int) -> list[Path]:
    """Select ``count`` deterministic, evenly-spaced calibration images."""
    if count < 300:
        raise ValueError("--n-calib must be at least 300 to satisfy the Issue #51 protocol")
    images = _image_paths(directory)
    if len(images) < count:
        raise ValueError(f"calibration set contains {len(images)} images, but {count} were requested")
    _validate_unique_stems(images, "calibration")
    indices = [(i * len(images)) // count for i in range(count)]
    return [images[i] for i in indices]


def letterbox(img_path: Union[Path, str], imgsz: int = 640):
    """Load an image and return a contiguous float32 NCHW batch."""
    import cv2
    import numpy as np

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError(f"unable to read calibration image: {img_path}")
    if imgsz <= 0:
        raise ValueError("imgsz must be positive")
    h, w = img.shape[:2]
    ratio = min(imgsz / h, imgsz / w)
    nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
    canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
    # Match ultralytics.data.augment.LetterBox: odd padding goes to the
    # right/bottom edge via the -0.1 tie-break on the left/top half.
    px, py = round((imgsz - nw) / 2 - 0.1), round((imgsz - nh) / 2 - 0.1)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas[py : py + nh, px : px + nw] = resized
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None])


class CalibrationReader:
    """ONNX Runtime calibration reader with rewind support."""

    def __init__(self, images: Iterable[Path], input_name: str, imgsz: int):
        self.images = list(images)
        self.input_name = input_name
        self.imgsz = imgsz
        self._iter = iter(self.images)

    def get_next(self):
        image = next(self._iter, None)
        return None if image is None else {self.input_name: letterbox(image, self.imgsz)}

    def rewind(self):
        self._iter = iter(self.images)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", type=Path, default=Path("models/esmoe_n_visdrone_sim.onnx"))
    parser.add_argument("--train", type=Path, default=Path("/data/datasets/VisDrone/images/train"))
    parser.add_argument(
        "--validation-images", type=Path, default=None,
        help=(
            "optional validation image directory used to prove calibration/validation "
            "disjointness by SHA256; required for an acceptance-ready manifest"
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("models/esmoe_n_visdrone_int8.onnx"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--n-calib", type=int, default=500)
    parser.add_argument("--method", default="MinMax", choices=("MinMax", "Entropy", "Percentile"))
    parser.add_argument(
        "--format", default="QOperator", choices=("QDQ", "QOperator"),
        help="quantized graph format (QOperator is the Issue #51 CPU recipe)",
    )
    parser.add_argument(
        "--symmetric", action="store_true",
        help=(
            "use symmetric activation/weight quantization and leave bias in FP32; "
            "required when the QDQ artifact is parsed by TensorRT"
        ),
    )
    parser.add_argument(
        "--exclude", nargs="*", default=None,
        help="node-name substrings to keep in FP32 (default: head, attention, routing)",
    )
    parser.add_argument(
        "--no-default-exclude", action="store_true",
        help="quantize every eligible node; diagnostic only and not the acceptance recipe",
    )
    return parser.parse_args()


def _restore_metadata(source, target) -> None:
    props = [(prop.key, prop.value) for prop in source.metadata_props]
    del target.metadata_props[:]
    for key, value in props:
        entry = target.metadata_props.add()
        entry.key, entry.value = key, value


def main() -> int:
    args = parse_args()
    if not args.fp32.is_file():
        raise FileNotFoundError(f"FP32 ONNX model not found: {args.fp32}")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive")
    if args.out.resolve() == args.fp32.resolve():
        raise ValueError("--out must differ from --fp32; in-place quantization would destroy the source model")
    prep_path = args.fp32.with_name(args.fp32.stem + ".prep.onnx")
    if args.out.resolve() == prep_path.resolve():
        raise ValueError("--out must not use the temporary .prep.onnx path")

    # Heavy dependencies are imported only after argument/path validation so
    # ``--help`` and path errors remain useful on an edge-only host.
    import onnx
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self, images, input_name, imgsz):
            self.images = list(images)
            self.input_name = input_name
            self.imgsz = imgsz
            self._iter = iter(self.images)

        def get_next(self):
            image = next(self._iter, None)
            return None if image is None else {self.input_name: letterbox(image, self.imgsz)}

        def rewind(self):
            self._iter = iter(self.images)

    images = select_calibration_images(args.train, args.n_calib)
    validation_images = _image_paths(args.validation_images) if args.validation_images else []
    if args.validation_images is not None and len(validation_images) == 0:
        raise ValueError("--validation-images does not contain any supported images")
    _validate_unique_stems(validation_images, "validation")
    calibration_hashes = {_sha256(path) for path in images}
    validation_hashes = {_sha256(path) for path in validation_images}
    overlap = calibration_hashes.intersection(validation_hashes)
    if overlap:
        raise ValueError(
            "calibration and validation sets overlap by SHA256; use training images only "
            "(overlap count={})".format(len(overlap))
        )
    using_default_exclude = args.exclude is None and not args.no_default_exclude
    exclude = list(DEFAULT_EXCLUDE) if using_default_exclude else list(args.exclude or [])
    print(
        f"[quant] calibration images: {len(images)} (method={args.method}, "
        f"per_channel=True, format={args.format}, exclude={exclude or 'none'})"
    )

    source = onnx.load(str(args.fp32))
    if not source.graph.input:
        raise ValueError("FP32 ONNX graph has no input")
    input_name = source.graph.input[0].name
    input_dims = source.graph.input[0].type.tensor_type.shape.dim
    if len(input_dims) != 4:
        raise ValueError("FP32 ONNX input must be rank-4 NCHW for the shared calibration preprocessing")
    channel_dim = input_dims[1]
    if channel_dim.dim_value and channel_dim.dim_value != 3:
        raise ValueError(
            f"FP32 ONNX input channel dimension must be 3, got {channel_dim.dim_value}"
        )
    if len(input_dims) == 4 and input_dims[2].dim_value and input_dims[3].dim_value:
        model_h, model_w = input_dims[2].dim_value, input_dims[3].dim_value
        if model_h != args.imgsz or model_w != args.imgsz:
            raise ValueError(
                f"--imgsz={args.imgsz} does not match static ONNX input [{model_h}, {model_w}]"
            )
    prep = prep_path
    try:
        quant_pre_process(str(args.fp32), str(prep), skip_symbolic_shape=True)
    except TypeError:
        quant_pre_process(str(args.fp32), str(prep))

    prepared = onnx.load(str(prep))
    if not prepared.graph.input:
        raise ValueError("preprocessed ONNX graph has no input")
    # Shape preprocessing can rename graph inputs in some ORT releases.
    input_name = prepared.graph.input[0].name
    opset = next((op.version for op in prepared.opset_import if op.domain in ("", "ai.onnx")), 0)
    if opset < 13:
        print(f"[quant] upgrading opset {opset} -> 17 for per-channel quantization")
        from onnx import version_converter

        prepared = version_converter.convert_version(prepared, 17)
        onnx.save(prepared, str(prep))

    nodes_to_exclude = [
        node.name for node in prepared.graph.node
        if node.name and any(token.lower() in node.name.lower() for token in exclude)
    ]
    if exclude and using_default_exclude:
        missing_patterns = [
            token for token in exclude
            if not any(token.lower() in (node.name or "").lower() for node in prepared.graph.node)
        ]
        if missing_patterns:
            raise RuntimeError(
                "default INT8 exclusion pattern(s) matched no ONNX nodes: "
                + ", ".join(missing_patterns)
                + "; refusing to claim the mixed-precision acceptance recipe"
            )
    if exclude:
        print(f"[quant] excluding {len(nodes_to_exclude)} nodes from INT8: {exclude}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out.with_suffix(args.out.suffix + ".calibration.txt")
    manifest_names = [manifest_name(image, args.train) for image in images]
    manifest_text = "\n".join(manifest_names) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    reader = Reader(images, input_name, args.imgsz)
    extra_options = (
        {
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            # TensorRT handles convolution bias internally and rejects an
            # explicitly quantized int32 bias DequantizeLinear node.
            "QuantizeBias": False,
        }
        if args.symmetric
        else {}
    )
    quantize_static(
        str(prep), str(args.out),
        calibration_data_reader=reader,
        quant_format=getattr(QuantFormat, args.format),
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=getattr(CalibrationMethod, args.method),
        nodes_to_exclude=nodes_to_exclude,
        extra_options=extra_options,
    )

    quantized = onnx.load(str(args.out))
    _restore_metadata(source, quantized)
    onnx.checker.check_model(quantized)
    onnx.save(quantized, str(args.out))

    fp32_mb = args.fp32.stat().st_size / 1e6
    int8_mb = args.out.stat().st_size / 1e6
    summary = {
        "fp32": str(args.fp32),
        "output": str(args.out),
        "calibration_images": len(images),
        "calibration_image_list_sha256": hashlib.sha256(
            "\n".join(manifest_names).encode("utf-8")
        ).hexdigest(),
        "validation_images": len(validation_images),
        "calibration_validation_disjoint": bool(validation_images) and not overlap,
        "validation_image_list_sha256": (
            hashlib.sha256(
                "\n".join(manifest_name(path, args.validation_images) for path in validation_images).encode("utf-8")
            ).hexdigest()
            if validation_images
            else None
        ),
        "calibration_manifest": str(manifest_path),
        "calibration_manifest_sha256": manifest_sha256,
        "format": args.format,
        "symmetric": args.symmetric,
        "method": args.method,
        "excluded_substrings": exclude,
        "excluded_nodes": len(nodes_to_exclude),
        "imgsz": args.imgsz,
        "opset": next((op.version for op in quantized.opset_import if op.domain in ("", "ai.onnx")), None),
        "size_mb": {"fp32": round(fp32_mb, 3), "int8": round(int8_mb, 3)},
        # Quantization is not an accuracy claim until eval_map.py has consumed
        # the generated predictions.  Keep this false even when all structural
        # checks pass, and point callers to the next gate explicitly.
        "acceptance_ready": False,
        "accuracy_gate_required": True,
        "accuracy_gate_command": (
            "python scripts/eval_map.py --preds <int8_txt> --images <val_images> "
            "--labels <yolo_labels> --reference-json <pytorch_json> "
            "--max-abs-delta-pp 1.0"
        ),
    }
    summary_path = args.out.with_suffix(args.out.suffix + ".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[quant] done -> {args.out} ({fp32_mb:.1f} MB -> {int8_mb:.1f} MB)")
    print(f"[quant] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
