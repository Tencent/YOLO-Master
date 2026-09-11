"""Read-only CPU smoke/parity checks for already exported A3 artifacts.

No training, export, quantization, mAP evaluation, or latency benchmarking occurs.
Run with the SAME Python environment and config used for the CPU experiment lock.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch


VARIANTS = ("fp32", "fp16", "full_int8", "manual_fallback")


def save_json(path, payload):
    """Disallow non-standard NaN/Infinity tokens in the diagnostic report."""
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def error_record(exc):
    return {
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def array_metrics(reference, actual, atol, rtol):
    """Compare aligned elements, never broadcasting or discarding non-finite values."""
    import numpy as np

    a, b = np.asarray(reference), np.asarray(actual)
    result = {"reference_shape": list(a.shape), "actual_shape": list(b.shape), "atol": atol, "rtol": rtol}
    result["reference_nonfinite"] = int((~np.isfinite(a)).sum())
    result["actual_nonfinite"] = int((~np.isfinite(b)).sum())
    result["allclose"] = False
    if a.shape != b.shape or not a.size or result["reference_nonfinite"] or result["actual_nonfinite"]:
        result["reason"] = "shape_mismatch_empty_or_nonfinite"
        return result
    a, b = a.astype(np.float64), b.astype(np.float64)
    delta = b - a
    close = np.abs(delta) <= atol + rtol * np.abs(a)
    denominator = float(np.linalg.norm(a.ravel()))
    numerator = float(np.linalg.norm(delta.ravel()))
    result.update(
        max_abs=float(np.abs(delta).max()),
        mean_abs=float(np.abs(delta).mean()),
        rmse=float(np.sqrt(np.mean(delta * delta))),
        relative_l2=numerator / denominator if denominator else (0.0 if numerator == 0 else None),
        close_fraction=float(close.mean()),
        allclose=bool(close.all()),
    )
    return result


def compare_outputs(reference, actual, atol, rtol, end2end=False):
    """Require identical output count and shape; class IDs must match exactly."""
    import numpy as np

    if len(reference) != len(actual):
        return {"allclose": False, "reason": "output_count_mismatch", "counts": [len(reference), len(actual)]}
    rows = []
    for a, b in zip(reference, actual):
        row = array_metrics(a, b, atol, rtol)
        if end2end and a.shape == b.shape and a.ndim == 3 and a.shape[-1] == 6:
            row["boxes"] = array_metrics(a[..., :4], b[..., :4], atol, rtol)
            row["scores"] = array_metrics(a[..., 4], b[..., 4], atol, rtol)
            row["classes_equal"] = bool(np.array_equal(a[..., 5], b[..., 5]))
            row["allclose"] = row["allclose"] and row["classes_equal"]
            row["alignment_note"] = (
                "Row-aligned post-TopK detections; ties/reordering may require review, not necessarily an accuracy loss."
            )
        rows.append(row)
    return {"allclose": bool(rows) and all(row["allclose"] for row in rows), "outputs": rows}


def tensor_outputs(value):
    """Accept a detection tensor, (prediction, auxiliary dict), or tensor sequence."""
    import torch

    if isinstance(value, torch.Tensor):
        values = [value]
    elif (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and isinstance(value[0], torch.Tensor)
        and isinstance(value[1], dict)
    ):
        values = [value[0]]
    elif isinstance(value, (tuple, list)) and value and all(isinstance(item, torch.Tensor) for item in value):
        values = list(value)
    else:
        raise TypeError(
            f"Unsupported prediction structure: {type(value).__name__}; refusing to guess dictionary output order"
        )
    return [item.detach().cpu().numpy().copy() for item in values]


def prepare_reference(yolo, settings, destination):
    """Use the real exporter preparation but stop BEFORE writing/exporting ONNX."""
    from ultralytics.engine.exporter import Exporter
    from scripts.a3_precision.onnx_backend import _enable_masked_export

    class ReferenceReady(Exception):
        pass

    class CaptureExporter(Exporter):
        def run_callbacks(self, event):
            """Do not invoke external tracking integrations during a local check."""

        def export_onnx(self, *args, **kwargs):
            raise ReferenceReady

    _enable_masked_export(yolo.model)
    yolo.model.pt_path = str(destination / "reference_not_written.pt")
    custom = {"imgsz": yolo.model.args["imgsz"], "batch": 1, "data": None, "device": None, "verbose": False}
    options = {
        **yolo.overrides,
        **custom,
        "mode": "export",
        "format": "onnx",
        "device": "cpu",
        "imgsz": int(settings.get("imgsz", 640)),
        "batch": 1,
        "opset": int(settings.get("opset", 17)),
        "simplify": bool(settings.get("simplify", False)),
        "dynamic": bool(settings.get("dynamic", False)),
        "quantize": 32,
        "molora_export_mode": "routing_preserved",
    }
    exporter = CaptureExporter(overrides=options)
    try:
        exporter(model=yolo.model)
    except ReferenceReady:
        return exporter.model, exporter.metadata
    raise RuntimeError("Exporter did not reach the reference-capture point")


def predict_reference(model, inputs, onnx_branch=False):
    import torch

    outputs = []
    with torch.inference_mode():
        for index, array in enumerate(inputs):
            print(f"    PyTorch sample {index + 1}/{len(inputs)}, onnx_branch={onnx_branch}", flush=True)
            with patch("torch.onnx.is_in_onnx_export", return_value=onnx_branch):
                outputs.append(tensor_outputs(model(torch.from_numpy(array.copy()))))
    return outputs


def run_ort(path, inputs, threads):
    import numpy as np
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(f"Unexpected providers: {session.get_providers()}")
    descriptors = session.get_inputs()
    if len(descriptors) != 1:
        raise ValueError(f"Expected one image input, found {len(descriptors)}")
    descriptor = descriptors[0]
    dtypes = {"tensor(float)": np.float32, "tensor(float16)": np.float16}
    if descriptor.type not in dtypes:
        raise TypeError(f"Unsupported input type: {descriptor.type}")
    outputs = []
    for index, array in enumerate(inputs):
        if len(descriptor.shape) != array.ndim or any(
            isinstance(dim, int) and dim != size for dim, size in zip(descriptor.shape, array.shape)
        ):
            raise ValueError(f"Input shape mismatch: model={descriptor.shape}, image={array.shape}")
        print(f"    ORT {path.stem} sample {index + 1}/{len(inputs)}", flush=True)
        outputs.append(session.run(None, {descriptor.name: array.astype(dtypes[descriptor.type], copy=False)}))
    metadata = {
        "providers": session.get_providers(),
        "input_name": descriptor.name,
        "input_type": descriptor.type,
        "input_shape": descriptor.shape,
        "output_names": [item.name for item in session.get_outputs()],
        "graph_optimization": "ORT_ENABLE_ALL",
        "fp16_note": "CPU execution can internally promote FP16 ops; this is not a native-FP16 speed measurement.",
    }
    del session
    return outputs, metadata


def sample_comparisons(reference, actual, atol, rtol, end2end):
    if len(reference) != len(actual):
        raise ValueError("Sample count mismatch")
    return [compare_outputs(a, b, atol, rtol, end2end) for a, b in zip(reference, actual)]


def check_family(spec, config, inputs, destination, args):
    import numpy as np
    from scripts.a3_precision.common import sha256_file
    from scripts.a3_precision.onnx_backend import _load_yolo

    destination.mkdir()
    artifacts = config.output_dir / spec.family / "artifacts"
    manifest = json.loads((artifacts / "primary_variants.json").read_text(encoding="utf-8"))
    export = json.loads((artifacts / "fp32.export.json").read_text(encoding="utf-8"))
    for key, path in (("source_checkpoint_sha256", spec.checkpoint), ("source_config_sha256", spec.config)):
        if export.get(key) != sha256_file(path):
            raise RuntimeError(f"Export provenance mismatch: {key}")
    hashes = {}
    for variant in VARIANTS:
        path = artifacts / f"{variant}.onnx"
        digest = sha256_file(path)
        if manifest.get(variant, {}).get("status") != "success" or manifest[variant]["artifact"]["sha256"] != digest:
            raise RuntimeError(f"Artifact manifest mismatch: {path}")
        hashes[variant] = digest
    if export["artifact"]["sha256"] != hashes["fp32"]:
        raise RuntimeError("FP32 export metadata hash mismatch")
    result = {"family": spec.family, "artifact_sha256": hashes, "variants": {}, "references": {}}
    yolo = _load_yolo(spec)
    yolo.model.cpu().float().eval()
    original = predict_reference(yolo.model, inputs)
    model, metadata = prepare_reference(yolo, config.raw.get("experiment", {}), destination)
    end2end = bool(metadata.get("end2end", False))
    result["export_metadata"] = metadata
    eager = predict_reference(model, inputs)
    branch = predict_reference(model, inputs, onnx_branch=True)
    result["references"]["original_vs_export_prepared"] = sample_comparisons(
        original, eager, args.fp32_atol, args.fp32_rtol, end2end
    )
    result["references"]["export_prepared_vs_onnx_branch"] = sample_comparisons(
        eager, branch, args.fp32_atol, args.fp32_rtol, end2end
    )
    del model, yolo
    gc.collect()
    baseline = None
    for variant in VARIANTS:
        try:
            outputs, runtime = run_ort(artifacts / f"{variant}.onnx", inputs, args.threads)
            finite = all(np.isfinite(array).all() for sample in outputs for array in sample)
            row = {"status": "runnable" if finite else "nonfinite", "all_finite": bool(finite), "runtime": runtime}
            if variant == "fp32":
                baseline = outputs
                for name, reference in (
                    ("original_pytorch", original),
                    ("export_prepared_pytorch", eager),
                    ("onnx_branch_pytorch", branch),
                ):
                    row[name] = sample_comparisons(reference, outputs, args.fp32_atol, args.fp32_rtol, end2end)
                row["tolerance_pass"] = all(
                    item["allclose"]
                    for name in ("original_pytorch", "export_prepared_pytorch", "onnx_branch_pytorch")
                    for item in row[name]
                )
            elif baseline is not None:
                atol, rtol = (args.fp16_atol, args.fp16_rtol) if variant == "fp16" else (args.fp32_atol, args.fp32_rtol)
                row["vs_fp32_onnx"] = sample_comparisons(baseline, outputs, atol, rtol, end2end)
                if variant == "fp16":
                    row["tolerance_pass"] = all(item["allclose"] for item in row["vs_fp32_onnx"])
                else:
                    row["interpretation"] = (
                        "INT8 differences are descriptive only; floating-point allclose is NOT an INT8 accuracy acceptance criterion."
                    )
            else:
                row["comparison_skipped"] = "FP32 ONNX could not run"
            np.savez_compressed(
                destination / f"{variant}_outputs.npz",
                **{
                    f"sample_{i}_output_{j}": value
                    for i, sample in enumerate(outputs)
                    for j, value in enumerate(sample)
                },
            )
            result["variants"][variant] = row
        except Exception as exc:
            result["variants"][variant] = error_record(exc)
        save_json(destination / "report.json", result)
        print(f"  {variant}: {result['variants'][variant]['status']}", flush=True)
        gc.collect()
    for name, samples in (
        ("original_pytorch", original),
        ("export_prepared_pytorch", eager),
        ("onnx_branch_pytorch", branch),
    ):
        np.savez_compressed(
            destination / f"{name}_outputs.npz",
            **{f"sample_{i}_output_{j}": value for i, sample in enumerate(samples) for j, value in enumerate(sample)},
        )
    result["smoke_gate_pass"] = all(row.get("all_finite", False) for row in result["variants"].values()) and all(
        result["variants"][key].get("tolerance_pass", False) for key in ("fp32", "fp16")
    )
    result["status"] = "smoke_pass_int8_review_required" if result["smoke_gate_pass"] else "needs_review"
    save_json(destination / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New diagnostic directory; must not exist")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--family", action="append")
    parser.add_argument("--fp32-atol", type=float, default=1e-4)
    parser.add_argument("--fp32-rtol", type=float, default=1e-4)
    parser.add_argument("--fp16-atol", type=float, default=1e-3)
    parser.add_argument("--fp16-rtol", type=float, default=1e-2)
    parser.add_argument(
        "--allow-runtime-drift",
        action="store_true",
        help="Accept only runtime-contract drift while still verifying every locked asset and dataset hash",
    )
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    if not 1 <= args.samples <= 32 or args.threads < 1:
        parser.error("samples must be 1..32; threads must be positive")
    if any(not 0 <= value < float("inf") for value in (args.fp32_atol, args.fp32_rtol, args.fp16_atol, args.fp16_rtol)):
        parser.error("tolerances must be finite and non-negative")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(args.repo.resolve()))
    os.chdir(args.repo)
    summary = {
        "status": "running",
        "families": {},
        "claim_boundary": "Fixed-image CPU numerical smoke test only; no mAP, GPU latency, or INT8 accuracy acceptance.",
    }
    try:
        import numpy as np
        import onnxruntime as ort
        import torch
        from scripts.a3_precision.common import letterbox_image, sha256_file
        from scripts.a3_precision import manifest as manifest_module

        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        np.random.seed(0)
        if torch.cuda.is_available():
            raise RuntimeError("GPU unexpectedly visible; CPU-only process required")
        config = manifest_module.load_config(args.config)
        if int(config.raw.get("experiment", {}).get("batch", 1)) != 1:
            raise ValueError("This smoke test requires exported batch=1")
        print("Verifying existing experiment lock (read-only)...", flush=True)
        lock_path = config.output_dir / "experiment.lock.json"
        recorded_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        locked_runtime = recorded_lock["environment"]["contract"]
        current_runtime = manifest_module._runtime_contract()
        runtime_drift = current_runtime != locked_runtime
        if runtime_drift and not args.allow_runtime_drift:
            raise RuntimeError(
                "Runtime differs from the export lock. Re-run with --allow-runtime-drift only when validating in a "
                "deliberately different environment; the original lock will not be modified."
            )
        if runtime_drift:
            # Verify every other immutable contract field by substituting only the already-recorded runtime during
            # verification. The report below retains both contracts so this cannot masquerade as a locked-env run.
            with patch.object(manifest_module, "_runtime_contract", return_value=locked_runtime):
                lock = manifest_module.verify_lock(config)
            summary["claim_boundary"] += (
                " Validation ran in a deliberately different recorded runtime; it is not a reproduction of the "
                "locked export environment."
            )
        else:
            lock = manifest_module.verify_lock(config)
        available = lock["dataset"]["validation"]["images"]
        if args.samples > len(available):
            raise ValueError("Not enough locked validation images")
        indices = np.linspace(0, len(available) - 1, args.samples, dtype=int)
        paths = [Path(available[index]) for index in indices]
        size = int(config.raw.get("experiment", {}).get("imgsz", 640))
        inputs = [letterbox_image(path, size) for path in paths]
        summary.update(
            torch_version=torch.__version__,
            onnxruntime_version=ort.__version__,
            python=sys.version,
            threads=args.threads,
            config=str(config.source),
            lock_sha256=sha256_file(lock_path),
            runtime_contract_match=not runtime_drift,
            runtime_drift_allowed=bool(runtime_drift and args.allow_runtime_drift),
            locked_export_runtime=locked_runtime,
            current_validation_runtime=current_runtime,
            script_sha256=sha256_file(Path(__file__)),
            source_sha256={
                str(path.relative_to(args.repo)): sha256_file(path)
                for base in (args.repo / "ultralytics", args.repo / "scripts" / "a3_precision")
                for path in sorted(base.rglob("*.py"))
            },
            preprocessing="common.letterbox_image: fixed square 114 padding; BGR to RGB; float32 /255; BCHW; batch=1",
            samples=[
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "input_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                }
                for path, array in zip(paths, inputs)
            ],
        )
        specs = [spec for spec in config.models if spec.enabled and (not args.family or spec.family in args.family)]
        if not specs or (args.family and set(args.family) - {spec.family for spec in specs}):
            raise ValueError("Requested family missing or disabled")
        save_json(args.output / "summary.json", summary)
        for spec in specs:
            print(f"\n===== {spec.family} =====", flush=True)
            try:
                summary["families"][spec.family] = check_family(spec, config, inputs, args.output / spec.family, args)
            except Exception as exc:
                summary["families"][spec.family] = error_record(exc)
                print(traceback.format_exc(), flush=True)
            save_json(args.output / "summary.json", summary)
            gc.collect()
        passed = all(row.get("smoke_gate_pass", False) for row in summary["families"].values())
        summary["status"] = "smoke_pass_int8_review_required" if passed else "needs_review"
    except Exception as exc:
        summary.update(error_record(exc))
        print(traceback.format_exc(), flush=True)
    save_json(args.output / "summary.json", summary)
    print(f"\nStatus: {summary['status']}\nReport: {args.output / 'summary.json'}", flush=True)
    return 0 if summary["status"] == "smoke_pass_int8_review_required" else 2


if __name__ == "__main__":
    raise SystemExit(main())
