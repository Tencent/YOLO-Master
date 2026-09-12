"""Unified accuracy, latency, size, and failure validation for all artifacts."""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .common import json_dump, letterbox_image, sha256_file


VARIANT_FILES = {
    "fp32": "fp32.onnx",
    "fp16": "fp16.onnx",
    "full_int8": "full_int8.onnx",
    "manual_fallback": "manual_fallback.onnx",
    "auto_fallback": "auto_fallback.onnx",
}


def _failure(path: Path, stage: str, exc: Exception) -> dict[str, Any]:
    from .onnx_backend import _operators_from_error

    return {
        "status": "failed",
        "path": str(path.resolve()),
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "failed_operators": _operators_from_error(str(exc)),
    }


def _extract_accuracy(metrics: Any) -> dict[str, Any]:
    def number(value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return parsed if math.isfinite(parsed) else float("nan")

    results = dict(getattr(metrics, "results_dict", {}) or {})
    box = getattr(metrics, "box", None)

    def box_metric(attribute: str, result_key: str) -> float:
        value = getattr(box, attribute, None) if box is not None else None
        return number(results.get(result_key) if value is None else value)

    accuracy = {
        "map50_95": box_metric("map", "metrics/mAP50-95(B)"),
        "map50": box_metric("map50", "metrics/mAP50(B)"),
        "map75": box_metric("map75", "metrics/mAP75(B)"),
        "precision": box_metric("mp", "metrics/precision(B)"),
        "recall": box_metric("mr", "metrics/recall(B)"),
    }
    speed = {str(key): number(value) for key, value in dict(getattr(metrics, "speed", {}) or {}).items()}
    return {"accuracy": accuracy, "validator_speed_ms": speed, "raw_results": results}


def validate_accuracy(path: str | Path, *, data_yaml: Path, settings: dict[str, Any]) -> dict[str, Any]:
    """Run the same Ultralytics validator contract for every ONNX variant."""
    from ultralytics import YOLO

    target = Path(path)
    try:
        started = time.perf_counter()
        metrics = YOLO(str(target), task="detect").val(
            data=str(data_yaml),
            imgsz=int(settings.get("imgsz", 640)),
            batch=int(settings.get("validation_batch", settings.get("batch", 1))),
            device=str(settings.get("validation_device", settings.get("device", "cpu"))),
            workers=int(settings.get("workers", 4)),
            conf=float(settings.get("conf", 0.001)),
            iou=float(settings.get("iou", 0.7)),
            plots=False,
            save=False,
            save_json=False,
            verbose=False,
        )
        return {
            "status": "success",
            "elapsed_seconds": time.perf_counter() - started,
            "data_yaml": str(data_yaml.resolve()),
            "data_yaml_sha256": sha256_file(data_yaml),
            **_extract_accuracy(metrics),
        }
    except Exception as exc:
        return _failure(target, "accuracy", exc)


def _provider_list(settings: dict[str, Any]) -> list[Any]:
    configured = settings.get("ort_providers")
    if configured:
        return list(configured)
    device = str(settings.get("benchmark_device", settings.get("device", "cpu"))).lower()
    return (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device.startswith(("cuda", "0"))
        else ["CPUExecutionProvider"]
    )


def _profile_provider_assignment(
    target: Path, providers: list[Any], input_name: str, tensor: np.ndarray
) -> dict[str, Any]:
    """Run one profiled inference and report the provider used by every executed op."""
    import onnxruntime as ort

    profile_path: Path | None = None
    try:
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(target.parent / f".{target.stem}.provider_profile")
        session = ort.InferenceSession(str(target), sess_options=options, providers=providers)
        session.run(None, {input_name: tensor})
        profile_path = Path(session.end_profiling())
        events = json.loads(profile_path.read_text(encoding="utf-8"))
        provider_counts: Counter[str] = Counter()
        operator_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for event in events:
            args = event.get("args") if isinstance(event, dict) else None
            if not isinstance(args, dict):
                continue
            provider = args.get("provider")
            operator = args.get("op_name")
            if provider and operator:
                provider_counts[str(provider)] += 1
                operator_counts[str(provider)][str(operator)] += 1
        return {
            "status": "success",
            "providers": dict(sorted(provider_counts.items())),
            "operators_by_provider": {
                provider: dict(sorted(counts.items())) for provider, counts in sorted(operator_counts.items())
            },
        }
    except Exception as exc:
        return _failure(target, "provider_profile", exc)
    finally:
        if profile_path is not None:
            profile_path.unlink(missing_ok=True)


def benchmark_latency(path: str | Path, *, images: list[Path], settings: dict[str, Any]) -> dict[str, Any]:
    """Benchmark only ORT ``session.run`` with a shared provider and input list."""
    import onnxruntime as ort

    target = Path(path)
    providers = _provider_list(settings)
    available = set(ort.get_available_providers())
    requested_names = [item[0] if isinstance(item, (list, tuple)) else item for item in providers]
    missing = [name for name in requested_names if name not in available]
    if missing and all(name in missing for name in requested_names):
        return {
            "status": "failed",
            "stage": "latency",
            "error_type": "ProviderUnavailable",
            "error": f"requested ORT providers unavailable: {missing}; available={sorted(available)}",
            "failed_operators": [],
        }
    providers = [item for item in providers if (item[0] if isinstance(item, (list, tuple)) else item) in available]
    try:
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = int(settings.get("ort_intra_op_threads", 0))
        session_options.inter_op_num_threads = int(settings.get("ort_inter_op_threads", 0))
        session = ort.InferenceSession(str(target), sess_options=session_options, providers=providers)
        input_name = session.get_inputs()[0].name
        imgsz = int(settings.get("imgsz", 640))
        tensors = [letterbox_image(path, imgsz) for path in images]
        provider_assignment = (
            _profile_provider_assignment(target, providers, input_name, tensors[0])
            if bool(settings.get("profile_provider_assignment", True))
            else {"status": "disabled"}
        )
        warmup = int(settings.get("warmup", 10))
        repeats = int(settings.get("repeats", 100))
        for index in range(warmup):
            session.run(None, {input_name: tensors[index % len(tensors)]})
        samples = []
        for index in range(repeats):
            started = time.perf_counter_ns()
            session.run(None, {input_name: tensors[index % len(tensors)]})
            samples.append((time.perf_counter_ns() - started) / 1e6)
        values = np.asarray(samples, dtype=np.float64)
        return {
            "status": "success",
            "providers_requested": providers,
            "providers_used": session.get_providers(),
            "provider_assignment": provider_assignment,
            "warmup": warmup,
            "repeats": repeats,
            "latency_ms": {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "p90": float(np.percentile(values, 90)),
                "p95": float(np.percentile(values, 95)),
                "min": float(values.min()),
                "max": float(values.max()),
                "std": float(values.std()),
            },
        }
    except Exception as exc:
        return _failure(target, "latency", exc)


def validate_artifact(
    path: str | Path,
    *,
    data_yaml: Path,
    benchmark_images: list[Path],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Collect accuracy, latency, physical size, hash, and failures."""
    target = Path(path)
    if not target.is_file():
        return {
            "status": "failed",
            "path": str(target.resolve()),
            "stage": "preflight",
            "error_type": "FileNotFoundError",
            "error": "artifact does not exist",
            "failed_operators": [],
        }
    accuracy = validate_accuracy(target, data_yaml=data_yaml, settings=settings)
    latency = benchmark_latency(target, images=benchmark_images, settings=settings)
    return {
        "status": "success" if accuracy["status"] == latency["status"] == "success" else "failed",
        "path": str(target.resolve()),
        "size_bytes": target.stat().st_size,
        "size_mb": target.stat().st_size / (1024 * 1024),
        "sha256": sha256_file(target),
        "accuracy_result": accuracy,
        "latency_result": latency,
        "failed_operators": sorted(
            set(accuracy.get("failed_operators", [])) | set(latency.get("failed_operators", []))
        ),
    }


def compare_variants(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Attach deltas against FP32 without mixing failed measurements."""
    baseline = results.get("fp32", {})
    baseline_accuracy = baseline.get("accuracy_result", {}).get("accuracy", {})
    baseline_latency = baseline.get("latency_result", {}).get("latency_ms", {})
    base_map = baseline_accuracy.get("map50_95")
    base_ms = baseline_latency.get("median")
    base_size = baseline.get("size_bytes")
    compared = {}
    for name, row in results.items():
        item = dict(row)
        accuracy = row.get("accuracy_result", {}).get("accuracy", {})
        latency = row.get("latency_result", {}).get("latency_ms", {})
        if all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in (base_map, accuracy.get("map50_95"))
        ):
            item["map50_95_loss"] = float(base_map - accuracy["map50_95"])
            item["map50_95_loss_percentage_points"] = float((base_map - accuracy["map50_95"]) * 100.0)
        if all(
            isinstance(value, (int, float)) and math.isfinite(float(value)) and value != 0
            for value in (base_ms, latency.get("median"))
        ):
            item["latency_change_pct"] = float((latency["median"] / base_ms - 1.0) * 100.0)
        if all(isinstance(value, (int, float)) and value != 0 for value in (base_size, row.get("size_bytes"))):
            item["size_reduction_pct"] = float((1.0 - row["size_bytes"] / base_size) * 100.0)
        compared[name] = item
    return compared


def validate_family_variants(
    family_dir: Path,
    *,
    data_yaml: Path,
    benchmark_images: list[Path],
    settings: dict[str, Any],
    include_auto: bool,
) -> dict[str, Any]:
    """Validate one family's comparable primary artifacts."""
    names = tuple(VARIANT_FILES) if include_auto else tuple(name for name in VARIANT_FILES if name != "auto_fallback")
    results = {
        name: validate_artifact(
            family_dir / "artifacts" / VARIANT_FILES[name],
            data_yaml=data_yaml,
            benchmark_images=benchmark_images,
            settings=settings,
        )
        for name in names
    }
    compared = compare_variants(results)
    json_dump(family_dir / ("comparison.final.json" if include_auto else "comparison.pre_auto.json"), compared)
    return compared


def load_validation(path: Path) -> dict[str, Any]:
    """Read a saved validation JSON."""
    return json.loads(path.read_text(encoding="utf-8"))
