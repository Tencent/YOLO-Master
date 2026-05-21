from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any

from runtime.cli.contract import ensure_manifest_dir, json_safe, plan_response, response, write_manifest
from runtime.cli.dataset import collect_dataset_images, dedupe_images, expand_image_reference, select_image_sample
from runtime.cli.normalize import parse_bool


def _resolve_images(request: dict[str, Any], params: dict[str, Any]) -> tuple[list[Path], dict[str, Any]]:
    source = request.get("inputs", {}).get("source") or params.get("source")
    data = request.get("inputs", {}).get("data") or params.get("data")
    split = str(params.get("split", "val"))
    if source not in (None, ""):
        images = expand_image_reference(source)
        dataset_info = {"source": json_safe(source), "split": None, "root": None}
    else:
        images, dataset_info, _ = collect_dataset_images(data, split)
    images = dedupe_images(images)
    limit_raw = params.get("limit", params.get("max_images", 5))
    limit = int(limit_raw) if limit_raw not in (None, "") else 5
    seed_raw = params.get("seed")
    sample = select_image_sample(
        images,
        limit=None if limit <= 0 else limit,
        offset=int(params.get("offset", 0) or 0),
        stride=max(1, int(params.get("stride", 1) or 1)),
        shuffle=parse_bool(params.get("shuffle"), False),
        seed=int(seed_raw) if seed_raw not in (None, "") else None,
    )
    if not sample:
        raise ValueError("No local images were found for Sparse SAHI comparison.")
    dataset_info["images_total"] = len(images)
    dataset_info["sample_count"] = len(sample)
    dataset_info["sample_limit"] = None if limit <= 0 else limit
    return sample, dataset_info


def _load_sparse_sahi_predictor():
    try:
        return importlib.import_module("sparse_sahi_inference").SparseSAHIPredictor, None
    except Exception as exc:
        return None, exc


def _box_count(result: Any) -> int:
    boxes = getattr(result, "boxes", None)
    try:
        return len(boxes) if boxes is not None else 0
    except Exception:
        return 0


def _speed(result: Any) -> dict[str, Any]:
    speed = getattr(result, "speed", None)
    return json_safe(speed) if isinstance(speed, dict) else {}


def run_sahi_compare(request: dict[str, Any]) -> dict[str, Any]:
    params = dict(request.get("params", {}))
    model_ref = request.get("inputs", {}).get("model") or params.get("model")
    if not model_ref:
        raise ValueError("`inputs.model` is required for yolo.eval.sparse_sahi_compare.")
    images, dataset_info = _resolve_images(request, params)
    output_dir = Path(params.get("output_dir") or ensure_manifest_dir(request) / "sparse_sahi_compare")
    conf = float(params.get("conf", params.get("conf_thres", 0.25)))
    imgsz = int(params.get("imgsz", 224))
    include_full_sahi = parse_bool(params.get("include_full_sahi"), True)

    if request.get("policy", {}).get("dry_run"):
        return plan_response(
            request,
            "Sparse SAHI comparison dry run prepared",
            "python_api",
            "yolo.eval.sparse_sahi_compare",
            params={
                "model": model_ref,
                "images": [str(path) for path in images[:10]],
                "dataset": dataset_info,
                "output_dir": str(output_dir),
                "stages": ["standard_predict", "full_sahi_optional", "sparse_sahi_predict"],
                "conf": conf,
                "imgsz": imgsz,
            },
        )

    SparseSAHIPredictor, import_error = _load_sparse_sahi_predictor()
    if include_full_sahi and SparseSAHIPredictor is None:
        payload = response(
            request["skill"],
            "blocked",
            "SparseSAHIPredictor is unavailable; Sparse SAHI comparison was not run.",
            error={
                "category": "missing_optional_dependency",
                "missing_module": "sparse_sahi_inference",
                "message": str(import_error),
            },
            data={
                "dataset": dataset_info,
                "images": [str(path) for path in images],
                "hint": "Install or expose sparse_sahi_inference.py, or set params.include_full_sahi=false to compare standard vs integrated sparse_sahi only.",
            },
            next_actions=["yolo.predict"],
        )
        payload["manifest"] = str(write_manifest(request, payload))
        return payload

    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_ref)
    legacy_sahi = SparseSAHIPredictor(model_ref) if include_full_sahi and SparseSAHIPredictor else None
    items: list[dict[str, Any]] = []
    for image_path in images:
        item: dict[str, Any] = {"path": str(image_path)}
        start = time.perf_counter()
        standard = model.predict(str(image_path), imgsz=imgsz, conf=conf, verbose=False, sparse_sahi=False)[0]
        item["standard"] = {
            "latency_sec": round(time.perf_counter() - start, 6),
            "boxes": _box_count(standard),
            "speed_ms": _speed(standard),
        }
        if legacy_sahi is not None:
            start = time.perf_counter()
            boxes, scores, classes, meta = legacy_sahi.predict_standard(str(image_path), conf_thres=conf)
            item["full_sahi"] = {
                "latency_sec": round(float(meta.get("inference_time", time.perf_counter() - start)), 6) if isinstance(meta, dict) else round(time.perf_counter() - start, 6),
                "boxes": len(boxes),
                "slices": len(meta.get("slices", [])) if isinstance(meta, dict) else None,
                "classes": json_safe([int(value) for value in classes]) if classes is not None else [],
                "scores": json_safe([float(value) for value in scores]) if scores is not None else [],
            }
        start = time.perf_counter()
        sparse = model.predict(str(image_path), imgsz=imgsz, conf=conf, verbose=False, sparse_sahi=True)[0]
        sparse_meta = getattr(sparse, "sparse_sahi_metadata", {}) or {}
        item["sparse_sahi"] = {
            "latency_sec": round(time.perf_counter() - start, 6),
            "boxes": _box_count(sparse),
            "speed_ms": _speed(sparse),
            "slices": len(sparse_meta.get("slices", [])) if isinstance(sparse_meta, dict) else None,
            "metadata": json_safe(sparse_meta),
        }
        base_latency = item["standard"]["latency_sec"]
        sparse_latency = item["sparse_sahi"]["latency_sec"]
        item["delta"] = {
            "sparse_vs_standard_latency_sec": round(sparse_latency - base_latency, 6),
            "sparse_vs_standard_speedup": round(base_latency / sparse_latency, 6) if sparse_latency > 0 else None,
            "sparse_box_delta": int(item["sparse_sahi"]["boxes"]) - int(item["standard"]["boxes"]),
        }
        items.append(item)

    payload = response(
        request["skill"],
        "ok",
        f"Sparse SAHI comparison finished on {len(items)} images",
        data={"dataset": dataset_info, "items": items},
        evaluation={
            "images": len(items),
            "avg_standard_latency_sec": round(sum(item["standard"]["latency_sec"] for item in items) / len(items), 6),
            "avg_sparse_sahi_latency_sec": round(sum(item["sparse_sahi"]["latency_sec"] for item in items) / len(items), 6),
        },
        artifacts=[{"kind": "directory", "path": str(output_dir.resolve())}],
        next_actions=["yolo.predict", "yolo.val"],
    )
    payload["manifest"] = str(write_manifest(request, payload))
    return payload
