"""Benchmark the deployable (stripped) student: PyTorch + ONNX latency, params, GFLOPs.

This runner implements the deployment-cost side of the Foundation research
release gate (plan section 17.4 / Gate C). It proves that a Foundation-trained
checkpoint deploys as a pure student: no teacher/projector/foundation state in
the checkpoint, no forbidden tokens in the ONNX graph, and params/GFLOPs/latency
parity against the paired baseline.

Example::

    python scripts/foundation_student_latency_benchmark.py \
        --checkpoints runs/detect/p0-loss-ablation-mps-smoke/baseline-s20260818-w0.01/weights/best.pt,\
runs/detect/p0-loss-ablation-mps-smoke/cosine-s20260818-w0.01/weights/best.pt \
        --imgsz 128 --device mps --warmup 5 --iters 20 \
        --output reports/foundation/v0.1/p0-student-latency-mps.json

The report sets ``accuracy_claim=false`` and never fabricates a metric: when a
backend is unavailable the corresponding field records an explicit ``null``
with a reason.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Make direct ``python scripts/...`` invocation resolve this checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = 1
BENCHMARK = "foundation_student_latency"

#: Tokens that must never appear in a deployable student artifact (plan §16.3).
FORBIDDEN_TOKENS = ("dinov3", "dino", "siglip", "automodel", "teacher", "processor", "projector", "foundation")


def _csv_paths(value: str) -> list[Path]:
    """Parse a non-empty comma-separated list of checkpoint paths."""
    values = [Path(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list of checkpoints")
    return values


def scan_forbidden_keys(keys: list[str]) -> list[str]:
    """Return state-dict keys containing forbidden teacher/training tokens."""
    return [key for key in keys if any(token in key.lower() for token in FORBIDDEN_TOKENS)]


def scan_onnx_graph(path: str | Path) -> dict[str, Any]:
    """Scan an ONNX graph for forbidden teacher/training tokens.

    Checks node names, op types, and initializer/input/output names. Returns
    ``available=False`` with a reason when onnx is missing or the file cannot
    be parsed instead of pretending the graph is clean.
    """
    result: dict[str, Any] = {"available": False, "reason": None, "path": str(path), "forbidden": []}
    try:
        import onnx
    except ImportError:
        result["reason"] = "onnx is not installed"
        return result
    try:
        model = onnx.load(str(path))
    except OSError as exc:
        result["reason"] = f"cannot load onnx file: {exc}"
        return result
    graph = model.graph
    names = [node.name for node in graph.node] + [node.op_type for node in graph.node]
    names += [init.name for init in graph.initializer]
    names += [value.name for value in list(graph.input) + list(graph.output)]
    forbidden = sorted({name for name in names if any(token in name.lower() for token in FORBIDDEN_TOKENS)})
    result.update(
        {
            "available": True,
            "forbidden": forbidden,
            "nodes": len(graph.node),
            "initializers": len(graph.initializer),
            "size_bytes": Path(path).stat().st_size,
        }
    )
    return result


def load_stripped_student(checkpoint: str | Path) -> dict[str, Any]:
    """Load a checkpoint, strip any training wrapper, and audit the student state dict."""
    import torch

    from ultralytics.nn.foundation_distill_model import strip_foundation_distillation_model

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    model = payload.get("ema") or payload.get("model")
    if model is None:
        raise ValueError(f"checkpoint has neither 'model' nor 'ema': {checkpoint}")
    metadata = payload.get("foundation") or {}
    stripped = strip_foundation_distillation_model(model)
    keys = [str(key) for key in stripped.state_dict().keys()]
    return {
        "model": stripped,
        "metadata": metadata,
        "state_keys": len(keys),
        "forbidden_keys": scan_forbidden_keys(keys),
        "checkpoint_size_bytes": Path(checkpoint).stat().st_size,
    }


def _sync_device(device: str) -> None:
    """Synchronize asynchronous backends before/after timed sections."""
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _summarize_times(times: list[float]) -> dict[str, float]:
    """Return latency statistics in milliseconds."""
    ordered = sorted(times)
    n = len(ordered)

    def percentile(q: float) -> float:
        idx = min(n - 1, max(0, int(round(q * (n - 1)))))
        return ordered[idx]

    mean = sum(ordered) / n
    variance = sum((value - mean) ** 2 for value in ordered) / n
    return {
        "mean_ms": round(mean * 1000, 4),
        "std_ms": round(variance**0.5 * 1000, 4),
        "p50_ms": round(percentile(0.5) * 1000, 4),
        "p95_ms": round(percentile(0.95) * 1000, 4),
        "min_ms": round(ordered[0] * 1000, 4),
        "max_ms": round(ordered[-1] * 1000, 4),
        "iters": n,
    }


def benchmark_pytorch(model: Any, *, imgsz: int, device: str, warmup: int, iters: int) -> dict[str, Any]:
    """Time single-image forward passes of the stripped student on the requested device."""
    import torch

    result: dict[str, Any] = {"available": False, "reason": None, "device": device, "imgsz": imgsz}
    if device == "mps" and not torch.backends.mps.is_available():
        result["reason"] = "mps is not available"
        return result
    if device.startswith("cuda") and not torch.cuda.is_available():
        result["reason"] = "cuda is not available"
        return result
    model = model.float().to(device).eval()
    try:
        model.fuse()
        result["fused"] = True
    except (AttributeError, RuntimeError, TypeError) as exc:
        result["fused"] = False
        result["fuse_note"] = f"{type(exc).__name__}: {exc}"
        model = model.to(device).eval()
    params = sum(int(p.numel()) for p in model.parameters())
    result["params"] = params
    try:
        from ultralytics.utils.torch_utils import get_flops

        result["gflops"] = round(float(get_flops(model, imgsz)), 4)
    except (RuntimeError, TypeError, ValueError) as exc:
        result["gflops"] = None
        result["gflops_note"] = f"{type(exc).__name__}: {exc}"
    dummy = torch.rand(1, 3, imgsz, imgsz, device=device)
    times: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(dummy)
        _sync_device(device)
        for _ in range(iters):
            _sync_device(device)
            started = time.perf_counter()
            model(dummy)
            _sync_device(device)
            times.append(time.perf_counter() - started)
    result.update(_summarize_times(times))
    result["available"] = True
    return result


def export_and_benchmark_onnx(
    checkpoint: str | Path,
    *,
    imgsz: int,
    warmup: int,
    iters: int,
    workdir: str | Path,
) -> dict[str, Any]:
    """Export the student to ONNX, scan the graph, and time ONNX Runtime CPU inference."""
    result: dict[str, Any] = {"available": False, "reason": None, "imgsz": imgsz}
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        result["reason"] = "onnxruntime is not installed"
        return result
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    exported = Path(str(model.export(format="onnx", imgsz=imgsz, half=False, dynamic=False, simplify=True)))
    if not exported.is_file():
        result["reason"] = f"export did not produce a file: {exported}"
        return result
    # Move the artifact out of the checkpoint directory into the benchmark workdir.
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / exported.name
    if exported.resolve() != target.resolve():
        target.write_bytes(exported.read_bytes())
        exported.unlink()
    result["onnx"] = str(target)
    result["graph_scan"] = scan_onnx_graph(target)

    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    times: list[float] = []
    for _ in range(warmup):
        session.run(None, {input_name: dummy})
    for _ in range(iters):
        started = time.perf_counter()
        session.run(None, {input_name: dummy})
        times.append(time.perf_counter() - started)
    result["ort_cpu"] = _summarize_times(times)
    result["available"] = True
    return result


def gate_c_compare(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Gate C: deployable student params/GFLOPs must match the paired baseline."""
    gate: dict[str, Any] = {"checked": False, "params_match": None, "gflops_match": None, "details": []}
    if len(records) < 2:
        gate["reason"] = "need at least two checkpoints (baseline + variant)"
        return gate
    baseline = records[0]
    base_params = baseline.get("pytorch", {}).get("params")
    base_gflops = baseline.get("pytorch", {}).get("gflops")
    if base_params is None or base_gflops is None:
        gate["reason"] = "baseline record lacks params/gflops"
        return gate
    gate["checked"] = True
    gate["params_match"] = True
    gate["gflops_match"] = True
    for record in records[1:]:
        params = record.get("pytorch", {}).get("params")
        gflops = record.get("pytorch", {}).get("gflops")
        params_ok = params == base_params
        flops_ok = gflops is not None and abs(gflops - base_gflops) <= max(1e-6, abs(base_gflops) * 1e-4)
        gate["params_match"] = gate["params_match"] and params_ok
        gate["gflops_match"] = gate["gflops_match"] and flops_ok
        gate["details"].append(
            {
                "checkpoint": record.get("checkpoint"),
                "params": params,
                "params_delta": None if params is None else int(params) - int(base_params),
                "gflops": gflops,
                "gflops_delta": None if gflops is None else round(float(gflops) - float(base_gflops), 6),
                "params_match": params_ok,
                "gflops_match": flops_ok,
            }
        )
    return gate


def benchmark_checkpoints(
    checkpoints: list[Path],
    *,
    imgsz: int,
    device: str,
    warmup: int,
    iters: int,
    export: bool,
    workdir: str | Path,
) -> dict[str, Any]:
    """Benchmark every checkpoint and return the full report payload."""
    records: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        record: dict[str, Any] = {"checkpoint": str(checkpoint), "available": False, "reason": None}
        if not checkpoint.is_file():
            record["reason"] = "checkpoint not found"
            records.append(record)
            continue
        loaded = load_stripped_student(checkpoint)
        record.update(
            {
                "available": True,
                "student_class": type(loaded["model"]).__name__,
                "state_keys": loaded["state_keys"],
                "forbidden_keys": loaded["forbidden_keys"],
                "checkpoint_size_bytes": loaded["checkpoint_size_bytes"],
                "foundation_metadata": loaded["metadata"],
                "pytorch": benchmark_pytorch(loaded["model"], imgsz=imgsz, device=device, warmup=warmup, iters=iters),
            }
        )
        if export:
            record["onnx_export"] = export_and_benchmark_onnx(
                checkpoint, imgsz=imgsz, warmup=warmup, iters=iters, workdir=workdir
            )
        records.append(record)
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK,
        "accuracy_claim": False,
        "config": {"imgsz": imgsz, "device": device, "warmup": warmup, "iters": iters, "export": export},
        "forbidden_tokens": list(FORBIDDEN_TOKENS),
        "records": records,
        "gate_c": gate_c_compare(records),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse student latency benchmark arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=_csv_paths, required=True, help="Comma-separated student checkpoints.")
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--device", default="cpu", help="PyTorch latency device (cpu/mps/cuda:N).")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--export", dest="export", action="store_true", default=True)
    parser.add_argument("--no-export", dest="export", action="store_false", help="Skip ONNX export/ORT benchmark.")
    parser.add_argument("--workdir", type=Path, default=Path("runs/detect/student-latency"), help="ONNX artifact dir.")
    parser.add_argument("--output", type=Path, default=Path("reports/foundation/v0.1/student-latency.json"))
    args = parser.parse_args(argv)
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")
    if args.warmup < 0 or args.iters <= 0:
        parser.error("--warmup must be >= 0 and --iters must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    """Entry point for the student latency benchmark."""
    args = parse_args(argv)
    report = benchmark_checkpoints(
        args.checkpoints,
        imgsz=args.imgsz,
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        export=args.export,
        workdir=args.workdir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    gate = report["gate_c"]
    print(
        json.dumps(
            {
                "checkpoints": len(report["records"]),
                "gate_c": {
                    "checked": gate["checked"],
                    "params_match": gate["params_match"],
                    "gflops_match": gate["gflops_match"],
                },
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
