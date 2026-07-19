"""Executable export roundtrip validation for release and nightly gates."""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ExportRoundtripResult:
    format: str
    artifact_bytes: int
    outputs: int
    max_abs_error: float
    atol: float
    rtol: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tensor_outputs(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in _tensor_outputs(item)]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in _tensor_outputs(item)]
    return []


def validate_export_roundtrip(
    module: nn.Module,
    inputs: torch.Tensor | tuple[torch.Tensor, ...],
    fmt: str,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    opset: int = 17,
) -> dict[str, Any]:
    """Export, reload, execute, and numerically compare a module artifact."""
    module = module.cpu().eval()
    input_tuple = inputs if isinstance(inputs, tuple) else (inputs,)
    input_tuple = tuple(value.detach().cpu() for value in input_tuple)
    # Stateful heads cache anchors during the eager baseline. no_grad keeps
    # those tensors usable by the subsequent ONNX trace, unlike inference_mode.
    with torch.no_grad():
        eager = _tensor_outputs(module(*input_tuple))
    if not eager:
        raise RuntimeError("Export validation requires at least one tensor output.")

    fmt = fmt.lower()
    if fmt == "torchscript":
        with torch.inference_mode():
            artifact = torch.jit.trace(module, input_tuple, strict=False)
            actual = _tensor_outputs(artifact(*input_tuple))
        buffer = io.BytesIO()
        torch.jit.save(artifact, buffer)
        artifact_bytes = len(buffer.getvalue())
    elif fmt == "onnx":
        try:
            import onnx
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX roundtrip validation requires onnx and onnxruntime.") from exc
        buffer = io.BytesIO()
        input_names = [f"input_{index}" for index in range(len(input_tuple))]
        output_names = [f"output_{index}" for index in range(len(eager))]
        torch.onnx.export(
            module,
            input_tuple,
            buffer,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            do_constant_folding=False,
            dynamo=False,
        )
        raw = buffer.getvalue()
        onnx.checker.check_model(onnx.load_from_string(raw))
        session = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
        actual = [torch.from_numpy(value) for value in session.run(None, {name: value.numpy() for name, value in zip(input_names, input_tuple)})]
        artifact_bytes = len(raw)
    else:
        raise ValueError(f"Unsupported roundtrip format: {fmt}")

    if len(actual) != len(eager):
        raise RuntimeError(f"Export output count mismatch: eager={len(eager)}, exported={len(actual)}")
    errors = [float((expected.float() - observed.float()).abs().max().item()) for expected, observed in zip(eager, actual)]
    passed = all(torch.allclose(expected, observed.to(expected.dtype), atol=atol, rtol=rtol) for expected, observed in zip(eager, actual))
    result = ExportRoundtripResult(fmt, artifact_bytes, len(eager), max(errors, default=0.0), atol, rtol, passed)
    if not passed:
        raise RuntimeError(f"{fmt} roundtrip numerical mismatch: max_abs_error={result.max_abs_error:.6g}")
    return result.to_dict()


__all__ = ["ExportRoundtripResult", "validate_export_roundtrip"]
