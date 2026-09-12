"""Stable detector-output extraction and comparison helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def _array(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"detector output {name} must be non-empty and finite")
    return array.astype(np.float64, copy=False)


def detector_output_tensors(output: Any) -> dict[str, np.ndarray]:
    """Extract decoded output and fixed-grid head tensors from the YOLO eval contract."""

    if not isinstance(output, (list, tuple)) or len(output) < 2 or not isinstance(output[1], dict):
        raise TypeError("detector output must be (decoded, head-dict)")
    head = output[1]
    tensors = {"decoded_top300": _array(output[0], "decoded_top300")}
    for branch in ("one2one", "one2many"):
        branch_value = head.get(branch)
        if not isinstance(branch_value, dict):
            raise TypeError(f"detector output is missing {branch} dictionary")
        for field in ("scores", "boxes"):
            if field not in branch_value:
                raise TypeError(f"detector output is missing {branch}.{field}")
            tensors[f"{branch}_{field}"] = _array(branch_value[field], f"{branch}_{field}")
    return tensors


def detector_output_comparison(reference: Any, candidate: Any) -> dict[str, Any]:
    """Compare fixed-structure detector tensors without treating decoded row order as anchor-aligned."""

    left = detector_output_tensors(reference)
    right = detector_output_tensors(candidate)
    if left.keys() != right.keys():
        raise RuntimeError("detector output tensor keys changed")
    results = {}
    for name in left:
        if left[name].shape != right[name].shape:
            raise RuntimeError(f"detector output shape changed for {name}: {left[name].shape} != {right[name].shape}")
        difference = np.abs(left[name] - right[name])
        results[name] = {
            "shape": list(left[name].shape),
            "mean_absolute_change": float(difference.mean()),
            "root_mean_square_change": float(np.sqrt(np.mean(difference**2))),
            "maximum_absolute_change": float(difference.max()),
            "alignment": (
                "fixed head-grid and channel order"
                if name != "decoded_top300"
                else "decoded Top-300 row order is not treated as anchor-aligned"
            ),
        }
    return {
        "primary_tensor": "one2one_scores",
        "primary_reason": "fixed grid/channel order; closest retained detector class-score endpoint",
        "tensors": results,
    }
