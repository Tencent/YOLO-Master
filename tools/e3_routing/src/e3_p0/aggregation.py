"""Dataset-level aggregation and reproducibility checks for routing events."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

import numpy as np

from .adapters import routing_metrics


def _event_weight(event: dict[str, Any]) -> float:
    runtime = event.get("runtime", {})
    batch_size = runtime.get("batch_size")
    if batch_size is None:
        shape = runtime.get("input_shape") or []
        batch_size = shape[0] if shape else 1
    return float(batch_size)


def aggregate_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-forward loads by family/module with batch-size weighting."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[(event["family"], event["module"]["name"])].append(event)

    aggregates = []
    for (family, module_name), group in sorted(grouped.items()):
        loads = np.asarray([event["routing"]["expert_load"] for event in group], dtype=np.float64)
        weights = np.asarray([_event_weight(event) for event in group], dtype=np.float64)
        mean_load = np.average(loads, axis=0, weights=weights)
        variance = np.average((loads - mean_load) ** 2, axis=0, weights=weights)
        metrics = routing_metrics(mean_load)
        metric_stats = {}
        for name in ("entropy_normalized", "load_gini", "dominant_expert_share"):
            values = np.asarray([event["routing"][name] for event in group], dtype=np.float64)
            metric_stats[name] = {
                "mean": float(np.average(values, weights=weights)),
                "std": float(np.sqrt(np.average((values - np.average(values, weights=weights)) ** 2, weights=weights))),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        aggregates.append(
            {
                "family": family,
                "module": module_name,
                "num_experts": len(mean_load),
                "forward_observations": len(group),
                "sample_observations": int(weights.sum()),
                "expert_load_mean": mean_load.tolist(),
                "expert_load_std": np.sqrt(variance).tolist(),
                "aggregate_metrics": metrics,
                "per_forward_metric_stats": metric_stats,
                "mixing_weight_states": sorted({event["routing"]["mixing_weights"]["state"] for event in group}),
                "aux_loss_states": sorted({event["aux_loss"]["state"] for event in group}),
            }
        )
    return aggregates


def _stable_payload(event: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(event.get("runtime", {}))
    runtime.pop("pass_name", None)
    return {
        "family": event["family"],
        "module": event["module"],
        "routing": event["routing"],
        "aux_loss": event["aux_loss"],
        "runtime": runtime,
        "provenance": event["provenance"],
        "source_snapshot": event["source_snapshot"],
    }


def event_fingerprint(event: dict[str, Any]) -> str:
    """Hash deterministic event content while excluding run/time/sequence metadata."""

    encoded = json.dumps(_stable_payload(event), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compare_repeated_runs(reference: list[dict[str, Any]], repeated: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare two same-order passes using canonical event fingerprints."""

    if len(reference) != len(repeated):
        return {
            "status": "FAIL",
            "reference_events": len(reference),
            "repeated_events": len(repeated),
            "matching_events": 0,
            "mismatches": [{"reason": "event_count_mismatch"}],
        }
    mismatches = []
    for index, (left, right) in enumerate(zip(reference, repeated)):
        left_hash = event_fingerprint(left)
        right_hash = event_fingerprint(right)
        if left_hash != right_hash:
            mismatches.append(
                {
                    "index": index,
                    "family": left["family"],
                    "module": left["module"]["name"],
                    "sample_indices": left.get("runtime", {}).get("sample_indices"),
                    "reference_sha256": left_hash,
                    "repeated_sha256": right_hash,
                }
            )
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "reference_events": len(reference),
        "repeated_events": len(repeated),
        "matching_events": len(reference) - len(mismatches),
        "mismatches": mismatches,
    }


def compare_batch_aggregates(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    candidate_batch_size: int,
    tolerance: float,
) -> dict[str, Any]:
    """Compare module-level mean expert loads against the batch=1 reference."""

    left = {(item["family"], item["module"]): item for item in aggregate_events(reference)}
    right = {(item["family"], item["module"]): item for item in aggregate_events(candidate)}
    rows = []
    for key in sorted(left):
        if key not in right:
            rows.append({"family": key[0], "module": key[1], "status": "MISSING", "max_abs_load_delta": None})
            continue
        left_load = np.asarray(left[key]["expert_load_mean"], dtype=np.float64)
        right_load = np.asarray(right[key]["expert_load_mean"], dtype=np.float64)
        delta = float(np.max(np.abs(left_load - right_load)))
        rows.append(
            {
                "family": key[0],
                "module": key[1],
                "status": "PASS" if delta <= tolerance else "FAIL",
                "max_abs_load_delta": delta,
            }
        )
    missing_reference = sorted(set(right).difference(left))
    rows.extend(
        {"family": family, "module": module, "status": "UNEXPECTED", "max_abs_load_delta": None}
        for family, module in missing_reference
    )
    numeric = [row["max_abs_load_delta"] for row in rows if row["max_abs_load_delta"] is not None]
    return {
        "status": "PASS" if rows and all(row["status"] == "PASS" for row in rows) else "FAIL",
        "reference_batch_size": 1,
        "candidate_batch_size": candidate_batch_size,
        "tolerance": tolerance,
        "max_abs_load_delta": max(numeric, default=None),
        "modules": rows,
    }
