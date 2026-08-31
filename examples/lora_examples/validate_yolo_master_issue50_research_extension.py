#!/usr/bin/env python3
"""Validate the audited Issue #50 research-extension tables without running training."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "yolo_master_issue50_audited_results.csv"
SUMMARY = ROOT / "yolo_master_issue50_seed_summary.csv"
PROTOCOL_ID = "issue50_research_extension_v1"
METRICS = ("precision", "recall", "mAP50", "mAP50_95")
OBJECTIVES = ("mAP50_95", "trainable_params", "peak_gpu_mem_gib", "train_time_s")

RESULT_FIELDS = (
    "protocol_id",
    "dataset",
    "method",
    "run_name",
    "seed",
    "rank",
    "alpha",
    "epochs",
    "imgsz",
    "fraction",
    "amp_strategy",
    "head_lr_scale",
    "router_lr_scale",
    "adapter_lr_scale",
    "requested_batch",
    "actual_batch",
    "effective_batch",
    "precision",
    "recall",
    "mAP50",
    "mAP50_95",
    "best_epoch",
    "epochs_completed",
    "trainable_params",
    "trainable_pct",
    "peak_gpu_mem_gib",
    "train_time_s",
    "inference_ms_per_image",
    "formal_validity_status",
    "recovery_status",
    "version_scope",
)

SUMMARY_FIELDS = (
    "protocol_id",
    "dataset",
    "method",
    "n",
    "seeds",
    "precision_mean",
    "precision_sample_std",
    "precision_best",
    "recall_mean",
    "recall_sample_std",
    "recall_best",
    "mAP50_mean",
    "mAP50_sample_std",
    "mAP50_best",
    "mAP50_95_mean",
    "mAP50_95_sample_std",
    "mAP50_95_best",
)


def read_table(path: Path, expected_fields: tuple[str, ...]) -> list[dict[str, str]]:
    """Read a CSV file and enforce its exact public schema."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == expected_fields, f"Unexpected schema: {path}"
        rows = list(reader)
    assert rows, f"Empty table: {path}"
    assert all(None not in row for row in rows), f"Malformed row in {path}"
    return rows


def validate_formal_rows(rows: list[dict[str, str]]) -> None:
    """Reject invalid statuses, duplicate evidence, and ambiguous batch fields."""
    identities = set()
    for row in rows:
        assert row["protocol_id"] == PROTOCOL_ID
        assert row["formal_validity_status"] == "passed"
        assert row["recovery_status"] == "none"
        assert row["requested_batch"] and row["actual_batch"]
        assert row["version_scope"] == "research_branch_includes_pr124_pr125_excludes_pr170_pr177"
        identity = (row["dataset"], row["method"], row["seed"], row["actual_batch"])
        assert identity not in identities, f"Duplicate formal seed: {identity}"
        identities.add(identity)
        for field in (*METRICS, "trainable_params", "peak_gpu_mem_gib", "train_time_s"):
            assert math.isfinite(float(row[field])), f"Non-finite {field}: {row['run_name']}"


def validate_summary(rows: list[dict[str, str]], summary: list[dict[str, str]]) -> None:
    """Recompute means, sample standard deviations, and best values."""
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"])].append(row)
    assert len(grouped) == len(summary)

    for item in summary:
        key = (item["dataset"], item["method"])
        group = sorted(grouped[key], key=lambda row: int(row["seed"]))
        seeds = [row["seed"] for row in group]
        assert item["protocol_id"] == PROTOCOL_ID
        assert int(item["n"]) == len(group)
        assert item["seeds"].split(",") == seeds
        assert len(seeds) == len(set(seeds))
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            assert math.isclose(float(item[f"{metric}_mean"]), statistics.mean(values), rel_tol=0, abs_tol=1e-12)
            assert math.isclose(float(item[f"{metric}_best"]), max(values), rel_tol=0, abs_tol=1e-12)
            std = item[f"{metric}_sample_std"]
            if len(values) == 1:
                assert std == "not_estimable"
            else:
                assert math.isclose(float(std), statistics.stdev(values), rel_tol=0, abs_tol=1e-12)


def dominates(left: dict[str, str], right: dict[str, str]) -> bool:
    """Return whether left dominates right over accuracy, parameters, memory, and time."""
    left_values = (
        float(left["mAP50_95"]),
        -float(left["trainable_params"]),
        -float(left["peak_gpu_mem_gib"]),
        -float(left["train_time_s"]),
    )
    right_values = (
        float(right["mAP50_95"]),
        -float(right["trainable_params"]),
        -float(right["peak_gpu_mem_gib"]),
        -float(right["train_time_s"]),
    )
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


def pareto_front(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    """Compute the stability-gated four-objective non-dominated front by dataset."""
    fronts = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        peers = [row for row in rows if row["dataset"] == dataset]
        front = [row for row in peers if not any(dominates(other, row) for other in peers if other is not row)]
        assert front, f"Empty Pareto front: {dataset}"
        assert all(not any(dominates(other, row) for other in peers if other is not row) for row in front)
        fronts[dataset] = [row["run_name"] for row in front]
    return fronts


def main() -> None:
    results = read_table(RESULTS, RESULT_FIELDS)
    summary = read_table(SUMMARY, SUMMARY_FIELDS)
    validate_formal_rows(results)
    validate_summary(results, summary)
    fronts = pareto_front(results)
    print(f"Validated {len(results)} formal rows and {len(summary)} method summaries.")
    for dataset, names in fronts.items():
        print(f"{dataset} Pareto front ({len(names)}): {', '.join(names)}")


if __name__ == "__main__":
    main()
