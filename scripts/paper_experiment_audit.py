#!/usr/bin/env python3
"""Audit PEFT training manifests against the paper's review-critical protocol.

This tool deliberately separates smoke-test evidence from submission-grade
evidence. It consumes the ``runs.json`` records produced by
``planner_mps_coco128_calibration.py`` and requires a matched Full-SFT baseline
for every PEFT observation before computing a delta. It does not fit a model or
emit a performance claim.

Usage:
    python scripts/paper_experiment_audit.py --runs runs/*/runs.json --report reports/paper_audit.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ultralytics.utils.lora.planner import ArchitectureFingerprint, LOVODataPoint, LOVOValidator


METRIC = "metrics/mAP50-95(B)"
PROTOCOL_KEYS = (
    "dataset", "epochs", "imgsz", "batch", "optimizer", "lr0", "lrf",
    "weight_decay", "momentum", "amp", "cos_lr", "close_mosaic", "warmup_epochs",
)
CONTROL_KEYS = PROTOCOL_KEYS + (
    "lora_backend", "lora_type", "lora_dropout", "lora_alpha", "lora_use_rslora",
    "lora_lr_mult", "lora_include_attention", "lora_gradient_checkpointing",
    "lora_alpha_warmup", "lora_layer_decay", "training_budget", "deterministic", "workers",
)


def _load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list in {path}")
        for record in payload:
            if not isinstance(record, dict):
                raise ValueError(f"Invalid record in {path}")
            records.append({**record, "_source": str(path)})
    return records


def _baseline_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (record.get("model_name"), record.get("seed"), *(record.get(key) for key in PROTOCOL_KEYS))


def _metric(record: dict[str, Any]) -> float | None:
    value = (record.get("metrics") or {}).get(METRIC)
    return float(value) if isinstance(value, (int, float)) else None


def _family(record: dict[str, Any]) -> str:
    return str(record.get("architecture_family") or "unknown")


def _placement(record: dict[str, Any]) -> str | None:
    metadata = record.get("lora_runtime_metadata") or {}
    targets = metadata.get("target_modules") or record.get("target_modules_requested")
    if not isinstance(targets, list) or not targets:
        return None
    return json.dumps(sorted(map(str, targets)), separators=(",", ":"))


def _placement_label(record: dict[str, Any]) -> str:
    return str(record.get("target_set") or record.get("placement") or _placement(record) or "unknown")


def _control_signature(record: dict[str, Any]) -> tuple[Any, ...]:
    """Return all training controls that must match in a placement pair."""
    return tuple(record.get(key) for key in CONTROL_KEYS)


def _fingerprint(record: dict[str, Any]) -> ArchitectureFingerprint:
    payload = record.get("fingerprint") or {}
    return ArchitectureFingerprint(**{key: payload.get(key, 0.0) for key in ArchitectureFingerprint.__dataclass_fields__})


def _lovo_points(observations: list[dict[str, Any]]) -> list[LOVODataPoint]:
    points = []
    for item in observations:
        points.append(
            LOVODataPoint(
                fingerprint=_fingerprint(item["record"]),
                variant=str(item["variant"]),
                rank=max(int(item["rank"] or 1), 1),
                delta_mAP=float(item["delta_map"]),
                model_name=str(item["model"] or ""),
                dataset=str(item["dataset"] or ""),
                notes=str(item["experiment_id"] or ""),
            )
        )
    return points


def audit(records: list[dict[str, Any]], min_seeds: int = 3, min_datasets: int = 2) -> dict[str, Any]:
    successes = [record for record in records if record.get("status") == "success" and _metric(record) is not None]
    baselines = {
        _baseline_key(record): record
        for record in successes
        if record.get("variant") == "full"
    }
    observations = []
    unmatched = []
    for record in successes:
        if record.get("variant") == "full":
            continue
        baseline = baselines.get(_baseline_key(record))
        if baseline is None:
            unmatched.append(record.get("experiment_id"))
            continue
        observations.append(
            {
                "experiment_id": record.get("experiment_id"),
                "model": record.get("model_name"),
                "family": _family(record),
                "dataset": record.get("dataset"),
                "seed": record.get("seed"),
                "variant": record.get("variant"),
                "rank": record.get("rank"),
                "delta_map": _metric(record) - _metric(baseline),
                "placement": _placement(record),
                "target_set": _placement_label(record),
                "control_signature": _control_signature(record),
                "record": record,
                "source": record.get("_source", "inline"),
            }
        )

    by_family = Counter(item["family"] for item in observations)
    by_variant = Counter(item["variant"] for item in observations)
    datasets = sorted({str(item["dataset"]) for item in observations})
    variants = sorted({str(item["variant"]) for item in observations})
    placements = sorted({item["target_set"] for item in observations})
    seeds_by_protocol: dict[tuple[Any, ...], set[Any]] = defaultdict(set)
    for item in observations:
        seeds_by_protocol[
            (item["model"], item["dataset"], item["variant"], item["rank"], item["target_set"])
        ].add(item["seed"])
    insufficient_seed_groups = [
        {"model": key[0], "dataset": key[1], "variant": key[2], "rank": key[3], "target_set": key[4], "seeds": sorted(values)}
        for key, values in sorted(seeds_by_protocol.items(), key=lambda pair: str(pair[0]))
        if len(values) < min_seeds
    ]

    baseline_seed_groups: dict[tuple[Any, ...], set[Any]] = defaultdict(set)
    for record in successes:
        if record.get("variant") == "full":
            baseline_seed_groups[(record.get("model_name"), record.get("dataset"))].add(record.get("seed"))
    insufficient_baseline_groups = [
        {"model": key[0], "dataset": key[1], "seeds": sorted(values)}
        for key, values in sorted(baseline_seed_groups.items(), key=lambda pair: str(pair[0]))
        if len(values) < min_seeds
    ]

    placement_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in observations:
        placement_groups[(item["model"], item["dataset"], item["seed"], item["variant"], item["rank"])].append(item)
    controlled_placement = []
    placement_mismatches = []
    for key, items in placement_groups.items():
        placements = {_placement_label(item["record"]) for item in items}
        controls = {item["control_signature"] for item in items}
        target_payloads = {_placement(item["record"]) for item in items}
        if len(placements) >= 2 and None not in target_payloads:
            if len(controls) == 1 and len(target_payloads) >= 2:
                controlled_placement.append({"key": key, "placements": sorted(placements), "n_runs": len(items)})
            else:
                placement_mismatches.append(
                    {"key": key, "n_runs": len(items), "n_control_signatures": len(controls)}
                )

    points = _lovo_points(observations)
    variant_lovo = None
    architecture_loao = None
    grouped_error = None
    if len(points) >= 5 and len(by_variant) >= 3:
        try:
            validator = LOVOValidator()
            variant_lovo = validator.cross_validate_variant(points, paper=True)
        except ValueError as exc:
            grouped_error = str(exc)
    complete_models = {}
    incomplete_models = []
    for family in sorted(by_family):
        family_models = sorted({item["model"] for item in observations if item["family"] == family})
        for model in family_models:
            model_items = [item for item in observations if item["model"] == model]
            datasets_for_model = {item["dataset"] for item in model_items}
            variants_for_model = {item["variant"] for item in model_items}
            missing_cells = []
            for dataset in datasets:
                for variant in variants:
                    for placement in placements:
                        seeds = {
                            item["seed"]
                            for item in model_items
                            if item["dataset"] == dataset
                            and item["variant"] == variant
                            and item["target_set"] == placement
                        }
                        if len(seeds) < min_seeds:
                            missing_cells.append(
                                {"dataset": dataset, "variant": variant, "target_set": placement, "seeds": sorted(seeds)}
                            )
            baseline_datasets = {
                record.get("dataset")
                for record in successes
                if record.get("variant") == "full" and record.get("model_name") == model
            }
            baseline_seed_missing = []
            for dataset in datasets:
                seeds = {
                    record.get("seed")
                    for record in successes
                    if record.get("variant") == "full"
                    and record.get("model_name") == model
                    and record.get("dataset") == dataset
                }
                if len(seeds) < min_seeds:
                    baseline_seed_missing.append({"dataset": dataset, "seeds": sorted(seeds)})
            if (
                len(datasets_for_model) >= min_datasets
                and len(variants_for_model) >= 3
                and not missing_cells
                and not baseline_seed_missing
            ):
                complete_models[family] = model
                break
            incomplete_models.append(
                {"family": family, "model": model, "missing_cells": missing_cells, "missing_baselines": baseline_seed_missing}
            )
    if len(complete_models) >= 3 and len(points) >= 5:
        try:
            selected = set(complete_models.values())
            architecture_loao = LOVOValidator().cross_validate_architecture(
                [point for point in points if point.model_name in selected],
                holdout_models=sorted(selected),
                paper=True,
            )
        except ValueError as exc:
            grouped_error = str(exc)

    architecture_ready = len(complete_models) >= 3 and architecture_loao is not None
    variant_ready = len(by_variant) >= 3 and variant_lovo is not None
    data_ready = len(datasets) >= min_datasets
    seed_ready = not insufficient_seed_groups and not insufficient_baseline_groups and bool(observations)
    placement_ready = bool(controlled_placement)
    eligible = all((architecture_ready, variant_ready, data_ready, seed_ready, placement_ready))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metric": METRIC,
        "input_records": len(records),
        "successful_records": len(successes),
        "matched_peft_observations": len(observations),
        "unmatched_peft_experiments": unmatched,
        "coverage": {"architectures": dict(sorted(by_family.items())), "variants": dict(sorted(by_variant.items())), "datasets": datasets},
        "gates": {
            "leave_one_architecture_ready": architecture_ready,
            "leave_one_variant_ready": variant_ready,
            "cross_dataset_ready": data_ready,
            "three_seed_ready": seed_ready,
            "controlled_placement_ready": placement_ready,
            "submission_grade": eligible,
        },
        "insufficient_seed_groups": insufficient_seed_groups,
        "insufficient_baseline_seed_groups": insufficient_baseline_groups,
        "complete_architectures_for_loao": complete_models,
        "incomplete_architectures": incomplete_models,
        "variant_lovo": variant_lovo,
        "architecture_loao": architecture_loao,
        "grouped_validation_error": grouped_error,
        "controlled_placement_groups": controlled_placement,
        "placement_mismatches": placement_mismatches,
        "recommendation": (
            "Eligible for preregistered held-out analyses."
            if eligible
            else "Smoke-only: do not use for paper claims. Add matched runs across >=3 seeds, >=2 datasets, "
            "and explicit manual-vs-planner target-set pairs."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, action="append", required=True, help="Path to a runs.json file; repeatable")
    parser.add_argument("--report", type=Path, required=True, help="Output audit report")
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--min-datasets", type=int, default=2)
    args = parser.parse_args()
    report = audit(_load_records(args.runs), min_seeds=args.min_seeds, min_datasets=args.min_datasets)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["gates"], indent=2))
    return 0 if report["gates"]["submission_grade"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
