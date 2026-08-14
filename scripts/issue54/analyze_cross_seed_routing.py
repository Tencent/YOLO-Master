#!/usr/bin/env python3
"""Align independent MoT checkpoints and audit cross-seed routing stability."""

from __future__ import annotations

import argparse
import itertools
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.issue54.build_experiment_registry import inference_level, validate_registry  # noqa: E402
from scripts.issue54.schema import (  # noqa: E402
    ANALYSIS_SCHEMA_VERSION,
    FORMAL_STATUS,
    SchemaValidationError,
    canonical_payload_sha256,
    cli_error_message,
    ensure_outputs_available,
    load_json,
    load_jsonl,
    validate_routing_record,
    write_json,
)

RecordKey = Tuple[str, str, int]
ProtocolKey = Tuple[str, str, str, str]


def route_entropy(probabilities: Iterable[float]) -> tuple[float, float]:
    """Return raw and expert-count-normalized entropy for one probability vector."""
    values = [float(value) for value in probabilities]
    if not values:
        raise SchemaValidationError("entropy requires a non-empty probability vector")
    entropy = -math.fsum(value * math.log(max(value, 1e-12)) for value in values)
    normalized = entropy / math.log(len(values)) if len(values) > 1 else 0.0
    return entropy, normalized


def jensen_shannon_divergence(probabilities_a: Iterable[float], probabilities_b: Iterable[float]) -> float:
    """Return base-2 Jensen-Shannon divergence for two aligned probability vectors."""
    values_a = [float(value) for value in probabilities_a]
    values_b = [float(value) for value in probabilities_b]
    if not values_a or len(values_a) != len(values_b):
        raise SchemaValidationError("JSD requires aligned non-empty probability vectors")
    midpoint = [(left + right) / 2.0 for left, right in zip(values_a, values_b)]

    def kl(values: list[float], reference: list[float]) -> float:
        return math.fsum(
            value * math.log2(value / max(target, 1e-300)) for value, target in zip(values, reference) if value > 0.0
        )

    divergence = 0.5 * kl(values_a, midpoint) + 0.5 * kl(values_b, midpoint)
    return min(1.0, max(0.0, divergence))


def _mean(values: Iterable[float]) -> float | None:
    materialized = [float(value) for value in values]
    return math.fsum(materialized) / len(materialized) if materialized else None


def _sample_variance(values: Iterable[float]) -> float | None:
    materialized = [float(value) for value in values]
    return statistics.variance(materialized) if len(materialized) >= 2 else None


def _record_key(record: dict[str, Any]) -> RecordKey:
    return record["image_id"], record["layer_name"], record["layer_index"]


def _protocol_key(manifest: dict[str, Any]) -> ProtocolKey:
    return (
        manifest["model_variant"],
        manifest["dataset"],
        manifest["dataset_version"],
        manifest["split"],
    )


def _canonical_record(record: dict[str, Any], expert_names: tuple[str, ...]) -> dict[str, Any]:
    """Reorder probabilities and token indices by expert semantics, not original index."""
    source_names = record["expert_names"]
    if set(source_names) != set(expert_names):
        raise SchemaValidationError(
            f"expert schema mismatch for {record['experiment_id']}/{record['image_id']}/{record['layer_name']}: "
            f"{source_names} vs {list(expert_names)}"
        )
    source_index = {name: index for index, name in enumerate(source_names)}
    canonical_index = {name: index for index, name in enumerate(expert_names)}
    old_to_new = {old: canonical_index[name] for name, old in source_index.items()}
    normalized = dict(record)
    normalized["expert_names"] = list(expert_names)
    normalized["expert_probabilities"] = [record["expert_probabilities"][source_index[name]] for name in expert_names]
    normalized["token_top1_indices"] = [old_to_new[index] for index in record["token_top1_indices"]]
    normalized["selected_expert"] = expert_names[
        max(range(len(expert_names)), key=normalized["expert_probabilities"].__getitem__)
    ]
    return normalized


def _validate_record_identity(
    record: dict[str, Any],
    manifest: dict[str, Any],
    image_hashes: dict[tuple[str, str, str, str], str],
    layer_indices: dict[tuple[str, str], int],
) -> None:
    """Verify route metadata against the explicit registry rather than directory names."""
    for field in ("experiment_id", "model_variant", "seed", "dataset", "dataset_version", "split", "checkpoint_sha256"):
        if record[field] != manifest[field]:
            raise SchemaValidationError(
                f"routing record field {field!r} disagrees with manifest {manifest['experiment_id']!r}"
            )
    if record["status"] != manifest["status"]:
        raise SchemaValidationError("routing record status disagrees with its experiment manifest")
    image_key = (
        manifest["dataset"],
        manifest["dataset_version"],
        manifest["split"],
        record["image_id"],
    )
    previous_hash = image_hashes.setdefault(image_key, record["image_sha256"])
    if previous_hash != record["image_sha256"]:
        raise SchemaValidationError(f"image_id {record['image_id']!r} maps to multiple image_sha256 values")
    layer_key = (record["experiment_id"], record["layer_name"])
    previous_index = layer_indices.setdefault(layer_key, record["layer_index"])
    if previous_index != record["layer_index"]:
        raise SchemaValidationError(
            f"layer_name {record['layer_name']!r} maps to multiple indices in {record['experiment_id']!r}"
        )


def _prepare_records(
    registry: dict[str, Any], raw_records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manifests = {manifest["experiment_id"]: manifest for manifest in registry["experiments"]}
    image_hashes: dict[tuple[str, str, str, str], str] = {}
    layer_indices: dict[tuple[str, str], int] = {}
    duplicate_keys = set()
    normalized_records = []
    for raw in raw_records:
        record = validate_routing_record(raw)
        manifest = manifests.get(record["experiment_id"])
        if manifest is None:
            raise SchemaValidationError(f"routing record references unknown experiment_id={record['experiment_id']!r}")
        _validate_record_identity(record, manifest, image_hashes, layer_indices)
        duplicate_key = (
            record["experiment_id"],
            record["image_id"],
            record["layer_name"],
            record["layer_index"],
            record["inference_repeat"],
        )
        if duplicate_key in duplicate_keys:
            raise SchemaValidationError(f"duplicate routing record identity: {duplicate_key}")
        duplicate_keys.add(duplicate_key)
        normalized_records.append(record)
    normalized_records.sort(
        key=lambda item: (
            item["model_variant"],
            item["experiment_id"],
            item["image_id"],
            item["layer_index"],
            item["layer_name"],
            item["inference_repeat"],
        )
    )
    return normalized_records, manifests


def _analysis_experiment_ids(
    records: list[dict[str, Any]], manifests: dict[str, dict[str, Any]]
) -> tuple[set[str], str]:
    present = {record["experiment_id"] for record in records}
    formal = {experiment_id for experiment_id in present if manifests[experiment_id]["status"] == FORMAL_STATUS}
    if formal:
        return formal, "formal_passed_only"
    diagnostic = {experiment_id for experiment_id in present if manifests[experiment_id]["status"] == "diagnostic"}
    return diagnostic, "diagnostic_not_formal_evidence"


def _base_records(
    records: list[dict[str, Any]], experiment_ids: set[str]
) -> tuple[dict[str, dict[RecordKey, dict[str, Any]]], list[dict[str, Any]]]:
    by_experiment: dict[str, dict[RecordKey, dict[str, Any]]] = defaultdict(dict)
    issues = []
    repeats_by_key: dict[tuple[str, RecordKey], set[int]] = defaultdict(set)
    for record in records:
        if record["experiment_id"] not in experiment_ids:
            continue
        key = _record_key(record)
        repeats_by_key[(record["experiment_id"], key)].add(record["inference_repeat"])
        if record["inference_repeat"] == 0:
            by_experiment[record["experiment_id"]][key] = record
    for (experiment_id, key), repeats in sorted(repeats_by_key.items()):
        if 0 not in repeats:
            issues.append(
                {
                    "type": "missing_base_repeat",
                    "experiment_id": experiment_id,
                    "image_id": key[0],
                    "layer_name": key[1],
                    "layer_index": key[2],
                    "available_repeats": sorted(repeats),
                }
            )
    return by_experiment, issues


def _alignment_groups(experiment_ids: set[str], manifests: dict[str, dict[str, Any]]) -> dict[ProtocolKey, list[str]]:
    groups: dict[ProtocolKey, list[str]] = defaultdict(list)
    for experiment_id in experiment_ids:
        manifest = manifests[experiment_id]
        groups[_protocol_key(manifest)].append(experiment_id)
    return {key: sorted(values) for key, values in sorted(groups.items())}


def _alignment_issues(
    groups: dict[ProtocolKey, list[str]],
    base_records: dict[str, dict[RecordKey, dict[str, Any]]],
) -> list[dict[str, Any]]:
    issues = []
    for group_key, experiment_ids in groups.items():
        union = set().union(*(base_records.get(experiment_id, {}).keys() for experiment_id in experiment_ids))
        for key in sorted(union):
            for experiment_id in experiment_ids:
                if key not in base_records.get(experiment_id, {}):
                    image_present = any(item[0] == key[0] for item in base_records.get(experiment_id, {}))
                    issues.append(
                        {
                            "type": "missing_layer" if image_present else "missing_image",
                            "model_variant": group_key[0],
                            "experiment_id": experiment_id,
                            "image_id": key[0],
                            "layer_name": key[1],
                            "layer_index": key[2],
                        }
                    )
    return issues


def _pairwise_comparisons(
    groups: dict[ProtocolKey, list[str]],
    base_records: dict[str, dict[RecordKey, dict[str, Any]]],
    manifests: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for group_key, experiment_ids in groups.items():
        for experiment_a, experiment_b in itertools.combinations(experiment_ids, 2):
            keys = sorted(set(base_records.get(experiment_a, {})).intersection(base_records.get(experiment_b, {})))
            for key in keys:
                record_a = base_records[experiment_a][key]
                record_b = base_records[experiment_b][key]
                canonical_names = tuple(sorted(record_a["expert_names"]))
                record_a = _canonical_record(record_a, canonical_names)
                record_b = _canonical_record(record_b, canonical_names)
                if record_a["spatial_shape"] != record_b["spatial_shape"]:
                    token_agreement = None
                else:
                    indices_a = record_a["token_top1_indices"]
                    indices_b = record_b["token_top1_indices"]
                    token_agreement = _mean(float(left == right) for left, right in zip(indices_a, indices_b))
                entropy_a, normalized_entropy_a = route_entropy(record_a["expert_probabilities"])
                entropy_b, normalized_entropy_b = route_entropy(record_b["expert_probabilities"])
                rows.append(
                    {
                        "model_variant": group_key[0],
                        "dataset": group_key[1],
                        "dataset_version": group_key[2],
                        "split": group_key[3],
                        "experiment_a": experiment_a,
                        "experiment_b": experiment_b,
                        "seed_a": manifests[experiment_a]["seed"],
                        "seed_b": manifests[experiment_b]["seed"],
                        "checkpoint_sha256_a": manifests[experiment_a]["checkpoint_sha256"],
                        "checkpoint_sha256_b": manifests[experiment_b]["checkpoint_sha256"],
                        "image_id": key[0],
                        "layer_name": key[1],
                        "layer_index": key[2],
                        "expert_names": list(canonical_names),
                        "dominant_expert_agreement": int(record_a["selected_expert"] == record_b["selected_expert"]),
                        "token_top1_agreement": token_agreement,
                        "jensen_shannon_divergence": jensen_shannon_divergence(
                            record_a["expert_probabilities"], record_b["expert_probabilities"]
                        ),
                        "route_entropy_a": entropy_a,
                        "route_entropy_b": entropy_b,
                        "normalized_route_entropy_a": normalized_entropy_a,
                        "normalized_route_entropy_b": normalized_entropy_b,
                        "max_probability_absolute_difference": max(
                            abs(left - right)
                            for left, right in zip(record_a["expert_probabilities"], record_b["expert_probabilities"])
                        ),
                        "expert_probability_differences_b_minus_a": {
                            name: record_b["expert_probabilities"][index] - record_a["expert_probabilities"][index]
                            for index, name in enumerate(canonical_names)
                        },
                    }
                )
    return rows


def _summarize_pairwise(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    summaries = []
    for group_key, items in sorted(groups.items()):
        summary = {key: value for key, value in zip(keys, group_key)}
        summary.update(
            {
                "aligned_comparisons": len(items),
                "distinct_seed_pairs": len({(item["seed_a"], item["seed_b"]) for item in items}),
                "mean_dominant_expert_agreement": _mean(item["dominant_expert_agreement"] for item in items),
                "mean_token_top1_agreement": _mean(
                    item["token_top1_agreement"] for item in items if item["token_top1_agreement"] is not None
                ),
                "mean_jensen_shannon_divergence": _mean(item["jensen_shannon_divergence"] for item in items),
                "mean_max_probability_absolute_difference": _mean(
                    item["max_probability_absolute_difference"] for item in items
                ),
            }
        )
        summaries.append(summary)
    return summaries


def _expert_utilization(
    base_records: dict[str, dict[RecordKey, dict[str, Any]]],
    manifests: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accumulators: dict[tuple[str, str, str, str, str, str, int], dict[str, Any]] = {}
    for experiment_id, records in base_records.items():
        manifest = manifests[experiment_id]
        for record in records.values():
            key = (
                manifest["model_variant"],
                manifest["dataset"],
                manifest["dataset_version"],
                manifest["split"],
                experiment_id,
                record["layer_name"],
                record["layer_index"],
            )
            names = tuple(sorted(record["expert_names"]))
            canonical = _canonical_record(record, names)
            state = accumulators.setdefault(
                key,
                {"expert_names": names, "counts": Counter(), "tokens": 0},
            )
            if state["expert_names"] != names:
                raise SchemaValidationError(f"expert schema changes within experiment/layer: {key}")
            state["counts"].update(canonical["token_top1_indices"])
            state["tokens"] += len(canonical["token_top1_indices"])

    utilization = []
    for (
        model_variant,
        dataset,
        dataset_version,
        split,
        experiment_id,
        layer_name,
        layer_index,
    ), state in sorted(accumulators.items()):
        manifest = manifests[experiment_id]
        for expert_index, expert_name in enumerate(state["expert_names"]):
            utilization.append(
                {
                    "model_variant": model_variant,
                    "dataset": dataset,
                    "dataset_version": dataset_version,
                    "split": split,
                    "experiment_id": experiment_id,
                    "seed": manifest["seed"],
                    "checkpoint_sha256": manifest["checkpoint_sha256"],
                    "layer_name": layer_name,
                    "layer_index": layer_index,
                    "expert_name": expert_name,
                    "selected_tokens": state["counts"][expert_index],
                    "total_tokens": state["tokens"],
                    "utilization": state["counts"][expert_index] / state["tokens"] if state["tokens"] else None,
                }
            )

    grouped: dict[tuple[str, str, str, str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in utilization:
        grouped[
            (
                row["model_variant"],
                row["dataset"],
                row["dataset_version"],
                row["split"],
                row["layer_name"],
                row["layer_index"],
                row["expert_name"],
            )
        ].append(row)
    between_seed = []
    for key, items in sorted(grouped.items()):
        values = [item["utilization"] for item in items if item["utilization"] is not None]
        seed_count = len({item["seed"] for item in items})
        mean_value = _mean(values)
        variance = _sample_variance(values)
        between_seed.append(
            {
                "model_variant": key[0],
                "dataset": key[1],
                "dataset_version": key[2],
                "split": key[3],
                "layer_name": key[4],
                "layer_index": key[5],
                "expert_name": key[6],
                "seed_count": seed_count,
                "seeds": sorted({item["seed"] for item in items}),
                "utilization_by_seed": [
                    {
                        "experiment_id": item["experiment_id"],
                        "seed": item["seed"],
                        "utilization": item["utilization"],
                    }
                    for item in sorted(items, key=lambda value: (value["seed"], value["experiment_id"]))
                ],
                "mean_utilization": mean_value,
                "between_seed_variance": variance,
                "between_seed_standard_deviation": math.sqrt(variance) if variance is not None else None,
                "inference_level": inference_level(seed_count),
            }
        )
    return utilization, between_seed


def _scene_reproducibility(
    base_records: dict[str, dict[RecordKey, dict[str, Any]]],
    manifests: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    observations: dict[tuple[str, str, int, str, str, str], list[float]] = defaultdict(list)
    for experiment_id, records in base_records.items():
        for record in records.values():
            names = tuple(sorted(record["expert_names"]))
            canonical = _canonical_record(record, names)
            for dimension, level in canonical["scene_groups"].items():
                for expert_index, expert_name in enumerate(names):
                    observations[
                        (
                            experiment_id,
                            canonical["layer_name"],
                            canonical["layer_index"],
                            dimension,
                            level,
                            expert_name,
                        )
                    ].append(canonical["expert_probabilities"][expert_index])

    level_index: dict[tuple[str, str, str, str, str, int, str, str], set[str]] = defaultdict(set)
    for experiment_id, layer_name, layer_index, dimension, level, expert_name in observations:
        protocol = _protocol_key(manifests[experiment_id])
        level_index[(*protocol, layer_name, layer_index, dimension, expert_name)].add(level)

    rows = []
    for key, levels in sorted(level_index.items()):
        for level_a, level_b in itertools.combinations(sorted(levels), 2):
            effects = []
            for experiment_id, manifest in sorted(manifests.items()):
                if _protocol_key(manifest) != key[:4] or experiment_id not in base_records:
                    continue
                values_a = observations.get((experiment_id, key[4], key[5], key[6], level_a, key[7]), [])
                values_b = observations.get((experiment_id, key[4], key[5], key[6], level_b, key[7]), [])
                if not values_a or not values_b:
                    continue
                effects.append(
                    {
                        "experiment_id": experiment_id,
                        "seed": manifest["seed"],
                        "effect_b_minus_a": _mean(values_b) - _mean(values_a),
                        "images_a": len(values_a),
                        "images_b": len(values_b),
                    }
                )
            if not effects:
                continue
            signs = Counter(
                "positive" if item["effect_b_minus_a"] > 0 else "negative" if item["effect_b_minus_a"] < 0 else "zero"
                for item in effects
            )
            consensus_direction, consensus_count = sorted(signs.items(), key=lambda item: (-item[1], item[0]))[0]
            seed_count = len({item["seed"] for item in effects})
            rows.append(
                {
                    "model_variant": key[0],
                    "dataset": key[1],
                    "dataset_version": key[2],
                    "split": key[3],
                    "layer_name": key[4],
                    "layer_index": key[5],
                    "scene_dimension": key[6],
                    "level_a": level_a,
                    "level_b": level_b,
                    "expert_name": key[7],
                    "seed_count": seed_count,
                    "effects_by_seed": effects,
                    "consensus_direction": consensus_direction,
                    "direction_reproduction_rate": consensus_count / len(effects),
                    "mean_effect_b_minus_a": _mean(item["effect_b_minus_a"] for item in effects),
                    "inference_level": inference_level(seed_count),
                }
            )
    return rows


def _determinism(
    records: list[dict[str, Any]],
    experiment_ids: set[str],
    *,
    probability_tolerance: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, RecordKey], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["experiment_id"] in experiment_ids:
            grouped[(record["experiment_id"], _record_key(record))].append(record)
    rows = []
    for (experiment_id, key), items in sorted(grouped.items()):
        items.sort(key=lambda item: item["inference_repeat"])
        baseline_items = [item for item in items if item["inference_repeat"] == 0]
        if not baseline_items:
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "image_id": key[0],
                    "layer_name": key[1],
                    "layer_index": key[2],
                    "status": "invalid_missing_base_repeat",
                    "repeat_comparisons": [],
                }
            )
            continue
        if len(items) < 2:
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "image_id": key[0],
                    "layer_name": key[1],
                    "layer_index": key[2],
                    "status": "not_checked",
                    "repeat_comparisons": [],
                }
            )
            continue
        canonical_names = tuple(sorted(baseline_items[0]["expert_names"]))
        baseline = _canonical_record(baseline_items[0], canonical_names)
        comparisons = []
        for changed in (item for item in items if item["inference_repeat"] != 0):
            changed = _canonical_record(changed, canonical_names)
            same_shape = baseline["spatial_shape"] == changed["spatial_shape"]
            token_agreement = (
                _mean(
                    float(left == right)
                    for left, right in zip(baseline["token_top1_indices"], changed["token_top1_indices"])
                )
                if same_shape
                else None
            )
            max_difference = max(
                abs(left - right)
                for left, right in zip(baseline["expert_probabilities"], changed["expert_probabilities"])
            )
            comparisons.append(
                {
                    "baseline_repeat": baseline["inference_repeat"],
                    "repeat": changed["inference_repeat"],
                    "token_top1_agreement": token_agreement,
                    "jensen_shannon_divergence": jensen_shannon_divergence(
                        baseline["expert_probabilities"], changed["expert_probabilities"]
                    ),
                    "max_probability_absolute_difference": max_difference,
                    "passed": bool(token_agreement == 1.0 and max_difference <= probability_tolerance),
                }
            )
        rows.append(
            {
                "experiment_id": experiment_id,
                "image_id": key[0],
                "layer_name": key[1],
                "layer_index": key[2],
                "status": "passed" if all(item["passed"] for item in comparisons) else "failed",
                "repeat_comparisons": comparisons,
            }
        )
    return rows


def analyze_cross_seed_routing(
    registry: dict[str, Any],
    raw_records: list[dict[str, Any]],
    *,
    determinism_tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Return deterministic cross-seed summaries without image-level significance claims."""
    registry = validate_registry(registry)
    if not raw_records:
        raise SchemaValidationError("routing analysis requires at least one routing record")
    records, manifests = _prepare_records(registry, raw_records)
    experiment_ids, analysis_mode = _analysis_experiment_ids(records, manifests)
    if not experiment_ids:
        raise SchemaValidationError("routing analysis found no passed or diagnostic routing evidence")
    base_records, base_issues = _base_records(records, experiment_ids)
    groups = _alignment_groups(experiment_ids, manifests)
    validation_issues = base_issues + _alignment_issues(groups, base_records)
    pairwise = _pairwise_comparisons(groups, base_records, manifests)
    utilization, utilization_variance = _expert_utilization(base_records, manifests)
    scene_reproducibility = _scene_reproducibility(base_records, manifests)
    determinism = _determinism(records, experiment_ids, probability_tolerance=determinism_tolerance)

    analyzed_seeds: dict[str, set[int]] = defaultdict(set)
    analysis_groups = []
    for group_key, ids in groups.items():
        seeds = sorted({manifests[experiment_id]["seed"] for experiment_id in ids})
        analyzed_seeds[group_key[0]].update(seeds)
        analysis_groups.append(
            {
                "model_variant": group_key[0],
                "dataset": group_key[1],
                "dataset_version": group_key[2],
                "split": group_key[3],
                "experiment_ids": ids,
                "seeds": seeds,
                "seed_count": len(seeds),
                "inference_level": inference_level(len(seeds)),
            }
        )
    analyzed_seed_counts = {variant: len(seeds) for variant, seeds in sorted(analyzed_seeds.items())}
    formal_seed_counts = {
        variant: summary["formal_seed_count"] for variant, summary in registry["variant_summary"].items()
    }
    analysis = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_mode": analysis_mode,
        "registry_sha256": registry["registry_sha256"],
        "record_count": len(records),
        "image_count": len(
            {
                (record["dataset"], record["dataset_version"], record["split"], record["image_id"])
                for record in records
                if record["experiment_id"] in experiment_ids
            }
        ),
        "formal_seed_counts_from_registry": formal_seed_counts,
        "analyzed_seed_counts": analyzed_seed_counts,
        "analysis_groups": analysis_groups,
        "inference_guard": (
            "Independent training seed/checkpoint is the highest-level unit. Image, layer, token, inference repeat, "
            "and pairwise seed rows never increase formal seed count. Three seeds are exploratory; fewer than three "
            "are insufficient; stronger claims require at least five pre-registered seeds."
        ),
        "validation_issues": validation_issues,
        "checkpoint_determinism": determinism,
        "pairwise_seed_comparisons": pairwise,
        "per_image_summary": _summarize_pairwise(
            pairwise,
            ("model_variant", "dataset", "dataset_version", "split", "image_id"),
        ),
        "per_layer_summary": _summarize_pairwise(
            pairwise,
            ("model_variant", "dataset", "dataset_version", "split", "layer_name", "layer_index"),
        ),
        "global_summary": _summarize_pairwise(
            pairwise,
            ("model_variant", "dataset", "dataset_version", "split"),
        ),
        "expert_utilization_by_seed": utilization,
        "expert_utilization_between_seed": utilization_variance,
        "scene_conclusion_reproducibility": scene_reproducibility,
    }
    analysis["analysis_sha256"] = canonical_payload_sha256(analysis, exclude=("analysis_sha256",))
    return analysis


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--routes", type=Path, action="append", required=True, help="Routing JSONL; repeatable.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--determinism-tolerance", type=float, default=1e-7)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file explicitly.")
    return parser.parse_args()


def main() -> int:
    """Load evidence, run the cross-seed audit, and write deterministic JSON."""
    args = parse_args()
    records = []
    for path in args.routes:
        records.extend(load_jsonl(path))
    analysis = analyze_cross_seed_routing(
        load_json(args.registry),
        records,
        determinism_tolerance=args.determinism_tolerance,
    )
    ensure_outputs_available([args.output], overwrite=args.overwrite)
    write_json(args.output, analysis, overwrite=args.overwrite)
    print(
        f"[issue54-analysis] wrote {ascii(args.output.name)}; mode={analysis['analysis_mode']}; "
        f"formal_seeds={analysis['formal_seed_counts_from_registry']}; "
        f"pairwise_rows={len(analysis['pairwise_seed_comparisons'])}"
    )
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except (SchemaValidationError, OSError) as error:
        print(f"[issue54-analysis] error: {cli_error_message(error)}", file=sys.stderr)
        exit_code = 2
    raise SystemExit(exit_code)
