"""Synthetic cross-seed alignment and validity tests for Issue #54."""

from __future__ import annotations

import copy
from collections import Counter
import hashlib

import pytest

from scripts.issue54.analyze_cross_seed_routing import analyze_cross_seed_routing
from scripts.issue54.build_experiment_registry import build_registry
from scripts.issue54.schema import SchemaValidationError


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _manifest(seed: int, *, experiment_id: str | None = None):
    return {
        "schema_version": 1,
        "experiment_id": experiment_id or f"mot-seed-{seed}",
        "model_variant": "mot",
        "seed": seed,
        "dataset": "synthetic",
        "dataset_version": "phase1-v1",
        "dataset_manifest_sha256": _hash("dataset"),
        "split": "val",
        "requested_epochs": 10,
        "epochs": 10,
        "requested_batch": 2,
        "batch": 2,
        "effective_batch": 2,
        "imgsz": 8,
        "optimizer": "SGD",
        "precision_mode": "fp32",
        "checkpoint_path": f"synthetic/seed-{seed}/best.pt",
        "checkpoint_sha256": _hash(f"checkpoint-{seed}"),
        "config_path": "synthetic/mot.yaml",
        "config_sha256": _hash("config"),
        "git_commit": "a" * 40,
        "timestamp": "2026-07-30T00:00:00Z",
        "status": "passed",
        "failure_reason": None,
    }


def _record(
    seed: int,
    *,
    image_id: str = "image-001",
    layer_name: str = "model.14.m.0",
    layer_index: int = 0,
    probabilities=(0.8, 0.1, 0.1),
    token_indices=(0, 0, 0, 0),
    expert_names=("Local", "Window", "Deformable"),
    repeat: int = 0,
    scene_groups=None,
):
    probabilities = list(probabilities)
    expert_names = list(expert_names)
    return {
        "schema_version": 1,
        "experiment_id": f"mot-seed-{seed}",
        "model_variant": "mot",
        "seed": seed,
        "dataset": "synthetic",
        "dataset_version": "phase1-v1",
        "split": "val",
        "checkpoint_sha256": _hash(f"checkpoint-{seed}"),
        "image_id": image_id,
        "image_path": f"val/{image_id}.tensor",
        "image_sha256": _hash(image_id),
        "scene_groups": scene_groups or {"density": "dense"},
        "layer_name": layer_name,
        "layer_index": layer_index,
        "expert_names": expert_names,
        "expert_probabilities": probabilities,
        "selected_expert": expert_names[max(range(len(probabilities)), key=probabilities.__getitem__)],
        "top_k": 2,
        "token_top1_indices": list(token_indices),
        "spatial_shape": [2, 2],
        "inference_repeat": repeat,
        "inference_batch_actual": 1,
        "timestamp": "2026-07-30T00:00:00Z",
        "status": "passed",
        "failure_reason": None,
    }


def _analysis(seeds, records):
    return analyze_cross_seed_routing(build_registry([_manifest(seed) for seed in seeds]), records)


def test_identical_routes_align_by_id_not_input_order_and_are_reproducible():
    records = [
        _record(17, image_id="image-001"),
        _record(17, image_id="image-002"),
        _record(42, image_id="image-002"),
        _record(42, image_id="image-001"),
    ]

    result = _analysis([17, 42], records)
    reversed_result = _analysis([17, 42], list(reversed(records)))

    assert result == reversed_result
    assert len(result["pairwise_seed_comparisons"]) == 2
    assert all(row["dominant_expert_agreement"] == 1 for row in result["pairwise_seed_comparisons"])
    assert all(row["token_top1_agreement"] == 1.0 for row in result["pairwise_seed_comparisons"])
    assert all(row["jensen_shannon_divergence"] == pytest.approx(0.0) for row in result["pairwise_seed_comparisons"])
    assert result["formal_seed_counts_from_registry"]["mot"] == 2
    assert result["image_count"] == 2


def test_opposite_routes_and_small_probability_perturbations_are_preserved_pairwise():
    records = [
        _record(17, probabilities=(1.0, 0.0, 0.0), token_indices=(0, 0, 0, 0)),
        _record(42, probabilities=(0.0, 1.0, 0.0), token_indices=(1, 1, 1, 1)),
        _record(73, probabilities=(0.79, 0.11, 0.10), token_indices=(0, 0, 0, 0)),
    ]

    result = _analysis([17, 42, 73], records)
    rows = result["pairwise_seed_comparisons"]

    assert len(rows) == 3
    opposite = next(row for row in rows if {row["seed_a"], row["seed_b"]} == {17, 42})
    assert opposite["dominant_expert_agreement"] == 0
    assert opposite["token_top1_agreement"] == 0.0
    assert opposite["jensen_shannon_divergence"] == pytest.approx(1.0)
    assert result["formal_seed_counts_from_registry"]["mot"] == 3
    assert result["global_summary"][0]["distinct_seed_pairs"] == 3


def test_expert_order_is_aligned_by_name_semantics():
    reordered = _record(
        42,
        probabilities=(0.1, 0.8, 0.1),
        token_indices=(1, 1, 1, 1),
        expert_names=("Window", "Local", "Deformable"),
    )
    result = _analysis([17, 42], [_record(17), reordered])
    row = result["pairwise_seed_comparisons"][0]

    assert row["expert_names"] == ["Deformable", "Local", "Window"]
    assert row["dominant_expert_agreement"] == 1
    assert row["token_top1_agreement"] == 1.0
    assert row["jensen_shannon_divergence"] == pytest.approx(0.0)


def test_missing_image_and_layer_are_reported_without_inventing_alignment():
    records = [
        _record(17, image_id="shared"),
        _record(17, image_id="only-seed-17"),
        _record(17, image_id="shared", layer_name="model.20.m.0", layer_index=1),
        _record(42, image_id="shared"),
    ]

    result = _analysis([17, 42], records)
    issue_types = Counter(item["type"] for item in result["validation_issues"])

    assert issue_types["missing_image"] == 1
    assert issue_types["missing_layer"] == 1
    assert len(result["pairwise_seed_comparisons"]) == 1


def test_expert_schema_mismatch_is_fatal():
    malformed = _record(
        42,
        probabilities=(0.8, 0.2),
        token_indices=(0, 0, 0, 0),
        expert_names=("Local", "Window"),
    )
    malformed["top_k"] = 1

    with pytest.raises(SchemaValidationError, match="expert schema mismatch"):
        _analysis([17, 42], [_record(17), malformed])


def test_same_image_id_with_different_content_hash_is_fatal():
    malformed = _record(42)
    malformed["image_sha256"] = _hash("different-image-content")

    with pytest.raises(SchemaValidationError, match="multiple image_sha256"):
        _analysis([17, 42], [_record(17), malformed])


def test_repeat_exports_check_determinism_but_do_not_create_more_seeds():
    repeat = _record(17, repeat=1)
    result = _analysis([17], [_record(17, repeat=0), repeat])

    assert result["formal_seed_counts_from_registry"]["mot"] == 1
    assert result["analyzed_seed_counts"]["mot"] == 1
    assert result["pairwise_seed_comparisons"] == []
    assert result["checkpoint_determinism"][0]["status"] == "passed"


def test_probability_repeat_difference_fails_determinism_gate():
    changed = _record(17, repeat=1, probabilities=(0.79, 0.11, 0.10))
    result = _analysis([17], [_record(17), changed])

    assert result["checkpoint_determinism"][0]["status"] == "failed"
    comparison = result["checkpoint_determinism"][0]["repeat_comparisons"][0]
    assert comparison["token_top1_agreement"] == 1.0
    assert comparison["max_probability_absolute_difference"] == pytest.approx(0.01)


def test_image_count_never_changes_seed_level_and_all_seed_pairs_are_saved():
    seeds = [17, 42, 73]
    records = [_record(seed, image_id=f"image-{image:03d}") for seed in seeds for image in range(20)]
    result = _analysis(seeds, records)

    assert result["image_count"] == 20
    assert result["formal_seed_counts_from_registry"]["mot"] == 3
    assert result["analyzed_seed_counts"]["mot"] == 3
    assert len(result["pairwise_seed_comparisons"]) == 20 * 3
    assert result["global_summary"][0]["distinct_seed_pairs"] == 3


def test_one_and_two_seed_inputs_degrade_safely_without_significance_claims():
    one = _analysis([17], [_record(17)])
    two = _analysis([17, 42], [_record(17), _record(42)])

    assert one["global_summary"] == []
    assert one["formal_seed_counts_from_registry"]["mot"] == 1
    assert two["formal_seed_counts_from_registry"]["mot"] == 2
    assert len(two["pairwise_seed_comparisons"]) == 1
    assert "significance" not in str(two).lower()


def test_scene_reproducibility_keeps_raw_seed_effects_and_exploratory_label():
    records = []
    for seed in (17, 42, 73):
        records.extend(
            [
                _record(
                    seed,
                    image_id=f"sparse-{seed}",
                    probabilities=(0.8, 0.1, 0.1),
                    scene_groups={"density": "sparse"},
                ),
                _record(
                    seed,
                    image_id=f"dense-{seed}",
                    probabilities=(0.6, 0.2, 0.2),
                    scene_groups={"density": "dense"},
                ),
            ]
        )
    result = _analysis([17, 42, 73], records)
    local = next(
        row
        for row in result["scene_conclusion_reproducibility"]
        if row["expert_name"] == "Local" and {row["level_a"], row["level_b"]} == {"dense", "sparse"}
    )

    assert local["seed_count"] == 3
    assert len(local["effects_by_seed"]) == 3
    assert local["direction_reproduction_rate"] == 1.0
    assert local["inference_level"] == "exploratory_only"


def test_analysis_does_not_mutate_input_records():
    records = [_record(17), _record(42)]
    original = copy.deepcopy(records)

    _analysis([17, 42], records)

    assert records == original


def test_empty_routing_input_is_fatal():
    with pytest.raises(SchemaValidationError, match="at least one routing record"):
        _analysis([17], [])


def test_missing_repeat_zero_cannot_pass_determinism():
    result = _analysis([17], [_record(17, repeat=1), _record(17, repeat=2)])

    assert result["checkpoint_determinism"][0]["status"] == "invalid_missing_base_repeat"
    assert result["checkpoint_determinism"][0]["repeat_comparisons"] == []


def test_protocol_dimensions_are_not_combined_in_summaries_or_seed_counts():
    manifests = [_manifest(17), _manifest(42), _manifest(73), _manifest(101)]
    records = [_record(17), _record(42), _record(73), _record(101)]
    for manifest, record in zip(manifests[2:], records[2:]):
        manifest.update(dataset="other", dataset_version="v2", split="test")
        record.update(dataset="other", dataset_version="v2", split="test")

    result = analyze_cross_seed_routing(build_registry(manifests), records)

    assert result["analyzed_seed_counts"]["mot"] == 4
    assert len(result["analysis_groups"]) == 2
    assert {group["dataset"] for group in result["analysis_groups"]} == {"other", "synthetic"}
    assert len(result["global_summary"]) == 2
    assert result["image_count"] == 2
    assert {row["dataset"] for row in result["expert_utilization_between_seed"]} == {"other", "synthetic"}
