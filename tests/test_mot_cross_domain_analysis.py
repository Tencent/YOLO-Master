"""Tests for the same-checkpoint MoT cross-domain routing audit."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import scripts.compare_mot_ablation as mot_ablation
from scripts.analyze_mot_cross_domain import (
    EXPERT_NAMES,
    benjamini_hochberg,
    bootstrap_mean_diff_ci,
    jensen_shannon_divergence,
    load_image_tensor,
    normalize_image_array,
    pairwise_statistics,
    parse_domain_specs,
    permutation_p_value_two_sided,
    resolve_cluster_id,
    robustness_statistics,
    sample_overlap_summary,
)
from scripts.compare_mot_ablation import (
    SPECS,
    aggregate_benchmark_rounds,
    benchmark_row,
    build_model,
    deterministic_benchmark_input,
    read_best_observed_metrics,
    stability_from_results,
)
from scripts.prepare_mot_routing_scenes import (
    ImageStats,
    OcclusionStats,
    match_paired_occlusion_scenes,
    parse_visdrone_occlusion,
)
from ultralytics.nn.modules.mot import MoTBlock


def test_parse_domain_specs_resolves_paths_and_rejects_duplicates(tmp_path: Path):
    specs = parse_domain_specs([f"medical={tmp_path}", "aerial=images"], root=tmp_path)

    assert specs[0].name == "medical"
    assert specs[0].path == tmp_path.resolve()
    assert specs[1].path == (tmp_path / "images").resolve()
    with pytest.raises(ValueError, match="duplicate"):
        parse_domain_specs([f"medical={tmp_path}", f"MEDICAL={tmp_path}"])
    with pytest.raises(ValueError, match="NAME=PATH"):
        parse_domain_specs([str(tmp_path)])


def test_normalize_uint16_grayscale_to_rgb():
    image = np.array([[0, 512], [4096, 65535]], dtype=np.uint16)

    normalized = normalize_image_array(image)

    assert normalized.shape == (2, 2, 3)
    assert normalized.dtype == np.float32
    assert normalized.min() == pytest.approx(0.0)
    assert normalized.max() == pytest.approx(1.0)
    np.testing.assert_allclose(normalized[..., 0], normalized[..., 1])


def test_load_image_tensor_supports_16_bit_tiff_letterbox(tmp_path: Path):
    array = np.arange(8 * 16, dtype=np.uint16).reshape(8, 16) * 257
    path = tmp_path / "mri.tiff"
    Image.fromarray(array).save(path)

    tensor = load_image_tensor(path, imgsz=32)

    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0
    assert torch.allclose(tensor[:, 0, 0], torch.full((3,), 114.0 / 255.0))


def test_statistical_helpers_are_deterministic_and_directional():
    values_a = np.array([0.1, 0.2, 0.3, 0.4])
    values_b = np.array([0.7, 0.8, 0.9, 1.0])

    interval_1 = bootstrap_mean_diff_ci(values_a, values_b, samples=200, seed=7)
    interval_2 = bootstrap_mean_diff_ci(values_a, values_b, samples=200, seed=7)
    p_value_1 = permutation_p_value_two_sided(values_a, values_b, permutations=199, seed=7)
    p_value_2 = permutation_p_value_two_sided(values_a, values_b, permutations=199, seed=7)

    assert interval_1 == interval_2
    assert interval_1[0] > 0
    assert p_value_1 == p_value_2
    assert 0.0 < p_value_1 <= 1.0
    q_values = benjamini_hochberg([0.01, 0.04, 0.03])
    assert q_values == pytest.approx([0.03, 0.04, 0.04])


def test_cluster_regex_uses_capture_and_falls_back_to_sample_identity():
    pattern = re.compile(r"(?:^|__)(?P<cluster>[0-9]{7})_")

    assert resolve_cluster_id("images__val__0000291_01001_d_1.jpg", "sha-a", pattern) == "0000291"
    assert resolve_cluster_id("unclustered.jpg", "sha-b", pattern) == "sha-b"
    assert resolve_cluster_id("anything.jpg", "sha-c", None) == "sha-c"


def make_routing_records() -> list[dict]:
    records = []
    distributions = {
        "aerial": (0.75, 0.20, 0.05),
        "medical": (0.10, 0.20, 0.70),
    }
    for domain, probabilities in distributions.items():
        for image_index in range(5):
            for layer in ("layer_1", "layer_2"):
                for expert_id, (expert, probability) in enumerate(zip(EXPERT_NAMES, probabilities)):
                    records.append(
                        {
                            "domain": domain,
                            "perturbation": "base",
                            "image_id": f"{image_index}.png",
                            "layer": layer,
                            "expert_id": expert_id,
                            "expert": expert,
                            "top1_share": probability,
                            "mean_weight": probability,
                            "mean_probability": probability,
                            "normalized_entropy": 0.5,
                            "effective_experts": 2.0,
                            "top1_margin": 0.4,
                        }
                    )
                    changed = list(probabilities)
                    changed[0], changed[2] = changed[2], changed[0]
                    records.append(
                        {
                            "domain": domain,
                            "perturbation": "hflip",
                            "image_id": f"{image_index}.png",
                            "layer": layer,
                            "expert_id": expert_id,
                            "expert": expert,
                            "top1_share": changed[expert_id],
                            "mean_weight": changed[expert_id],
                            "mean_probability": changed[expert_id],
                            "normalized_entropy": 0.6,
                            "effective_experts": 2.1,
                            "top1_margin": 0.2,
                        }
                    )
    return records


def test_pairwise_and_robustness_statistics_use_image_level_units():
    records = make_routing_records()

    comparisons = pairwise_statistics(records, bootstrap_samples=100, permutations=99, seed=42, alpha=0.05)
    detailed, summary = robustness_statistics(records)

    local_probability = next(
        row for row in comparisons if row["expert"] == "LocalConvTransformer" and row["metric"] == "mean_probability"
    )
    assert local_probability["n_a"] == 5
    assert local_probability["n_b"] == 5
    assert local_probability["n_shared"] == 0
    assert local_probability["comparison_valid"] is True
    assert local_probability["mean_diff_b_minus_a"] == pytest.approx(-0.65)
    assert len(detailed) == 20
    assert len(summary) == 2
    assert all(row["dominant_expert_agreement_rate"] == 0.0 for row in summary)


def test_pairwise_statistics_rejects_shared_images_as_independent_samples():
    records = make_routing_records()
    for row in records:
        row["sample_fingerprint"] = f"shared-{row['image_id']}"

    comparisons = pairwise_statistics(records, bootstrap_samples=100, permutations=99, seed=42, alpha=0.05)
    overlaps = sample_overlap_summary(records)
    local_probability = next(
        row for row in comparisons if row["expert"] == "LocalConvTransformer" and row["metric"] == "mean_probability"
    )

    assert local_probability["n_shared"] == 5
    assert local_probability["shared_fraction_min"] == pytest.approx(1.0)
    assert local_probability["comparison_valid"] is False
    assert np.isnan(local_probability["permutation_p_value_two_sided"])
    assert np.isnan(local_probability["fdr_q_value"])
    assert local_probability["significant_after_fdr"] is False
    assert overlaps == [
        {
            "domain_a": "aerial",
            "domain_b": "medical",
            "n_a": 5,
            "n_b": 5,
            "n_shared": 5,
            "shared_fraction_min": 1.0,
            "independent_sample_test_valid": False,
        }
    ]


def test_pairwise_statistics_uses_paired_sequence_clusters():
    records = make_routing_records()
    for row in records:
        row["sample_fingerprint"] = f"{row['domain']}-{row['image_id']}"
        row["cluster_id"] = f"sequence-{row['image_id']}"

    comparisons = pairwise_statistics(
        records,
        bootstrap_samples=200,
        permutations=199,
        seed=42,
        alpha=0.05,
        cluster_aware=True,
    )
    overlaps = sample_overlap_summary(records, cluster_aware=True)
    local_probability = next(
        row for row in comparisons if row["expert"] == "LocalConvTransformer" and row["metric"] == "mean_probability"
    )

    assert local_probability["analysis_unit"] == "sequence_cluster"
    assert local_probability["comparison_design"] == "paired_sequence_clusters"
    assert local_probability["n_paired_clusters"] == 5
    assert local_probability["n_a"] == local_probability["n_b"] == 5
    assert local_probability["mean_diff_b_minus_a"] == pytest.approx(-0.65)
    assert local_probability["bootstrap_ci95_high"] < 0
    assert overlaps[0]["n_shared"] == 0
    assert overlaps[0]["n_shared_clusters"] == 5
    assert overlaps[0]["paired_cluster_test_available"] is True


def test_jensen_shannon_divergence_bounds_and_identity():
    distribution = np.array([0.2, 0.3, 0.5])

    assert jensen_shannon_divergence(distribution, distribution) == pytest.approx(0.0)
    assert jensen_shannon_divergence(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)


def test_mot_p5_budget_config_builds_with_one_router():
    model = build_model(SPECS["v10_mot_p5"], device="cpu")

    assert sum(isinstance(module, MoTBlock) for module in model.modules()) == 1


def test_benchmark_input_is_reproducible_and_grad_mode_does_not_leak(monkeypatch: pytest.MonkeyPatch):
    model = torch.nn.Conv2d(3, 4, 1).eval()
    first = deterministic_benchmark_input(model, imgsz=8, seed=17)
    second = deterministic_benchmark_input(model, imgsz=8, seed=17)
    third = deterministic_benchmark_input(model, imgsz=8, seed=18)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)

    monkeypatch.setattr(mot_ablation, "build_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(mot_ablation, "profile_flops", lambda *_args, **_kwargs: (1.0, "test"))
    previous_grad_mode = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(True)
        row = benchmark_row(
            SPECS["v10"],
            "cpu",
            imgsz=8,
            warmup=0,
            reps=1,
            input_seed=17,
            min_warmup_seconds=0.0,
        )
        assert torch.is_grad_enabled()
    finally:
        torch.set_grad_enabled(previous_grad_mode)
    assert row["input_seed"] == "17"
    assert row["input_distribution"] == "standard_normal"


def test_benchmark_rounds_report_median_and_dispersion():
    rows = []
    for round_index, p50 in enumerate((12.0, 10.0, 11.0), start=1):
        rows.append(
            {
                "key": "v10",
                "benchmark_round": str(round_index),
                "warmup_iterations": str(50 + round_index),
                **{
                    key: str(value)
                    for key, value in {
                        "latency_ms_mean": p50 + 0.1,
                        "latency_ms_p50": p50,
                        "latency_ms_p95": p50 + 1.0,
                        "latency_ms_p99": p50 + 2.0,
                        "latency_ms_min": p50 - 1.0,
                        "latency_ms_max": p50 + 3.0,
                    }.items()
                },
            }
        )

    aggregate = aggregate_benchmark_rounds(rows, [SPECS["v10"]])[0]

    assert aggregate["latency_ms_p50"] == "11.000"
    assert aggregate["latency_ms_p50_run_min"] == "10.000"
    assert aggregate["latency_ms_p50_run_max"] == "12.000"
    assert aggregate["benchmark_rounds"] == "3"


def test_mot_sparse_eval_handles_autocast_dtype_transition():
    block = MoTBlock(dim=16, num_heads=4, top_k=2).eval()
    inputs = torch.randn(2, 16, 8, 8)

    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        output, auxiliary_loss = block(inputs)

    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()
    assert auxiliary_loss.item() == 0.0


def test_stability_summary_includes_transient_recovery_events(tmp_path: Path):
    results = tmp_path / "results.csv"
    events = tmp_path / "recovery_events.jsonl"
    results.write_text(
        "epoch,train/box_loss,train/cls_loss,train/dfl_loss\n0,1.0,2.0,3.0\n",
        encoding="utf-8",
    )
    events.write_text(
        '{"epoch": 0, "reason": "Gradient NaN/Inf", "amp_fallback_triggered": true}\n',
        encoding="utf-8",
    )

    summary = stability_from_results(results, events)

    assert summary["nan_detected"] == "False"
    assert summary["nonfinite_recovery_detected"] == "True"
    assert summary["recovery_events"] == "1"
    assert summary["recovery_reasons"] == "Gradient NaN/Inf"
    assert summary["amp_fallback_triggered"] == "True"


def test_best_observed_metrics_are_distinct_from_the_final_row(tmp_path: Path):
    results = tmp_path / "results.csv"
    results.write_text(
        "epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "1,0.20,0.10\n"
        "2,0.25,0.15\n"
        "3,0.23,0.12\n",
        encoding="utf-8",
    )

    best = read_best_observed_metrics(results)

    assert best["epoch"] == "2"
    assert best["metrics/mAP50-95(B)"] == "0.15"


def test_original_visdrone_occlusion_pairs_are_sequence_matched():
    parsed = parse_visdrone_occlusion(
        "0,0,10,10,1,1,0,0\n"
        "0,0,10,10,1,2,0,1\n"
        "0,0,10,10,1,10,0,2\n"
        "0,0,10,10,0,1,0,2\n"
        "0,0,10,10,1,11,0,2\n"
    )
    assert parsed is not None
    assert parsed.valid_objects == 3
    assert parsed.occluded_fraction == pytest.approx(2 / 3)
    assert parsed.heavy_occluded_fraction == pytest.approx(1 / 3)

    stats = []
    occlusion = {}
    for sequence_index in range(4):
        sequence = f"{sequence_index:07d}"
        for frame, fraction, objects, area in (
            ("00100", 0.1, 20 + sequence_index, 0.0010),
            ("00200", 0.9, 21 + sequence_index, 0.0011),
        ):
            image = Path(f"{sequence}_{frame}_d_0000001.jpg")
            stats.append(
                ImageStats(
                    image=image,
                    label=image.with_suffix(".txt"),
                    objects=objects,
                    mean_area=area,
                    median_area=area,
                    area_cv=0.1,
                    aspect_cv=0.1,
                )
            )
            occlusion[image.stem] = OcclusionStats(
                valid_objects=objects,
                occluded_fraction=fraction,
                heavy_occluded_fraction=max(fraction - 0.5, 0.0),
                mean_occlusion_level=fraction,
            )

    pairs, metadata = match_paired_occlusion_scenes(
        stats,
        occlusion,
        limit=8,
        q_low=0.25,
        q_high=0.75,
    )

    assert len(pairs) == 4
    assert all(pair.sequence_id == pair.lower.image.stem.split("_", 1)[0] for pair in pairs)
    assert all(pair.sequence_id == pair.higher.image.stem.split("_", 1)[0] for pair in pairs)
    assert all(pair.lower.image != pair.higher.image for pair in pairs)
    assert all(pair.lower_occlusion.occluded_fraction < pair.higher_occlusion.occluded_fraction for pair in pairs)
    assert metadata["paired_sequences"] == 4
