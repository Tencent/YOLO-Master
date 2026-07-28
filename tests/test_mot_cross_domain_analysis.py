"""Tests for the same-checkpoint MoT cross-domain routing audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

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
    robustness_statistics,
)
from scripts.compare_mot_ablation import SPECS, build_model, stability_from_results
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
    assert local_probability["mean_diff_b_minus_a"] == pytest.approx(-0.65)
    assert len(detailed) == 20
    assert len(summary) == 2
    assert all(row["dominant_expert_agreement_rate"] == 0.0 for row in summary)


def test_jensen_shannon_divergence_bounds_and_identity():
    distribution = np.array([0.2, 0.3, 0.5])

    assert jensen_shannon_divergence(distribution, distribution) == pytest.approx(0.0)
    assert jensen_shannon_divergence(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)


def test_mot_p5_budget_config_builds_with_one_router():
    model = build_model(SPECS["v10_mot_p5"], device="cpu")

    assert sum(isinstance(module, MoTBlock) for module in model.modules()) == 1


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
