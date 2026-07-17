"""Statistical regression tests for real-image MoT routing analysis."""

import numpy as np
import pytest

from scripts.analyze_mot_routing import (
    aggregate,
    aggregate_marginal,
    derive_scene_thresholds,
    letterbox_image,
    scene_contrasts,
    scene_tags,
)


def synthetic_rows():
    rows = []
    for index in range(40):
        high = index < 20
        for module_index, module in enumerate(("model.5.m.0", "model.8.m.0")):
            deform = (0.8 if high else 0.2) + module_index * 0.01
            rows.append(
                {
                    "image": f"image_{index}.jpg",
                    "module": module,
                    "density": "dense" if high else "sparse",
                    "scale": "small" if high else "large",
                    "shape": "irregular" if high else "regular",
                    "occlusion": "occluded" if high else "clear",
                    "LocalConvTransformer_mean_weight": 1.0 - deform,
                    "LocalConvTransformer_top1_token_frac": 1.0 - deform,
                    "WindowTransformer_mean_weight": 0.0,
                    "WindowTransformer_top1_token_frac": 0.0,
                    "DeformableTransformer_mean_weight": deform,
                    "DeformableTransformer_top1_token_frac": deform,
                }
            )
    return rows


def test_scene_aggregation_preserves_layer_identity_and_image_counts():
    rows = synthetic_rows()
    joint = aggregate(rows)
    marginal = aggregate_marginal(rows)

    assert {row["module"] for row in joint} == {"model.5.m.0", "model.8.m.0"}
    global_occlusion = [
        row for row in marginal if row["scope"] == "global" and row["factor"] == "occlusion"
    ]
    assert {row["images"] for row in global_occlusion} == {20}


def test_letterbox_preserves_aspect_ratio_and_centers_padding():
    image = np.full((100, 200, 3), 255, dtype=np.uint8)

    output = letterbox_image(image, 640)

    assert output.shape == (640, 640, 3)
    assert np.all(output[:160] == 114)
    assert np.all(output[160:480] == 255)
    assert np.all(output[480:] == 114)


def test_scene_quantiles_create_reproducible_balanced_thresholds():
    labels = []
    for count, area in ((1, 0.001), (2, 0.004), (10, 0.02), (20, 0.08)):
        rows = np.zeros((count, 5), dtype=np.float32)
        rows[:, 3] = np.sqrt(area)
        rows[:, 4] = np.sqrt(area)
        labels.append(rows)

    thresholds = derive_scene_thresholds(labels, 0.25, 0.75)

    assert thresholds["sparse_threshold"] == pytest.approx(1.75)
    assert thresholds["dense_threshold"] == pytest.approx(12.5)
    assert scene_tags(labels[0], **thresholds)["density"] == "sparse"
    assert scene_tags(labels[-1], **thresholds)["density"] == "dense"
    assert scene_tags(labels[0], **thresholds)["scale"] == "small"
    assert scene_tags(labels[-1], **thresholds)["scale"] == "large"


def test_scene_contrasts_report_effect_uncertainty_permutation_and_fdr():
    contrasts = scene_contrasts(synthetic_rows(), n_resamples=499, seed=7)
    result = next(
        row
        for row in contrasts
        if row["scope"] == "global"
        and row["factor"] == "occlusion"
        and row["metric"] == "DeformableTransformer_mean_weight"
    )

    assert result["n_a"] == result["n_b"] == 20
    assert result["mean_difference"] > 0.5
    assert result["ci95_low"] > 0
    assert result["hedges_g"] > 5
    assert result["permutation_p"] <= 0.01
    assert result["fdr_q"] <= 0.05
