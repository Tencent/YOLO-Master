"""Tests for the dense / small / irregular scene contrasts in the MoT routing suite.

The point of these tests is not that the statistics run — the doctests cover that —
but that the covariate control actually works. On VisDrone, crowding, object scale
and occlusion move together, so a scene test that cannot reject a planted confound
would report a "significant" result for whichever axis happened to correlate with
the routing signal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.mot_routing_interpret import (
    VisDroneObject,
    image_scene_features,
    is_irregular_box,
    local_density_masks,
    quantile_contrast_split,
    token_group_masks,
    token_local_density,
    token_object_masks,
)
from scripts.run_mot_routing_interpret import (
    SCENE_CONTRASTS,
    SCENE_TOKEN_PAIRS,
    ImageRecord,
    SceneContrast,
    apply_fdr,
    area_terciles,
    matched_area_band,
    scene_assignments,
    scene_contrast_image_tests,
    scene_token_groups,
    stratified_scene_tests,
    token_level_tests,
)

LAYERS = ("mot.0", "mot.1")


def make_record(stem: str, features: dict[str, float], values: dict[str, list[float]]) -> ImageRecord:
    """Build a synthetic :class:`ImageRecord` with the given per-layer expert weights."""
    defaults = {
        "n_objects": 10.0,
        "median_area": 0.001,
        "mean_area": 0.001,
        "area_cv": 0.1,
        "aspect_cv": 0.1,
        "occlusion_rate": 0.2,
        "heavy_occlusion_rate": 0.1,
        "truncation_rate": 0.1,
        "irregular_rate": 0.0,
    }
    return ImageRecord(
        stem=stem,
        image_path=Path(stem),
        features={**defaults, **features},
        layer_expert_weight={layer: np.asarray(value, dtype=np.float64) for layer, value in values.items()},
        layer_expert_top1={layer: np.asarray(value, dtype=np.float64) for layer, value in values.items()},
        token_group_weight={},
        token_group_sizes={},
    )


def confounded_records(*, value_follows: str, seed: int = 0) -> list[ImageRecord]:
    """Images where crowding and object scale co-vary, as they do in VisDrone.

    ``value_follows`` picks which of the two drives the routing weight, so the same
    generator produces both a confound-only sample and a real-effect sample.
    """
    rng = np.random.default_rng(seed)
    records: list[ImageRecord] = []
    for index in range(180):
        area_level = index % 3
        median_area = 0.0005 * (area_level + 1)
        # Small objects sit in crowded frames, but with enough noise that every area
        # stratum still contains both dense and sparse images.
        n_objects = 45.0 - 10.0 * area_level + rng.normal(0.0, 10.0)
        driver = n_objects / 45.0 if value_follows == "n_objects" else (2 - area_level) / 2.0
        weight = 0.2 + 0.3 * driver + rng.normal(0.0, 0.005)
        records.append(
            make_record(
                f"img{index}",
                {"n_objects": n_objects, "median_area": median_area},
                {layer: [weight, 1.0 - weight, 0.0] for layer in LAYERS},
            )
        )
    return records


def dense_contrast_report(records: list[ImageRecord]) -> dict:
    """Run the dense-vs-sparse contrast and return its pooled-layer row."""
    rows = stratified_scene_tests(
        records,
        LAYERS,
        SCENE_CONTRASTS[0],
        n_permutations=400,
        n_strata=3,
        seed=0,
        alternative="two-sided",
    )
    return next(row for row in rows if row["expert"] == "LocalConvTransformer" and row["layer"] == "__pooled__")


def test_stratification_removes_a_pure_object_scale_confound():
    report = dense_contrast_report(confounded_records(value_follows="median_area"))

    assert report["raw_difference"] > 0.1, "the confound must be visible before stratification"
    assert abs(report["stratified_difference"]) < 0.02
    assert report["p_value"] > 0.05


def test_stratification_keeps_a_real_density_effect():
    report = dense_contrast_report(confounded_records(value_follows="n_objects"))

    assert report["stratified_difference"] > 0.1
    assert report["p_value"] < 0.01


def test_scene_contrast_family_covers_every_contrast_and_excludes_pooled_rows():
    rows = apply_fdr(
        scene_contrast_image_tests(
            confounded_records(value_follows="n_objects"),
            LAYERS,
            n_permutations=100,
            n_strata=3,
            seed=0,
        )
    )

    assert {row["contrast"] for row in rows} == {contrast.name for contrast in SCENE_CONTRASTS}
    assert len(rows) == len(SCENE_CONTRASTS) * 3 * (len(LAYERS) + 1)
    assert all(row["alternative"] == "two-sided" for row in rows)
    assert all(not row["in_fdr_family"] for row in rows if row["layer"] == "__pooled__")
    assert all(np.isnan(row["q_value_bh"]) for row in rows if row["layer"] == "__pooled__")


def test_irregular_contrast_survives_a_tie_heavy_feature():
    """Most VisDrone frames have ``irregular_rate == 0``; the split must not collapse."""
    rng = np.random.default_rng(1)
    records = [
        make_record(
            f"img{index}",
            {"irregular_rate": 0.0 if index % 4 else 0.4},
            {layer: [0.3 + rng.normal(0.0, 0.01), 0.4, 0.3] for layer in LAYERS},
        )
        for index in range(80)
    ]
    contrast = SceneContrast("irregular_vs_regular", "irregular_rate", True, ("n_objects", "median_area"))

    rows = stratified_scene_tests(
        records, LAYERS, contrast, n_permutations=100, n_strata=2, seed=0, alternative="two-sided"
    )

    assert all(row["n_group"] == 20 and row["n_baseline"] == 60 for row in rows)
    assert all(row["n_strata"] >= 1 for row in rows)


def test_quantile_contrast_split_arms_are_disjoint_when_ties_dominate():
    high, low = quantile_contrast_split([0.0] * 90 + [0.25] * 10)

    assert not (high & low).any()
    assert high.sum() == 10 and low.sum() == 90


def test_scene_assignments_exposes_both_shape_groups():
    records = [make_record(f"img{index}", {"irregular_rate": 0.5 * (index % 2)}, {}) for index in range(20)]

    scenes = scene_assignments(records)

    assert scenes["irregular_objects"].sum() == 10
    assert scenes["regular_objects"].sum() == 10
    assert not (scenes["irregular_objects"] & scenes["regular_objects"]).any()


def test_irregular_rate_uses_pixel_aspect_ratio():
    wide = VisDroneObject(0, 0, 40, 10, 1, 0, 0)
    tall = VisDroneObject(0, 0, 10, 40, 1, 0, 0)
    compact = VisDroneObject(0, 0, 20, 15, 1, 0, 0)

    assert is_irregular_box(wide) and is_irregular_box(tall) and not is_irregular_box(compact)
    # A 4:1 image would turn the compact box into a 1:3 one under normalized coordinates.
    assert image_scene_features([compact], 400, 100).irregular_rate == 0.0
    assert image_scene_features([wide, compact], 400, 100).irregular_rate == 0.5


def test_middle_tercile_boxes_break_solo_status_without_joining_either_arm():
    small = VisDroneObject(0, 0, 4, 4, 1, 0, 0)
    medium = VisDroneObject(0, 0, 30, 30, 1, 0, 0)
    large = VisDroneObject(60, 0, 40, 100, 1, 0, 0)
    groups = scene_token_groups(
        [small, medium, large],
        100,
        100,
        1,
        2,
        size_split=(0.01, 0.1),
        shape_band=(0.0, 1.0),
        density_radius=1,
        box_count=token_object_masks([small, medium, large], 100, 100, 1, 2)["box_count"],
    )

    # The small box shares its token with the medium one, so it is no longer solo,
    # while the large box on the right-hand token still is.
    assert groups["small_solo"].tolist() == [[False, False]]
    assert groups["large_solo"].tolist() == [[False, True]]


def test_matched_area_band_narrows_to_the_shared_size_range():
    irregular = [VisDroneObject(0, 0, 3 * side, side, 1, 0, 0) for side in (4, 6, 8, 10, 12)]
    regular = [VisDroneObject(0, 0, side, side, 1, 0, 0) for side in (7, 10, 14, 17, 21)]

    band = matched_area_band([(irregular + regular, 100, 100)], is_irregular_box)

    assert band == pytest.approx((0.0108, 0.0289))
    # Strictly inside both arms' ranges, so the extremes of each are excluded.
    assert min(box.width * box.height / 10000 for box in irregular + regular) < band[0] < band[1]
    assert band[1] < max(box.width * box.height / 10000 for box in irregular + regular)


def test_out_of_band_boxes_leave_the_shape_arms_but_still_occupy_tokens():
    in_band_wide = VisDroneObject(0, 0, 24, 8, 1, 0, 0)
    in_band_square = VisDroneObject(50, 0, 17, 17, 1, 0, 0)
    huge_square = VisDroneObject(50, 0, 90, 90, 1, 0, 0)
    band = (0.0108, 0.0289)

    masks = token_group_masks(
        [in_band_wide, in_band_square, huge_square], 100, 100, 1, 2, is_irregular_box, area_range=band
    )

    assert masks["positive_solo"].tolist() == [[True, False]]
    # The right token holds an in-band square plus the out-of-band one, so it is not solo.
    assert masks["negative_solo"].tolist() == [[False, False]]
    assert masks["box_count"].tolist() == [[1, 2]]


def test_local_density_separates_crowded_from_isolated_object_tokens():
    crowd = [VisDroneObject(x, 0, 2, 2, 1, 0, 0) for x in (0, 4, 8, 12)]
    loner = VisDroneObject(90, 0, 2, 2, 1, 0, 0)
    objects = [*crowd, loner]
    density = token_local_density(objects, 100, 10, 1, 10, radius=1)
    object_mask = ~token_object_masks(objects, 100, 10, 1, 10)["background"]

    masks = local_density_masks(density, object_mask)

    assert density[0, 0] > density[0, 9]
    assert masks["crowded"].tolist() == [[True, True, False, False, False, False, False, False, False, False]]
    assert masks["isolated"].tolist() == [[False, False, False, False, False, False, False, False, False, True]]


def test_area_terciles_split_the_pooled_box_distribution():
    objects = [VisDroneObject(0, 0, size, size, 1, 0, 0) for size in range(1, 31)]

    small_max, large_min = area_terciles([(objects, 100, 100)])

    assert small_max < large_min
    assert sum(1 for obj in objects if (obj.width * obj.height) / 10000 <= small_max) == pytest.approx(10, abs=1)


def test_scene_token_pairs_run_two_sided_and_report_their_contrast():
    rng = np.random.default_rng(2)
    groups = {pair.group_a for pair in SCENE_TOKEN_PAIRS} | {pair.group_b for pair in SCENE_TOKEN_PAIRS}
    records = []
    for index in range(30):
        record = make_record(f"img{index}", {}, {layer: [0.3, 0.4, 0.3] for layer in LAYERS})
        record.token_group_weight = {
            layer: {group: np.asarray([0.3, 0.4, 0.3]) + rng.normal(0.0, 0.01, 3) for group in groups}
            for layer in LAYERS
        }
        records.append(record)

    rows = token_level_tests(records, LAYERS, SCENE_TOKEN_PAIRS, alternative="two-sided")

    assert len(rows) == 3 * len(SCENE_TOKEN_PAIRS) * (len(LAYERS) + 1)
    assert {row["contrast"] for row in rows} == {pair.contrast for pair in SCENE_TOKEN_PAIRS}
    assert all(row["alternative"] == "two-sided" for row in rows)
    # Pure noise must not look significant once the family is FDR-corrected.
    assert not any(row["significant_after_fdr"] for row in apply_fdr(rows))
