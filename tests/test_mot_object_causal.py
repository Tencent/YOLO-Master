from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.analyze_mot_object_causal import (
    LetterboxGeometry,
    ObjectPair,
    VisDroneObject,
    feature_roi,
    frame_position,
    match_occlusion_objects,
    object_pair_statistics,
    parse_visdrone_objects,
    plot_object_delta_heatmap,
    roi_routing_metrics,
)


def _object(
    object_id: str,
    image_id: str,
    *,
    category: int = 3,
    area: float = 0.01,
    occlusion: int = 0,
    truncation: int = 0,
) -> VisDroneObject:
    return VisDroneObject(
        object_id=object_id,
        image_id=image_id,
        sequence_id=image_id.split("_", 1)[0],
        frame_position=int(image_id.split("_")[1]),
        category_id=category,
        x=10.0,
        y=20.0,
        width=20.0,
        height=10.0,
        normalized_area=area,
        truncation=truncation,
        occlusion=occlusion,
    )


def test_parse_visdrone_objects_filters_ignored_regions_and_preserves_metadata():
    text = "10,20,30,40,1,4,1,2\n0,0,50,50,0,0,0,0\n3,4,5,6,1,11,0,1"

    objects = parse_visdrone_objects(text, "0000001_00200_d_0000003.jpg", 100, 200)

    assert len(objects) == 1
    assert objects[0].category_id == 3
    assert objects[0].truncation == 1
    assert objects[0].occlusion == 2
    assert objects[0].normalized_area == pytest.approx(0.06)
    assert objects[0].sequence_id == "0000001"


def test_parse_visdrone_objects_clips_boxes_and_uses_terminal_frame_id():
    objects = parse_visdrone_objects(
        "-5,-4,10,12,1,1,0,0",
        "0000001_00200_d_0000017.jpg",
        100,
        80,
    )

    assert len(objects) == 1
    assert objects[0].x == 0
    assert objects[0].y == 0
    assert objects[0].width == 5
    assert objects[0].height == 8
    assert frame_position(objects[0].image_id) == 17


def test_feature_roi_projects_letterboxed_box_and_keeps_small_object_visible():
    geometry = LetterboxGeometry(
        original_width=200,
        original_height=100,
        input_size=640,
        scale=3.2,
        left=0,
        top=160,
        resized_width=640,
        resized_height=320,
    )
    annotation = _object("a", "0000001_00100_x.jpg")

    roi = feature_roi(annotation, geometry, feature_height=20, feature_width=20)

    assert roi == (1, 7, 3, 8)
    tiny = VisDroneObject(**{**annotation.__dict__, "object_id": "tiny", "width": 0.01, "height": 0.01})
    x0, y0, x1, y1 = feature_roi(tiny, geometry, feature_height=20, feature_width=20)
    assert x1 > x0 and y1 > y0


def test_roi_routing_metrics_separates_inside_and_ring():
    probabilities = torch.full((3, 5, 5), 0.2)
    probabilities[0, 2:4, 2:4] = 0.8
    probabilities[1, 2:4, 2:4] = 0.1
    probabilities[2, 2:4, 2:4] = 0.1
    probabilities /= probabilities.sum(dim=0, keepdim=True)

    metrics = roi_routing_metrics(probabilities, (2, 2, 4, 4))

    assert metrics[0]["feature_cells"] == 4
    assert metrics[0]["inside_probability"] > metrics[0]["ring_probability"]
    assert metrics[0]["inside_top1_share"] == pytest.approx(1.0)
    assert metrics[1]["inside_top1_share"] == pytest.approx(0.0)


def test_occlusion_matching_prioritizes_same_image_then_area_without_reuse():
    objects = [
        _object("low_same", "0000001_00100_x.jpg", area=0.011),
        _object("low_other", "0000001_00200_x.jpg", area=0.010),
        _object("high", "0000001_00100_x.jpg", area=0.010, occlusion=2),
        _object("wrong_class", "0000001_00100_x.jpg", category=4, area=0.010, occlusion=2),
    ]

    pairs = match_occlusion_objects(objects, min_high_occlusion=1, max_pairs=0)

    matched = next(pair for pair in pairs if pair.high_object_id == "high")
    assert matched.low_object_id == "low_same"
    assert matched.same_image is True
    assert len({pair.low_object_id for pair in pairs}) == len(pairs)
    assert len({pair.high_object_id for pair in pairs}) == len(pairs)


def test_occlusion_matching_enforces_truncation_and_area_caliper():
    objects = [
        _object("low_valid", "0000001_00001_d_0000001.jpg", area=0.010, truncation=0),
        _object("low_truncated", "0000001_00001_d_0000001.jpg", area=0.010, truncation=1),
        _object("high_valid", "0000001_00001_d_0000002.jpg", area=0.012, occlusion=2, truncation=0),
        _object("high_too_large", "0000001_00001_d_0000003.jpg", area=0.100, occlusion=2, truncation=0),
    ]

    pairs = match_occlusion_objects(
        objects,
        min_high_occlusion=1,
        max_pairs=0,
        max_log_area_distance=0.5,
    )

    assert [(pair.low_object_id, pair.high_object_id) for pair in pairs] == [("low_valid", "high_valid")]


def test_object_statistics_use_sequence_aggregates_and_compound_significance():
    records = []
    pairs = []
    for sequence_index in range(6):
        sequence = f"{sequence_index:07d}"
        low_id, high_id = f"{sequence}:low", f"{sequence}:high"
        pairs.append(
            ObjectPair(
                pair_id=sequence_index,
                sequence_id=sequence,
                category_id=0,
                low_object_id=low_id,
                high_object_id=high_id,
                same_image=True,
                log_area_distance=0.0,
                frame_distance=0,
            )
        )
        for object_id, probability, occlusion in ((low_id, 0.2, 0), (high_id, 0.7, 2)):
            for expert in ("LocalConvTransformer",):
                records.append(
                    {
                        "object_id": object_id,
                        "occlusion": occlusion,
                        "layer": "model.1",
                        "expert": expert,
                        "inside_probability": probability,
                        "inside_top1_share": probability,
                        "inside_minus_ring_probability": probability - 0.1,
                    }
                )

    statistics = object_pair_statistics(records, pairs, bootstrap_samples=500, permutations=500, seed=7)

    assert len(statistics) == 3
    probability = next(row for row in statistics if row["metric"] == "inside_probability")
    assert probability["analysis_unit"] == "video_sequence"
    assert probability["n_sequences"] == 6
    assert probability["mean_diff_high_minus_low"] == pytest.approx(0.5)
    assert probability["bootstrap_ci95_low"] > 0
    assert np.isfinite(probability["fdr_q_value"])
    assert probability["significant_after_fdr"] is True


def test_object_delta_heatmap_writes_png(tmp_path):
    output = tmp_path / "delta.png"
    plot_object_delta_heatmap(
        [
            {
                "layer": "model.1",
                "expert": "LocalConvTransformer",
                "metric": "inside_probability",
                "mean_diff_high_minus_low": 0.1,
            }
        ],
        output,
    )

    assert output.is_file()
    assert output.stat().st_size > 0
