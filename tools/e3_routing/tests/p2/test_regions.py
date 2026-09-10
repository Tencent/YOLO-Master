from pathlib import Path

import numpy as np
import pytest

from e3_p2.geometry import LetterboxMeta
from e3_p2.regions import (
    YoloBox,
    aggregate_region_diagnostics,
    label_path_for_image,
    parse_yolo_labels,
    region_routing_diagnostics,
    token_region_masks,
)


def _meta() -> LetterboxMeta:
    return LetterboxMeta(
        original_width=100,
        original_height=50,
        input_size=100,
        resized_width=100,
        resized_height=50,
        pad_left=0,
        pad_top=25,
        pad_right=0,
        pad_bottom=25,
        scale=1.0,
    )


def test_label_path_replaces_images_component(tmp_path: Path):
    image = tmp_path / "dataset" / "images" / "val" / "sample.jpg"
    assert label_path_for_image(image) == tmp_path / "dataset" / "labels" / "val" / "sample.txt"


def test_strict_yolo_label_parser(tmp_path: Path):
    labels = tmp_path / "sample.txt"
    labels.write_text("2 0.5 0.5 0.4 0.2\n", encoding="utf-8")
    assert parse_yolo_labels(labels) == [YoloBox(2, 0.5, 0.5, 0.4, 0.2)]
    labels.write_text("2 1.2 0.5 0.4 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="center"):
        parse_yolo_labels(labels)


def test_token_masks_exclude_letterbox_padding_and_use_cell_centers():
    masks = token_region_masks([YoloBox(0, 0.5, 0.5, 0.5, 1.0)], _meta(), 4, 4)
    assert masks["padding"].tolist() == [
        [True, True, True, True],
        [False, False, False, False],
        [False, False, False, False],
        [True, True, True, True],
    ]
    assert masks["foreground"].sum() == 4
    assert masks["background"].sum() == 4
    assert np.all(masks["foreground"][:, 1:3] == ~masks["padding"][:, 1:3])


def test_region_diagnostics_has_known_contrast():
    weights = np.asarray(
        [
            [[0.9, 0.8], [0.2, 0.1]],
            [[0.1, 0.2], [0.8, 0.9]],
        ],
        dtype=np.float32,
    )
    foreground = np.asarray([[True, True], [False, False]])
    background = ~foreground
    diagnostics = region_routing_diagnostics(
        weights,
        {"foreground": foreground, "background": background, "padding": np.zeros((2, 2), dtype=bool)},
    )
    assert diagnostics["status"] == "SUPPORTED"
    assert diagnostics["foreground"]["mean_expert_probability"] == pytest.approx([0.85, 0.15])
    assert diagnostics["background"]["mean_expert_probability"] == pytest.approx([0.15, 0.85])
    assert diagnostics["contrast"]["total_variation_distance"] == pytest.approx(0.7)


def test_region_diagnostics_marks_empty_group_without_inventing_values():
    weights = np.full((2, 2, 2), 0.5, dtype=np.float32)
    diagnostics = region_routing_diagnostics(
        weights,
        {
            "foreground": np.zeros((2, 2), dtype=bool),
            "background": np.ones((2, 2), dtype=bool),
            "padding": np.zeros((2, 2), dtype=bool),
        },
    )
    assert diagnostics["status"] == "INSUFFICIENT_TOKENS"
    assert diagnostics["foreground"]["mean_expert_probability"] is None
    assert diagnostics["contrast"] is None


def test_aggregate_reports_pooled_and_equal_weight_paired_contrasts():
    first = region_routing_diagnostics(
        np.asarray([[[0.9, 0.2], [0.9, 0.2]], [[0.1, 0.8], [0.1, 0.8]]], dtype=np.float32),
        {
            "foreground": np.asarray([[True, False], [True, False]]),
            "background": np.asarray([[False, True], [False, True]]),
            "padding": np.asarray([[False, False], [False, False]]),
        },
    )
    second = region_routing_diagnostics(
        np.asarray([[[0.6, 0.4], [0.6, 0.4]], [[0.4, 0.6], [0.4, 0.6]]], dtype=np.float32),
        {
            "foreground": np.asarray([[True, False], [True, False]]),
            "background": np.asarray([[False, True], [False, True]]),
            "padding": np.asarray([[False, False], [False, False]]),
        },
    )
    aggregate = aggregate_region_diagnostics(
        [
            {"family": "mot", "module": "router", "region_diagnostics": first},
            {"family": "mot", "module": "router", "region_diagnostics": second},
        ]
    )["by_family"]["mot"]
    assert aggregate["pooled_contrast"]["total_variation_distance"] == pytest.approx(0.45)
    assert aggregate["paired_capture_contrast"]["capture_count"] == 2
    assert aggregate["paired_capture_contrast"]["total_variation_distance"]["mean"] == pytest.approx(0.45)
