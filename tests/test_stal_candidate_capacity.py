"""Verify the opt-in geometric capacity filter for expanded small-target candidates."""

import pytest
import torch

from ultralytics.cfg import check_cfg, get_cfg
from ultralytics.utils.tal import TaskAlignedAssigner


def test_capacity_keeps_inside_and_feasible_outside_candidates():
    """Keep the 0.75 boundary but reject the 0.60-capacity point that fixed STAL admitted."""
    points = torch.tensor([[3.0, 3.0], [8.0, 3.0], [10.0, 3.0]])
    boxes = torch.tensor([[[0.0, 0.0, 6.0, 6.0]]])
    valid = torch.ones(1, 1, 1, dtype=torch.bool)
    before = boxes.clone()
    fixed = TaskAlignedAssigner(stal_candidate_mode="fixed")
    capped = TaskAlignedAssigner(stal_candidate_mode="fixed", stal_candidate_iou_floor=0.75)
    assert fixed.select_candidates_in_gts(points, boxes, valid).tolist() == [[[True, True, True]]]
    assert capped.select_candidates_in_gts(points, boxes, valid, image_size=(100, 100)).tolist() == [
        [[True, True, False]]
    ]
    assert torch.equal(boxes, before)


def test_capacity_preserves_non_small_and_pure_candidates():
    """Non-small GTs are unaffected; pure TAL ignores the expansion-only filter."""
    points = torch.tensor([[3.0, 3.0], [8.0, 3.0], [10.0, 3.0]])
    boxes = torch.tensor([[[0.0, 0.0, 6.0, 200.0]]])
    valid = torch.ones(1, 1, 1, dtype=torch.bool)
    for mode in ("pure", "fixed", "adaptive"):
        original = TaskAlignedAssigner(stal_candidate_mode=mode)
        filtered = TaskAlignedAssigner(stal_candidate_mode=mode, stal_candidate_iou_floor=0.75)
        a = original.select_candidates_in_gts(points, boxes, valid, image_size=(100, 100))
        b = filtered.select_candidates_in_gts(points, boxes, valid, image_size=(100, 100))
        assert torch.equal(a, b)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_capacity_rejects_invalid_floor(value):
    """Reject invalid thresholds both at configuration and assigner boundaries."""
    with pytest.raises(ValueError):
        TaskAlignedAssigner(stal_candidate_iou_floor=value)
    with pytest.raises(ValueError):
        check_cfg({"stal_candidate_iou_floor": value})


def test_capacity_default_and_type_contract():
    """Default is disabled and string-like programmatic thresholds are rejected."""
    assert get_cfg().stal_candidate_iou_floor == 0.0
    with pytest.raises(TypeError):
        check_cfg({"stal_candidate_iou_floor": "0.75"})


def test_capacity_requires_image_size_when_enabled():
    """A missing scale must fail rather than silently apply the small-target gate incorrectly."""
    capped = TaskAlignedAssigner(stal_candidate_mode="fixed", stal_candidate_iou_floor=0.75)
    with pytest.raises(ValueError, match="image_size"):
        capped.select_candidates_in_gts(
            torch.tensor([[3.0, 3.0]]), torch.tensor([[[0.0, 0.0, 6.0, 6.0]]]), torch.ones(1, 1, 1)
        )


@pytest.mark.parametrize(
    "bbox,point,relaxation,floor",
    [
        ([100.0, 100.0, 120.0, 120.0], [500.0, 500.0], 800.0, 0.002),
        ([0.0, 0.0, 0.0001, 0.0001], [0.00005, 0.00005], 0.0, 0.75),
    ],
)
def test_capacity_half_precision_matches_fp32(bbox, point, relaxation, floor):
    """Neither overflowing enclosing areas nor underflowing tiny GT areas may reject feasible candidates."""
    assigner = TaskAlignedAssigner(
        stal_candidate_mode="adaptive",
        stal_relaxation=relaxation,
        stal_warmup_epochs=0,
        stal_candidate_iou_floor=floor,
    )
    masks = []
    for dtype in (torch.float32, torch.float16):
        masks.append(
            assigner.select_candidates_in_gts(
                torch.tensor([point], dtype=dtype),
                torch.tensor([[bbox]], dtype=dtype),
                torch.ones(1, 1, 1, dtype=torch.bool),
                image_size=(800, 800),
            )
        )
    assert masks[0].item()
    assert torch.equal(masks[0], masks[1])
