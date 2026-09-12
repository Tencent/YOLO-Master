"""Verify the opt-in SimD-inspired small-target TAL matching adaptation."""

import math

import pytest
import torch

from ultralytics.cfg import check_cfg, get_cfg
from ultralytics.utils.tal import TaskAlignedAssigner


def test_simd_matches_official_visdrone_formula_constants():
    """Match the official implementation's x=6.13, y=4.59 location-and-shape formula."""
    gt = torch.tensor([[0.0, 0.0, 4.0, 2.0]])
    pred = torch.tensor([[1.0, 0.0, 7.0, 4.0]])
    gt_xywh = torch.tensor([2.0, 1.0, 4.0, 2.0])
    pred_xywh = torch.tensor([4.0, 2.0, 6.0, 4.0])
    scale = (gt_xywh[2:] + pred_xywh[2:]) / torch.tensor([6.13, 4.59])
    location = (((gt_xywh[:2] - pred_xywh[:2]) / scale).square().sum()).sqrt()
    shape = (((gt_xywh[2:] - pred_xywh[2:]) / scale).square().sum()).sqrt()
    expected = math.exp(-float(location + shape))
    assert TaskAlignedAssigner.simd_similarity(gt, pred).item() == pytest.approx(expected)


def test_simd_changes_only_small_target_matching_and_keeps_ciou_targets():
    """Use SimD for small-target ranking while decoupling final soft targets to CIoU."""
    assigner = TaskAlignedAssigner(
        num_classes=1,
        topk=2,
        stal_candidate_mode="pure",
        stal_area_threshold=0.01,
        stal_simd_weight=1.0,
        stal_nwd_target_mode="ciou",
    )
    scores = torch.tensor([[[0.5], [0.5]]])
    predicted = torch.tensor([[[0.0, 0.0, 4.0, 4.0], [1.0, 0.0, 5.0, 4.0]]])
    anchors = torch.tensor([[2.0, 2.0], [3.0, 2.0]])
    labels = torch.zeros(1, 1, 1)
    boxes = torch.tensor([[[0.0, 0.0, 4.0, 4.0]]])
    valid = torch.ones(1, 1, 1, dtype=torch.bool)
    assigner.bs = 1
    assigner.n_max_boxes = 1
    mask, _, matching, _, targets = assigner.get_pos_mask(
        scores, predicted, labels, boxes, anchors, valid, image_size=(100, 100)
    )
    assert mask.any()
    assert matching[0, 0, 1] != pytest.approx(targets[0, 0, 1])
    assert targets[0, 0, 0] == pytest.approx(1.0)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_simd_rejects_invalid_weights(value):
    """Reject invalid weights at both public configuration boundaries."""
    with pytest.raises(ValueError):
        TaskAlignedAssigner(stal_simd_weight=value)
    with pytest.raises(ValueError):
        check_cfg({"stal_simd_weight": value})


def test_simd_default_and_type_contract():
    """Default is disabled and programmatic strings cannot bypass type checking."""
    assert get_cfg().stal_simd_weight == 0.0
    with pytest.raises(TypeError):
        check_cfg({"stal_simd_weight": "1.0"})
