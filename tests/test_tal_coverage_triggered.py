"""Regression tests for the C1 coverage-triggered candidate mode."""

import torch

from ultralytics.utils.tal import TaskAlignedAssigner


def _assigner(**kwargs):
    return TaskAlignedAssigner(topk=10, num_classes=1, stride=[8, 16, 32], **kwargs)


def test_coverage_triggered_preserves_core_and_caps_supplements():
    centers = torch.tensor([[4.0, 14.0], [8.0, 14.0], [12.0, 14.0], [16.0, 14.0], [20.0, 14.0], [24.0, 14.0]])
    gt = torch.tensor([[[8.0, 8.0, 20.0, 20.0]]])
    valid = torch.ones((1, 1, 1), dtype=torch.bool)
    assigner = _assigner(
        candidate_expand_coverage_triggered=True,
        candidate_expand_coverage_min=3,
        candidate_expand_coverage_target=20.0,
    )

    mask = assigner.select_candidates_in_gts(centers, gt, valid)

    # The original box contains centers 8 and 12; C1 adds only the nearest ring center.
    assert mask.sum().item() == 3
    assert mask[0, 0, 1].item() and mask[0, 0, 2].item()


def test_coverage_disabled_matches_default_candidate_selection():
    centers = torch.tensor([[4.0, 14.0], [8.0, 14.0], [12.0, 14.0], [16.0, 14.0], [20.0, 14.0]])
    gt = torch.tensor([[[8.0, 8.0, 20.0, 20.0]]])
    valid = torch.ones((1, 1, 1), dtype=torch.bool)
    default_mask = _assigner().select_candidates_in_gts(centers, gt, valid)
    disabled_mask = _assigner(candidate_expand_coverage_triggered=False).select_candidates_in_gts(centers, gt, valid)

    assert torch.equal(default_mask, disabled_mask)
