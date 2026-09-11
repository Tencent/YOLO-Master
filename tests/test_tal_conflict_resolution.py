"""Regression tests for task-aligned assignment conflict resolution."""

import torch

from ultralytics.utils.tal import TaskAlignedAssigner


def test_conflict_resolution_cannot_assign_anchor_to_non_candidate_gt():
    """Zero-overlap ties must stay within the GTs that proposed the contested anchor."""
    assigner = TaskAlignedAssigner(num_classes=3)
    mask_pos = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]]
    )
    overlaps = torch.zeros_like(mask_pos)
    align_metric = torch.zeros_like(mask_pos)
    original_counts = mask_pos.sum(-1)

    _, fg_mask, resolved = assigner.select_highest_overlaps(
        mask_pos, overlaps, n_max_boxes=3, align_metric=align_metric
    )

    assert resolved[0, 0, 2] == 0
    assert (resolved.sum(-1) <= original_counts).all()
    assert (fg_mask <= 1).all()
