"""Regression tests for task-aligned assignment on Apple MPS."""

import pytest
import torch

from ultralytics.utils.tal import TaskAlignedAssigner


def test_task_aligned_assigner_box_metrics_match_expanded_reference():
    """Explicit assignment indices must preserve the original expanded-view calculation."""
    batch_size, max_boxes, anchors, classes = 2, 5, 19, 7
    generator = torch.Generator().manual_seed(0)
    pd_scores = torch.rand(batch_size, anchors, classes, generator=generator)
    pd_bboxes = torch.rand(batch_size, anchors, 4, generator=generator)
    gt_labels = torch.randint(classes, (batch_size, max_boxes, 1), generator=generator)
    gt_bboxes = torch.rand(batch_size, max_boxes, 4, generator=generator)
    mask_gt = torch.rand(batch_size, max_boxes, anchors, generator=generator) > 0.4

    assigner = TaskAlignedAssigner(num_classes=classes)
    assigner.bs, assigner.n_max_boxes = batch_size, max_boxes
    align_metric, overlaps = assigner.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)

    reference_scores = torch.zeros(batch_size, max_boxes, anchors)
    reference_overlaps = torch.zeros_like(reference_scores)
    batch_idx = torch.arange(batch_size).view(-1, 1).expand(-1, max_boxes)
    class_idx = gt_labels.squeeze(-1)
    reference_scores[mask_gt] = pd_scores[batch_idx, :, class_idx][mask_gt]
    reference_pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, max_boxes, -1, -1)[mask_gt]
    reference_gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, anchors, -1)[mask_gt]
    reference_overlaps[mask_gt] = assigner.iou_calculation(reference_gt_boxes, reference_pd_boxes)
    reference_align_metric = reference_scores.pow(assigner.alpha) * reference_overlaps.pow(assigner.beta)

    torch.testing.assert_close(overlaps, reference_overlaps)
    torch.testing.assert_close(align_metric, reference_align_metric)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_task_aligned_assigner_mps_large_mask_keeps_index_alignment():
    """The MPS path must keep predicted and GT boxes aligned for large expanded masks."""
    device = torch.device("mps")
    batch_size, max_boxes, anchors, classes = 8, 184, 8400, 80
    generator = torch.Generator(device="cpu").manual_seed(0)

    pd_scores = torch.rand(batch_size, anchors, classes, generator=generator).to(device)
    pd_bboxes = torch.rand(batch_size, anchors, 4, generator=generator).to(device)
    anc_points = torch.rand(anchors, 2, generator=generator).to(device)
    gt_labels = torch.randint(classes, (batch_size, max_boxes, 1), generator=generator).to(device)
    gt_bboxes = torch.rand(batch_size, max_boxes, 4, generator=generator).to(device)
    gt_bboxes[..., 2:] += gt_bboxes[..., :2]
    mask_gt = torch.rand(batch_size, max_boxes, 1, generator=generator).to(device).bool()

    assigner = TaskAlignedAssigner(topk=10, num_classes=classes)
    result = assigner(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)

    assert result[0].shape == (batch_size, anchors)
    assert result[1].shape == (batch_size, anchors, 4)
    assert result[2].shape == (batch_size, anchors, classes)
    assert result[3].shape == (batch_size, anchors)
    assert result[4].shape == (batch_size, anchors)
