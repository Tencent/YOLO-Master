"""Tests for small-target adaptive task-aligned assignment."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ultralytics.cfg import get_cfg
from ultralytics.utils.tal import TaskAlignedAssigner


def _grid_points(size=3, stride=8):
    coordinates = torch.arange(size, dtype=torch.float32) * stride + stride / 2
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


def test_stal_default_config_preserves_fixed_stride_mode():
    """The new configuration contract must retain the repository's existing behavior by default."""
    cfg = get_cfg()
    assert cfg.stal_mode == "fixed"
    assert cfg.stal_small_area == 32**2
    assert cfg.stal_medium_area == 96**2
    assert cfg.stal_topk_small >= cfg.stal_min_candidates


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"stal_mode": "unknown"}, "stal_mode"),
        ({"stal_small_area": 1024.0, "stal_medium_area": 1024.0}, "0 < small < medium"),
        ({"stal_candidate_scale": 0.5}, "between 1.0 and 4.0"),
        ({"stal_min_candidates": 4, "stal_topk_small": 3}, "greater than or equal"),
        ({"stal_topk_small": 1.5}, "stal_topk_small"),
        ({"stal_small_area": float("nan")}, "finite"),
        ({"stal_medium_area": float("inf")}, "finite"),
        ({"stal_candidate_scale": float("nan")}, "finite"),
    ],
)
def test_stal_config_rejects_invalid_values_and_relationships(overrides, match):
    """STAL experiment parameters must fail fast instead of silently changing an experiment."""
    with pytest.raises((TypeError, ValueError), match=match):
        get_cfg(overrides=overrides)


def test_detection_loss_receives_stal_configuration():
    """Training configuration must reach the assigner used by detection loss."""
    import ultralytics.nn.tasks  # noqa: F401
    from ultralytics.utils.loss import v8DetectionLoss

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.args = get_cfg(overrides={"stal_mode": "adaptive", "stal_topk_small": 15})
            self.model = [SimpleNamespace(stride=torch.tensor([8.0, 16.0, 32.0]), nc=2, reg_max=16)]

    assigner = v8DetectionLoss(TinyModel()).assigner
    assert assigner.stal_mode == "adaptive"
    assert assigner.stal_topk[0] == 15


def test_tal_mode_uses_unmodified_gt_box():
    """Disabling STAL must be equivalent to raw TAL candidate geometry."""
    points = _grid_points(size=2)
    box = torch.tensor([[[7.0, 7.0, 9.0, 9.0]]])
    valid = torch.ones(1, 1, 1, dtype=torch.bool)
    mask = TaskAlignedAssigner(stal_mode="tal").select_candidates_in_gts(points, box, valid)
    lt, rb = box.unsqueeze(2).chunk(2, 3)
    expected = ((points - lt > 1e-9) & (rb - points > 1e-9)).all(3)
    torch.testing.assert_close(mask, expected)
    assert not mask.any()


def test_fixed_mode_keeps_legacy_stride_expansion():
    """The default fixed mode must keep expanding sub-stride dimensions to the middle stride."""
    points = _grid_points(size=2)
    box = torch.tensor([[[7.0, 7.0, 9.0, 9.0]]])
    valid = torch.ones(1, 1, 1, dtype=torch.bool)
    mask = TaskAlignedAssigner(stal_mode="fixed", stride=[8, 16, 32]).select_candidates_in_gts(points, box, valid)
    assert mask.sum().item() == 4


def test_adaptive_area_boundaries_use_coco_style_training_thresholds():
    """Boundary values are small only below 32^2 and large from 96^2 onward."""
    boxes = torch.tensor([[[0.0, 0.0, 31.0, 33.0], [0.0, 0.0, 32.0, 32.0], [0.0, 0.0, 96.0, 96.0]]])
    assigner = TaskAlignedAssigner(
        stal_mode="adaptive", stal_min_candidates=1, stal_topk_small=13, stal_topk_medium=7, stal_topk_large=3
    )
    assert assigner.get_adaptive_topks(boxes).tolist() == [[13, 7, 3]]


def test_adaptive_mode_guarantees_minimum_pre_conflict_candidates_for_tiny_gt():
    """A tiny valid GT receives the configured number of candidates before conflict resolution."""
    points = _grid_points(size=3)
    box = torch.tensor([[[7.5, 7.5, 8.5, 8.5]]])
    valid = torch.ones(1, 1, 1, dtype=torch.bool)
    assigner = TaskAlignedAssigner(
        stal_mode="adaptive", stal_min_candidates=5, stal_topk_small=5, stal_topk_medium=5, stal_topk_large=5
    )
    mask = assigner.select_candidates_in_gts(points, box, valid)
    assert mask.sum().item() >= 5
    assigner.bs, assigner.n_max_boxes = 1, 1
    positive_mask, _, _ = assigner.get_pos_mask(
        torch.zeros(1, points.shape[0], 1),
        torch.zeros(1, points.shape[0], 4),
        torch.zeros(1, 1, 1),
        box,
        points,
        valid,
    )
    assert positive_mask.sum().item() >= 5


def test_adaptive_assigner_handles_empty_gt():
    """Empty batches must return correctly shaped finite targets without positives."""
    points = _grid_points(size=2)
    anchors = points.shape[0]
    outputs = TaskAlignedAssigner(topk=3, num_classes=2, stal_mode="adaptive")(
        torch.full((1, anchors, 2), 0.5),
        torch.zeros(1, anchors, 4),
        points,
        torch.zeros(1, 0, 1),
        torch.zeros(1, 0, 4),
        torch.zeros(1, 0, 1, dtype=torch.bool),
    )
    assert not outputs[3].any()
    assert all(torch.isfinite(value).all() for value in outputs if value.is_floating_point())


def test_adaptive_assigner_resolves_overlapping_gt_conflicts_once_per_anchor():
    """Overlapping GT candidates must still resolve to at most one target for each anchor."""
    points = _grid_points(size=4)
    anchors = points.shape[0]
    boxes = torch.tensor([[[4.0, 4.0, 20.0, 20.0], [8.0, 8.0, 24.0, 24.0]]])
    labels = torch.tensor([[[0.0], [1.0]]])
    valid = torch.ones(1, 2, 1, dtype=torch.bool)
    pd_boxes = torch.cat((points - 6, points + 6), dim=-1).unsqueeze(0)
    pd_scores = torch.linspace(0.55, 0.95, anchors).view(1, anchors, 1).expand(-1, -1, 2)
    assigner = TaskAlignedAssigner(
        topk=5,
        num_classes=2,
        stal_mode="adaptive",
        stal_min_candidates=3,
        stal_topk_small=5,
        stal_topk_medium=5,
        stal_topk_large=5,
    )
    assigner.bs, assigner.n_max_boxes = 1, 2
    mask_pos, align_metric, overlaps = assigner.get_pos_mask(pd_scores, pd_boxes, labels, boxes, points, valid)
    _, fg_mask, resolved = assigner.select_highest_overlaps(mask_pos, overlaps, 2, align_metric)
    assert (resolved.sum(1) <= 1).all()
    assert fg_mask.sum() > 0
    assert torch.isfinite(align_metric).all()


def test_conflict_resolution_cannot_assign_anchor_to_non_candidate_gt():
    """Zero-overlap ties must stay within the GTs that proposed the contested anchor."""
    assigner = TaskAlignedAssigner(num_classes=3, stal_mode="adaptive")
    mask_pos = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]])
    overlaps = torch.zeros_like(mask_pos)
    align_metric = torch.zeros_like(mask_pos)
    original_counts = mask_pos.sum(-1)

    _, fg_mask, resolved = assigner.select_highest_overlaps(
        mask_pos, overlaps, n_max_boxes=3, align_metric=align_metric
    )

    assert resolved[0, 0, 2] == 0
    assert (resolved.sum(-1) <= original_counts).all()
    assert (fg_mask <= 1).all()


def _assignment_and_gradient(amp_enabled):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    points = _grid_points(size=4).to(device)
    anchors = points.shape[0]
    logits = torch.linspace(-1.0, 1.0, anchors, device=device).view(1, anchors, 1).requires_grad_()
    pd_boxes = torch.cat((points - 7, points + 7), dim=-1).unsqueeze(0)
    gt_boxes = torch.tensor([[[4.0, 4.0, 20.0, 20.0]]], device=device)
    gt_labels = torch.zeros(1, 1, 1, device=device)
    valid = torch.ones(1, 1, 1, dtype=torch.bool, device=device)
    assigner = TaskAlignedAssigner(
        topk=5,
        num_classes=1,
        stal_mode="adaptive",
        stal_min_candidates=3,
        stal_topk_small=5,
        stal_topk_medium=5,
        stal_topk_large=5,
    ).to(device)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        scores = logits.sigmoid()
        outputs = assigner(scores.detach(), pd_boxes, points, gt_labels, gt_boxes, valid)
        loss = F.binary_cross_entropy_with_logits(logits, outputs[2].to(logits.dtype))
    loss.backward()
    return outputs[3].cpu(), outputs[4].cpu(), float(loss.detach()), logits.grad.detach().cpu()


def test_adaptive_assigner_fp32_amp_masks_counts_loss_and_gradients_are_stable():
    """FP32 and autocast must keep assignment decisions aligned and all optimization values finite."""
    fp32_mask, fp32_idx, fp32_loss, fp32_grad = _assignment_and_gradient(False)
    amp_mask, amp_idx, amp_loss, amp_grad = _assignment_and_gradient(True)
    torch.testing.assert_close(amp_mask, fp32_mask)
    torch.testing.assert_close(amp_idx[amp_mask], fp32_idx[fp32_mask])
    assert amp_mask.sum() == fp32_mask.sum()
    assert fp32_loss == pytest.approx(amp_loss, rel=0.02, abs=1e-4)
    assert torch.isfinite(fp32_grad).all() and torch.isfinite(amp_grad).all()
