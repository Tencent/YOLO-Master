"""Tests for non-invasive TAL/STAL positive-assignment telemetry."""

import torch

from ultralytics import YOLO  # noqa: F401 - initialize task/loss imports in package order
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner


class _TelemetryAssigner:
    """Minimal assigner surface used by telemetry-only loss tests."""

    def __init__(self, stage_counts=None):
        self._stage_counts = stage_counts or {}

    def assignment_stage_counts(self):
        return self._stage_counts

    def coverage_stats(self):
        return {}


def _bare_loss(enabled: bool = True, stage_counts=None) -> v8DetectionLoss:
    """Construct only the telemetry state without building a detection model."""
    loss = object.__new__(v8DetectionLoss)
    loss.assignment_stats_enabled = enabled
    loss.assignment_small_area = 32.0**2
    loss.assignment_medium_area = 96.0**2
    loss.assigner = _TelemetryAssigner(stage_counts)
    loss._assignment_stats = torch.zeros(len(loss._ASSIGNMENT_STAT_NAMES), dtype=torch.long)
    return loss


def test_assignment_stats_are_binned_and_reset_without_changing_inputs():
    """Counters should preserve inputs and report post-conflict positives by GT area."""
    loss = _bare_loss(
        stage_counts={
            "candidate": torch.tensor([[2, 1, 0, 3]]),
            "preconflict": torch.tensor([[2, 1, 0, 2]]),
        }
    )
    gt_bboxes = torch.tensor(
        [[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 40.0, 40.0], [0.0, 0.0, 100.0, 100.0], [0, 0, 120, 120]]]
    )
    mask_gt = torch.ones(1, 4, 1, dtype=torch.bool)
    fg_mask = torch.tensor([[True, True, True, True, False]])
    target_gt_idx = torch.tensor([[0, 0, 1, 2, 0]])
    originals = tuple(x.clone() for x in (gt_bboxes, mask_gt, fg_mask, target_gt_idx))

    loss._update_assignment_stats(gt_bboxes, mask_gt, fg_mask, target_gt_idx)
    stats = {key: int(value) for key, value in loss.assignment_stats().items()}

    assert stats == {
        "gt_total": 4,
        "pos_total": 4,
        "zero_gt": 1,
        "gt_small": 1,
        "gt_medium": 1,
        "gt_large": 2,
        "pos_small": 2,
        "pos_medium": 1,
        "pos_large": 1,
        "zero_small": 0,
        "zero_medium": 0,
        "zero_large": 1,
        "candidate_total": 6,
        "candidate_small": 2,
        "candidate_medium": 1,
        "candidate_large": 3,
        "zero_candidate_gt": 1,
        "zero_candidate_small": 0,
        "zero_candidate_medium": 0,
        "zero_candidate_large": 1,
        "preconflict_pos_total": 5,
        "preconflict_pos_small": 2,
        "preconflict_pos_medium": 1,
        "preconflict_pos_large": 2,
        "zero_preconflict_gt": 1,
        "zero_preconflict_small": 0,
        "zero_preconflict_medium": 0,
        "zero_preconflict_large": 1,
        "coverage_gt_base0": 0,
        "coverage_gt_base1": 0,
        "coverage_gt_base2": 0,
        "coverage_gt_base3plus": 0,
        "coverage_gt_ordinary": 0,
        "coverage_gt_elongated": 0,
        "coverage_triggered_gt": 0,
        "coverage_supplement_candidates": 0,
        "coverage_supplement_topk": 0,
        "coverage_supplement_conflict": 0,
        "coverage_supplement_final": 0,
    }
    for actual, original in zip((gt_bboxes, mask_gt, fg_mask, target_gt_idx), originals):
        torch.testing.assert_close(actual, original)

    loss.reset_assignment_stats()
    assert all(int(value) == 0 for value in loss.assignment_stats().values())


def test_assignment_stats_disabled_is_a_noop():
    """The default-off path should expose no metrics and perform no counter work."""
    loss = _bare_loss(enabled=False)
    loss._update_assignment_stats(
        torch.tensor([[[0.0, 0.0, 10.0, 10.0]]]),
        torch.ones(1, 1, 1, dtype=torch.bool),
        torch.tensor([[True]]),
        torch.tensor([[0]]),
    )
    assert loss.assignment_stats() == {}
    assert not loss._assignment_stats.any()


def test_dynamic_topk_changes_only_small_gt_candidate_count():
    """Dynamic TopK should use ceil(lambda*x) for small GTs and retain fixed K for larger GTs."""
    assigner = TaskAlignedAssigner(topk=3, small_area_threshold=32.0**2, dynamic_topk_small=True, dynamic_topk_lambda=0.8)
    metrics = torch.tensor([[[9.0, 8.0, 7.0, 6.0, 5.0, 0.0], [0.0, 9.0, 8.0, 7.0, 6.0, 5.0]]])
    candidates = torch.tensor([[[True, True, True, True, True, False], [False, True, True, True, True, True]]])
    gt_bboxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 40.0, 40.0]]])
    valid_gts = torch.ones(1, 2, 1, dtype=torch.bool)

    selected = assigner.select_topk_candidates(
        metrics,
        topk_mask=valid_gts.expand(-1, -1, 3),
        candidate_mask=candidates,
        gt_bboxes=gt_bboxes,
        mask_gt=valid_gts,
    )

    assert selected[0, 0].sum() == 4  # ceil(0.8 * 5)
    assert selected[0, 1].sum() == 3  # unchanged fixed K
    assert not selected.bool().logical_and(~candidates).any()


def test_assigner_reports_candidate_topk_and_conflict_stages_without_changing_assignment():
    """Two GTs competing for one anchor should be distinguishable from true zero-candidate GTs."""
    assigner = TaskAlignedAssigner(
        topk=1,
        num_classes=1,
        candidate_expand_0_8=-1,
        collect_coverage_stats=True,
    )
    scores = torch.tensor([[[0.9], [0.1]]])
    predicted_boxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [10.0, 0.0, 20.0, 10.0]]])
    anchor_points = torch.tensor([[5.0, 5.0], [15.0, 5.0]])
    gt_labels = torch.zeros(1, 2, 1)
    gt_boxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]]])
    valid = torch.ones(1, 2, 1, dtype=torch.bool)

    _, _, _, foreground, _ = assigner(scores, predicted_boxes, anchor_points, gt_labels, gt_boxes, valid)
    stages = assigner.assignment_stage_counts()

    assert stages["candidate"].tolist() == [[1, 1]]
    assert stages["preconflict"].tolist() == [[1, 1]]
    assert stages["final"].sum().item() == foreground.sum().item() == 1
    assert stages["final"].eq(0).sum().item() == 1
