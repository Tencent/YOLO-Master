"""Protect candidate diagnostic semantics and unchanged assignment outputs."""

import torch

from scripts.stal.diagnose_rescue_candidates import DiagnosticAssigner, candidate_counts, outside_capacity_counts
from ultralytics.utils.tal import TaskAlignedAssigner


def test_outside_anchor_capacity_and_occupied_candidate():
    """A 2x2 GT at the origin cannot reach AP50 with its only point at (4,4)."""
    boxes = torch.tensor([[[0.0, 0.0, 2.0, 2.0], [3.0, 3.0, 5.0, 5.0]]])
    anchors = torch.tensor([[4.0, 4.0], [12.0, 12.0]])
    selected = torch.ones(1, 2, dtype=torch.bool)
    counts = outside_capacity_counts(boxes, anchors, selected, torch.tensor([[True, False]]))
    assert counts["outside_gt"] == 2
    assert counts["outside_any_iou_cap_ge_0.5"] == 1
    assert counts["outside_any_iou_cap_ge_0.95"] == 1
    assert counts["outside_free_iou_cap_ge_0.5"] == 0


def test_counts_separate_no_candidate_occupied_and_threshold_suppression():
    """A positive CIoU can be filtered by alignment eps; occupied anchors cannot be rescued."""
    mask = torch.tensor([[[1, 0], [0, 0], [0, 0], [0, 0]]])
    legal = torch.tensor([[[1, 0], [0, 1], [1, 0], [0, 0]]], dtype=torch.bool)
    quality = torch.tensor([[[1.0, 0.0], [0.0, 0.03], [0.8, 0.0], [0.0, 0.0]]])
    metric = quality.pow(6) * 0.1
    counts = candidate_counts(mask, quality, metric, legal, torch.ones(1, 4, dtype=torch.bool), 1e-9)
    assert counts["uncovered"] == 3
    assert counts["uncovered_without_legal"] == 1
    assert counts["uncovered_with_free_legal"] == 1
    assert counts["positive_ciou_but_alignment_below_eps"] == 1
    assert counts["suppressed_ciou_ge_0.03"] == 1
    assert counts["suppressed_ciou_ge_0.05"] == 0


def test_diagnostic_preserves_all_assigner_outputs():
    """Observation must not change pure TAL labels, masks, boxes, scores, or GT ownership."""
    inputs = (
        torch.tensor([[[0.5], [0.1]]]),
        torch.tensor([[[0.0, 0.0, 6.0, 6.0], [0.0, 0.0, 10.0, 6.0]]]),
        torch.tensor([[3.0, 3.0], [8.0, 3.0]]),
        torch.zeros(1, 2, 1),
        torch.tensor([[[0.0, 0.0, 6.0, 6.0], [0.0, 0.0, 10.0, 6.0]]]),
        torch.ones(1, 2, 1, dtype=torch.bool),
    )
    baseline = TaskAlignedAssigner(topk=1, num_classes=1, stal_candidate_mode="pure")
    diagnostic = DiagnosticAssigner(topk=1, num_classes=1, stal_candidate_mode="pure")
    DiagnosticAssigner.active_epoch = 0
    DiagnosticAssigner.records = []
    try:
        expected = baseline(*inputs, image_size=(100, 100))
        actual = diagnostic(*inputs, image_size=(100, 100))
        assert all(torch.equal(a, b) for a, b in zip(expected, actual))
        assert len(DiagnosticAssigner.records) == 1
    finally:
        DiagnosticAssigner.active_epoch = None
