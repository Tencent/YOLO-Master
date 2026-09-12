"""Guard scale gates and telemetry at the formal 800-pixel training resolution."""

from types import SimpleNamespace

import torch

from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel  # noqa: F401
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner


def test_half_canvas_does_not_turn_medium_targets_into_stal_targets():
    """800*800 overflows FP16: 100*100/inf must not become a false small-target classification."""
    assigner = TaskAlignedAssigner()
    boxes = torch.tensor([[[0.0, 0.0, 20.0, 20.0], [0.0, 0.0, 80.0, 80.0], [0.0, 0.0, 100.0, 100.0]]])
    size = torch.tensor([800.0, 800.0], dtype=torch.float16)
    assert assigner.small_target_mask(boxes, size).tolist() == [[True, False, False]]


def test_half_candidate_geometry_matches_fp32_at_formal_resolution():
    """A 100px box is above 1% at 800px and must not receive adaptive expansion."""
    assigner = TaskAlignedAssigner(stal_candidate_mode="adaptive", stal_warmup_epochs=0)
    boxes = torch.tensor([[[100.0, 100.0, 200.0, 200.0]]])
    anchors = torch.tensor([[98.0, 150.0], [150.0, 150.0]])
    valid = torch.ones(1, 1, 1, dtype=torch.bool)
    for dtype in (torch.float32, torch.float16):
        selected = assigner.select_candidates_in_gts(anchors.to(dtype), boxes.to(dtype), valid, image_size=(800, 800))
        assert selected.tolist() == [[[False, True]]]


def test_half_assignment_statistics_preserve_stal_area_gate():
    """A medium-sized target must not enter the relative-area STAL statistics group."""
    assigner = TaskAlignedAssigner()
    stats = assigner.assignment_statistics(
        torch.tensor([[[100.0, 100.0, 200.0, 200.0]]], dtype=torch.float16),
        torch.ones(1, 1, 1, dtype=torch.bool),
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 1, dtype=torch.long),
        (800, 800),
    )
    assert stats[:3].tolist() == [0, 0, 0]
    assert stats[-3:].tolist() == [1, 1, 0]


def test_fp16_loss_score_statistics_exclude_non_stal_gt():
    """Exercise the actual loss telemetry with an FP16 800px canvas and FP32 training labels."""
    model = torch.nn.Linear(1, 1)
    model.args = get_cfg(overrides={"stal_stats": True})
    model.model = [SimpleNamespace(stride=torch.tensor([8.0, 16.0, 32.0]), nc=1, reg_max=16)]
    loss = v8DetectionLoss(model)
    predictions = {
        "feats": [torch.zeros(1, 1, s, s) for s in (100, 50, 25)],
        "boxes": torch.zeros(1, 64, 13125, dtype=torch.float16),
        "scores": torch.zeros(1, 1, 13125, dtype=torch.float16),
    }
    batch = {
        "batch_idx": torch.zeros(1),
        "cls": torch.zeros(1, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.125, 0.125]]),
        "epoch": 10,
    }
    # CPU autocast keeps the BCE computation supported while predictions remain genuinely FP16.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss.get_assigned_targets_and_loss(predictions, batch)
    stats = loss.pop_target_score_stats()
    assert stats[3] > 0
    assert stats[:3].tolist() == [0.0, 0.0, 0.0]
