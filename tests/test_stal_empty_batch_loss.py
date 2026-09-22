"""Exercise background-only assignment through detection loss and diagnostics."""

from types import SimpleNamespace

import pytest
import torch

from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel  # noqa: F401 - initialize the normal model/loss import path
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_empty_assignment_returns_indexable_types(dtype):
    """An empty target batch must preserve discrete output types and prediction precision."""
    scores = torch.zeros(2, 20, 3, dtype=dtype)
    boxes = torch.zeros(2, 20, 4, dtype=dtype)
    labels, targets, weights, foreground, indices = TaskAlignedAssigner(num_classes=3)(
        scores,
        boxes,
        torch.zeros(20, 2),
        torch.empty(2, 0, 1),
        torch.empty(2, 0, 4),
        torch.empty(2, 0, 1, dtype=torch.bool),
    )
    assert labels.dtype == indices.dtype == torch.long
    assert foreground.dtype == torch.bool
    assert targets.dtype == weights.dtype == dtype
    assert labels.shape == indices.shape == foreground.shape == (2, 20)
    assert (labels == 3).all()
    assert not foreground.any()
    assert not indices.any()
    assert not targets.any()
    assert not weights.any()


@pytest.mark.parametrize("mode", ["pure", "fixed", "adaptive", "rescue", "quality"])
@pytest.mark.parametrize("stats", [False, True])
def test_empty_batch_detection_loss_and_backward(mode, stats):
    """Diagnostics must not break background classification supervision or accumulate positive counts."""
    config = {
        "stal_candidate_mode": mode if mode in {"pure", "fixed"} else "adaptive",
        "stal_stats": stats,
        "stal_zero_positive_rescue": mode == "rescue",
        "stal_rescue_score_floor": 0.01 if mode == "rescue" else 0.0,
        "stal_expanded_quality_ratio": 0.25 if mode == "quality" else 0.0,
    }
    model = torch.nn.Linear(1, 1)
    model.args = get_cfg(overrides=config)
    model.model = [SimpleNamespace(stride=torch.tensor([8.0, 16.0, 32.0]), nc=3, reg_max=16)]
    criterion = v8DetectionLoss(model)
    predictions = {
        "feats": [torch.zeros(2, 1, size, size) for size in (8, 4, 2)],
        "boxes": torch.zeros(2, 64, 84, requires_grad=True),
        "scores": torch.zeros(2, 3, 84, requires_grad=True),
    }
    batch = {
        "batch_idx": torch.empty(0),
        "cls": torch.empty(0, 1),
        "bboxes": torch.empty(0, 4),
        "epoch": 10,
    }
    assigned, loss, _ = criterion.get_assigned_targets_and_loss(predictions, batch)
    assert assigned[0].dtype == torch.bool
    assert assigned[1].dtype == torch.long
    assert not assigned[0].any()
    expected_cls = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            predictions["scores"], torch.zeros_like(predictions["scores"]), reduction="sum"
        )
        * model.args.cls
    )
    torch.testing.assert_close(loss[1], expected_cls)
    assert loss[0] == loss[2] == 0
    loss.sum().backward()
    assert torch.isfinite(predictions["scores"].grad).all()
    assert (predictions["scores"].grad > 0).all()
    assert predictions["boxes"].grad is None  # No regression targets in a background-only batch.
    assert not criterion.pop_assignment_stats().any()
    assert not criterion.pop_rescue_stats().any()
    assert not criterion.pop_assignment_stage_stats().any()
    assert not criterion.pop_target_score_stats().any()
