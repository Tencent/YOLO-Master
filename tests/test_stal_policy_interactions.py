"""Regression coverage for branch budgets and interacting candidate policies."""

from types import SimpleNamespace

import pytest
import torch

from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel  # noqa: F401 - initialize loss imports
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner


@pytest.mark.parametrize("topk,topk2", [(1, None), (7, 1), (10, None)])
@pytest.mark.parametrize("budget", [10, 15])
def test_loss_branch_budget(topk, topk2, budget):
    """Neutral budgets follow the branch and overrides remain one-to-many only."""
    model = torch.nn.Linear(1, 1)
    model.args = get_cfg(
        overrides={"stal_candidate_mode": "adaptive", "stal_small_topk": budget, "stal_small_topk_min": budget}
    )
    model.model = [SimpleNamespace(stride=torch.tensor([8.0, 16.0, 32.0]), nc=1, reg_max=16)]
    assigner = v8DetectionLoss(model, tal_topk=topk, tal_topk2=topk2).assigner
    assert assigner.stal_small_topk == (budget if topk == 10 else topk)
    anchors = torch.stack((torch.arange(1.0, 21.0), torch.full((20,), 5.0)), dim=1)
    boxes = torch.tensor([[[0.0, 0.0, 22.0, 12.0]]])
    result = assigner(
        torch.linspace(0.1, 1.0, 20).view(1, 20, 1),
        boxes.expand(1, 20, 4).clone(),
        anchors,
        torch.zeros(1, 1, 1),
        boxes,
        torch.ones(1, 1, 1, dtype=torch.bool),
        image_size=(800, 800),
        epoch=10,
    )
    assert result[3].sum() == (budget if topk == 10 else 1)


@pytest.mark.parametrize("budget", [3, 10, 15])
@pytest.mark.parametrize("cap", [1, 2])
def test_adaptive_topk_preserves_extra_cap(budget, cap):
    """Combining strategies must satisfy both upper bounds."""
    anchors = torch.stack((torch.arange(1.0, 15.0), torch.full((14,), 5.0)), dim=1)
    boxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0]]])
    valid = torch.ones(1, 1, 1, dtype=torch.bool)
    assigner = TaskAlignedAssigner(
        topk=10,
        num_classes=1,
        stal_candidate_mode="adaptive",
        stal_relaxation=10.0,
        stal_warmup_epochs=0,
        stal_area_threshold=0.5,
        stal_max_extra_candidates=cap,
        stal_small_topk=budget,
        stal_small_topk_min=budget,
        stride=[1, 1, 1],
    )
    assigner.bs = assigner.n_max_boxes = 1
    selected = assigner.get_pos_mask(
        torch.linspace(0.1, 0.9, 14).view(1, 14, 1),
        boxes.expand(1, 14, 4).clone(),
        torch.zeros(1, 1, 1),
        boxes,
        anchors,
        valid,
        image_size=(100, 100),
    )[0].bool()
    base = assigner.select_candidates_in_gts(anchors, boxes, valid, image_size=(100, 100), relaxation_override=0.0)
    assert (selected & ~base).sum() == cap
    if budget != 10:
        assert selected.sum() <= budget
    else:
        assert (selected & base).sum() == base.sum()


@pytest.mark.parametrize("epoch", [0, 1, 5, 10])
@pytest.mark.parametrize("scale_mode", ["constant", "sqrt_area"])
def test_crowding_respects_warmup_and_never_enlarges(epoch, scale_mode):
    """Crowding caps the area-scaled expansion with the same epoch warmup."""
    x = torch.linspace(-5.0, 15.0, 401) + 0.013
    anchors = torch.stack((x, torch.full_like(x, 5.0)), dim=1)
    boxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]]])
    valid = torch.ones(1, 2, 1, dtype=torch.bool)
    common = {
        "stal_candidate_mode": "adaptive",
        "stal_relaxation": 8.0,
        "stal_warmup_epochs": 10,
        "stal_area_threshold": 0.5,
        "stal_relaxation_scale_mode": scale_mode,
    }
    plain = TaskAlignedAssigner(**common)
    crowd = TaskAlignedAssigner(**common, stal_crowding_mode="candidate_overlap", stal_crowded_relaxation=4.0)
    baseline = plain.select_candidates_in_gts(anchors, boxes, valid, image_size=(100, 100), epoch=epoch)
    actual = crowd.select_candidates_in_gts(anchors, boxes, valid, image_size=(100, 100), epoch=epoch)
    assert not (actual & ~baseline).any()
    if scale_mode == "constant":
        expected = (x > -2.0 * epoch / 10 + 1e-9) & (x < 10.0 + 2.0 * epoch / 10 - 1e-9)
        assert torch.equal(actual, expected.view(1, 1, -1).expand_as(actual))


def test_one_to_one_final_filter_keeps_zero_metric_candidate():
    """An invalid zero-score anchor cannot displace a nominated zero-score anchor."""
    assigner = TaskAlignedAssigner(topk=1)
    nominations = torch.tensor([[[0.0, 1.0, 1.0]]])
    _, foreground, selected = assigner.select_highest_overlaps(
        nominations, torch.ones_like(nominations), 1, torch.zeros_like(nominations)
    )
    assert foreground.sum() == 1
    assert not selected[..., 0].any()
