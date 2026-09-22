"""Keep relative STAL and absolute small-target telemetry populations distinct."""

from types import SimpleNamespace

import pytest
import torch

from ultralytics.cfg import get_cfg
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.tasks import DetectionModel  # noqa: F401
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner


@pytest.mark.parametrize("side", [24, 32, 40, 80])
def test_score_groups_match_their_gt_area_definitions(side):
    model = torch.nn.Linear(1, 1)
    model.args = get_cfg(overrides={"stal_stats": True})
    model.model = [SimpleNamespace(stride=torch.tensor([8.0, 16.0, 32.0]), nc=1, reg_max=16)]
    loss = v8DetectionLoss(model)
    predictions = {
        "feats": [torch.zeros(1, 1, s, s) for s in (100, 50, 25)],
        "boxes": torch.zeros(1, 64, 13125),
        "scores": torch.zeros(1, 1, 13125),
    }
    batch = {
        "batch_idx": torch.zeros(1),
        "cls": torch.zeros(1, 1),
        "bboxes": torch.tensor([[0.5, 0.5, side / 800, side / 800]]),
        "epoch": 10,
    }
    loss.get_assigned_targets_and_loss(predictions, batch)
    score = loss.pop_target_score_stats()
    assignment = loss.pop_assignment_stats()
    assert score.shape == (9,)  # STAL, all, COCO-style small; three values each
    assert score[3] > 0
    assert score[0] == assignment[1]
    assert score[6] == assignment[4]
    assert bool(score[0]) == (side < 80)
    assert bool(score[6]) == (side < 32)
    assert not loss.pop_target_score_stats().any()


@pytest.mark.parametrize("constructor", [TaskAlignedAssigner, lambda **kw: get_cfg(overrides=kw)])
def test_guarantee_cannot_override_a_configured_capacity_floor(constructor):
    with pytest.raises(ValueError, match="guarantee.*floor"):
        constructor(stal_candidate_mode="adaptive", stal_min_candidate_guarantee=True, stal_candidate_iou_floor=0.75)


def test_epoch_attempt_discards_all_abandoned_statistics():
    model = torch.nn.Linear(1, 1)
    model.args = get_cfg(overrides={"stal_stats": True})
    model.model = [SimpleNamespace(stride=torch.tensor([8.0, 16.0, 32.0]), nc=1, reg_max=16)]
    criterion = v8DetectionLoss(model)
    model.criterion = criterion
    counters = [
        criterion._assignment_stats,
        criterion._rescue_stats,
        criterion._assignment_stage_stats,
        criterion._target_score_stats,
    ]
    for counter in counters:
        counter.fill_(7)
    BaseTrainer._reset_assignment_metric_state(SimpleNamespace(model=model))
    assert all(not counter.any() for counter in counters)
    criterion._assignment_stats[0] = 3
    assert criterion.pop_assignment_stats()[0] == 3


def test_initial_epoch_reset_accepts_lazy_criterion():
    BaseTrainer._reset_assignment_metric_state(SimpleNamespace(model=torch.nn.Linear(1, 1)))


@pytest.mark.parametrize(
    "name,buffer",
    [
        ("pop_assignment_stats", "_assignment_stats"),
        ("pop_rescue_stats", "_rescue_stats"),
        ("pop_assignment_stage_stats", "_assignment_stage_stats"),
        ("pop_target_score_stats", "_target_score_stats"),
    ],
)
def test_e2e_reports_one_to_many_without_double_counting_and_drains_both(name, buffer):
    model = torch.nn.Linear(1, 1)
    model.args = get_cfg(overrides={"stal_stats": True})
    model.model = [SimpleNamespace(stride=torch.tensor([8.0, 16.0, 32.0]), nc=1, reg_max=16)]
    loss = E2ELoss(model)
    getattr(loss.one2many, buffer).fill_(3)
    getattr(loss.one2one, buffer).fill_(7)
    assert (getattr(loss, name)() == 3).all()
    assert not getattr(loss.one2many, buffer).any()
    assert not getattr(loss.one2one, buffer).any()


def test_new_stats_cannot_append_to_historical_csv_schema(tmp_path):
    path = tmp_path / "results.csv"
    previous = "epoch,time,assign/small_nonzero_score_pos\n1,1,3\n"
    path.write_text(previous, encoding="utf-8")
    trainer = SimpleNamespace(csv=path, train_time_start=0, epoch=1)
    metrics = {"assign/stal_nonzero_score_pos": 3, "assign/small_nonzero_score_pos": 1}
    with pytest.raises(ValueError, match="schema changed"):
        BaseTrainer.save_metrics(trainer, metrics)
    assert path.read_text() == previous


def test_new_stats_append_consistently_in_a_fresh_run(tmp_path):
    trainer = SimpleNamespace(csv=tmp_path / "results.csv", train_time_start=0, epoch=0)
    metrics = {"assign/stal_nonzero_score_pos": 3, "assign/small_nonzero_score_pos": 1}
    BaseTrainer.save_metrics(trainer, metrics)
    trainer.epoch = 1
    BaseTrainer.save_metrics(trainer, metrics)
    assert len(trainer.csv.read_text().splitlines()) == 3
