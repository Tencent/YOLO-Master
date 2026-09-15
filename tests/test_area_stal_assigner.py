"""Tests for the A2 Area-Threshold STAL assigner and paired bootstrap."""

from __future__ import annotations

import csv
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from scripts.reproduce.a2._config import A2_CONFIG
from scripts.reproduce.a2._scale_stats import empty_scale_stats, flatten_scale_stats, update_scale_stats
from scripts.reproduce.a2.bootstrap_scale_delta import _bootstrap_area_ranges, paired_bootstrap
from scripts.reproduce.a2.evaluate_scale_aware import _qa_area_ranges
from scripts.reproduce.a2.reproduce_esmoe_baseline import build_parser as build_baseline_parser
from scripts.reproduce.a2.reproduce_esmoe_stal import _install_positive_stats, build_parser
from ultralytics.utils.tal import AreaAwareTaskAlignedAssigner, TaskAlignedAssigner


def _anchors() -> torch.Tensor:
    return torch.tensor([[4.0, 4.0], [12.0, 4.0], [4.0, 12.0], [12.0, 12.0], [20.0, 12.0], [12.0, 20.0]])


def test_area_threshold_boundary_is_inclusive():
    assigner = AreaAwareTaskAlignedAssigner(stride=[8, 16, 32], area_threshold=64.0)
    mask_gt = torch.ones((1, 2, 1), dtype=torch.bool)
    boxes = torch.tensor([[[7.0, 7.0, 15.0, 15.0], [7.0, 7.0, 16.0, 15.0]]])

    candidates = assigner.select_candidates_in_gts(_anchors(), boxes, mask_gt)

    assert int(candidates[0, 0].sum().item()) == 4
    assert int(candidates[0, 1].sum().item()) == 1


def test_area_floor_adds_only_missing_candidates():
    assigner = AreaAwareTaskAlignedAssigner(stride=[8, 16, 32], area_threshold=256.0)
    reference = TaskAlignedAssigner(stride=[8, 16, 32])
    mask_gt = torch.ones((1, 2, 1), dtype=torch.bool)
    boxes = torch.tensor([[[7.0, 7.0, 15.0, 15.0], [0.0, 0.0, 16.0, 16.0]]])

    baseline = reference.select_candidates_in_gts(_anchors(), boxes, mask_gt)
    candidates = assigner.select_candidates_in_gts(_anchors(), boxes, mask_gt)

    assert int(baseline[0, 0].sum().item()) == 1
    assert int(candidates[0, 0].sum().item()) == 4
    assert torch.equal(candidates[0, 1], baseline[0, 1])
    assert assigner.last_area_stats["floor_added"] == 3


def test_target_above_area_threshold_matches_baseline():
    kwargs = {"stride": [8, 16, 32]}
    assigner = AreaAwareTaskAlignedAssigner(**kwargs, area_threshold=64.0)
    reference = TaskAlignedAssigner(**kwargs)
    mask_gt = torch.ones((1, 1, 1), dtype=torch.bool)
    boxes = torch.tensor([[[7.0, 7.0, 16.0, 15.0]]])

    actual = assigner.select_candidates_in_gts(_anchors(), boxes, mask_gt)
    expected = reference.select_candidates_in_gts(_anchors(), boxes, mask_gt)

    assert torch.equal(actual, expected)


def test_one_to_one_candidate_geometry_matches_baseline():
    kwargs = {"topk": 4, "topk2": 1, "num_classes": 1, "stride": [8, 16, 32]}
    assigner = AreaAwareTaskAlignedAssigner(**kwargs, area_threshold=256.0)
    reference = TaskAlignedAssigner(**kwargs)
    mask_gt = torch.ones((1, 1, 1), dtype=torch.bool)
    boxes = torch.tensor([[[7.0, 7.0, 15.0, 15.0]]])

    actual = assigner.select_candidates_in_gts(_anchors(), boxes, mask_gt)
    expected = reference.select_candidates_in_gts(_anchors(), boxes, mask_gt)

    assert torch.equal(actual, expected)
    assert assigner.last_area_stats["assignment_branch"] == "one2one"


def test_full_forward_matches_baseline_outside_threshold():
    kwargs = {"topk": 4, "topk2": 4, "num_classes": 1, "stride": [8, 16, 32]}
    assigner = AreaAwareTaskAlignedAssigner(**kwargs, area_threshold=64.0)
    reference = TaskAlignedAssigner(**kwargs)
    anchors = _anchors()
    scores = torch.tensor([[[0.8], [0.7], [0.6], [0.5], [0.4], [0.3]]])
    predicted = torch.stack([torch.tensor([x - 4.0, y - 4.0, x + 4.0, y + 4.0]) for x, y in anchors]).unsqueeze(0)
    labels = torch.zeros((1, 1, 1), dtype=torch.long)
    targets = torch.tensor([[[4.0, 4.0, 20.0, 20.0]]])
    mask_gt = torch.ones((1, 1, 1), dtype=torch.bool)

    actual = assigner(scores, predicted, anchors, labels, targets, mask_gt)
    expected = reference(scores, predicted, anchors, labels, targets, mask_gt)

    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.equal(actual_tensor, expected_tensor)


def test_area_floor_keeps_native_alignment_quality():
    kwargs = {"topk": 4, "topk2": 4, "num_classes": 1, "stride": [8, 16, 32]}
    assigner = AreaAwareTaskAlignedAssigner(**kwargs, area_threshold=64.0)
    reference = TaskAlignedAssigner(**kwargs)
    anchors = _anchors()
    scores = torch.tensor([[[0.8], [0.7], [0.6], [0.5], [0.4], [0.3]]])
    predicted = torch.stack([torch.tensor([x - 4.0, y - 4.0, x + 4.0, y + 4.0]) for x, y in anchors]).unsqueeze(0)
    labels = torch.zeros((1, 1, 1), dtype=torch.long)
    targets = torch.tensor([[[7.0, 7.0, 15.0, 15.0]]])
    metric_mask = torch.ones((1, 1, len(anchors)), dtype=torch.bool)
    assigner.bs = reference.bs = 1
    assigner.n_max_boxes = reference.n_max_boxes = 1

    actual_metric, actual_overlaps = assigner.get_box_metrics(scores, predicted, labels, targets, metric_mask)
    expected_metric, expected_overlaps = reference.get_box_metrics(scores, predicted, labels, targets, metric_mask)

    assert torch.equal(actual_metric, expected_metric)
    assert torch.equal(actual_overlaps, expected_overlaps)


def test_area_floor_tie_breaking_keeps_added_candidates_in_topk():
    """All-zero early metrics must rank eligible candidates before outside anchors."""
    kwargs = {"topk": 4, "topk2": 4, "num_classes": 1, "stride": [8, 16, 32]}
    assigner = AreaAwareTaskAlignedAssigner(**kwargs, area_threshold=64.0)
    anchors = torch.tensor([[0.0, 0.0], [0.0, 40.0], [40.0, 0.0], [40.0, 40.0], [12.0, 12.0], [20.0, 20.0]])
    scores = torch.zeros((1, len(anchors), 1))
    predicted = torch.zeros((1, len(anchors), 4))
    labels = torch.zeros((1, 1, 1), dtype=torch.long)
    targets = torch.tensor([[[7.0, 7.0, 15.0, 15.0]]])
    mask_gt = torch.ones((1, 1, 1), dtype=torch.bool)

    _, _, _, fg_mask, _ = assigner(scores, predicted, anchors, labels, targets, mask_gt)

    assert int(fg_mask.sum().item()) == 4


def test_conflict_statistics_are_recorded_after_native_resolution():
    assigner = AreaAwareTaskAlignedAssigner(
        topk=4,
        topk2=4,
        num_classes=1,
        stride=[8, 16, 32],
        area_threshold=64.0,
    )
    anchors = _anchors()
    scores = torch.full((1, len(anchors), 1), 0.8)
    predicted = torch.stack([torch.tensor([x - 4.0, y - 4.0, x + 4.0, y + 4.0]) for x, y in anchors]).unsqueeze(0)
    labels = torch.zeros((1, 2, 1), dtype=torch.long)
    targets = torch.tensor([[[7.0, 7.0, 15.0, 15.0], [7.0, 7.0, 15.0, 15.0]]])
    mask_gt = torch.ones((1, 2, 1), dtype=torch.bool)

    assigner(scores, predicted, anchors, labels, targets, mask_gt)

    stats = assigner.last_area_stats
    assert stats["conflict_anchors"] > 0
    assert stats["post_assigned"] <= stats["pre_assigned"]
    assert stats["eligible_gt"] == 2


def test_area_threshold_must_be_positive():
    with pytest.raises(ValueError, match="area_threshold"):
        AreaAwareTaskAlignedAssigner(area_threshold=0.0)

    with pytest.raises(ValueError, match="min_candidates"):
        AreaAwareTaskAlignedAssigner(min_candidates=0)


def test_a2_runner_reads_final_protocol_from_default_yaml():
    args = build_parser().parse_args([])

    assert args.epochs == A2_CONFIG["epochs"] == 120
    assert args.imgsz == A2_CONFIG["imgsz"] == 800
    assert args.batch == A2_CONFIG["batch"] == 8
    assert args.seed == A2_CONFIG["seed"] == 42
    assert args.model == A2_CONFIG["model"] == "v0.1-N"
    assert args.stal_area_threshold == A2_CONFIG["area_threshold"] == 16.0
    assert args.stal_min_candidates == A2_CONFIG["min_candidates"] == 4

    for forbidden_args in (["--stal-area-threshold", "16"], ["--epochs", "1"]):
        with pytest.raises(SystemExit):
            build_parser().parse_args(forbidden_args)

    baseline_args = build_baseline_parser().parse_args([])
    assert baseline_args.epochs == A2_CONFIG["epochs"]
    assert baseline_args.imgsz == A2_CONFIG["imgsz"]
    assert baseline_args.batch == A2_CONFIG["batch"]
    assert baseline_args.seed == A2_CONFIG["seed"]
    assert baseline_args.model == A2_CONFIG["model"]


def test_scale_stats_use_strict_qa_area_boundaries_and_final_assignments():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 31.0, 33.0],  # 1023: small
            [0.0, 0.0, 32.0, 32.0],  # 1024: medium
            [0.0, 0.0, 95.0, 97.0],  # 9215: medium
            [0.0, 0.0, 96.0, 96.0],  # 9216: large
        ]
    ).unsqueeze(0)
    mask_gt = torch.ones((1, 4, 1), dtype=torch.bool)
    fg_mask = torch.ones((1, 6), dtype=torch.bool)
    target_gt_idx = torch.tensor([[0, 0, 1, 2, 2, 3]])
    result = (None, None, torch.ones((1, 6, 1)), fg_mask, target_gt_idx)
    stats = empty_scale_stats()

    update_scale_stats(stats, boxes, mask_gt, result)
    flat = flatten_scale_stats(stats)

    assert flat["small_gt"] == 1
    assert flat["small_post_pos_total"] == 2
    assert flat["small_post_avg_pos"] == pytest.approx(2.0)
    assert flat["medium_gt"] == 2
    assert flat["medium_post_pos_total"] == 3
    assert flat["medium_post_avg_pos"] == pytest.approx(1.5)
    assert flat["large_gt"] == 1
    assert flat["large_post_pos_total"] == 1
    assert flat["large_zero_post_ratio"] == pytest.approx(0.0)


def test_scale_stats_empty_gt_is_zeroed():
    stats = empty_scale_stats()
    boxes = torch.zeros((2, 0, 4))
    mask_gt = torch.zeros((2, 0, 1), dtype=torch.bool)
    result = (
        torch.zeros((2, 5), dtype=torch.long),
        torch.zeros((2, 5, 4)),
        torch.zeros((2, 5, 1)),
        torch.zeros((2, 5), dtype=torch.bool),
        torch.zeros((2, 5), dtype=torch.long),
    )

    update_scale_stats(stats, boxes, mask_gt, result)
    assert all(values == 0 for values in flatten_scale_stats(stats).values())


def test_eval_area_ranges_match_qa_half_open_boundaries():
    ranges = _qa_area_ranges()

    assert ranges[1][1] < 32**2
    assert ranges[2][0] == 32**2
    assert ranges[2][1] < 96**2
    assert ranges[3][0] == 96**2


def test_bootstrap_uses_the_same_area_boundaries_as_eval():
    assert _bootstrap_area_ranges() == _qa_area_ranges()


def test_epoch_stats_callback_writes_compact_assignment_csv(tmp_path):
    class ConfiguredAreaAssigner(AreaAwareTaskAlignedAssigner):
        pass

    callbacks, restore = _install_positive_stats(ConfiguredAreaAssigner, tmp_path, resume=False)
    try:
        assigner = ConfiguredAreaAssigner(
            topk=4,
            topk2=4,
            num_classes=1,
            stride=[8, 16, 32],
            area_threshold=64.0,
        )
        anchors = _anchors()
        scores = torch.full((1, len(anchors), 1), 0.8)
        predicted = torch.stack([torch.tensor([x - 4.0, y - 4.0, x + 4.0, y + 4.0]) for x, y in anchors]).unsqueeze(0)
        labels = torch.zeros((1, 1, 1), dtype=torch.long)
        targets = torch.tensor([[[7.0, 7.0, 15.0, 15.0]]])
        mask_gt = torch.ones((1, 1, 1), dtype=torch.bool)

        callbacks["on_train_epoch_start"](SimpleNamespace(epoch=0))
        assigner(scores, predicted, anchors, labels, targets, mask_gt)
        callbacks["on_fit_epoch_end"](SimpleNamespace(epoch=0))
    finally:
        restore()

    with (tmp_path / "a2_area_stal_positive_stats.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["epoch"] == "1"
    assert rows[0]["eligible_gt"] == "1"
    assert rows[0]["floor_added"] == "3"
    assert int(rows[0]["train_pos_total"]) > 0
    assert rows[0]["small_gt"] == "1"
    assert float(rows[0]["small_post_avg_pos"]) > 0
    assert float(rows[0]["small_zero_post_ratio"]) == pytest.approx(0.0)


def _bootstrap_fixture():
    ground_truth = {
        "images": [{"id": index, "width": 10, "height": 10} for index in range(1, 5)],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}],
    }
    baseline = [{"image_id": index, "score": float(index) / 10} for index in range(1, 5)]
    return ground_truth, baseline


def _mean_score(dataset: dict, predictions: list[dict]) -> float:
    scores = defaultdict(list)
    for prediction in predictions:
        scores[int(prediction["image_id"])].append(float(prediction["score"]))
    return sum(max(scores.get(int(image["id"]), [0.0])) for image in dataset["images"]) / len(dataset["images"])


def test_paired_bootstrap_zero_difference_fails_p1():
    ground_truth, baseline = _bootstrap_fixture()
    result = paired_bootstrap(
        ground_truth,
        baseline,
        baseline,
        replicates=40,
        seed=42,
        evaluator=_mean_score,
        progress_every=0,
    )

    assert result["point_delta"] == pytest.approx(0.0)
    assert result["ci_low"] == pytest.approx(0.0)
    assert not result["p1_passed"]


def test_paired_bootstrap_positive_difference_passes_p1():
    ground_truth, baseline = _bootstrap_fixture()
    candidate = [{**item, "score": item["score"] + 0.02} for item in baseline]
    result = paired_bootstrap(
        ground_truth,
        baseline,
        candidate,
        replicates=40,
        seed=42,
        evaluator=_mean_score,
        progress_every=0,
    )

    assert result["point_delta"] == pytest.approx(0.02)
    assert result["ci_low"] > 0
    assert result["p1_passed"]


def test_paired_bootstrap_is_deterministic_for_fixed_seed():
    ground_truth, baseline = _bootstrap_fixture()
    candidate = [
        {**item, "score": item["score"] + increment} for item, increment in zip(baseline, (0.00, 0.01, 0.02, 0.03))
    ]
    kwargs = {
        "replicates": 50,
        "seed": 13,
        "evaluator": _mean_score,
        "progress_every": 0,
    }

    first = paired_bootstrap(ground_truth, baseline, candidate, **kwargs)
    second = paired_bootstrap(ground_truth, baseline, candidate, **kwargs)

    assert first == second
