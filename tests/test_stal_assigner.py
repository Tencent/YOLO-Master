# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest
import torch

from ultralytics.cfg import check_cfg, get_cfg
from ultralytics.utils.tal import TaskAlignedAssigner


def _candidate_fixture():
    """Return anchors and a 10x8 box that does not trigger the locked baseline's <8 pixel rule."""
    anchors = torch.tensor([[4.0, 4.0], [12.0, 4.0], [20.0, 4.0]])
    boxes = torch.tensor([[[7.0, 0.0, 17.0, 8.0]]])
    valid = torch.ones((1, 1, 1))
    return anchors, boxes, valid


def test_stal_disabled_preserves_locked_candidate_selection():
    """Explicitly disabling STAL must match the locked baseline's default behavior."""
    anchors, boxes, valid = _candidate_fixture()
    baseline = TaskAlignedAssigner(stride=[8, 16, 32])
    disabled = TaskAlignedAssigner(stride=[8, 16, 32], stal_enabled=False)

    expected = baseline.select_candidates_in_gts(anchors, boxes, valid)
    actual = disabled.select_candidates_in_gts(anchors, boxes, valid, image_size=(100, 100), epoch=10)

    assert torch.equal(actual, expected)
    assert actual.tolist() == [[[False, True, False]]]


def test_pure_tal_uses_unmodified_ground_truth_region():
    """Pure TAL must not inherit the repository's fixed-stride candidate expansion."""
    anchors = torch.tensor([[4.0, 4.0], [12.0, 4.0]])
    boxes = torch.tensor([[[7.0, 1.0, 9.0, 7.0]]])
    valid = torch.ones((1, 1, 1))

    candidates = TaskAlignedAssigner(stal_candidate_mode="pure").select_candidates_in_gts(anchors, boxes, valid)

    assert candidates.tolist() == [[[False, False]]]


def test_fixed_mode_preserves_locked_stride_expansion():
    """The default fixed mode must retain the locked repository behavior for sub-stride boxes."""
    anchors = torch.tensor([[4.0, 4.0], [12.0, 4.0]])
    boxes = torch.tensor([[[7.0, 1.0, 9.0, 7.0]]])
    valid = torch.ones((1, 1, 1))

    candidates = TaskAlignedAssigner(stal_candidate_mode="fixed").select_candidates_in_gts(anchors, boxes, valid)

    assert candidates.tolist() == [[[True, True]]]


def test_stal_area_rule_expands_candidate_region_after_warmup():
    """A relative-area target should admit extra anchors once symmetric relaxation is active."""
    anchors, boxes, valid = _candidate_fixture()
    assigner = TaskAlignedAssigner(
        stride=[8, 16, 32],
        stal_enabled=True,
        stal_area_threshold=0.01,
        stal_relaxation=8.0,
        stal_warmup_epochs=10,
    )

    candidates = assigner.select_candidates_in_gts(anchors, boxes, valid, image_size=(100, 100), epoch=10)

    assert candidates.tolist() == [[[True, True, True]]]


def test_stal_trigger_uses_original_area_before_locked_width_expansion():
    """The locked <stride expansion must not change whether a target satisfies the STAL area threshold."""
    anchors = torch.tensor([[0.0, 20.0], [10.0, 20.0], [20.0, 20.0]])
    boxes = torch.tensor([[[9.0, 0.0, 11.0, 40.0]]])  # 80 / 10_000 = 0.8%
    valid = torch.ones((1, 1, 1))
    assigner = TaskAlignedAssigner(
        stride=[8, 16, 32],
        stal_enabled=True,
        stal_area_threshold=0.01,
        stal_relaxation=8.0,
        stal_warmup_epochs=0,
    )

    candidates = assigner.select_candidates_in_gts(anchors, boxes, valid, image_size=(100, 100), epoch=0)

    assert candidates.tolist() == [[[True, True, True]]]


def test_stal_area_threshold_is_strict_at_boundary():
    """A target exactly at the configured threshold must remain outside the adaptive group."""
    anchors, boxes, valid = _candidate_fixture()  # 10 * 8 / (100 * 80) == 1%
    assigner = TaskAlignedAssigner(
        stal_candidate_mode="adaptive", stal_area_threshold=0.01, stal_relaxation=8.0, stal_warmup_epochs=0
    )

    candidates = assigner.select_candidates_in_gts(anchors, boxes, valid, image_size=(100, 80), epoch=0)

    assert candidates.tolist() == [[[False, True, False]]]


def test_adaptive_mode_keeps_extremely_small_valid_box_finite():
    """A positive-area subpixel GT should remain valid and gain a bounded candidate region."""
    anchors = torch.tensor([[4.0, 4.0], [12.0, 4.0]])
    boxes = torch.tensor([[[7.9, 3.9, 8.1, 4.1]]])
    valid = torch.ones((1, 1, 1))
    assigner = TaskAlignedAssigner(stal_candidate_mode="adaptive", stal_relaxation=8.0, stal_warmup_epochs=0)

    candidates = assigner.select_candidates_in_gts(anchors, boxes, valid, image_size=(64, 64), epoch=0)

    assert candidates.dtype == torch.bool
    assert candidates.any()


def test_stal_warmup_is_linear_and_capped():
    """Relaxation should start at zero, increase linearly, and stop at the configured maximum."""
    assigner = TaskAlignedAssigner(stal_enabled=True, stal_relaxation=12.0, stal_warmup_epochs=6)

    assert assigner.stal_relaxation_at_epoch(0) == pytest.approx(0.0)
    assert assigner.stal_relaxation_at_epoch(2) == pytest.approx(4.0)
    assert assigner.stal_relaxation_at_epoch(6) == pytest.approx(12.0)
    assert assigner.stal_relaxation_at_epoch(10) == pytest.approx(12.0)


def test_minimum_stride_size_is_warmup_scaled_and_uses_the_smallest_stride():
    """The minimum candidate size should track both warmup progress and the model's minimum stride."""
    assigner = TaskAlignedAssigner(
        stride=[16, 4, 8],
        stal_candidate_mode="adaptive",
        stal_relaxation=0.0,
        stal_min_size_stride_ratio=1.5,
        stal_warmup_epochs=10,
    )

    assert assigner.stal_min_candidate_size_at_epoch(0) == pytest.approx(0.0)
    assert assigner.stal_min_candidate_size_at_epoch(5) == pytest.approx(3.0)
    assert assigner.stal_min_candidate_size_at_epoch(10) == pytest.approx(6.0)


def test_minimum_stride_size_expands_candidates_without_changing_regression_gt():
    """A small GT should gain grid candidates while its caller-owned regression box remains unchanged."""
    anchors = torch.tensor([[3.0, 10.0], [10.0, 10.0], [17.0, 10.0]])
    boxes = torch.tensor([[[5.0, 5.0, 15.0, 15.0]]])
    original = boxes.clone()
    valid = torch.ones((1, 1, 1))
    assigner = TaskAlignedAssigner(
        stride=[8, 16, 32],
        stal_candidate_mode="adaptive",
        stal_area_threshold=0.02,
        stal_relaxation=0.0,
        stal_min_size_stride_ratio=2.0,
        stal_warmup_epochs=0,
    )

    candidates = assigner.select_candidates_in_gts(anchors, boxes, valid, image_size=(100, 100), epoch=0)

    assert candidates.tolist() == [[[True, True, True]]]
    assert torch.equal(boxes, original)


def test_zero_minimum_stride_size_preserves_unexpanded_adaptive_candidates():
    """The disabled minimum-size guarantee must not change an r0 adaptive candidate mask."""
    anchors, boxes, valid = _candidate_fixture()
    baseline = TaskAlignedAssigner(stal_candidate_mode="adaptive", stal_relaxation=0.0, stal_min_size_stride_ratio=0.0)

    candidates = baseline.select_candidates_in_gts(anchors, boxes, valid, image_size=(100, 100), epoch=10)

    assert candidates.tolist() == [[[False, True, False]]]


def test_area_scaled_relaxation_preserves_warmup_and_interpolates_between_bounds():
    """Both area-dependent bounds should warm up together before reaching their configured values."""
    assigner = TaskAlignedAssigner(
        stal_candidate_mode="adaptive",
        stal_area_threshold=0.01,
        stal_relaxation=8.0,
        stal_relaxation_scale_mode="sqrt_area",
        stal_relaxation_min=4.0,
        stal_warmup_epochs=10,
    )
    relative_area = torch.tensor([0.0, 0.0025, 0.01])

    assert assigner.stal_relaxation_for_area(relative_area, epoch=0).tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert assigner.stal_relaxation_for_area(relative_area, epoch=5).tolist() == pytest.approx([4.0, 3.0, 2.0])
    assert assigner.stal_relaxation_for_area(relative_area, epoch=10).tolist() == pytest.approx([8.0, 6.0, 4.0])


def test_default_constant_relaxation_matches_explicit_legacy_behavior():
    """The new area-scaling controls must leave existing r8 configurations unchanged by default."""
    anchors, boxes, valid = _candidate_fixture()
    common = {
        "stal_candidate_mode": "adaptive",
        "stal_area_threshold": 0.01,
        "stal_relaxation": 8.0,
        "stal_warmup_epochs": 10,
    }

    baseline = TaskAlignedAssigner(**common).select_candidates_in_gts(
        anchors, boxes, valid, image_size=(100, 100), epoch=5
    )
    explicit = TaskAlignedAssigner(
        **common, stal_relaxation_scale_mode="constant", stal_relaxation_min=0.0
    ).select_candidates_in_gts(anchors, boxes, valid, image_size=(100, 100), epoch=5)

    assert torch.equal(explicit, baseline)


def test_area_scaled_relaxation_gives_tinier_target_more_candidate_coverage():
    """A tinier target should receive more expansion than a target near the adaptive-area threshold."""
    assigner = TaskAlignedAssigner(
        stride=[4],
        stal_candidate_mode="adaptive",
        stal_area_threshold=0.01,
        stal_relaxation=8.0,
        stal_relaxation_scale_mode="sqrt_area",
        stal_relaxation_min=0.0,
        stal_warmup_epochs=0,
    )
    anchors = torch.tensor([[5.0, 10.0], [10.0, 10.0], [15.0, 10.0]])
    boxes = torch.tensor([[[9.0, 9.0, 11.0, 11.0], [5.1, 5.1, 14.9, 14.9]]])
    valid = torch.ones((1, 2, 1))

    candidates = assigner.select_candidates_in_gts(anchors, boxes, valid, image_size=(100, 100), epoch=0)

    assert candidates.sum(-1).tolist() == [[3, 1]]


def test_adaptive_small_topk_only_expands_small_target_selection():
    """Adaptive top-k should select more small-target candidates without changing a large target's top-k."""
    assigner = TaskAlignedAssigner(
        topk=2,
        num_classes=1,
        stal_candidate_mode="adaptive",
        stal_area_threshold=0.01,
        stal_small_topk=4,
        stal_relaxation=0,
    )
    anchors = torch.tensor([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [1.0, 2.0]])
    boxes = torch.tensor([[[0.0, 0.0, 5.0, 5.0], [0.0, 0.0, 80.0, 80.0]]])
    predicted = torch.tensor([[[0.0, 0.0, 5.0, 5.0]] * 5])
    scores = torch.full((1, 5, 1), 0.9)
    labels = torch.zeros((1, 2, 1))
    valid = torch.ones((1, 2, 1), dtype=torch.bool)
    assigner.bs = 1
    assigner.n_max_boxes = 2

    mask_pos, _, _, _, _ = assigner.get_pos_mask(
        scores, predicted, labels, boxes, anchors, valid, image_size=(100, 100)
    )
    baseline = TaskAlignedAssigner(
        topk=2,
        num_classes=1,
        stal_candidate_mode="adaptive",
        stal_area_threshold=0.01,
        stal_small_topk=2,
        stal_relaxation=0,
    )
    baseline.bs = 1
    baseline.n_max_boxes = 2
    baseline_mask, _, _, _, _ = baseline.get_pos_mask(
        scores, predicted, labels, boxes, anchors, valid, image_size=(100, 100)
    )

    assert mask_pos.sum(-1)[0, 0].item() == 4
    assert baseline_mask.sum(-1)[0, 0].item() == 2
    assert mask_pos[0, 1].equal(baseline_mask[0, 1])


def test_area_adaptive_topk_uses_fewer_slots_for_tinier_targets():
    """Square-root area interpolation should vary small-target top-k while leaving its bounds explicit."""
    assigner = TaskAlignedAssigner(
        topk=10,
        num_classes=1,
        stal_candidate_mode="adaptive",
        stal_area_threshold=0.01,
        stal_small_topk=10,
        stal_small_topk_min=2,
        stal_relaxation=0,
    )
    anchors = torch.tensor([[float(i), 5.0] for i in range(1, 11)])
    scores = torch.full((1, 10, 1), 0.9)
    labels = torch.zeros((1, 1, 1))
    valid = torch.ones((1, 1, 1), dtype=torch.bool)
    assigner.bs, assigner.n_max_boxes = 1, 1

    tiny_box = torch.tensor([[[0.0, 0.0, 10.0, 10.0]]])
    tiny_predictions = tiny_box.expand(1, 10, 4).clone()
    tiny_mask = assigner.get_pos_mask(
        scores, tiny_predictions, labels, tiny_box, anchors, valid, image_size=(1000, 1000)
    )[0]

    near_threshold_box = torch.tensor([[[0.0, 0.0, 90.0, 90.0]]])
    near_predictions = near_threshold_box.expand(1, 10, 4).clone()
    near_mask = assigner.get_pos_mask(
        scores, near_predictions, labels, near_threshold_box, anchors, valid, image_size=(1000, 1000)
    )[0]

    assert tiny_mask.sum().item() == 3
    assert near_mask.sum().item() == 9


def test_variable_topk_mask_does_not_erase_real_anchor_zero():
    """Disabled top-k slots must not collide with a legitimately selected candidate at index zero."""
    assigner = TaskAlignedAssigner(topk=3, num_classes=1)
    metrics = torch.tensor([[[0.9, 0.8, 0.7]]])
    topk_mask = torch.tensor([[[True, False, False]]])

    selected = assigner.select_topk_candidates(metrics, topk_mask=topk_mask)

    assert selected.tolist() == [[[1.0, 0.0, 0.0]]]


def test_expanded_quality_gate_preserves_base_candidate_and_rejects_weak_extension():
    """Relative quality gating must affect only weak candidates added by adaptive relaxation."""
    common = {
        "topk": 2,
        "num_classes": 1,
        "stal_candidate_mode": "adaptive",
        "stal_area_threshold": 0.5,
        "stal_relaxation": 8.0,
        "stal_warmup_epochs": 0,
    }
    anchors = torch.tensor([[5.0, 5.0], [12.0, 5.0]])
    boxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0]]])
    predicted = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [7.0, 0.0, 17.0, 10.0]]])
    scores = torch.full((1, 2, 1), 0.9)
    labels = torch.zeros((1, 1, 1))
    valid = torch.ones((1, 1, 1), dtype=torch.bool)

    def positive_mask(ratio):
        assigner = TaskAlignedAssigner(**common, stal_expanded_quality_ratio=ratio)
        assigner.bs, assigner.n_max_boxes = 1, 1
        return assigner.get_pos_mask(scores, predicted, labels, boxes, anchors, valid, image_size=(100, 100))[0]

    assert positive_mask(0.0).tolist() == [[[1.0, 1.0]]]
    assert positive_mask(0.5).tolist() == [[[1.0, 0.0]]]


def test_expanded_quality_gate_keeps_coverage_when_no_base_candidate_exists():
    """A target with no pre-relaxation grid point must retain its adaptive candidates."""
    common = {
        "topk": 2,
        "num_classes": 1,
        "stal_candidate_mode": "adaptive",
        "stal_area_threshold": 0.5,
        "stal_relaxation": 8.0,
        "stal_warmup_epochs": 0,
    }
    anchors = torch.tensor([[8.0, 15.0], [22.0, 15.0]])
    boxes = torch.tensor([[[10.0, 10.0, 20.0, 20.0]]])
    predicted = torch.tensor([[[7.0, 10.0, 17.0, 20.0], [13.0, 10.0, 23.0, 20.0]]])
    scores = torch.full((1, 2, 1), 0.9)
    labels = torch.zeros((1, 1, 1))
    valid = torch.ones((1, 1, 1), dtype=torch.bool)

    masks = []
    for ratio in (0.0, 0.5):
        assigner = TaskAlignedAssigner(**common, stal_expanded_quality_ratio=ratio)
        assigner.bs, assigner.n_max_boxes = 1, 1
        masks.append(assigner.get_pos_mask(scores, predicted, labels, boxes, anchors, valid, image_size=(100, 100))[0])

    assert torch.equal(masks[0], masks[1])
    assert masks[1].sum().item() == 2


def test_expanded_quality_gate_rejects_zero_alignment_extensions_when_base_is_zero():
    """A zero base metric must not make every zero-quality extension pass a relative gate."""
    assigner = TaskAlignedAssigner(
        topk=2,
        num_classes=1,
        stride=[1, 1, 1],
        stal_candidate_mode="adaptive",
        stal_area_threshold=0.5,
        stal_relaxation=8.0,
        stal_warmup_epochs=0,
        stal_expanded_quality_ratio=0.5,
    )
    assigner.bs = assigner.n_max_boxes = 1
    anchors = torch.tensor([[5.0, 5.0], [12.0, 5.0]])
    boxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0]]])
    predicted = torch.tensor([[[100.0, 100.0, 110.0, 110.0], [100.0, 100.0, 110.0, 110.0]]])
    scores = torch.full((1, 2, 1), 0.9)
    labels = torch.zeros((1, 1, 1))
    valid = torch.ones((1, 1, 1), dtype=torch.bool)

    mask_pos, align_metric, _, mask_in_gts, _ = assigner.get_pos_mask(
        scores, predicted, labels, boxes, anchors, valid, image_size=(100, 100)
    )

    assert not align_metric.any()
    assert mask_in_gts.tolist() == [[[True, False]]]
    assert mask_pos.tolist() == [[[1.0, 0.0]]]


def test_expanded_score_weight_continuously_attenuates_only_adaptive_candidate():
    """Adaptive-only supervision should be softened while base-candidate supervision stays unchanged."""
    common = {
        "topk": 2,
        "num_classes": 1,
        "stal_candidate_mode": "adaptive",
        "stal_area_threshold": 0.5,
        "stal_relaxation": 8.0,
        "stal_warmup_epochs": 0,
    }
    anchors = torch.tensor([[5.0, 5.0], [12.0, 5.0]])
    boxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0]]])
    predicted = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [7.0, 0.0, 17.0, 10.0]]])
    scores = torch.full((1, 2, 1), 0.9)
    labels = torch.zeros((1, 1, 1))
    valid = torch.ones((1, 1, 1), dtype=torch.bool)

    baseline = TaskAlignedAssigner(**common)
    softened = TaskAlignedAssigner(**common, stal_expanded_score_floor=0.25)
    baseline_output = baseline(scores, predicted, anchors, labels, boxes, valid, image_size=(100, 100))
    softened_output = softened(scores, predicted, anchors, labels, boxes, valid, image_size=(100, 100))
    baseline_scores, softened_scores = baseline_output[2], softened_output[2]

    assert torch.equal(softened_output[3], baseline_output[3])
    assert torch.equal(softened_output[4], baseline_output[4])
    assert softened_scores[0, 0].item() == pytest.approx(baseline_scores[0, 0].item())
    assert 0.25 * baseline_scores[0, 1].item() <= softened_scores[0, 1].item() < baseline_scores[0, 1].item()


def test_expanded_score_weight_keeps_full_weight_when_gt_has_no_base_candidate():
    """Coverage-only adaptive candidates must not be weakened when no base candidate exists."""
    common = {
        "topk": 2,
        "num_classes": 1,
        "stal_candidate_mode": "adaptive",
        "stal_area_threshold": 0.5,
        "stal_relaxation": 8.0,
        "stal_warmup_epochs": 0,
    }
    anchors = torch.tensor([[8.0, 15.0], [22.0, 15.0]])
    boxes = torch.tensor([[[10.0, 10.0, 20.0, 20.0]]])
    predicted = torch.tensor([[[7.0, 10.0, 17.0, 20.0], [13.0, 10.0, 23.0, 20.0]]])
    scores = torch.full((1, 2, 1), 0.9)
    labels = torch.zeros((1, 1, 1))
    valid = torch.ones((1, 1, 1), dtype=torch.bool)

    baseline = TaskAlignedAssigner(**common)
    softened = TaskAlignedAssigner(**common, stal_expanded_score_floor=0.25)
    baseline_output = baseline(scores, predicted, anchors, labels, boxes, valid, image_size=(100, 100))
    softened_output = softened(scores, predicted, anchors, labels, boxes, valid, image_size=(100, 100))

    assert torch.equal(softened_output[2], baseline_output[2])


@pytest.mark.parametrize("extra_limit", [1, 2])
def test_extra_candidate_limit_preserves_base_tal_and_caps_adaptive_additions(extra_limit):
    """Limited supplementation must keep base selections and add no more than the configured number."""
    anchors = torch.tensor([[2.0, 5.0], [5.0, 5.0], [8.0, 5.0], [12.0, 5.0], [14.0, 5.0]])
    boxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0]]])
    predicted = boxes.expand(1, 5, 4).clone()
    scores = torch.tensor([[[0.5], [0.9], [0.8], [0.7], [0.6]]])
    labels = torch.zeros((1, 1, 1))
    valid = torch.ones((1, 1, 1), dtype=torch.bool)
    assigner = TaskAlignedAssigner(
        topk=2,
        num_classes=1,
        stal_candidate_mode="adaptive",
        stal_area_threshold=0.5,
        stal_relaxation=10.0,
        stal_warmup_epochs=0,
        stal_max_extra_candidates=extra_limit,
        stride=[1, 1, 1],
    )
    assigner.bs = assigner.n_max_boxes = 1

    mask = assigner.get_pos_mask(scores, predicted, labels, boxes, anchors, valid, image_size=(100, 100))[0]

    assert mask[0, 0, 1].item() == 1
    assert mask[0, 0, 2].item() == 1
    assert mask.sum().item() == 2 + extra_limit


def test_minimum_base_candidate_trigger_expands_only_undercovered_small_targets():
    """Coverage-triggered adaptive STAL should leave an already covered small GT at its base candidate set."""
    anchors = torch.tensor([[5.0, 5.0], [12.0, 5.0], [28.0, 5.0], [42.0, 5.0]])
    boxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [30.0, 0.0, 40.0, 10.0]]])
    valid = torch.ones((1, 2, 1), dtype=torch.bool)
    common = {
        "stal_candidate_mode": "adaptive",
        "stal_area_threshold": 0.5,
        "stal_relaxation": 8.0,
        "stal_warmup_epochs": 0,
    }

    all_small = TaskAlignedAssigner(**common).select_candidates_in_gts(anchors, boxes, valid, image_size=(100, 100))
    undercovered_only = TaskAlignedAssigner(**common, stal_min_base_candidates=1).select_candidates_in_gts(
        anchors, boxes, valid, image_size=(100, 100)
    )

    assert all_small.tolist() == [[[True, True, False, False], [False, False, True, True]]]
    assert undercovered_only.tolist() == [[[True, False, False, False], [False, False, True, True]]]


def test_minimum_candidate_guarantee_adds_only_nearest_point_for_candidate_free_small_gt():
    """A candidate-free small GT should gain exactly its nearest point, while a covered GT stays unchanged."""
    anchors = torch.tensor([[4.0, 4.0], [12.0, 4.0], [20.0, 4.0]])
    boxes = torch.tensor([[[7.0, 1.0, 9.0, 3.0], [11.0, 3.0, 13.0, 5.0]]])
    valid = torch.ones(1, 2, 1, dtype=torch.bool)
    assigner = TaskAlignedAssigner(
        stal_candidate_mode="adaptive",
        stal_relaxation=0.0,
        stal_min_candidate_guarantee=True,
        stride=[1, 1, 1],
    )

    candidates = assigner.select_candidates_in_gts(anchors, boxes, valid, image_size=(100, 100))

    assert candidates.tolist() == [[[True, False, False], [False, True, False]]]


def test_minimum_candidate_guarantee_ignores_invalid_and_non_small_gt():
    """The guarantee must not create candidates for padded rows or targets outside the configured small area."""
    anchors = torch.tensor([[4.0, 4.0], [12.0, 4.0]])
    boxes = torch.tensor([[[7.0, 1.0, 9.0, 3.0], [30.0, 30.0, 80.0, 80.0]]])
    valid = torch.tensor([[[False], [True]]])
    assigner = TaskAlignedAssigner(
        stal_candidate_mode="adaptive",
        stal_relaxation=0.0,
        stal_min_candidate_guarantee=True,
        stride=[1, 1, 1],
    )

    candidates = assigner.select_candidates_in_gts(anchors, boxes, valid, image_size=(100, 100))

    assert not candidates.any()


def test_candidate_overlap_crowding_shrinks_only_overlapping_small_targets():
    """Crowded small GTs should use r4 while an isolated small GT preserves r8 candidates."""
    anchors = torch.tensor([[5.0, 5.0], [12.0, 5.0], [17.0, 5.0], [25.0, 5.0], [35.0, 5.0], [42.0, 5.0]])
    boxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [12.0, 0.0, 22.0, 10.0], [30.0, 0.0, 40.0, 10.0]]])
    valid = torch.ones((1, 3, 1), dtype=torch.bool)
    common = {
        "stal_candidate_mode": "adaptive",
        "stal_area_threshold": 0.5,
        "stal_relaxation": 8.0,
        "stal_warmup_epochs": 0,
    }

    r8 = TaskAlignedAssigner(**common).select_candidates_in_gts(anchors, boxes, valid, image_size=(100, 100))
    hybrid = TaskAlignedAssigner(
        **common, stal_crowding_mode="candidate_overlap", stal_crowded_relaxation=4.0
    ).select_candidates_in_gts(anchors, boxes, valid, image_size=(100, 100))

    assert hybrid[0, 0].sum() < r8[0, 0].sum()
    assert hybrid[0, 1].sum() < r8[0, 1].sum()
    assert torch.equal(hybrid[0, 2], r8[0, 2])


def test_stal_ignores_padded_ground_truth_rows():
    """Padded GT rows must never gain candidates even though their zero area is below the threshold."""
    anchors = torch.tensor([[4.0, 4.0], [12.0, 4.0]])
    boxes = torch.zeros((1, 1, 4))
    invalid = torch.zeros((1, 1, 1))
    assigner = TaskAlignedAssigner(stal_enabled=True, stal_relaxation=8.0, stal_warmup_epochs=0)

    candidates = assigner.select_candidates_in_gts(anchors, boxes, invalid, image_size=(100, 100), epoch=1)

    assert not candidates.any()


def test_stal_assignment_statistics_follow_foreground_to_gt_mapping():
    """STAL-target counts must use the same foreground-to-GT mapping returned by assignment."""
    assigner = TaskAlignedAssigner(stal_enabled=False, stal_area_threshold=0.01)
    boxes = torch.tensor([[[0.0, 0.0, 5.0, 5.0], [10.0, 10.0, 30.0, 30.0]]])
    valid = torch.ones((1, 2, 1), dtype=torch.bool)
    foreground = torch.tensor([[True, True, False, True, False, True]])
    target_gt_idx = torch.tensor([[0, 0, 0, 1, 1, 1]])

    stats = assigner.assignment_statistics(boxes, valid, foreground, target_gt_idx, image_size=(100, 100))

    assert stats.tolist() == [1, 2, 0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 2, 4, 0]


def test_stal_assignment_statistics_accept_empty_ground_truth_batches():
    """Background-only batches should contribute zero counts instead of indexing an empty GT dimension."""
    assigner = TaskAlignedAssigner(stal_enabled=False)
    stats = assigner.assignment_statistics(
        torch.empty((1, 0, 4)),
        torch.empty((1, 0, 1), dtype=torch.bool),
        torch.zeros((1, 3), dtype=torch.bool),
        torch.zeros((1, 3), dtype=torch.long),
        image_size=(100, 100),
    )

    assert stats.tolist() == [0] * 15


def test_stal_statistics_report_zero_positive_targets_by_size():
    """Mechanism logs must expose targets that receive no final positive after conflict resolution."""
    assigner = TaskAlignedAssigner(stal_area_threshold=0.01)
    boxes = torch.tensor([[[0.0, 0.0, 5.0, 5.0], [10.0, 10.0, 50.0, 50.0]]])
    valid = torch.ones((1, 2, 1), dtype=torch.bool)
    foreground = torch.tensor([[True, False]])
    target_gt_idx = torch.tensor([[0, 0]])

    stats = assigner.assignment_statistics(boxes, valid, foreground, target_gt_idx, image_size=(100, 100))

    assert stats.tolist() == [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 2, 1, 1]


def test_training_size_bins_use_coco_boundary_convention():
    """Exactly 32-square belongs to medium and exactly 96-square belongs to large."""
    assigner = TaskAlignedAssigner()
    boxes = torch.tensor([[[0.0, 0.0, 32.0, 32.0], [0.0, 0.0, 96.0, 96.0]]])
    valid = torch.ones((1, 2, 1), dtype=torch.bool)
    foreground = torch.zeros((1, 1), dtype=torch.bool)
    target_gt_idx = torch.zeros((1, 1), dtype=torch.long)

    stats = assigner.assignment_statistics(boxes, valid, foreground, target_gt_idx, image_size=(200, 200))

    assert stats[3:6].tolist() == [0, 0, 0]  # small
    assert stats[6:9].tolist() == [1, 0, 1]  # medium
    assert stats[9:12].tolist() == [1, 0, 1]  # large


def test_overlapping_ground_truth_conflict_assigns_each_anchor_once():
    """Overlapping GT candidates must resolve to one owner per foreground anchor."""
    assigner = TaskAlignedAssigner(topk=3, num_classes=1, stal_candidate_mode="adaptive", stal_warmup_epochs=0)
    scores = torch.tensor([[[0.9], [0.8], [0.7]]])
    predicted = torch.tensor([[[0.0, 0.0, 8.0, 8.0], [4.0, 0.0, 12.0, 8.0], [8.0, 0.0, 16.0, 8.0]]])
    anchors = torch.tensor([[4.0, 4.0], [8.0, 4.0], [12.0, 4.0]])
    labels = torch.zeros((1, 2, 1))
    boxes = torch.tensor([[[0.0, 0.0, 10.0, 8.0], [6.0, 0.0, 16.0, 8.0]]])
    valid = torch.ones((1, 2, 1), dtype=torch.bool)

    _, _, target_scores, foreground, target_gt_idx = assigner(
        scores, predicted, anchors, labels, boxes, valid, image_size=(32, 32), epoch=1
    )

    assert torch.isfinite(target_scores).all()
    assert foreground.shape == (1, 3)
    assert set(target_gt_idx[foreground].tolist()).issubset({0, 1})
    assert int(foreground.sum()) <= anchors.shape[0]


def test_conflict_resolution_cannot_assign_anchor_to_non_candidate_gt():
    """A higher-IoU GT that did not nominate an anchor must not acquire it during conflict resolution."""
    assigner = TaskAlignedAssigner(topk=3, num_classes=1)
    mask_pos = torch.tensor([[[1.0], [1.0], [0.0]]])
    overlaps = torch.tensor([[[0.2], [0.3], [0.9]]])
    align_metric = torch.tensor([[[0.2], [0.3], [0.9]]])

    target_gt_idx, foreground, resolved = assigner.select_highest_overlaps(
        mask_pos, overlaps, n_max_boxes=3, align_metric=align_metric
    )

    assert target_gt_idx.item() == 1
    assert foreground.item() == 1
    assert resolved[:, :, 0].tolist() == [[0.0, 1.0, 0.0]]
    assert torch.all(resolved <= mask_pos)


def test_assignment_stage_statistics_distinguish_candidate_alignment_topk_and_conflict_failures():
    """The diagnostic funnel should classify small-target failures without changing assignment tensors."""
    assigner = TaskAlignedAssigner(stal_area_threshold=0.5)
    boxes = torch.tensor([[[0.0, 0.0, 4.0, 4.0]] * 4])
    valid = torch.ones((1, 4, 1), dtype=torch.bool)
    legal = torch.tensor([[[0, 0], [1, 0], [1, 1], [1, 1]]], dtype=torch.bool)
    alignment = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [1e-10, 0.0], [0.7, 0.1]]])
    pre_conflict = torch.tensor([[[0, 0], [0, 0], [0, 0], [1, 0]]], dtype=torch.float32)
    post_conflict = torch.zeros_like(pre_conflict)

    assigner.record_assignment_stages(
        boxes, valid, legal, alignment, pre_conflict, post_conflict, image_size=(100, 100)
    )

    assert assigner.assignment_stage_statistics().tolist() == [4, 1, 1, 1, 1, 1, 1, 1, 4]


def test_assignment_stage_diagnostics_do_not_change_assignment_outputs():
    """Enabling stage counters must leave labels, boxes, scores, masks, and indices unchanged."""
    scores = torch.tensor([[[0.9], [0.4]]])
    predicted = torch.tensor([[[0.0, 0.0, 6.0, 6.0], [4.0, 0.0, 10.0, 6.0]]])
    anchors = torch.tensor([[3.0, 3.0], [7.0, 3.0]])
    labels = torch.zeros((1, 1, 1))
    boxes = torch.tensor([[[0.0, 0.0, 6.0, 6.0]]])
    valid = torch.ones((1, 1, 1), dtype=torch.bool)

    outputs = []
    for enabled in (False, True):
        assigner = TaskAlignedAssigner(topk=2, num_classes=1, stal_stats=enabled)
        outputs.append(assigner(scores, predicted, anchors, labels, boxes, valid, image_size=(32, 32)))

    assert all(torch.equal(left, right) for left, right in zip(*outputs))


def test_zero_positive_rescue_uses_best_free_legal_candidate_without_stealing():
    """Rescue should skip an occupied high-quality point and preserve an already covered GT."""
    assigner = TaskAlignedAssigner(
        topk=3, num_classes=1, stal_candidate_mode="adaptive", stal_zero_positive_rescue=True
    )
    mask_pos = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    align_metric = torch.tensor([[[0.2, 0.8, 0.9], [0.1, 0.7, 0.3]]])
    mask_in_gts = torch.tensor([[[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]])
    boxes = torch.tensor([[[0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 6.0, 6.0]]])
    valid = torch.ones((1, 2, 1), dtype=torch.bool)

    rescued = assigner.rescue_zero_positive_small_targets(
        mask_pos,
        align_metric,
        mask_in_gts,
        boxes,
        valid,
        image_size=(100, 100),
        available_anchors=torch.tensor([[True, False, True]]),
    )

    assert rescued.tolist() == [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
    assert torch.all(rescued <= mask_in_gts)


def test_zero_positive_rescue_rejects_zero_quality_and_non_small_candidates():
    """Rescue must not invent a positive from a zero-quality candidate or alter a non-small target."""
    assigner = TaskAlignedAssigner(stal_candidate_mode="adaptive", stal_zero_positive_rescue=True)
    mask_pos = torch.zeros((1, 2, 2))
    align_metric = torch.tensor([[[0.0, 0.0], [0.8, 0.7]]])
    mask_in_gts = torch.ones_like(mask_pos)
    boxes = torch.tensor([[[0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 40.0, 40.0]]])
    valid = torch.ones((1, 2, 1), dtype=torch.bool)

    rescued = assigner.rescue_zero_positive_small_targets(
        mask_pos, align_metric, mask_in_gts, boxes, valid, image_size=(100, 100)
    )

    assert torch.equal(rescued, mask_pos)


def test_zero_positive_rescue_survives_conflict_resolution_without_stealing():
    """End-to-end rescue should use a free fallback after two GTs initially select the same anchor."""
    scores = torch.tensor([[[0.9], [0.01]]])
    predicted = torch.tensor([[[0.0, 0.0, 6.0, 6.0], [2.0, 0.0, 10.0, 6.0]]])
    anchors = torch.tensor([[3.0, 3.0], [8.0, 3.0]])
    labels = torch.zeros((1, 2, 1))
    boxes = torch.tensor([[[0.0, 0.0, 6.0, 6.0], [0.0, 0.0, 10.0, 6.0]]])
    valid = torch.ones((1, 2, 1), dtype=torch.bool)

    def assigned(rescue):
        assigner = TaskAlignedAssigner(
            topk=1,
            num_classes=1,
            stal_candidate_mode="adaptive",
            stal_area_threshold=0.5,
            stal_relaxation=0,
            stal_zero_positive_rescue=rescue,
        )
        return assigner(scores, predicted, anchors, labels, boxes, valid, image_size=(100, 100), epoch=1)

    baseline = assigned(False)
    rescued = assigned(True)

    assert baseline[3].tolist() == [[True, False]]
    assert rescued[3].tolist() == [[True, True]]
    assert rescued[4].tolist() == [[0, 1]]


def test_zero_positive_rescue_reports_static_same_input_effect_and_failure_stages():
    """The same tensors should expose one successful rescue and one rescue-candidate collision."""
    scores = torch.tensor([[[0.9], [0.2], [0.1]]])
    predicted = torch.tensor([[[0.0, 0.0, 6.0, 6.0], [4.0, 0.0, 10.0, 6.0], [20.0, 20.0, 24.0, 24.0]]])
    anchors = torch.tensor([[3.0, 3.0], [7.0, 3.0], [22.0, 22.0]])
    labels = torch.zeros((1, 3, 1))
    boxes = torch.tensor([[[0.0, 0.0, 6.0, 6.0], [0.0, 0.0, 10.0, 6.0], [0.0, 0.0, 6.0, 6.0]]])
    valid = torch.ones((1, 3, 1), dtype=torch.bool)

    def assigned(rescue):
        assigner = TaskAlignedAssigner(
            topk=1,
            num_classes=1,
            stal_candidate_mode="adaptive",
            stal_area_threshold=0.5,
            stal_relaxation=0,
            stal_zero_positive_rescue=rescue,
        )
        output = assigner(scores, predicted, anchors, labels, boxes, valid, image_size=(100, 100), epoch=1)
        return output, assigner.rescue_statistics()

    baseline, baseline_stats = assigned(False)
    rescued, rescue_stats = assigned(True)

    assert baseline[3].sum() == 1
    assert rescued[3].sum() == 2
    assert baseline_stats.tolist() == [0] * 9
    # attempted, legal, free, positive-quality, proposed, succeeded, conflict-lost, bootstrap proposed/succeeded
    assert rescue_stats.tolist() == [2, 2, 2, 2, 2, 1, 1, 0, 0]


def test_zero_quality_bootstrap_uses_nearest_free_candidate_and_nonzero_score_floor():
    """A zero-CIoU small GT should receive one center-prior candidate with a controlled supervision weight."""
    assigner = TaskAlignedAssigner(
        topk=1,
        num_classes=1,
        stal_candidate_mode="adaptive",
        stal_area_threshold=0.5,
        stal_relaxation=0,
        stal_zero_positive_rescue=True,
        stal_rescue_score_floor=0.2,
    )
    scores = torch.tensor([[[0.9], [0.9]]])
    predicted = torch.tensor([[[0.0, 0.0, 6.0, 6.0], [40.0, 40.0, 44.0, 44.0]]])
    anchors = torch.tensor([[3.0, 3.0], [8.0, 3.0]])
    labels = torch.zeros((1, 2, 1))
    boxes = torch.tensor([[[0.0, 0.0, 6.0, 6.0], [0.0, 0.0, 10.0, 6.0]]])
    valid = torch.ones((1, 2, 1), dtype=torch.bool)

    _, _, target_scores, foreground, target_gt_idx = assigner(
        scores, predicted, anchors, labels, boxes, valid, image_size=(100, 100), epoch=0
    )

    assert foreground.tolist() == [[True, True]]
    assert target_gt_idx[foreground].tolist() == [0, 1]
    assert target_scores[0, 1, 0].item() == pytest.approx(0.2)
    assert torch.isfinite(target_scores).all()
    assert assigner.rescue_statistics().tolist() == [1, 1, 1, 0, 1, 1, 0, 1, 1]


def test_pure_tal_can_apply_rescue_without_candidate_expansion():
    """Rescue-only mode should preserve pure candidates and add one uncovered small-target anchor."""
    assigner = TaskAlignedAssigner(
        topk=1,
        num_classes=1,
        stal_candidate_mode="pure",
        stal_area_threshold=0.5,
        stal_zero_positive_rescue=True,
        stal_rescue_score_floor=0.01,
    )
    scores = torch.tensor([[[0.9], [0.9]]])
    predicted = torch.tensor([[[0.0, 0.0, 6.0, 6.0], [40.0, 40.0, 44.0, 44.0]]])
    anchors = torch.tensor([[3.0, 3.0], [8.0, 3.0]])
    labels = torch.zeros((1, 2, 1))
    boxes = torch.tensor([[[0.0, 0.0, 6.0, 6.0], [0.0, 0.0, 10.0, 6.0]]])
    valid = torch.ones((1, 2, 1), dtype=torch.bool)

    _, _, target_scores, foreground, target_gt_idx = assigner(
        scores, predicted, anchors, labels, boxes, valid, image_size=(100, 100), epoch=0
    )

    assert foreground.tolist() == [[True, True]]
    assert target_gt_idx[foreground].tolist() == [0, 1]
    assert target_scores[0, 1, 0].item() == pytest.approx(0.01)


def test_bootstrap_score_floor_decays_linearly_and_stops_at_zero():
    """Bootstrap supervision should be strongest initially and disappear after its configured decay."""
    assigner = TaskAlignedAssigner(
        stal_candidate_mode="adaptive",
        stal_zero_positive_rescue=True,
        stal_rescue_score_floor=0.2,
        stal_rescue_floor_decay_epochs=4,
    )

    assert assigner.stal_rescue_score_floor_at_epoch(0) == pytest.approx(0.2)
    assert assigner.stal_rescue_score_floor_at_epoch(2) == pytest.approx(0.1)
    assert assigner.stal_rescue_score_floor_at_epoch(4) == pytest.approx(0.0)
    assert assigner.stal_rescue_score_floor_at_epoch(8) == pytest.approx(0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for AMP bootstrap-rescue coverage")
def test_bootstrap_rescue_is_finite_and_assignment_stable_in_fp32_and_amp():
    """The bootstrap candidate and its nonzero target score must survive autocast unchanged."""
    device = torch.device("cuda")
    assigner = TaskAlignedAssigner(
        topk=1,
        num_classes=1,
        stal_candidate_mode="adaptive",
        stal_area_threshold=0.5,
        stal_relaxation=0,
        stal_zero_positive_rescue=True,
        stal_rescue_score_floor=0.2,
    ).to(device)
    anchors = torch.tensor([[3.0, 3.0], [8.0, 3.0]], device=device)
    predicted = torch.tensor([[[0.0, 0.0, 6.0, 6.0], [40.0, 40.0, 44.0, 44.0]]], device=device)
    labels = torch.zeros((1, 2, 1), device=device)
    boxes = torch.tensor([[[0.0, 0.0, 6.0, 6.0], [0.0, 0.0, 10.0, 6.0]]], device=device)
    valid = torch.ones((1, 2, 1), dtype=torch.bool, device=device)

    outputs = []
    for amp in (False, True):
        logits = torch.tensor([[[2.0], [1.0]]], device=device, requires_grad=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            assigned = assigner(logits.detach().sigmoid(), predicted, anchors, labels, boxes, valid, (100, 100), 0)
            target_scores, foreground, indices = assigned[2], assigned[3], assigned[4]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target_scores.to(logits.dtype))
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(logits.grad).all()
        assert target_scores[0, 1, 0].item() == pytest.approx(0.2)
        outputs.append((foreground, indices, assigner.rescue_statistics()))

    assert torch.equal(outputs[0][0], outputs[1][0])
    assert torch.equal(outputs[0][1], outputs[1][1])
    assert torch.equal(outputs[0][2], outputs[1][2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FP16 autocast assignment coverage")
def test_fp32_and_amp_assignment_masks_counts_loss_and_gradients_are_finite():
    """FP32 and AMP should keep assignment discrete outputs stable and downstream gradients finite."""
    device = torch.device("cuda")
    assigner = TaskAlignedAssigner(
        topk=3, num_classes=1, stal_candidate_mode="adaptive", stal_relaxation=8.0, stal_warmup_epochs=0
    ).to(device)
    anchors = torch.tensor([[4.0, 4.0], [8.0, 4.0], [12.0, 4.0]], device=device)
    predicted = torch.tensor([[[0.0, 0.0, 8.0, 8.0], [4.0, 0.0, 12.0, 8.0], [8.0, 0.0, 16.0, 8.0]]], device=device)
    labels = torch.zeros((1, 1, 1), device=device)
    boxes = torch.tensor([[[6.0, 1.0, 10.0, 7.0]]], device=device)
    valid = torch.ones((1, 1, 1), dtype=torch.bool, device=device)

    outputs = []
    for amp in (False, True):
        logits = torch.tensor([[[2.0], [1.0], [0.0]]], device=device, requires_grad=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            assigned = assigner(logits.detach().sigmoid(), predicted, anchors, labels, boxes, valid, (32, 32), 1)
            target_scores, foreground, indices = assigned[2], assigned[3], assigned[4]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target_scores.to(logits.dtype))
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(logits.grad).all()
        outputs.append((foreground, indices, int(foreground.sum())))

    assert torch.equal(outputs[0][0], outputs[1][0])
    assert torch.equal(outputs[0][1], outputs[1][1])
    assert outputs[0][2] == outputs[1][2]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"stal_area_threshold": 0.0}, "stal_area_threshold"),
        ({"stal_area_threshold": 1.1}, "stal_area_threshold"),
        ({"stal_small_topk": 0}, "stal_small_topk"),
        ({"stal_small_topk_min": 0}, "stal_small_topk_min"),
        ({"stal_small_topk": 4, "stal_small_topk_min": 5}, "stal_small_topk_min"),
        ({"stal_min_base_candidates": -1}, "stal_min_base_candidates"),
        ({"stal_max_extra_candidates": -1}, "stal_max_extra_candidates"),
        ({"stal_relaxation": -1.0}, "stal_relaxation"),
        ({"stal_min_size_stride_ratio": -1.0}, "stal_min_size_stride_ratio"),
        ({"stal_min_size_stride_ratio": 2.0}, "stal_min_size_stride_ratio"),
        (
            {
                "stal_relaxation": 0.0,
                "stal_relaxation_scale_mode": "sqrt_area",
                "stal_min_size_stride_ratio": 2.0,
            },
            "stal_min_size_stride_ratio",
        ),
        ({"stal_relaxation_scale_mode": "unknown"}, "stal_relaxation_scale_mode"),
        ({"stal_relaxation_min": -1.0}, "stal_relaxation_min"),
        ({"stal_relaxation": 4.0, "stal_relaxation_min": 5.0}, "stal_relaxation_min"),
        ({"stal_warmup_epochs": -1}, "stal_warmup_epochs"),
        ({"stal_rescue_score_floor": -0.1}, "stal_rescue_score_floor"),
        ({"stal_rescue_score_floor": 1.1}, "stal_rescue_score_floor"),
        ({"stal_rescue_floor_decay_epochs": -1}, "stal_rescue_floor_decay_epochs"),
        ({"stal_nwd_weight": -0.1}, "stal_nwd_weight"),
        ({"stal_nwd_weight": 1.1}, "stal_nwd_weight"),
        ({"stal_nwd_constant": 0.0}, "stal_nwd_constant"),
        ({"stal_nwd_target_mode": "unknown"}, "stal_nwd_target_mode"),
        ({"stal_expanded_quality_ratio": -0.1}, "stal_expanded_quality_ratio"),
        ({"stal_expanded_quality_ratio": 1.1}, "stal_expanded_quality_ratio"),
        ({"stal_expanded_score_floor": -0.1}, "stal_expanded_score_floor"),
        ({"stal_expanded_score_floor": 1.1}, "stal_expanded_score_floor"),
        ({"stal_candidate_mode": "unknown"}, "stal_candidate_mode"),
    ],
)
def test_stal_rejects_invalid_configuration(kwargs, match):
    """Invalid settings should fail before a long training job starts."""
    with pytest.raises(ValueError, match=match):
        TaskAlignedAssigner(stal_enabled=True, **kwargs)


def test_stal_defaults_are_disabled_and_typed():
    """Global defaults must preserve the locked baseline while exposing CLI-compatible values."""
    cfg = get_cfg()

    assert cfg.stal_enabled is False
    assert cfg.stal_stats is False
    assert cfg.stal_candidate_mode == "fixed"
    assert cfg.stal_area_threshold == pytest.approx(0.01)
    assert cfg.stal_small_topk == 10
    assert cfg.stal_min_candidate_guarantee is False
    assert cfg.stal_max_extra_candidates == 0
    assert cfg.stal_expanded_score_floor == pytest.approx(1.0)
    assert cfg.stal_relaxation == pytest.approx(8.0)
    assert cfg.stal_relaxation_scale_mode == "constant"
    assert cfg.stal_relaxation_min == pytest.approx(0.0)
    assert cfg.stal_min_size_stride_ratio == pytest.approx(0.0)
    assert cfg.stal_warmup_epochs == pytest.approx(10.0)
    assert cfg.stal_zero_positive_rescue is False
    assert cfg.stal_rescue_score_floor == pytest.approx(0.0)
    assert cfg.stal_rescue_floor_decay_epochs == pytest.approx(0.0)
    assert cfg.stal_nwd_weight == pytest.approx(0.0)
    assert cfg.stal_nwd_constant == pytest.approx(12.8)
    assert cfg.stal_nwd_target_mode == "match"
    assert cfg.stal_nwd_target_weight == pytest.approx(0.0)
    assert cfg.stal_nwd_zero_score_floor == pytest.approx(0.0)


def test_nwd_is_less_brittle_than_ciou_for_one_pixel_tiny_box_shift():
    """NWD should retain useful quality when a one-pixel shift sharply reduces tiny-box CIoU."""
    assigner = TaskAlignedAssigner(stal_nwd_weight=1.0, stal_nwd_constant=12.8)
    gt = torch.tensor([[10.0, 10.0, 12.0, 12.0]])
    shifted = torch.tensor([[11.0, 10.0, 13.0, 12.0]])

    ciou = assigner.iou_calculation(gt, shifted)
    nwd = assigner.nwd_similarity(gt, shifted)

    assert 0.0 < ciou.item() < nwd.item() < 1.0


def test_nwd_blend_is_size_gated_and_default_quality_is_unchanged():
    """Only small GT quality should change; weight zero must exactly preserve existing TAL quality."""
    scores = torch.full((1, 2, 1), 0.8)
    predicted = torch.tensor([[[11.0, 10.0, 13.0, 12.0], [30.0, 20.0, 51.0, 40.0]]])
    labels = torch.zeros((1, 2, 1))
    boxes = torch.tensor([[[10.0, 10.0, 12.0, 12.0], [30.0, 20.0, 50.0, 40.0]]])
    legal = torch.tensor([[[True, False], [False, True]]])

    baseline = TaskAlignedAssigner(num_classes=1)
    baseline.bs = 1
    baseline.n_max_boxes = 2
    _, ciou_quality, _ciou_targets = baseline.get_box_metrics(
        scores, predicted, labels, boxes, legal, image_size=(100, 100)
    )

    mixed = TaskAlignedAssigner(num_classes=1, stal_nwd_weight=0.5)
    mixed.bs = 1
    mixed.n_max_boxes = 2
    _, mixed_quality, mixed_targets = mixed.get_box_metrics(
        scores, predicted, labels, boxes, legal, image_size=(100, 100)
    )

    assert mixed_quality[0, 0, 0] > ciou_quality[0, 0, 0]
    assert mixed_quality[0, 1, 1] == pytest.approx(ciou_quality[0, 1, 1])
    assert torch.equal(ciou_quality[~legal], mixed_quality[~legal])
    assert torch.equal(mixed_quality, mixed_targets)


def test_nwd_ciou_target_mode_decouples_matching_from_supervision_quality():
    """NWD may rank tiny candidates while CIoU continues to set the final target-score scale."""
    scores = torch.full((1, 1, 1), 0.8)
    predicted = torch.tensor([[[11.0, 10.0, 13.0, 12.0]]])
    labels = torch.zeros((1, 1, 1))
    boxes = torch.tensor([[[10.0, 10.0, 12.0, 12.0]]])
    legal = torch.ones((1, 1, 1), dtype=torch.bool)
    assigner = TaskAlignedAssigner(num_classes=1, stal_nwd_weight=0.25, stal_nwd_target_mode="ciou")
    assigner.bs = assigner.n_max_boxes = 1

    _, match_quality, target_quality = assigner.get_box_metrics(
        scores, predicted, labels, boxes, legal, image_size=(100, 100)
    )

    assert match_quality.item() > target_quality.item()
    assert target_quality.item() == pytest.approx(assigner.iou_calculation(boxes[0], predicted[0]).item())


def test_nwd_ciou_target_mode_can_select_zero_supervision_candidate():
    """A zero-CIoU candidate may be matched by NWD but receive no target-score supervision in ciou mode."""
    scores = torch.full((1, 1, 1), 0.8)
    predicted = torch.tensor([[[12.0, 10.0, 14.0, 12.0]]])
    labels = torch.zeros((1, 1, 1))
    boxes = torch.tensor([[[10.0, 10.0, 12.0, 12.0]]])
    anchors = torch.tensor([[12.0, 11.0]])
    valid = torch.ones((1, 1, 1), dtype=torch.bool)
    assigner = TaskAlignedAssigner(num_classes=1, topk=1, stal_nwd_weight=0.25, stal_nwd_target_mode="ciou")

    _, _, target_scores, foreground, _ = assigner(
        scores, predicted, anchors, labels, boxes, valid, image_size=(100, 100)
    )

    assert foreground.item()
    assert target_scores.sum().item() == 0.0


def test_nwd_weighted_target_mode_restores_bounded_supervision():
    """A smaller target blend should supervise NWD matches without using the full matching quality."""
    scores = torch.full((1, 1, 1), 0.8)
    predicted = torch.tensor([[[12.0, 10.0, 14.0, 12.0]]])
    labels = torch.zeros((1, 1, 1))
    boxes = torch.tensor([[[10.0, 10.0, 12.0, 12.0]]])
    legal = torch.ones((1, 1, 1), dtype=torch.bool)
    assigner = TaskAlignedAssigner(
        num_classes=1,
        stal_nwd_weight=0.25,
        stal_nwd_target_mode="weighted",
        stal_nwd_target_weight=0.05,
    )
    assigner.bs = assigner.n_max_boxes = 1

    _, match_quality, target_quality = assigner.get_box_metrics(
        scores, predicted, labels, boxes, legal, image_size=(100, 100)
    )

    assert 0.0 < target_quality.item() < match_quality.item()


def test_nwd_zero_score_floor_only_changes_selected_zero_score_small_anchor():
    """The NWD floor should activate after matching without replacing positive CIoU target scores."""
    scores = torch.full((1, 2, 1), 0.8)
    predicted = torch.tensor([[[12.0, 10.0, 14.0, 12.0], [20.5, 20.0, 22.5, 22.0]]])
    labels = torch.zeros((1, 2, 1))
    boxes = torch.tensor([[[10.0, 10.0, 12.0, 12.0], [20.0, 20.0, 22.0, 22.0]]])
    anchors = torch.tensor([[12.0, 11.0], [21.5, 21.0]])
    valid = torch.ones((1, 2, 1), dtype=torch.bool)
    assigner = TaskAlignedAssigner(
        num_classes=1,
        topk=1,
        stal_nwd_weight=0.25,
        stal_nwd_target_mode="ciou",
        stal_nwd_zero_score_floor=0.01,
    )

    _, _, target_scores, foreground, target_gt_idx = assigner(
        scores, predicted, anchors, labels, boxes, valid, image_size=(100, 100)
    )

    assigned_scores = target_scores.sum(-1)[foreground]
    assigned_gt = target_gt_idx[foreground]
    assert assigned_scores[assigned_gt == 0].item() == pytest.approx(0.01)
    assert assigned_scores[assigned_gt == 1].item() > 0.01


@pytest.mark.parametrize(
    "config",
    [
        {"stal_enabled": "true"},
        {"stal_stats": "true"},
        {"stal_candidate_mode": 1},
        {"stal_area_threshold": "0.01"},
        {"stal_small_topk": "13"},
        {"stal_small_topk_min": "3"},
        {"stal_min_base_candidates": "1"},
        {"stal_max_extra_candidates": "1"},
        {"stal_min_candidate_guarantee": "true"},
        {"stal_expanded_quality_ratio": "0.5"},
        {"stal_expanded_score_floor": "0.5"},
        {"stal_relaxation": "8"},
        {"stal_relaxation_scale_mode": 1},
        {"stal_relaxation_min": "4"},
        {"stal_min_size_stride_ratio": "2"},
        {"stal_warmup_epochs": "10"},
        {"stal_zero_positive_rescue": "true"},
        {"stal_rescue_score_floor": "0.05"},
        {"stal_rescue_floor_decay_epochs": "10"},
    ],
)
def test_stal_cli_configuration_is_type_checked(config):
    """String-like programmatic values must not silently bypass configuration validation."""
    with pytest.raises(TypeError):
        check_cfg(config)


@pytest.mark.parametrize(
    "config",
    [
        {"stal_candidate_mode": "unknown"},
        {"stal_area_threshold": 0.0},
        {"stal_small_topk": 0},
        {"stal_small_topk_min": 0},
        {"stal_small_topk": 4, "stal_small_topk_min": 5},
        {"stal_min_base_candidates": -1},
        {"stal_max_extra_candidates": -1},
        {"stal_expanded_quality_ratio": -0.1},
        {"stal_expanded_quality_ratio": 1.1},
        {"stal_expanded_score_floor": -0.1},
        {"stal_expanded_score_floor": 1.1},
        {"stal_relaxation": -1.0},
        {"stal_relaxation_scale_mode": "unknown"},
        {"stal_relaxation_min": -1.0},
        {"stal_relaxation": 4.0, "stal_relaxation_min": 5.0},
        {"stal_min_size_stride_ratio": -1.0},
        {"stal_min_size_stride_ratio": 2.0},
        {
            "stal_relaxation": 0.0,
            "stal_relaxation_scale_mode": "sqrt_area",
            "stal_min_size_stride_ratio": 2.0,
        },
        {"stal_warmup_epochs": -1.0},
        {"stal_enabled": True, "stal_candidate_mode": "pure"},
        {"stal_rescue_score_floor": 0.05},
        {
            "stal_candidate_mode": "adaptive",
            "stal_zero_positive_rescue": True,
            "stal_rescue_floor_decay_epochs": -1.0,
        },
    ],
)
def test_stal_central_contract_rejects_invalid_values_and_relationships(config):
    """CLI configuration validation must fail before constructing a model or starting training."""
    with pytest.raises(ValueError):
        check_cfg(config)
