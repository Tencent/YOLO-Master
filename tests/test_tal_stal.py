"""Unit tests for STAL (area-aware task-aligned assignment) in the task-aligned assigner.

These cover the P1 acceptance criteria: STAL-off == pure TAL, the legacy fixed-stride rescue, area-tier
top-k boundaries, empty-GT early return, adaptive candidate expansion, min-positive fallback, and
overlap-conflict resolution, plus fp32/fp16 finiteness of the assigner forward.
"""

import torch

from ultralytics.utils.tal import TaskAlignedAssigner

NUM_CLASSES = 3
NUM_ANCHORS = 16  # 4x4 grid of integer anchor centers (0..3)


def _grid_anchor_points(rows=4, cols=4):
    """Return integer anchor centers on a rows x cols grid."""
    return torch.tensor([[float(i), float(j)] for i in range(rows) for j in range(cols)])


def _run_assigner(assigner, gt_bboxes, mask_gt, num_anchors=NUM_ANCHORS, dtype=torch.float32):
    """Run a full assigner forward on synthetic inputs and return the five output tensors."""
    batch_size, n_boxes = gt_bboxes.shape[:2]
    anc_points = _grid_anchor_points()[:num_anchors].to(dtype)
    pd_scores = torch.full((batch_size, num_anchors, NUM_CLASSES), 0.6, dtype=dtype)
    pd_bboxes = torch.zeros((batch_size, num_anchors, 4), dtype=dtype)
    pd_bboxes[..., 2:] = 1.5  # every anchor predicts the same unit box (0.0, 0.0, 1.5, 1.5)
    gt_labels = torch.zeros((batch_size, n_boxes, 1), dtype=torch.long)
    return assigner(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes.to(dtype), mask_gt)


def test_stal_none_matches_pure_tal_no_rescue():
    """Pure TAL (stal_mode='none') must not fabricate positives for a tiny GT with no in-box anchor."""
    assigner = TaskAlignedAssigner(topk=4, num_classes=NUM_CLASSES, stal_mode="none")
    gt_bboxes = torch.tensor([[[0.1, 0.1, 0.9, 0.9]]])  # area 0.64, no integer center strictly inside
    mask_gt = torch.tensor([[[True]]])
    fg_mask = _run_assigner(assigner, gt_bboxes, mask_gt)[3]
    assert fg_mask.sum().item() == 0


def test_none_is_pure_tal():
    """``stal_mode='none'`` leaves every STAL hook (clamp / adaptive top-k / min-positive) inert, so none == pure TAL."""
    stride = [8, 16, 32]
    none = TaskAlignedAssigner(topk=4, num_classes=NUM_CLASSES, stal_mode="none", stride=stride)
    fixed = TaskAlignedAssigner(topk=4, num_classes=NUM_CLASSES, stal_mode="fixed", stride=stride)

    # (1) Adaptive top-k is disabled outside adaptive mode.
    gt_bboxes = torch.tensor([[[0.0, 0.0, 8.0, 8.0], [1.0, 1.0, 9.0, 9.0]]])  # both w,h >= stride[0]
    assert none._adaptive_topk(gt_bboxes) is None
    assert fixed._adaptive_topk(gt_bboxes) is None

    # (2) Full forward: none == fixed when the clamp is a no-op (overlapping GTs trigger conflict resolution).
    # Each anchor predicts an 8x8 box (same scale as the GTs) so CIoU is positive and positives are assigned;
    # the assigner's iou_calculation uses CIoU, which goes negative for a tiny pred box vs. a large GT.
    mask_gt = torch.tensor([[[True], [True]]])

    def _forward(assigner):
        anc = _grid_anchor_points()
        pd_scores = torch.full((1, NUM_ANCHORS, NUM_CLASSES), 0.6)
        pd_bboxes = torch.zeros((1, NUM_ANCHORS, 4))
        pd_bboxes[..., 2:] = 8.0  # [0,0,8,8] per anchor
        gt_labels = torch.zeros((1, 2, 1), dtype=torch.long)
        return assigner(pd_scores, pd_bboxes, anc, gt_labels, gt_bboxes, mask_gt)

    out_none = _forward(none)
    out_fixed = _forward(fixed)
    assert out_none[3].sum().item() > 0  # non-degenerate: at least one positive anchor
    for a, b in zip(out_none, out_fixed):
        assert torch.equal(a, b)

    # (3) Clamp-free: a box narrower than stride[0] is NOT enlarged in none mode (no in-box integer center).
    tiny = torch.tensor([[[0.1, 0.1, 0.9, 0.9]]])
    mask_tiny = torch.tensor([[[True]]])
    xy = _grid_anchor_points()
    assert none.select_candidates_in_gts(xy, tiny, mask_tiny).sum().item() == 0
    assert fixed.select_candidates_in_gts(xy, tiny, mask_tiny).sum().item() > 0  # fixed clamps -> recovers a center


def test_fixed_stride_rescues_tiny_gt():
    """The legacy fixed-stride clamp must enlarge a tiny GT enough to recover at least one positive."""
    assigner = TaskAlignedAssigner(topk=4, num_classes=NUM_CLASSES, stal_mode="fixed", stride=[8, 16, 32])
    gt_bboxes = torch.tensor([[[0.1, 0.1, 0.9, 0.9]]])
    mask_gt = torch.tensor([[[True]]])
    fg_mask = _run_assigner(assigner, gt_bboxes, mask_gt)[3]
    assert fg_mask.sum().item() > 0


def test_adaptive_topk_area_boundary():
    """Small-tier GT (area < stal_area_small) gets stal_topk_small; others keep the base topk."""
    assigner = TaskAlignedAssigner(
        topk=4, num_classes=NUM_CLASSES, stal_mode="adaptive", stal_area_small=1024, stal_topk_small=8
    )
    # 16x16 -> area 256 (small); 40x40 -> area 1600 (not small).
    gt_bboxes = torch.tensor([[[0.0, 0.0, 16.0, 16.0], [0.0, 0.0, 40.0, 40.0]]])
    k = assigner._adaptive_topk(gt_bboxes)
    assert k[0, 0].item() == 8
    assert k[0, 1].item() == 4


def test_adaptive_expands_small_candidates():
    """Adaptive expansion enlarges small-tier boxes by stal_expand * stride[0]; larger boxes stay unchanged."""
    assigner = TaskAlignedAssigner(stal_mode="adaptive", stal_area_small=1024, stal_expand=1.0, stride=[8, 16, 32])
    mask_gt = torch.tensor([[[True]]])
    small = torch.tensor([[[10.0, 10.0, 12.0, 12.0]]])  # area 4 < 1024 -> radius 8
    expanded = assigner._expand_small_candidates(small, mask_gt)
    assert torch.allclose(expanded[0, 0], torch.tensor([2.0, 2.0, 20.0, 20.0]))
    large = torch.tensor([[[0.0, 0.0, 40.0, 40.0]]])  # area 1600 >= 1024 -> unchanged
    assert torch.allclose(assigner._expand_small_candidates(large, mask_gt)[0, 0], large[0, 0])


def test_ensure_min_positive_forces_anchor():
    """A valid GT with an in-box anchor but no positive must be promoted to its best (highest-IoU) anchor."""
    assigner = TaskAlignedAssigner(num_classes=NUM_CLASSES, stal_min_positive=True)
    mask_pos = torch.zeros(1, 1, 4)
    overlaps = torch.zeros(1, 1, 4)
    overlaps[0, 0, 2] = 0.7  # highest-IoU in-box anchor
    mask_in_gts = torch.zeros(1, 1, 4, dtype=torch.bool)
    mask_in_gts[0, 0, 1] = True
    mask_in_gts[0, 0, 2] = True
    mask_gt = torch.ones(1, 1, 1, dtype=torch.bool)

    out = assigner._ensure_min_positive(mask_pos, overlaps, mask_in_gts, mask_gt)

    assert out[0, 0, 2].item() == 1.0
    assert out[0, 0].sum().item() == 1.0


def test_ensure_min_positive_no_anchor_noop():
    """When a GT has no in-box anchor at all, min-positive must not fabricate a bogus positive."""
    assigner = TaskAlignedAssigner(num_classes=NUM_CLASSES, stal_min_positive=True)
    mask_pos = torch.zeros(1, 1, 4)
    overlaps = torch.zeros(1, 1, 4)
    mask_in_gts = torch.zeros(1, 1, 4, dtype=torch.bool)
    mask_gt = torch.ones(1, 1, 1, dtype=torch.bool)

    out = assigner._ensure_min_positive(mask_pos, overlaps, mask_in_gts, mask_gt)

    assert out.sum().item() == 0


def test_select_highest_overlaps_resolves_conflict():
    """An anchor inside two GTs must be assigned to the higher-IoU GT, never shared."""
    assigner = TaskAlignedAssigner(num_classes=NUM_CLASSES)
    mask_pos = torch.tensor([[[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]])  # anchor 0 positive for both GTs
    overlaps = torch.tensor([[[0.9, 0.8, 0.0], [0.4, 0.0, 0.0]]])
    align_metric = torch.ones(1, 2, 3)

    target_gt_idx, fg_mask, _ = assigner.select_highest_overlaps(mask_pos, overlaps, 2, align_metric)

    assert target_gt_idx[0, 0].item() == 0  # GT 0 has the higher IoU (0.9 > 0.4)
    assert fg_mask[0, 0].item() == 1
    assert (fg_mask > 1).sum().item() == 0  # no anchor remains shared


def test_empty_gt_early_return():
    """n_max_boxes == 0 must short-circuit to the background-label / zero tensors."""
    assigner = TaskAlignedAssigner(topk=4, num_classes=NUM_CLASSES)
    batch_size = 2
    anc_points = _grid_anchor_points()
    pd_scores = torch.zeros(batch_size, NUM_ANCHORS, NUM_CLASSES)
    pd_bboxes = torch.zeros(batch_size, NUM_ANCHORS, 4)
    gt_labels = torch.zeros(batch_size, 0, 1, dtype=torch.long)
    gt_bboxes = torch.zeros(batch_size, 0, 4)
    mask_gt = torch.zeros(batch_size, 0, 1, dtype=torch.bool)

    target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = assigner(
        pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt
    )

    assert (target_labels == NUM_CLASSES).all()
    assert target_scores.sum().item() == 0
    assert fg_mask.sum().item() == 0
    assert target_bboxes.shape == pd_bboxes.shape
    assert target_gt_idx.shape == pd_scores[..., 0].shape


def test_assigner_forward_finite_fp32():
    """The assigner forward must produce finite outputs in fp32."""
    assigner = TaskAlignedAssigner(topk=4, num_classes=NUM_CLASSES, stal_mode="adaptive", stal_topk_small=8)
    gt_bboxes = torch.tensor([[[0.0, 0.0, 3.0, 3.0], [1.0, 1.0, 2.5, 2.5]]])
    mask_gt = torch.tensor([[[True], [True]]])

    result = _run_assigner(assigner, gt_bboxes, mask_gt)

    for out in result:
        assert torch.isfinite(out.float()).all()


def test_assigner_forward_finite_mixed_precision():
    """The assigner forward must stay finite under AMP-style mixed precision.

    Mirrors the dtypes ``loss.py`` actually passes under autocast: predicted scores and anchor points are fp16,
    while predicted boxes and GT boxes remain fp32 (``pred_bboxes`` is cast via ``.type(gt_bboxes.dtype)``).
    """
    assigner = TaskAlignedAssigner(topk=4, num_classes=NUM_CLASSES, stal_mode="adaptive", stal_topk_small=8)
    gt_bboxes = torch.tensor([[[0.0, 0.0, 3.0, 3.0], [1.0, 1.0, 2.5, 2.5]]])  # fp32
    mask_gt = torch.tensor([[[True], [True]]])
    anc_points = _grid_anchor_points().half()  # fp16
    pd_scores = torch.full((1, NUM_ANCHORS, NUM_CLASSES), 0.6, dtype=torch.float16)
    pd_bboxes = torch.zeros((1, NUM_ANCHORS, 4), dtype=torch.float32)  # fp32, as loss.py casts it
    pd_bboxes[..., 2:] = 1.5
    gt_labels = torch.zeros((1, 2, 1), dtype=torch.long)

    result = assigner(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)

    for out in result:
        assert torch.isfinite(out.float()).all()


def test_assigner_forward_finite_with_min_positive():
    """Min-positive (re-applied after conflict resolution) must keep the forward finite with overlapping GTs."""
    assigner = TaskAlignedAssigner(
        topk=4, num_classes=NUM_CLASSES, stal_mode="adaptive", stal_topk_small=8, stal_min_positive=True
    )
    gt_bboxes = torch.tensor([[[0.0, 0.0, 3.0, 3.0], [0.5, 0.5, 3.0, 3.0]]])  # overlapping GTs
    mask_gt = torch.tensor([[[True], [True]]])

    result = _run_assigner(assigner, gt_bboxes, mask_gt)

    for out in result:
        assert torch.isfinite(out.float()).all()


def test_select_topk_adaptive_truncation_preserves_anchor_zero():
    """Adaptive per-GT truncation must not scatter masked tail columns into anchor 0 and drop a genuine candidate."""
    assigner = TaskAlignedAssigner(topk=2, num_classes=NUM_CLASSES, stal_mode="adaptive", stal_topk_small=4)
    metrics = torch.tensor([[[0.9, 0.7, 0.5, 0.3]]])  # anchor 0 is the highest
    mask_gt = torch.ones(1, 1, 1, dtype=torch.bool)
    k = torch.tensor([[2]])  # non-small GT -> base topk, truncated to the top-2

    out = assigner.select_topk_candidates(metrics, topk_mask=mask_gt, k=k)

    assert out[0, 0, 0].item() == 1.0  # anchor 0 must survive truncation
    assert out[0, 0, 1].item() == 1.0
    assert out[0, 0].sum().item() == 2.0  # exactly top-2, no spurious extras


def test_expand_small_candidates_uses_preclamp_area():
    """Small-tier decision must use the pre-clamp area, so a thin box clamped over the threshold is still expanded."""
    assigner = TaskAlignedAssigner(stal_mode="adaptive", stal_area_small=1024, stal_expand=1.0, stride=[8, 16, 32])
    mask_gt = torch.tensor([[[True]]])
    # Pre-clamp 5x200 (area 1000 < 1024 -> small); post-clamp 16x200 (area 3200 >= 1024 -> not small).
    preclamp = torch.tensor([[[0.0, 0.0, 5.0, 200.0]]])
    clamped = torch.tensor([[[0.0, 0.0, 16.0, 200.0]]])

    expanded = assigner._expand_small_candidates(clamped, mask_gt, area_gt_bboxes=preclamp)

    assert torch.allclose(expanded[0, 0], torch.tensor([-8.0, -8.0, 24.0, 208.0]))


def test_select_candidates_in_gts_expands_thin_small_box():
    """A thin box whose clamped area crosses stal_area_small must still be expanded (pre-clamp area decides)."""
    assigner = TaskAlignedAssigner(stal_mode="adaptive", stal_area_small=1024, stal_expand=1.0, stride=[8, 16, 32])
    gt_bboxes = torch.tensor([[[0.0, 0.0, 5.0, 200.0]]])  # w=5 < stride[0], area 1000 < 1024
    mask_gt = torch.tensor([[[True]]])
    # Clamp centers the box: [0,0,5,200] -> clamped [-5.5,0,10.5,200], then expanded by radius 8 -> [-13.5,-8,18.5,208].
    xy_centers = torch.tensor([[15.0, 100.0]])  # inside expanded x-range, outside the clamped-only range

    mask = assigner.select_candidates_in_gts(xy_centers, gt_bboxes, mask_gt)

    assert mask[0, 0, 0].item() is True


def test_ensure_min_positive_does_not_steal_foreground_anchor():
    """Min-positive must promote a free in-box anchor, never an anchor already claimed by another GT."""
    assigner = TaskAlignedAssigner(num_classes=NUM_CLASSES, stal_min_positive=True)
    mask_pos = torch.zeros(1, 2, 4)
    mask_pos[0, 1, 2] = 1.0  # anchor 2 already owned by GT 1
    overlaps = torch.zeros(1, 2, 4)
    overlaps[0, 0, 2] = 0.9  # GT 0's highest-IoU in-box anchor (but it is foreground)
    overlaps[0, 0, 1] = 0.4  # GT 0's next-best free in-box anchor
    mask_in_gts = torch.zeros(1, 2, 4, dtype=torch.bool)
    mask_in_gts[0, 0, 1] = True
    mask_in_gts[0, 0, 2] = True
    mask_in_gts[0, 1, 2] = True
    mask_gt = torch.ones(1, 2, 1, dtype=torch.bool)

    out = assigner._ensure_min_positive(mask_pos, overlaps, mask_in_gts, mask_gt)

    assert out[0, 0, 1].item() == 1.0  # GT 0 promoted to free anchor 1
    assert out[0, 0, 2].item() == 0.0  # anchor 2 not stolen
    assert out[0, 1, 2].item() == 1.0  # GT 1 keeps anchor 2


def test_min_positive_recovers_gt_stripped_by_conflict():
    """A GT stripped to zero positives by conflict resolution must get a fallback on a free in-box anchor."""
    assigner = TaskAlignedAssigner(num_classes=NUM_CLASSES, stal_min_positive=True)
    mask_pos = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])  # anchor 0 positive for both GTs
    overlaps = torch.tensor([[[0.4, 0.0, 0.0], [0.9, 0.0, 0.0]]])  # GT 1 higher IoU on anchor 0
    align_metric = torch.ones(1, 2, 3)
    mask_in_gts = torch.tensor([[[True, True, False], [True, False, False]]])
    mask_gt = torch.ones(1, 2, 1, dtype=torch.bool)

    _, _, mask_pos = assigner.select_highest_overlaps(mask_pos, overlaps, 2, align_metric)
    assert mask_pos[0, 0].sum().item() == 0.0  # GT 0 stripped to zero

    mask_pos = assigner._ensure_min_positive(mask_pos, overlaps, mask_in_gts, mask_gt)

    assert mask_pos[0, 0, 1].item() == 1.0  # GT 0 recovers on free anchor 1
    assert mask_pos[0, 0].sum().item() == 1.0
    assert mask_pos[0, 1, 0].item() == 1.0  # GT 1 keeps anchor 0


def _run_assigner_scores(assigner, gt_bboxes, mask_gt, amp):
    """Run the assigner with distinct per-anchor class scores; amp=True uses fp16 scores + anchors (fp32 boxes/GT)."""
    score_dtype = torch.float16 if amp else torch.float32
    anc_dtype = torch.float16 if amp else torch.float32
    logits = torch.linspace(-2.0, 2.0, NUM_ANCHORS * NUM_CLASSES).reshape(NUM_ANCHORS, NUM_CLASSES)
    pd_scores = torch.sigmoid(logits.to(score_dtype)).unsqueeze(0)  # (1, NUM_ANCHORS, NUM_CLASSES)
    anc_points = _grid_anchor_points()[:NUM_ANCHORS].to(anc_dtype)
    pd_bboxes = torch.zeros((1, NUM_ANCHORS, 4), dtype=torch.float32)
    pd_bboxes[..., 2:] = 1.5
    gt_labels = torch.zeros((1, gt_bboxes.shape[1], 1), dtype=torch.long)
    return assigner(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)


def test_assignment_consistent_fp32_vs_amp():
    """Adaptive STAL assignment (mask / top-k / count / labels / gt-index) must match between fp32 and AMP-style fp16."""
    assigner = TaskAlignedAssigner(
        topk=4, num_classes=NUM_CLASSES, stal_mode="adaptive", stal_topk_small=8, stride=[8, 16, 32]
    )
    gt_bboxes = torch.tensor([[[0.0, 0.0, 3.0, 3.0], [1.0, 1.0, 2.5, 2.5]]])  # small + medium, overlapping
    mask_gt = torch.tensor([[[True], [True]]])

    labels32, _, scores32, fg32, idx32 = _run_assigner_scores(assigner, gt_bboxes, mask_gt, amp=False)
    labels16, _, scores16, fg16, idx16 = _run_assigner_scores(assigner, gt_bboxes, mask_gt, amp=True)

    assert torch.equal(fg32, fg16)  # identical positive mask (top-k selection)
    assert fg32.sum().item() == fg16.sum().item() > 0  # identical positive count
    assert torch.equal(labels32, labels16)  # identical target labels
    assert torch.equal(idx32, idx16)  # identical GT assignment
    assert torch.allclose(scores32.float(), scores16.float(), atol=1e-2)  # continuous score, fp16 rounding tolerance
