"""Unit tests for the loss-side STAL integration (per-tier positive-sample statistics).

These cover the config-driven area-tier boundaries of ``v8DetectionLoss._record_tier_positives``: GTs must
bucket into small/medium/large at ``stal_area_small`` / ``stal_area_medium``, zero-positive GTs must be counted,
and non-default thresholds must re-bucket GTs (proving the tiering is not hardcoded).
"""

from types import SimpleNamespace

import torch
from torch import nn

import ultralytics.nn.tasks  # noqa: F401  # import first to break the loss <-> nn.tasks circular import
from ultralytics.utils.loss import v8DetectionLoss


class _Stub:
    def __init__(self, area_small, area_medium):
        self.device = torch.device("cpu")
        self.stal_area_small = area_small
        self.stal_area_medium = area_medium
        self.fg_tier_pos = {}
        self.fg_tier_gt = {}
        self.fg_tier_zero = {}

    def record(self, gt_bboxes, mask_gt, target_gt_idx, fg_mask):
        v8DetectionLoss._record_tier_positives(self, gt_bboxes, mask_gt, target_gt_idx, fg_mask)


def test_record_tier_positives_buckets_by_config_thresholds():
    """Per-tier positive stats must bucket GTs at the config-driven 32²/96² boundaries."""
    stub = _Stub(area_small=1024, area_medium=9216)
    # Three GTs: small (16²=256), medium (64²=4096), large (100²=10000); one positive anchor each.
    gt_bboxes = torch.tensor([[[0.0, 0.0, 16.0, 16.0], [0.0, 0.0, 64.0, 64.0], [0.0, 0.0, 100.0, 100.0]]])
    mask_gt = torch.ones(1, 3, 1, dtype=torch.bool)
    fg_mask = torch.tensor([[True, True, True, False]], dtype=torch.bool)
    target_gt_idx = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)

    stub.record(gt_bboxes, mask_gt, target_gt_idx, fg_mask)

    assert stub.fg_tier_pos == {"small": 1, "medium": 1, "large": 1}
    assert stub.fg_tier_gt == {"small": 1, "medium": 1, "large": 1}
    assert stub.fg_tier_zero == {"small": 0, "medium": 0, "large": 0}


def test_record_tier_positives_zero_positive_and_config_driven():
    """Zero-positive GTs are counted, and non-default thresholds re-bucket GTs (config-driven tiering)."""
    stub = _Stub(area_small=400, area_medium=1600)  # 20² / 40² — non-default
    # area 576 (24x24) is 400 <= area < 1600 -> "medium" under the custom thresholds (would be "small" at 32²/96²).
    gt_bboxes = torch.tensor([[[0.0, 0.0, 24.0, 24.0]]])
    mask_gt = torch.ones(1, 1, 1, dtype=torch.bool)
    fg_mask = torch.tensor([[False]], dtype=torch.bool)  # zero positives for this GT
    target_gt_idx = torch.tensor([[0]], dtype=torch.long)

    stub.record(gt_bboxes, mask_gt, target_gt_idx, fg_mask)

    assert stub.fg_tier_gt == {"medium": 1}
    assert stub.fg_tier_zero == {"medium": 1}
    assert stub.fg_tier_pos == {"medium": 0}
    assert "small" not in stub.fg_tier_gt and "large" not in stub.fg_tier_gt


def test_record_tier_positives_exact_area_boundaries():
    """Exact 32²/96² areas bucket into medium/large; the small/medium upper bounds use strict ``<``."""
    stub = _Stub(area_small=1024, area_medium=9216)
    # 31²=961 -> small; 32²=1024 -> medium; 95²=9025 -> medium; 96²=9216 -> large.
    gt_bboxes = torch.tensor(
        [
            [
                [0.0, 0.0, 31.0, 31.0],
                [0.0, 0.0, 32.0, 32.0],
                [0.0, 0.0, 95.0, 95.0],
                [0.0, 0.0, 96.0, 96.0],
            ]
        ]
    )
    mask_gt = torch.ones(1, 4, 1, dtype=torch.bool)
    fg_mask = torch.tensor([[True, True, True, True]], dtype=torch.bool)
    target_gt_idx = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    stub.record(gt_bboxes, mask_gt, target_gt_idx, fg_mask)

    assert stub.fg_tier_gt == {"small": 1, "medium": 2, "large": 1}
    assert stub.fg_tier_pos == {"small": 1, "medium": 2, "large": 1}
    assert stub.fg_tier_zero == {"small": 0, "medium": 0, "large": 0}


def _make_detection_loss(stal_mode="adaptive", stal_min_positive=False, reg_max=16, nc=3):
    """Build a ``v8DetectionLoss`` against a minimal model stub (single stride level, 3x3 feature map)."""
    args = SimpleNamespace(
        box=7.5,
        cls=0.5,
        dfl=1.5,
        stal_mode=stal_mode,
        stal_min_positive=stal_min_positive,
        stal_area_small=32**2,
        stal_area_medium=96**2,
        stal_topk_small=10,
        stal_expand=1.0,
    )
    head = SimpleNamespace(stride=torch.tensor([8.0]), nc=nc, reg_max=reg_max)
    model = nn.Module()
    model.register_parameter("dummy", nn.Parameter(torch.zeros(1)))  # so next(model.parameters()) resolves a device
    model.args = args
    model.model = [head]
    model.class_weights = None
    return v8DetectionLoss(model, tal_topk=10)


def _preds_batch(nc=3, reg_max=16, hw=4, amp=False, seed=0):
    """Synthetic preds dict + batch with two GTs (small + medium); amp=True casts model outputs to fp16.

    ``seed`` is reset so the fp32 and fp16 variants draw the *same* underlying values (dtype differs only), which is
    the fair basis for a fp32-vs-AMP comparison.
    """
    if seed is not None:
        torch.manual_seed(seed)
    dtype = torch.float16 if amp else torch.float32
    a = hw * hw
    preds = {
        "feats": [torch.randn(1, 16, hw, hw, dtype=dtype)],
        "boxes": torch.randn(1, reg_max * 4, a, dtype=dtype),
        "scores": torch.randn(1, nc, a, dtype=dtype),
    }
    batch = {
        "batch_idx": torch.tensor([[0], [0]]),
        "cls": torch.tensor([[0], [1]]),
        "bboxes": torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.6, 0.6, 0.5, 0.5]]),
    }
    return preds, batch


def test_loss_and_grad_finite_fp32():
    """Full v8DetectionLoss forward+backward must be finite (loss value and gradients) in fp32."""
    crit = _make_detection_loss(stal_mode="adaptive", stal_min_positive=True)
    preds, batch = _preds_batch(amp=False)
    preds["boxes"].requires_grad_(True)
    preds["scores"].requires_grad_(True)

    loss, _ = crit.loss(preds, batch)
    total = loss.sum()

    assert torch.isfinite(total)
    total.backward()
    assert torch.isfinite(preds["boxes"].grad).all()
    assert torch.isfinite(preds["scores"].grad).all()


def test_loss_and_grad_finite_amp():
    """Full v8DetectionLoss forward+backward must stay finite under AMP-style fp16 model outputs."""
    crit = _make_detection_loss(stal_mode="adaptive", stal_min_positive=True)
    preds, batch = _preds_batch(amp=True)
    preds["boxes"].requires_grad_(True)
    preds["scores"].requires_grad_(True)

    loss, _ = crit.loss(preds, batch)
    total = loss.sum()

    assert torch.isfinite(total.float())
    total.backward()
    assert torch.isfinite(preds["boxes"].grad.float()).all()
    assert torch.isfinite(preds["scores"].grad.float()).all()


def test_fg_mask_consistent_fp32_vs_amp():
    """The STAL assignment (fg_mask, positive count) must be identical between fp32 and AMP-style fp16 inputs."""
    crit = _make_detection_loss(stal_mode="adaptive", stal_min_positive=True)
    preds32, batch = _preds_batch(amp=False)
    preds16, _ = _preds_batch(amp=True)

    (fg32, _, _, _, _), _, _ = crit.get_assigned_targets_and_loss(preds32, batch)
    (fg16, _, _, _, _), _, _ = crit.get_assigned_targets_and_loss(preds16, batch)

    assert torch.equal(fg32, fg16)
    assert fg32.sum().item() == fg16.sum().item() > 0
