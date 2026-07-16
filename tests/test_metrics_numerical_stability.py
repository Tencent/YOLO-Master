import pytest
import torch

from ultralytics.utils.metrics import bbox_iou


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ciou_low_precision_degenerate_box_has_finite_gradient(dtype):
    pred = torch.tensor([[10.0, 10.0, 20.0, 10.0]], dtype=dtype, requires_grad=True)
    target = torch.tensor([[10.0, 10.0, 20.0, 20.0]], dtype=dtype)

    iou = bbox_iou(pred, target, xywh=False, CIoU=True)
    iou.sum().backward()

    assert torch.isfinite(iou).all()
    assert torch.isfinite(pred.grad).all()
