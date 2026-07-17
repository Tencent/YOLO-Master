from unittest.mock import patch

import torch

from ultralytics.nn.modules.moa.moa import _moa_router_aux_loss
from ultralytics.nn.modules.mot.mot import differentiable_balance_loss


def test_moa_local_gradient_global_value_formula():
    logits = torch.tensor([[[[0.4]], [[0.1]]]], requires_grad=True)
    weights = logits.softmax(1)

    def reduce(tensor, op=None):
        # Synthetic rank 1 contributes usage [0.2, 0.8] and one sample.
        tensor.add_(torch.tensor([0.2, 0.8]) if tensor.numel() == 2 else 1.0)

    with patch("torch.distributed.is_initialized", return_value=True), patch(
        "torch.distributed.get_world_size", return_value=2
    ), patch("torch.distributed.get_backend", return_value="gloo"), patch(
        "torch.distributed.all_reduce", side_effect=reduce
    ):
        loss = _moa_router_aux_loss(weights, logits, 1.0)
        loss.backward()

    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_mot_only_reduces_detached_usage():
    probabilities = torch.tensor([[0.7, 0.3]], requires_grad=True)
    usage = torch.tensor([1.0, 0.0])
    with patch(
        "ultralytics.nn.modules.moe.loss.all_reduce_mean", side_effect=lambda _: torch.tensor([0.25, 0.75])
    ) as reduce:
        differentiable_balance_loss(probabilities, usage, 2, reduce_ddp=True).backward()
    reduce.assert_called_once()
    assert torch.isfinite(probabilities.grad).all()


def test_eval_local_only():
    with torch.no_grad():
        probabilities = torch.tensor([[0.7, 0.3]])
        differentiable_balance_loss(probabilities, torch.tensor([1.0, 0.0]), 2, reduce_ddp=False)
