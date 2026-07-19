"""Regression coverage for non-autograd c10d all-reduce handling."""
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

from ultralytics.nn.modules.moe.loss import all_reduce_mean as moe_all_reduce_mean
from ultralytics.nn.modules.moa.moa import all_reduce_mean as moa_all_reduce_mean
from ultralytics.nn.modules.mot.mot import all_reduce_mean as mot_all_reduce_mean
from ultralytics.nn.peft.molora.loss import all_reduce_mean as molora_all_reduce_mean


@pytest.mark.parametrize(
    "reduce_mean",
    [moe_all_reduce_mean, moa_all_reduce_mean, mot_all_reduce_mean, molora_all_reduce_mean],
)
def test_all_reduce_mean_uses_global_value_and_local_gradient(reduce_mean):
    """Raw c10d mutates only a detached copy; input retains its local Jacobian."""
    local = torch.tensor([0.2, 0.8], requires_grad=True)

    def remote_rank_sum(value, op=None):
        value.add_(torch.tensor([0.4, 0.6]))

    with ExitStack() as stack:
        stack.enter_context(patch("torch.distributed.is_available", return_value=True))
        stack.enter_context(patch("torch.distributed.is_initialized", return_value=True))
        stack.enter_context(patch("torch.distributed.get_world_size", return_value=2))
        stack.enter_context(patch("torch.distributed.get_backend", return_value="gloo"))
        stack.enter_context(patch("torch.distributed.all_reduce", side_effect=remote_rank_sum))
        result = reduce_mean(local)
        result.sum().backward()

    assert torch.allclose(result, torch.tensor([0.3, 0.7]))
    assert torch.allclose(local.grad, torch.ones_like(local))


def test_moe_global_mean_uses_detached_collective_and_local_gradient():
    """MoELoss global mean has global forward value without a c10d autograd edge."""
    from ultralytics.nn.modules.moe.loss import MoELoss

    values = torch.tensor([[0.2, 0.8], [0.4, 0.6]], requires_grad=True)
    collective_requires_grad = []

    def remote_rank_sum(value, op=None):
        collective_requires_grad.append(value.requires_grad)
        if value.numel() == 2:
            value.add_(torch.tensor([0.6, 1.4]))
        else:
            value.add_(2.0)

    with ExitStack() as stack:
        stack.enter_context(patch("torch.distributed.is_available", return_value=True))
        stack.enter_context(patch("torch.distributed.is_initialized", return_value=True))
        stack.enter_context(patch("torch.distributed.get_world_size", return_value=2))
        stack.enter_context(patch("torch.distributed.get_backend", return_value="gloo"))
        stack.enter_context(patch("torch.distributed.all_reduce", side_effect=remote_rank_sum))
        result = MoELoss(num_experts=2)._get_global_mean(values)
        result.sum().backward()

    assert collective_requires_grad == [False, False]
    assert torch.allclose(result, torch.tensor([0.3, 0.7]))
    assert torch.allclose(values.grad, torch.full_like(values, 0.5))
