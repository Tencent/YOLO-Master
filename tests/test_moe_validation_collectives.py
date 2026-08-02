from unittest.mock import patch

import torch
from torch import nn

from ultralytics.nn.modules.moe.loss import MoELoss, differentiable_balance_loss, should_reduce_ddp


def test_eval_and_no_grad_local():
    model = nn.Linear(2, 2).eval()
    with patch("torch.distributed.is_initialized", return_value=True), patch(
        "torch.distributed.get_world_size", return_value=2
    ), patch("torch.distributed.all_reduce") as reduce:
        assert not should_reduce_ddp(model)
        with torch.no_grad():
            differentiable_balance_loss(
                torch.tensor([[0.6, 0.4]]),
                torch.tensor([0.5, 0.5]),
                2,
                reduce_ddp=should_reduce_ddp(model),
            )
        reduce.assert_not_called()


def test_train_grad_global():
    model = nn.Linear(2, 2).train()
    with patch("torch.distributed.is_initialized", return_value=True), patch(
        "torch.distributed.get_world_size", return_value=2
    ):
        assert should_reduce_ddp(model)


def test_moe_loss_eval_hidden_usage_local():
    loss = MoELoss(num_experts=2).eval()
    probabilities = torch.tensor([[0.7, 0.3]])
    with patch("torch.distributed.is_initialized", return_value=True), patch(
        "torch.distributed.get_world_size", return_value=2
    ), patch("torch.distributed.all_reduce") as reduce:
        with torch.no_grad():
            loss(probabilities, probabilities.log(), torch.tensor([[0]]))
        reduce.assert_not_called()


def test_moe_loss_packs_ddp_statistics_into_one_collective():
    """Importance sums/counts and hard usage counts share one all-reduce."""
    loss = MoELoss(num_experts=2).train()
    probabilities = torch.tensor([[0.2, 0.8], [0.4, 0.6]], requires_grad=True)
    indices = torch.tensor([[0], [1]])

    def add_remote_rank(value, op=None):
        # Packed layout: [importance_sum(2), sample_count, expert_counts(2), selected_count].
        value.add_(torch.tensor([0.4, 1.6, 2.0, 1.0, 1.0, 2.0]))

    with patch("torch.distributed.is_initialized", return_value=True), patch(
        "torch.distributed.get_world_size", return_value=2
    ), patch("torch.distributed.get_backend", return_value="gloo"), patch(
        "torch.distributed.all_reduce", side_effect=add_remote_rank
    ) as reduce:
        result = loss(probabilities, probabilities.log(), indices)
        result.backward()

    assert reduce.call_count == 1
    # Global importance is mean([0.2, 0.8], [0.4, 0.6], [0.2, 0.8], [0.4, 0.6]).
    assert torch.isfinite(result)
    assert torch.allclose(probabilities.grad, torch.full_like(probabilities, 0.5), atol=1e-5)
