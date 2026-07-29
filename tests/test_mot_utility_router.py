from __future__ import annotations

import csv

import pytest
import torch
from torch import nn

from scripts.train_mot_utility_router import (
    read_utility_matrix,
    routing_metrics,
    split_sequence_indices,
    utility_importance,
    utility_objective,
    utility_probabilities,
)


class ToyUtilityRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(2, 3, 1, bias=False)
        self.register_buffer("temperature", torch.tensor(1.0))

    def _compute_logits(self, features):
        return self.projection(features)


def test_read_utility_matrix_validates_and_aligns_experts(tmp_path):
    path = tmp_path / "matrix.csv"
    fieldnames = [
        "image_id",
        "sequence_id",
        "expert_0_name",
        "expert_0_total",
        "expert_0_target_probability",
        "expert_0_router_probability",
        "expert_1_name",
        "expert_1_total",
        "expert_1_target_probability",
        "expert_1_router_probability",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "/private/path/frame.jpg",
                "sequence_id": "seq",
                "expert_0_name": "local",
                "expert_0_total": 1.0,
                "expert_0_target_probability": 0.8,
                "expert_0_router_probability": 0.4,
                "expert_1_name": "window",
                "expert_1_total": 2.0,
                "expert_1_target_probability": 0.2,
                "expert_1_router_probability": 0.6,
            }
        )

    records, names = read_utility_matrix(path)

    assert names == ("local", "window")
    assert records[0].image_id == "frame.jpg"
    assert records[0].target.tolist() == pytest.approx([0.8, 0.2])


def test_sequence_split_has_no_sequence_overlap_and_is_reproducible():
    sequences = ["a", "a", "b", "b", "c", "d"]

    train, validation, held_out = split_sequence_indices(sequences, validation_fraction=0.25, seed=3)
    train_again, validation_again, held_out_again = split_sequence_indices(sequences, 0.25, seed=3)

    assert (train, validation, held_out) == (train_again, validation_again, held_out_again)
    assert {sequences[index] for index in train}.isdisjoint({sequences[index] for index in validation})


def test_utility_objective_backpropagates_and_weights_larger_effects():
    probabilities = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]], requires_grad=True)
    targets = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])
    anchors = torch.full((2, 3), 1 / 3)
    forced_losses = torch.tensor([[1.0, 1.1, 1.2], [1.0, 2.0, 3.0]])

    importance = utility_importance(forced_losses, power=0.5)
    loss = utility_objective(
        probabilities,
        targets,
        anchors,
        forced_losses,
        anchor_weight=0.1,
        importance_power=0.5,
    )
    loss.backward()

    assert importance[1] > importance[0]
    assert probabilities.grad is not None
    assert torch.isfinite(probabilities.grad).all()


def test_router_probabilities_are_normalized_and_trainable():
    router = ToyUtilityRouter()
    features = torch.randn(4, 2, 3, 3)

    probabilities = utility_probabilities(router, features)
    probabilities[:, 0].sum().backward()

    assert probabilities.shape == (4, 3)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(4), atol=1e-6)
    assert router.projection.weight.grad is not None


def test_routing_metrics_use_detection_regret_not_probability_distance():
    probabilities = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
    targets = torch.tensor([[0.9, 0.05, 0.05], [0.1, 0.2, 0.7]])
    forced_losses = torch.tensor([[1.0, 2.0, 3.0], [1.5, 1.2, 1.0]])

    metrics = routing_metrics(probabilities, targets, forced_losses)

    assert metrics["oracle_accuracy"] == pytest.approx(0.5)
    assert metrics["mean_regret"] == pytest.approx(0.1)
    assert metrics["cross_entropy"] > 0
