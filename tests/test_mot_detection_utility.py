from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from scripts.build_mot_detection_utility import (
    detection_loss_from_items,
    resolve_mot_layer,
    select_indices,
    soft_utility_targets,
    summarize_labels,
    summarize_matrix,
)
from ultralytics.nn.modules.mot import MoTBlock


def test_detection_loss_excludes_routing_auxiliary_term():
    loss = detection_loss_from_items(torch.tensor([1.0, 2.0, 3.0, 9.0]))

    assert loss.total == pytest.approx(6.0)
    assert loss.routing_aux == pytest.approx(9.0)


def test_soft_utility_targets_prefer_low_loss_and_are_stable():
    targets = soft_utility_targets([1.0, 0.5, 2.0], temperature=0.1)

    assert targets.sum() == pytest.approx(1.0)
    assert targets.argmax() == 1
    assert targets[1] > targets[0] > targets[2]
    with pytest.raises(ValueError, match="positive"):
        soft_utility_targets([1.0, 2.0], temperature=0.0)


def test_label_summary_reports_density_and_scale():
    batch = {
        "bboxes": torch.tensor([[0.5, 0.5, 0.01, 0.01], [0.5, 0.5, 0.3, 0.3]]),
        "cls": torch.tensor([[0.0], [3.0]]),
    }

    summary = summarize_labels(batch, imgsz=640)

    assert summary["num_targets"] == 2
    assert summary["num_classes"] == 2
    assert summary["small_target_share"] == pytest.approx(0.5)
    assert summary["large_target_share"] == pytest.approx(0.5)


def test_random_subset_is_reproducible_and_not_prefix_only():
    first = select_indices(dataset_size=100, max_images=12, seed=7)
    second = select_indices(dataset_size=100, max_images=12, seed=7)

    assert first == second
    assert len(first) == 12
    assert first != set(range(12))


def test_layer_resolution_requires_explicit_choice_for_multi_layer_model():
    class TwoBlocks(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = MoTBlock(8, num_heads=2, top_k=2, window_size=2, n_points=2)
            self.second = MoTBlock(8, num_heads=2, top_k=2, window_size=2, n_points=2)

    model = TwoBlocks()

    with pytest.raises(ValueError, match="--layer is required"):
        resolve_mot_layer(model, requested=None)
    name, block = resolve_mot_layer(model, requested="second")
    assert name == "second"
    assert block is model.second


def test_matrix_summary_reports_router_regret_and_oracle_shares():
    rows = [
        {
            "sequence_id": "a",
            "router_regret": 0.2,
            "oracle_gain_over_natural": 0.3,
            "selected_gain_over_natural": 0.1,
            "router_matches_oracle": False,
            "oracle_expert": 1,
            "router_selected_expert": 0,
            "expert_0_total": 1.2,
            "expert_1_total": 1.0,
            "expert_2_total": 1.4,
            "expert_0_target_probability": 0.2,
            "expert_1_target_probability": 0.7,
            "expert_2_target_probability": 0.1,
        },
        {
            "sequence_id": "b",
            "router_regret": 0.0,
            "oracle_gain_over_natural": -0.1,
            "selected_gain_over_natural": -0.1,
            "router_matches_oracle": True,
            "oracle_expert": 2,
            "router_selected_expert": 2,
            "expert_0_total": 2.3,
            "expert_1_total": 2.2,
            "expert_2_total": 2.0,
            "expert_0_target_probability": 0.1,
            "expert_1_target_probability": 0.2,
            "expert_2_target_probability": 0.7,
        },
    ]

    summary = summarize_matrix(rows, ("local", "window", "deformable"))

    assert summary["images"] == 2
    assert summary["router_oracle_accuracy"] == pytest.approx(0.5)
    assert summary["mean_router_regret"] == pytest.approx(0.1)
    assert summary["natural_better_than_every_forced_expert_share"] == pytest.approx(0.5)
    assert summary["oracle_share_1_window"] == pytest.approx(0.5)
    assert summary["median_utility_span"] == pytest.approx(0.35)
    assert summary["low_signal_share_below_1e_4"] == pytest.approx(0.0)
    assert np.isfinite(summary["p95_router_regret"])
