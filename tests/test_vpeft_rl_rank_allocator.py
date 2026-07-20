"""Tests for the PPO-based V-PEFT rank allocator."""

import torch

from ultralytics.vpeft import (
    ComputationGraph,
    GreedyRankAllocator,
    HybridTrainingProtocol,
    ModuleNode,
    RLRankAllocator,
)


def _graph(*, embeddings=True):
    features = torch.tensor(
        [[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5], [0.5, 0.5, 1.0, 0.0]],
        dtype=torch.float32,
    )
    return ComputationGraph(
        modules=[
            ModuleNode("backbone.fc0", "Linear", 8, 8),
            ModuleNode("neck.fc1", "Linear", 8, 8),
            ModuleNode("head.fc2", "Linear", 8, 8),
        ],
        node_features=features if embeddings else None,
    )


def _force_action(allocator, action_index):
    with torch.no_grad():
        for parameter in allocator.state_encoder.parameters():
            parameter.zero_()
        allocator.policy_head.weight.zero_()
        allocator.policy_head.bias.fill_(-10.0)
        allocator.policy_head.bias[action_index] = 10.0


def test_feasible_action_mask_respects_remaining_budget():
    allocator = RLRankAllocator(hidden_dim=4, rank_set=[4, 8])
    graph = _graph()

    assert allocator.feasible_action_mask(graph, 0, 63, "lora").tolist() == [True, False, False]
    assert allocator.feasible_action_mask(graph, 0, 64, "lora").tolist() == [True, True, False]
    assert allocator.feasible_action_mask(graph, 0, 128, "lora").tolist() == [True, True, True]


def test_trajectory_applies_action_masks_and_never_exceeds_budget():
    allocator = RLRankAllocator(hidden_dim=4, rank_set=[4, 8])
    _force_action(allocator, action_index=2)

    trajectory = allocator.collect_trajectory(
        _graph(), torch.ones(3), budget=128, variant="lora", deterministic=True
    )

    assert trajectory.actions.tolist() == [2, 0, 0]
    assert trajectory.allocation.tolist() == [8.0, 0.0, 0.0]
    assert trajectory.budget_used == 128
    assert all(bool(mask[action]) for mask, action in zip(trajectory.action_masks, trajectory.actions))


def test_generalized_advantage_estimation_matches_manual_returns():
    advantages, returns = RLRankAllocator.compute_gae(
        rewards=torch.tensor([1.0, 2.0]),
        values=torch.tensor([0.5, 0.25]),
        dones=torch.tensor([0.0, 1.0]),
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert torch.allclose(advantages, torch.tensor([2.5, 1.75]))
    assert torch.allclose(returns, torch.tensor([3.0, 2.0]))


def test_ppo_update_trains_policy_and_value_heads():
    torch.manual_seed(7)
    allocator = RLRankAllocator(hidden_dim=4, rank_set=[4, 8])
    trajectory = allocator.collect_trajectory(_graph(), torch.ones(3), budget=256, variant="lora")
    before = [parameter.detach().clone() for parameter in allocator.parameters()]

    metrics = allocator.ppo_update(trajectory, epochs=3, minibatch_size=2, lr=1e-2)

    assert allocator.is_trained
    assert allocator.ppo_updates.item() == 6
    assert metrics["updates"] == 6.0
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert any(not torch.allclose(old, new) for old, new in zip(before, allocator.parameters()))


def test_allocate_uses_greedy_cold_start_and_trained_policy_after_update_marker():
    placement = torch.ones(3)
    cold_graph = _graph(embeddings=False)
    allocator = RLRankAllocator(hidden_dim=4, rank_set=[4, 8])

    expected = GreedyRankAllocator(rank_set=[4, 8]).allocate(cold_graph, placement, 128, "lora")
    assert torch.equal(allocator.allocate(cold_graph, placement, 128, "lora"), expected)

    _force_action(allocator, action_index=1)
    allocator.ppo_updates.fill_(1)
    learned = allocator.allocate(_graph(), placement, 128, "lora")
    assert learned.tolist() == [4.0, 4.0, 0.0]


def test_hybrid_training_protocol_runs_real_ppo_updates():
    torch.manual_seed(11)
    allocator = RLRankAllocator(hidden_dim=4, rank_set=[4, 8])
    episode = {
        "graph": _graph(),
        "placement": torch.ones(3),
        "budget": 192,
        "variant": "lora",
        "episode_reward": 0.5,
    }

    result = HybridTrainingProtocol.train_rl(
        allocator,
        episode,
        total_timesteps=6,
        lr=1e-2,
        ppo_epochs=2,
        minibatch_size=3,
        log_interval=100,
    )

    assert result is allocator
    assert allocator.is_trained
    assert allocator.ppo_updates.item() == 4
    assert allocator.last_training_metrics["updates"] == 2.0
