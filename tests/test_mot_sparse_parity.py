from unittest.mock import patch

import torch

from ultralytics.nn.modules.mot import MoTBlock


def test_mot_sparse_and_dense_eval_are_close():
    torch.manual_seed(0)
    block = MoTBlock(24, num_heads=3, top_k=1, sparse_train=False).eval()
    x = torch.randn(4, 24, 4, 4)
    with torch.no_grad():
        weights, indices, _ = block.router(x, return_logits=True)
        dense_mixture = sum(
            (expert(x) * weights[:, i : i + 1] for i, expert in enumerate(block.experts)), torch.zeros_like(x)
        )
        out_dense = block.out_norm(block.out_proj(dense_mixture)) + x
        out_sparse, _ = block(x)
    assert torch.isfinite(out_sparse).all()
    assert out_dense.shape == out_sparse.shape
    assert torch.allclose(out_dense, out_sparse, atol=1e-5, rtol=1e-4)
    assert block._last_dispatch_stats["mode"] == "sample_sparse"


def test_mot_dense_export_path_records_all_experts():
    block = MoTBlock(24, num_heads=3, top_k=1, sparse_train=False).train()
    block(torch.randn(2, 24, 4, 4))
    assert block._last_dispatch_stats["mode"] == "dense"
    assert block._last_dispatch_stats["expert_calls"] == len(block.experts)
    assert block.last_routing_snapshot["dispatch"]["policy"] == "dense_train"


def test_mot_sparse_dispatch_reports_policy_and_sparsity():
    block = MoTBlock(24, num_heads=3, top_k=1, sparse_train=True, exploration_eps=0.0, balance_loss_coeff=0.0).train()
    block(torch.randn(2, 24, 4, 4))
    dispatch = block.last_routing_snapshot["dispatch"]

    assert dispatch["policy"] == "sparse_train"
    assert 0.0 <= dispatch["sparsity_ratio"] <= 1.0
    assert dispatch["selected_experts"] <= block.num_experts
    assert block.export_capabilities()["sparse_train"] is True


def test_mot_sparse_training_uses_persistent_dense_warmup_before_switching():
    block = MoTBlock(
        24,
        num_heads=3,
        top_k=1,
        sparse_train=True,
        exploration_eps=0.0,
        sparse_train_warmup_steps=2,
    ).train()
    x = torch.randn(2, 24, 4, 4)

    policies = []
    for _ in range(3):
        block(x)
        policies.append(block.last_routing_snapshot["dispatch"]["policy"])

    assert policies == ["dense_warmup", "dense_warmup", "sparse_train"]
    assert int(block._sparse_train_step) == 3
    assert "_sparse_train_step" in block.state_dict()
    capabilities = block.export_capabilities()
    assert capabilities["sparse_train_warmup_steps"] == 2
    assert capabilities["sparse_train_ready"] is True

    restored = MoTBlock(
        24,
        num_heads=3,
        top_k=1,
        sparse_train=True,
        exploration_eps=0.0,
        sparse_train_warmup_steps=2,
    ).train()
    restored.load_state_dict(block.state_dict())
    restored(x)
    assert restored.last_routing_snapshot["dispatch"]["policy"] == "sparse_train"

    legacy_state = block.state_dict()
    legacy_state.pop("_sparse_train_step")
    legacy = MoTBlock(24, num_heads=3, top_k=1, sparse_train=True, exploration_eps=0.0, sparse_train_warmup_steps=2)
    legacy.load_state_dict(legacy_state, strict=True)
    assert int(legacy._sparse_train_step) == 0


def test_mot_ddp_sparse_training_falls_back_until_find_unused_is_confirmed():
    block = MoTBlock(24, num_heads=3, top_k=1, sparse_train=True, balance_loss_coeff=0.0).train()
    with patch("torch.distributed.is_available", return_value=True), patch(
        "torch.distributed.is_initialized", return_value=True
    ), patch("torch.distributed.get_world_size", return_value=2):
        block(torch.randn(2, 24, 4, 4))
        dispatch = block.last_routing_snapshot["dispatch"]
        capabilities = block.export_capabilities()

    assert block.sparse_train is True
    assert dispatch["policy"] == "dense_ddp_fallback"
    assert dispatch["ddp_active"] is True
    assert dispatch["ddp_find_unused_parameters"] is None
    assert dispatch["ddp_fallback_reason"] == "find_unused_parameters_not_confirmed"
    assert capabilities["ddp_sparse_train_safe"] is False


def test_mot_ddp_sparse_training_runs_after_safe_handshake():
    block = MoTBlock(24, num_heads=3, top_k=1, sparse_train=True, exploration_eps=0.0, balance_loss_coeff=0.0).train()
    block.configure_ddp_sparse_training(find_unused_parameters=True, source="trainer")
    with patch("torch.distributed.is_available", return_value=True), patch(
        "torch.distributed.is_initialized", return_value=True
    ), patch("torch.distributed.get_world_size", return_value=2):
        block(torch.randn(2, 24, 4, 4))
        dispatch = block.last_routing_snapshot["dispatch"]

    assert dispatch["policy"] == "sparse_train"
    assert dispatch["mode"] == "sample_sparse"
    assert dispatch["ddp_find_unused_parameters"] is True
    assert dispatch["ddp_contract_source"] == "trainer"
    assert dispatch["ddp_fallback_reason"] is None


def test_mot_ddp_sparse_training_reports_disabled_find_unused():
    block = MoTBlock(24, num_heads=3, top_k=1, sparse_train=True, balance_loss_coeff=0.0).train()
    block.configure_ddp_sparse_training(find_unused_parameters=False, source="custom_ddp")
    with patch("torch.distributed.is_available", return_value=True), patch(
        "torch.distributed.is_initialized", return_value=True
    ), patch("torch.distributed.get_world_size", return_value=2):
        block(torch.randn(2, 24, 4, 4))

    dispatch = block.last_routing_snapshot["dispatch"]
    assert dispatch["policy"] == "dense_ddp_fallback"
    assert dispatch["ddp_find_unused_parameters"] is False
    assert dispatch["ddp_fallback_reason"] == "find_unused_parameters_disabled"
