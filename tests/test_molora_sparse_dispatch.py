import torch
import torch.nn as nn

from ultralytics.nn.peft.molora.layer import MoLoRALayer


def test_molora_grouped_dispatch_records_actual_calls():
    torch.manual_seed(0)
    layer = MoLoRALayer(nn.Linear(16, 16), r=2, num_experts=4, top_k=1).eval()
    layer(torch.randn(8, 16))
    stats = layer._last_dispatch_stats
    assert stats["mode"] == "grouped_sparse"
    assert 1 <= stats["expert_calls"] <= 4
    assert stats["selected_samples"] == 8


def test_molora_small_batch_falls_back_to_dense_when_not_vectorizable():
    layer = MoLoRALayer(nn.Linear(16, 16), r=2, num_experts=4, top_k=1).eval()
    layer._can_vectorize_linear_experts = lambda: False
    layer(torch.randn(2, 16))
    assert layer._last_dispatch_stats["mode"] == "dense_small_batch"


def test_molora_all_linear_experts_use_batched_dispatch_without_changing_values():
    torch.manual_seed(1)
    layer = MoLoRALayer(nn.Linear(8, 6), r=2, num_experts=3, top_k=3).eval()
    x = torch.randn(5, 8)
    weights = torch.softmax(torch.randn(5, 3), dim=-1)
    indices = torch.tensor([[0, 1, 2]]).expand(5, -1)

    batched = layer._compute_sparse_experts(x, weights, indices, torch.zeros(5, 6))
    reference = sum(weights[:, i : i + 1] * layer.experts[i](x) for i in range(3))

    assert layer._last_dispatch_stats["mode"] == "batched_dense_linear"
    assert torch.allclose(batched, reference, atol=1e-6, rtol=1e-5)


def test_molora_small_batch_uses_vectorized_dense_linear_path():
    torch.manual_seed(1)
    layer = MoLoRALayer(nn.Linear(16, 16), r=2, num_experts=4, top_k=2).eval()
    x = torch.randn(2, 16)
    output = layer(x)
    stats = layer._last_dispatch_stats

    assert output.shape == (2, 16)
    assert stats["mode"] == "vectorized_dense_linear"
    assert stats["expert_calls"] == 4
    assert stats["dispatch_shape"] == (2, 4)


def test_molora_vectorized_dense_linear_matches_sparse_fallback():
    torch.manual_seed(2)
    layer = MoLoRALayer(nn.Linear(16, 12), r=3, num_experts=4, top_k=2).eval()
    x = torch.randn(2, 16)
    router_logits = layer.router(x)
    router_probs = torch.softmax(router_logits, dim=-1)
    weights, indices = torch.topk(router_probs, 2, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    template = layer.base_layer(x)

    dense = layer._compute_vectorized_linear_experts(x, weights, indices, template)
    dense_stats = dict(layer._last_dispatch_stats)
    original_guard = layer._can_vectorize_linear_experts
    layer._can_vectorize_linear_experts = lambda: False
    sparse = layer._compute_sparse_experts(x, weights, indices, template)
    layer._can_vectorize_linear_experts = original_guard

    torch.testing.assert_close(dense, sparse)
    assert dense_stats["mode"] == "vectorized_dense_linear"
    assert layer._last_dispatch_stats["mode"] == "dense_small_batch"
    assert layer._last_dispatch_stats["dispatch_shape"][0] <= x.shape[0] * layer.top_k
