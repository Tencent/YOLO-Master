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


def test_molora_small_batch_keeps_dense_fast_path():
    layer = MoLoRALayer(nn.Linear(16, 16), r=2, num_experts=4, top_k=1).eval()
    layer(torch.randn(2, 16))
    assert layer._last_dispatch_stats["mode"] == "dense_small_batch"
