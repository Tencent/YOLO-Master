"""Regression tests for MoLoRA mixed-precision execution."""

import pytest
import torch
import torch.nn as nn

from ultralytics.nn.peft.molora.layer import MoLoRAExpert, MoLoRALayer


@pytest.mark.parametrize(
    "base_layer, input_shape",
    [
        (nn.Conv2d(3, 8, 3, padding=1), (2, 3, 8, 8)),
        (nn.Linear(16, 4), (2, 16)),
    ],
)
def test_molora_expert_matches_adapter_dtype(base_layer, input_shape):
    """Low-rank execution should support half parameters without dtype errors."""
    expert = MoLoRAExpert(base_layer, r=2, alpha=4).half()
    x = torch.randn(*input_shape, dtype=torch.float16)

    out = expert(x)

    assert out.dtype == torch.float16
    assert all(p.dtype == torch.float16 for p in expert.parameters())


def test_molora_expert_float32_params_accept_low_precision_input():
    """AMP-style low-precision activations should use float32 adapter weights."""
    expert = MoLoRAExpert(nn.Linear(8, 4), r=2, alpha=4)
    x = torch.randn(2, 8, dtype=torch.float16)

    out = expert(x)

    assert out.dtype == torch.float16
    assert all(p.dtype == torch.float32 for p in expert.parameters())


def test_molora_grouped_sparse_dispatch_matches_half_output_template():
    """FP32 router weights must not promote an FP16 index_add contribution."""
    layer = MoLoRALayer(nn.Linear(8, 6), r=2, num_experts=4, top_k=2).eval()
    x = torch.randn(5, 8, dtype=torch.float16)
    weights = torch.softmax(torch.randn(5, 2, dtype=torch.float32), dim=-1)
    indices = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]])
    template = torch.zeros(5, 6, dtype=torch.float16)

    output = layer._compute_sparse_experts(x, weights, indices, template)

    assert output.dtype == torch.float16
    assert output.shape == template.shape
    assert layer._last_dispatch_stats["mode"] == "grouped_sparse"
