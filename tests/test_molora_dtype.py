"""Regression tests for MoLoRA mixed-precision execution."""

import pytest
import torch
import torch.nn as nn

from ultralytics.nn.peft.molora.layer import MoLoRAExpert


@pytest.mark.parametrize("base_layer, input_shape", [
    (nn.Conv2d(3, 8, 3, padding=1), (2, 3, 8, 8)),
    (nn.Linear(16, 4), (2, 16)),
])
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
