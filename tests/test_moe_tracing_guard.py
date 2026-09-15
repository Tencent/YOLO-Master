# 🐧Please note that this file has been modified by Tencent on 2026/08/30. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Tracing must not freeze one batch's routing into the graph."""

import pytest
import torch

from ultralytics.nn.modules.moe.modules import OptimizedMOE, OptimizedMOEImproved


def _distinguishable(block):
    """Give every expert a different constant so a wrong route shows up in the output."""
    with torch.no_grad():
        for index, expert in enumerate(block.experts):
            for parameter in expert.parameters():
                parameter.mul_(0).add_((index + 1) * 0.05)
    return block.eval()


@pytest.mark.parametrize("cls", [OptimizedMOE, OptimizedMOEImproved])
def test_torchscript_trace_matches_eager_for_other_routes(cls):
    """torch.jit.trace records one batch's Top-K decision; the dense path keeps it faithful."""
    torch.manual_seed(0)
    block = _distinguishable(cls(in_channels=16, out_channels=16, num_experts=4, top_k=1))
    samples = [torch.randn(1, 16, 8, 8) * (1 + 3 * scale) for scale in range(6)]

    traced = torch.jit.trace(block, samples[0], check_trace=False)
    for sample in samples:
        with torch.no_grad():
            eager, scripted = block(sample), traced(sample)
        if isinstance(eager, tuple):
            eager, scripted = eager[0], scripted[0]
        assert torch.allclose(eager, scripted, atol=1e-4), "traced graph routes differently from eager"
