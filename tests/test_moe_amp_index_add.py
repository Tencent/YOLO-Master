"""Regression: AMP dtype mismatch in MoE sparse index_add_ paths.

Under autocast, GroupNorm/BatchNorm promote activations to fp32 while sparse
accumulators often stay fp16. ``index_add_`` then raises:

    RuntimeError: index_add_(): self (Half) and source (Float) must have the same scalar type
"""

import pytest
import torch
import torch.nn as nn

from ultralytics.nn.modules.moe import _common
from ultralytics.nn.modules.moe.experts import SharedInvertedExpertGroup
from ultralytics.nn.modules.moe.utils import index_add_aligned_


def _require_cpu_fp16_nn() -> None:
    """Skip only when this PyTorch build lacks the CPU fp16 kernels exercised here."""
    sample = torch.ones(1, 2, 2, 2, dtype=torch.float16)
    try:
        torch.softmax(sample, dim=1)
        nn.SiLU()(sample)
        nn.GroupNorm(1, 2)(sample)
    except RuntimeError as exc:
        if "not implemented for 'Half'" in str(exc):
            pytest.skip(f"CPU fp16 operator unavailable: {exc}")
        raise


class _Fp32Proj(nn.Module):
    """Force projection outputs to fp32 to simulate AMP GroupNorm promotion."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, t):
        return self.inner(t).float()


def test_autocast_supports_legacy_torch_cuda_api(monkeypatch):
    sentinel = object()
    calls = []

    def legacy_autocast(*, enabled):
        calls.append(enabled)
        return sentinel

    monkeypatch.setattr(_common, "_device_autocast", None)
    monkeypatch.setattr(_common, "_cuda_autocast", legacy_autocast)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert _common.autocast(enabled=True) is sentinel
    assert calls == [True]


def test_index_add_aligned_casts_float_into_half():
    acc = torch.zeros(3, 2, 2, dtype=torch.float16)
    src = torch.ones(2, 2, 2, dtype=torch.float32) * 0.5
    idx = torch.tensor([0, 2], dtype=torch.long)
    index_add_aligned_(acc, 0, idx, src)
    assert acc.dtype == torch.float16
    assert float(acc[0, 0, 0]) == 0.5
    assert float(acc[2, 0, 0]) == 0.5


def test_shared_inverted_index_add_amp_fp16_safe():
    _require_cpu_fp16_nn()
    torch.manual_seed(0)
    B, C, H, W, E, K = 4, 32, 8, 8, 4, 2
    group = SharedInvertedExpertGroup(C, C, num_experts=E, top_k=K).train().half()
    group.expert_projections = nn.ModuleList([_Fp32Proj(p) for p in group.expert_projections])
    x = torch.randn(B, C, H, W, dtype=torch.float16)
    w = torch.softmax(torch.randn(B, K, dtype=torch.float16), dim=-1)
    idx = torch.randint(0, E, (B, K), dtype=torch.long)
    out = group(x, w, idx, K)
    assert out.dtype == torch.float16
    assert out.shape == (B, C, H, W)
    assert torch.isfinite(out.float()).all()


def test_gated_diversified_expert_index_add_amp_fp16_safe():
    """Crash site from training stack: gated.py Heterogeneous/Diversified path."""
    from ultralytics.nn.modules.moe.gated import DiversifiedExpertGroup

    _require_cpu_fp16_nn()

    torch.manual_seed(0)
    B, C, H, W, E, K = 4, 32, 8, 8, 4, 2
    group = DiversifiedExpertGroup(C, C, num_experts=E, top_k=K).train().half()
    group.expert_projections = nn.ModuleList([_Fp32Proj(p) for p in group.expert_projections])
    x = torch.randn(B, C, H, W, dtype=torch.float16)
    w = torch.softmax(torch.randn(B, K, dtype=torch.float16), dim=-1)
    idx = torch.randint(0, E, (B, K), dtype=torch.long)
    out = group(x, w, idx, K)
    assert out.dtype == torch.float16
    assert torch.isfinite(out.float()).all()


def test_adaptive_gate_moe_half_forward_backward():
    from ultralytics.nn.modules.moe import AdaptiveGateMoE

    _require_cpu_fp16_nn()

    torch.manual_seed(0)
    moe = AdaptiveGateMoE(32, 32, num_experts=4, top_k=2).train().half()
    fused = moe.fused_experts
    if hasattr(fused, "expert_projections"):
        fused.expert_projections = nn.ModuleList([_Fp32Proj(p) for p in fused.expert_projections])
    x = torch.randn(2, 32, 8, 8, dtype=torch.float16, requires_grad=True)
    out = moe(x)
    assert out.dtype == torch.float16
    out.float().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad.float()).all()
