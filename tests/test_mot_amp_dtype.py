"""Regression tests for MoT expert-blending dtype alignment."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ultralytics.nn.modules.mot import MoTBlock
from ultralytics.nn.tasks import DetectionModel


ROOT = Path(__file__).resolve().parents[1]
MOT_CONFIG = ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-n.yaml"
CUDA_AVAILABLE = torch.cuda.is_available()


class _FloatExpert(nn.Module):
    """Return a predictable FP32 tensor regardless of input dtype."""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() * self.scale


def _lightweight_block() -> MoTBlock:
    block = MoTBlock(8, num_heads=2, top_k=2, window_size=2, n_points=1)
    block.experts = nn.ModuleList((_FloatExpert(1.0), _FloatExpert(2.0), _FloatExpert(3.0)))
    return block


def test_mot_blend_fp32_matches_reference_and_preserves_routing():
    """FP32 blending must remain mathematically identical and leave routing untouched."""
    torch.manual_seed(54)
    block = _lightweight_block().train()
    x = torch.randn(2, 8, 4, 4)
    weights, indices = block.router(x)
    weights_before = weights.clone()
    indices_before = indices.clone()

    actual = block._blend_experts(x, weights, indices)
    expected = sum(expert(x) * weights[:, i : i + 1] for i, expert in enumerate(block.experts))

    assert actual.shape == x.shape
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert torch.equal(weights, weights_before)
    assert torch.equal(indices, indices_before)
    assert torch.allclose(actual, expected, rtol=0, atol=0)


def test_mot_blend_fp32_router_weights_with_fp32_experts():
    """The ordinary FP32 expert-output path must remain finite and shape preserving."""
    block = _lightweight_block().eval()
    x = torch.randn(1, 8, 3, 3)
    weights = torch.full((1, 3, 3, 3), 1 / 3)
    indices = torch.tensor([0, 1]).view(1, 2, 1, 1).expand(1, 2, 3, 3)

    output = block._blend_experts(x, weights, indices)

    assert output.shape == x.shape
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for FP16 expert-blending regression")
def test_mot_sparse_blend_aligns_fp32_probabilities_with_fp16_experts():
    """Blend weights are aligned locally without mutating FP32 router probabilities."""

    class _HalfExpert(nn.Module):
        def __init__(self, scale: float):
            super().__init__()
            self.scale = scale

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.half() * self.scale

    block = _lightweight_block().cuda().eval()
    block.experts = nn.ModuleList((_HalfExpert(1.0), _HalfExpert(2.0), _HalfExpert(3.0)))
    x = torch.randn(1, 8, 3, 3, device="cuda", dtype=torch.float16)
    weights = torch.full((1, 3, 3, 3), 1 / 3, device="cuda", dtype=torch.float32)
    weights_before = weights.clone()
    indices = torch.tensor([0, 1], device="cuda").view(1, 2, 1, 1).expand(1, 2, 3, 3)

    output = block._blend_experts(x, weights, indices)

    assert output.shape == x.shape
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    assert torch.equal(weights, weights_before)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for AMP forward/backward regression")
def test_mot_cuda_amp_forward_backward_is_finite_and_stateless():
    """CUDA AMP must support sparse eval, dense train, backward, and repeated forward."""
    torch.manual_seed(54)
    torch.cuda.manual_seed_all(54)
    block = MoTBlock(16, num_heads=4, top_k=2, window_size=4, n_points=2).cuda()
    x = torch.randn(1, 16, 8, 8, device="cuda", dtype=torch.float16, requires_grad=True)
    initial_hook_count = sum(len(module._forward_hooks) for module in block.modules())

    block.train()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        train_output, aux = block(x)
        loss = train_output.square().mean() + aux
    loss.backward()

    assert train_output.shape == x.shape
    assert train_output.dtype == torch.float32
    assert torch.isfinite(train_output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in block.parameters())

    block.eval()
    x_eval = x.detach()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        first, _ = block(x_eval)
        second, _ = block(x_eval)

    assert first.shape == x.shape
    assert first.dtype == torch.float32
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert block.training is False
    assert sum(len(module._forward_hooks) for module in block.modules()) == initial_hook_count
    assert not torch.is_autocast_enabled("cuda")


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for official MoT AMP integration")
def test_official_mot_model_cuda_amp_eval_forward():
    """The official MoT YAML must complete the formerly failing sparse AMP eval path."""
    torch.manual_seed(54)
    torch.cuda.manual_seed_all(54)
    model = DetectionModel(str(MOT_CONFIG), ch=3, nc=10, verbose=False).cuda().eval()
    x = torch.randn(1, 3, 64, 64, device="cuda")

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(x)

    predictions = output[0] if isinstance(output, tuple) else output
    assert torch.isfinite(predictions).all()
    assert model.training is False
    assert not torch.is_autocast_enabled("cuda")
