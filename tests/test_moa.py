from pathlib import Path

import torch

from ultralytics.nn.modules.moa import C2fMoA, MoABlock, NeckMoAFusion, anneal_moa_temperature, collect_moa_aux_loss
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import _collect_moa_aux_loss


ROOT = Path(__file__).resolve().parents[1]


def _has_grad(module):
    return any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in module.parameters()
        if p.requires_grad
    )


def test_moa_modules_forward_backward():
    torch.manual_seed(0)
    cases = [
        (MoABlock(48, num_heads=6), torch.randn(2, 48, 8, 8), (2, 48, 8, 8)),
        (C2fMoA(64, 64, n=2, num_heads=6), torch.randn(2, 64, 8, 8), (2, 64, 8, 8)),
    ]
    for module, x, expected_shape in cases:
        module.train()
        out = module(x)
        assert out.shape == expected_shape
        out.mean().backward()
        assert _has_grad(module)


def test_neck_moa_fusion_forward_backward():
    torch.manual_seed(0)
    module = NeckMoAFusion(64, 128, 64, num_heads=4).train()
    out = module(torch.randn(2, 64, 16, 16), torch.randn(2, 128, 8, 8))
    assert out.shape == (2, 64, 16, 16)
    out.mean().backward()
    assert _has_grad(module)


def test_moa_aux_loss_collected_for_c2f_and_neck():
    torch.manual_seed(0)
    c2f = C2fMoA(32, 32, n=1, num_heads=3).train()
    out = c2f(torch.randn(2, 32, 6, 6))
    aux = collect_moa_aux_loss(c2f)
    assert out.shape == (2, 32, 6, 6)
    assert aux.requires_grad and torch.isfinite(aux)
    assert _collect_moa_aux_loss(c2f, torch.device("cpu")).requires_grad

    neck = NeckMoAFusion(32, 64, 32, num_heads=2).train()
    neck(torch.randn(2, 32, 8, 8), torch.randn(2, 64, 4, 4))
    neck_aux = collect_moa_aux_loss(neck)
    assert neck_aux.requires_grad and torch.isfinite(neck_aux)


def test_c2fmoa_small_channels_keep_valid_head_count():
    module = C2fMoA(8, 8, n=1, num_heads=6, e=0.5).train()
    out = module(torch.randn(1, 8, 4, 4))
    assert out.shape == (1, 8, 4, 4)


def test_moa_temperature_anneal():
    module = C2fMoA(64, 64, n=1, num_heads=6)
    before = [m.router.temperature for m in module.m]
    anneal_moa_temperature(module, factor=0.5, min_temp=0.3)
    after = [m.router.temperature for m in module.m]
    assert after == [max(t * 0.5, 0.3) for t in before]


def test_moa_model_configs_parse():
    configs = [
        ROOT / "ultralytics/cfg/models/master/v0_8/det/yolo-master-moa-n.yaml",
        ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-n.yaml",
    ]
    for cfg in configs:
        model = DetectionModel(str(cfg), ch=3, nc=80, verbose=False)
        assert sum(isinstance(m, C2fMoA) for m in model.modules()) == 3
        assert sum(isinstance(m, MoABlock) for m in model.modules()) == 6
