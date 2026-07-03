from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.moa import C2fMoA, MoABlock, NeckMoAFusion, anneal_moa_temperature, collect_moa_aux_loss
from ultralytics.nn.modules.moa.moa import _GlobalAttnHead, _LocalAttnHead
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


def test_neck_moa_fusion_handles_non_2x_spatial_mismatch():
    torch.manual_seed(0)
    module = NeckMoAFusion(32, 48, 32, num_heads=4).train()
    hi = torch.randn(2, 32, 15, 15)
    lo = torch.randn(2, 48, 7, 7)
    out = module(hi, lo)
    assert out.shape == hi.shape
    assert torch.isfinite(out).all()
    out.mean().backward()
    assert _has_grad(module)


def test_moa_router_tiny_temperature_remains_finite_and_non_uniform():
    torch.manual_seed(0)
    block = MoABlock(48, num_heads=6).train()
    block.router.temperature = 1e-6
    with torch.no_grad():
        block.router.router[-1].bias.copy_(torch.tensor([-0.25, 0.0, 0.25]))

    x = torch.randn(2, 48, 5, 5)
    probs, logits = block.router(x, return_logits=True)
    uniform = torch.full_like(probs, 1.0 / probs.shape[1])

    assert torch.isfinite(logits).all()
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=1), torch.ones_like(probs[:, 0]), atol=1e-6)
    assert not torch.allclose(probs, uniform, atol=1e-3)

    out = block(x)
    assert torch.isfinite(out).all()


def test_attention_heads_degrade_when_heads_do_not_divide_dim():
    torch.manual_seed(0)
    x = torch.randn(1, 30, 6, 6)
    for head in (_LocalAttnHead(30, num_heads=8), _GlobalAttnHead(30, num_heads=8)):
        out = head(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


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


def test_c2fmoa_aux_loss_does_not_double_count_nested_blocks():
    torch.manual_seed(0)
    module = C2fMoA(32, 32, n=3, num_heads=6).train()
    module(torch.randn(2, 32, 6, 6))

    block_total = sum((m.last_aux_loss for m in module.m), module.last_aux_loss.new_zeros(()))
    collected = collect_moa_aux_loss(module)
    collected_via_loss = _collect_moa_aux_loss(module, torch.device("cpu"))
    legacy_recursive_total = module.last_aux_loss + block_total

    assert module.last_aux_loss.requires_grad
    assert torch.allclose(module.last_aux_loss, block_total)
    assert torch.allclose(collected, module.last_aux_loss)
    assert torch.allclose(collected_via_loss, module.last_aux_loss)
    assert legacy_recursive_total > collected * 1.5


def test_c2fmoa_small_channels_keep_valid_head_count():
    module = C2fMoA(8, 8, n=1, num_heads=6, e=0.5).train()
    out = module(torch.randn(1, 8, 4, 4))
    assert out.shape == (1, 8, 4, 4)


def test_c2fmoa_rounds_head_count_to_expert_groups():
    module = C2fMoA(256, 256, n=1, num_heads=4, e=0.5).train()
    out = module(torch.randn(1, 256, 2, 2))

    assert out.shape == (1, 256, 2, 2)
    assert module.m[0].local_head.num_heads == 2


def test_neck_moa_fusion_eval_projects_self_path_and_zero_aux_loss():
    module = NeckMoAFusion(16, 32, 24, num_heads=4).eval()
    out = module(torch.randn(1, 16, 5, 5), torch.randn(1, 32, 3, 3))

    assert out.shape == (1, 24, 5, 5)
    assert torch.isfinite(out).all()
    assert module.last_aux_loss.device == out.device
    assert module.last_aux_loss.item() == 0


def test_collect_moa_aux_loss_handles_empty_module_and_standalone_block():
    empty_aux = collect_moa_aux_loss(nn.Identity())
    assert empty_aux.device.type == "cpu"
    assert empty_aux.item() == 0

    module = MoABlock(48, num_heads=6).train()
    module(torch.randn(1, 48, 4, 4))
    aux = collect_moa_aux_loss(module)

    assert torch.isfinite(aux)
    assert torch.allclose(aux, module.last_aux_loss)


def test_moa_temperature_anneal():
    module = C2fMoA(64, 64, n=1, num_heads=6)
    before = [m.router.temperature for m in module.m]
    anneal_moa_temperature(module, factor=0.5, min_temp=0.3)
    after = [m.router.temperature for m in module.m]
    assert after == [max(t * 0.5, 0.3) for t in before]


def test_moa_global_head_per_block_seed():
    """Relocated from test_mot.py — verifies per-block RF seed diversity."""
    b0 = MoABlock(64, num_heads=6, block_index=0)
    b1 = MoABlock(64, num_heads=6, block_index=1)
    assert not torch.allclose(b0.global_head._rf_matrix, b1.global_head._rf_matrix)


def test_moa_model_configs_parse():
    configs = [
        ROOT / "ultralytics/cfg/models/master/v0_8/det/yolo-master-moa-n.yaml",
        ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-n.yaml",
    ]
    for cfg in configs:
        model = DetectionModel(str(cfg), ch=3, nc=80, verbose=False)
        assert sum(isinstance(m, C2fMoA) for m in model.modules()) == 3
        assert sum(isinstance(m, MoABlock) for m in model.modules()) == 6
