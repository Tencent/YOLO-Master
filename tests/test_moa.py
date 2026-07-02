from pathlib import Path

import torch
import torch.nn.functional as F

from ultralytics.nn.modules.moa import C2fMoA, MoABlock, NeckMoAFusion, anneal_moa_temperature, collect_moa_aux_loss
from ultralytics.nn.modules.moa.moa import _GlobalAttnHead, _LocalAttnHead, _flash_attn
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


def test_neck_moa_fusion_handles_non_strict_scale_ratio():
    torch.manual_seed(0)
    module = NeckMoAFusion(64, 128, 64, num_heads=4).train()
    out = module(torch.randn(1, 64, 15, 15), torch.randn(1, 128, 7, 7))
    assert out.shape == (1, 64, 15, 15)
    assert torch.isfinite(out).all()
    assert module.last_aux_loss.requires_grad and torch.isfinite(module.last_aux_loss)
    (out.mean() + module.last_aux_loss).backward()
    assert _has_grad(module)


def test_moa_router_stays_finite_at_tiny_annealed_temperature():
    torch.manual_seed(0)
    module = MoABlock(48, num_heads=6, temperature=1.0).train()
    anneal_moa_temperature(module, factor=1e-8, min_temp=1e-8)
    assert module.router.temperature < 1e-4

    with torch.no_grad():
        module.router.router[-1].bias.copy_(torch.tensor([0.0, 1e-4, -1e-4]))

    x = torch.randn(1, 48, 4, 4)
    probs, logits = module.router(x, return_logits=True)
    assert torch.isfinite(probs).all()
    assert torch.isfinite(logits).all()
    assert torch.allclose(probs.sum(dim=1), torch.ones_like(probs[:, 0]), atol=1e-6)
    assert not torch.allclose(probs, torch.full_like(probs, 1.0 / probs.shape[1]))

    out = module(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert torch.isfinite(module.last_aux_loss)


def test_moa_attention_heads_handle_non_divisible_dim_and_heads():
    torch.manual_seed(0)
    cases = [
        (_LocalAttnHead(17, num_heads=5), torch.randn(2, 17, 5, 5)),
        (_GlobalAttnHead(17, num_heads=5), torch.randn(1, 17, 17, 17)),
    ]
    for module, x in cases:
        module.train()
        out = module(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
        assert module.norm.num_channels % module.norm.num_groups == 0
        out.mean().backward()
        assert _has_grad(module)


def test_c2fmoa_aux_loss_not_double_counted_for_nested_blocks():
    torch.manual_seed(0)
    module = C2fMoA(32, 32, n=3, num_heads=6).train()
    out = module(torch.randn(2, 32, 6, 6))
    assert out.shape == (2, 32, 6, 6)

    inner_sum = torch.stack([m.last_aux_loss for m in module.m]).sum()
    collected = collect_moa_aux_loss(module)
    loss_collector = _collect_moa_aux_loss(module, torch.device("cpu"))
    naive_sum = torch.stack(
        [m.last_aux_loss for m in module.modules() if isinstance(getattr(m, "last_aux_loss", None), torch.Tensor)]
    ).sum()

    assert inner_sum.requires_grad and torch.isfinite(inner_sum)
    assert torch.allclose(module.last_aux_loss, inner_sum)
    assert torch.allclose(collected, inner_sum)
    assert torch.allclose(loss_collector, inner_sum)
    assert naive_sum > collected * 1.5


def test_flash_attn_supports_sdpa_without_scale_keyword(monkeypatch):
    original_sdpa = F.scaled_dot_product_attention

    def torch20_sdpa(q, k, v):
        return original_sdpa(q, k, v)

    monkeypatch.setattr("ultralytics.nn.modules.moa.moa.F.scaled_dot_product_attention", torch20_sdpa)
    q = torch.randn(1, 2, 4, 4)
    k = torch.randn(1, 2, 4, 4)
    v = torch.randn(1, 2, 4, 4)
    scale = 0.25
    out = _flash_attn(q, k, v, scale=scale)
    expected = ((q @ k.transpose(-2, -1)) * scale).softmax(dim=-1) @ v
    assert torch.allclose(out, expected)


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
