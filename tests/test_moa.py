from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.moa import C2fMoA, MoABlock, NeckMoAFusion, anneal_moa_temperature, collect_moa_aux_loss
from ultralytics.nn.modules.moa.moa import _GlobalAttnHead, _LocalAttnHead, _MoARouter, _flash_attn
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
    """NeckMoAFusion should align lo to hi size even when scales are not exactly 2x."""
    torch.manual_seed(0)
    module = NeckMoAFusion(32, 64, 32, num_heads=4).train()
    hi = torch.randn(2, 32, 15, 15)
    lo = torch.randn(2, 64, 7, 7)

    out = module(hi, lo)

    assert out.shape == hi.shape
    assert torch.isfinite(out).all()
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


def test_moa_router_extreme_temperature_stays_finite_and_nonuniform():
    """Very low router temperature should not create NaN/Inf or collapse to uniform probabilities."""
    torch.manual_seed(0)
    block = MoABlock(48, num_heads=6, temperature=1.0).train()
    block.router.temperature = 1e-6
    with torch.no_grad():
        block.router.router[-1].weight.normal_(0, 0.02)
        block.router.router[-1].bias.copy_(torch.tensor([0.25, -0.15, 0.05]))

    probs, logits = block.router(torch.randn(2, 48, 5, 5), return_logits=True)

    assert torch.isfinite(logits).all()
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=1), torch.ones_like(probs[:, 0]), atol=1e-5)
    uniform = probs.new_full(probs.shape, 1.0 / probs.shape[1])
    assert not torch.allclose(probs, uniform, atol=1e-3)


def test_attention_heads_handle_dim_not_divisible_by_heads():
    """Local/global heads should degrade safely when dim is not divisible by num_heads."""
    x = torch.randn(1, 10, 4, 4)
    for cls in (_LocalAttnHead, _GlobalAttnHead):
        module = cls(dim=10, num_heads=3).train()
        out = module(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


def test_flash_attention_fallback_without_sdpa(monkeypatch):
    """The manual attention path should remain finite when SDPA is unavailable."""
    monkeypatch.delattr(F, "scaled_dot_product_attention", raising=False)
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

    out = _flash_attn(q, k, v, scale=8**-0.5)

    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def test_global_head_large_map_uses_linear_attention_path():
    module = _GlobalAttnHead(dim=10, num_heads=3, nb_features=8).eval()
    x = torch.randn(1, 10, 17, 17)

    out = module(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_moa_router_probs_without_logits_are_finite_and_normalized():
    router = _MoARouter(dim=8, num_groups=3).train()
    probs = router(torch.randn(2, 8, 3, 3))

    assert probs.shape == (2, 3, 3, 3)
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=1), torch.ones_like(probs[:, 0]), atol=1e-6)


def test_moablock_without_shortcut_keeps_shape_and_aux_loss():
    module = MoABlock(48, num_heads=6, shortcut=False).train()
    x = torch.randn(1, 48, 4, 4)

    out = module(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert module.last_aux_loss.requires_grad
    assert torch.isfinite(module.last_aux_loss)


def test_c2fmoa_aux_loss_not_double_counted_for_nested_blocks():
    """C2fMoA wrapper aux loss should equal child block losses once, not wrapper + children twice."""
    torch.manual_seed(0)
    module = C2fMoA(32, 32, n=3, num_heads=3).train()
    module(torch.randn(2, 32, 6, 6))

    child_total = sum(m.last_aux_loss for m in module.m)
    collected = collect_moa_aux_loss(module)
    public_collected = _collect_moa_aux_loss(module, torch.device("cpu"))

    assert torch.isfinite(collected)
    assert torch.allclose(collected, child_total)
    assert torch.allclose(public_collected, child_total)
    assert not torch.allclose(collected, child_total * 2)


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
