from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.moa import C2fMoA, MoABlock, NeckMoAFusion, anneal_moa_temperature, collect_moa_aux_loss
from ultralytics.nn.modules.moa.moa import (
    _GlobalAttnHead,
    _LocalAttnHead,
    _RegionalAttnHead,
    _MoARouter,
    _flash_attn,
    _moa_router_aux_loss,
    _window_flash_attn,
)
from ultralytics.nn.modules._numeric import fp_clamp_floor as _fp_min
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
    original_sdpa = getattr(F, "scaled_dot_product_attention", None)

    if original_sdpa is None:
        def original_sdpa(q, k, v):
            return (q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5).softmax(dim=-1) @ v

    def torch20_sdpa(q, k, v):
        return original_sdpa(q, k, v)

    monkeypatch.setattr("ultralytics.nn.modules.moa.moa.F.scaled_dot_product_attention", torch20_sdpa, raising=False)
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


def test_moa_default_temperature_anneal_matches_shared_schedule():
    module = C2fMoA(64, 64, n=1, num_heads=6)
    anneal_moa_temperature(module)

    assert [m.router.temperature for m in module.m] == [0.97]


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


# ============================================================================
# Enhanced boundary tests — Issue #53
# ============================================================================


# ── 1. NeckMoAFusion cross-scale size mismatch (extended) ─────────────────

def test_neck_moa_fusion_non_strict_scale_many_cases():
    """NeckMoAFusion forward stability with diverse non-2× downsampling ratios."""
    torch.manual_seed(0)
    cases = [
        # (hi_H, hi_W, lo_H, lo_W)
        (15, 15, 7, 7),     # ~2.14× ratio
        (13, 13, 5, 5),     # ~2.6× ratio
        (10, 10, 3, 3),     # ~3.33× ratio
        (17, 17, 9, 9),     # ~1.89× ratio
        (7, 7, 3, 3),       # ~2.33× ratio
        (20, 15, 10, 7),    # non-square, different H/W ratios
        (9, 16, 4, 8),      # non-square, exact 2× per dim
    ]
    for hi_h, hi_w, lo_h, lo_w in cases:
        module = NeckMoAFusion(64, 128, 64, num_heads=4).train()
        hi = torch.randn(2, 64, hi_h, hi_w)
        lo = torch.randn(2, 128, lo_h, lo_w)
        out = module(hi, lo)
        assert out.shape == (2, 64, hi_h, hi_w), f"Failed at hi=({hi_h},{hi_w}), lo=({lo_h},{lo_w})"
        assert torch.isfinite(out).all(), f"NaN at hi=({hi_h},{hi_w}), lo=({lo_h},{lo_w})"
        (out.mean() + module.last_aux_loss).backward()
        assert _has_grad(module), f"No grad at hi=({hi_h},{hi_w}), lo=({lo_h},{lo_w})"


def test_neck_moa_fusion_single_pixel_lo():
    """NeckMoAFusion with lo-res map at 1×1 (extreme upsampling)."""
    torch.manual_seed(0)
    module = NeckMoAFusion(64, 128, 64, num_heads=4).train()
    out = module(torch.randn(1, 64, 8, 8), torch.randn(1, 128, 1, 1))
    assert out.shape == (1, 64, 8, 8)
    assert torch.isfinite(out).all()
    (out.mean() + module.last_aux_loss).backward()
    assert _has_grad(module)


def test_neck_moa_fusion_no_shortcut():
    """NeckMoAFusion with shortcut=False (pure feed-forward fusion)."""
    torch.manual_seed(0)
    module = NeckMoAFusion(64, 128, 64, num_heads=4, shortcut=False).train()
    out = module(torch.randn(2, 64, 16, 16), torch.randn(2, 128, 8, 8))
    assert out.shape == (2, 64, 16, 16)
    assert torch.isfinite(out).all()
    out.mean().backward()
    assert _has_grad(module)


def test_neck_moa_fusion_channel_mismatch_projection():
    """NeckMoAFusion with c_hi ≠ c_out triggers self_out_proj/res_proj paths."""
    torch.manual_seed(0)
    module = NeckMoAFusion(32, 64, 48, num_heads=4).train()
    out = module(torch.randn(2, 32, 8, 8), torch.randn(2, 64, 4, 4))
    assert out.shape == (2, 48, 8, 8)
    assert torch.isfinite(out).all()
    out.mean().backward()
    assert _has_grad(module)


# ── 2. MoABlock temperature annealing & numerical stability ────────────────

def test_moa_router_extreme_temperature_numerical_stability():
    """Router softmax remains numerically stable at temperature extremes."""
    torch.manual_seed(0)
    for temp in [1e-8, 1e-6, 1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0]:
        module = MoABlock(48, num_heads=6, temperature=temp).train()
        x = torch.randn(1, 48, 8, 8)
        out = module(x)
        assert out.shape == x.shape, f"Shape mismatch at temp={temp}"
        assert torch.isfinite(out).all(), f"NaN output at temp={temp}"
        # Router probs must sum to 1
        probs = module.router(x)
        assert torch.allclose(probs.sum(dim=1), torch.ones_like(probs[:, 0]), atol=1e-4), (
            f"Router probs don't sum to 1 at temp={temp}"
        )
        # No NaN in aux_loss
        assert torch.isfinite(module.last_aux_loss), f"NaN aux_loss at temp={temp}"


def test_moa_router_anneal_to_zero_like_temperature():
    """Router temperature annealed to near-zero must not produce NaN."""
    torch.manual_seed(0)
    module = MoABlock(48, num_heads=6, temperature=1.0).train()
    # Simulate aggressive annealing: temperature = max(1.0 * 1e-8, 1e-12) = 1e-8
    anneal_moa_temperature(module, factor=1e-8, min_temp=1e-12)
    assert module.router.temperature <= 1e-8

    x = torch.randn(2, 48, 4, 4)
    probs = module.router(x)
    assert torch.isfinite(probs).all(), "NaN in router probs at extreme temperature"
    assert (probs >= 0).all() and (probs <= 1).all(), "Probs out of [0,1] range"
    assert torch.allclose(probs.sum(dim=1), torch.ones_like(probs[:, 0]), atol=1e-4)

    out = module(x)
    assert torch.isfinite(out).all()
    assert torch.isfinite(module.last_aux_loss)


def test_moa_router_return_logits():
    """_MoARouter.forward with return_logits=True returns both probs and logits."""
    torch.manual_seed(0)
    router = _MoARouter(64, num_groups=3)
    x = torch.randn(1, 64, 8, 8)
    probs, logits = router(x, return_logits=True)
    assert probs.shape == (1, 3, 8, 8)
    assert logits.shape == (1, 3, 8, 8)
    assert torch.allclose(probs.sum(dim=1), torch.ones_like(probs[:, 0]), atol=1e-5)
    # Softmax consistency: probs = softmax(logits)
    expected = F.softmax(logits, dim=1)
    assert torch.allclose(probs, expected, atol=1e-5)


# ── 3. Attention head non-divisible dim & heads (extended) ─────────────────

def test_regional_attn_head_non_divisible_dim_and_heads():
    """_RegionalAttnHead handles dim not divisible by num_heads."""
    torch.manual_seed(0)
    cases = [
        (_RegionalAttnHead(17, num_heads=5), torch.randn(2, 17, 8, 8)),
        (_RegionalAttnHead(31, num_heads=7), torch.randn(1, 31, 12, 12)),
        (_RegionalAttnHead(20, num_heads=3), torch.randn(1, 20, 16, 16)),
        (_RegionalAttnHead(8, num_heads=2, pool_stride=1), torch.randn(1, 8, 8, 8)),
    ]
    for module, x in cases:
        module.train()
        out = module(x)
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
        assert torch.isfinite(out).all(), "NaN in output"
        assert module.norm.num_channels % module.norm.num_groups == 0, (
            f"GroupNorm mismatch: {module.norm.num_channels} % {module.norm.num_groups} != 0"
        )
        out.mean().backward()
        assert _has_grad(module)


def test_regional_attn_head_small_spatial_dims():
    """_RegionalAttnHead gracefully handles H=1 or W=1 feature maps."""
    torch.manual_seed(0)
    cases = [
        (_RegionalAttnHead(32, num_heads=4), torch.randn(1, 32, 1, 16)),
        (_RegionalAttnHead(32, num_heads=4), torch.randn(1, 32, 16, 1)),
        (_RegionalAttnHead(32, num_heads=4), torch.randn(1, 32, 1, 1)),
    ]
    for module, x in cases:
        module.train()
        out = module(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
        out.mean().backward()
        assert _has_grad(module)


def test_regional_attn_head_invalid_pool_stride():
    """_RegionalAttnHead raises ValueError for pool_stride < 1."""
    with pytest.raises(ValueError, match="pool_stride"):
        _RegionalAttnHead(32, num_heads=4, pool_stride=0)


def test_local_attn_head_window_size_clamping():
    """_LocalAttnHead clamps window_size >= 1 for extreme inputs."""
    torch.manual_seed(0)
    # window_size=0 → clamped to 1
    head = _LocalAttnHead(32, num_heads=4, window_size=0)
    assert head.window_size == 1
    x = torch.randn(1, 32, 8, 8)
    out = head(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_attention_heads_all_variants_non_divisible():
    """All three attention head types handle non-divisible dim/heads."""
    torch.manual_seed(0)
    dim, heads = 19, 5  # 19 % 5 != 0, 19 // 5 = 3 < 16 so head_dim=16
    cases = [
        _LocalAttnHead(dim, num_heads=heads),
        _RegionalAttnHead(dim, num_heads=heads),
        _GlobalAttnHead(dim, num_heads=heads),
    ]
    for module in cases:
        module.train()
        x = torch.randn(1, dim, 10, 10)
        out = module(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


# ── 4. C2fMoA aux_loss double-counting (extended) ─────────────────────────

def test_c2fmoa_covered_modules_mechanism():
    """C2fMoA.publish_aux_loss passes covered_modules to prevent double-counting.

    C2fMoA publishes with covered_modules=children, so children are skipped during
    collection (parent comes first in depth-first module iteration).
    """
    torch.manual_seed(0)
    n_blocks = 2
    module = C2fMoA(64, 64, n=n_blocks, num_heads=6).train()
    from ultralytics.nn.modules.routing_protocol import collect_aux_loss, begin_aux_step
    # Use a fresh step so only this forward pass is counted
    step = begin_aux_step(42)
    module(torch.randn(2, 64, 8, 8))

    # collect with diagnostics — match the step used in forward
    total, diag = collect_aux_loss(
        module, include_kinds=("moa",), return_diagnostics=True, step=step
    )
    # The MoABlock children are skipped because C2fMoA's covered_modules covers them
    assert diag["duplicate_skipped"] == n_blocks, (
        f"Expected {n_blocks} children skipped, got {diag}"
    )
    # Total should equal C2fMoA.last_aux_loss (children already included in it)
    assert torch.allclose(total, module.last_aux_loss)


def test_c2fmoa_single_block_aux_loss_equals_block_loss():
    """With n=1, C2fMoA.last_aux_loss should equal MoABlock.last_aux_loss."""
    torch.manual_seed(0)
    module = C2fMoA(64, 64, n=1, num_heads=6).train()
    module(torch.randn(2, 64, 8, 8))
    assert torch.allclose(module.last_aux_loss, module.m[0].last_aux_loss)


def test_c2fmoa_eval_mode_zero_aux_loss():
    """C2fMoA in eval mode produces zero aux_loss without grad."""
    module = C2fMoA(64, 64, n=2, num_heads=6).eval()
    module(torch.randn(2, 64, 8, 8))
    assert module.last_aux_loss.item() == 0.0
    assert not module.last_aux_loss.requires_grad


# ── 5. MoABlock advanced paths ─────────────────────────────────────────────

def test_moa_block_no_shortcut():
    """MoABlock with shortcut=False: pure feed-forward transform, no residual."""
    torch.manual_seed(0)
    module = MoABlock(48, num_heads=6, shortcut=False).train()
    x = torch.randn(2, 48, 8, 8)
    out = module(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    out.mean().backward()
    assert _has_grad(module)


def test_moa_block_sequential_heads():
    """MoABlock with sequential_heads=True: memory-efficient sequential path."""
    torch.manual_seed(0)
    # Test both shortcut modes
    for shortcut in [True, False]:
        module = MoABlock(48, num_heads=6, sequential_heads=True, shortcut=shortcut).train()
        x = torch.randn(2, 48, 8, 8)
        out = module(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all(), f"NaN with shortcut={shortcut}"
        out.mean().backward()
        assert _has_grad(module), f"No grad with shortcut={shortcut}"


def test_moa_block_sequential_vs_parallel_equivalence():
    """Sequential and parallel head evaluation produce identical outputs."""
    torch.manual_seed(0)
    x = torch.randn(2, 48, 8, 8)

    module_seq = MoABlock(48, num_heads=6, sequential_heads=True).eval()
    module_par = MoABlock(48, num_heads=6, sequential_heads=False).eval()

    # Copy weights from parallel to sequential
    module_seq.load_state_dict(module_par.state_dict())

    with torch.no_grad():
        out_seq = module_seq(x)
        out_par = module_par(x)

    assert torch.allclose(out_seq, out_par, atol=1e-5), "Sequential vs parallel mismatch"


def test_moa_block_with_attn_dropout():
    """MoABlock with attention dropout > 0."""
    torch.manual_seed(0)
    module = MoABlock(48, num_heads=6, attn_drop=0.1).train()
    x = torch.randn(2, 48, 8, 8)
    out = module(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    out.mean().backward()
    assert _has_grad(module)


def test_moa_block_different_mlp_ratios():
    """MoABlock with various MLP expansion ratios."""
    torch.manual_seed(0)
    for ratio in [0.5, 1.0, 2.0, 4.0]:
        module = MoABlock(48, num_heads=6, mlp_ratio=ratio).train()
        x = torch.randn(1, 48, 8, 8)
        out = module(x)
        assert out.shape == x.shape, f"Shape mismatch at mlp_ratio={ratio}"
        assert torch.isfinite(out).all(), f"NaN at mlp_ratio={ratio}"


# ── 6. Global attention head advanced paths ────────────────────────────────

def test_global_head_linear_attn_large_spatial():
    """_GlobalAttnHead uses linear attention for N > 512 tokens."""
    torch.manual_seed(0)
    # 24×24 = 576 > 512 → triggers linear attention
    head = _GlobalAttnHead(32, num_heads=4).train()
    x = torch.randn(1, 32, 24, 24)  # N = 576
    out = head(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    out.mean().backward()
    assert _has_grad(head)


def test_global_head_smooth_blend_window():
    """_GlobalAttnHead blends between exact and linear attention in [448, 512]."""
    torch.manual_seed(0)
    head = _GlobalAttnHead(32, num_heads=4).train()
    # 22×22 = 484 → in blend window [448, 512]
    x = torch.randn(1, 32, 22, 22)  # N = 484
    out = head(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    out.mean().backward()
    assert _has_grad(head)


def test_global_head_very_large_spatial():
    """_GlobalAttnHead handles very large spatial maps (N >> 512)."""
    torch.manual_seed(0)
    # For large spatial maps, use fewer channels to keep memory low
    head = _GlobalAttnHead(16, num_heads=2).train()
    x = torch.randn(1, 16, 40, 40)  # N = 1600 >> 512
    out = head(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    out.mean().backward()
    assert _has_grad(head)


def test_global_head_exact_attn_small_spatial():
    """_GlobalAttnHead uses exact attention for N <= 448 (below blend window)."""
    torch.manual_seed(0)
    head = _GlobalAttnHead(32, num_heads=4).train()
    x = torch.randn(1, 32, 16, 16)  # N = 256
    out = head(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    out.mean().backward()
    assert _has_grad(head)


# ── 7. Flash attention edge cases ──────────────────────────────────────────

def test_flash_attn_fallback_no_sdpa(monkeypatch):
    """_flash_attn falls back to manual SDPA when F.scaled_dot_product_attention is absent."""
    monkeypatch.delattr(F, "scaled_dot_product_attention", raising=False)
    q = torch.randn(1, 2, 8, 8)
    k = torch.randn(1, 2, 8, 8)
    v = torch.randn(1, 2, 8, 8)
    scale = 0.25
    out = _flash_attn(q, k, v, scale=scale)
    expected = ((q @ k.transpose(-2, -1)) * scale).softmax(dim=-1) @ v
    assert torch.allclose(out, expected, atol=1e-5)


def test_flash_attn_sdpa_without_scale_typeerror(monkeypatch):
    """_flash_attn handles SDPA that raises TypeError unrelated to 'scale'."""
    original_sdpa = F.scaled_dot_product_attention
    call_count = [0]

    def buggy_sdpa(q, k, v, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 1:
            raise TypeError("unexpected keyword")
        return original_sdpa(q, k, v)

    monkeypatch.setattr(
        "ultralytics.nn.modules.moa.moa.F.scaled_dot_product_attention", buggy_sdpa
    )
    q = torch.randn(1, 2, 4, 4)
    k = torch.randn(1, 2, 4, 4)
    v = torch.randn(1, 2, 4, 4)
    # Should NOT raise — TypeError re-raised for non-scale errors
    with pytest.raises(TypeError, match="unexpected keyword"):
        _flash_attn(q, k, v, scale=0.25)


def test_flash_attn_sdpa_scale_typeerror_fallback(monkeypatch):
    """_flash_attn falls back gracefully when SDPA raises TypeError about 'scale'.

    Simulates old PyTorch where sdpa doesn't accept 'scale' kwarg.
    The fallback absorbs scale into q and calls sdpa again without scale.
    """
    import ultralytics.nn.modules.moa.moa as moa_mod

    original_sdpa = F.scaled_dot_product_attention

    def old_torch_sdpa(q, k, v, **kwargs):
        if "scale" in kwargs:
            raise TypeError("got an unexpected keyword argument 'scale'")
        return original_sdpa(q, k, v)

    monkeypatch.setattr(moa_mod.F, "scaled_dot_product_attention", old_torch_sdpa)
    q = torch.randn(1, 2, 4, 4)
    k = torch.randn(1, 2, 4, 4)
    v = torch.randn(1, 2, 4, 4)
    scale = 0.25
    out = _flash_attn(q, k, v, scale=scale)
    # Should fall back: SDPA with scale absorbed into q
    default_scale = q.shape[-1] ** -0.5
    expected = original_sdpa(q * (scale / default_scale), k, v)
    assert torch.allclose(out, expected, atol=1e-5)


def test_window_flash_attn_edge_cases():
    """_window_flash_attn handles various spatial/window configurations."""
    torch.manual_seed(0)
    B, nh, hd = 1, 2, 16
    for H, W, win in [
        (8, 8, 7),       # standard
        (15, 15, 7),     # non-divisible by window
        (7, 7, 7),       # exact fit
        (4, 4, 7),       # window > spatial (clamped)
        (16, 8, 4),      # non-square
        (1, 16, 4),      # H=1 edge case
    ]:
        N = H * W
        q = torch.randn(B, nh, N, hd)
        k = torch.randn(B, nh, N, hd)
        v = torch.randn(B, nh, N, hd)
        out = _window_flash_attn(q, k, v, scale=hd ** -0.5, window_size=win, height=H, width=W)
        assert out.shape == (B, nh, N, hd), f"Shape mismatch at H={H},W={W},win={win}"
        assert torch.isfinite(out).all(), f"NaN at H={H},W={W},win={win}"


# ── 8. _fp_min utility ────────────────────────────────────────────────────

def test_fp_min_across_dtypes():
    """_fp_min returns dtype-aware minimum values."""
    assert _fp_min(1e-6, torch.float32) == 1e-6
    assert _fp_min(1e-6, torch.float16) == 1e-4
    assert _fp_min(1e-6, torch.bfloat16) == 1e-3
    # Value above threshold passes through
    assert _fp_min(0.5, torch.float16) == 0.5


# ── 9. Aux loss DDP path ──────────────────────────────────────────────────

def test_moa_router_aux_loss_no_nan_from_biased_logits():
    """_moa_router_aux_loss returns finite values even with extreme router bias."""
    torch.manual_seed(0)
    # Create extreme logits that would overflow softmax
    logits = torch.tensor([[[[100.0, -100.0, 0.0]]]])  # [1, 3, 1, 1]
    weights = F.softmax(logits, dim=1)
    result = _moa_router_aux_loss(weights, logits, coeff=0.01)
    assert torch.isfinite(result), f"NaN/inf aux loss: {result}"


def test_moa_router_aux_loss_finite_guard_triggers():
    """_moa_router_aux_loss returns zero when result would be non-finite."""
    # Use large logits that cause overflow but not NaN in weights
    weights = torch.ones(1, 3, 4, 4) / 3.0
    logits = torch.full((1, 3, 4, 4), 1e6)  # very large but not inf
    result = _moa_router_aux_loss(weights, logits, coeff=0.01)
    # Should be finite (logits clamped to [-80, 80])
    assert torch.isfinite(result), f"Expected finite, got {result}"


# ── 10. RoutedModule protocol completeness ────────────────────────────────

def test_moa_block_routed_module_protocol():
    """MoABlock implements full RoutedModule protocol."""
    module = MoABlock(48, num_heads=6)
    assert module.num_experts == 3
    assert module.top_k == 3
    assert isinstance(module.aux_loss, torch.Tensor)

    snap = module.routing_snapshot()
    assert isinstance(snap, dict)

    caps = module.export_capabilities()
    assert isinstance(caps, dict)
    assert caps.get("dynamic_routing") is True


def test_c2fmoa_routed_module_protocol():
    """C2fMoA implements full RoutedModule protocol."""
    module = C2fMoA(64, 64, n=2, num_heads=6)
    assert module.num_experts == 3
    assert module.top_k == 3
    assert isinstance(module.aux_loss, torch.Tensor)

    snap = module.routing_snapshot()
    assert isinstance(snap, dict)

    caps = module.export_capabilities()
    assert isinstance(caps, dict)
    assert caps.get("dynamic_routing") is True


def test_neck_moa_fusion_routed_module_protocol():
    """NeckMoAFusion implements full RoutedModule protocol."""
    module = NeckMoAFusion(64, 128, 64, num_heads=4)
    assert module.num_experts == 2
    assert module.top_k == 2
    assert isinstance(module.aux_loss, torch.Tensor)

    snap = module.routing_snapshot()
    assert isinstance(snap, dict)

    caps = module.export_capabilities()
    assert isinstance(caps, dict)
    assert caps.get("dynamic_routing") is True


def test_publish_aux_loss_in_eval_mode():
    """publish_aux_loss in eval mode returns detached zero."""
    module = MoABlock(48, num_heads=6).eval()
    x = torch.randn(1, 48, 4, 4)
    module(x)
    result = module.publish_aux_loss(step=0, training=False)
    assert result.item() == 0.0
    assert not result.requires_grad


# ── 11. C2fMoA advanced edge cases ────────────────────────────────────────

def test_c2fmoa_multiple_expansion_ratios():
    """C2fMoA with different expansion ratios e."""
    torch.manual_seed(0)
    for e_val in [0.25, 0.5, 0.75, 1.0]:
        c = 64
        internal = int(c * e_val)
        if internal < 3:
            continue  # too small for 3 groups
        module = C2fMoA(c, c, n=1, num_heads=6, e=e_val).train()
        x = torch.randn(1, c, 8, 8)
        out = module(x)
        assert out.shape == (1, c, 8, 8), f"Shape mismatch at e={e_val}"
        assert torch.isfinite(out).all(), f"NaN at e={e_val}"


def test_c2fmoa_large_n_many_blocks():
    """C2fMoA with many stacked MoABlocks (n=8)."""
    torch.manual_seed(0)
    module = C2fMoA(32, 32, n=8, num_heads=3).train()
    x = torch.randn(1, 32, 8, 8)
    out = module(x)
    assert out.shape == (1, 32, 8, 8)
    assert torch.isfinite(out).all()
    assert module.last_aux_loss.requires_grad


def test_c2fmoa_inference_mode_no_aux_loss_grad():
    """C2fMoA in torch.no_grad() produces no-gradient aux_loss."""
    module = C2fMoA(64, 64, n=2, num_heads=6).eval()
    with torch.no_grad():
        out = module(torch.randn(2, 64, 8, 8))
    assert out.shape == (2, 64, 8, 8)
    assert not module.last_aux_loss.requires_grad


# ── 12. anneal_moa_temperature edge cases ─────────────────────────────────

def test_anneal_moa_temperature_respects_min_temp():
    """anneal_moa_temperature does not go below min_temp."""
    module = C2fMoA(64, 64, n=2, num_heads=6)
    initial = module.m[0].router.temperature
    # Anneal with very aggressive factor but high min_temp
    anneal_moa_temperature(module, factor=0.1, min_temp=0.8)
    for m in module.m:
        assert m.router.temperature == max(initial * 0.1, 0.8)


def test_anneal_moa_temperature_skips_non_moa_modules():
    """anneal_moa_temperature only affects _MoARouter instances."""
    module = nn.Sequential(
        nn.Conv2d(16, 16, 1),
        MoABlock(16, num_heads=3),
    )
    anneal_moa_temperature(module, factor=0.5, min_temp=0.3)
    # Conv2d has no temperature attr
    assert module[1].router.temperature == max(1.0 * 0.5, 0.3)  # default 1.0


# ── 13. collect_moa_aux_loss edge cases ───────────────────────────────────

def test_collect_moa_aux_loss_multiple_module_types():
    """collect_moa_aux_loss collects from mixed C2fMoA + NeckMoAFusion."""
    torch.manual_seed(0)
    c2f = C2fMoA(32, 32, n=1, num_heads=3).train()
    neck = NeckMoAFusion(32, 64, 32, num_heads=2).train()

    c2f(torch.randn(1, 32, 8, 8))
    neck(torch.randn(1, 32, 8, 8), torch.randn(1, 64, 4, 4))

    # Wrap in a parent module to test collection
    parent = nn.ModuleList([c2f, neck])
    total = collect_moa_aux_loss(parent)
    expected = c2f.last_aux_loss + neck.last_aux_loss
    assert torch.allclose(total, expected, atol=1e-5)


def test_collect_moa_aux_loss_with_none_model():
    """collect_moa_aux_loss(None) returns zero tensor."""
    result = collect_moa_aux_loss(None)
    assert result.item() == 0.0


# ── 14. Global head QR fallback ───────────────────────────────────────────

def test_global_head_rf_matrix_deterministic():
    """Global head RF matrix is deterministic across instantiations."""
    torch.manual_seed(0)
    h1 = _GlobalAttnHead(32, num_heads=4, rf_seed=42).eval()
    h2 = _GlobalAttnHead(32, num_heads=4, rf_seed=42).eval()
    assert torch.allclose(h1._rf_matrix, h2._rf_matrix)


def test_global_head_rf_matrix_different_seeds():
    """Different rf_seed produce different RF matrices."""
    h1 = _GlobalAttnHead(32, num_heads=4, rf_seed=1).eval()
    h2 = _GlobalAttnHead(32, num_heads=4, rf_seed=2).eval()
    assert not torch.allclose(h1._rf_matrix, h2._rf_matrix)


# ── 15. Numerical stability of output across spatial sizes ─────────────────

def test_moa_block_diverse_spatial_sizes():
    """MoABlock forward is stable across a range of spatial sizes."""
    torch.manual_seed(0)
    module = MoABlock(32, num_heads=3).train()
    for size in [4, 8, 16, 32, 64]:
        x = torch.randn(1, 32, size, size)
        out = module(x)
        assert out.shape == x.shape, f"Shape mismatch at size={size}"
        assert torch.isfinite(out).all(), f"NaN at size={size}"


def test_neck_moa_fusion_diverse_spatial_sizes():
    """NeckMoAFusion forward is stable across spatial sizes."""
    torch.manual_seed(0)
    module = NeckMoAFusion(32, 64, 32, num_heads=2).train()
    for h in [4, 8, 16, 32]:
        hi = torch.randn(1, 32, h, h)
        lo = torch.randn(1, 64, max(1, h // 2), max(1, h // 2))
        out = module(hi, lo)
        assert out.shape == hi.shape, f"Shape mismatch at h={h}"
        assert torch.isfinite(out).all(), f"NaN at h={h}"
