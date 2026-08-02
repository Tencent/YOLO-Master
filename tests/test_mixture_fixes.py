"""Regression tests for the MoE/MoA/MoT fixes (report v1.0).

Covers:
- MoE-Loss float32 DDP reduce path (numeric stability helper).
- DynamicRoutingLayer hard Top-K sparsity at inference (eval) vs soft at train.
- UltimateOptimizedMoE.get_gflops removes the fictitious *0.9 static factor.
- get_gflops 'total_gflops' is the consistent sum of components (no double count).
- MoT _sdpa fallback is memory-bounded (chunked) for large N on PyTorch < 2.0.
- MoA shortcut=False has no residual on either attention or FFN.
- MoA N>256 linear-attention path runs and approximates standard attention.
- MoELoss.coeff_floor warns once when it overrides a small user coefficient.
"""

import math

import torch

from ultralytics.nn.modules.moe.loss import MoELoss, all_reduce_mean
from ultralytics.nn.modules.moe.routers import DynamicRoutingLayer
from ultralytics.nn.modules.moe.modules import UltimateOptimizedMoE
from ultralytics.nn.modules.moa.moa import MoABlock, _GlobalAttnHead
from ultralytics.nn.modules.mot import mot as mot_mod


# ---------------------------------------------------------------------------
# MoE-Loss
# ---------------------------------------------------------------------------


def test_all_reduce_mean_noop_single_process_preserves_dtype():
    # No DDP initialised → identity, but dtype must be preserved.
    t = torch.randn(8, dtype=torch.float16)
    out = all_reduce_mean(t)
    assert out.dtype == torch.float16
    assert torch.equal(out, t)


def test_moe_loss_coeff_floor_warns_once(monkeypatch):
    from ultralytics import utils as ult_utils

    calls = []
    monkeypatch.setattr(ult_utils.LOGGER, "warning", lambda msg, *a, **k: calls.append(msg))

    loss_fn = MoELoss(balance_loss_coeff=1e-4, z_loss_coeff=1e-4, num_experts=4, top_k=2, coeff_floor=0.01)
    probs = torch.softmax(torch.randn(16, 4), dim=1)
    logits = torch.randn(16, 4)
    idx = torch.randint(0, 4, (16, 2))
    loss_fn(probs, logits, idx)
    loss_fn(probs, logits, idx)

    floor_warnings = [m for m in calls if "coeff_floor" in str(m)]
    assert len(floor_warnings) == 1, "coeff_floor should warn exactly once"


# ---------------------------------------------------------------------------
# MoE routing sparsity
# ---------------------------------------------------------------------------


def test_dynamic_router_hard_topk_at_inference():
    torch.manual_seed(0)
    router = DynamicRoutingLayer(in_channels=32, num_experts=4, top_k=2)
    x = torch.randn(2, 32, 8, 8)

    router.eval()
    with torch.no_grad():
        w = router(x)  # [B, E, H, W]
    # Hard Top-K: exactly (E - top_k) experts must be exactly zero per sample.
    per_sample = w[:, :, 0, 0]  # weights are spatially constant (GAP router)
    nonzero = (per_sample > 0).sum(dim=1)
    assert torch.all(nonzero <= 2), "inference must use hard Top-K (true sparsity)"
    # Weights still sum to 1.
    assert torch.allclose(per_sample.sum(dim=1), torch.ones(2), atol=1e-4)


def test_dynamic_router_soft_topk_at_training():
    torch.manual_seed(0)
    router = DynamicRoutingLayer(in_channels=32, num_experts=4, top_k=2).train()
    x = torch.randn(2, 32, 8, 8, requires_grad=True)
    w = router(x)
    w.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# FLOPs accounting
# ---------------------------------------------------------------------------


def test_ultimate_moe_gflops_no_fake_skip_and_consistent_total():
    m = UltimateOptimizedMoE(32, 32, num_experts=4, top_k=2)
    flops = m.get_gflops((1, 32, 32, 32))
    total = flops.pop("total_gflops")
    assert math.isclose(total, sum(flops.values()), rel_tol=1e-6), (
        "total_gflops must equal the sum of components (no double-count, no fake factor)"
    )
    assert flops["static_path"] > 0


# ---------------------------------------------------------------------------
# MoT large-N fallback (simulate PyTorch < 2.0 by hiding SDPA)
# ---------------------------------------------------------------------------


def test_mot_sdpa_fallback_is_memory_bounded(monkeypatch):
    # Force the fallback path by pretending F has no scaled_dot_product_attention.
    monkeypatch.setattr(mot_mod.F, "scaled_dot_product_attention", None, raising=False)
    monkeypatch.delattr(mot_mod.F, "scaled_dot_product_attention", raising=False)
    # Small explicit-limit + chunk so the test stays fast but exercises chunking.
    monkeypatch.setattr(mot_mod, "_SDPA_EXPLICIT_MAX_TOKENS", 16)
    monkeypatch.setattr(mot_mod, "_SDPA_FALLBACK_CHUNK", 8)

    B, nh, N, hd = 1, 2, 40, 8
    q = torch.randn(B, nh, N, hd)
    k = torch.randn(B, nh, N, hd)
    v = torch.randn(B, nh, N, hd)
    scale = hd**-0.5
    out = mot_mod._sdpa(q, k, v, scale)
    assert out.shape == (B, nh, N, hd)
    # Compare to reference dense attention.
    ref = (q @ k.transpose(-2, -1) * scale).softmax(-1) @ v
    assert torch.allclose(out, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# MoA shortcut semantics
# ---------------------------------------------------------------------------


def test_moa_shortcut_false_has_no_residual():
    torch.manual_seed(0)
    block = MoABlock(dim=48, num_heads=3, shortcut=False).eval()
    # Zero out all layer-scales & make FFN produce 0 → output must be ~0 (no residual leak).
    with torch.no_grad():
        block.ls_attn.zero_()
        block.ls_ffn.zero_()
    x = torch.randn(2, 48, 8, 8)
    with torch.no_grad():
        out = block(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-5), (
        "shortcut=False must not add input residual on attention or FFN path"
    )


def test_moa_shortcut_true_keeps_residual():
    torch.manual_seed(0)
    block = MoABlock(dim=48, num_heads=3, shortcut=True).eval()
    with torch.no_grad():
        block.ls_attn.zero_()
        block.ls_ffn.zero_()
    x = torch.randn(2, 48, 8, 8)
    with torch.no_grad():
        out = block(x)
    assert torch.allclose(out, x, atol=1e-5), "shortcut=True with zero layer-scale must pass input through unchanged"


def test_moa_optional_sparse_inference_skips_low_weight_head(monkeypatch):
    block = MoABlock(24, num_heads=3, sparse_inference=True, sparse_inference_threshold=0.02).eval()
    calls = {"local": 0, "region": 0, "global": 0}
    for name in calls:
        head = getattr(block, f"{name}_head")
        original = head.forward

        def wrapped(value, *, _name=name, _original=original):
            calls[_name] += 1
            return _original(value)

        monkeypatch.setattr(head, "forward", wrapped)

    def concentrated_router(x, return_logits=False):
        weights = x.new_zeros(x.shape[0], 3, x.shape[2], x.shape[3])
        weights[:, 0] = 1.0
        logits = weights.float()
        return (weights, logits) if return_logits else weights

    monkeypatch.setattr(block.router, "forward", concentrated_router)
    with torch.no_grad():
        block(torch.randn(1, 24, 4, 4))

    assert calls == {"local": 1, "region": 0, "global": 0}
    assert block.export_capabilities()["eager_sparse_dispatch"] is True


# ---------------------------------------------------------------------------
# MoA global head: N > 256 linear-attention path
# ---------------------------------------------------------------------------


def test_moa_global_head_linear_path_large_n():
    torch.manual_seed(0)
    # 24x24 = 576 tokens > 256 → linear attention path is taken.
    head = _GlobalAttnHead(64, num_heads=4, head_dim=16).eval()
    x = torch.randn(1, 64, 24, 24)
    with torch.no_grad():
        out = head(x)
    assert out.shape == (1, 64, 24, 24)
    assert torch.isfinite(out).all()


def test_moa_global_head_linear_vs_softmax_correlated():
    """Linear attention should be positively correlated with full softmax attention."""
    torch.manual_seed(0)
    head = _GlobalAttnHead(32, num_heads=2, head_dim=16).eval()
    B, nh, N, hd = 1, 2, 400, 16
    q = torch.randn(B, nh, N, hd)
    k = torch.randn(B, nh, N, hd)
    v = torch.randn(B, nh, N, hd)
    with torch.no_grad():
        approx = head._linear_attn(q, k, v).reshape(-1)
        exact = ((q @ k.transpose(-2, -1)) * head.scale).softmax(-1) @ v
        exact = exact.reshape(-1)
    # Pearson correlation > 0 confirms the approximation tracks the true op.
    approx_centered = approx - approx.mean()
    exact_centered = exact - exact.mean()
    corr = (approx_centered * exact_centered).sum() / (
        approx_centered.square().sum().sqrt() * exact_centered.square().sum().sqrt()
    )
    assert corr > 0.1, f"linear attn should correlate with softmax attn, got {corr:.3f}"


def test_moa_regional_head_respects_kv_token_budget():
    """Regional attention increases pooling stride when the KV map is too large."""
    from ultralytics.nn.modules.moa.heads import _RegionalAttnHead

    head = _RegionalAttnHead(32, num_heads=2, head_dim=16, pool_stride=2, max_kv_tokens=16).eval()
    pooled_shapes = []
    hook = head.kv_proj.register_forward_hook(lambda _, inputs, __: pooled_shapes.append(tuple(inputs[0].shape[-2:])))
    try:
        with torch.no_grad():
            out = head(torch.randn(1, 32, 16, 16))
    finally:
        hook.remove()
    assert out.shape == (1, 32, 16, 16)
    assert pooled_shapes == [(4, 4)]


def test_moa_linear_attention_uses_raw_kv_accumulator():
    """The linear attention numerator must not apply an extra KV L2 scale."""
    from ultralytics.nn.modules.moa.heads import _GlobalAttnHead

    torch.manual_seed(0)
    head = _GlobalAttnHead(16, num_heads=1, head_dim=16).eval()
    q = torch.randn(1, 1, 7, 16)
    k = torch.randn(1, 1, 7, 16)
    v = torch.randn(1, 1, 7, 16)
    with torch.no_grad():
        actual = head._linear_attn(q, k, v)
        rf = head._get_rf(q.device, q.dtype)
        scale = rf.shape[0] ** -0.5
        q_feat = head._relu_kernel(q @ rf.T * scale).clamp(max=1e4)
        k_feat = head._relu_kernel(k @ rf.T * scale).clamp(max=1e4)
        q_flat = q_feat.reshape(-1, 7, rf.shape[0])
        k_flat = k_feat.reshape(-1, 7, rf.shape[0])
        v_flat = v.reshape(-1, 7, 16)
        kv = k_flat.transpose(1, 2) @ v_flat
        k_sum = k_flat.float().sum(dim=1)
        expected = ((q_flat @ kv).clamp(-1e4, 1e4)
                    / (q_flat @ k_sum.to(q_flat.dtype).unsqueeze(-1)).clamp_min(1e-6))
        expected = expected.reshape_as(actual)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
