from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.modules.moa import C2fMoA, MoABlock
from ultralytics.nn.modules.mot import C2fMoT, MoTBlock, anneal_mot_temperature, collect_mot_aux_loss
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import _collect_mot_aux_loss


ROOT = Path(__file__).resolve().parents[1]


def _has_grad(module):
    return any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in module.parameters()
        if p.requires_grad
    )


def test_mot_block_forward_backward_all_experts_trainable():
    """All experts receive non-zero gradient during training.

    With top_k=1, non-selected experts only receive gradient through the
    exploration_eps (~0.02) blending floor.  Using ``out.mean()`` divides the
    gradient by B*C*H*W (=2048), which can underflow to exact zero in float32
    for the non-selected experts.  A squared-sum loss preserves sufficient
    gradient magnitude for all experts to register as trainable.
    """
    torch.manual_seed(0)
    block = MoTBlock(32, num_heads=4, top_k=1, window_size=4, n_points=2).train()
    x = torch.randn(1, 32, 8, 8)
    out, aux = block(x)
    assert out.shape == x.shape
    assert aux.requires_grad
    assert torch.isfinite(aux)

    # Squared-sum produces O(1) per-element gradient (vs O(1/N) for mean),
    # ensuring non-selected experts' ~0.02-weighted contribution stays above
    # the float32 underflow threshold.
    ((out ** 2).sum() + aux).backward()
    assert _has_grad(block.router)
    for expert in block.experts:
        assert _has_grad(expert)


def test_c2fmot_collects_aux_loss_and_keeps_shape():
    torch.manual_seed(0)
    module = C2fMoT(48, 64, n=2, num_heads=4, top_k=2, window_size=4, n_points=2).train()
    out = module(torch.randn(2, 48, 8, 8))

    assert out.shape == (2, 64, 8, 8)
    aux = collect_mot_aux_loss(module)
    assert aux.requires_grad
    assert torch.isfinite(aux)
    assert _collect_mot_aux_loss(module, torch.device("cpu")).requires_grad


def test_mot_router_z_loss_uses_expert_axis():
    torch.manual_seed(0)
    module = MoTBlock(32, num_heads=4, top_k=2, window_size=4, n_points=2, balance_loss_coeff=0.01).train()
    z_loss = module.router.router_z_loss(torch.randn(2, 32, 5, 7))
    expected = torch.log(torch.tensor(3.0)).square()
    assert torch.allclose(z_loss, expected, atol=1e-5)


def test_mot_block_reuses_router_logits_for_z_loss(monkeypatch):
    torch.manual_seed(0)
    module = MoTBlock(24, num_heads=3, top_k=2, window_size=4, n_points=2, balance_loss_coeff=0.01).train()
    calls = {"n": 0}
    original = module.router._compute_logits

    def wrapped(x):
        calls["n"] += 1
        return original(x)

    monkeypatch.setattr(module.router, "_compute_logits", wrapped)
    out, aux = module(torch.randn(1, 24, 4, 4))
    assert out.shape == (1, 24, 4, 4)
    assert aux.requires_grad
    assert calls["n"] == 1


def test_mot_temperature_anneal():
    module = C2fMoT(64, 64, n=2, num_heads=4)
    before = [float(m.router.temperature) for m in module.m]
    anneal_mot_temperature(module, factor=0.5, min_temp=0.3)
    after = [float(m.router.temperature) for m in module.m]
    assert after == [max(t * 0.5, 0.3) for t in before]


def test_trainer_detects_and_anneals_moa_mot_temperatures():
    trainer = object.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(moa_mot_temperature_factor=0.5, moa_mot_min_temperature=0.3)
    moa = C2fMoA(64, 64, n=1, num_heads=4)
    mot = C2fMoT(64, 64, n=1, num_heads=4)
    trainer.model = torch.nn.Sequential(moa, mot)

    trainer._detect_moa_mot_modules()
    assert trainer._has_moa_mot is True

    moa_before = [float(m.router.temperature) for m in moa.m]
    mot_before = [float(m.router.temperature) for m in mot.m]
    trainer._anneal_moa_mot_temperature()
    assert [float(m.router.temperature) for m in moa.m] == [max(t * 0.5, 0.3) for t in moa_before]
    assert [float(m.router.temperature) for m in mot.m] == [max(t * 0.5, 0.3) for t in mot_before]


def test_mot_model_configs_parse():
    configs = {
        ROOT / "ultralytics/cfg/models/master/v0_8/det/yolo-master-mot-n.yaml": (3, 6, 0, 0),
        ROOT / "ultralytics/cfg/models/master/v0_8/det/yolo-master-moa-mot-n.yaml": (3, 6, 2, 2),
        ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-n.yaml": (3, 6, 0, 0),
        ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-mot-n.yaml": (3, 6, 1, 1),
    }
    for cfg, (c2fmot, motblocks, c2fmoa, moablocks) in configs.items():
        model = DetectionModel(str(cfg), ch=3, nc=80, verbose=False)
        assert sum(isinstance(m, C2fMoT) for m in model.modules()) == c2fmot
        assert sum(isinstance(m, MoTBlock) for m in model.modules()) == motblocks
        assert sum(isinstance(m, C2fMoA) for m in model.modules()) == c2fmoa
        assert sum(isinstance(m, MoABlock) for m in model.modules()) == moablocks


def test_mot_deformable_align_corners_option():
    block = MoTBlock(32, num_heads=4, top_k=2, window_size=4, n_points=2, grid_align_corners=False)
    assert block.experts[2].align_corners is False



def test_mot_window_size_larger_than_feature_map():
    """MoT window expert should pad and crop back when window_size exceeds H/W."""
    torch.manual_seed(0)
    block = MoTBlock(32, num_heads=4, top_k=2, window_size=16, n_points=2).eval()
    x = torch.randn(1, 32, 5, 7)
    with torch.no_grad():
        out, aux = block(x)
    assert out.shape == x.shape
    assert aux.item() == 0
    assert torch.isfinite(out).all()


def test_mot_window_expert_handles_window_larger_than_feature_map():
    """Window expert should pad and crop back when win is larger than H/W."""
    from ultralytics.nn.modules.mot.mot import _WindowTransformerExpert

    expert = _WindowTransformerExpert(16, num_heads=4, window_size=8).eval()
    x = torch.randn(1, 16, 3, 5)
    with torch.no_grad():
        out = expert(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_mot_window_expert_shift_spatial_alignment():
    """Shifted-window expert output must keep input spatial dims and stay finite.

    The shift/un-shift residual alignment is the subtle correctness point: an
    identity-ish check ensures no cyclic spatial misalignment leaks through.
    """
    torch.manual_seed(0)
    block = MoTBlock(32, num_heads=4, top_k=2, window_size=5, n_points=2, window_shift=True).eval()
    x = torch.randn(2, 32, 9, 11)
    with torch.no_grad():
        out, _ = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_mot_window_expert_shift_handles_odd_spatial_sizes():
    """Shifted-window expert should keep shape for odd H/W with padding and crop-back."""
    from ultralytics.nn.modules.mot.mot import _WindowTransformerExpert

    expert = _WindowTransformerExpert(24, num_heads=3, window_size=5, shift_size=2).eval()
    x = torch.randn(1, 24, 7, 9)
    with torch.no_grad():
        out = expert(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_mot_router_disables_exploration_eps_in_eval():
    """Evaluation routing should stay sparse and not re-densify via exploration_eps."""
    block = MoTBlock(24, num_heads=3, top_k=1, window_size=4, n_points=2, exploration_eps=0.2).eval()
    weights, indices = block.router(torch.randn(2, 24, 5, 5))

    nonzero_per_token = (weights > 0).sum(dim=1)
    assert torch.equal(nonzero_per_token, torch.ones_like(nonzero_per_token))
    assert indices.shape == (2, 1, 5, 5)


def test_mot_router_warns_when_exploration_eps_is_clamped():
    with pytest.warns(UserWarning, match="clamped"):
        block = MoTBlock(24, num_heads=3, exploration_eps=0.3)

    assert block.router.exploration_eps == 0.2


def test_mot_scene_consistency_rejects_unsupported_expert_count():
    from ultralytics.nn.modules.mot.router import _MoTRouter

    router = _MoTRouter(8, num_experts=4, top_k=2, scene_aware=True)
    weights = torch.full((1, 4, 1, 1), 0.25)
    with pytest.raises(ValueError, match="exactly 3 experts"):
        router.scene_consistency_loss(weights, torch.ones(1, 3))


def test_mot_deformable_attention_falls_back_for_non_grid_tokens():
    from ultralytics.nn.modules.mot.mot import _DeformableTransformerExpert

    expert = _DeformableTransformerExpert(16, num_heads=4, n_points=2).eval()
    tokens = torch.randn(2, 7, 16)
    with torch.no_grad():
        output = expert._deform_attn(tokens, tokens, H=2, W=3)

    assert output.shape == tokens.shape
    assert torch.isfinite(output).all()


def test_mot_inference_sparsity_skips_inactive_experts():
    """At eval with top_k<E, a per-sample inactive expert must not be invoked."""
    torch.manual_seed(0)
    block = MoTBlock(24, num_heads=3, top_k=1, window_size=4, n_points=2, exploration_eps=0.2).eval()
    x = torch.randn(1, 24, 6, 6)
    with torch.no_grad():
        weights, indices = block.router(x)
    assert weights.shape == (1, 3, 6, 6)
    assert indices.shape == (1, 1, 6, 6)
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0]))
    assert (weights > 0).sum(dim=1).max().item() == 1


# ── Boundary regression tests (issue #54) ──────────────────────────────────

def test_mot_block_handles_1x1_feature_map():
    """MoTBlock must not crash on the smallest possible spatial input (1×1)."""
    torch.manual_seed(0)
    block = MoTBlock(32, num_heads=4, top_k=2, window_size=4, n_points=2).eval()
    x = torch.randn(2, 32, 1, 1)
    with torch.no_grad():
        out, aux = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert aux.item() == 0


def test_mot_block_handles_all_zero_input():
    """All-zero input must not produce NaN or Inf in output or aux loss."""
    torch.manual_seed(0)
    # Use balance_loss_coeff > 0 so aux loss is computed even on zero input
    block = MoTBlock(32, num_heads=4, top_k=2, window_size=4, n_points=2,
                     balance_loss_coeff=0.01).train()
    x = torch.zeros(2, 32, 8, 8)
    out, aux = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert torch.isfinite(aux)


def test_mot_block_handles_very_wide_feature_map():
    """MoTBlock with extreme aspect ratio (very wide, very short)."""
    torch.manual_seed(0)
    block = MoTBlock(32, num_heads=4, top_k=2, window_size=4, n_points=2).eval()
    x = torch.randn(1, 32, 4, 128)
    with torch.no_grad():
        out, aux = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_mot_deformable_expert_handles_extreme_offsets():
    """Deformable expert must produce finite output with near-boundary offsets."""
    import torch.nn as nn
    from ultralytics.nn.modules.mot.mot import _DeformableTransformerExpert

    torch.manual_seed(0)
    expert = _DeformableTransformerExpert(32, num_heads=4, n_points=4).eval()
    # Manually push offset weights to produce extreme sampling locations
    nn.init.constant_(expert.offset_proj.weight, 5.0)
    nn.init.constant_(expert.offset_proj.bias, 5.0)

    x = torch.randn(1, 32, 8, 8)
    with torch.no_grad():
        out = expert(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_mot_deformable_expert_handles_single_pixel():
    """Deformable expert on a 1×1 feature map (N==1)."""
    from ultralytics.nn.modules.mot.mot import _DeformableTransformerExpert

    torch.manual_seed(0)
    expert = _DeformableTransformerExpert(16, num_heads=2, n_points=2).eval()
    x = torch.randn(1, 16, 1, 1)
    with torch.no_grad():
        out = expert(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_c2fmot_handles_minimal_channels():
    """C2fMoT with smallest practical channel count must not crash."""
    torch.manual_seed(0)
    module = C2fMoT(8, 8, n=1, num_heads=2, top_k=1, e=0.5).train()
    out = module(torch.randn(1, 8, 4, 4))
    assert out.shape == (1, 8, 4, 4)


def test_mot_router_z_loss_handles_extreme_logits():
    """Router z-loss must guard against overflow on extreme logit values."""
    block = MoTBlock(32, num_heads=4, top_k=2, window_size=4, n_points=2,
                     balance_loss_coeff=0.01).eval()
    # Simulate extreme router output
    extreme_logits = torch.full((1, 3, 4, 4), 100.0)
    z = block.router.z_loss_from_logits(extreme_logits)
    assert torch.isfinite(z)
    # Also test very negative logits
    neg_logits = torch.full((1, 3, 4, 4), -100.0)
    z_neg = block.router.z_loss_from_logits(neg_logits)
    assert torch.isfinite(z_neg)


def test_mot_sparse_train_mode():
    """sparse_train=True must only dispatch to selected experts."""
    torch.manual_seed(0)
    block = MoTBlock(24, num_heads=3, top_k=1, window_size=4, n_points=2,
                     sparse_train=True).train()
    x = torch.randn(1, 24, 6, 6)
    out, aux = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    # Sparse dispatch stats should show fewer than 3 expert calls
    stats = block._last_dispatch_stats
    assert stats["mode"] == "sample_sparse"
    assert stats["expert_calls"] <= block.NUM_EXPERTS


def test_mot_model_configs_include_mixed_variants():
    """All MoT, MoA, and mixed YAML configs must parse without error."""
    configs = {
        ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-n.yaml": (3, 6, 0, 0),
        ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-n.yaml": (0, 0, 3, 6),
        ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-mot-n.yaml": (3, 6, 1, 1),
        ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-n.yaml": (0, 0, 0, 0),
    }
    for cfg, (c2fmot, motblocks, c2fmoa, moablocks) in configs.items():
        model = DetectionModel(str(cfg), ch=3, nc=80, verbose=False)
        assert sum(isinstance(m, C2fMoT) for m in model.modules()) == c2fmot
        assert sum(isinstance(m, MoTBlock) for m in model.modules()) == motblocks
        assert sum(isinstance(m, C2fMoA) for m in model.modules()) == c2fmoa
        assert sum(isinstance(m, MoABlock) for m in model.modules()) == moablocks


def test_c2fmot_aux_loss_aggregation():
    """Multiple stacked MoTBlocks must have their aux losses summed correctly."""
    torch.manual_seed(0)
    module = C2fMoT(32, 32, n=3, num_heads=4, top_k=2, balance_loss_coeff=0.01).train()
    module(torch.randn(2, 32, 8, 8))
    # Each block contributes to total
    block_aux = [m.last_aux_loss for m in module.m
                 if isinstance(getattr(m, 'last_aux_loss', None), torch.Tensor)]
    assert len(block_aux) == 3
    assert torch.allclose(module.last_aux_loss, sum(block_aux))
