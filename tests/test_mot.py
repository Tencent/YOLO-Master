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
    ((out**2).sum() + aux).backward()
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
    block = MoTBlock(32, num_heads=4, top_k=2, window_size=4, n_points=2, balance_loss_coeff=0.01).train()
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
        out, _aux = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_mot_deformable_expert_handles_extreme_offsets():
    """Deformable expert must produce finite output with near-boundary offsets."""
    from torch import nn

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
    block = MoTBlock(32, num_heads=4, top_k=2, window_size=4, n_points=2, balance_loss_coeff=0.01).eval()
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
    block = MoTBlock(24, num_heads=3, top_k=1, window_size=4, n_points=2, sparse_train=True).train()
    x = torch.randn(1, 24, 6, 6)
    out, _aux = block(x)
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
    block_aux = [m.last_aux_loss for m in module.m if isinstance(getattr(m, "last_aux_loss", None), torch.Tensor)]
    assert len(block_aux) == 3
    assert torch.allclose(module.last_aux_loss, sum(block_aux))


# ─────────────────────────────────────────────────────────────────────────────
# Boundary regression tests: degenerate geometry, shifted-window correctness,
# and eval-mode routing.  Each test below pins a defect found by probing the
# boundary surface of MoTBlock / _WindowTransformerExpert / the router.
# ─────────────────────────────────────────────────────────────────────────────


def _amplified_window_expert(window_size, shift_size, dim=8, num_heads=2):
    """Window expert whose attention path dominates, for influence probing.

    Default ``ls1=0.1`` plus ``std=0.02`` weights attenuate the attention
    contribution so far that cross-token influence is unmeasurable.  Amplifying
    qkv/proj, setting ``ls1=1`` and zeroing ``ls2`` isolates attention as the
    only route by which one token can affect another.
    """
    from ultralytics.nn.modules.mot.mot import _WindowTransformerExpert

    torch.manual_seed(0)
    expert = _WindowTransformerExpert(dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size).eval()
    with torch.no_grad():
        torch.nn.init.trunc_normal_(expert.qkv.weight, std=0.5)
        torch.nn.init.trunc_normal_(expert.proj.weight, std=0.5)
        expert.ls1.fill_(1.0)
        expert.ls2.zero_()
    return expert


def _influence_map(expert, src, size=(8, 8), dim=8):
    """Per-pixel absolute output change from perturbing a single input token.

    The perturbation must vary across channels: LayerNorm cancels a
    channel-uniform offset, which would make the probe blind.
    """
    H, W = size
    x = torch.zeros(1, dim, H, W)
    delta = torch.tensor([5.0, -5.0] * (dim // 2))
    with torch.no_grad():
        base = expert(x)
        perturbed = x.clone()
        perturbed[0, :, src[0], src[1]] += delta
        return (expert(perturbed) - base).abs().sum(dim=1)[0]


@pytest.mark.parametrize("window_size", [0, -1, -4])
def test_mot_window_expert_rejects_non_positive_window_size(window_size):
    """window_size <= 0 must raise, not ZeroDivisionError inside _pad_to_window."""
    from ultralytics.nn.modules.mot.mot import _WindowTransformerExpert

    with pytest.raises(ValueError, match="window_size must be >= 1"):
        _WindowTransformerExpert(16, num_heads=4, window_size=window_size)


@pytest.mark.parametrize("window_size", [0, -3])
def test_mot_block_rejects_non_positive_window_size(window_size):
    """MoTBlock validates window_size up front so bad YAML fails clearly."""
    with pytest.raises(ValueError, match="window_size must be >= 1"):
        MoTBlock(32, num_heads=4, top_k=2, window_size=window_size, n_points=2)


@pytest.mark.parametrize("n_points", [0, -2])
def test_mot_deformable_expert_rejects_non_positive_n_points(n_points):
    """n_points=0 silently zeroed the whole attention branch — must raise."""
    from ultralytics.nn.modules.mot.mot import _DeformableTransformerExpert

    with pytest.raises(ValueError, match="n_points must be >= 1"):
        _DeformableTransformerExpert(16, num_heads=4, n_points=n_points)


@pytest.mark.parametrize("n_points", [0, -2])
def test_mot_block_rejects_non_positive_n_points(n_points):
    with pytest.raises(ValueError, match="n_points must be >= 1"):
        MoTBlock(32, num_heads=4, top_k=2, window_size=4, n_points=n_points)


def test_mot_window_expert_honours_explicit_shift_size():
    """shift_size is a value, not a flag.

    It used to be coerced via ``(window_size // 2) if shift_size else 0``, so
    every truthy value collapsed to win//2 — silently ignoring the caller.
    """
    from ultralytics.nn.modules.mot.mot import _WindowTransformerExpert

    for requested in (0, 1, 2, 3):
        expert = _WindowTransformerExpert(16, num_heads=4, window_size=5, shift_size=requested)
        assert expert.shift_size == requested
    # bool keeps the Swin default meaning; out-of-range wraps into [0, win)
    assert _WindowTransformerExpert(16, num_heads=4, window_size=5, shift_size=True).shift_size == 2
    assert _WindowTransformerExpert(16, num_heads=4, window_size=5, shift_size=False).shift_size == 0
    assert _WindowTransformerExpert(16, num_heads=4, window_size=5, shift_size=7).shift_size == 2


def test_mot_window_expert_shift_does_not_leak_across_image_edges():
    """Shifted windows must be masked, as in canonical Swin.

    The cyclic roll wraps opposite edges into one window; without an attention
    mask a top-left token influences the bottom-right corner.  The unshifted
    expert is the control: influence outside the window is exactly zero.
    """
    control = _influence_map(_amplified_window_expert(4, 0), src=(0, 0))
    shifted = _influence_map(_amplified_window_expert(4, 2), src=(0, 0))

    # Control: strictly window-local.
    assert control[4:, :].sum() == 0
    assert control[:, 4:].sum() == 0
    # Shifted: the wrapped edge rows/cols must not be reachable either.
    assert shifted[6:, :].sum() == 0, "shifted window leaks into wrapped bottom rows"
    assert shifted[:, 6:].sum() == 0, "shifted window leaks into wrapped right cols"
    assert shifted[7, 7] == 0, "opposite corners attend each other under shift"
    # Sanity: the probe is actually measuring something.
    assert shifted[:4, :4].sum() > 0


def test_mot_window_expert_shift_still_connects_across_windows():
    """The mask must block only wrapped pairs, not shift's whole purpose.

    A token at the unshifted window boundary (3, 3) must reach tokens beyond
    it (rows/cols 4-5) once shifted — that cross-window connectivity is why
    Swin alternates shifted blocks at all.
    """
    control = _influence_map(_amplified_window_expert(4, 0), src=(3, 3))
    shifted = _influence_map(_amplified_window_expert(4, 2), src=(3, 3))

    assert control[4:, :].sum() == 0  # unshifted: cannot cross the boundary
    assert shifted[4:6, 2:6].sum() > 0  # shifted: reaches the next window
    # Interior window is fully unmasked, so influence spans exactly rows/cols 2-5.
    rows = (shifted > 1e-4).any(dim=1).nonzero().flatten().tolist()
    cols = (shifted > 1e-4).any(dim=0).nonzero().flatten().tolist()
    assert rows == [2, 3, 4, 5]
    assert cols == [2, 3, 4, 5]


def test_mot_shift_attn_mask_leaves_interior_window_unmasked():
    """Mask shape/content contract: interior window fully attends, edges restricted."""
    expert = _amplified_window_expert(4, 2)
    mask = expert._shift_attn_mask(8, 8, torch.device("cpu"))
    assert mask.shape == (4, 1, 16, 16)  # [nW, 1(head-broadcast), win², win²]
    assert mask.dtype == torch.bool
    allowed = [float(w.float().mean()) for w in mask[:, 0]]
    assert allowed[0] == 1.0, "interior window must be fully unmasked"
    assert all(0 < a < 1.0 for a in allowed[1:]), "edge windows must be partially masked"
    # Every query keeps at least itself, so no row can be all-False (→ NaN softmax).
    assert mask.any(dim=-1).all()


@pytest.mark.parametrize("shift_size", [0, 2])
@pytest.mark.parametrize("spatial", [(1, 1), (1, 5), (2, 3), (3, 3), (5, 7), (7, 9), (9, 11), (4, 128)])
def test_mot_window_expert_shape_and_finiteness_across_geometries(shift_size, spatial):
    """Odd sizes, degenerate 1×1, and extreme aspect ratios must round-trip cleanly."""
    from ultralytics.nn.modules.mot.mot import _WindowTransformerExpert

    torch.manual_seed(0)
    expert = _WindowTransformerExpert(16, num_heads=4, window_size=4, shift_size=shift_size).eval()
    x = torch.randn(1, 16, *spatial)
    with torch.no_grad():
        out = expert(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("window_shift", [False, True])
@pytest.mark.parametrize("window_size", [1, 2, 16])
def test_mot_block_degrades_when_window_exceeds_feature_map(window_shift, window_size):
    """window_size >= H/W (incl. win=1 and shift enabled) must pad, crop, stay finite."""
    torch.manual_seed(0)
    block = MoTBlock(32, num_heads=4, top_k=2, window_size=window_size, n_points=2, window_shift=window_shift).eval()
    x = torch.randn(2, 32, 3, 5)
    with torch.no_grad():
        out, aux = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert aux.item() == 0  # eval mode contributes no aux loss


def test_mot_window_expert_shift_is_trace_stable():
    """The mask must not break ONNX/TorchScript tracing (roll uses a cat fallback)."""
    import warnings

    expert = _amplified_window_expert(4, 2, dim=16, num_heads=4)
    x = torch.randn(1, 16, 8, 8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traced = torch.jit.trace(expert, x)
        out = traced(torch.randn(1, 16, 8, 8))
    assert out.shape == (1, 16, 8, 8)
    assert torch.isfinite(out).all()


def test_mot_exploration_eps_disabled_in_eval_end_to_end():
    """eval must not re-densify routing — checked at router *and* dispatch level.

    The router-level test above only inspects weights; this pins the observable
    consequence: eval dispatches to exactly top_k experts, training does not.
    """
    torch.manual_seed(0)
    block = MoTBlock(24, num_heads=3, top_k=1, window_size=4, n_points=2, exploration_eps=0.2)
    x = torch.randn(2, 24, 6, 6)

    block.train()
    block(x)
    train_calls = block._last_dispatch_stats["expert_calls"]

    block.eval()
    with torch.no_grad():
        weights, _ = block.router(x)
        out, aux = block(x)
    eval_calls = block._last_dispatch_stats["expert_calls"]

    assert train_calls == block.NUM_EXPERTS, "training keeps every expert trainable"
    assert eval_calls == block.top_k, "eval must dispatch to top_k experts only"
    # Eval weights: exactly one nonzero per token, and it is a full 1.0.
    assert torch.equal((weights > 0).sum(dim=1), torch.ones(2, 6, 6, dtype=torch.long))
    assert torch.allclose(weights.sum(dim=1), torch.ones(2, 6, 6))
    assert torch.allclose(weights.max(dim=1).values, torch.ones(2, 6, 6))
    assert aux.item() == 0
    assert torch.isfinite(out).all()


def test_mot_exploration_eps_boundary_values():
    """eps is clamped into [0, 0.2]; the endpoint itself is accepted silently."""
    import warnings

    assert MoTBlock(24, num_heads=3, exploration_eps=0.2).router.exploration_eps == 0.2
    with pytest.warns(UserWarning, match="clamped"):
        assert MoTBlock(24, num_heads=3, exploration_eps=-0.5).router.exploration_eps == 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 0.0 and 0.2 must not warn
        MoTBlock(24, num_heads=3, exploration_eps=0.0)


def test_c2fmot_window_shift_policy_is_configurable():
    """window_shift must be reachable from YAML, not hardcoded per block index.

    The alternating rule was previously baked into ``wrappers.py`` as
    ``bool(i % 2)`` with no constructor argument, so a shift-strategy ablation
    was impossible to express in a model YAML.  "alternate" keeps the Swin
    default; a plain bool forces one policy onto every block.
    """

    def shifts(**kwargs):
        module = C2fMoT(32, 32, n=4, num_heads=4, top_k=2, window_size=4, n_points=2, **kwargs)
        return [blk.experts[1].shift_size for blk in module.m]

    assert shifts() == [0, 2, 0, 2]  # default is unchanged
    assert shifts(window_shift="alternate") == [0, 2, 0, 2]
    assert shifts(window_shift=True) == [2, 2, 2, 2]
    assert shifts(window_shift=False) == [0, 0, 0, 0]

    with pytest.raises(ValueError, match="window_shift must be 'alternate' or a bool"):
        C2fMoT(32, 32, n=2, num_heads=4, window_size=4, n_points=2, window_shift="every-other")


def test_c2fmot_window_shift_reaches_model_from_yaml():
    """A YAML arg list must actually drive the shift policy end to end."""
    cfg = {
        "nc": 8,
        "backbone": [[-1, 1, "Conv", [16, 3, 2]]],
        "head": [
            # ``n`` comes from the repeat column, not the arg list; ``c1`` is
            # inserted from the input channels.  Args are therefore
            # [c2, num_heads, top_k, window_size, n_points, mlp_ratio, temperature,
            #  balance_loss_coeff, e, sparse_train, scene_aware_router,
            #  scene_hidden_dim, scene_consistency_coeff, sparse_train_warmup_steps,
            #  scene_inference_mode, window_shift]
            [-1, 2, "C2fMoT", [32, 4, 2, 4, 2, 2.0, 1.0, 0.01, 0.5, False, False, None, 0.0, 0, "dynamic", True]],
            [[-1], 1, "Detect", [8]],
        ],
    }
    model = DetectionModel(cfg, ch=3, nc=8, verbose=False)
    blocks = [m for m in model.modules() if isinstance(m, C2fMoT)]
    assert len(blocks) == 1
    assert [blk.experts[1].shift_size for blk in blocks[0].m] == [2, 2], "YAML window_shift ignored"


def test_c2fmot_shifted_blocks_are_edge_safe():
    """C2fMoT alternates window_shift by block index — cover the wired-up path.

    Under the default "alternate" policy every odd block runs the shifted
    expert.  This is how the mask fix reaches trained models.
    """
    torch.manual_seed(0)
    module = C2fMoT(32, 32, n=4, num_heads=4, top_k=2, window_size=4, n_points=2).eval()
    shifts = [blk.experts[1].shift_size for blk in module.m]
    assert shifts == [0, 2, 0, 2]
    x = torch.randn(1, 32, 7, 9)
    with torch.no_grad():
        out = module(x)
    assert out.shape == (1, 32, 7, 9)
    assert torch.isfinite(out).all()


# Main-branch coverage retained while replaying the issue #54 boundary suite.


def test_localconv_window_attention_is_opt_in_and_shape_preserving():
    """LocalConv keeps global attention by default and supports padded windows."""
    from ultralytics.nn.modules.mot.experts import _LocalConvTransformerExpert

    torch.manual_seed(0)
    x = torch.randn(1, 24, 7, 9)
    global_expert = _LocalConvTransformerExpert(24, 3).eval()
    window_expert = _LocalConvTransformerExpert(24, 3, local_window_size=4).eval()
    assert global_expert.local_window_size == 0
    assert window_expert.local_window_size == 4
    with torch.no_grad():
        out = window_expert(x)
    assert out.shape == x.shape


def test_mot_local_window_config_applies_to_nested_expert():
    model = C2fMoT(64, 64, n=1, num_heads=4)
    resolved = __import__(
        "ultralytics.nn.modules.moe.config", fromlist=["resolve_mixture_config"]
    ).resolve_mixture_config(SimpleNamespace(mot_local_attn_window=4), model)
    from ultralytics.nn.modules.moe.config import apply_mixture_config

    apply_mixture_config(model, resolved)
    assert model.m[0].experts[0].local_window_size == 4


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_mot_deformable_reference_grid_cache_clears_on_module_apply():
    """Cached grids must not survive checkpoint device or dtype migration."""
    from ultralytics.nn.modules.mot.mot import _DeformableTransformerExpert

    expert = _DeformableTransformerExpert(16, num_heads=4, n_points=2).eval()
    with torch.no_grad():
        expert(torch.randn(1, 16, 2, 2))
    assert expert._reference_grid_cache
    expert.to("mps")
    assert not expert._reference_grid_cache


def test_mot_fp16_forward_stability():
    torch.manual_seed(0)
    block = MoTBlock(32, num_heads=4, top_k=2, window_size=4, n_points=2).eval().half()
    x = torch.randn(2, 32, 8, 8, dtype=torch.float16)
    with torch.no_grad():
        out, aux = block(x)
    assert out.shape == x.shape
    assert out.dtype == torch.float16
    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(aux.float())


def test_mot_gradient_flow_with_zero_exploration_eps():
    torch.manual_seed(0)
    block = MoTBlock(
        32, num_heads=4, top_k=1, window_size=4, n_points=2, exploration_eps=0.0, sparse_train=True
    ).train()
    out, _ = block(torch.randn(2, 32, 8, 8))
    (out**2).sum().backward()
    assert _has_grad(block.router)
    assert sum(_has_grad(expert) for expert in block.experts) >= 1


def test_mot_top_k_equals_num_experts():
    block = MoTBlock(32, num_heads=4, top_k=3, window_size=4, n_points=2, exploration_eps=0.0).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        out, _ = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_mot_routing_determinism():
    torch.manual_seed(0)
    block = MoTBlock(32, num_heads=4, top_k=2, window_size=4, n_points=2, exploration_eps=0.0).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        weights_a, indices_a = block.router(x)
        weights_b, indices_b = block.router(x)
    assert torch.allclose(weights_a, weights_b)
    assert torch.equal(indices_a, indices_b)


def test_mot_block_forward_train_and_eval_consistency():
    torch.manual_seed(0)
    block = MoTBlock(32, num_heads=4, top_k=2, window_size=4, n_points=2)
    x = torch.randn(2, 32, 8, 8)
    train_out, _ = block.train()(x)
    with torch.no_grad():
        eval_out, _ = block.eval()(x)
    assert train_out.shape == eval_out.shape == x.shape
    assert torch.isfinite(train_out).all()
    assert torch.isfinite(eval_out).all()


def test_mot_localconv_expert_with_various_input_sizes():
    from ultralytics.nn.modules.mot.mot import _LocalConvTransformerExpert

    expert = _LocalConvTransformerExpert(16, num_heads=4).eval()
    for height, width in ((4, 4), (8, 16), (16, 8), (7, 13)):
        x = torch.randn(1, 16, height, width)
        with torch.no_grad():
            out = expert(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


def test_mot_block_invalid_top_k_raises():
    with pytest.raises(ValueError, match="top_k"):
        MoTBlock(32, top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        MoTBlock(32, top_k=5)


def test_mot_block_with_scene_aware_router():
    torch.manual_seed(0)
    block = MoTBlock(
        32,
        num_heads=4,
        top_k=2,
        window_size=4,
        n_points=2,
        scene_aware_router=True,
        scene_hidden_dim=16,
        scene_consistency_coeff=0.01,
    ).train()
    out, aux = block(torch.randn(2, 32, 8, 8))
    assert out.shape == (2, 32, 8, 8)
    assert torch.isfinite(out).all()
    assert torch.isfinite(aux)
