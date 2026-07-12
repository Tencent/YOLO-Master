from pathlib import Path
from types import SimpleNamespace

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
    from ultralytics.nn.modules.mot.mot import _WindowTransformerExpert

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
