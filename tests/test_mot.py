from pathlib import Path
from types import SimpleNamespace

import torch

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.modules.moa import C2fMoA
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
    torch.manual_seed(0)
    module = MoTBlock(32, num_heads=4, top_k=2, window_size=4, n_points=2, mlp_ratio=1.5).train()
    x = torch.randn(2, 32, 8, 8)

    out, aux = module(x)
    assert out.shape == x.shape
    assert aux.requires_grad
    assert torch.isfinite(aux)

    (out.mean() + aux).backward()
    assert _has_grad(module.router)
    for expert in module.experts:
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
    before = [m.router.temperature for m in module.m]
    anneal_mot_temperature(module, factor=0.5, min_temp=0.3)
    after = [m.router.temperature for m in module.m]
    assert after == [max(t * 0.5, 0.3) for t in before]


def test_trainer_detects_and_anneals_moa_mot_temperatures():
    trainer = object.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(moa_mot_temperature_factor=0.5, moa_mot_min_temperature=0.3)
    moa = C2fMoA(64, 64, n=1, num_heads=4)
    mot = C2fMoT(64, 64, n=1, num_heads=4)
    trainer.model = torch.nn.Sequential(moa, mot)

    trainer._detect_moa_mot_modules()
    assert trainer._has_moa_mot is True

    moa_before = [m.router.temperature for m in moa.m]
    mot_before = [m.router.temperature for m in mot.m]
    trainer._anneal_moa_mot_temperature()
    assert [m.router.temperature for m in moa.m] == [max(t * 0.5, 0.3) for t in moa_before]
    assert [m.router.temperature for m in mot.m] == [max(t * 0.5, 0.3) for t in mot_before]


def test_mot_model_configs_parse():
    configs = [
        ROOT / "ultralytics/cfg/models/master/v0_8/det/yolo-master-mot-n.yaml",
        ROOT / "ultralytics/cfg/models/master/v0_8/det/yolo-master-moa-mot-n.yaml",
    ]
    for cfg in configs:
        model = DetectionModel(str(cfg), ch=3, nc=80, verbose=False)
        assert sum(isinstance(m, C2fMoT) for m in model.modules()) == 3
        assert sum(isinstance(m, MoTBlock) for m in model.modules()) == 6
