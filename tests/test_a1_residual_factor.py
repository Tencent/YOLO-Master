"""A1 zero-gated factors preserve pretrained outputs and frozen state."""

import copy
import io

import pytest
import torch
from torch import nn

from ultralytics.nn.modules.moe.factor_adapter import ResidualFactorAdapter


def make_adapter():
    """Use a BN-equipped base to detect accidental buffer updates."""
    torch.manual_seed(29)
    base = nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4)).eval()
    base.i, base.f, base.type = 4, -1, "test-base"
    factor = nn.Sequential(nn.Conv2d(4, 4, 1), nn.SiLU())
    return ResidualFactorAdapter(base, factor, 4)


def test_zero_gain_and_checkpoint_preserve_outputs():
    model = make_adapter().eval()
    x = torch.randn(2, 4, 8, 8)
    expected = model.base(x)
    assert torch.equal(model(x), expected)
    assert (model.i, model.f, model.type) == (4, -1, "test-base")
    assert torch.count_nonzero(model.gain) == 0
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    loaded = make_adapter().eval()
    loaded.load_state_dict(torch.load(stream, map_location="cpu"))
    assert torch.equal(loaded(x), expected)


def test_optimizer_step_keeps_base_weights_and_bn_frozen():
    model = make_adapter().train()
    before = copy.deepcopy(model.base.state_dict())
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.01)
    x = torch.randn(2, 4, 8, 8, requires_grad=True)
    model(x).square().mean().backward()
    assert model.gain.grad.abs().sum() > 0
    assert x.grad.abs().sum() > 0  # freezing weights must not detach earlier layers
    assert all(p.grad is None for p in model.base.parameters())
    optimizer.step()
    optimizer.zero_grad()
    model(x).square().mean().backward()
    assert sum(p.grad.abs().sum() for p in model.factor.parameters()) > 0
    assert all(torch.equal(v, model.base.state_dict()[k]) for k, v in before.items())


def test_freeze_can_be_reapplied_after_trainer_reset():
    model = make_adapter()
    model.requires_grad_(True)
    assert model.freeze_base_parameters() == sum(p.numel() for p in model.base.parameters())
    assert all(not p.requires_grad for p in model.base.parameters())
    assert all(p.requires_grad for p in model.factor.parameters())
    assert model.gain.requires_grad


def test_unfrozen_base_follows_training_mode():
    model = ResidualFactorAdapter(nn.BatchNorm2d(4), nn.Identity(), 4, freeze_base=False)
    model.train()
    assert model.base.training
    assert all(p.requires_grad for p in model.base.parameters())
    model.eval()
    assert not model.base.training


def test_invalid_channels():
    with pytest.raises(ValueError, match="channels"):
        ResidualFactorAdapter(nn.Identity(), nn.Identity(), 0)


@pytest.mark.parametrize("moe", [False, True])
def test_native_dense_and_moe_factors_have_exact_initial_output(moe):
    from ultralytics.nn.modules.block import A2C2f, C3k2
    from ultralytics.nn.modules.moe.modules import A2C2fMoE

    base = C3k2(64, 64).eval()
    factor = A2C2fMoE(64, 64, n=1, area=1, num_experts=2, top_k=1) if moe else A2C2f(64, 64, n=1, area=1)
    adapter = ResidualFactorAdapter(base, factor, 64).eval()
    x = torch.randn(1, 64, 8, 8)
    with torch.no_grad():
        assert torch.equal(adapter(x), base(x))
