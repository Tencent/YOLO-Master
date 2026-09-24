"""Protect native forward/head gradients and the opt-in shared-feature gradient scale."""

import copy
import io

import pytest
import torch

from ultralytics.nn.modules.head import Detect
from ultralytics.utils.patches import torch_load


def make_head():
    torch.manual_seed(260829)
    head = Detect(nc=3, reg_max=1, end2end=True, ch=(8, 16, 32))
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    head.max_det = 20
    head.train()
    for m in head.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
    return head


def features():
    return [torch.randn(1, c, s, s, requires_grad=True) for c, s in ((8, 8), (16, 4), (32, 2))]


def measure(head, x):
    outputs = head(x)
    loss = sum(outputs["one2one"][k].square().mean() for k in ("boxes", "scores"))
    params = [p for n, p in head.named_parameters() if "one2one_" in n]
    grads = torch.autograd.grad(loss, x + params, allow_unused=True)
    return outputs, grads


def test_bridge_forward_and_gradient_scaling():
    head = make_head()
    x = features()
    native, native_grads = measure(head, x)
    head.p2_o2o_gradient_alpha = 0.0
    zero, zero_grads = measure(head, x)
    head.p2_o2o_gradient_alpha = 1.0
    full, full_grads = measure(head, x)
    head.p2_o2o_gradient_alpha = 0.1
    bridged, bridge_grads = measure(head, x)
    for branch in ("one2one", "one2many"):
        for key in ("boxes", "scores"):
            assert torch.equal(native[branch][key], zero[branch][key])
            assert torch.equal(native[branch][key], full[branch][key])
            assert torch.equal(native[branch][key], bridged[branch][key])
    assert all(g is None for g in native_grads[:3] + zero_grads[:3])
    for whole, partial in zip(full_grads[:3], bridge_grads[:3]):
        assert partial.abs().sum() > 0
        assert torch.allclose(partial, whole * 0.1, atol=1e-7, rtol=1e-5)
    for native_g, zero_g, bridge_g in zip(native_grads[3:], zero_grads[3:], bridge_grads[3:]):
        assert torch.equal(native_g, zero_g)
        assert torch.equal(native_g, bridge_g)


@pytest.mark.parametrize("export", [False, True])
def test_bridge_evaluation_and_export_unchanged(export):
    head = make_head().eval()
    head.export = export
    x = features()
    baseline = head(x)
    head.p2_o2o_gradient_alpha = 0.1
    result = head(x)
    assert torch.equal(baseline if export else baseline[0], result if export else result[0])


def test_bridge_checkpoint_roundtrip():
    head = make_head()
    head.p2_o2o_gradient_alpha = 0.1
    stream = io.BytesIO()
    torch.save(copy.deepcopy(head), stream)
    stream.seek(0)
    loaded = torch_load(stream, weights_only=False)
    assert loaded.p2_o2o_gradient_alpha == 0.1
    assert all(torch.equal(t, loaded.state_dict()[n]) for n, t in head.state_dict().items())
    _, grads = measure(loaded, features())
    assert all(g.abs().sum() > 0 for g in grads[:3])


def test_probe_reports_detach_and_scaled_gradients_without_rng_side_effects():
    from scripts.a1.diagnose_o2o_gradients import diagnose

    before = torch.random.get_rng_state().clone()
    native, full, partial = diagnose(), diagnose(1.0), diagnose(0.1)
    assert torch.equal(before, torch.random.get_rng_state())
    assert native["input_gradient_l2"]["one2one"] == [None] * 3
    assert native["hooks_remaining"] == full["hooks_remaining"] == partial["hooks_remaining"] == 0
    for a, b in zip(full["input_gradient_l2"]["one2one"], partial["input_gradient_l2"]["one2one"]):
        assert b == pytest.approx(a * 0.1, rel=1e-5)
    assert native["input_gradient_l2"]["one2many"] == partial["input_gradient_l2"]["one2many"]


def test_non_e2e_training_ignores_bridge():
    head = make_head()
    head.end2end = False
    x = features()
    native = head(x)
    head.p2_o2o_gradient_alpha = 0.1
    actual = head(x)
    for key in ("boxes", "scores"):
        assert torch.equal(native[key], actual[key])


def test_detect_subclass_rejects_training_intervention():
    class OtherHead(Detect):
        pass

    head = OtherHead(nc=3, reg_max=1, end2end=True, ch=(8, 16, 32)).eval()
    head.training = True
    head.p2_o2o_gradient_alpha = 0.1
    with pytest.raises(ValueError, match="requires Detect"):
        head(features())


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
def test_bridge_rejects_invalid_alpha(alpha):
    head = make_head()
    head.p2_o2o_gradient_alpha = alpha
    with pytest.raises(ValueError, match="finite alpha"):
        head(features())
