"""Exercise the production A1 hook entrypoints on an ordinary Detect model."""

from contextlib import ExitStack

import pytest
import torch

from scripts.a1.diagnostic_hooks import capture_head_inputs, install_first_forward_check
from ultralytics.nn.tasks import DetectionModel


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [t for v in value.values() for t in tensors(v)]
    return [t for v in value for t in tensors(v)] if isinstance(value, (list, tuple)) else []


@pytest.fixture
def model():
    torch.manual_seed(17)
    return DetectionModel("yolo26n.yaml", nc=3, verbose=False).eval()


@pytest.mark.parametrize("end2end", [False, True])
def test_native_model_capture_and_one_shot_preserve_outputs(model, tmp_path, end2end):
    head = model.model[-1]
    head.end2end = end2end
    x = torch.randn(1, 3, 64, 64)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        baseline = tensors(model(x))
        with capture_head_inputs(head) as captured:
            actual = tensors(model(x))
            assert len(captured) == 3
            assert all(torch.equal(a, b) for a, b in zip(baseline, actual))
        assert captured == []
        assert not head._forward_pre_hooks
        assert all(torch.equal(a, b) for a, b in zip(baseline, tensors(model(x))))
    if end2end:
        # Keep BN in eval mode but exercise native training-mode head outputs.
        head.training = True
        baseline = tensors(model(x))
        with ExitStack() as cleanup:
            install_first_forward_check(head, 0, tmp_path / "first.json", cleanup)
            actual = tensors(model(x))
            assert len(actual) == len(baseline)
            assert all(torch.equal(a, b) for a, b in zip(baseline, actual))
            assert not head._forward_hooks
            assert (tmp_path / "first.json").is_file()
        assert all(torch.equal(a, b) for a, b in zip(baseline, tensors(model(x))))
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())


@pytest.mark.parametrize("failure", ["forward", "assertion", "write"])
def test_real_hooks_cleanup_and_propagate_errors(model, tmp_path, monkeypatch, failure):
    head = model.model[-1]
    head.training = True
    expected = {"forward": ValueError, "assertion": RuntimeError, "write": OSError}[failure]
    if failure == "forward":

        def broken(*args, **kwargs):
            raise ValueError("forward failed")

        monkeypatch.setattr(head, "forward", broken)
    output = tmp_path if failure == "write" else tmp_path / "first.json"
    with pytest.raises(expected), ExitStack() as cleanup, capture_head_inputs(head) as captured:
        install_first_forward_check(head, 1 if failure == "assertion" else 0, output, cleanup)
        model(torch.randn(1, 3, 64, 64))
    assert captured == []
    assert not head._forward_hooks
    assert not head._forward_pre_hooks


def test_cleanup_before_any_forward(model, tmp_path):
    head = model.model[-1]
    with ExitStack() as cleanup:
        install_first_forward_check(head, 0, tmp_path / "first.json", cleanup)
    assert not head._forward_hooks
    assert not (tmp_path / "first.json").exists()
