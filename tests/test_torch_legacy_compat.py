"""Regression tests for the minimum supported PyTorch APIs."""

from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import torch

from ultralytics.nn.modules._numeric import disabled_autocast
from ultralytics.utils import patches, torch_utils


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps"])
def test_disabled_autocast_uses_modern_api(monkeypatch, device_type):
    context = nullcontext()
    modern_autocast = Mock(return_value=context)
    legacy_autocast = Mock(return_value=nullcontext())
    monkeypatch.setattr(torch, "autocast", modern_autocast, raising=False)
    monkeypatch.setattr(torch.cuda.amp, "autocast", legacy_autocast)

    assert disabled_autocast(device_type) is context
    modern_autocast.assert_called_once_with(device_type=device_type, enabled=False)
    legacy_autocast.assert_not_called()


def test_disabled_autocast_uses_legacy_cuda_api(monkeypatch):
    context = nullcontext()
    legacy_autocast = Mock(return_value=context)
    monkeypatch.delattr(torch, "autocast", raising=False)
    monkeypatch.setattr(torch.cuda.amp, "autocast", legacy_autocast)

    assert disabled_autocast("cuda") is context
    legacy_autocast.assert_called_once_with(enabled=False)


def test_disabled_autocast_legacy_cpu_is_noop(monkeypatch):
    legacy_autocast = Mock(return_value=nullcontext())
    monkeypatch.delattr(torch, "autocast", raising=False)
    monkeypatch.setattr(torch.cuda.amp, "autocast", legacy_autocast)

    with disabled_autocast("cpu") as value:
        assert value is None
    legacy_autocast.assert_not_called()


def test_disabled_autocast_legacy_mps_is_noop(monkeypatch):
    legacy_autocast = Mock(return_value=nullcontext())
    monkeypatch.delattr(torch, "autocast", raising=False)
    monkeypatch.setattr(torch.cuda.amp, "autocast", legacy_autocast)

    with disabled_autocast("mps") as value:
        assert value is None
    legacy_autocast.assert_not_called()


def test_disabled_autocast_unknown_device_is_noop(monkeypatch):
    modern_autocast = Mock(return_value=nullcontext())
    legacy_autocast = Mock(return_value=nullcontext())
    monkeypatch.setattr(torch, "autocast", modern_autocast, raising=False)
    monkeypatch.setattr(torch.cuda.amp, "autocast", legacy_autocast)

    with disabled_autocast("unknown") as value:
        assert value is None
    modern_autocast.assert_not_called()
    legacy_autocast.assert_not_called()


def test_torch_load_drops_unsupported_weights_only(monkeypatch):
    loader = Mock(return_value={"ok": True})
    monkeypatch.setattr(torch_utils, "TORCH_1_13", False)
    monkeypatch.setattr(patches.torch, "load", loader)

    assert patches.torch_load("checkpoint.pt", map_location="cpu", weights_only=False) == {"ok": True}
    loader.assert_called_once_with("checkpoint.pt", map_location="cpu")
