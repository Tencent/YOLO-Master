"""Regression tests for minimum-supported PyTorch APIs."""

from contextlib import nullcontext
from unittest.mock import Mock

import torch

from ultralytics.nn.modules import _numeric
from ultralytics.utils import patches, torch_utils


def test_torch_load_drops_unsupported_weights_only(monkeypatch):
    loader = Mock(return_value={"ok": True})
    monkeypatch.setattr(torch_utils, "TORCH_1_13", False)
    monkeypatch.setattr(patches.torch, "load", loader)

    assert patches.torch_load("checkpoint.pt", map_location="cpu", weights_only=False) == {"ok": True}
    loader.assert_called_once_with("checkpoint.pt", map_location="cpu")


def test_disabled_autocast_uses_legacy_cuda_context_without_unified_api(monkeypatch):
    legacy_autocast = Mock(return_value=nullcontext())
    monkeypatch.delattr(_numeric.torch, "autocast", raising=False)
    monkeypatch.setattr(_numeric.torch.cuda.amp, "autocast", legacy_autocast)

    with _numeric.disabled_autocast("cuda"):
        pass

    legacy_autocast.assert_called_once_with(enabled=False)


def test_disabled_autocast_is_noop_for_unsupported_legacy_device(monkeypatch):
    legacy_autocast = Mock(side_effect=AssertionError("CPU must not use CUDA autocast"))
    monkeypatch.delattr(_numeric.torch, "autocast", raising=False)
    monkeypatch.setattr(_numeric.torch.cuda.amp, "autocast", legacy_autocast)

    with _numeric.disabled_autocast("cpu"):
        result = torch.ones(1) + 1

    assert result.item() == 2
    legacy_autocast.assert_not_called()
