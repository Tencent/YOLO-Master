"""Regression tests for the minimum supported PyTorch loader API."""

from unittest.mock import Mock

from ultralytics.utils import patches, torch_utils


def test_torch_load_drops_unsupported_weights_only(monkeypatch):
    loader = Mock(return_value={"ok": True})
    monkeypatch.setattr(torch_utils, "TORCH_1_13", False)
    monkeypatch.setattr(patches.torch, "load", loader)

    assert patches.torch_load("checkpoint.pt", map_location="cpu", weights_only=False) == {"ok": True}
    loader.assert_called_once_with("checkpoint.pt", map_location="cpu")
