import torch
import torch.nn as nn
import pytest

from ultralytics.nn.peft.molora.layer import MoLoRALayer


def test_dynamic_merge_requires_explicit_mode_and_is_approximate():
    layer = MoLoRALayer(nn.Linear(8, 8), r=2, num_experts=2, top_k=1).eval()
    x = torch.randn(3, 8)
    dynamic = layer(x)
    original = layer.base_layer.weight.detach().clone()
    layer.merge_weights(mode="uniform")
    assert layer._merge_metadata["approximate"] is True
    merged = layer(x)
    assert merged.shape == dynamic.shape
    layer.unmerge_weights()
    assert torch.allclose(layer.base_layer.weight, original)


def test_calibrated_merge_validates_and_records_weights():
    layer = MoLoRALayer(nn.Linear(8, 8), r=2, num_experts=2, top_k=1).eval()
    with pytest.raises(ValueError):
        layer.merge_weights(mode="calibrated")
    layer.merge_weights(mode="calibrated", calibration=[0.8, 0.2])
    assert layer._merge_metadata["mode"] == "calibrated"
    assert sum(layer._merge_metadata["expert_weights"]) == pytest.approx(1.0)
