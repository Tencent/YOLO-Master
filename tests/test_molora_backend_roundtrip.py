"""Focused MoLoRA backend roundtrip and calibrated merge coverage."""

import torch
import torch.nn as nn

from ultralytics.nn.peft.molora import MoLoRAConfig, MoLoRAModel
from ultralytics.utils.lora import load_adapters, merge_adapters, save_adapters


def _make_model() -> MoLoRAModel:
    return MoLoRAModel(
        nn.Sequential(nn.Linear(8, 8)),
        MoLoRAConfig(r=2, alpha=4, num_experts=2, top_k=1, target_modules=["0"]),
    )


def test_molora_backend_roundtrip_preserves_forward_and_usage(tmp_path):
    torch.manual_seed(7)
    source = _make_model().eval()
    batch = torch.randn(4, 8)
    with torch.no_grad():
        expected = source(batch)

    path = tmp_path / "adapter"
    assert save_adapters(source, path)

    restored = _make_model().eval()
    restored.model[0].base_layer.load_state_dict(source.model[0].base_layer.state_dict())
    assert load_adapters(restored, path)
    with torch.no_grad():
        actual = restored(batch)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.equal(
        restored.model[0]._usage_ema,
        source.model[0]._usage_ema,
    )


def test_calibrated_merge_records_publishable_metadata(tmp_path):
    model = _make_model().eval()
    calibration = [torch.randn(4, 8), torch.randn(4, 8)]
    assert merge_adapters(
        model,
        mode="calibrated",
        calibration_data=calibration,
    )
    metadata_path = tmp_path / "merged"
    assert save_adapters(model, metadata_path)
    payload = (metadata_path / "runtime_metadata.json").read_text()
    assert '"merge_mode": "dynamic"' in payload
    assert '"mode": "calibrated"' in payload
