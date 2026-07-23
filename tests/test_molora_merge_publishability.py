"""MoLoRA merge artifact publishability contracts."""

import torch
import torch.nn as nn

from ultralytics.nn.peft.molora import MoLoRAConfig, MoLoRAModel
from ultralytics.utils.lora import adapter_metadata, merge_adapters


def _model():
    return MoLoRAModel(
        nn.Sequential(nn.Linear(8, 8)),
        MoLoRAConfig(r=2, alpha=4, num_experts=2, top_k=1, target_modules=["0"]),
    ).eval()


def test_calibrated_merge_has_reproducibility_fingerprint():
    model = _model()
    assert merge_adapters(model, mode="calibrated", calibration_data=[torch.randn(2, 8)])
    metadata = adapter_metadata(model)
    records = metadata["merge_records"]
    assert records[0]["calibration_fingerprint"]
    assert metadata["publishable_merge"] is True


def test_ema_merge_is_not_publishable_without_calibration():
    metadata_model = _model()
    assert merge_adapters(metadata_model, mode="ema")
    assert adapter_metadata(metadata_model)["publishable_merge"] is False
