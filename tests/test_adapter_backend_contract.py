import json

import torch
import torch.nn as nn

from ultralytics.nn.peft.molora import MoLoRAConfig, MoLoRAModel
from ultralytics.utils.lora import (
    adapter_metadata,
    discover_adapter_backend,
    load_adapters,
    merge_adapters,
    save_adapters,
)
from ultralytics.utils.lora.io import save_lora_adapters


def _model():
    base = nn.Sequential(nn.Linear(8, 8))
    return MoLoRAModel(
        base,
        MoLoRAConfig(r=2, alpha=4, num_experts=2, top_k=1, target_modules=["0"]),
    )


def test_molora_backend_discovery_and_artifact(tmp_path):
    model = _model().train()
    model(torch.randn(2, 8))
    backend = discover_adapter_backend(model)
    assert backend is not None and backend.name == "molora"
    path = tmp_path / "adapter"
    assert save_adapters(model, path)
    payload = json.loads((path / "runtime_metadata.json").read_text())
    assert payload["backend"] == "molora"
    assert payload["schema_version"] == 1
    assert (path / "molora_adapter.pt").exists()


def test_legacy_save_lora_entrypoint_dispatches_molora(tmp_path):
    model = _model()
    path = tmp_path / "legacy-entrypoint"

    assert save_lora_adapters(model, path)
    assert (path / "molora_adapter.pt").exists()


def test_molora_metadata_declares_dynamic_non_exact_merge():
    model = _model()
    metadata = adapter_metadata(model)
    assert metadata["backend"] == "molora"
    assert metadata["exact_merge"] is False
    assert metadata["merge_mode"] == "dynamic"


def test_molora_backend_defaults_to_ema_merge():
    model = _model()
    assert merge_adapters(model)
    records = adapter_metadata(model)["merge_records"]
    assert records and all(record["mode"] == "ema" for record in records)


def test_molora_checkpoint_round_trip_via_backend(tmp_path):
    model = _model().train()
    model(torch.randn(2, 8))
    path = tmp_path / "adapter"
    save_adapters(model, path)
    restored = _model()
    assert load_adapters(restored, path)
