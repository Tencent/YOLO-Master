import hashlib
import json
from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from ultralytics.nn.peft.molora import MoLoRAConfig, MoLoRAModel
from ultralytics.utils.checkpoint_compat import (
    checkpoint_runtime_metadata,
    convert_checkpoint_artifact,
    graph_metadata,
    inspect_checkpoint_artifact,
    load_compatible_checkpoint,
    remap_checkpoint_state,
)
from ultralytics.utils.patches import torch_load


class ToyHead(nn.Module):
    def __init__(self, *, reg_max=1, end2end=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.reg_max = reg_max
        self.end2end = end2end
        if end2end:
            self.one2many = nn.Identity()
            self.one2one = nn.Identity()

    def forward(self, x):
        return x * self.weight


class ToyModel(nn.Module):
    def __init__(self, *, reg_max=1, end2end=True):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(2, 2), ToyHead(reg_max=reg_max, end2end=end2end))
        self.task = "detect"

    def forward(self, x):
        return self.model(x)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _full_checkpoint(path, model, *, version="8.3.240", ema=True):
    torch.save(
        {
            "model": deepcopy(model),
            "ema": deepcopy(model) if ema else None,
            "version": version,
            "train_args": {},
        },
        path,
    )


def test_graph_metadata_preserves_yolo26_contract_fields():
    metadata = graph_metadata(ToyModel())
    assert metadata["reg_max"] == 1
    assert metadata["end2end"] is True
    assert metadata["one2many"] is True
    assert metadata["one2one"] is True


def test_state_remap_handles_parallel_prefixes_only():
    source = {"module.weight": torch.ones(2, 2), "wrong": torch.ones(1)}
    target = {"weight": torch.zeros(2, 2), "bias": torch.zeros(2)}
    mapped, remap, missing, unexpected, mismatches = remap_checkpoint_state(source, target)
    assert set(mapped) == {"weight"}
    assert remap == {"module.weight": "weight"}
    assert missing == ["bias"]
    assert unexpected == ["wrong"]
    assert not mismatches


def test_full_checkpoint_conversion_is_read_only_and_preserves_graph(tmp_path):
    source = tmp_path / "legacy.pt"
    destination = tmp_path / "converted.pt"
    _full_checkpoint(source, ToyModel())
    before = _sha256(source)

    report = convert_checkpoint_artifact(source, destination)
    converted = torch_load(destination, map_location="cpu", weights_only=False)

    assert _sha256(source) == before
    assert report.source_version == "8.3.240"
    assert converted["version"] == "8.4.101"
    assert converted["model"].model[-1].reg_max == 1
    assert converted["model"].model[-1].end2end is True
    assert converted["mixture_checkpoint"]["graph"]["one2one"] is True


def test_head_mismatch_is_explicit_and_opt_in(tmp_path):
    source = tmp_path / "legacy.pt"
    _full_checkpoint(source, ToyModel(reg_max=16, end2end=False))
    target = ToyModel(reg_max=1, end2end=True)

    with pytest.raises(ValueError, match="allow_head_mismatch"):
        load_compatible_checkpoint(target, source)

    report = load_compatible_checkpoint(target, source, allow_head_mismatch=True)
    assert any("reg_max differs" in risk for risk in report.semantic_risks)
    assert any("end2end differs" in risk for risk in report.semantic_risks)


def test_conversion_rebuilds_online_and_ema_state(tmp_path):
    source = tmp_path / "legacy.pt"
    destination = tmp_path / "converted.pt"
    source_model = ToyModel()
    source_model.model[0].weight.data.fill_(3)
    _full_checkpoint(source, source_model)

    target = ToyModel()
    convert_checkpoint_artifact(source, destination, target_model=target)
    converted = torch_load(destination, map_location="cpu", weights_only=False)

    assert torch.equal(converted["model"].model[0].weight, torch.full_like(target.model[0].weight, 3))
    assert torch.equal(converted["ema"].model[0].weight, torch.full_like(target.model[0].weight, 3))
    assert set(converted["checkpoint_compat"]["state_reports"]) == {"model", "ema"}


def test_adapter_directory_conversion_preserves_payload(tmp_path):
    source = tmp_path / "adapter"
    source.mkdir()
    (source / "runtime_metadata.json").write_text(json.dumps({"backend": "peft", "variant": "lora"}))
    (source / "adapter_model.safetensors").write_bytes(b"adapter")
    destination = tmp_path / "converted-adapter"

    report = convert_checkpoint_artifact(source, destination)

    assert report.artifact_type == "adapter_directory"
    assert (destination / "adapter_model.safetensors").read_bytes() == b"adapter"
    assert (destination / "checkpoint_compatibility.json").exists()


def test_molora_checkpoint_conversion_and_inspection(tmp_path):
    model = MoLoRAModel(
        nn.Sequential(nn.Linear(4, 4)),
        MoLoRAConfig(r=2, alpha=4, num_experts=2, top_k=1, target_modules=["0"]),
    )
    source = tmp_path / "molora_adapter.pt"
    destination = tmp_path / "molora_adapter_v84101.pt"
    model.save_checkpoint(str(source))

    report = inspect_checkpoint_artifact(source)
    assert report.artifact_type == "molora_adapter"
    convert_checkpoint_artifact(source, destination)
    payload = torch_load(destination, map_location="cpu", weights_only=False)
    assert payload["format"] == "molora_adapter"
    assert payload["target_version"] == "8.4.101"


def test_checkpoint_runtime_metadata_records_adapter_and_graph():
    model = ToyModel()
    model.lora_enabled = True
    model.lora_backend = "fallback"
    model.lora_variant = "lora"
    metadata = checkpoint_runtime_metadata(model)
    assert metadata["graph"]["head_class"].endswith("ToyHead")
    assert metadata["adapter"]["backend"] == "lora"
