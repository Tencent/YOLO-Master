"""End-to-end compatibility matrix for YOLO26 and routed model variants."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ultralytics.cfg import get_cfg
from ultralytics.engine.exporter import Exporter
from ultralytics.engine.model import Model
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.peft.molora import MoLoRAConfig, MoLoRAConfigBuilder, get_peft_molora_model
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils.checkpoint_compat import graph_metadata
from ultralytics.utils.export_preflight import export_preflight
from ultralytics.utils.export_validation import validate_export_roundtrip

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "dense": ("yolo26.yaml", set()),
    "moe": ("yolo26-master-n.yaml", {"moe"}),
    "moa": ("yolo26-master-moa-n.yaml", {"moa"}),
    "mot": ("yolo26-master-mot-n.yaml", {"mot"}),
    "moa_mot": ("yolo26-master-moa-mot-n.yaml", {"moa", "mot"}),
    "molora": ("yolo26.yaml", {"molora"}),
}


def _build_variant(name: str) -> DetectionModel:
    config = ROOT / "ultralytics/cfg/models/26" / VARIANTS[name][0]
    model = DetectionModel(config, ch=3, nc=5, verbose=False)
    model.args = get_cfg(overrides={"box": 7.5, "cls": 0.5, "dfl": 1.5})
    if name == "molora":
        get_peft_molora_model(
            model,
            MoLoRAConfig(
                r=2,
                alpha=4,
                num_experts=2,
                top_k=1,
                target_modules=["model.4.cv1.conv"],
            ),
        )
    return model


def _detection_batch() -> dict[str, torch.Tensor]:
    return {
        "img": torch.rand(2, 3, 64, 64),
        "batch_idx": torch.tensor([0, 1]),
        "cls": torch.tensor([[1.0], [2.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25], [0.4, 0.4, 0.2, 0.2]]),
    }


class _PredictionTensor(nn.Module):
    """Expose the decoded end-to-end tensor as a stable export output."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        return output[0] if isinstance(output, tuple) else output


@pytest.mark.parametrize("variant", VARIANTS)
def test_yolo26_mixture_train_inference_and_graph_contract(variant):
    model = _build_variant(variant)
    head = model.model[-1]
    metadata = graph_metadata(model)

    assert head.reg_max == 1
    assert head.end2end is True
    assert hasattr(head, "one2many") and hasattr(head, "one2one")
    assert set(metadata["routed_families"]) == VARIANTS[variant][1]

    model.eval()
    with torch.inference_mode():
        prediction = model(torch.zeros(1, 3, 64, 64))[0]
    assert prediction.shape == (1, 84, 6)
    assert torch.isfinite(prediction).all()

    model.train()
    loss, items = model(_detection_batch())
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    assert items.numel() == (3 if variant == "dense" else 4)
    loss.sum().backward()
    if variant != "dense":
        routed = [
            module
            for module in model.modules()
            if module.__class__.__name__
            in {"A2C2fMoE", "C2fMoA", "C2fMoT", "MoLoRALayer"}
        ]
        assert routed
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
            for module in routed
            for parameter in module.parameters()
            if parameter.requires_grad
        )


@pytest.mark.parametrize("variant", VARIANTS)
def test_yolo26_mixture_export_preflight_matrix(variant):
    report = export_preflight(_build_variant(variant), "onnx", strict=True)
    assert report["supported"] is True
    assert {item["module_family"].lower() for item in report["decisions"]} == VARIANTS[variant][1]
    assert all(item["strategy"] == "dense_fallback" for item in report["decisions"])


def test_yolo26_molora_auto_targeting_excludes_head_by_default():
    model = DetectionModel(ROOT / "ultralytics/cfg/models/26/yolo26.yaml", ch=3, nc=5, verbose=False)
    head_config = MoLoRAConfigBuilder.create_molora_config(
        model,
        r=2,
        alpha=4,
        num_experts=2,
        top_k=1,
        include_head=True,
    )
    assert head_config is not None
    assert any(name.startswith("model.23.") for name in head_config["target_modules"])

    config = MoLoRAConfig(r=2, alpha=4, num_experts=2, top_k=1, target_modules=None)
    get_peft_molora_model(model, config)

    assert config.target_modules
    assert not any(name.startswith("model.23.") for name in config.target_modules)


@pytest.mark.slow
@pytest.mark.parametrize("variant", VARIANTS)
def test_yolo26_mixture_full_checkpoint_roundtrip(tmp_path, variant):
    model = _build_variant(variant).eval()
    facade = Model.__new__(Model)
    nn.Module.__init__(facade)
    facade.model = model
    facade.ckpt = {}
    checkpoint = tmp_path / f"{variant}.pt"

    facade.save(checkpoint)
    restored, payload = load_checkpoint(checkpoint, device="cpu")

    assert set(payload["mixture_checkpoint"]["graph"]["routed_families"]) == VARIANTS[variant][1]
    assert payload["mixture_checkpoint"]["graph"]["reg_max"] == 1
    assert payload["mixture_checkpoint"]["graph"]["end2end"] is True
    with torch.inference_mode():
        prediction = restored(torch.zeros(1, 3, 64, 64))[0]
    assert prediction.shape == (1, 84, 6)
    assert torch.isfinite(prediction).all()


@pytest.mark.slow
@pytest.mark.parametrize("fmt", ["torchscript", "onnx"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_yolo26_mixture_full_model_export_roundtrip(tmp_path, variant, fmt):
    if fmt == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    model = _build_variant(variant).eval()
    report = export_preflight(model, fmt, strict=True)
    assert report["supported"] is True
    if fmt == "torchscript":
        result = validate_export_roundtrip(_PredictionTensor(deepcopy(model)), torch.zeros(1, 3, 64, 64), fmt)
        assert result["passed"] is True
        assert result["artifact_bytes"] > 0
        return

    model.task = "detect"
    model.pt_path = str(tmp_path / f"{variant}.pt")
    exporter = Exporter(
        overrides={"format": "onnx", "imgsz": 64, "batch": 1, "device": "cpu", "simplify": False, "opset": 17}
    )
    artifact = exporter(model=model)
    backend = AutoBackend(artifact, device=torch.device("cpu"))
    output = backend(torch.zeros(1, 3, 64, 64))

    assert output.shape == (1, 84, 6)
    assert torch.isfinite(output).all()
    assert exporter.export_preflight_report["supported"] is True
    if variant != "dense":
        assert "mixture_export_preflight" in exporter.metadata
