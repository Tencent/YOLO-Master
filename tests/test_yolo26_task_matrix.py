"""Lifecycle coverage for YOLO26 specialized task heads."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ultralytics.cfg import get_cfg
from ultralytics.engine.exporter import Exporter
from ultralytics.engine.model import Model
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.peft.molora import MoLoRAConfig, MoLoRALayer, get_peft_molora_model
from ultralytics.nn.tasks import OBBModel, PoseModel, SegmentationModel, SemanticSegmentationModel, load_checkpoint
from ultralytics.utils.checkpoint_compat import graph_metadata
from ultralytics.utils.export_preflight import export_preflight

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "ultralytics/cfg/models/26"
TASKS = {
    "segment": (SegmentationModel, "yolo26-seg.yaml", 3, "Segment26", 5),
    "pose": (PoseModel, "yolo26-pose.yaml", 1, "Pose26", 6),
    "obb": (OBBModel, "yolo26-obb.yaml", 3, "OBB26", 4),
    "semantic": (SemanticSegmentationModel, "yolo26-sem.yaml", 3, "SemanticSegment", 3),
}


def _build_task(task: str) -> nn.Module:
    model_cls, yaml_name, nc, _, _ = TASKS[task]
    model = model_cls(MODEL_ROOT / yaml_name, ch=3, nc=nc, verbose=False)
    model.task = task
    model.args = get_cfg(
        overrides={
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "pose": 12.0,
            "kobj": 1.0,
            "data": "cityscapes8.yaml" if task == "semantic" else "coco8.yaml",
        }
    )
    return model


def _training_batch(task: str) -> dict[str, torch.Tensor]:
    if task == "semantic":
        return {
            "img": torch.rand(2, 3, 64, 64),
            "semantic_mask": torch.randint(0, TASKS[task][2], (2, 64, 64)),
        }
    batch = {
        "img": torch.rand(2, 3, 64, 64),
        "batch_idx": torch.empty(0, dtype=torch.long),
        "cls": torch.empty(0, 1),
        "bboxes": torch.empty(0, 5 if task == "obb" else 4),
    }
    if task == "segment":
        batch["masks"] = torch.empty(0, 64, 64)
    elif task == "pose":
        batch["keypoints"] = torch.empty(0, 17, 3)
    return batch


def _tensor_outputs(value) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in _tensor_outputs(item)]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in _tensor_outputs(item)]
    return []


def _assert_inference_output(task: str, output) -> None:
    tensors = _tensor_outputs(output)
    assert tensors and all(torch.isfinite(tensor).all() for tensor in tensors)
    primary = tensors[0]
    expected = {
        "segment": {(1, 84, 38)},
        "pose": {(1, 84, 57)},
        "obb": {(1, 84, 7)},
        "semantic": {(1, 3, 8, 8), (1, 3, 64, 64), (1, 64, 64)},
    }
    assert tuple(primary.shape) in expected[task]


@pytest.mark.parametrize("task", TASKS)
def test_yolo26_specialized_task_train_inference_and_graph_contract(task):
    model = _build_task(task)
    metadata = graph_metadata(model)

    assert metadata["head_class"].endswith(TASKS[task][3])
    if task != "semantic":
        assert metadata["reg_max"] == 1
        assert metadata["end2end"] is True
        assert metadata["one2many"] is True
        assert metadata["one2one"] is True

    model.eval()
    with torch.no_grad():
        _assert_inference_output(task, model(torch.zeros(1, 3, 64, 64)))

    model.train()
    loss, items = model(_training_batch(task))
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    assert items.numel() == TASKS[task][4]
    loss.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


@pytest.mark.parametrize("task", TASKS)
def test_yolo26_specialized_task_molora_routed_smoke(task):
    model = _build_task(task)
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

    assert export_preflight(model, "onnx", strict=True)["supported"] is True
    model.train()
    loss, items = model(_training_batch(task))
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    loss.sum().backward()

    layer = next(module for module in model.modules() if isinstance(module, MoLoRALayer))
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        for parameter in layer.parameters()
        if parameter.requires_grad
    )


@pytest.mark.slow
@pytest.mark.parametrize("task", TASKS)
def test_yolo26_specialized_task_checkpoint_roundtrip(tmp_path, task):
    model = _build_task(task).eval()
    facade = Model.__new__(Model)
    nn.Module.__init__(facade)
    facade.model = model
    facade.ckpt = {}
    checkpoint = tmp_path / f"{task}.pt"

    facade.save(checkpoint)
    restored, payload = load_checkpoint(checkpoint, device="cpu")

    assert payload["mixture_checkpoint"]["graph"]["head_class"].endswith(TASKS[task][3])
    with torch.no_grad():
        _assert_inference_output(task, restored(torch.zeros(1, 3, 64, 64)))


@pytest.mark.slow
@pytest.mark.parametrize("fmt", ["torchscript", "onnx"])
@pytest.mark.parametrize("task", TASKS)
def test_yolo26_specialized_task_export_roundtrip(tmp_path, task, fmt):
    if fmt == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    model = _build_task(task).eval()
    model.pt_path = str(tmp_path / f"{task}.pt")

    report = export_preflight(model, fmt, strict=True)
    assert report["supported"] is True
    artifact = Exporter(
        overrides={"format": fmt, "imgsz": 64, "batch": 1, "device": "cpu", "simplify": False, "opset": 17}
    )(model=model)
    output = AutoBackend(artifact, device=torch.device("cpu"))(torch.zeros(1, 3, 64, 64))

    _assert_inference_output(task, output)


@pytest.mark.slow
def test_yolo26_segment_molora_onnx_roundtrip(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    model = _build_task("segment").eval()
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
    model.pt_path = str(tmp_path / "segment-molora.pt")
    exporter = Exporter(
        overrides={"format": "onnx", "imgsz": 64, "batch": 1, "device": "cpu", "simplify": False, "opset": 17}
    )

    artifact = exporter(model=model)
    output = AutoBackend(artifact, device=torch.device("cpu"))(torch.zeros(1, 3, 64, 64))

    _assert_inference_output("segment", output)
    assert exporter.export_preflight_report["supported"] is True
    assert "mixture_export_preflight" in exporter.metadata
