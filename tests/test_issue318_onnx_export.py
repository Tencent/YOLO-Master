"""End-to-end regression for trained ES-MoE checkpoint export in issue #318."""

from pathlib import Path

import numpy as np
import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules.moe.modules import ES_MOE


def _flatten_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _flatten_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_tensors(item)


def test_trained_esmoe_checkpoint_opset12_export_matches_onnxruntime(tmp_path):
    """A trained checkpoint must export at opset 12 and preserve dense-export numerics."""
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    config = Path(__file__).resolve().parents[1] / "ultralytics/cfg/models/master/v0/det/yolo-master-esmoe-n-coco.yaml"
    checkpoint = tmp_path / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)

    # Exercise the same train -> save -> reload path as a fine-tuned user checkpoint
    # without requiring a dataset download in CI.
    model = YOLO(str(config))
    model.model.train()
    parameter = next(model.model.parameters())
    before = parameter.detach().clone()
    optimizer = torch.optim.SGD(model.model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    outputs = model.model(torch.randn(1, 3, 64, 64))
    loss = sum(tensor.float().sum() for tensor in _flatten_tensors(outputs) if tensor.requires_grad)
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, parameter.detach()), "optimizer did not update model weights"

    model.ckpt = {}
    model.save(checkpoint)
    reloaded = YOLO(str(checkpoint))
    assert sum(isinstance(module, ES_MOE) for module in reloaded.model.modules()) == 4

    exported = reloaded.export(format="onnx", opset=12, imgsz=640, dynamic=False, simplify=False)
    onnx_path = Path(exported)
    proto = onnx.load(onnx_path)
    onnx.checker.check_model(proto)
    opsets = {item.domain or "ai.onnx": item.version for item in proto.opset_import}
    assert opsets["ai.onnx"] == 12

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    assert input_info.shape == [1, 3, 640, 640]
    x = np.random.default_rng(318).standard_normal((1, 3, 640, 640), dtype=np.float32)
    onnx_output = session.run(None, {input_info.name: x})[0]
    assert np.isfinite(onnx_output).all()

    # Full-model ONNX export uses ES_MOE's dense fallback. Compare against the
    # same dense semantics rather than the intentionally different sparse path.
    reference = YOLO(str(checkpoint)).model.eval()
    for module in reference.modules():
        if isinstance(module, ES_MOE):
            module.use_sparse_inference = False
    with torch.no_grad():
        torch_output = reference(torch.from_numpy(x))
    if isinstance(torch_output, (list, tuple)):
        torch_output = torch_output[0]
    torch_output = torch_output.numpy()

    assert torch_output.shape == onnx_output.shape
    np.testing.assert_allclose(torch_output, onnx_output, atol=2e-3, rtol=2e-3)
