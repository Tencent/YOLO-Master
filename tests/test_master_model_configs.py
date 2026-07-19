"""Smoke tests for Master and YOLO26 model configuration compatibility."""

from pathlib import Path

import pytest
import torch

from ultralytics.nn.tasks import DetectionModel, YOLOEModel, YOLOESegModel
from ultralytics.nn.modules import C3k2


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "config",
    [
        ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-n.yaml",
        ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-n.yaml",
        ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-mot-n.yaml",
        ROOT / "ultralytics/cfg/models/26/yolo26-master-n.yaml",
        ROOT / "ultralytics/cfg/models/26/yolo26.yaml",
        ROOT / "ultralytics/cfg/models/26/yolo26-seg.yaml",
    ],
)
def test_model_config_parses_and_runs_minimal_forward(config):
    """Supported research configs must construct and produce finite tensors."""
    model = DetectionModel(str(config), ch=3, nc=80, verbose=False)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 64, 64))

    assert output is not None
    if isinstance(output, torch.Tensor):
        assert torch.isfinite(output).all()


def test_yolo26_c3k2_shortcut_form_keeps_conv_groups_integer():
    """The versioned C3k2 YAML form must not route a bool into Conv2d.groups."""
    model = DetectionModel(str(ROOT / "ultralytics/cfg/models/26/yolo26.yaml"), verbose=False)
    c3k2_layers = [layer for layer in model.model if isinstance(layer, C3k2)]

    assert c3k2_layers
    assert all(
        isinstance(module.groups, int)
        and not isinstance(module.groups, bool)
        and module.in_channels % module.groups == 0
        and module.out_channels % module.groups == 0
        for layer in c3k2_layers
        for module in layer.modules()
        if isinstance(module, torch.nn.Conv2d)
    )


@pytest.mark.parametrize(
    ("model_cls", "config"),
    [
        (YOLOEModel, ROOT / "ultralytics/cfg/models/26/yoloe-26.yaml"),
        (YOLOESegModel, ROOT / "ultralytics/cfg/models/26/yoloe-26-seg.yaml"),
    ],
)
def test_yoloe_26_configs_use_prompt_aware_model_path(model_cls, config):
    """YOLOE configs must be built through their prompt-aware task models."""
    model = model_cls(str(config), ch=3, nc=80, verbose=False)

    assert model.model[-1].__class__.__name__ in {"YOLOEDetect", "YOLOESegment", "YOLOESegment26"}
    assert tuple(model.stride.tolist()) == (8.0, 16.0, 32.0)
