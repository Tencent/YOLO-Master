"""Build and minimal-forward regression tests for Master configurations."""

from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules.moe import SharedExpertMoE


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative_path",
    [
        "ultralytics/cfg/models/26/yolo26-master-n.yaml",
        "ultralytics/cfg/models/26/yolo26.yaml",
        "ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-n.yaml",
        "ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-n.yaml",
        "ultralytics/cfg/models/master/v0_8/det/yolo-master-moe-mot-shared-n.yaml",
    ],
)
def test_master_config_builds_and_forwards(relative_path):
    model = YOLO(ROOT / relative_path).model
    with torch.no_grad():
        result = model(torch.zeros(1, 3, 64, 64))
    assert result is not None


def test_yolo26_master_uses_current_sppf_signature():
    text = (ROOT / "ultralytics/cfg/models/26/yolo26-master-n.yaml").read_text()
    assert "SPPF, [1024, 5]" in text


def test_shared_expert_config_reuses_pool_and_isolates_model_builds():
    config = ROOT / "ultralytics/cfg/models/master/v0_8/det/yolo-master-moe-mot-shared-n.yaml"

    first_model = YOLO(config).model
    first_blocks = [module for module in first_model.modules() if isinstance(module, SharedExpertMoE)]
    assert len(first_blocks) == 2
    assert first_blocks[0].fused_experts is first_blocks[1].fused_experts
    assert first_blocks[0].get_pool_info()["is_owner"] is True
    assert first_blocks[1].get_pool_info()["is_owner"] is False

    second_model = YOLO(config).model
    second_blocks = [module for module in second_model.modules() if isinstance(module, SharedExpertMoE)]
    assert second_blocks[0].fused_experts is second_blocks[1].fused_experts
    assert first_blocks[0].fused_experts is not second_blocks[0].fused_experts
