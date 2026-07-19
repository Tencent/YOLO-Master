"""Tests for additive mixture YAML registration on the v8.4.101 parser."""

from pathlib import Path

import torch

from ultralytics.nn.mixture_registry import MIXTURE_MODULES, MIXTURE_REPEAT_MODULES
from ultralytics.nn.modules import C2fMoA, C2fMoT, Detect
from ultralytics.nn.tasks import DetectionModel

ROOT = Path(__file__).resolve().parents[1]


def test_registry_covers_preserved_yaml_names():
    assert set(MIXTURE_MODULES) == {
        "A2C2fMoE",
        "AdaptiveGateMoE",
        "C2fMoA",
        "C2fMoT",
        "DetailAwareLowRankHybridAdaptiveGateMoE",
        "DiversifiedExpertMoE",
        "DyC2f",
        "DyMoEBlock",
        "ES_MOE",
        "FusedAdaptiveGateMoE",
        "GatedFusionMoE",
        "HybridAdaptiveGateMoE",
        "HybridAdaptiveGateMoEv2",
        "LowRankHybridAdaptiveGateMoE",
        "ModularRouterExpertMoE",
        "MultiHeadRouterMoE",
        "OptimalHybridGateMoE",
        "RefinedLowRankHybridAdaptiveGateMoE",
        "UltimateOptimizedMoE",
        "UltraOptimizedMoE",
        "VisualEnhancedAdaptiveGateMoE",
    }
    assert {module.__name__ for module in MIXTURE_REPEAT_MODULES} == {"A2C2fMoE", "C2fMoA", "C2fMoT", "DyC2f"}


def test_official_yolo26_path_remains_native():
    model = DetectionModel(ROOT / "ultralytics/cfg/models/26/yolo26.yaml", ch=3, nc=80, verbose=False)
    head = model.model[-1]
    assert isinstance(head, Detect)
    assert head.reg_max == 1
    assert head.end2end is True
    assert hasattr(head, "one2many") and hasattr(head, "one2one")


def test_moa_and_mot_configs_build_and_run():
    configs = (
        ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-n.yaml",
        ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-n.yaml",
        ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-mot-n.yaml",
    )
    for config in configs:
        model = DetectionModel(config, ch=3, nc=80, verbose=False).eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 64, 64))
        assert output is not None
    assert any(isinstance(module, C2fMoA) for module in DetectionModel(configs[0], verbose=False).modules())
    assert any(isinstance(module, C2fMoT) for module in DetectionModel(configs[1], verbose=False).modules())
