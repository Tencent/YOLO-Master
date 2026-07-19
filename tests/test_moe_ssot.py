import io
from pathlib import Path

import torch
from scripts.check_moe_ssot import find_duplicate_classes
from ultralytics.utils.patches import torch_load


def test_moe_public_classes_have_one_implementation_source():
    assert find_duplicate_classes() == {}


def test_legacy_modules_reexport_canonical_class_objects():
    from ultralytics.nn.modules.moe import gated, modules
    from ultralytics.nn.modules.moe import base, blocks_advanced, experts_advanced, hybrid, integration, routers_advanced

    aliases = {
        base.UltraOptimizedMoE: modules.UltraOptimizedMoE,
        base.OptimizedMOEImproved: modules.OptimizedMOEImproved,
        blocks_advanced.AdaptiveGateMoE: gated.AdaptiveGateMoE,
        routers_advanced.DualStreamGateRouterV2: gated.DualStreamGateRouterV2,
        experts_advanced.FusedExpertGroup: gated.FusedExpertGroup,
        hybrid.OptimalHybridGateMoE: gated.OptimalHybridGateMoE,
        integration.UltimateOptimizedMoE: modules.UltimateOptimizedMoE,
    }
    assert all(alias is canonical for alias, canonical in aliases.items())


def test_legacy_modules_preserve_historical_class_names():
    from ultralytics.nn.modules.moe import base, blocks_advanced, experts_advanced, hybrid, integration, routers_advanced

    expected = {
        base: {"UltraOptimizedMoE", "AdaptiveCapacityMoE", "ES_MOE", "OptimizedMOE", "OptimizedMOEImproved", "ABlockMoE", "A2C2fMoE"},
        blocks_advanced: {"AdaptiveGateMoE", "HyperSplitMoE", "HyperFusedMoE"},
        experts_advanced: {"FusedExpertGroup", "LowRankFusedExpertGroup"},
        routers_advanced: {"DualStreamGateRouter", "DualStreamGateRouterV2", "ZeroCostRouter"},
        hybrid: {
            "VisualDetailGate", "PyramidContextMixer", "FusedAdaptiveGateMoE", "HybridAdaptiveGateMoE",
            "HybridAdaptiveGateMoEv2", "LowRankHybridAdaptiveGateMoE", "RefinedLowRankHybridAdaptiveGateMoE",
            "DetailAwareLowRankHybridAdaptiveGateMoE", "ContextRefinedLowRankHybridAdaptiveGateMoE",
            "VisualEnhancedAdaptiveGateMoE", "AdaptiveBalanceController", "OptimalHybridGateMoE",
        },
        integration: {
            "MultiHeadRouterV3", "DiversifiedExpertGroup", "CrossPathGate", "MultiHeadRouterMoE",
            "DiversifiedExpertMoE", "GatedFusionMoE", "UltraLightRouter", "MatMulFusedExperts",
            "HyperUltimateMoE", "UltimateOptimizedMoE",
        },
    }
    assert all(all(hasattr(module, name) for name in names) for module, names in expected.items())


def test_base_debug_switch_still_controls_canonical_ablock(monkeypatch):
    import torch
    from torch import nn
    from ultralytics.nn.modules.moe import base

    block = object.__new__(base.ABlockMoE)
    nn.Module.__init__(block)
    block.attn = nn.Identity()
    block.mlp = nn.Identity()
    monkeypatch.setattr(base, "_MOE_FINITE_DIAGNOSTICS", True)
    block.attn.forward = lambda x: torch.full_like(x, float("nan"))

    try:
        block(torch.ones(1, 1, 1, 1))
    except RuntimeError as exc:
        assert "attention output" in str(exc)
    else:
        raise AssertionError("legacy debug switch did not enable canonical ABlockMoE diagnostics")


def test_compatibility_alias_survives_torch_serialization():
    from ultralytics.nn.modules.moe import base, modules

    model = base.UltraOptimizedMoE(8, 8, num_experts=2, top_k=1)
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    loaded = torch_load(buffer, map_location="cpu", weights_only=False)

    assert type(loaded) is modules.UltraOptimizedMoE
    assert loaded.state_dict().keys() == model.state_dict().keys()


def test_legacy_modules_contain_no_public_class_definitions():
    moe_root = Path(__file__).parents[1] / "ultralytics/nn/modules/moe"
    for name in ("base.py", "blocks_advanced.py", "experts_advanced.py", "hybrid.py", "integration.py", "routers_advanced.py"):
        source = (moe_root / name).read_text(encoding="utf-8")
        assert "\nclass " not in source
