import math
from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from ultralytics.utils.lora import LoRAConfig
from ultralytics.utils.lora.fallback import (
    FewShotLoRAConv,
    ManualLoRAConv,
    _collect_fallback_adapter_state,
    _load_fallback_adapter_state,
    _merge_fallback_modules,
    apply_manual_lora,
)


def _manual_layer(*, use_rslora: bool) -> ManualLoRAConv:
    return ManualLoRAConv(nn.Conv2d(4, 4, 1), r=8, alpha=16, use_rslora=use_rslora)


def test_fallback_honors_requested_rslora_scaling():
    assert _manual_layer(use_rslora=True).scaling == pytest.approx(16 / math.sqrt(8))
    assert _manual_layer(use_rslora=False).scaling == pytest.approx(16 / 8)
    assert FewShotLoRAConv(nn.Conv2d(4, 4, 1), r=8, alpha=16, use_rslora=True).scaling == pytest.approx(
        16 / math.sqrt(8)
    )


def test_fallback_adapter_round_trip_preserves_effective_rslora(tmp_path):
    base = nn.Sequential(nn.Conv2d(4, 4, 1))
    restored_base = deepcopy(base)
    source = apply_manual_lora(
        base,
        LoRAConfig(
            r=8,
            alpha=16,
            dropout=0.0,
            backend="fallback",
            target_modules=["0"],
            skip_stem=False,
            use_rslora=True,
        ),
    )
    source[0].lora_B.data.normal_()
    source.eval()
    assert source.lora_runtime_metadata["requested_use_rslora"] is True
    assert source.lora_runtime_metadata["effective_use_rslora"] is True
    sample = torch.randn(2, 4, 3, 3)
    expected = source(sample)
    saved = _collect_fallback_adapter_state(source)
    assert saved["modules"]["0"]["use_rslora"] is True
    torch.save(saved, tmp_path / "fallback_adapter.pt")
    payload = {"backend": "fallback", "weight_file": "fallback_adapter.pt"}

    restored = _load_fallback_adapter_state(restored_base, tmp_path, payload)
    restored.eval()

    assert restored[0].use_rslora is True
    assert restored[0].scaling == pytest.approx(16 / math.sqrt(8))
    torch.testing.assert_close(restored(sample), expected)
    assert _merge_fallback_modules(restored) == 1
    torch.testing.assert_close(restored(sample), expected)


def test_few_shot_round_trip_preserves_dropconnect_and_adaptive_rank(tmp_path):
    source = nn.Sequential(
        FewShotLoRAConv(
            nn.Conv2d(4, 4, 1),
            r=8,
            alpha=16,
            dropconnect=0.27,
            adaptive_rank=False,
        )
    )
    saved = _collect_fallback_adapter_state(source)
    torch.save(saved, tmp_path / "fallback_adapter.pt")
    payload = {"backend": "fallback", "weight_file": "fallback_adapter.pt"}

    restored = _load_fallback_adapter_state(nn.Sequential(nn.Conv2d(4, 4, 1)), tmp_path, payload)

    assert isinstance(restored[0], FewShotLoRAConv)
    assert restored[0].dropconnect_rate == pytest.approx(0.27)
    assert restored[0].adaptive_rank is False
    assert not hasattr(restored[0], "rank_mask")


def test_legacy_fallback_adapter_without_scaling_mode_keeps_lora_scaling(tmp_path):
    source = nn.Sequential(_manual_layer(use_rslora=False))
    saved = _collect_fallback_adapter_state(source)
    for module_config in saved["modules"].values():
        module_config.pop("use_rslora", None)
    torch.save(saved, tmp_path / "fallback_adapter.pt")
    payload = {"backend": "fallback", "weight_file": "fallback_adapter.pt"}

    restored = _load_fallback_adapter_state(nn.Sequential(nn.Conv2d(4, 4, 1)), tmp_path, payload)

    assert restored[0].use_rslora is False
    assert restored[0].scaling == pytest.approx(16 / 8)


def test_layers_outside_explicit_targets_are_frozen():
    model = apply_manual_lora(
        nn.Sequential(nn.Conv2d(4, 4, 1), nn.Conv2d(4, 4, 1)),
        LoRAConfig(r=2, alpha=4, backend="fallback", target_modules=["1"], skip_stem=False),
    )

    assert isinstance(model[1], ManualLoRAConv)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {"1.lora_A", "1.lora_B"}


def test_skipped_depthwise_and_head_like_layers_are_frozen():
    class _TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(4, 4, 1)
            self.depthwise = nn.Conv2d(4, 4, 3, padding=1, groups=4)
            self.dfl = nn.Sequential(nn.Conv2d(4, 4, 1))

    model = apply_manual_lora(_TinyModel(), LoRAConfig(r=2, alpha=4, backend="fallback", skip_stem=False))

    assert isinstance(model.stem, ManualLoRAConv)
    assert not any(parameter.requires_grad for parameter in model.depthwise.parameters())
    assert not any(parameter.requires_grad for parameter in model.dfl.parameters())
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {"stem.lora_A", "stem.lora_B"}
