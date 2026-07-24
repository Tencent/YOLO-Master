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
