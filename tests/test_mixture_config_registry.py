"""Contracts for additive registration of preserved mixture and PEFT settings."""

from pathlib import Path

import yaml

from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/governance/mixture-preservation-manifest.yaml"


def test_preserved_config_keys_are_registered_or_upstream_precision_aliases():
    keys = {item["name"] for item in yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["config_keys"]}
    assert keys - set(DEFAULT_CFG_DICT) == {"half", "int8"}


def test_adapter_runtime_metadata_can_flow_into_validator_args():
    args = get_cfg(overrides={"lora_r": 2, "lora_backend": "fallback"})
    args.requested_lora_backend = "fallback"
    args.effective_lora_backend = "fallback"
    args.lora_target_audit = {"selected_count": 1}

    cloned = get_cfg(overrides=args)

    assert cloned.lora_r == 2
    assert cloned.effective_lora_backend == "fallback"
    assert cloned.lora_target_audit == {"selected_count": 1}


def test_custom_config_types_are_validated_additively():
    args = get_cfg(
        overrides={
            "lora_r": 4,
            "lora_dropout": 0.1,
            "lora_include_head": True,
            "lora_backend": "fallback",
            "moe_map_saturation_window_size": 3,
        }
    )

    assert args.lora_r == 4
    assert args.lora_dropout == 0.1
    assert args.lora_include_head is True
    assert args.lora_backend == "fallback"
    assert args.moe_map_saturation_window_size == 3


def test_lora_init_accepts_bool_or_named_initialization():
    assert get_cfg(overrides={"lora_init_lora_weights": True}).lora_init_lora_weights is True
    assert get_cfg(overrides={"lora_init_lora_weights": "gaussian"}).lora_init_lora_weights == "gaussian"
