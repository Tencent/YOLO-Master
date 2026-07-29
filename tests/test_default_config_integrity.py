"""Configuration integrity tests for mixture and adapter options."""

from pathlib import Path

import yaml

from ultralytics.cfg import check_cfg, get_cfg
from ultralytics.nn.peft.molora import MoLoRAConfig


ROOT = Path(__file__).resolve().parents[1]


def _yaml_keys(path: Path):
    values = yaml.safe_load(path.read_text())
    return values


def test_default_yaml_has_unique_top_level_keys():
    text = (ROOT / "ultralytics/cfg/default.yaml").read_text().splitlines()
    keys = []
    for line in text:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and ":" in stripped and not line.startswith(" "):
            keys.append(stripped.split(":", 1)[0])
    assert len(keys) == len(set(keys))


def test_mixture_defaults_parse_with_expected_types():
    cfg = get_cfg()
    assert isinstance(cfg.latent_aux_gain, float)
    assert cfg.latent_aux_gain == 0.1
    assert isinstance(cfg.molora_top_k_warmup, (int, type(None)))
    assert isinstance(cfg.molora_domain_experts, (dict, type(None)))
    assert isinstance(cfg.molora_freeze_experts, (list, type(None)))
    assert cfg.molora_capacity_factor == 0.0
    assert cfg.mot_sparse_train_warmup_steps == 0
    assert cfg.mot_scene_hidden_dim is None
    assert cfg.mot_scene_inference_mode == "dynamic"


def test_new_mixture_float_key_is_type_checked():
    check_cfg({"latent_aux_gain": 0.25})
    try:
        check_cfg({"latent_aux_gain": "0.25"})
    except TypeError as exc:
        assert "latent_aux_gain" in str(exc)
    else:
        raise AssertionError("latent_aux_gain must reject string values")


def test_scene_inference_mode_is_type_checked():
    check_cfg({"mot_scene_inference_mode": "bypass"})
    try:
        check_cfg({"mot_scene_inference_mode": 1})
    except TypeError as exc:
        assert "mot_scene_inference_mode" in str(exc)
    else:
        raise AssertionError("mot_scene_inference_mode must reject non-string values")


def test_molora_none_and_empty_optional_values_have_stable_semantics():
    class Args:
        molora_num_experts = 2
        molora_top_k = 1
        molora_r = 2
        molora_alpha = 4
        molora_domain_experts = None
        molora_freeze_experts = None

    cfg = MoLoRAConfig.from_args(Args())
    assert cfg.domain_experts is None
    assert cfg.freeze_experts is None

    cfg = MoLoRAConfig.from_args(Args(), molora_domain_experts={}, molora_freeze_experts=[])
    assert cfg.domain_experts == {}
    assert cfg.freeze_experts == []
