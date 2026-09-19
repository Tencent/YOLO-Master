"""Configuration integrity tests for mixture and adapter options."""

from pathlib import Path

import yaml

from ultralytics.cfg import check_cfg, get_cfg
from ultralytics.nn.peft.molora import MoLoRAConfig


ROOT = Path(__file__).resolve().parents[1]


def _yaml_keys(path: Path):
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    return values


def test_default_yaml_has_unique_top_level_keys():
    text = (ROOT / "ultralytics/cfg/default.yaml").read_text(encoding="utf-8").splitlines()
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
    assert cfg.moa_regional_max_kv_tokens == 4096


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


def test_stal_defaults_parse_with_expected_types():
    cfg = get_cfg()
    assert isinstance(cfg.tal_topk, int)
    assert cfg.tal_topk == 10
    assert cfg.stal_mode == "fixed"
    assert cfg.stal_min_positive is False
    assert isinstance(cfg.stal_area_small, int)
    assert cfg.stal_area_small == 32**2
    assert isinstance(cfg.stal_area_medium, int)
    assert cfg.stal_area_medium == 96**2
    assert isinstance(cfg.stal_topk_small, int)
    assert isinstance(cfg.stal_expand, float)
    assert isinstance(cfg.tal_alpha, float)
    assert cfg.tal_alpha == 0.5
    assert isinstance(cfg.tal_beta, float)
    assert cfg.tal_beta == 6.0


def test_stal_mode_is_enum_checked():
    check_cfg({"stal_mode": "adaptive"})
    try:
        get_cfg(overrides={"stal_mode": "bogus"})
    except ValueError as exc:
        assert "stal_mode" in str(exc)
    else:
        raise AssertionError("stal_mode must reject unknown values")


def test_stal_area_ordering_is_validated():
    try:
        get_cfg(overrides={"stal_area_small": 9216, "stal_area_medium": 1024})
    except ValueError as exc:
        assert "stal_area_small" in str(exc)
    else:
        raise AssertionError("stal_area_small >= stal_area_medium must be rejected")


def test_stal_keys_are_type_checked():
    check_cfg({"stal_expand": 2.0})
    try:
        check_cfg({"stal_expand": "2.0"})
    except TypeError as exc:
        assert "stal_expand" in str(exc)
    else:
        raise AssertionError("stal_expand must reject string values")
    try:
        check_cfg({"stal_topk_small": "13"})
    except TypeError as exc:
        assert "stal_topk_small" in str(exc)
    else:
        raise AssertionError("stal_topk_small must reject string values")


def test_tal_alpha_beta_type_and_value_checked():
    check_cfg({"tal_alpha": 0.25, "tal_beta": 4.0})
    try:
        check_cfg({"tal_beta": "6.0"})
    except TypeError as exc:
        assert "tal_beta" in str(exc)
    else:
        raise AssertionError("tal_beta string must reject string values")
    try:
        get_cfg(overrides={"tal_alpha": 0.0})
    except ValueError as exc:
        assert "tal_alpha" in str(exc)
    else:
        raise AssertionError("tal_alpha=0 must be rejected")


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
