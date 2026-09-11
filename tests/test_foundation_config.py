"""Configuration contracts for the opt-in Foundation Teacher boundary."""

import pytest

import ultralytics.cfg as cfg_module
from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg


def _enabled_overrides(**overrides):
    values = {
        "foundation_enabled": True,
        "foundation_teacher": "dinov3",
        "foundation_model": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "foundation_loss_weight": 1.0,
    }
    values.update(overrides)
    return values


def test_foundation_defaults_are_opt_in_and_preserve_existing_distillation_defaults():
    args = get_cfg()

    assert args.foundation_enabled is False
    assert args.foundation_teacher == "none"
    assert args.foundation_backend == "transformers"
    assert args.foundation_target_levels == ["p4"]
    assert args.foundation_multiscale is False
    assert args.foundation_align_dim == 256
    assert args.foundation_relation_mode == "sampled"
    assert args.foundation_relation_samples == 256
    assert args.foundation_loss == "relational"
    assert args.foundation_loss_weight == 0.0
    assert args.foundation_foreground_weighting is False
    assert args.foundation_foreground_weight == 1.5
    assert args.foundation_boundary_weight == 1.0
    assert args.foundation_background_weight == 0.25
    assert args.foundation_router_temperature == 1.0
    assert args.distill_model is DEFAULT_CFG_DICT["distill_model"] is None


def test_disabled_foundation_is_a_noop_even_when_optional_dependency_is_missing(monkeypatch):
    monkeypatch.setattr(cfg_module, "_foundation_transformers_available", lambda: False)

    args = get_cfg(
        overrides={
            "foundation_model": "unused-model",
            "foundation_teacher_dtype": "fp16",
            "foundation_teacher_device": "cpu",
        }
    )

    assert args.foundation_enabled is False
    assert args.foundation_model == "unused-model"


def test_invalid_foundation_teacher_is_rejected():
    with pytest.raises(ValueError, match="foundation_teacher"):
        get_cfg(overrides={"foundation_teacher": "unknown"})


def test_siglip2_foundation_teacher_is_valid():
    args = get_cfg(
        overrides={
            "foundation_enabled": True,
            "foundation_teacher": "siglip2",
            "foundation_model": "google/siglip2-base-patch16-512",
            "foundation_loss_weight": 0.0,
        }
    )
    assert args.foundation_teacher == "siglip2"


def test_enabled_without_teacher_or_cache_is_rejected():
    with pytest.raises(ValueError, match="requires 'foundation_cache_dir'"):
        get_cfg(overrides={"foundation_enabled": True})


def test_negative_foundation_loss_weights_are_rejected():
    with pytest.raises(ValueError, match="foundation_loss_weight"):
        get_cfg(overrides={"foundation_loss_weight": -0.1})


def test_foundation_and_yolo_distillation_are_mutually_exclusive():
    with pytest.raises(ValueError, match="distill_model"):
        get_cfg(overrides=_enabled_overrides(distill_model="yolo11n.pt"))


def test_foundation_and_compile_are_mutually_exclusive():
    with pytest.raises(ValueError, match="compile"):
        get_cfg(overrides=_enabled_overrides(compile=True))


def test_positive_foundation_weight_requires_a_model_reference():
    with pytest.raises(ValueError, match="foundation_model.*foundation_weights"):
        get_cfg(
            overrides={
                "foundation_enabled": True,
                "foundation_teacher": "dinov3",
                "foundation_loss_weight": 1.0,
            }
        )


def test_positive_transformers_foundation_weight_checks_optional_dependency(monkeypatch):
    monkeypatch.setattr(cfg_module, "_foundation_transformers_available", lambda: False)

    with pytest.raises(ImportError, match="transformers>=4.56.0"):
        get_cfg(overrides=_enabled_overrides())


def test_zero_foundation_weight_skips_optional_dependency_check(monkeypatch):
    monkeypatch.setattr(cfg_module, "_foundation_transformers_available", lambda: False)

    args = get_cfg(overrides=_enabled_overrides(foundation_loss_weight=0.0))
    assert args.foundation_enabled is True
    assert args.foundation_loss_weight == 0.0


def test_router_only_foundation_requires_teacher_and_optional_backend(monkeypatch):
    monkeypatch.setattr(cfg_module, "_foundation_transformers_available", lambda: False)
    with pytest.raises(ImportError, match="transformers>=4.56.0"):
        get_cfg(
            overrides={
                "foundation_enabled": True,
                "foundation_teacher": "dinov3",
                "foundation_model": "facebook/dinov3-vits16-pretrain-lvd1689m",
                "foundation_loss_weight": 0.0,
                "foundation_router_distill": True,
                "foundation_router_loss_weight": 0.1,
            }
        )


def test_foundation_is_training_only():
    with pytest.raises(ValueError, match="training-only"):
        get_cfg(overrides=_enabled_overrides(mode="val"))


def test_foundation_target_levels_are_validated():
    with pytest.raises(ValueError, match="unsupported levels"):
        get_cfg(overrides={"foundation_target_levels": ["p6"]})
    with pytest.raises(ValueError, match="duplicate"):
        get_cfg(overrides={"foundation_target_levels": ["p4", "p4"]})


def test_multiscale_requires_multiple_target_levels():
    with pytest.raises(ValueError, match="at least two target levels"):
        get_cfg(overrides={"foundation_multiscale": True, "foundation_target_levels": ["p4"]})


@pytest.mark.parametrize("value", [0, -1, True, "256"])
def test_foundation_align_dim_must_be_positive_integer(value):
    with pytest.raises((TypeError, ValueError), match="foundation_align_dim"):
        get_cfg(overrides={"foundation_align_dim": value})


def test_foundation_relation_mode_is_validated():
    with pytest.raises(ValueError, match="foundation_relation_mode"):
        get_cfg(overrides={"foundation_relation_mode": "invalid"})


def test_foundation_router_temperature_is_positive():
    with pytest.raises(ValueError, match="foundation_router_temperature"):
        get_cfg(overrides={"foundation_router_temperature": 0})


@pytest.mark.parametrize(
    "key", ["foundation_foreground_weight", "foundation_boundary_weight", "foundation_background_weight"]
)
def test_foreground_weights_must_be_nonnegative(key):
    with pytest.raises(ValueError, match=key):
        get_cfg(overrides={key: -0.1})


@pytest.mark.parametrize("value", [0, -1, True, "256"])
def test_foundation_relation_samples_must_be_positive_integer(value):
    with pytest.raises((TypeError, ValueError), match="foundation_relation_samples"):
        get_cfg(overrides={"foundation_relation_samples": value})
