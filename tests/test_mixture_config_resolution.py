"""Tests for unified mixture configuration resolution and audit semantics."""

from types import SimpleNamespace

from ultralytics.nn.modules import C2fMoA, C2fMoT
from ultralytics.nn.modules.moe.config import (
    apply_mixture_config,
    annotate_mixture_yaml_config,
    resolve_mixture_config,
)
from ultralytics.nn.modules.moe.modules import OptimizedMOEImproved
from ultralytics.nn.tasks import DetectionModel
from ultralytics.engine.trainer import BaseTrainer


def _args(**overrides):
    defaults = dict(
        moe_balance_loss=1.7,
        moe_router_z_loss=0.8,
        moe_noise_std=0.2,
        moe_temperature=0.6,
        moe_weight_threshold=0.03,
        moa_local_window_size=9,
        moa_aux_loss_coeff=0.2,
        moa_sparse_inference=True,
        moa_sparse_inference_threshold=0.15,
        mot_balance_loss=0.4,
        mot_router_z_loss=0.5,
        mot_sparse_train=True,
        mot_sparse_train_warmup_steps=3,
        mot_scene_aware_router=True,
        mot_scene_hidden_dim=5,
        mot_scene_consistency=0.03,
        mot_scene_inference_mode="bypass",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_yaml_explicit_values_override_cli_and_are_inherited_by_children():
    """A wrapper's YAML temperature must beat the trainer-wide override."""
    model = C2fMoA(64, 64, n=1, num_heads=6, temperature=1.0, local_window_size=7)
    annotate_mixture_yaml_config(model, "C2fMoA", [64, 6, 2.0, 1.0, True])

    resolved = resolve_mixture_config(_args(moa_local_window_size=11), model)
    audit = [item for item in resolved.audit if item["kind"] == "moa"]

    assert audit
    assert all(item["values"]["temperature"] == 1.0 for item in audit)
    assert audit[0]["sources"]["temperature"] == "yaml"
    assert all(item["values"]["local_window_size"] == 11 for item in audit)

    apply_mixture_config(model, resolved)
    assert model.m[0].router.temperature == 1.0
    assert model.m[0].local_head.window_size == 11
    assert model.sparse_inference is True
    assert model.sparse_inference_threshold == 0.15
    assert model.m[0].sparse_inference is True
    assert model.m[0].sparse_inference_threshold == 0.15


def test_mot_cli_values_apply_to_wrapper_and_nested_blocks():
    """CLI values should apply consistently to C2fMoT and every child block."""
    model = C2fMoT(64, 64, n=2)
    resolved = resolve_mixture_config(_args(), model)
    apply_mixture_config(model, resolved)

    assert model.last_aux_loss.requires_grad is False
    assert all(block.balance_loss_coeff == 0.4 for block in model.m)
    assert all(block.router_z_loss_coeff == 0.5 for block in model.m)
    assert all(block.sparse_train is True for block in model.m)
    assert all(block.sparse_train_warmup_steps == 3 for block in model.m)
    assert all(block.router.scene_aware is True for block in model.m)
    assert all(block.router.scene_hidden_dim == 5 for block in model.m)
    assert all(block.scene_consistency_coeff == 0.03 for block in model.m)
    assert all(block.router.scene_inference_mode == "bypass" for block in model.m)


def test_moe_module_receives_cli_values_and_audit_records_sources():
    """Core MoE runtime attributes use the same resolver and audit schema."""
    model = OptimizedMOEImproved(32, 32, num_experts=4, top_k=2)
    resolved = resolve_mixture_config(_args(), model)
    apply_mixture_config(model, resolved)

    assert model.balance_loss_coeff == 1.7
    assert model.router_z_loss_coeff == 0.8
    assert model.routing.noise_std == 0.2
    assert not hasattr(model.routing, "temperature")
    assert not hasattr(model, "weight_threshold")
    assert model.mixture_config_audit == resolved.audit


def test_yaml_only_cli_only_and_resume_like_resolution_are_deterministic():
    """Repeated resolution from equivalent args produces identical results."""
    yaml_model = DetectionModel(
        "ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-mot-n.yaml", verbose=False
    )
    args = _args()
    first = resolve_mixture_config(args, yaml_model).to_dict()
    resumed = resolve_mixture_config(SimpleNamespace(**vars(args)), yaml_model).to_dict()

    assert first == resumed
    assert first["audit"]


def test_resolver_has_safe_defaults_without_args_or_model():
    """No CLI/model input must still return a complete, serializable config."""
    resolved = resolve_mixture_config()

    assert resolved.values["moe"]["temperature"] == 1.0
    assert resolved.values["moe"]["router_z_loss_coeff"] == 0.1
    assert resolved.values["moa"]["local_window_size"] == 7
    assert resolved.values["moa"]["sparse_inference"] is False
    assert resolved.values["moa"]["sparse_inference_threshold"] == 0.02
    assert resolved.values["mot"]["sparse_train"] is False
    assert resolved.values["mot"]["sparse_train_warmup_steps"] == 0
    assert resolved.values["mot"]["scene_aware_router"] is False
    assert resolved.values["mot"]["scene_inference_mode"] == "dynamic"
    assert resolved.values["molora"]["router_hidden_dim"] is None


def test_base_trainer_resolves_and_attaches_runtime_audit():
    """The trainer entry point uses the canonical resolver exactly once."""
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.args = _args()
    trainer.model = C2fMoT(64, 64, n=1)

    trainer._resolve_mixture_runtime_config()

    assert trainer.mixture_config.audit
    assert trainer.model.mixture_config_audit == trainer.mixture_config.audit
    assert trainer.model.m[0].balance_loss_coeff == 0.4
    assert trainer.model.m[0].router_z_loss_coeff == 0.5
