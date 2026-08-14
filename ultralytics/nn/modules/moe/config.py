"""Unified configuration resolution for MoE, MoA, MoT, and MoLoRA.

The resolver is deliberately small and side-effect free. Model construction
can annotate values that came from YAML; training-time values then fill only
fields that were not explicitly set by the model definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn


MIXTURE_DEFAULTS: dict[str, dict[str, Any]] = {
    "moe": {
        "balance_loss_coeff": 1.0,
        "router_z_loss_coeff": 0.1,
        "noise_std": 0.5,
        "temperature": 1.0,
        "weight_threshold": 0.01,
        "aux_gain": 1.0,
        "aux_budget": 3.0,
    },
    "moa": {
        "temperature": 1.0,
        "local_window_size": 7,
        "regional_max_kv_tokens": 4096,
        "aux_loss_coeff": 0.01,
        "aux_gain": 1.0,
        "aux_budget": 3.0,
        "sparse_inference": False,
        "sparse_inference_threshold": 0.02,
    },
    "mot": {
        "balance_loss_coeff": 0.01,
        "router_z_loss_coeff": 0.01,
        "temperature": 1.0,
        "sparse_train": False,
        "sparse_train_warmup_steps": 0,
        "scene_aware_router": False,
        "scene_hidden_dim": None,
        "scene_consistency_coeff": 0.0,
        "scene_inference_mode": "dynamic",
        "local_attn_window": 0,
        "aux_gain": 1.0,
        "aux_budget": 3.0,
    },
    "latent": {
        "aux_gain": 0.1,
        "aux_budget": 3.0,
        "inference_top_k": None,
    },
    "molora": {
        "balance_loss_coef": 0.01,
        "z_loss_coef": 0.001,
        "diversity_loss_coef": 0.0,
        "router_hidden_dim": None,
        "top_k_warmup": None,
        "domain_experts": None,
        "freeze_experts": None,
    },
}


CLI_FIELDS: dict[str, dict[str, str]] = {
    "moe": {
        "balance_loss_coeff": "moe_balance_loss",
        "router_z_loss_coeff": "moe_router_z_loss",
        "noise_std": "moe_noise_std",
        "temperature": "moe_temperature",
        "weight_threshold": "moe_weight_threshold",
        "aux_gain": "moe_aux_gain",
        "aux_budget": "mixture_aux_budget",
    },
    "moa": {
        "temperature": "moa_temperature",
        "local_window_size": "moa_local_window_size",
        "regional_max_kv_tokens": "moa_regional_max_kv_tokens",
        "aux_loss_coeff": "moa_aux_loss_coeff",
        "aux_gain": "moa_aux_gain",
        "aux_budget": "mixture_aux_budget",
        "sparse_inference": "moa_sparse_inference",
        "sparse_inference_threshold": "moa_sparse_inference_threshold",
    },
    "mot": {
        "balance_loss_coeff": "mot_balance_loss",
        "router_z_loss_coeff": "mot_router_z_loss",
        "temperature": "mot_temperature",
        "sparse_train": "mot_sparse_train",
        "sparse_train_warmup_steps": "mot_sparse_train_warmup_steps",
        "scene_aware_router": "mot_scene_aware_router",
        "scene_hidden_dim": "mot_scene_hidden_dim",
        "scene_consistency_coeff": "mot_scene_consistency",
        "scene_inference_mode": "mot_scene_inference_mode",
        "local_attn_window": "mot_local_attn_window",
        "aux_gain": "mot_aux_gain",
        "aux_budget": "mixture_aux_budget",
    },
    "latent": {
        "aux_gain": "latent_aux_gain",
        "aux_budget": "mixture_aux_budget",
        "inference_top_k": "latent_inference_top_k",
    },
    "molora": {
        "balance_loss_coef": "molora_balance_loss",
        "z_loss_coef": "molora_router_z_loss",
        "diversity_loss_coef": "molora_diversity_loss",
        "router_hidden_dim": "molora_router_hidden_dim",
        "top_k_warmup": "molora_top_k_warmup",
        "domain_experts": "molora_domain_experts",
        "freeze_experts": "molora_freeze_experts",
    },
}


@dataclass
class ResolvedMixtureConfig:
    """Resolved values plus a per-module audit trail."""

    values: dict[str, dict[str, Any]]
    audit: list[dict[str, Any]] = field(default_factory=list)

    def for_kind(self, kind: str) -> dict[str, Any]:
        return self.values[kind]

    def to_dict(self) -> dict[str, Any]:
        return {"values": self.values, "audit": self.audit}


def annotate_mixture_yaml_config(module: nn.Module, module_name: str, yaml_args: list[Any]) -> None:
    """Record explicit mixture constructor values before CLI injection.

    ``parse_model`` calls this with the unmodified YAML argument list. The
    metadata is non-persistent and therefore does not affect checkpoints.
    """

    name = module_name.rsplit(".", 1)[-1]
    explicit: dict[str, Any] = {}
    kind = None
    if name == "C2fMoA":
        kind = "moa"
        for index, key in (
            (3, "temperature"),
            (6, "aux_loss_coeff"),
            (7, "local_window_size"),
            (9, "regional_max_kv_tokens"),
            (10, "sparse_inference"),
            (11, "sparse_inference_threshold"),
        ):
            if len(yaml_args) > index:
                explicit[key] = yaml_args[index]
    elif name == "C2fMoT":
        kind = "mot"
        for index, key in (
            (6, "temperature"),
            (7, "balance_loss_coeff"),
            (9, "sparse_train"),
            (10, "scene_aware_router"),
            (11, "scene_hidden_dim"),
            (12, "scene_consistency_coeff"),
            (13, "sparse_train_warmup_steps"),
            (14, "scene_inference_mode"),
            (15, "local_attn_window"),
        ):
            if len(yaml_args) > index:
                explicit[key] = yaml_args[index]
    elif name == "MoABlock":
        kind = "moa"
        for index, key in (
            (3, "temperature"),
            (6, "aux_loss_coeff"),
            (8, "local_window_size"),
            (10, "regional_max_kv_tokens"),
            (11, "sparse_inference"),
            (12, "sparse_inference_threshold"),
        ):
            if len(yaml_args) > index:
                explicit[key] = yaml_args[index]
    elif name == "MoTBlock":
        kind = "mot"
        for index, key in (
            (5, "temperature"),
            (6, "balance_loss_coeff"),
            (7, "router_z_loss_coeff"),
            (14, "sparse_train"),
            (15, "scene_aware_router"),
            (16, "scene_hidden_dim"),
            (17, "scene_consistency_coeff"),
            (18, "sparse_train_warmup_steps"),
            (19, "scene_inference_mode"),
            (20, "local_attn_window"),
        ):
            if len(yaml_args) > index:
                explicit[key] = yaml_args[index]
    elif name == "A2C2fMoE":
        # Architecture values are recorded for audit, but are not dynamically
        # changed by the runtime resolver because changing them changes shape.
        kind = "moe"
        for index, key in ((8, "num_experts"), (9, "top_k")):
            if len(yaml_args) > index:
                explicit[key] = yaml_args[index]

    if kind and explicit:
        setattr(module, "_mixture_config_kind", kind)
        setattr(module, "_mixture_config_explicit", explicit)


def _module_kind(module: nn.Module) -> str | None:
    """Return the runtime mixture kind without importing task modules."""

    name = module.__class__.__name__
    if name in {"MoABlock", "C2fMoA", "NeckMoAFusion"}:
        return "moa"
    if name in {"MoTBlock", "C2fMoT"}:
        return "mot"
    if name in {"LatentMixture", "MultiScaleLatentMixture"}:
        return "latent"
    if name in {"MoLoRALayer", "MoLoRAMoEAwareLayer"}:
        return "molora"
    try:
        from .utils import is_core_moe_block

        if is_core_moe_block(module):
            return "moe"
    except (ImportError, AttributeError):
        pass
    return getattr(module, "_mixture_config_kind", None)


def _cli_value(args: Any, attr: str) -> tuple[Any, bool]:
    if args is None or not hasattr(args, attr):
        return None, False
    value = getattr(args, attr)
    return value, value is not None


def resolve_mixture_config(args: Any = None, model: nn.Module | None = None) -> ResolvedMixtureConfig:
    """Resolve mixture settings using explicit YAML > CLI > safe defaults."""

    values = {kind: dict(defaults) for kind, defaults in MIXTURE_DEFAULTS.items()}
    audit: list[dict[str, Any]] = []

    for kind, fields in CLI_FIELDS.items():
        for field_name, attr in fields.items():
            value, present = _cli_value(args, attr)
            if present:
                values[kind][field_name] = value

    if model is None:
        return ResolvedMixtureConfig(values=values, audit=audit)

    explicit_by_path = {path: getattr(module, "_mixture_config_explicit", {}) for path, module in model.named_modules()}

    for path, module in model.named_modules():
        kind = _module_kind(module)
        if kind is None:
            continue
        explicit = dict(getattr(module, "_mixture_config_explicit", {}))
        # Wrapper YAML values govern its nested routed blocks. This is the
        # important distinction between a module-level YAML contract and a
        # trainer-wide override.
        for parent_path, parent_explicit in explicit_by_path.items():
            is_child = path.startswith(f"{parent_path}.") if parent_path else bool(path)
            if parent_path != path and is_child:
                explicit = {**parent_explicit, **explicit}
        module_values = {}
        module_sources = {}
        for field_name, default in values[kind].items():
            if field_name in explicit:
                module_values[field_name] = explicit[field_name]
                module_sources[field_name] = "yaml"
            else:
                attr = CLI_FIELDS[kind].get(field_name)
                cli_value, present = _cli_value(args, attr) if attr else (None, False)
                module_values[field_name] = cli_value if present else default
                module_sources[field_name] = "cli" if present else "default"
        audit.append(
            {
                "module": path or module.__class__.__name__,
                "kind": kind,
                "values": module_values,
                "sources": module_sources,
                "explicit_fields": sorted(explicit),
                "resolved_sources": dict(module_sources),
            }
        )

    return ResolvedMixtureConfig(values=values, audit=audit)


def apply_mixture_config(model: nn.Module, resolved: ResolvedMixtureConfig) -> int:
    """Apply resolved runtime values once, respecting YAML annotations."""

    applied = 0
    explicit_by_ancestor: list[tuple[str, dict[str, Any]]] = []
    for path, module in model.named_modules():
        explicit_by_ancestor.append((path, getattr(module, "_mixture_config_explicit", {})))

    for path, module in model.named_modules():
        kind = _module_kind(module)
        if kind is None:
            continue
        inherited_explicit: set[str] = set(getattr(module, "_mixture_config_explicit", {}))
        for parent_path, parent_explicit in explicit_by_ancestor:
            is_child = path.startswith(f"{parent_path}.") if parent_path else bool(path)
            if is_child and parent_path != path:
                inherited_explicit.update(parent_explicit)

        config = resolved.values[kind]
        if kind == "moe":
            targets = {
                "balance_loss_coeff": (module, "balance_loss_coeff"),
                "router_z_loss_coeff": (module, "router_z_loss_coeff"),
                "weight_threshold": (module, "weight_threshold"),
                "noise_std": (getattr(module, "routing", None), "noise_std"),
                "temperature": (getattr(module, "routing", None), "temperature"),
            }
            for key, (target, attr) in targets.items():
                if key in inherited_explicit or target is None or not hasattr(target, attr):
                    continue
                setattr(target, attr, config[key])
                applied += 1
            loss_fn = getattr(module, "moe_loss_fn", None)
            if loss_fn is not None and "balance_loss_coeff" not in inherited_explicit:
                loss_fn.balance_loss_coeff = config["balance_loss_coeff"]
            if loss_fn is not None and "router_z_loss_coeff" not in inherited_explicit:
                loss_fn.z_loss_coeff = config["router_z_loss_coeff"]
        elif kind == "moa":
            router = getattr(module, "router", None)
            if router is not None and hasattr(router, "temperature") and "temperature" not in inherited_explicit:
                router.temperature = config["temperature"]
            local_head = getattr(module, "local_head", None)
            if (
                local_head is not None
                and hasattr(local_head, "window_size")
                and "local_window_size" not in inherited_explicit
            ):
                local_head.window_size = max(1, int(config["local_window_size"]))
            regional_head = getattr(module, "region_head", None)
            if (
                regional_head is not None
                and hasattr(regional_head, "max_kv_tokens")
                and "regional_max_kv_tokens" not in inherited_explicit
            ):
                budget = config["regional_max_kv_tokens"]
                regional_head.max_kv_tokens = None if budget in (None, 0) else max(1, int(budget))
            if hasattr(module, "aux_loss_coeff") and "aux_loss_coeff" not in inherited_explicit:
                module.aux_loss_coeff = config["aux_loss_coeff"]
            if hasattr(module, "sparse_inference") and "sparse_inference" not in inherited_explicit:
                module.sparse_inference = bool(config["sparse_inference"])
                for child in module.modules():
                    if child is not module and hasattr(child, "sparse_inference"):
                        child.sparse_inference = module.sparse_inference
            if hasattr(module, "sparse_inference_threshold") and "sparse_inference_threshold" not in inherited_explicit:
                threshold = float(config["sparse_inference_threshold"])
                if not 0.0 <= threshold < 1.0:
                    raise ValueError("moa_sparse_inference_threshold must be in [0, 1)")
                module.sparse_inference_threshold = threshold
                for child in module.modules():
                    if child is not module and hasattr(child, "sparse_inference_threshold"):
                        child.sparse_inference_threshold = threshold
        elif kind == "mot":
            if hasattr(module, "balance_loss_coeff") and "balance_loss_coeff" not in inherited_explicit:
                module.balance_loss_coeff = config["balance_loss_coeff"]
            if hasattr(module, "router_z_loss_coeff") and "router_z_loss_coeff" not in inherited_explicit:
                module.router_z_loss_coeff = config["router_z_loss_coeff"]
            if hasattr(module, "sparse_train") and "sparse_train" not in inherited_explicit:
                module.sparse_train = bool(config["sparse_train"])
            if hasattr(module, "sparse_train_warmup_steps") and "sparse_train_warmup_steps" not in inherited_explicit:
                module.sparse_train_warmup_steps = max(0, int(config["sparse_train_warmup_steps"]))
            router = getattr(module, "router", None)
            if router is not None and "scene_aware_router" not in inherited_explicit:
                if bool(config["scene_aware_router"]):
                    router.enable_scene_aware(config["scene_hidden_dim"])
                else:
                    router.scene_aware = False
            if hasattr(module, "scene_consistency_coeff") and "scene_consistency_coeff" not in inherited_explicit:
                module.scene_consistency_coeff = max(0.0, float(config["scene_consistency_coeff"]))
            if router is not None and "scene_inference_mode" not in inherited_explicit:
                router.set_scene_inference_mode(config["scene_inference_mode"])
            if router is not None and hasattr(router, "temperature") and "temperature" not in inherited_explicit:
                if hasattr(router.temperature, "fill_"):
                    router.temperature.fill_(float(config["temperature"]))
                else:
                    router.temperature = float(config["temperature"])
            if "local_attn_window" not in inherited_explicit:
                experts = getattr(module, "experts", None)
                if experts and hasattr(experts[0], "local_window_size"):
                    experts[0].local_window_size = max(0, int(config["local_attn_window"]))
                    applied += 1
        elif kind == "molora":
            loss_fn = getattr(module, "loss_fn", None)
            if loss_fn is not None:
                for key, attr in (
                    ("balance_loss_coef", "balance_loss_coef"),
                    ("z_loss_coef", "z_loss_coef"),
                    ("diversity_loss_coef", "diversity_loss_coef"),
                ):
                    if hasattr(loss_fn, attr) and key not in inherited_explicit:
                        setattr(loss_fn, attr, config[key])
        elif kind == "latent":
            if (
                hasattr(module, "inference_top_k")
                and config["inference_top_k"] is not None
                and "inference_top_k" not in inherited_explicit
            ):
                module.inference_top_k = max(1, min(int(config["inference_top_k"]), int(module.num_experts)))
        # Keep a compact runtime audit on the model for logging/checkpoints.
    model.mixture_config_audit = resolved.audit
    return applied


__all__ = [
    "MIXTURE_DEFAULTS",
    "CLI_FIELDS",
    "ResolvedMixtureConfig",
    "annotate_mixture_yaml_config",
    "resolve_mixture_config",
    "apply_mixture_config",
]
