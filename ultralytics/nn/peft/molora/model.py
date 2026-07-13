"""MoLoRA model wrapper and PEFT-style entry point.

Provides:
  - get_peft_molora_model(model, config): wrap an Ultralytics model with MoLoRA
  - MoLoRAModel: convenience wrapper with aux_loss, merge/unmerge, checkpointing
"""
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from ultralytics.utils import LOGGER
from .config import MoLoRAConfig, MoLoRAConfigBuilder
from .layer import MoLoRALayer
from .utils import mark_only_molora_as_trainable, count_parameters


def get_peft_molora_model(
    model: nn.Module,
    config: Union[MoLoRAConfig, Dict[str, Any]],
) -> nn.Module:
    """Wrap an Ultralytics DetectionModel (or any nn.Module) with MoLoRA layers.

    Args:
        model: The base model to adapt.
        config: Either a MoLoRAConfig dataclass or a dict with keys:
            r, alpha, num_experts, top_k, router_type,
            target_modules (list of module names), dropout, etc.

    Returns:
        The model with selected layers replaced by MoLoRALayer wrappers.
        Note: this modifies the model **in-place**.
    """
    # Prevent double-wrapping
    if getattr(model, "molora_enabled", False):
        LOGGER.warning("[MoLoRA] Model already has MoLoRA enabled. Skipping re-application.")
        return model

    if isinstance(config, MoLoRAConfig):
        cfg = config
    else:
        cfg = MoLoRAConfig(**{k: v for k, v in config.items() if k in MoLoRAConfig.__dataclass_fields__})

    # Resolve target modules
    target_modules = getattr(cfg, "target_modules", None)
    if target_modules is None or not target_modules:
        LOGGER.warning("[MoLoRA] No target_modules specified; running auto-detection.")
        target_modules = MoLoRAConfigBuilder.auto_detect_targets(
            model,
            r=cfg.r,
            include_moe=getattr(cfg, "include_moe", True),
            include_attention=getattr(cfg, "include_attention", False),
            only_backbone=getattr(cfg, "only_backbone", False),
            exclude_modules=getattr(cfg, "exclude_modules", None),
            allow_depthwise=getattr(cfg, "allow_depthwise", False),
            kernels=getattr(cfg, "kernels", None),
            skip_stem=getattr(cfg, "skip_stem", False),
            min_channels=getattr(cfg, "min_channels", 0),
            only_3x3=getattr(cfg, "only_3x3", False),
        )
        if not target_modules:
            LOGGER.warning("[MoLoRA] Auto-detection found no compatible layers. Returning model unchanged.")
            return model

    # Wrap each target module in-place by name
    wrapped = 0
    modules_dict = dict(model.named_modules())
    for name in target_modules:
        if name not in modules_dict:
            continue
        base_layer = modules_dict[name]
        if not isinstance(base_layer, (nn.Conv2d, nn.Linear)):
            continue

        # Parent module and local attribute name for in-place replacement
        parent_name, child_name = _parent_child_name(name)
        parent = _get_submodule(model, parent_name) if parent_name else model
        if parent is None or not hasattr(parent, child_name):
            continue

        molora_layer = MoLoRALayer(
            base_layer=base_layer,
            r=cfg.r,
            alpha=cfg.alpha,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            router_type=cfg.router_type,
            dropout=cfg.dropout,
            use_rslora=getattr(cfg, "use_rslora", True),
            balance_loss_coef=cfg.balance_loss_coef,
            z_loss_coef=cfg.z_loss_coef,
            diversity_loss_coef=cfg.diversity_loss_coef,
            expert_init=cfg.expert_init,
            share_moe_registry=cfg.share_moe_registry,
            router_hidden_dim=getattr(cfg, "router_hidden_dim", None),
            capacity_factor=cfg.capacity_factor,
            expert_dropout=cfg.expert_dropout,
            top_k_warmup=cfg.top_k_warmup,
            warmup_steps=cfg.warmup_steps,
            domain_experts=getattr(cfg, "domain_experts", None),
        )

        setattr(parent, child_name, molora_layer)
        wrapped += 1

    LOGGER.info(f"[MoLoRA] Wrapped {wrapped} layers with MoLoRA (E={cfg.num_experts}, K={cfg.top_k}).")

    # Attach metadata
    model.molora_config = cfg  # type: ignore[union-attr]
    model.molora_enabled = True  # type: ignore[union-attr]

    # Freeze all non-MoLoRA parameters so only adapter weights are trainable
    mark_only_molora_as_trainable(model)
    LOGGER.info("[MoLoRA] Frozen non-MoLoRA parameters. Only adapter weights are trainable.")

    return model


def _parent_child_name(full_name: str) -> tuple:
    """Split 'model.5.m.0.cv1' -> ('model.5.m.0', 'cv1')."""
    parts = full_name.rsplit(".", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "", parts[0]


def _get_submodule(model: nn.Module, path: str) -> Optional[nn.Module]:
    """Navigate to a submodule by dot-separated path."""
    if not path:
        return model
    parts = path.split(".")
    mod = model
    for p in parts:
        if hasattr(mod, p):
            mod = getattr(mod, p)
        elif p.isdigit() and isinstance(mod, (nn.Sequential, nn.ModuleList)):
            mod = mod[int(p)]
        else:
            return None
    return mod


# ---------------------------------------------------------------------------
# Wrapper class (optional convenience)
# ---------------------------------------------------------------------------

class MoLoRAModel(nn.Module):
    """Thin wrapper around a base model that adds MoLoRA bookkeeping.

    Not required for training; you can use `get_peft_molora_model` directly
    and then call `mark_only_molora_as_trainable`. This wrapper is useful
    when you want a single object that exposes:
      - compute_aux_loss()
      - merge() / unmerge()
      - save_checkpoint() / load_checkpoint()
    """

    def __init__(self, model: nn.Module, config: Union[MoLoRAConfig, Dict[str, Any]]):
        super().__init__()
        self.model = get_peft_molora_model(model, config)
        self.config = self.model.molora_config if hasattr(self.model, "molora_config") else config
        # get_peft_molora_model already called mark_only_molora_as_trainable

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_aux_loss(self) -> torch.Tensor:
        """Collect MoLoRA aux losses from MOE_LOSS_REGISTRY.

        Call this after forward() in the training loop and add it to the
        total loss. The registry is automatically cleared by tasks.py
        before each training forward.
        """
        aux_loss = torch.tensor(0.0)
        try:
            from ultralytics.nn.modules.moe.modules import MOE_LOSS_REGISTRY
        except Exception:
            return aux_loss

        device = next(self.model.parameters()).device
        aux_loss = aux_loss.to(device)
        # the `seen` set was redundant — `model.modules()` yields each
        # module exactly once, so MoLoRALayer instances cannot be double-counted.
        for m in self.model.modules():
            if isinstance(m, MoLoRALayer):
                loss_t = MOE_LOSS_REGISTRY.get(m)
                if isinstance(loss_t, torch.Tensor):
                    aux_loss = aux_loss + loss_t.to(device)
        return aux_loss

    def merge(self) -> None:
        """Merge all MoLoRALayer weights for inference."""
        for m in self.model.modules():
            if isinstance(m, MoLoRALayer):
                m.merge_weights()
        LOGGER.info("[MoLoRA] All layers merged.")

    def unmerge(self) -> None:
        """Unmerge all MoLoRALayer weights."""
        for m in self.model.modules():
            if isinstance(m, MoLoRALayer):
                m.unmerge_weights()
        LOGGER.info("[MoLoRA] All layers unmerged.")

    # ------------------------------------------------------------------
    # Domain & Continual Learning
    # ------------------------------------------------------------------

    def set_domain(self, domain: str) -> None:
        """Restrict all MoLoRALayers to domain-specific experts."""
        for m in self.model.modules():
            if isinstance(m, MoLoRALayer):
                m.set_domain(domain)
        LOGGER.info(f"[MoLoRA] Domain set to '{domain}'.")

    def clear_domain(self) -> None:
        """Clear domain restrictions on all MoLoRALayers."""
        for m in self.model.modules():
            if isinstance(m, MoLoRALayer):
                m.clear_domain()
        LOGGER.info("[MoLoRA] Domain restrictions cleared.")

    def freeze_experts(self, expert_indices: List[int]) -> None:
        """Freeze specified experts across all MoLoRALayers."""
        for m in self.model.modules():
            if isinstance(m, MoLoRALayer):
                m.freeze_experts(expert_indices)

    def unfreeze_experts(self, expert_indices: Optional[List[int]] = None) -> None:
        """Unfreeze specified or all experts across all MoLoRALayers."""
        for m in self.model.modules():
            if isinstance(m, MoLoRALayer):
                m.unfreeze_experts(expert_indices)

    # ------------------------------------------------------------------
    # Expert Replay (continual learning)
    # ------------------------------------------------------------------

    def save_expert_replay_buffer(self, domain: str, path: Optional[str] = None) -> Dict[str, Any]:
        """Save current expert weights for a domain into a replay buffer.

        This allows restoring old-domain experts later to prevent catastrophic
        forgetting when training on new domains.
        """
        buffer: Dict[str, Any] = {"domain": domain, "experts": {}}
        for name, m in self.model.named_modules():
            if isinstance(m, MoLoRALayer):
                buffer["experts"][name] = {
                    idx: {
                        "lora_A": m.experts[idx].lora_A.state_dict(),
                        "lora_B": m.experts[idx].lora_B.state_dict(),
                    }
                    for idx in range(m.num_experts)
                }
        if path is not None:
            torch.save(buffer, path)
            LOGGER.info(f"[MoLoRA] Expert replay buffer saved to {path} for domain '{domain}'")
        return buffer

    def load_expert_replay_buffer(self, buffer: Union[str, Dict[str, Any]], domain: Optional[str] = None) -> None:
        """Load expert weights from a replay buffer.

        Args:
            buffer: Either a file path or a dict returned by save_expert_replay_buffer.
            domain: Optional domain name to verify (if buffer is a dict with 'domain' key).
        """
        if isinstance(buffer, str):
            buffer = torch.load(buffer, map_location="cpu")
        if domain is not None and buffer.get("domain") != domain:
            LOGGER.warning(f"[MoLoRA] Replay buffer domain mismatch: {buffer.get('domain')} vs {domain}")
        for name, m in self.model.named_modules():
            if isinstance(m, MoLoRALayer) and name in buffer["experts"]:
                for idx, states in buffer["experts"][name].items():
                    m.experts[idx].lora_A.load_state_dict(states["lora_A"])
                    m.experts[idx].lora_B.load_state_dict(states["lora_B"])
        LOGGER.info(f"[MoLoRA] Expert replay buffer loaded for domain '{buffer.get('domain')}'")

    def save_checkpoint(self, path: str) -> None:
        """Save a versioned MoLoRA-only checkpoint with explicit compatibility metadata.

        Includes registered buffers (e.g. ``_step_count``, ``_usage_ema``)
        alongside trainable parameters, since these carry training state
        needed for correct resume.
        """
        molora_keys = ("lora_A", "lora_B", "router", "molora",
                       "_step_count", "_usage_ema", "_domain_active_mask")
        config = asdict(self.config) if is_dataclass(self.config) else dict(self.config)
        state = {
            "schema_version": 1,
            "format": "molora_adapter",
            "config": config,
            "structure": _molora_structure(self.model),
            "state_dict": {
                k: v for k, v in self.model.state_dict().items()
                if any(p in k for p in molora_keys)
            },
        }
        torch.save(state, path)
        LOGGER.info(f"[MoLoRA] Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load a checkpoint, rejecting incompatible configuration or partial state."""
        state = torch.load(path, map_location="cpu")
        if not isinstance(state, dict) or "state_dict" not in state:
            raise ValueError(
                "Invalid MoLoRA checkpoint: expected a dict with 'state_dict'. "
                "Legacy/unversioned checkpoints must be re-exported with save_checkpoint()."
            )
        if state.get("schema_version") != 1 or state.get("format") != "molora_adapter":
            raise ValueError(
                f"Unsupported MoLoRA checkpoint schema: version={state.get('schema_version')!r}, "
                f"format={state.get('format')!r}; expected version=1, format='molora_adapter'."
            )
        saved_config = state.get("config")
        if not isinstance(saved_config, dict):
            raise ValueError("Invalid MoLoRA checkpoint: missing complete 'config' dictionary.")
        current_config = asdict(self.config) if is_dataclass(self.config) else dict(self.config)
        for key in ("r", "num_experts", "top_k", "router_type", "target_modules"):
            saved_value = saved_config.get(key)
            current_value = current_config.get(key)
            if key == "target_modules":
                saved_value = sorted(saved_value or [])
                current_value = sorted(current_value or [])
            if saved_value != current_value:
                raise ValueError(
                    f"MoLoRA checkpoint config mismatch for {key}: "
                    f"checkpoint={saved_value!r}, model={current_value!r}."
                )
        saved_structure = state.get("structure")
        current_structure = _molora_structure(self.model)
        if saved_structure != current_structure:
            raise ValueError(
                "MoLoRA checkpoint structure mismatch (target names, layer types, or dimensions differ)."
            )
        checkpoint_sd = state["state_dict"]
        if not isinstance(checkpoint_sd, dict):
            raise ValueError("Invalid MoLoRA checkpoint: 'state_dict' must be a dictionary.")
        expected_keys = {k for k in self.model.state_dict() if _is_molora_state_key(k)}
        missing = sorted(expected_keys - set(checkpoint_sd))
        unexpected = sorted(set(checkpoint_sd) - expected_keys)
        if missing or unexpected:
            raise RuntimeError(
                "MoLoRA checkpoint state mismatch: "
                f"missing={missing[:5]} ({len(missing)} total), "
                f"unexpected={unexpected[:5]} ({len(unexpected)} total)."
            )
        try:
            self.model.load_state_dict(checkpoint_sd, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(f"MoLoRA checkpoint tensor shape mismatch: {exc}") from exc
        LOGGER.info(f"[MoLoRA] Checkpoint loaded from {path}")

    def param_stats(self) -> Dict[str, Any]:
        """Return parameter statistics for the wrapped model."""
        return count_parameters(self.model)


def _is_molora_state_key(key: str) -> bool:
    """Return whether a state key belongs to adapter parameters or state buffers."""
    return any(token in key for token in (
        "lora_A", "lora_B", "router", "loss_fn", "molora",
        "_step_count", "_usage_ema", "_domain_active_mask",
    ))


def _molora_structure(model: nn.Module) -> List[Dict[str, Any]]:
    """Describe wrapped layers sufficiently to reject incompatible adapters."""
    structure = []
    for name, layer in model.named_modules():
        if not isinstance(layer, MoLoRALayer):
            continue
        base = layer.base_layer
        item: Dict[str, Any] = {
            "name": name,
            "base_type": type(base).__name__,
            "r": layer.r,
            "num_experts": layer.num_experts,
            "in_features": getattr(base, "in_features", getattr(base, "in_channels", None)),
            "out_features": getattr(base, "out_features", getattr(base, "out_channels", None)),
        }
        if isinstance(base, nn.Conv2d):
            item.update({"kernel_size": tuple(base.kernel_size), "groups": base.groups})
        structure.append(item)
    return structure
