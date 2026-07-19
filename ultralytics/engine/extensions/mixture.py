"""Runtime lifecycle controller for routed model extensions."""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.moe.config import apply_mixture_config, resolve_mixture_config
from ultralytics.nn.modules.routing_protocol import (
    anneal_mixture_temperatures,
    configure_mixture_temperature_schedule,
    reset_routing_runtime_state,
)
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model


class MixtureRuntimeController:
    """Own routed configuration, scheduling, DDP safety, and runtime reset."""

    def __init__(self, trainer):
        self.trainer = trainer
        self._warmup_expert_params = []

    @property
    def model(self) -> nn.Module:
        return unwrap_model(self.trainer.model)

    def setup(self) -> None:
        self.detect_modules()
        self.resolve_config()
        self._configure_map_saturation()
        if getattr(self.trainer, "_has_moe", False):
            from ultralytics.nn.modules.moe.utils import iter_core_moe_expert_params

            self._warmup_expert_params = [
                parameter for parameter in iter_core_moe_expert_params(self.model) if parameter.requires_grad
            ]
        if getattr(self.trainer, "world_size", 1) > 1:
            self.prepare_ddp()

    def _configure_map_saturation(self) -> None:
        """Attach opt-in validation-driven balance schedulers to core MoE modules."""
        if not getattr(self.trainer.args, "moe_map_saturation_enabled", False):
            return
        from ultralytics.nn.modules.moe.scheduler import MapSaturationScheduler, MapSaturationSchedulerConfig
        from ultralytics.nn.modules.moe.utils import is_core_moe_block

        config = MapSaturationSchedulerConfig(
            enabled=True,
            window_size=int(getattr(self.trainer.args, "moe_map_saturation_window_size", 5)),
            saturation_threshold=float(getattr(self.trainer.args, "moe_map_saturation_threshold", 0.001)),
            decay_factor=float(getattr(self.trainer.args, "moe_map_saturation_decay_factor", 0.8)),
            min_scale=float(getattr(self.trainer.args, "moe_map_saturation_min_scale", 0.1)),
        )
        for module in self.model.modules():
            if not is_core_moe_block(module):
                continue
            if hasattr(module, "balance_loss_coeff"):
                module.map_saturation_scheduler = MapSaturationScheduler(config)
            loss_fn = getattr(module, "moe_loss_fn", None)
            if loss_fn is not None and hasattr(loss_fn, "balance_loss_coeff"):
                loss_fn.map_saturation_scheduler = MapSaturationScheduler(config)

    def detect_modules(self) -> bool:
        from ultralytics.nn.modules.moa import C2fMoA
        from ultralytics.nn.modules.moe.utils import model_has_core_moe
        from ultralytics.nn.modules.mot import C2fMoT

        model = self.model
        self.trainer._has_moa_mot = any(isinstance(module, (C2fMoA, C2fMoT)) for module in model.modules())
        self.trainer._has_moe = model_has_core_moe(model)
        return bool(self.trainer._has_moa_mot or self.trainer._has_moe)

    def resolve_config(self):
        model = self.model
        resolved = resolve_mixture_config(self.trainer.args, model)
        self.trainer.mixture_config = resolved
        apply_mixture_config(model, resolved)
        configure_mixture_temperature_schedule(model, external=True)
        return resolved

    def anneal_temperature(self) -> int:
        factor = float(getattr(self.trainer.args, "moa_mot_temperature_factor", 0.97))
        min_temp = float(getattr(self.trainer.args, "moa_mot_min_temperature", 0.3))
        updated = anneal_mixture_temperatures(self.model, factor=factor, min_temp=min_temp)
        if updated == 0 and getattr(self.trainer, "_has_moa_mot", False) and RANK in {-1, 0}:
            LOGGER.warning("[Mixture] temperature scheduler found no routable temperature buffers")
        return updated

    def prepare_ddp(self) -> tuple[int, int, int]:
        """Disable checkpoint recomputation and sparse dispatch combinations unsafe under DDP."""
        root = self.model
        disabled = frozen = dense = 0
        for module in root.modules():
            if getattr(module, "use_gradient_checkpointing", False):
                module.use_gradient_checkpointing = False
                disabled += 1
            if hasattr(module, "sparse_train") and module.sparse_train:
                module.sparse_train = False
                dense += 1
            if hasattr(module, "expert_projections") and hasattr(module, "ddp_safe_dense"):
                if not module.ddp_safe_dense:
                    module.ddp_safe_dense = True
                    dense += 1
        for name, parameter in root.named_parameters():
            lname = name.lower()
            if "lora_" in lname and any(token in lname for token in ("complexity_estimator", "se_gate")):
                if parameter.requires_grad:
                    parameter.requires_grad_(False)
                    frozen += 1
        if disabled or frozen or dense:
            LOGGER.warning(
                f"[Mixture+DDP] disabled checkpointing={disabled}, "
                f"enabled dense routing={dense}, froze control-path adapters={frozen}."
            )
        return disabled, frozen, dense

    def begin_forward(self) -> int:
        try:
            from ultralytics.nn.modules.moe.modules import MOE_LOSS_REGISTRY

            MOE_LOSS_REGISTRY.clear()
        except Exception:
            pass
        return reset_routing_runtime_state(self.model)

    def begin_epoch(self, epoch: int) -> None:
        if epoch > int(getattr(self.trainer, "start_epoch", 0)):
            self.anneal_temperature()
        if not getattr(self.trainer, "_has_moe", False):
            return
        warmup = int(getattr(self.trainer.args, "moe_expert_warmup_epochs", 3))
        trainable = epoch >= warmup
        for parameter in self._warmup_expert_params:
            parameter.requires_grad = trainable

    def reset_runtime(self) -> int:
        return reset_routing_runtime_state(self.model)

    def finalize_epoch(self, *, recovered: bool, validated: bool) -> int:
        """Advance mAP-saturation schedulers only for accepted validation epochs."""
        if recovered or not validated or not getattr(self.trainer, "_has_moe", False):
            return 0
        if not getattr(self.trainer.args, "moe_map_saturation_enabled", False):
            return 0
        fitness = getattr(self.trainer, "fitness", None)
        if fitness is None or not torch.isfinite(torch.as_tensor(fitness)):
            return 0
        updated, seen = 0, set()
        for module in self.model.modules():
            scheduler = getattr(module, "map_saturation_scheduler", None)
            if scheduler is None or id(scheduler) in seen:
                continue
            scheduler.update(float(fitness))
            seen.add(id(scheduler))
            updated += 1
        return updated
