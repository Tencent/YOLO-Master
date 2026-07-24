"""Runtime lifecycle controller for routed model extensions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

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
        self._gini_scheduler = None
        self._gini_usage_totals: dict[str, torch.Tensor] = {}
        self._gini_usage_weights: dict[str, float] = {}

    @property
    def model(self) -> nn.Module:
        return unwrap_model(self.trainer.model)

    def setup(self) -> None:
        self.detect_modules()
        self.resolve_config()
        self._configure_gini_schedule()
        self._configure_map_saturation()
        if getattr(self.trainer, "_has_moe", False):
            from ultralytics.nn.modules.moe.utils import iter_core_moe_expert_params

            self._warmup_expert_params = [
                parameter for parameter in iter_core_moe_expert_params(self.model) if parameter.requires_grad
            ]
        if getattr(self.trainer, "world_size", 1) > 1:
            self.prepare_ddp()

    def _configure_gini_schedule(self) -> None:
        """Configure the opt-in epoch-level expert-usage Gini scheduler."""
        mode = str(getattr(self.trainer.args, "moe_dynamic_schedule", "none")).strip().lower()
        if mode in {"", "none", "off", "false"}:
            return
        if mode not in {"gini", "gini_balance"}:
            raise ValueError(f"Unsupported moe_dynamic_schedule={mode!r}; expected 'none' or 'gini'")
        if not getattr(self.trainer, "_has_moe", False):
            LOGGER.warning("[MoE dynamic] requested Gini scheduling, but the model has no core MoE blocks")
            return

        from ultralytics.nn.modules.moe.schedule import GiniBalanceScheduler, apply_balance_loss_coeff
        from ultralytics.nn.modules.moe.utils import is_core_moe_block

        self._gini_scheduler = GiniBalanceScheduler(
            base=float(getattr(self.trainer.args, "moe_balance_loss", 1.0)),
            target=float(getattr(self.trainer.args, "moe_dynamic_gini_target", 0.25)),
            alpha=float(getattr(self.trainer.args, "moe_dynamic_gini_alpha", 1.0)),
            beta=float(getattr(self.trainer.args, "moe_dynamic_gini_beta", 0.8)),
            min_coeff=float(getattr(self.trainer.args, "moe_dynamic_balance_min", 0.5)),
            max_coeff=float(getattr(self.trainer.args, "moe_dynamic_balance_max", 2.0)),
        )
        state = getattr(self.model, "_moe_gini_schedule_state", None)
        if isinstance(state, dict):
            ema = state.get("ema_gini")
            self._gini_scheduler.ema = float(ema) if ema is not None else None
            if state.get("balance_loss_coeff") is not None:
                apply_balance_loss_coeff(self.model, float(state["balance_loss_coeff"]))
        for module in self.model.modules():
            if is_core_moe_block(module):
                module._moe_force_snapshot = True

    def collect_routing_usage(self, *, batch_weight: float = 1.0) -> int:
        """Accumulate detached per-layer expert usage for the current training epoch."""
        if self._gini_scheduler is None:
            return 0
        from ultralytics.nn.modules.moe.utils import is_core_moe_block

        collected = 0
        weight = max(float(batch_weight), 1.0)
        for name, module in self.model.named_modules():
            if not is_core_moe_block(module):
                continue
            snapshot = getattr(module, "last_routing_snapshot", None)
            if not isinstance(snapshot, dict):
                continue
            usage = snapshot.get("topk_counts")
            raw_counts = isinstance(usage, torch.Tensor) and bool(usage.numel())
            if not raw_counts:
                usage = snapshot.get("expert_usage")
            if not isinstance(usage, torch.Tensor) or not usage.numel():
                continue
            usage = torch.nan_to_num(usage.detach().float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
            if not raw_counts:
                usage = usage / usage.sum().clamp_min(1e-12) * weight
            observation_weight = weight
            previous = self._gini_usage_totals.get(name)
            self._gini_usage_totals[name] = usage.clone() if previous is None else previous + usage
            self._gini_usage_weights[name] = self._gini_usage_weights.get(name, 0.0) + observation_weight
            collected += 1
        return collected

    def _write_gini_trace(self, *, mean_gini: float, layer_gini: dict[str, float], coeff: float) -> None:
        if RANK not in {-1, 0}:
            return
        trace = Path(self.trainer.save_dir) / "moe_dynamic_schedule.csv"
        trace.parent.mkdir(parents=True, exist_ok=True)
        fields = (
            "epoch",
            "mean_gini",
            "ema_gini",
            "balance_loss_coeff",
            "layers",
            "routing_observations",
            "layer_gini",
        )
        is_new = not trace.exists() or trace.stat().st_size == 0
        with trace.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if is_new:
                writer.writeheader()
            writer.writerow(
                {
                    "epoch": int(getattr(self.trainer, "epoch", -1)) + 1,
                    "mean_gini": mean_gini,
                    "ema_gini": self._gini_scheduler.ema,
                    "balance_loss_coeff": coeff,
                    "layers": len(layer_gini),
                    "routing_observations": sum(self._gini_usage_weights.values()),
                    "layer_gini": json.dumps(layer_gini, sort_keys=True),
                }
            )

    def _finalize_gini_epoch(self, *, recovered: bool) -> int:
        """Commit a successful epoch's Gini state and update the next epoch's coefficient."""
        if recovered or self._gini_scheduler is None or not self._gini_usage_totals:
            return 0
        from ultralytics.nn.modules.moe.schedule import apply_balance_loss_coeff, usage_gini

        layer_gini = {}
        for name, usage in self._gini_usage_totals.items():
            reduced = usage.clone()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
            layer_gini[name] = usage_gini(reduced)
        mean_gini = float(sum(layer_gini.values()) / len(layer_gini))
        coeff = self._gini_scheduler.update(mean_gini)
        updated = apply_balance_loss_coeff(self.model, coeff)
        state = {
            "ema_gini": self._gini_scheduler.ema,
            "balance_loss_coeff": coeff,
            "mean_gini": mean_gini,
            "epoch": int(getattr(self.trainer, "epoch", -1)) + 1,
        }
        self.model._moe_gini_schedule_state = state
        ema_model = getattr(getattr(self.trainer, "ema", None), "ema", None)
        if isinstance(ema_model, nn.Module):
            ema_model = unwrap_model(ema_model)
            apply_balance_loss_coeff(ema_model, coeff)
            ema_model._moe_gini_schedule_state = dict(state)
        self._write_gini_trace(mean_gini=mean_gini, layer_gini=layer_gini, coeff=coeff)
        return updated

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
        self._gini_usage_totals.clear()
        self._gini_usage_weights.clear()
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
        """Advance dynamic schedulers only for successful, accepted epochs."""
        updated = self._finalize_gini_epoch(recovered=recovered)
        if recovered or not validated or not getattr(self.trainer, "_has_moe", False):
            return updated
        if not getattr(self.trainer.args, "moe_map_saturation_enabled", False):
            return updated
        fitness = getattr(self.trainer, "fitness", None)
        if fitness is None or not torch.isfinite(torch.as_tensor(fitness)):
            return updated
        map_updates, seen = 0, set()
        for module in self.model.modules():
            scheduler = getattr(module, "map_saturation_scheduler", None)
            if scheduler is None or id(scheduler) in seen:
                continue
            scheduler.update(float(fitness))
            seen.add(id(scheduler))
            map_updates += 1
        return updated + map_updates
