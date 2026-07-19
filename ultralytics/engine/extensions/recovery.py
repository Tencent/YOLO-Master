"""Checkpoint health, non-finite recovery, and distributed validation helpers."""

from __future__ import annotations

import io
import math
import os
import pickle
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from torch import distributed as dist
from torch import nn

from ultralytics import __version__
from ultralytics.utils import GIT, LOGGER
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import TORCH_2_4, convert_optimizer_state_dict_to_fp16, unwrap_model


class TrainingRecoveryController:
    """Own healthy checkpoint serialization and coordinated NaN/Inf recovery."""

    def __init__(self, trainer):
        self.trainer = trainer

    @staticmethod
    def rank() -> int:
        """Read the trainer module rank so tests and torchrun initialization share one source."""
        from ultralytics.engine import trainer as trainer_module

        return int(trainer_module.RANK)

    @staticmethod
    def state_is_finite(value) -> bool:
        """Return whether every floating tensor nested in a state object is finite."""
        if isinstance(value, torch.Tensor):
            return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all().item())
        if isinstance(value, nn.Module):
            return all(TrainingRecoveryController.state_is_finite(item) for item in value.state_dict().values())
        if isinstance(value, dict):
            return all(TrainingRecoveryController.state_is_finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(TrainingRecoveryController.state_is_finite(item) for item in value)
        return True

    def sync_nonfinite_flag(self, local_nonfinite: bool) -> bool:
        """Reduce a local non-finite flag across all initialized ranks."""
        if self.rank() == -1 or not dist.is_initialized():
            return bool(local_nonfinite)
        backend = dist.get_backend()
        device = self.trainer.device if backend == "nccl" else torch.device("cpu")
        flag = torch.tensor(int(local_nonfinite), dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def sync_ema_buffers(self) -> None:
        """Broadcast EMA buffers while keeping non-persistent CPU diagnostics off NCCL."""
        trainer = self.trainer
        if not getattr(trainer, "ema", None) or getattr(trainer, "world_size", 1) <= 1 or not dist.is_initialized():
            return
        backend, skipped = dist.get_backend(), []
        for module_name, module in trainer.ema.ema.named_modules():
            for name, buffer in module.named_buffers(recurse=False):
                full_name = f"{module_name}.{name}" if module_name else name
                persistent = name not in module._non_persistent_buffers_set
                if backend == "nccl" and buffer.device.type != "cuda":
                    if not persistent:
                        skipped.append(full_name)
                        continue
                    try:
                        buffer = buffer.to(trainer.device, non_blocking=True).detach()
                        module._buffers[name] = buffer
                    except RuntimeError:
                        skipped.append(full_name)
                        continue
                dist.broadcast(buffer, src=0)
        if skipped and not getattr(trainer, "_warned_ema_cpu_diagnostics", False):
            LOGGER.warning(f"Skipping {len(skipped)} non-persistent CPU EMA diagnostic buffer(s) for validation.")
            trainer._warned_ema_cpu_diagnostics = True

    @staticmethod
    def reset_runtime(model=None) -> None:
        """Clear per-process routed state omitted from checkpoints."""
        from ultralytics.nn.modules.moe._common import MOE_LOSS_REGISTRY, _MOE_LOSS_REGISTRY_LOCK
        from ultralytics.nn.modules.routing_protocol import reset_routing_runtime_state

        with _MOE_LOSS_REGISTRY_LOCK:
            MOE_LOSS_REGISTRY.clear()
        reset_routing_runtime_state(unwrap_model(model) if model is not None else None)

    def serialize_checkpoint(self) -> bytes:
        """Serialize complete online, EMA, optimizer, scaler, and metadata state."""
        trainer = self.trainer
        from ultralytics.nn.mixture_loss import _get_mixture_loss_ema
        from ultralytics.utils.checkpoint_compat import checkpoint_runtime_metadata

        _get_mixture_loss_ema(unwrap_model(trainer.model))
        if getattr(trainer, "ema", None):
            _get_mixture_loss_ema(unwrap_model(trainer.ema.ema))
        buffer = io.BytesIO()
        model = deepcopy(unwrap_model(trainer.model)).half()
        ema = deepcopy(unwrap_model(trainer.ema.ema)).half() if getattr(trainer, "ema", None) else None
        for snapshot in (model, ema):
            if snapshot is None:
                continue
            if hasattr(snapshot, "criterion"):
                snapshot.criterion = None
            for value in snapshot.state_dict().values():
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    torch.nan_to_num_(value)
        torch.save(
            {
                "epoch": getattr(trainer, "epoch", trainer.start_epoch - 1),
                "best_fitness": trainer.best_fitness,
                "model": model,
                "ema": ema,
                "updates": trainer.ema.updates if trainer.ema else 0,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(trainer.optimizer.state_dict())),
                "scaler": trainer.scaler.state_dict(),
                "train_args": vars(trainer.args),
                "train_metrics": {**getattr(trainer, "metrics", {}), "fitness": trainer.fitness},
                "train_results": trainer.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "git": {"root": str(GIT.root), "branch": GIT.branch, "commit": GIT.commit, "origin": GIT.origin},
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
                "mixture_checkpoint": checkpoint_runtime_metadata(model),
            },
            buffer,
        )
        return buffer.getvalue()

    def resync_nonfinite_ema(self) -> bool:
        """Replace poisoned EMA tensors with finite online tensors when structures match."""
        trainer = self.trainer
        ema = getattr(getattr(trainer, "ema", None), "ema", None)
        model = getattr(trainer, "model", None)
        if ema is None or model is None:
            return ema is None
        ema_state = unwrap_model(ema).state_dict()
        model_state = unwrap_model(model).state_dict()
        with torch.no_grad():
            for key, value in ema_state.items():
                if not isinstance(value, torch.Tensor) or self.state_is_finite(value):
                    continue
                source = model_state.get(key)
                if (
                    isinstance(source, torch.Tensor)
                    and source.shape == value.shape
                    and self.state_is_finite(source)
                ):
                    value.copy_(source.to(device=value.device, dtype=value.dtype))
        return self.state_is_finite(ema)

    def checkpoint_forward_smoke(self, checkpoint) -> tuple[bool, str]:
        """Run small fused FP32 inference samples and reject non-finite activations."""
        model = checkpoint.get("ema") or checkpoint.get("model")
        if not isinstance(model, nn.Module):
            return False, "checkpoint has no loadable model or EMA module"
        try:
            model = model.float().cpu().eval()
            fuse = getattr(model, "fuse", None)
            if callable(fuse):
                model = fuse(verbose=False)
            yaml = getattr(model, "yaml", {}) or {}
            channels = int(yaml.get("channels", 3))
            stride = max(1, int(torch.as_tensor(getattr(model, "stride", torch.tensor([32.0]))).max().item()))
            configured = getattr(getattr(self.trainer, "args", None), "imgsz", 64)
            configured = max(configured) if isinstance(configured, (list, tuple)) else configured
            is_rtdetr = any(module.__class__.__name__ == "RTDETRDecoder" for module in model.modules())
            smoke_min, smoke_max = (128, 128) if is_rtdetr else (32, 64)
            imgsz = math.ceil(max(smoke_min, min(int(configured), smoke_max)) / stride) * stride
            first = next(model.parameters(), None)
            sample = (
                torch.zeros(1, first.shape[1], dtype=torch.float32)
                if not yaml and first is not None and first.ndim == 2
                else torch.zeros(1, channels, imgsz, imgsz, dtype=torch.float32)
            )
            with torch.no_grad():
                for index, smoke_input in enumerate(
                    (sample, torch.linspace(-1.0, 1.0, sample.numel(), dtype=torch.float32).reshape_as(sample))
                ):
                    if not self.state_is_finite(model(smoke_input)):
                        return False, f"forward smoke sample {index} produced non-finite output"
        except Exception as exc:
            return False, f"forward smoke failed: {type(exc).__name__}: {exc}"
        return True, ""

    def validate_artifact(self, path) -> tuple[bool, str]:
        """Verify that a checkpoint is readable, finite, and executable."""
        try:
            checkpoint = torch_load(Path(path), map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
            return False, f"unreadable checkpoint: {type(exc).__name__}: {exc}"
        if not isinstance(checkpoint, dict) or not self.state_is_finite(checkpoint):
            return False, "checkpoint contains missing or non-finite state"
        return self.checkpoint_forward_smoke(checkpoint)

    def select_final_eval_checkpoints(self):
        """Select healthy best/recovery artifacts on rank 0 and share the decision."""
        trainer, decision = self.trainer, None
        rank = self.rank()
        if rank in {-1, 0}:
            candidates, rejected = [], []
            for path in (trainer.best, trainer.healthy):
                path = Path(path)
                if not path.exists() or path in candidates:
                    continue
                healthy, reason = trainer._validate_checkpoint_artifact(path)
                (candidates if healthy else rejected).append(path if healthy else f"{path.name}: {reason}")
            decision = ([str(path) for path in candidates], rejected)
        if rank != -1 and dist.is_initialized():
            shared = [decision]
            dist.broadcast_object_list(shared, src=0)
            decision = shared[0]
        paths, rejected = decision or ([], ["rank 0 did not provide a checkpoint decision"])
        return [Path(path) for path in paths], rejected

    def save_healthy(self, serialized_checkpoint: bytes) -> bool:
        """Atomically replace the recovery checkpoint only with finite executable state."""
        trainer = self.trainer
        checkpoint = torch_load(io.BytesIO(serialized_checkpoint), map_location="cpu", weights_only=False)
        if not self.state_is_finite(checkpoint):
            LOGGER.warning("Skipping non-finite recovery checkpoint state.")
            return False
        healthy, reason = self.checkpoint_forward_smoke(checkpoint)
        if not healthy:
            LOGGER.warning(f"Skipping recovery checkpoint that failed inference health check: {reason}")
            return False
        trainer.healthy.parent.mkdir(parents=True, exist_ok=True)
        temporary = trainer.healthy.with_suffix(".tmp")
        temporary.write_bytes(serialized_checkpoint)
        os.replace(temporary, trainer.healthy)
        return True

    @staticmethod
    def aux_state_is_finite() -> bool:
        """Check non-checkpointed legacy MoE auxiliary registry entries."""
        from ultralytics.nn.modules.moe._common import MOE_LOSS_REGISTRY, _MOE_LOSS_REGISTRY_LOCK

        with _MOE_LOSS_REGISTRY_LOCK:
            entries = list(MOE_LOSS_REGISTRY.values())
        return TrainingRecoveryController.state_is_finite(entries)

    def bootstrap(self) -> None:
        """Create and globally acknowledge a finite pre-step recovery point."""
        trainer, healthy = self.trainer, True
        rank = self.rank()
        if rank in {-1, 0}:
            try:
                healthy = all(
                    self.state_is_finite(state)
                    for state in (unwrap_model(trainer.model), trainer.optimizer.state_dict(), trainer.scaler.state_dict())
                ) and trainer._save_healthy_checkpoint(trainer._serialize_checkpoint())
            except (OSError, RuntimeError, ValueError) as exc:
                LOGGER.warning(f"Initial healthy checkpoint creation failed: {exc}")
                healthy = False
        if rank != -1:
            backend = dist.get_backend()
            device = trainer.device if backend == "nccl" else torch.device("cpu")
            status = torch.tensor(int(healthy), dtype=torch.int32, device=device)
            dist.broadcast(status, src=0)
            healthy = bool(status.item())
        if not healthy:
            raise RuntimeError("Initial training state is nonfinite; refusing to start without a healthy recovery checkpoint.")

    def recover(self, epoch: int) -> bool:
        """Restore globally confirmed non-finite state from the latest healthy online snapshot."""
        trainer = self.trainer
        loss_nonfinite = bool(getattr(trainer, "_loss_nonfinite", False)) or (
            trainer.loss is not None and not bool(torch.isfinite(trainer.loss.detach()).all().item())
        )
        fitness_nonfinite = trainer.fitness is not None and not bool(torch.isfinite(torch.as_tensor(trainer.fitness)))
        gradient_nonfinite = bool(getattr(trainer, "_gradient_nonfinite", False))
        ema_nonfinite = bool(getattr(trainer, "_ema_nonfinite", False))
        flags = (loss_nonfinite, fitness_nonfinite, gradient_nonfinite, ema_nonfinite)
        rank = self.rank()
        if rank != -1 and dist.is_initialized():
            backend = dist.get_backend()
            device = trainer.device if backend == "nccl" else torch.device("cpu")
            shared = torch.tensor(flags, dtype=torch.int32, device=device)
            dist.all_reduce(shared, op=dist.ReduceOp.MAX)
            flags = tuple(bool(item) for item in shared.cpu().tolist())
        if not any(flags):
            return False
        reason = ", ".join(
            name for name, active in zip(("Loss NaN/Inf", "Fitness NaN/Inf", "Gradient NaN/Inf", "EMA NaN/Inf"), flags) if active
        )
        path = getattr(trainer, "healthy", None) or getattr(trainer, "last", None)
        payload = None
        if rank in {-1, 0} and path is not None and Path(path).exists():
            try:
                candidate = torch_load(path, map_location="cpu", weights_only=False)
                if self.state_is_finite(candidate):
                    payload = Path(path).read_bytes()
            except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError):
                payload = None
        if rank != -1 and dist.is_initialized():
            shared = [payload]
            dist.broadcast_object_list(shared, src=0)
            payload = shared[0]
        if payload is None:
            raise RuntimeError(f"Global nonfinite training state detected ({reason}) without a healthy recovery checkpoint.")

        trainer.nan_recovery_attempts += 1
        if trainer.nan_recovery_attempts > 3:
            raise RuntimeError(f"Training failed: NaN persisted for {trainer.nan_recovery_attempts} epochs")
        checkpoint = torch_load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        snapshot = checkpoint.get("model")
        if snapshot is None:
            raise RuntimeError("Healthy checkpoint lacks online model state; refusing to restore EMA with optimizer state.")
        trainer._model_train()
        target = unwrap_model(trainer.model)
        state = snapshot.float().state_dict()
        if getattr(target, "lora_enabled", False):
            from ultralytics.utils.lora import load_lora_compatible_state_dict

            load_lora_compatible_state_dict(target, state, context="NaN recovery model", adapter_only=True)
        else:
            target.load_state_dict(state, strict=False)

        scaler_state = None
        amp_recovery = bool(getattr(trainer, "amp", False)) and (flags[0] or flags[2])
        if not amp_recovery and (loss_nonfinite or gradient_nonfinite):
            scaler = getattr(trainer, "scaler", None)
            if scaler is not None:
                if loss_nonfinite and not gradient_nonfinite:
                    scaler.update(new_scale=max(scaler.get_scale() * 0.5, 1.0))
                scaler_state = deepcopy(scaler.state_dict())
        trainer._load_checkpoint_state(checkpoint)
        optimizer = getattr(trainer, "optimizer", None)
        if optimizer is not None:
            optimizer.zero_grad()
        self.reset_runtime(trainer.model)
        if amp_recovery:
            trainer.amp = False
            trainer.scaler = (
                torch.amp.GradScaler("cuda", enabled=False)
                if TORCH_2_4
                else torch.cuda.amp.GradScaler(enabled=False)
            )
        elif scaler_state is not None:
            trainer.scaler.load_state_dict(scaler_state)
        trainer._loss_nonfinite = trainer._gradient_nonfinite = trainer._ema_nonfinite = False
        trainer._nonfinite_diagnostic = None
        trainer.scheduler.last_epoch = epoch - 1
        return True
