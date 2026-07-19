"""Runtime lifecycle controller for LoRA, MoLoRA, and planner extensions."""

from __future__ import annotations

import math
from functools import partial

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model


def _hierarchical_hook(storage, key, module, inputs, output) -> None:
    """Store one intermediate feature tensor for few-shot distillation."""
    storage[key] = output.detach() if not torch.is_grad_enabled() else output


def update_args_with_lora_runtime_metadata(args, model) -> None:
    """Copy runtime LoRA metadata from the adapted model onto trainer args."""
    metadata = getattr(unwrap_model(model), "lora_runtime_metadata", {}) or {}
    mapping = {
        "requested_backend": "requested_lora_backend",
        "effective_backend": "effective_lora_backend",
        "requested_variant": "requested_lora_variant",
        "effective_variant": "effective_lora_variant",
        "peft_type": "effective_lora_type",
        "requested_init_lora_weights": "requested_lora_init_lora_weights",
        "effective_init_lora_weights": "effective_lora_init_lora_weights",
        "safety_profile": "lora_safety_profile",
        "safety_overrides": "lora_safety_overrides",
        "target_audit": "lora_target_audit",
    }
    for source, target in mapping.items():
        if source in metadata and metadata[source] not in (None, {}, []):
            setattr(args, target, metadata[source])
    decision = metadata.get("planner_decision")
    if decision:
        args.planner_decision = decision["status"]
        args.planner_predicted_delta = decision["predicted_delta"]
        args.planner_recommended_variant = decision["recommended_variant"]
        args.planner_recommended_rank = decision["recommended_rank"]
        args.planner_refusal_reason = decision["refusal_reason"]
        if decision["status"] == "ADAPT":
            args.lora_planner_adapted = True
        elif decision["status"] == "REFUSE":
            args.lora_planner_refused = True


def validate_adapter_configuration(args) -> None:
    """Reject mutually exclusive standard PEFT and MoLoRA requests."""
    peft_type = str(getattr(args, "lora_type", "lora") or "lora").lower()
    rankless = peft_type in {"boft", "oft", "ia3", "hra"}
    lora_enabled = (
        int(getattr(args, "lora_r", 0) or 0) > 0
        or float(getattr(args, "lora_auto_r_ratio", 0.0) or 0.0) > 0
        or rankless
    )
    molora_enabled = int(getattr(args, "molora_num_experts", 0) or 0) > 0
    if lora_enabled and molora_enabled:
        raise ValueError(
            "Standard LoRA and MoLoRA cannot be enabled in the same training run. "
            "Disable the selected lora_type (including rankless OFT/BOFT/IA3/HRA variants) "
            "or set molora_num_experts=0."
        )


class AdapterRuntimeController:
    """Own adapter construction, optimizer policy, and scheduled regularization."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.strategy = None
        self._alpha_prepared = False
        self.ortho_weight = 0.0
        self.ortho_frequency = 10
        self.ortho_batch_counter = 0
        self.optimizer_steps = 0

    def setup(self) -> None:
        validate_adapter_configuration(self.trainer.args)
        self._prepare_adalora_placeholder()
        self._apply_lora()
        self._apply_molora()
        self._setup_few_shot_teacher()

    @property
    def model(self):
        """Return the unwrapped training model."""
        return unwrap_model(self.trainer.model)

    @property
    def enabled(self) -> bool:
        """Return whether standard PEFT adapters are active."""
        return bool(getattr(self.model, "lora_enabled", False))

    @property
    def active(self) -> bool:
        """Return whether standard PEFT or MoLoRA adapters are active."""
        return self.enabled or bool(getattr(self.model, "molora_enabled", False))

    def _prepare_adalora_placeholder(self) -> None:
        """Give PEFT a valid bootstrap budget until dataloader iterations are known."""
        args = self.trainer.args
        if str(getattr(args, "lora_type", "lora") or "lora").lower() != "adalora":
            return
        if int(getattr(args, "lora_total_step", 0) or 0) <= 0:
            args.lora_total_step = 1
            self._adalora_total_step_pending = True

    def _apply_lora(self) -> None:
        from ultralytics.utils.lora import apply_lora

        args = self.trainer.args
        peft_type = str(getattr(args, "lora_type", "lora") or "lora").lower()
        enabled = (
            int(getattr(args, "lora_r", 0) or 0) > 0
            or float(getattr(args, "lora_auto_r_ratio", 0.0) or 0.0) > 0
            or peft_type in {"boft", "oft", "ia3", "hra"}
        )
        if not enabled:
            return
        self.trainer.model = apply_lora(self.trainer.model, args)
        update_args_with_lora_runtime_metadata(args, self.trainer.model)

    def _apply_molora(self) -> None:
        if int(getattr(self.trainer.args, "molora_num_experts", 0) or 0) <= 0:
            return
        from ultralytics.nn.peft.molora import MoLoRAConfig, get_peft_molora_model

        config = MoLoRAConfig.from_args(self.trainer.args)
        self.trainer.model = get_peft_molora_model(self.trainer.model, config)
        LOGGER.info(f"[MoLoRA] Initialized {config.num_experts} experts with top_k={config.top_k}")

    def _setup_few_shot_teacher(self) -> None:
        """Load the optional few-shot teacher without affecting native distillation."""
        args = self.trainer.args
        teacher_path = getattr(args, "lora_few_shot_teacher", None)
        if not getattr(args, "lora_few_shot_mode", False) or not teacher_path:
            return
        from ultralytics import YOLO

        teacher = YOLO(teacher_path).model.to(self.trainer.device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        self.trainer.teacher_model = teacher
        if getattr(args, "lora_few_shot_hierarchical_distill", False):
            self.init_hierarchical_distill_cache()

    def prepare_optimizer(self, iterations: int) -> None:
        """Finalize iteration-dependent adapter configuration before optimizer creation."""
        if str(getattr(self.trainer.args, "lora_type", "lora") or "lora").lower() != "adalora":
            return
        from ultralytics.utils.lora import resolve_adalora_total_step

        requested = None if getattr(self, "_adalora_total_step_pending", False) else getattr(
            self.trainer.args, "lora_total_step", None
        )
        total_step = resolve_adalora_total_step("adalora", requested, iterations)
        if total_step is None:
            raise ValueError("AdaLoRA requires a positive training iteration count.")
        self.trainer.args.lora_total_step = total_step
        for module in self.model.modules():
            configs = getattr(module, "peft_config", None)
            if not isinstance(configs, dict):
                continue
            for config in configs.values():
                if str(getattr(config, "peft_type", "")).lower().endswith("adalora"):
                    config.total_step = total_step
        self._adalora_total_step_pending = False
        LOGGER.info(f"[LoRA] AdaLoRA total_step resolved to {total_step}.")

    def configure_optimizer(self, optimizer) -> None:
        """Apply adapter strategies that require finalized optimizer parameter groups."""
        self.trainer.lora_strategy = None
        if not self.enabled:
            return
        from ultralytics.utils.lora import LoraTrainingStrategy

        args = self.trainer.args
        if self.strategy is None:
            self.strategy = LoraTrainingStrategy(
                model=self.trainer.model,
                config=getattr(self.model, "lora_config", None),
                epochs=self.trainer.epochs,
            )
        else:
            self.strategy.model = self.trainer.model
            self.strategy.epochs = self.trainer.epochs
        self.trainer.lora_strategy = self.strategy
        layer_decay = float(getattr(args, "lora_layer_decay", 0.0) or 0.0)
        if layer_decay > 0:
            self.strategy.apply_layer_decay_to_optimizer(optimizer, decay_rate=layer_decay)

        alpha_warmup = int(getattr(args, "lora_alpha_warmup", 0) or 0)
        if alpha_warmup > 0 and not self._alpha_prepared:
            if any(hasattr(module, "lora_A") for module in self.model.modules()):
                self.strategy.prepare_alpha_warmup()
                self._alpha_prepared = True
            else:
                LOGGER.info("[LoRA] Alpha warmup skipped: active adapter type has no LoRA alpha layers.")
                args.lora_alpha_warmup = 0

        self.ortho_weight = float(getattr(args, "lora_ortho_weight", 0.0) or 0.0)
        self.ortho_frequency = max(int(getattr(args, "lora_ortho_frequency", 10) or 10), 1)
        self.ortho_batch_counter = getattr(self, "ortho_batch_counter", 0)
        self.trainer.lora_ortho_weight = self.ortho_weight
        self.trainer.lora_ortho_frequency = self.ortho_frequency
        self.trainer.lora_ortho_batch_counter = 0

    def begin_epoch(self, epoch: int) -> None:
        """Advance alpha warmup and adapter dropout schedules."""
        if self.strategy is None:
            return
        args = self.trainer.args
        alpha_warmup = int(getattr(args, "lora_alpha_warmup", 0) or 0)
        if 0 <= epoch < alpha_warmup:
            self.strategy.step_alpha_warmup(epoch, warmup_epochs=alpha_warmup)
        elif alpha_warmup > 0 and epoch == alpha_warmup:
            self.strategy.finalize_alpha_warmup()
        self.strategy.update_dropout_schedule(
            self.trainer.model,
            epoch=epoch,
            epochs_total=self.trainer.epochs,
            start_dropout=float(getattr(args, "lora_dropout", 0.0) or 0.0),
            end_dropout=float(getattr(args, "lora_dropout_end", 0.15) or 0.15),
            schedule_start_ratio=float(getattr(args, "lora_dropout_start_ratio", 0.3) or 0.3),
        )

    def augment_loss(self, loss):
        """Add scheduled adapter regularization without changing native loss items."""
        if self.strategy is None or self.ortho_weight <= 0:
            return loss
        self.ortho_batch_counter += 1
        self.trainer.lora_ortho_batch_counter = self.ortho_batch_counter
        if self.ortho_batch_counter % self.ortho_frequency:
            return loss
        from ultralytics.utils.lora import LoraTrainingStrategy

        regularizer = LoraTrainingStrategy.compute_orthogonal_loss(
            self.trainer.model, weight=self.ortho_weight
        )
        return loss + regularizer.to(device=loss.device, dtype=loss.dtype)

    def after_optimizer_step(self) -> None:
        """Advance optimizer-step-dependent adapter schedules such as AdaLoRA rank allocation."""
        if not self.enabled:
            return
        self.optimizer_steps += 1
        for module in self.model.modules():
            update = getattr(module, "update_and_allocate", None)
            if callable(update):
                update(self.optimizer_steps)
                return

    def compute_prediction_entropy(self, predictions):
        """Compute normalized channel entropy for adaptive distillation temperature."""
        predictions = self._first_tensor(predictions)
        if predictions is None or predictions.ndim != 4:
            return torch.tensor(1.0, device=next(self.model.parameters()).device)
        probabilities = torch.nn.functional.softmax(predictions, dim=1)
        entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum(dim=1).mean(dim=(1, 2))
        return (entropy / max(math.log(probabilities.shape[1]), 1e-8)).mean()

    def compute_distillation_loss(self, student, teacher, adaptive_temp=False):
        """Compute generic tensor distillation without changing native DistillationModel behavior."""
        student, teacher = self._first_tensor(student), self._first_tensor(teacher)
        device = next(self.model.parameters()).device
        if student is None or teacher is None:
            return torch.tensor(0.0, device=device)
        temperature = (
            float(torch.clamp(2.0 + self.compute_prediction_entropy(teacher) * 4.0, 1.0, 8.0).item())
            if adaptive_temp
            else 4.0
        )
        if student.ndim == teacher.ndim == 4:
            if student.shape[2:] != teacher.shape[2:]:
                teacher = torch.nn.functional.interpolate(
                    teacher, size=student.shape[2:], mode="bilinear", align_corners=False
                )
            if student.shape[1] == teacher.shape[1]:
                return torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(student / temperature, dim=1),
                    torch.nn.functional.softmax(teacher / temperature, dim=1),
                    reduction="batchmean",
                ) * temperature**2
        if student.ndim == teacher.ndim == 3 and student.shape[-1] == teacher.shape[-1]:
            length = min(student.shape[1], teacher.shape[1])
            return torch.nn.functional.mse_loss(student[:, :length], teacher[:, :length])
        length = min(student.numel(), teacher.numel())
        return torch.nn.functional.mse_loss(student.flatten()[:length], teacher.flatten()[:length])

    def compute_response_distillation_loss(self, student, teacher):
        """Align shape-compatible detection response tensors."""
        device = next(self.model.parameters()).device
        students, teachers = self._tensor_list(student), self._tensor_list(teacher)
        losses = []
        for student_prediction in students:
            for teacher_prediction in teachers:
                if student_prediction.shape != teacher_prediction.shape:
                    continue
                if student_prediction.ndim == 3 and student_prediction.shape[-1] >= 6:
                    bbox_loss = torch.nn.functional.l1_loss(student_prediction[..., :4], teacher_prediction[..., :4])
                    classification_loss = torch.nn.functional.mse_loss(
                        student_prediction[..., 4:], teacher_prediction[..., 4:]
                    )
                    losses.append(bbox_loss + classification_loss)
                else:
                    losses.append(torch.nn.functional.mse_loss(student_prediction, teacher_prediction))
        return sum(losses) / len(losses) if losses else torch.tensor(0.0, device=device)

    def init_hierarchical_distill_cache(self):
        """Register persistent intermediate-feature hooks for few-shot distillation."""
        layers = getattr(self.trainer.args, "lora_few_shot_distill_layers", None)
        if not layers:
            self.trainer._hierarchical_cache = None
            return None
        student, teacher = self.model, getattr(self.trainer, "teacher_model", None)
        cache = {
            "student_features": {},
            "teacher_features": {},
            "student_hooks": [],
            "teacher_hooks": [],
            "layer_indices": list(layers),
        }
        for index in layers:
            if hasattr(student, "model") and index < len(student.model):
                cache["student_hooks"].append(
                    student.model[index].register_forward_hook(partial(_hierarchical_hook, cache["student_features"], index))
                )
            if teacher is not None and hasattr(teacher, "model") and index < len(teacher.model):
                cache["teacher_hooks"].append(
                    teacher.model[index].register_forward_hook(partial(_hierarchical_hook, cache["teacher_features"], index))
                )
        self.trainer._hierarchical_cache = cache
        return cache

    def compute_hierarchical_distillation_loss(self, images, layer_indices):
        """Compute attention-transfer loss for cached intermediate feature pairs."""
        if not layer_indices:
            return torch.tensor(0.0, device=images.device)
        cache = getattr(self.trainer, "_hierarchical_cache", None) or self.init_hierarchical_distill_cache()
        teacher = getattr(self.trainer, "teacher_model", None)
        if cache is None or teacher is None:
            return torch.tensor(0.0, device=images.device)
        cache["student_features"].clear()
        cache["teacher_features"].clear()
        with torch.no_grad():
            self.model(images)
            teacher(images)
        losses = []
        for index in layer_indices:
            student_feature = cache["student_features"].get(index)
            teacher_feature = cache["teacher_features"].get(index)
            if student_feature is None or teacher_feature is None or student_feature.ndim != teacher_feature.ndim:
                continue
            if student_feature.ndim == 4 and student_feature.shape[2:] != teacher_feature.shape[2:]:
                teacher_feature = torch.nn.functional.interpolate(
                    teacher_feature, size=student_feature.shape[2:], mode="bilinear", align_corners=False
                )
            if student_feature.shape[1] != teacher_feature.shape[1]:
                continue
            student_attention = torch.abs(student_feature).sum(dim=1, keepdim=True)
            teacher_attention = torch.abs(teacher_feature).sum(dim=1, keepdim=True)
            student_attention = student_attention / (student_attention.norm(2, dim=(2, 3), keepdim=True) + 1e-8)
            teacher_attention = teacher_attention / (teacher_attention.norm(2, dim=(2, 3), keepdim=True) + 1e-8)
            losses.append(torch.nn.functional.mse_loss(student_attention, teacher_attention))
        return sum(losses) / len(losses) if losses else torch.tensor(0.0, device=images.device)

    def augment_few_shot_loss(self, loss, images, epoch):
        """Add opt-in few-shot teacher distillation to the native task loss."""
        args = self.trainer.args
        teacher = getattr(self.trainer, "teacher_model", None)
        if not getattr(args, "lora_few_shot_mode", False) or teacher is None:
            return loss
        student_predictions = self.trainer.model(images)
        with torch.no_grad():
            teacher_predictions = teacher(images)
        distillation = self.compute_distillation_loss(
            student_predictions,
            teacher_predictions,
            adaptive_temp=getattr(args, "lora_few_shot_adaptive_temperature", False),
        )
        if getattr(args, "lora_few_shot_response_distill", False):
            distillation += float(getattr(args, "lora_few_shot_response_distill_weight", 0.3)) * (
                self.compute_response_distillation_loss(student_predictions, teacher_predictions)
            )
        layers = getattr(args, "lora_few_shot_distill_layers", None)
        if getattr(args, "lora_few_shot_hierarchical_distill", False) and layers:
            distillation += 0.3 * self.compute_hierarchical_distillation_loss(images, layers)
        progress = epoch / max(self.trainer.epochs - 1, 1)
        maximum = float(getattr(args, "lora_few_shot_distill_weight_max", 1.0))
        minimum = float(getattr(args, "lora_few_shot_distill_weight_min", 0.1))
        schedule = getattr(args, "lora_few_shot_distill_schedule", "cosine")
        if schedule == "linear":
            weight = maximum - (maximum - minimum) * progress
        elif schedule == "exponential":
            weight = minimum + (maximum - minimum) * math.exp(-5 * progress)
        elif schedule == "constant":
            weight = float(getattr(args, "lora_few_shot_distill_weight", maximum))
        else:
            weight = minimum + (maximum - minimum) * 0.5 * (1 + math.cos(math.pi * progress))
        return loss + weight * distillation.to(device=loss.device, dtype=loss.dtype)

    @staticmethod
    def _tensor_list(value):
        if isinstance(value, torch.Tensor):
            return [value]
        if isinstance(value, dict):
            return [item for item in value.values() if isinstance(item, torch.Tensor)]
        if isinstance(value, (list, tuple)):
            tensors = []
            for item in value:
                tensors.extend(AdapterRuntimeController._tensor_list(item))
            return tensors
        return []

    @staticmethod
    def _first_tensor(value):
        tensors = AdapterRuntimeController._tensor_list(value)
        return tensors[0] if tensors else None
