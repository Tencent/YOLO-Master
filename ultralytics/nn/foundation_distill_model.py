"""Training-only Foundation Teacher distillation wrapper.

This module deliberately lives beside the existing YOLO-to-YOLO ``DistillationModel``.  A Foundation Teacher exposes
an image encoder rather than a YOLO graph, so sharing the latter's hook and checkpoint assumptions would make the
existing distillation path unnecessarily fragile.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.foundation import (
    DEFAULT_DINOV3_MODEL,
    DEFAULT_SIGLIP2_MODEL,
    DINOv3Teacher,
    FoundationFeatures,
    MultiFoundationTeacher,
    P4AlignmentProjector,
    SigLIP2Teacher,
    StudentFeatureTap,
    cosine_kd_loss,
    foreground_token_weights,
    relational_kd_loss,
    RegionSemanticProjector,
    positive_region_pool,
    semantic_distillation_loss,
)
from ultralytics.nn.modules.routing_protocol import publish_aux_loss
from ultralytics.nn.foundation.routing import (
    FoundationTeacherRouter,
    foundation_multiteacher_summary,
    foundation_teacher_summary,
    routing_kd_loss,
)


def _get(config: Any, name: str, default: Any = None) -> Any:
    """Read a setting from a mapping, namespace, or object."""
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _as_probe_size(value: Any, stride: int = 32) -> int:
    """Resolve a small, stride-compatible dry-run image size for lazy channel discovery."""
    if isinstance(value, (list, tuple)):
        value = max(value)
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = 64
    # A 64x64 probe is enough to discover channels and avoids an expensive full-resolution teacher pass.
    value = max(32, min(value, 128))
    return max(stride, (value + stride - 1) // stride * stride)


def _teacher_device(request: Any, student_device: torch.device | None) -> torch.device | str:
    """Resolve the integer device spelling accepted by the public YAML configuration."""
    if request in (None, "auto"):
        return student_device or "cpu"
    if isinstance(request, int):
        return f"cuda:{request}" if torch.cuda.is_available() else "cpu"
    return request


def _dense_p4(features: FoundationFeatures | Mapping[str, Any]) -> torch.Tensor:
    """Extract and validate the P4 dense feature from a teacher response."""
    dense = features.dense if isinstance(features, FoundationFeatures) else features.get("dense")
    if not isinstance(dense, Mapping) or "p4" not in dense:
        raise KeyError("Foundation Teacher output must contain dense['p4'] for F06 distillation.")
    feature = dense["p4"]
    if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
        raise ValueError(
            f"Foundation Teacher dense['p4'] must be a BCHW tensor, got {getattr(feature, 'shape', None)}."
        )
    if not torch.isfinite(feature).all():
        raise ValueError("Foundation Teacher dense['p4'] contains NaN or Inf values.")
    return feature


class FoundationDistillationModel(nn.Module):
    """Training-only wrapper that adds Foundation Teacher KD to a YOLO student.

    The teacher is intentionally kept outside PyTorch's registered module tree.  It is frozen, never placed in the
    optimizer or DDP graph, and therefore cannot enter a student checkpoint through ``state_dict``.  ``eval`` and
    tensor prediction paths are transparent and execute only the student.

    Args:
        student_model (nn.Module): Trainable YOLO model.
        teacher_manager (FoundationTeacher): Frozen image encoder implementing ``encode``.
        config (Mapping | object): Foundation settings, normally the trainer's ``args`` namespace.
    """

    def __init__(self, student_model: nn.Module, teacher_manager: Any | None, config: Any) -> None:
        super().__init__()
        if not isinstance(student_model, nn.Module):
            raise TypeError(f"student_model must be an nn.Module, got {type(student_model).__name__}.")
        self.student_model = student_model
        self.config = config
        self.loss_weight = float(_get(config, "foundation_loss_weight", 0.0) or 0.0)
        if self.loss_weight < 0:
            raise ValueError("foundation_loss_weight must be non-negative.")
        # Weight schedule: "constant" (legacy behavior) or "gate_decay" (cosine-gated ramp-in + late decay).
        self.weight_schedule = str(_get(config, "foundation_weight_schedule", "constant") or "constant").lower()
        if self.weight_schedule not in {"constant", "gate_decay"}:
            raise ValueError(f"Unsupported foundation_weight_schedule={self.weight_schedule!r}.")
        gate_cosine = _get(config, "foundation_gate_cosine", 1.0)
        self.gate_cosine = float(1.0 if gate_cosine is None else gate_cosine)
        gate_cosine_low = _get(config, "foundation_gate_cosine_low", 0.9)
        self.gate_cosine_low = None if not gate_cosine_low else float(gate_cosine_low)
        if self.gate_cosine_low is not None and not 0.0 < self.gate_cosine_low < self.gate_cosine:
            raise ValueError("foundation_gate_cosine_low must be in (0, foundation_gate_cosine), or 0 to disable.")
        gate_width = _get(config, "foundation_gate_width", 0.05)
        self.gate_width = float(0.05 if gate_width is None else gate_width)
        if self.gate_width <= 0:
            raise ValueError("foundation_gate_width must be positive.")
        decay_start = _get(config, "foundation_decay_start", 0.7)
        self.decay_start = float(0.7 if decay_start is None else decay_start)
        if not 0.0 <= self.decay_start < 1.0:
            raise ValueError("foundation_decay_start must be in [0, 1).")
        warmup_floor = _get(config, "foundation_warmup_floor", 0.2)
        self.warmup_floor = float(0.2 if warmup_floor is None else warmup_floor)
        if not 0.0 <= self.warmup_floor <= 1.0:
            raise ValueError("foundation_warmup_floor must be in [0, 1].")
        gate_ema = _get(config, "foundation_gate_ema", 0.9)
        self.gate_ema = float(0.9 if gate_ema is None else gate_ema)
        if not 0.0 <= self.gate_ema < 1.0:
            raise ValueError("foundation_gate_ema must be in [0, 1).")
        self.__dict__["_cosine_ema"] = None
        self.__dict__["_train_progress"] = 0.0
        self.router_distill = bool(_get(config, "foundation_router_distill", False))
        self.router_loss_weight = float(_get(config, "foundation_router_loss_weight", 0.0) or 0.0)
        if self.router_loss_weight < 0:
            raise ValueError("foundation_router_loss_weight must be non-negative.")
        router_temperature = _get(config, "foundation_router_temperature", 1.0)
        self.router_temperature = float(1.0 if router_temperature is None else router_temperature)
        if self.router_temperature <= 0:
            raise ValueError("foundation_router_temperature must be positive.")
        configured_teacher_name = _get(config, "foundation_teacher", None)
        if configured_teacher_name in (None, "none") and teacher_manager is not None:
            configured_teacher_name = getattr(teacher_manager, "name", configured_teacher_name)
        self.foundation_teacher_name = str(configured_teacher_name or "none").lower()
        configured_route_teachers = _get(config, "foundation_router_teachers", ("dinov3", "siglip2"))
        if isinstance(configured_route_teachers, str):
            configured_route_teachers = (configured_route_teachers,)
        self.router_teachers = tuple(str(name).lower() for name in (configured_route_teachers or ()))
        self.router_native_state = bool(_get(config, "foundation_router_native_state", True))
        if self.foundation_teacher_name == "multi" and self.router_teachers != ("dinov3", "siglip2"):
            raise ValueError("F14 requires foundation_router_teachers=['dinov3', 'siglip2'] in that order.")
        self._router_enabled = self.router_distill and self.router_loss_weight > 0.0
        self.semantic_distill = bool(_get(config, "foundation_semantic_distill", False))
        self.semantic_loss_weight = float(_get(config, "foundation_semantic_loss_weight", 0.0) or 0.0)
        self.semantic_text_weight = float(_get(config, "foundation_semantic_text_weight", 1.0) or 0.0)
        self.semantic_image_weight = float(_get(config, "foundation_semantic_image_weight", 1.0) or 0.0)
        self.semantic_temperature = float(_get(config, "foundation_semantic_temperature", 0.07) or 0.0)
        if min(self.semantic_loss_weight, self.semantic_text_weight, self.semantic_image_weight) < 0:
            raise ValueError("Foundation semantic loss weights must be non-negative.")
        if self.semantic_temperature <= 0:
            raise ValueError("foundation_semantic_temperature must be positive.")
        self._semantic_enabled = self.semantic_distill and self.semantic_loss_weight > 0.0
        # F15 is deliberately explicit.  Existing Foundation runs, including a
        # MultiTask model used without the F15 flag, retain the historical
        # single Foundation metrics and loss contract.
        self.multitask_enabled = bool(
            _get(config, "foundation_multitask", False) or _get(config, "foundation_multitask_enabled", False)
        )
        multitask_threshold = _get(config, "foundation_multitask_negative_transfer_threshold", 4.0)
        self.multitask_negative_transfer_threshold = float(4.0 if multitask_threshold is None else multitask_threshold)
        if self.multitask_negative_transfer_threshold <= 0:
            raise ValueError("foundation_multitask_negative_transfer_threshold must be positive.")
        self._disabled = (
            self.loss_weight <= 0 and not self._router_enabled and not self._semantic_enabled
        ) or teacher_manager is None
        self.__dict__["_student_only"] = False
        # Bypass nn.Module.__setattr__: teacher state must never be part of parameters/state_dict/DDP/EMA.
        self.__dict__["_teacher_manager"] = None if self._disabled else teacher_manager
        self.__dict__["_tap"] = None
        self.__dict__["_taps"] = {}
        self.__dict__["_projector"] = None
        self.__dict__["_semantic_tap"] = None
        self.__dict__["_semantic_projector"] = None
        self.__dict__["_semantic_prompts"] = None
        self.__dict__["_semantic_text_cache"] = None
        self.__dict__["_target_levels"] = ("p4",)
        self.__dict__["_multiscale"] = False
        self.__dict__["_route_teachers"] = {}
        self.__dict__["_route_specs"] = []
        self.__dict__["_router_teacher_dim"] = None
        self.__dict__["_last_foundation_loss"] = torch.zeros((), dtype=torch.float32)
        self.__dict__["_last_foundation_metrics"] = {}
        self.__dict__["_multitask_active_tasks"] = ()
        self.__dict__["_multitask_expected_tasks"] = ()

        if self._disabled:
            return
        if self.multitask_enabled:
            self._validate_multitask_contract()
        freeze = getattr(teacher_manager, "freeze", None)
        if callable(freeze):
            freeze()
        else:
            train = getattr(teacher_manager, "train", None)
            if callable(train):
                train(False)
            if isinstance(teacher_manager, nn.Module):
                teacher_manager.requires_grad_(False)
        target_levels = tuple(str(x).lower() for x in (_get(config, "foundation_target_levels", ("p4",)) or ()))
        multiscale = bool(_get(config, "foundation_multiscale", False))
        if not multiscale and target_levels != ("p4",):
            raise ValueError(
                "F06 currently supports exactly foundation_target_levels=['p4']; enable foundation_multiscale for F10."
            )
        if multiscale and len(target_levels) < 2:
            raise ValueError("F10 foundation_multiscale requires at least two target levels.")
        self.__dict__["_target_levels"] = target_levels
        self.__dict__["_multiscale"] = multiscale
        self.__dict__["_taps"] = {level: StudentFeatureTap(student_model, target=level) for level in target_levels}
        self.__dict__["_tap"] = self.__dict__["_taps"].get("p4") or next(iter(self.__dict__["_taps"].values()))
        # Register projectors as wrapper-owned trainable modules. The teacher remains deliberately unregistered above.
        if multiscale:
            self._projectors = self._build_projectors()
        else:
            self._projector = self._build_projector()
        if self._semantic_enabled:
            self._build_semantic_components()

    @property
    def teacher_manager(self):
        """Return the unregistered, frozen teacher backend (or ``None`` on the disabled path)."""
        return self.__dict__.get("_teacher_manager")

    @property
    def student(self):
        """Compatibility alias matching the concise wrapper naming used by integration callers."""
        return self.student_model

    @property
    def tap(self):
        """Return the student P4 tap, if Foundation KD is active."""
        return self.__dict__.get("_tap")

    @property
    def taps(self) -> dict[str, StudentFeatureTap]:
        """Return active student feature taps keyed by P-level name."""
        return dict(self.__dict__.get("_taps", {}))

    @property
    def target_levels(self) -> tuple[str, ...]:
        """Return the configured Foundation distillation levels."""
        return tuple(self.__dict__.get("_target_levels", ("p4",)))

    @property
    def multiscale(self) -> bool:
        """Return whether F10 multi-scale distillation is active."""
        return bool(self.__dict__.get("_multiscale", False))

    @property
    def multitask_active_tasks(self) -> tuple[str, ...]:
        """Return the F15 task set captured from the student's existing MultiTask head."""
        return tuple(self.__dict__.get("_multitask_active_tasks", ()))

    @property
    def projector(self):
        """Return the trainable alignment projector(s), if Foundation KD is active."""
        return self._modules.get("_projectors") or self._modules.get("_projector")

    @property
    def semantic_projector(self):
        """Return the F13 region projector, if semantic distillation is active."""
        return self._modules.get("_semantic_projector")

    def projector_for(self, level: str) -> P4AlignmentProjector:
        """Return the alignment projector assigned to one P-level."""
        level = str(level).lower()
        if self.multiscale:
            projectors = self._modules.get("_projectors")
            if projectors is None or level not in projectors:
                raise KeyError(f"No Foundation projector configured for {level!r}.")
            return projectors[level]
        if level != "p4":
            raise KeyError("Single-scale Foundation KD only has a p4 projector.")
        projector = self._modules.get("_projector")
        if projector is None:
            raise KeyError("Foundation p4 projector is not initialized.")
        return projector

    @property
    def last_foundation_loss(self) -> torch.Tensor:
        """Return the detached scalar KD loss from the most recent forward."""
        return self.__dict__.get("_last_foundation_loss", torch.zeros(())).detach()

    def foundation_metrics(self) -> dict[str, float]:
        """Return a copy of the most recent scalar Foundation training metrics."""
        return dict(self.__dict__.get("_last_foundation_metrics", {}))

    def multitask_metrics(self) -> dict[str, float]:
        """Return only the F15 task-level observations from the latest training batch."""
        return {
            key: value
            for key, value in self.__dict__.get("_last_foundation_metrics", {}).items()
            if key.startswith("foundation_multitask_")
        }

    def reset_foundation_metrics(self) -> None:
        """Clear the most recent Foundation metrics so eval or skipped batches cannot reuse stale values."""
        self.__dict__["_last_foundation_metrics"] = {}

    def has_foundation_metrics(self) -> bool:
        """Return whether a valid Foundation metric snapshot is available."""
        return bool(self.__dict__.get("_last_foundation_metrics", {}))

    def _resolve_multitask_tasks(self) -> tuple[str, ...]:
        """Read active tasks without changing the student's TaskRouter semantics."""
        student = self.student_model
        tasks = getattr(student, "active_tasks", None)
        if tasks is None:
            head = getattr(student, "model", None)
            head = head[-1] if isinstance(head, (list, tuple, nn.ModuleList)) and head else None
            tasks = getattr(head, "active_tasks", None)
            if tasks is None:
                tasks = getattr(head, "_active_tasks", None)
        if isinstance(tasks, str):
            tasks = (tasks,)
        if not isinstance(tasks, (list, tuple, set, frozenset)):
            return ()
        return tuple(sorted({str(task).lower() for task in tasks}))

    def _validate_multitask_contract(self) -> None:
        """Validate the F15 boundary while leaving TaskRouter as an existing local router."""
        task_mode = _get(self.config, "task", None)
        if task_mode is not None and str(task_mode).lower() != "multitask":
            raise ValueError("F15 foundation_multitask=True requires task='multitask'.")
        active = self._resolve_multitask_tasks()
        if len(active) < 2:
            raise ValueError("F15 Foundation MultiTask distillation requires at least two active visual tasks.")
        unsupported = sorted(
            set(active).difference({"detect", "segment", "pose", "classify", "depth", "normal", "semantic"})
        )
        if unsupported:
            raise ValueError(
                "F15 MultiTask Foundation currently supports only loss-backed visual tasks; "
                f"unsupported active tasks={unsupported}."
            )
        requested = _get(self.config, "foundation_multitask_tasks", None)
        if requested:
            if len(set(requested)) != len(requested):
                raise ValueError("foundation_multitask_tasks must not contain duplicate task names.")
            expected = tuple(sorted({str(task).lower() for task in requested}))
            if len(expected) < 2:
                raise ValueError("foundation_multitask_tasks must contain at least two active tasks for F15.")
            if set(expected) != set(active):
                raise ValueError(
                    "foundation_multitask_tasks does not match the student's active tasks: "
                    f"configured={list(expected)}, model={list(active)}."
                )
        else:
            expected = active
        self.__dict__["_multitask_active_tasks"] = active
        self.__dict__["_multitask_expected_tasks"] = expected

    @staticmethod
    def _task_loss_snapshot(task_items: Any) -> dict[str, float]:
        """Map the stable nine-item MultiTaskLoss vector to task-level scalar diagnostics."""
        if isinstance(task_items, torch.Tensor):
            values = task_items.detach().float().reshape(-1).cpu().tolist()
        elif isinstance(task_items, (list, tuple)):
            values = [
                float(value.detach().float().item()) if isinstance(value, torch.Tensor) else float(value)
                for value in task_items
            ]
        else:
            values = []
        values = (values + [0.0] * 9)[:9]
        return {
            "detect": max(0.0, float(sum(values[:3]))),
            "segment": max(0.0, float(values[3])),
            "pose": max(0.0, float(values[4])),
            "classify": max(0.0, float(values[5])),
            "depth": max(0.0, float(values[6])),
            "normal": max(0.0, float(values[7])),
            "semantic": max(0.0, float(values[8])),
        }

    def _multitask_metrics(
        self, task_loss: torch.Tensor, task_items: Any, foundation_loss: torch.Tensor, preds: Any
    ) -> dict[str, float]:
        """Collect F15 diagnostics for transfer, imbalance, and router conflict.

        These values are observations only.  In particular, the method never
        changes task weights or selects a task expert, so the pre-existing
        TaskRouter remains a task-local router rather than a global selector.
        """
        if not self.multitask_enabled:
            return {}
        active = self.multitask_active_tasks or self._resolve_multitask_tasks()
        losses = self._task_loss_snapshot(task_items)
        active_losses = {task: losses.get(task, 0.0) for task in active if task in losses}
        positive = [value for value in active_losses.values() if value > 1e-8]
        task_total = float(task_loss.detach().float().mean().item())
        foundation_value = float(foundation_loss.detach().float().item())
        imbalance = (max(positive) / max(min(positive), 1e-8)) if positive else 0.0
        metrics = {
            "foundation_multitask_enabled": 1.0,
            "foundation_multitask_active_tasks": float(len(active)),
            "foundation_multitask_supervised_tasks": float(len(positive)),
            "foundation_multitask_task_loss": task_total,
            "foundation_multitask_task_loss_imbalance": float(imbalance),
            "foundation_multitask_loss_imbalance": float(imbalance),
            "foundation_multitask_negative_transfer_threshold": float(self.multitask_negative_transfer_threshold),
            "foundation_multitask_negative_transfer_risk": float(
                imbalance >= self.multitask_negative_transfer_threshold if len(positive) >= 2 else 0.0
            ),
            "foundation_multitask_negative_transfer": float(
                imbalance >= self.multitask_negative_transfer_threshold if len(positive) >= 2 else 0.0
            ),
            "foundation_multitask_representation_transfer_ready": float(len(positive) >= 2),
            "foundation_multitask_representation_prior_loss": foundation_value,
            "foundation_multitask_shared_kd_loss": foundation_value,
            "foundation_multitask_foundation_task_ratio": foundation_value / max(task_total, 1e-8),
        }
        for task in active:
            metrics[f"foundation_multitask_task_loss_{task}"] = float(active_losses.get(task, 0.0))

        raw = preds
        if isinstance(raw, tuple):
            raw = raw[1] if len(raw) > 1 and isinstance(raw[1], dict) else {}
        if isinstance(raw, dict) and "one2many" in raw:
            raw = raw["one2many"]
        stats = raw.get("routing_stats") if isinstance(raw, dict) else None
        if isinstance(stats, Mapping):
            entropy = stats.get("entropy")
            if entropy is not None:
                metrics["foundation_multitask_task_router_entropy"] = float(entropy)
            usage = stats.get("task_usage")
            if isinstance(usage, torch.Tensor):
                usage = usage.detach().float().reshape(-1).cpu().tolist()
            if isinstance(usage, (list, tuple)):
                model_graph = getattr(self.student_model, "model", None)
                head = (
                    model_graph[-1] if isinstance(model_graph, (list, tuple, nn.ModuleList)) and model_graph else None
                )
                names = getattr(head, "_task_router_names", active)
                for name, value in zip(names, usage):
                    metrics[f"foundation_multitask_task_router_usage_{str(name).lower()}"] = float(value)
            if entropy is not None:
                foundation_entropy = self.__dict__.get("_last_foundation_metrics", {}).get(
                    "foundation_router_student_entropy"
                )
                conflict = abs(float(entropy) - float(foundation_entropy)) if foundation_entropy is not None else 0.0
                metrics["foundation_multitask_task_foundation_router_conflict"] = conflict
                metrics["foundation_multitask_router_conflict"] = conflict
        if "foundation_router_teacher_entropy" in self.__dict__.get("_last_foundation_metrics", {}):
            teacher_entropy = self.__dict__["_last_foundation_metrics"]["foundation_router_teacher_entropy"]
            student_entropy = self.__dict__["_last_foundation_metrics"].get("foundation_router_student_entropy")
            if student_entropy is not None:
                metrics["foundation_multitask_foundation_router_entropy_gap"] = abs(
                    float(teacher_entropy) - float(student_entropy)
                )
        return metrics

    def _latent_route_modules(self) -> list[tuple[str, nn.Module]]:
        """Return the F11-supported LatentMixture image-level routers only."""

        modules = []
        for name, module in self.student_model.named_modules():
            if module.__class__.__name__ != "LatentMixture":
                continue
            if not hasattr(module, "routing_logits") or not hasattr(module, "routing_summary"):
                continue
            modules.append((name or "root", module))
        return modules

    @staticmethod
    def _route_seed(index: int, student_dim: int, teacher_dim: int, num_experts: int) -> int:
        """Derive a stable seed so route targets survive checkpoint reconstruction."""

        return int(1101 + index * 1009 + student_dim * 17 + teacher_dim * 3 + num_experts * 7)

    def _ensure_route_teachers(
        self, teacher_summary: torch.Tensor, routes: list[tuple[str, nn.Module]]
    ) -> dict[str, FoundationTeacherRouter]:
        """Create frozen, unregistered route heads for the current student graph."""

        route_teachers = self.__dict__.setdefault("_route_teachers", {})
        self.__dict__["_router_teacher_dim"] = int(teacher_summary.shape[1])
        specs = []
        for index, (name, module) in enumerate(routes):
            summary = getattr(module, "routing_summary", None)
            logits = getattr(module, "routing_logits", None)
            if not isinstance(summary, torch.Tensor) or not isinstance(logits, torch.Tensor):
                continue
            if summary.ndim != 2 or logits.ndim != 2:
                continue
            student_dim, num_experts = int(summary.shape[1]), int(logits.shape[1])
            seed = self._route_seed(index, student_dim, int(teacher_summary.shape[1]), num_experts)
            current = route_teachers.get(name)
            if (
                current is None
                or current.student_dim != student_dim
                or current.teacher_dim != int(teacher_summary.shape[1])
                or current.num_experts != num_experts
            ):
                current = FoundationTeacherRouter(
                    student_dim,
                    int(teacher_summary.shape[1]),
                    num_experts,
                    seed=seed,
                ).to(device=summary.device)
                route_teachers[name] = current
            else:
                current.to(device=summary.device)
            current.eval()
            current.requires_grad_(False)
            specs.append(
                {
                    "name": name,
                    "student_dim": student_dim,
                    "teacher_dim": int(teacher_summary.shape[1]),
                    "num_experts": num_experts,
                    "hidden_dim": int(current.hidden_dim),
                    "seed": int(current.seed),
                }
            )
        self.__dict__["_route_specs"] = specs
        return route_teachers

    def _routing_kd(self, teacher_summary: torch.Tensor, *, batch_size: int) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute F11 route KD over LatentMixture modules and publish one aux record."""

        routes = self._latent_route_modules()
        if not self._router_enabled:
            zero = teacher_summary.sum() * 0.0
            return zero, {}
        if not routes:
            zero = teacher_summary.sum() * 0.0
            return zero, {
                "foundation_router_loss": 0.0,
                "foundation_router_kl": 0.0,
                "foundation_router_modules": 0.0,
                "foundation_router_teacher_entropy": 0.0,
                "foundation_router_student_entropy": 0.0,
            }
        route_teachers = self._ensure_route_teachers(teacher_summary, routes)
        losses = []
        teacher_entropies = []
        student_entropies = []
        for name, module in routes:
            student_logits = getattr(module, "routing_logits", None)
            student_summary = getattr(module, "routing_summary", None)
            teacher_router = route_teachers.get(name)
            if (
                teacher_router is None
                or not isinstance(student_logits, torch.Tensor)
                or not isinstance(student_summary, torch.Tensor)
                or student_logits.ndim != 2
                or student_summary.ndim != 2
            ):
                continue
            native_summary = (
                student_summary.detach() if self.router_native_state else torch.zeros_like(student_summary.detach())
            )
            with torch.inference_mode():
                teacher_logits = teacher_router(native_summary, teacher_summary.to(device=student_summary.device))
            loss = routing_kd_loss(student_logits, teacher_logits, temperature=self.router_temperature)
            losses.append(loss)
            with torch.no_grad():
                teacher_probs = F.softmax(teacher_logits.float() / self.router_temperature, dim=-1)
                student_probs = F.softmax(student_logits.detach().float() / self.router_temperature, dim=-1)
                teacher_entropies.append(
                    float((-(teacher_probs * teacher_probs.clamp_min(1e-12).log()).sum(-1).mean()))
                )
                student_entropies.append(
                    float((-(student_probs * student_probs.clamp_min(1e-12).log()).sum(-1).mean()))
                )
        if not losses:
            zero = teacher_summary.sum() * 0.0
            return zero, {
                "foundation_router_loss": 0.0,
                "foundation_router_kl": 0.0,
                "foundation_router_modules": 0.0,
                "foundation_router_teacher_entropy": 0.0,
                "foundation_router_student_entropy": 0.0,
            }
        route_kl = torch.stack(losses).mean()
        route_loss = route_kl * self.router_loss_weight * int(batch_size)
        if not torch.isfinite(route_loss):
            raise ValueError("Foundation routing distillation loss is NaN or Inf.")
        for _, module in routes:
            # The route loss now owns the autograd graph.  Do not retain a
            # second graph reference on every LatentMixture across batches.
            for attr in ("_last_routing_logits", "_last_routing_probs", "_last_routing_summary"):
                if hasattr(module, attr):
                    setattr(module, attr, None)
        # The wrapper owns this publication.  It is intentionally not included
        # in the legacy mixture collector, avoiding a second addition to loss.
        publish_aux_loss(self, route_loss, kind="foundation_route", training=self.training)
        return route_loss, {
            "foundation_router_loss": float(route_loss.detach().float().item()),
            "foundation_router_kl": float(route_kl.detach().float().item()),
            "foundation_router_modules": float(len(losses)),
            "foundation_router_teacher_entropy": sum(teacher_entropies) / len(teacher_entropies),
            "foundation_router_student_entropy": sum(student_entropies) / len(student_entropies),
        }

    def _build_projector(self, level: str = "p4", teacher_feature: torch.Tensor | None = None) -> P4AlignmentProjector:
        """Discover one student/teacher channel pair with a no-grad, small-image dry run."""
        tap = self.taps[level]
        teacher = self.teacher_manager
        device = next(self.student_model.parameters(), torch.empty(0)).device
        yaml = getattr(self.student_model, "yaml", {}) or {}
        ch = int(yaml.get("channels", 3)) if isinstance(yaml, Mapping) else int(getattr(yaml, "channels", 3))
        stride = max(1, int(torch.as_tensor(getattr(self.student_model, "stride", 32)).max().item()))
        size = _as_probe_size(_get(self.config, "imgsz", 64), stride)
        was_training = self.student_model.training
        self.student_model.eval()
        tap.clear()
        with torch.inference_mode():
            self.student_model(torch.zeros(2, ch, size, size, device=device))
            student_feature = tap.feature
            if teacher_feature is None:
                teacher_features = teacher.encode(torch.zeros(2, 3, size, size, device=device))
                teacher_feature = _dense_p4(teacher_features)
        self.student_model.train(was_training)
        if student_feature.shape[1] <= 0 or teacher_feature.shape[1] <= 0:
            raise ValueError("Foundation projector channel dimensions must be positive.")
        return P4AlignmentProjector(
            student_channels=int(student_feature.shape[1]),
            teacher_channels=int(teacher_feature.shape[1]),
            align_dim=int(_get(self.config, "foundation_align_dim", 256)),
        ).to(device=device)

    def _build_projectors(self) -> nn.ModuleDict:
        """Discover and build independent F10 adapters for each requested P-level."""
        device = next(self.student_model.parameters(), torch.empty(0)).device
        yaml = getattr(self.student_model, "yaml", {}) or {}
        ch = int(yaml.get("channels", 3)) if isinstance(yaml, Mapping) else int(getattr(yaml, "channels", 3))
        stride = max(1, int(torch.as_tensor(getattr(self.student_model, "stride", 32)).max().item()))
        size = _as_probe_size(_get(self.config, "imgsz", 64), stride)
        was_training = self.student_model.training
        self.student_model.eval()
        for tap in self.taps.values():
            tap.clear()
        with torch.inference_mode():
            self.student_model(torch.zeros(2, ch, size, size, device=device))
            teacher_features = self.teacher_manager.encode(torch.zeros(2, 3, size, size, device=device))
            teacher_feature = _dense_p4(teacher_features)
        self.student_model.train(was_training)
        projectors = nn.ModuleDict()
        for level in self.target_levels:
            student_feature = self.taps[level].feature
            if student_feature.shape[1] <= 0 or teacher_feature.shape[1] <= 0:
                raise ValueError("Foundation projector channel dimensions must be positive.")
            projectors[level] = P4AlignmentProjector(
                student_channels=int(student_feature.shape[1]),
                teacher_channels=int(teacher_feature.shape[1]),
                align_dim=int(_get(self.config, "foundation_align_dim", 256)),
            ).to(device=device)
        return projectors

    def _build_semantic_components(self) -> None:
        """Discover P4/teacher semantic dimensions and construct the F13 adapter."""
        teacher = self.teacher_manager
        if not callable(getattr(teacher, "encode", None)):
            raise ValueError("F13 semantic distillation requires a teacher exposing encode().")
        if "p4" in self.taps:
            tap = self.taps["p4"]
        else:
            tap = StudentFeatureTap(self.student_model, target="p4")
            self.__dict__["_semantic_tap"] = tap
        device = next(self.student_model.parameters(), torch.empty(0)).device
        yaml = getattr(self.student_model, "yaml", {}) or {}
        ch = int(yaml.get("channels", 3)) if isinstance(yaml, Mapping) else int(getattr(yaml, "channels", 3))
        stride = max(1, int(torch.as_tensor(getattr(self.student_model, "stride", 32)).max().item()))
        size = _as_probe_size(_get(self.config, "imgsz", 64), stride)
        was_training = self.student_model.training
        self.student_model.eval()
        tap.clear()
        with torch.inference_mode():
            self.student_model(torch.zeros(2, ch, size, size, device=device))
            student_feature = tap.feature
            teacher_features = teacher.encode(torch.zeros(2, 3, size, size, device=device))
        self.student_model.train(was_training)
        semantic = getattr(teacher_features, "semantic", None)
        if semantic is None and isinstance(teacher_features, Mapping):
            semantic = teacher_features.get("semantic")
        if not isinstance(semantic, torch.Tensor) or semantic.ndim != 2 or semantic.shape[1] <= 0:
            raise ValueError(
                "F13 semantic distillation requires teacher.encode() to return semantic features; "
                "DINOv3 does not provide text/image semantic embeddings in this phase."
            )
        projector = RegionSemanticProjector(int(student_feature.shape[1]), int(semantic.shape[1]))
        self._semantic_projector = projector.to(device=device)

    def _semantic_prompt_list(self) -> list[str]:
        """Resolve deterministic class prompts from explicit prompts or student names."""
        configured = _get(self.config, "foundation_semantic_prompts", None)
        if configured:
            prompts = [str(item) for item in configured]
        else:
            names = getattr(self.student_model, "names", {}) or {}
            if isinstance(names, Mapping):
                names = [names[key] for key in sorted(names)]
            names = list(names) if isinstance(names, (list, tuple)) else []
            template = str(_get(self.config, "foundation_semantic_prompt_template", "a photo of a {class_name}"))
            prompts = [template.format(class_name=str(name)) for name in names]
        if not prompts:
            raise ValueError("F13 semantic distillation requires class names or foundation_semantic_prompts.")
        return prompts

    def _semantic_text_prototypes(self) -> torch.Tensor:
        """Encode and cache text prototypes outside the student state_dict."""
        teacher = self.teacher_manager
        encode_text = getattr(teacher, "encode_text", None)
        if not callable(encode_text):
            raise ValueError("F13 semantic distillation requires a SigLIP2-like teacher exposing encode_text().")
        prompts = tuple(self._semantic_prompt_list())
        cached = self.__dict__.get("_semantic_text_cache")
        if cached is not None and self.__dict__.get("_semantic_prompts") == prompts:
            return cached.to(device=next(self.student_model.parameters()).device)
        prototypes = encode_text(prompts)
        if not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 2:
            raise ValueError("Foundation text prototypes must have shape (num_classes, semantic_dim).")
        self.__dict__["_semantic_prompts"] = prompts
        self.__dict__["_semantic_text_cache"] = prototypes.detach().cpu()
        return prototypes.to(device=next(self.student_model.parameters()).device)

    @staticmethod
    def _semantic_detection_assignment(student_model: nn.Module, preds: Any, batch: dict) -> tuple[Any, Any, Any]:
        """Run the native detection assigner once and return masks, matches, and raw feature maps."""
        raw = preds
        if isinstance(raw, tuple):
            raw = raw[1] if len(raw) > 1 and isinstance(raw[1], dict) else raw[0]
        if isinstance(raw, dict) and "one2many" in raw:
            raw = raw["one2many"]
        criterion = getattr(student_model, "criterion", None)
        for candidate in (criterion, getattr(criterion, "native_criterion", None)):
            if candidate is None:
                continue
            for attr in ("get_assigned_targets_and_loss",):
                fn = getattr(candidate, attr, None)
                if callable(fn) and isinstance(raw, dict) and "feats" in raw:
                    assigned = fn(raw, batch)
                    return assigned[0][0], assigned[0][1], raw["feats"]
            for branch in ("one2many", "one2one"):
                nested = getattr(candidate, branch, None)
                fn = getattr(nested, "get_assigned_targets_and_loss", None)
                if callable(fn) and isinstance(raw, dict) and branch in raw:
                    assigned = fn(raw[branch], batch)
                    return assigned[0][0], assigned[0][1], raw[branch].get("feats", [])
            vp_criterion = getattr(candidate, "vp_criterion", None)
            fn = getattr(vp_criterion, "get_assigned_targets_and_loss", None)
            if callable(fn) and isinstance(raw, dict) and "feats" in raw:
                assigned = fn(raw, batch)
                return assigned[0][0], assigned[0][1], raw["feats"]
        return None, None, raw.get("feats", []) if isinstance(raw, dict) else []

    def _semantic_kd(self, batch: dict, preds: Any, teacher_output: Any) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute positive-region text/image semantic KD and metrics."""
        projector = self.semantic_projector
        semantic = getattr(teacher_output, "semantic", None)
        if semantic is None and isinstance(teacher_output, Mapping):
            semantic = teacher_output.get("semantic")
        if projector is None or not isinstance(semantic, torch.Tensor):
            source = next(self.student_model.parameters())
            return source.sum() * 0.0, {"foundation_semantic_regions": 0.0}
        fg_mask, target_gt_idx, feature_shapes = self._semantic_detection_assignment(self.student_model, preds, batch)
        if not isinstance(fg_mask, torch.Tensor) or not isinstance(target_gt_idx, torch.Tensor):
            return projector.proj[0].weight.sum() * 0.0, {"foundation_semantic_regions": 0.0}
        tap = self.__dict__.get("_semantic_tap") or self.taps.get("p4")
        if tap is None or not tap.has_feature:
            return projector.proj[0].weight.sum() * 0.0, {"foundation_semantic_regions": 0.0}
        pooled = [
            positive_region_pool(
                tap.feature,
                fg_mask,
                target_gt_idx,
                batch,
                level_index=level_index,
                feature_shapes=feature_shapes,
                source_level_index=1,
            )
            for level_index in range(len(feature_shapes))
        ]
        regions = torch.cat([item[0] for item in pooled], dim=0)
        image_indices = torch.cat([item[1] for item in pooled], dim=0)
        labels = torch.cat([item[2] for item in pooled], dim=0)
        if regions.shape[0] == 0:
            return projector.proj[0].weight.sum() * 0.0, {"foundation_semantic_regions": 0.0}
        projected = projector(regions)
        image_targets = semantic.to(device=projected.device, dtype=projected.dtype)[image_indices]
        text = self._semantic_text_prototypes()
        total, text_loss, image_loss = semantic_distillation_loss(
            projected,
            labels,
            image_targets,
            text,
            text_weight=self.semantic_text_weight,
            image_weight=self.semantic_image_weight,
            temperature=self.semantic_temperature,
        )
        return total * self.semantic_loss_weight * int(batch["img"].shape[0]), {
            "foundation_semantic_loss": float(
                (total * self.semantic_loss_weight * int(batch["img"].shape[0])).detach()
            ),
            "foundation_semantic_text_loss": float((text_loss * self.semantic_loss_weight).detach()),
            "foundation_semantic_image_loss": float((image_loss * self.semantic_loss_weight).detach()),
            "foundation_semantic_regions": float(regions.shape[0]),
        }

    def _kd_loss(self, student_feature: torch.Tensor, teacher_feature: torch.Tensor) -> torch.Tensor:
        """Dispatch the configured Foundation loss while keeping the teacher detached."""
        return self._kd_components(student_feature, teacher_feature)[0]

    def _kd_components(self, student_feature: torch.Tensor, teacher_feature: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return total, cosine, and relational KD components in the shared aligned feature space."""
        return self._kd_components_with_weights(student_feature, teacher_feature, token_weights=None)

    def _kd_components_with_weights(
        self,
        student_feature: torch.Tensor,
        teacher_feature: torch.Tensor,
        *,
        token_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        """Return total, cosine, and relational KD components with optional detached token weights."""
        loss_name = str(_get(self.config, "foundation_loss", "relational")).lower()
        cosine_weight = float(_get(self.config, "foundation_cosine_weight", 1.0))
        relation_weight = float(_get(self.config, "foundation_relation_weight", 1.0))
        relation_mode = str(_get(self.config, "foundation_relation_mode", "sampled"))
        relation_samples = int(_get(self.config, "foundation_relation_samples", 256))
        zero = student_feature.sum() * 0.0
        if loss_name == "cosine":
            cosine = cosine_kd_loss(student_feature, teacher_feature, token_weights=token_weights)
            return cosine, cosine, zero
        if loss_name == "l2":
            difference = (student_feature.float() - teacher_feature.detach().float()).square().mean(dim=1)
            if token_weights is None:
                l2 = difference.mean()
            else:
                weights = token_weights.to(device=difference.device, dtype=difference.dtype).reshape_as(difference)
                l2 = (difference * weights).sum() / weights.sum().clamp_min(1e-6)
            return l2, zero, zero
        if loss_name == "relational":
            relation = relational_kd_loss(
                student_feature,
                teacher_feature,
                mode=relation_mode,
                num_samples=relation_samples,
                token_weights=token_weights,
            )
            return relation, zero, relation
        if loss_name == "hybrid":
            cosine = (
                cosine_kd_loss(student_feature, teacher_feature, token_weights=token_weights) if cosine_weight else zero
            )
            relation = (
                relational_kd_loss(
                    student_feature,
                    teacher_feature,
                    mode=relation_mode,
                    num_samples=relation_samples,
                    token_weights=token_weights,
                )
                if relation_weight
                else zero
            )
            total = cosine_weight * cosine + relation_weight * relation
            return total, cosine_weight * cosine, relation_weight * relation
        raise ValueError(f"Unsupported foundation_loss={loss_name!r}.")

    def set_foundation_progress(self, epoch: int, epochs: int) -> None:
        """Update fractional training progress (0-1) for the Foundation weight schedule."""
        total = max(int(epochs) - 1, 1)
        self.__dict__["_train_progress"] = min(max(float(epoch) / total, 0.0), 1.0)

    def _gate_factor(self) -> float:
        """Cosine-gated ramp: opens as the alignment EMA drops below gate_cosine, and backs off
        again below gate_cosine_low so the student is not over-aligned to teacher semantics.

        Empirical basis (F08, 4 runs): final detection gap vs final cosine_raw was
        0.79 -> -0.008, 0.87 -> -0.013, 0.92 -> +0.017, 0.93 -> +0.009 — alignment past
        ~0.92 hurts. The gate therefore peaks near the band centre and fades both ways.
        """
        ema = self.__dict__.get("_cosine_ema")
        if ema is None:
            return 0.0
        ramp_up = min(max((self.gate_cosine - ema) / self.gate_width, 0.0), 1.0)
        if self.gate_cosine_low is not None:
            ramp_down = min(max((ema - self.gate_cosine_low) / self.gate_width, 0.0), 1.0)
            return min(ramp_up, ramp_down)
        return ramp_up

    def _decay_factor(self) -> float:
        """Linear decay to zero over the final (1 - decay_start) fraction of training."""
        progress = float(self.__dict__.get("_train_progress", 0.0))
        if progress <= self.decay_start:
            return 1.0
        return max((1.0 - progress) / (1.0 - self.decay_start), 0.0)

    def effective_loss_weight(self) -> float:
        """Return the scheduled Foundation loss weight for the current training step.

        The "gate_decay" schedule keeps a warmup floor so the projector still learns while
        features are near-orthogonal (cosine gate closed), ramps to the full weight as the
        cosine EMA opens the gate, and decays to zero late in training so maturing detection
        features are not pulled back toward teacher semantics.
        """
        if self.weight_schedule != "gate_decay":
            return self.loss_weight
        ramp = self.warmup_floor + (1.0 - self.warmup_floor) * self._gate_factor()
        return self.loss_weight * ramp * self._decay_factor()

    def _update_cosine_ema(self, cosine: torch.Tensor) -> None:
        """Track an EMA of the raw cosine KD loss to drive the weight gate."""
        cosine = cosine.detach().float()
        if not torch.isfinite(cosine):
            return
        value = float(cosine.item())
        ema = self.__dict__.get("_cosine_ema")
        momentum = self.gate_ema
        self.__dict__["_cosine_ema"] = value if ema is None else momentum * ema + (1.0 - momentum) * value

    def forward(self, x, *args, **kwargs):
        """Run student inference, or compute task plus Foundation loss for a training batch."""
        if not isinstance(x, dict) or self._disabled or not self.training:
            self.reset_foundation_metrics()
            return self.student_model(x, *args, **kwargs)
        return self.loss(x, *args, **kwargs)

    def loss(self, batch: dict, preds=None):
        """Return ``(task_loss + foundation_loss, task_items + foundation_item)``."""
        if self._disabled and not self.__dict__.get("_student_only", False):
            self.reset_foundation_metrics()
            if preds is None:
                preds = self.student_model(batch["img"])
            return self.student_model.loss(batch, preds)
        if not self.training:
            self.reset_foundation_metrics()
            if preds is None:
                preds = self.student_model(batch["img"])
            task_loss, task_items = self.student_model.loss(batch, preds)
            zero = task_loss.new_zeros(1)
            return torch.cat([task_loss.reshape(-1), zero]), torch.cat([task_items.reshape(-1), zero])

        taps = self.taps or {"p4": self.tap}
        for tap in taps.values():
            tap.clear()
        if preds is None:
            preds = self.student_model(batch["img"])
        if not all(tap.has_feature for tap in taps.values()):
            # A caller may provide precomputed predictions; capture the student feature without changing the result.
            self.student_model(batch["img"])
        student_features = {level: tap.feature for level, tap in taps.items()}
        with torch.inference_mode():
            teacher_output = self.teacher_manager.encode(batch["img"])
            teacher_feature = _dense_p4(teacher_output)
            if self.foundation_teacher_name == "multi":
                teacher_summary = foundation_multiteacher_summary(teacher_output)
            else:
                teacher_summary = foundation_teacher_summary(teacher_output)
        foreground_enabled = bool(_get(self.config, "foundation_foreground_weighting", False))
        components = []
        foreground_means = []
        if self.loss_weight > 0:
            for level, student_feature in student_features.items():
                projector = self.projector_for(level)
                student_aligned, teacher_aligned = projector(student_feature, teacher_feature)
                token_weights = None
                if foreground_enabled:
                    image_height, image_width = batch["img"].shape[-2:]
                    token_weights = foreground_token_weights(
                        batch,
                        height=student_aligned.shape[-2],
                        width=student_aligned.shape[-1],
                        image_height=image_height,
                        image_width=image_width,
                        foreground_weight=float(_get(self.config, "foundation_foreground_weight", 1.5)),
                        boundary_weight=float(_get(self.config, "foundation_boundary_weight", 1.0)),
                        background_weight=float(_get(self.config, "foundation_background_weight", 0.25)),
                    )
                    foreground_means.append(token_weights.mean())
                components.append(
                    (
                        level,
                        self._kd_components_with_weights(student_aligned, teacher_aligned, token_weights=token_weights),
                    )
                )
            if not components:
                raise RuntimeError("Foundation distillation has no configured target levels.")
            kd = torch.stack([values[0] for _, values in components]).mean()
            cosine = torch.stack([values[1] for _, values in components]).mean()
            relation = torch.stack([values[2] for _, values in components]).mean()
            if not torch.isfinite(kd):
                raise ValueError("Foundation distillation loss is NaN or Inf.")
            effective_weight = self.effective_loss_weight()
            self._update_cosine_ema(cosine)
        else:
            source = next((value for value in student_features.values() if isinstance(value, torch.Tensor)), None)
            kd = source.sum() * 0.0 if source is not None else teacher_summary.sum() * 0.0
            cosine = kd
            relation = kd
            effective_weight = self.loss_weight
        batch_size = int(batch["img"].shape[0])
        feature_loss = kd * effective_weight * batch_size
        route_loss, route_metrics = self._routing_kd(teacher_summary, batch_size=batch_size)
        task_loss, task_items = self.student_model.loss(batch, preds)
        semantic_loss, semantic_metrics = (
            self._semantic_kd(batch, preds, teacher_output) if self._semantic_enabled else (kd * 0.0, {})
        )
        foundation_loss = feature_loss + route_loss + semantic_loss
        self.__dict__["_last_foundation_loss"] = foundation_loss.detach()
        task_scalar = float(task_loss.detach().float().mean().item())
        foundation_scalar = float(foundation_loss.detach().float().item())
        batch_size = max(int(batch["img"].shape[0]), 1)
        self.__dict__["_last_foundation_metrics"] = {
            "foundation_loss": foundation_scalar,
            "foundation_cosine_loss": float((cosine.detach().float() * effective_weight * batch_size).item()),
            "foundation_relational_loss": float((relation.detach().float() * effective_weight * batch_size).item()),
            # Raw (unweighted) KD components: comparable across loss_weight and batch_size settings.
            "foundation_cosine_raw": float(cosine.detach().float().item()),
            "foundation_relational_raw": float(relation.detach().float().item()),
            "foundation_task_ratio": foundation_scalar / max(task_scalar, 1e-8),
            "foundation_loss_weight": float(self.loss_weight),
            "foundation_effective_weight": float(effective_weight),
            "foundation_foreground_enabled": float(foreground_enabled),
            "foundation_foreground_mean_weight": float(torch.stack(foreground_means).mean().item())
            if foreground_means
            else 1.0,
        }
        self.__dict__["_last_foundation_metrics"].update(route_metrics)
        self.__dict__["_last_foundation_metrics"].update(semantic_metrics)
        self.__dict__["_last_foundation_metrics"].update(
            self._multitask_metrics(task_loss, task_items, foundation_loss, preds)
        )
        if self.semantic_distill:
            self.__dict__["_last_foundation_metrics"]["foundation_semantic_enabled"] = float(self._semantic_enabled)
        if self.multiscale:
            for level, values in components:
                self.__dict__["_last_foundation_metrics"][f"foundation_{level}_loss"] = float(
                    (values[0].detach().float() * effective_weight * batch_size).item()
                )
        return torch.cat([task_loss.reshape(-1), foundation_loss.reshape(1)]), torch.cat(
            [task_items.reshape(-1), foundation_loss.detach().reshape(1)]
        )

    def train(self, mode: bool = True):
        """Set student/projector mode while forcing the Foundation Teacher to eval/frozen mode."""
        super().train(mode)
        if self.teacher_manager is not None:
            freeze = getattr(self.teacher_manager, "freeze", None)
            if callable(freeze):
                freeze()
            else:
                train = getattr(self.teacher_manager, "train", None)
                if callable(train):
                    train(False)
        return self

    def deployment_model(self) -> nn.Module:
        """Return the pure student model for deployment/export callers."""
        student = self.student_model
        # Checkpoint-loaded DetectionModel instances may not carry the task metadata that the outer YOLO object
        # owns. Exporters consume these attributes after the wrapper has been stripped, so preserve them explicitly.
        for name in ("task", "names", "nc", "args", "end2end"):
            if hasattr(self, name):
                try:
                    setattr(student, name, getattr(self, name))
                except (AttributeError, TypeError):
                    pass
        self.close()
        return student

    def checkpoint_metadata(self) -> dict[str, Any]:
        """Return JSON-safe Foundation settings needed to reconstruct training-time distillation."""
        projector = self.projector
        teacher = self.teacher_manager
        route_specs = copy.deepcopy(self.__dict__.get("_route_specs", []))
        if self.router_distill and not route_specs:
            # EMA/checkpoint copies intentionally have no live route heads.  The
            # static LatentMixture graph still provides enough information to
            # describe deterministic F11 reconstruction without serializing the
            # training-only teacher routers.
            teacher_channels = (
                int(self.projector_for("p4").teacher_channels)
                if not self.multiscale
                else int(self.projector_for(self.target_levels[0]).teacher_channels)
            )
            teacher_dim = int(self.__dict__.get("_router_teacher_dim") or teacher_channels * 3)
            if self.foundation_teacher_name == "multi" and teacher_dim == teacher_channels * 3:
                siglip_channels = int(getattr(getattr(teacher, "siglip2", None), "hidden_size", 0) or teacher_channels)
                # Before the first route forward a copied wrapper has no live
                # teacher. Keep a deterministic shape marker; live wrappers
                # replace it with the exact DINO+SigLIP summary width.
                teacher_dim = teacher_channels * 3 + siglip_channels * 3 + 2
            for index, (name, module) in enumerate(self._latent_route_modules()):
                student_dim = int(getattr(getattr(module, "router", None), "latent_dim", 0) or 0)
                num_experts = int(getattr(module, "num_experts", 0) or 0)
                if student_dim <= 0 or num_experts <= 0:
                    continue
                hidden_dim = max(32, min(256, 2 * max(student_dim, teacher_dim)))
                route_specs.append(
                    {
                        "name": name,
                        "student_dim": student_dim,
                        "teacher_dim": teacher_dim,
                        "num_experts": num_experts,
                        "hidden_dim": hidden_dim,
                        "seed": self._route_seed(index, student_dim, teacher_dim, num_experts),
                    }
                )
        teacher_name = str(_get(self.config, "foundation_teacher", getattr(teacher, "name", "none"))).lower()
        router_kind = "multi_foundation_image_level" if teacher_name == "multi" else "latent_mixture_image_level"
        router_teachers = list(
            str(name).lower()
            for name in (_get(self.config, "foundation_router_teachers", self.router_teachers) or self.router_teachers)
        )
        dino_model = _get(self.config, "foundation_dinov3_model", None) or _get(self.config, "foundation_model", None)
        siglip_model = _get(self.config, "foundation_siglip2_model", None) or DEFAULT_SIGLIP2_MODEL
        if teacher_name == "multi" and teacher is not None:
            dino_model = dino_model or getattr(getattr(teacher, "dinov3", None), "model_id", None)
            siglip_model = siglip_model or getattr(getattr(teacher, "siglip2", None), "model_id", None)
        return {
            "schema_version": 1,
            # Deep-copied EMA/checkpoint wrappers intentionally have no live teacher, but still describe the
            # training configuration so resume can reconstruct it.
            "enabled": bool(
                _get(self.config, "foundation_enabled", False)
                and (
                    float(_get(self.config, "foundation_loss_weight", self.loss_weight) or 0.0) > 0
                    or (
                        bool(_get(self.config, "foundation_router_distill", self.router_distill))
                        and float(_get(self.config, "foundation_router_loss_weight", self.router_loss_weight) or 0.0)
                        > 0
                    )
                    or (
                        bool(_get(self.config, "foundation_semantic_distill", self.semantic_distill))
                        and float(
                            _get(self.config, "foundation_semantic_loss_weight", self.semantic_loss_weight) or 0.0
                        )
                        > 0
                    )
                )
            ),
            "training_only": True,
            "teacher": teacher_name,
            "backend": str(_get(self.config, "foundation_backend", "transformers")),
            "model": str(
                _get(
                    self.config,
                    "foundation_model",
                    getattr(
                        teacher,
                        "model_id",
                        DEFAULT_SIGLIP2_MODEL
                        if str(_get(self.config, "foundation_teacher", "dinov3")).lower() == "siglip2"
                        else DEFAULT_DINOV3_MODEL,
                    ),
                )
                or getattr(
                    teacher,
                    "model_id",
                    DEFAULT_SIGLIP2_MODEL
                    if str(_get(self.config, "foundation_teacher", "dinov3")).lower() == "siglip2"
                    else DEFAULT_DINOV3_MODEL,
                )
            ),
            "models": {"dinov3": str(dino_model or DEFAULT_DINOV3_MODEL), "siglip2": str(siglip_model)},
            "weights": str(
                _get(self.config, "foundation_weights", None) or getattr(teacher, "weights_path", None) or ""
            ),
            "dtype": str(_get(self.config, "foundation_teacher_dtype", getattr(teacher, "dtype", "auto"))),
            "device": str(_get(self.config, "foundation_teacher_device", getattr(teacher, "device", "auto"))),
            "target_levels": [str(level) for level in (_get(self.config, "foundation_target_levels", ("p4",)) or ())],
            "multiscale": bool(_get(self.config, "foundation_multiscale", False)),
            "align_dim": int(_get(self.config, "foundation_align_dim", getattr(projector, "align_dim", 256))),
            "loss": str(_get(self.config, "foundation_loss", "relational")),
            "loss_weight": float(_get(self.config, "foundation_loss_weight", self.loss_weight) or 0.0),
            "relation_mode": str(_get(self.config, "foundation_relation_mode", "sampled")),
            "relation_samples": int(_get(self.config, "foundation_relation_samples", 256)),
            "cosine_weight": float(_get(self.config, "foundation_cosine_weight", 1.0)),
            "relation_weight": float(_get(self.config, "foundation_relation_weight", 1.0)),
            "foreground_weighting": bool(_get(self.config, "foundation_foreground_weighting", False)),
            "foreground_weight": float(_get(self.config, "foundation_foreground_weight", 1.5)),
            "boundary_weight": float(_get(self.config, "foundation_boundary_weight", 1.0)),
            "background_weight": float(_get(self.config, "foundation_background_weight", 0.25)),
            "semantic_available": bool(teacher_name in {"siglip2", "multi"}),
            "text_prototypes_training_only": True,
            "semantic_distill": bool(_get(self.config, "foundation_semantic_distill", self.semantic_distill)),
            "semantic_loss_weight": float(
                _get(self.config, "foundation_semantic_loss_weight", self.semantic_loss_weight) or 0.0
            ),
            "semantic_text_weight": float(
                _get(self.config, "foundation_semantic_text_weight", self.semantic_text_weight)
            ),
            "semantic_image_weight": float(
                _get(self.config, "foundation_semantic_image_weight", self.semantic_image_weight)
            ),
            "semantic_temperature": float(
                _get(self.config, "foundation_semantic_temperature", self.semantic_temperature)
            ),
            "semantic_prompts": list(self.__dict__.get("_semantic_prompts") or self._semantic_prompt_list())
            if self._semantic_enabled
            else [],
            "semantic_dim": int(getattr(self.semantic_projector, "semantic_dim", 0) or 0),
            "router_distill": bool(_get(self.config, "foundation_router_distill", self.router_distill)),
            "router_loss_weight": float(
                _get(self.config, "foundation_router_loss_weight", self.router_loss_weight) or 0.0
            ),
            "router_temperature": float(self.router_temperature),
            "router_kind": router_kind,
            "router_teachers": router_teachers,
            "router_native_state": bool(_get(self.config, "foundation_router_native_state", self.router_native_state)),
            "router_input_dims": {
                "student": sorted({int(spec.get("student_dim", 0)) for spec in route_specs}),
                "teacher": sorted({int(spec.get("teacher_dim", 0)) for spec in route_specs}),
            },
            "router_specs": route_specs,
            "multitask": {
                "enabled": bool(self.multitask_enabled),
                "active_tasks": list(self.multitask_active_tasks),
                "expected_tasks": list(self.__dict__.get("_multitask_expected_tasks", ())),
                "negative_transfer_threshold": float(self.multitask_negative_transfer_threshold),
                "task_router_semantics": "existing_task_local_router",
                "representation_transfer_gate": "at_least_two_supervised_tasks",
            },
            "multitask_enabled": bool(self.multitask_enabled),
            "multitask_active_tasks": list(self.multitask_active_tasks),
            "multitask_tasks": list(self.multitask_active_tasks),
            "student_channels": (
                {
                    level: int(getattr(self.projector_for(level), "student_channels", 0) or 0)
                    for level in self.target_levels
                }
                if self.multiscale
                else int(getattr(projector, "student_channels", 0) or 0)
            ),
            "teacher_channels": (
                {
                    level: int(getattr(self.projector_for(level), "teacher_channels", 0) or 0)
                    for level in self.target_levels
                }
                if self.multiscale
                else int(getattr(projector, "teacher_channels", 0) or 0)
            ),
        }

    def close(self) -> None:
        """Remove the student hook and release transient references."""
        for tap in self.taps.values():
            tap.close()
        semantic_tap = self.__dict__.get("_semantic_tap")
        if semantic_tap is not None and semantic_tap not in self.taps.values():
            semantic_tap.close()
        self.__dict__["_tap"] = None
        self.__dict__["_taps"] = {}
        self.__dict__["_semantic_tap"] = None
        self.__dict__["_semantic_text_cache"] = None
        self.__dict__["_route_teachers"] = {}
        self.__dict__["_route_specs"] = []

    def __getstate__(self):
        """Exclude the external teacher and live hook from copies/checkpoints."""
        # The tap is also referenced by the bound hook stored inside the student's module graph.  Clear its captured
        # autograd tensor before deepcopy so checkpoint/EMA serialization cannot traverse a live computation graph.
        active_taps = self.taps
        semantic_tap = self.__dict__.get("_semantic_tap")
        clean_student = None
        if active_taps:
            for tap in active_taps.values():
                tap.clear()
                tap.close()
            if semantic_tap is not None and semantic_tap not in active_taps.values():
                semantic_tap.clear()
                semantic_tap.close()
            try:
                # LatentMixture retains graph-connected routing tensors for F11
                # until the wrapper has composed the loss.  Clear these before
                # deepcopy just like feature taps, then restore them for the
                # live training wrapper.
                route_runtime = []
                for module in self.student_model.modules():
                    values = tuple(
                        getattr(module, name, None)
                        for name in ("_last_routing_logits", "_last_routing_probs", "_last_routing_summary")
                    )
                    if any(value is not None for value in values):
                        route_runtime.append((module, values))
                        for name in ("_last_routing_logits", "_last_routing_probs", "_last_routing_summary"):
                            if hasattr(module, name):
                                setattr(module, name, None)
                clean_student = copy.deepcopy(self.student_model)
            finally:
                for module, values in locals().get("route_runtime", ()):
                    for name, value in zip(
                        ("_last_routing_logits", "_last_routing_probs", "_last_routing_summary"), values
                    ):
                        setattr(module, name, value)
                self.__dict__["_taps"] = {
                    level: StudentFeatureTap(self.student_model, target=level) for level in self.target_levels
                }
                self.__dict__["_tap"] = self.__dict__["_taps"].get("p4") or next(iter(self.__dict__["_taps"].values()))
                if self._semantic_enabled and "p4" not in self.target_levels:
                    self.__dict__["_semantic_tap"] = StudentFeatureTap(self.student_model, target="p4")
        state = self.__dict__.copy()
        modules = dict(self._modules)
        if clean_student is not None:
            state["student_model"] = clean_student
            modules["student_model"] = clean_student
        state["_modules"] = modules
        state.pop("_teacher_manager", None)
        state.pop("_tap", None)
        state.pop("_taps", None)
        state.pop("_route_teachers", None)
        state.pop("_route_specs", None)
        state.pop("_semantic_tap", None)
        state.pop("_semantic_text_cache", None)
        state.pop("_semantic_prompts", None)
        state.pop("_last_foundation_loss", None)
        state.pop("_last_foundation_metrics", None)
        return state

    def __setstate__(self, state):
        """Restore a checkpoint/EMA copy as a student-only transparent wrapper."""
        self.__dict__.update(state)
        self.__dict__["_teacher_manager"] = None
        self.__dict__["_tap"] = None
        self.__dict__["_route_teachers"] = {}
        self.__dict__["_route_specs"] = []
        self.__dict__["_semantic_tap"] = None
        self.__dict__["_semantic_text_cache"] = None
        self.__dict__["_semantic_prompts"] = None
        self.__dict__["_disabled"] = True
        self.__dict__["_student_only"] = True
        self.__dict__["_last_foundation_loss"] = torch.zeros(())
        self.__dict__["_last_foundation_metrics"] = {}

    @property
    def criterion(self):
        return self.student_model.criterion

    @criterion.setter
    def criterion(self, value) -> None:
        self.student_model.criterion = value

    def init_criterion(self):
        return self.student_model.init_criterion()

    @property
    def end2end(self):
        return getattr(self.student_model, "end2end", False)

    @end2end.setter
    def end2end(self, value):
        self.student_model.end2end = value

    def set_head_attr(self, **kwargs):
        return self.student_model.set_head_attr(**kwargs)

    # Assignment-aware proxies are required because DetectionTrainer attaches dataset metadata after wrapping.  Plain
    # ``__getattr__`` only handles reads and would otherwise leave ``nc``/``names`` on the wrapper rather than on YOLO.
    @property
    def nc(self):
        return self.student_model.nc

    @nc.setter
    def nc(self, value):
        self.student_model.nc = value

    @property
    def names(self):
        return self.student_model.names

    @names.setter
    def names(self, value):
        self.student_model.names = value

    @property
    def args(self):
        return self.student_model.args

    @args.setter
    def args(self, value):
        self.student_model.args = value

    @property
    def class_weights(self):
        return getattr(self.student_model, "class_weights", None)

    @class_weights.setter
    def class_weights(self, value):
        self.student_model.class_weights = value

    @property
    def stride(self):
        return self.student_model.stride

    @stride.setter
    def stride(self, value):
        self.student_model.stride = value

    @property
    def yaml(self):
        return self.student_model.yaml

    @yaml.setter
    def yaml(self, value):
        self.student_model.yaml = value

    def __getattr__(self, name: str):
        """Proxy common model attributes used by trainers, validators, and predictors."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = object.__getattribute__(self, "_modules")
            student = modules.get("student_model")
            if student is not None and hasattr(student, name):
                return getattr(student, name)
            raise


def build_foundation_distillation_wrapper(
    student_model: nn.Module,
    args: Any,
    *,
    device: torch.device | str | None = None,
    teacher_manager: Any | None = None,
    model_loader: Callable[..., nn.Module] | None = None,
    processor_loader: Callable[..., Any] | None = None,
) -> FoundationDistillationModel | nn.Module:
    """Build the F06 wrapper and DINOv3 backend from trainer arguments.

    ``teacher_manager`` is an explicit injection point for offline tests and local backends.  A disabled or zero-weight
    run returns the original student unchanged, so the default training path has no Foundation side effects.
    """
    enabled = bool(_get(args, "foundation_enabled", False))
    weight = float(_get(args, "foundation_loss_weight", 0.0) or 0.0)
    router_enabled = (
        bool(_get(args, "foundation_router_distill", False))
        and float(_get(args, "foundation_router_loss_weight", 0.0) or 0.0) > 0
    )
    semantic_enabled = (
        bool(_get(args, "foundation_semantic_distill", False))
        and float(_get(args, "foundation_semantic_loss_weight", 0.0) or 0.0) > 0
    )
    if not enabled or (weight <= 0 and not router_enabled and not semantic_enabled):
        return student_model
    if teacher_manager is None:
        backend = str(_get(args, "foundation_backend", "transformers")).lower()
        if backend != "transformers":
            raise ValueError(
                f"F06 does not construct foundation_backend={backend!r}; inject a teacher_manager instead."
            )
        teacher_name = str(_get(args, "foundation_teacher", "dinov3")).lower()
        dtype = _get(args, "foundation_teacher_dtype", "auto")
        teacher_device = _teacher_device(_get(args, "foundation_teacher_device", "auto"), device)
        if teacher_name == "multi":
            # F14 keeps explicit per-backend model ids so a DINOv3 cache and a
            # SigLIP2 cache can coexist. ``foundation_model`` remains a
            # backwards-compatible alias for the DINOv3 id.
            dino_model_id = _get(args, "foundation_dinov3_model", None) or _get(args, "foundation_model", None)
            siglip_model_id = _get(args, "foundation_siglip2_model", None) or DEFAULT_SIGLIP2_MODEL
            if dino_model_id is None:
                dino_model_id = DEFAULT_DINOV3_MODEL
            dino = DINOv3Teacher(
                model_id=dino_model_id,
                dtype=dtype,
                device=teacher_device,
                weights_path=_get(args, "foundation_dinov3_weights", None) or _get(args, "foundation_weights", None),
                model_loader=model_loader,
            )
            siglip = SigLIP2Teacher(
                model_id=siglip_model_id,
                dtype=dtype,
                device=teacher_device,
                weights_path=_get(args, "foundation_siglip2_weights", None),
                model_loader=model_loader,
                processor_loader=processor_loader,
            )
            teacher_manager = MultiFoundationTeacher(dinov3=dino, siglip2=siglip)
        elif teacher_name in {"dinov3", "siglip2"}:
            model_id = _get(args, "foundation_model", None) or (
                DEFAULT_SIGLIP2_MODEL if teacher_name == "siglip2" else DEFAULT_DINOV3_MODEL
            )
            teacher_cls = SigLIP2Teacher if teacher_name == "siglip2" else DINOv3Teacher
            teacher_kwargs = {
                "model_id": model_id,
                "dtype": dtype,
                "device": teacher_device,
                "weights_path": _get(args, "foundation_weights", None),
                "model_loader": model_loader,
            }
            if teacher_name == "siglip2":
                teacher_kwargs["processor_loader"] = processor_loader
            teacher_manager = teacher_cls(**teacher_kwargs)
        else:
            raise ValueError(f"Unsupported foundation_teacher={teacher_name!r}; use dinov3, siglip2, or multi.")
    elif str(_get(args, "foundation_teacher", "none")).lower() == "multi":
        if not isinstance(teacher_manager, MultiFoundationTeacher):
            if isinstance(teacher_manager, Mapping):
                teacher_manager = MultiFoundationTeacher(
                    dinov3=teacher_manager.get("dinov3"), siglip2=teacher_manager.get("siglip2")
                )
            elif not all(hasattr(teacher_manager, name) for name in ("dinov3", "siglip2")):
                raise TypeError(
                    "F14 injected teacher_manager must be MultiFoundationTeacher or a mapping with "
                    "'dinov3' and 'siglip2'."
                )
    return FoundationDistillationModel(student_model=student_model, teacher_manager=teacher_manager, config=args)


def strip_foundation_distillation_model(model: nn.Module) -> nn.Module:
    """Return a pure student model for deployment artifacts and export graphs."""
    if isinstance(model, FoundationDistillationModel):
        return model.deployment_model()
    return model


def rebuild_foundation_distillation_wrapper(
    student_model: nn.Module,
    args: Any,
    checkpoint_model: nn.Module | None = None,
    *,
    device: torch.device | str | None = None,
    teacher_manager: Any | None = None,
    processor_loader: Callable[..., Any] | None = None,
) -> FoundationDistillationModel | nn.Module:
    """Rebuild a Foundation wrapper and restore its trainable projector from a checkpoint copy."""
    wrapper = build_foundation_distillation_wrapper(
        student_model,
        args,
        device=device,
        teacher_manager=teacher_manager,
        processor_loader=processor_loader,
    )
    if not isinstance(wrapper, FoundationDistillationModel) or not isinstance(checkpoint_model, nn.Module):
        return wrapper
    checkpoint_projector = getattr(checkpoint_model, "projector", None)
    if checkpoint_projector is not None:
        try:
            wrapper.projector.load_state_dict(checkpoint_projector.state_dict())
        except (KeyError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "Foundation checkpoint projector is incompatible with the rebuilt student/teacher."
            ) from exc
    checkpoint_semantic_projector = getattr(checkpoint_model, "semantic_projector", None)
    if checkpoint_semantic_projector is not None and wrapper.semantic_projector is not None:
        try:
            wrapper.semantic_projector.load_state_dict(checkpoint_semantic_projector.state_dict())
        except (KeyError, RuntimeError, ValueError) as exc:
            raise RuntimeError("Foundation semantic projector is incompatible with the rebuilt student.") from exc
    # The cosine-gate EMA is pickled with the checkpoint model; restore it so resume does not
    # silently close the gate and drop the distillation weight back to the warmup floor.
    cosine_ema = checkpoint_model.__dict__.get("_cosine_ema")
    if cosine_ema is not None:
        wrapper.__dict__["_cosine_ema"] = float(cosine_ema)
    return wrapper


__all__ = [
    "FoundationDistillationModel",
    "build_foundation_distillation_wrapper",
    "rebuild_foundation_distillation_wrapper",
    "strip_foundation_distillation_model",
]
