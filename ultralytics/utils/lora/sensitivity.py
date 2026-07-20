from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn


@dataclass
class LayerSensitivity:
    """Gradient-based sensitivity score for one candidate target."""

    name: str
    score: float
    selected: bool = True
    reason: str = "gradient norm"


@dataclass
class SensitivityReport:
    """Serializable result of calibration-based target selection."""

    layers: list[LayerSensitivity] = field(default_factory=list)
    selected_targets: list[str] = field(default_factory=list)
    skipped_targets: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class GradientSensitivitySelector:
    """Select targets using parameter gradient norms on calibration batches."""

    def __init__(self, model: nn.Module | None = None, data_loader=None, **kwargs) -> None:
        self.model = model
        self.data_loader = data_loader
        self.num_batches = max(1, int(kwargs.get("num_batches", 4)))
        self.top_ratio = float(kwargs.get("top_ratio", 0.5))
        self.max_layers = kwargs.get("max_layers")
        self.loss_fn = kwargs.get("loss_fn")

    @staticmethod
    def _inputs(batch: Any):
        if isinstance(batch, dict):
            for key in ("img", "image", "images", "x"):
                value = batch.get(key)
                if isinstance(value, torch.Tensor):
                    return (value,)
            return None
        if isinstance(batch, (tuple, list)):
            return (batch[0],) if batch and isinstance(batch[0], torch.Tensor) else None
        return (batch,) if isinstance(batch, torch.Tensor) else None

    @staticmethod
    def _energy(output: Any) -> torch.Tensor | None:
        if isinstance(output, torch.Tensor):
            tensors = [output]
        elif isinstance(output, (tuple, list)):
            tensors = [x for x in output if isinstance(x, torch.Tensor)]
        elif isinstance(output, dict):
            tensors = [x for x in output.values() if isinstance(x, torch.Tensor)]
        else:
            tensors = []
        return sum(x.float().pow(2).mean() for x in tensors) / len(tensors) if tensors else None

    def select_targets(self, targets: Sequence[str] | None = None, **kwargs) -> SensitivityReport:
        targets = list(targets or [])
        model = kwargs.get("model", self.model)
        loader = kwargs.get("data_loader", self.data_loader)
        if model is None or loader is None or not targets:
            return SensitivityReport(
                layers=[LayerSensitivity(name=t, score=0.0, reason="no calibration data") for t in targets],
                selected_targets=targets,
                notes=["sensitivity selection skipped: model and data_loader are required"],
            )

        modules = dict(model.named_modules())
        scores = {name: 0.0 for name in targets if isinstance(modules.get(name), (nn.Conv2d, nn.Linear))}
        if not scores:
            return SensitivityReport(skipped_targets=targets, notes=["no scoreable Conv2d/Linear targets"])

        was_training = model.training
        model.eval()
        try:
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= self.num_batches:
                    break
                inputs = self._inputs(batch)
                if inputs is None:
                    continue
                model.zero_grad(set_to_none=True)
                output = model(*inputs)
                loss = self.loss_fn(output, batch) if self.loss_fn is not None else self._energy(output)
                if loss is None or not loss.requires_grad:
                    continue
                loss.backward()
                for name, module in ((name, modules[name]) for name in scores):
                    grads = [p.grad.detach().float().norm().item() for p in module.parameters() if p.grad is not None]
                    if grads:
                        scores[name] += sum(grads) / len(grads)
        finally:
            model.train(was_training)
            model.zero_grad(set_to_none=True)

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        keep = max(1, int(round(len(ordered) * min(max(self.top_ratio, 0.0), 1.0))))
        if self.max_layers is not None:
            keep = min(keep, max(1, int(self.max_layers)))
        selected = [name for name, score in ordered[:keep] if score > 0.0] or [name for name, _ in ordered[:keep]]
        return SensitivityReport(
            layers=[LayerSensitivity(name=name, score=float(score), selected=name in selected) for name, score in ordered],
            selected_targets=selected,
            skipped_targets=[name for name in targets if name not in selected],
            notes=[f"gradient energy over up to {self.num_batches} calibration batches"],
        )

    def __call__(self, targets: Sequence[str] | None = None, **kwargs) -> SensitivityReport:
        return self.select_targets(targets=targets, **kwargs)


def select_targets_by_sensitivity(targets: Iterable[str] | None = None, *args, **kwargs):
    selector = GradientSensitivitySelector(*args, **kwargs)
    report = selector.select_targets(targets=list(targets or []))
    return report.selected_targets, report
