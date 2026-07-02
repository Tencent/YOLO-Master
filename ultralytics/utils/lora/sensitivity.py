from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass
class LayerSensitivity:
    """Minimal compatibility record for LoRA sensitivity selection."""

    name: str
    score: float
    selected: bool = True
    reason: str = "compatibility shim"


@dataclass
class SensitivityReport:
    """Compatibility report returned by the lightweight selector."""

    layers: list[LayerSensitivity] = field(default_factory=list)
    selected_targets: list[str] = field(default_factory=list)
    skipped_targets: list[str] = field(default_factory=list)
    notes: list[str] = field(
        default_factory=lambda: [
            "ultralytics.utils.lora.sensitivity was missing in this checkout",
            "a no-op compatibility shim was used",
        ]
    )


class GradientSensitivitySelector:
    """No-op selector that preserves requested targets when the real module is absent."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def select_targets(self, targets: Sequence[str] | None = None, **kwargs) -> SensitivityReport:
        targets = list(targets or [])
        return SensitivityReport(
            layers=[LayerSensitivity(name=t, score=0.0, selected=True) for t in targets],
            selected_targets=targets,
            skipped_targets=[],
        )

    def __call__(self, targets: Sequence[str] | None = None, **kwargs) -> SensitivityReport:
        return self.select_targets(targets=targets, **kwargs)


def select_targets_by_sensitivity(
    targets: Iterable[str] | None = None, *args, **kwargs
) -> tuple[list[str], SensitivityReport]:
    """Return the requested targets unchanged with a compatibility report."""

    selected = list(targets or [])
    report = SensitivityReport(
        layers=[LayerSensitivity(name=t, score=0.0, selected=True) for t in selected],
        selected_targets=selected,
        skipped_targets=[],
    )
    return selected, report
