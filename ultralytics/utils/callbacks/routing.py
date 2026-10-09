# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Opt-in routing telemetry callbacks for existing training loggers."""

from __future__ import annotations


def on_fit_epoch_end(trainer) -> None:
    """Temporarily expose the latest routing snapshot through standard logger metrics."""
    telemetry = getattr(trainer, "training_telemetry", None)
    if not bool(getattr(telemetry, "enabled", False)):
        return
    scalar_factory = getattr(telemetry, "tensorboard_scalars", None)
    if not callable(scalar_factory):
        return
    try:
        scalars = scalar_factory()
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        trainer._routing_callback_error = f"{type(exc).__name__}: {exc}"
        return
    if not scalars:
        return
    metrics = getattr(trainer, "metrics", None)
    trainer._routing_callback_original_metrics = metrics
    trainer.metrics = {**(metrics if isinstance(metrics, dict) else {}), **scalars}


def on_fit_epoch_end_restore(trainer) -> None:
    """Restore metrics after all standard logger callbacks have consumed routing scalars."""
    marker = "_routing_callback_original_metrics"
    if not hasattr(trainer, marker):
        return
    trainer.metrics = getattr(trainer, marker)
    delattr(trainer, marker)


callbacks = {"on_fit_epoch_end": on_fit_epoch_end}
