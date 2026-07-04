# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Optional MoE diagnostic callbacks.

The callbacks module imports these helpers unconditionally, so they must remain
lightweight and dependency-safe even when plotting packages are unavailable.
"""

from __future__ import annotations

from pathlib import Path


def create_moe_diagnostic_callback(interval: int = 1, output_dir: str | Path | None = None):
    """Create an epoch-end callback that can run MoE diagnostics when enabled.

    The returned callback is intentionally defensive: if the model has no MoE
    router modules or optional plotting dependencies are missing, training
    should continue without failing.
    """

    def on_train_epoch_end(trainer):
        epoch = int(getattr(trainer, "epoch", 0))
        if interval <= 0 or (epoch + 1) % interval:
            return
        try:
            from ultralytics.nn.modules.moe.analysis import ExpertUsageTracker

            tracker = ExpertUsageTracker(trainer.model)
            setattr(trainer, "_moe_diagnostic_tracker", tracker)
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            setattr(trainer, "_moe_diagnostic_error", str(exc))

    return on_train_epoch_end


def create_moe_diagnostic_train_end_callback(output_dir: str | Path | None = None):
    """Create a train-end callback that prints a best-effort MoE diagnostic report."""

    def on_train_end(trainer):
        tracker = getattr(trainer, "_moe_diagnostic_tracker", None)
        if tracker is None:
            return
        try:
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            tracker.print_report()
            tracker.remove_hooks()
        except Exception as exc:
            setattr(trainer, "_moe_diagnostic_error", str(exc))

    return on_train_end
