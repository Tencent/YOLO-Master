# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .base import add_integration_callbacks as _add_integration_callbacks
from .base import default_callbacks, get_default_callbacks


def add_integration_callbacks(instance) -> None:
    """Register opt-in routing telemetry before the standard logger integrations."""
    telemetry = getattr(instance, "training_telemetry", None)
    routing_enabled = "Trainer" in instance.__class__.__name__ and bool(getattr(telemetry, "enabled", False))
    if routing_enabled:
        from .routing import callbacks as routing_callbacks

        for event, callback in routing_callbacks.items():
            if callback not in instance.callbacks[event]:
                instance.callbacks[event].append(callback)
    _add_integration_callbacks(instance)
    if routing_enabled:
        from .routing import on_fit_epoch_end_restore

        if on_fit_epoch_end_restore not in instance.callbacks["on_fit_epoch_end"]:
            instance.callbacks["on_fit_epoch_end"].append(on_fit_epoch_end_restore)


__all__ = "add_integration_callbacks", "default_callbacks", "get_default_callbacks"
