"""Optional, process-local training cooperation; no lifecycle decisions or runtime imports."""

from __future__ import annotations

import os
import tempfile
import threading
from contextvars import ContextVar
from pathlib import Path

CURRENT_SHUTDOWN = ContextVar("studio_training_shutdown", default=None)
CHECKPOINT_ID = "weights/shutdown.pt"


class TrainingShutdown:
    """Carry stop/ack facts from IPC to the training thread's safe callback."""

    def __init__(self, emit, output_root=None, job_id=None):
        self.requested = threading.Event()
        self.acknowledged = threading.Event()
        self.emit = emit
        self.saved = False
        self.output_root = Path(output_root) if output_root is not None else None
        self.job_id = job_id

    def on_epoch_end(self, trainer):
        """Preserve an unstripped epoch checkpoint, then wait for parent authorization."""
        if not self.requested.is_set() or self.saved:
            return
        # An outer model callback cannot coordinate DDP ranks or collectives.
        if getattr(trainer, "world_size", 1) > 1 or int(os.environ.get("WORLD_SIZE", "1")) > 1:
            return
        # on_fit_epoch_end runs after epoch metrics/validation and before final_eval.
        root = Path(trainer.save_dir).resolve(strict=True)
        if self.output_root is not None:
            base = self.output_root.resolve(strict=True)
            expected = (base / self.job_id).resolve(strict=True)
            expected.relative_to(base)
            if root != expected:
                raise ValueError("Trainer output does not match the owned job directory")
        target = root / CHECKPOINT_ID
        target.parent.mkdir(parents=True, exist_ok=True)
        target.parent.resolve(strict=True).relative_to(root)
        # Keep the online model aligned with optimizer state, plus EMA/scaler,
        # even when normal saving was disabled. final_eval may strip last/best.
        serialized = trainer._serialize_checkpoint(include_online_model=True)
        descriptor, name = tempfile.mkstemp(prefix="shutdown-", suffix=".pt.tmp", dir=target.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as outgoing:
                outgoing.write(serialized)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        self.saved = True
        self.emit("checkpoint", {"path": CHECKPOINT_ID, "epoch": trainer.epoch, "size": target.stat().st_size})
        # Only the watcher sets this token. It never touches the trainer or saves.
        # The parent enforces the absolute grace deadline even if ack never arrives.
        self.acknowledged.wait()
        trainer.stop = True


def install_training_shutdown(model, device):
    """Install only for managed single-process training; direct Core calls stay unchanged."""
    control = CURRENT_SHUTDOWN.get()
    if control is None:
        return
    devices = device if isinstance(device, (list, tuple)) else str(device).split(",")
    if len(devices) > 1 or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        control.emit(
            "shutdown_limit", {"reason": "DDP requires forced bounded cleanup; checkpoint cooperation unsupported"}
        )
        return
    model.add_callback("on_fit_epoch_end", control.on_epoch_end)
