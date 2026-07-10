"""DDP-safe trainer callbacks for the reproduction scripts."""

from __future__ import annotations

import os

from ultralytics.models.yolo.detect import DetectionTrainer

from _reproduce_common import _make_dense_inference_callback, _make_wandb_callbacks


class ReproductionTrainer(DetectionTrainer):
    """Detection trainer that preserves reproduction callbacks under DDP."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if os.getenv("YOLO_MASTER_REPRO_DENSE_EVAL") == "1":
            callback = _make_dense_inference_callback()
            self.add_callback("on_pretrain_routine_end", callback)
            self.add_callback("on_train_start", callback)
        if os.getenv("YOLO_MASTER_REPRO_WANDB") == "1":
            for event, callback in _make_wandb_callbacks().items():
                self.add_callback(event, callback)
