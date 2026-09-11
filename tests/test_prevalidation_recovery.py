"""Exercise epoch replay when callbacks corrupt online state before validation."""

from collections import Counter

import numpy as np
import pytest
import torch
from PIL import Image

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model


@pytest.mark.parametrize("persistent", [False, True])
def test_prevalidation_recovery_replays_or_exhausts_budget(tmp_path, persistent):
    """Discard rolled-back attempts, preserve recovery limits and reset optimizer accumulation."""
    for split in ("train", "val"):
        images, labels = tmp_path / "images" / split, tmp_path / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for index in range(4):
            pixels = np.full((32, 32, 3), 40 + index * 30, dtype=np.uint8)
            Image.fromarray(pixels).save(images / f"{index}.jpg")
            (labels / f"{index}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    data = tmp_path / "data.yaml"
    data.write_text(f"path: {tmp_path.as_posix()}\ntrain: images/train\nval: images/val\nnames: [object]\n")
    trainer = DetectionTrainer(
        overrides={
            "model": "yolo11n.yaml",
            "data": str(data),
            "imgsz": 32,
            "epochs": 1,
            "batch": 2,
            "nbs": 2,
            "workers": 0,
            "device": "cpu",
            "amp": False,
            "mosaic": 0.0,
            "close_mosaic": 0,
            "optimizer": "SGD",
            "warmup_epochs": 0,
            "plots": False,
            "project": str(tmp_path / "runs"),
            "name": "recovery",
        }
    )
    trainer.final_eval = lambda: None
    attempts, finalized, validated, saved = [], [], [], []
    steps = Counter()
    original_step, original_validate = trainer.optimizer_step, trainer.validate
    original_save, original_finalize = trainer.save_metrics, trainer._finalize_moe_map_saturation_epoch

    def step():
        steps[len(attempts)] += 1
        return original_step()

    def validate():
        validated.append(len(attempts))
        return original_validate()

    def save(metrics):
        saved.append((len(attempts), trainer.epoch))
        return original_save(metrics)

    def finalize(**kwargs):
        finalized.append(kwargs)
        return original_finalize(**kwargs)

    def poison(t):
        if persistent or len(attempts) == 1:
            with torch.no_grad():
                next(unwrap_model(t.model).parameters()).flatten()[0] = float("nan")

    trainer.optimizer_step, trainer.validate = step, validate
    trainer.save_metrics, trainer._finalize_moe_map_saturation_epoch = save, finalize
    trainer.add_callback("on_train_epoch_start", lambda t: attempts.append(t.epoch))
    trainer.add_callback("on_train_epoch_end", poison)
    if persistent:
        with pytest.raises(RuntimeError, match="NaN persisted"):
            trainer.train()
        assert attempts == [0, 0, 0, 0]
        assert trainer.nan_recovery_attempts == 4
        assert not saved and not validated
        assert finalized == [{"recovered": True, "validated": False}] * 3
    else:
        trainer.train()
        assert attempts == [0, 0]
        assert steps == {1: 2, 2: 2}
        assert validated == [2] and saved == [(2, 0)]
        assert finalized == [{"recovered": True, "validated": False}, {"recovered": False, "validated": True}]
        assert trainer.nan_recovery_attempts == 0
        assert len(trainer.csv.read_text().splitlines()) == 2
