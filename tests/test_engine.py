# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tests import MODEL
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.engine import trainer as trainer_module
from ultralytics.engine import validator as validator_module
from ultralytics.engine.exporter import Exporter
from ultralytics.models.yolo import classify, detect, segment
from ultralytics.utils import ASSETS, DEFAULT_CFG, WEIGHTS_DIR, YAML


def test_func(*args):
    """Test function callback for evaluating YOLO model performance metrics."""
    print("callback test passed")


def test_export():
    """Test model exporting functionality by adding a callback and verifying its execution."""
    exporter = Exporter()
    exporter.add_callback("on_export_start", test_func)
    assert test_func in exporter.callbacks["on_export_start"], "callback test failed"
    f = exporter(model=YOLO("yolo11n.yaml").model)
    YOLO(f)(ASSETS)  # exported model inference


def test_save_trainer_args_yaml_persists_runtime_lora_total_step(tmp_path):
    """Test that saved trainer args reflect runtime-updated AdaLoRA total_step."""
    args = SimpleNamespace(augmentations=None, lora_total_step=7, save_dir=str(tmp_path))

    trainer_module.save_trainer_args_yaml(tmp_path, args)

    saved = YAML.load(tmp_path / "args.yaml")
    assert saved["lora_total_step"] == 7


def test_save_trainer_args_yaml_serializes_augmentations_repr(tmp_path):
    """Test that saved trainer args serialize augmentation objects via repr for resumability."""

    class _DummyAug:
        def __repr__(self):
            return "DummyAug(p=0.5)"

    args = SimpleNamespace(augmentations=[_DummyAug()], lora_total_step=1, save_dir=str(tmp_path))

    trainer_module.save_trainer_args_yaml(tmp_path, args)

    saved = YAML.load(tmp_path / "args.yaml")
    assert saved["augmentations"] == ["DummyAug(p=0.5)"]


def test_save_trainer_args_yaml_persists_effective_lora_backend(tmp_path):
    args = SimpleNamespace(
        augmentations=None,
        lora_total_step=7,
        requested_lora_backend="auto",
        effective_lora_backend="fallback",
        requested_lora_init_lora_weights="pissa",
        effective_lora_init_lora_weights="gaussian",
        save_dir=str(tmp_path),
    )

    trainer_module.save_trainer_args_yaml(tmp_path, args)

    saved = YAML.load(tmp_path / "args.yaml")
    assert saved["requested_lora_backend"] == "auto"
    assert saved["effective_lora_backend"] == "fallback"
    assert saved["requested_lora_init_lora_weights"] == "pissa"
    assert saved["effective_lora_init_lora_weights"] == "gaussian"


def test_update_args_with_lora_runtime_metadata_sets_requested_and_effective_fields():
    args = SimpleNamespace()
    model = SimpleNamespace(
        lora_runtime_metadata={
            "requested_backend": "auto",
            "effective_backend": "fallback",
            "requested_init_lora_weights": "pissa",
            "effective_init_lora_weights": "gaussian",
            "safety_profile": "rtdetr_lora",
            "safety_overrides": {"lora_lr_mult": {"from": 2.0, "to": 1.0}},
        }
    )

    trainer_module.update_args_with_lora_runtime_metadata(args, model)

    assert args.requested_lora_backend == "auto"
    assert args.effective_lora_backend == "fallback"
    assert args.requested_lora_init_lora_weights == "pissa"
    assert args.effective_lora_init_lora_weights == "gaussian"
    assert args.lora_safety_profile == "rtdetr_lora"
    assert args.lora_safety_overrides == {"lora_lr_mult": {"from": 2.0, "to": 1.0}}


def test_convert_ndjson_to_yolo_if_needed_leaves_yaml_unchanged():
    """Test that validator dataset conversion is a no-op for YAML paths."""
    assert validator_module.convert_ndjson_to_yolo_if_needed("coco8.yaml") == "coco8.yaml"


def test_convert_ndjson_to_yolo_if_needed_converts_ndjson(monkeypatch, tmp_path):
    """Test that validator dataset conversion resolves NDJSON paths to generated YOLO YAML paths."""
    converted_yaml = tmp_path / "coco8-ndjson" / "data.yaml"

    async def fake_convert_ndjson_to_yolo(path):
        assert path == "coco8-ndjson.ndjson"
        return converted_yaml

    monkeypatch.setattr(validator_module, "convert_ndjson_to_yolo", fake_convert_ndjson_to_yolo)

    actual = validator_module.convert_ndjson_to_yolo_if_needed("coco8-ndjson.ndjson")
    assert Path(actual).resolve() == converted_yaml.resolve()


def test_rtdetr_lora_safety_guard_mutates_training_args():
    from ultralytics.utils.lora import LoRAConfig, _apply_rtdetr_lora_safety

    class RTDETRDecoder(torch.nn.Module):
        pass

    args = SimpleNamespace(amp=True, lora_alpha_warmup=0, lora_lr_mult=2.0, lora_include_attention=False)
    config = LoRAConfig(
        r=16,
        alpha=32,
        lr_mult=2.0,
        alpha_warmup=0,
        include_attention=False,
        use_dora=True,
    )
    kwargs = {}

    changes = _apply_rtdetr_lora_safety(torch.nn.Sequential(RTDETRDecoder()), args, config, kwargs)

    assert args.amp is True
    assert args.lora_alpha_warmup == 3
    assert args.lora_lr_mult == 1.0
    assert args.lora_use_dora is False
    assert args.lora_include_attention is True
    assert config.alpha_warmup == 3
    assert config.lr_mult == 1.0
    assert config.use_dora is False
    assert config.include_attention is True
    assert "amp" not in kwargs
    assert kwargs["lora_alpha_warmup"] == 3
    assert kwargs["lora_lr_mult"] == 1.0
    assert kwargs["lora_use_dora"] is False
    assert kwargs["lora_include_attention"] is True
    assert changes == {
        "lora_alpha_warmup": {"from": 0, "to": 3},
        "lora_lr_mult": {"from": 2.0, "to": 1.0},
        "lora_include_attention": {"from": False, "to": True},
        "lora_use_dora": {"from": True, "to": False},
    }


def test_build_optimizer_separates_lora_params_with_lr_multiplier():
    class _AdapterBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))
            self.bias = torch.nn.Parameter(torch.zeros(2))
            self.lora_A = torch.nn.Parameter(torch.ones(1, 2))
            self.lora_B = torch.nn.Parameter(torch.ones(2, 1))

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = _AdapterBlock()
            self.bn = torch.nn.BatchNorm1d(2)

    trainer = trainer_module.BaseTrainer.__new__(trainer_module.BaseTrainer)
    trainer.args = SimpleNamespace(lora_lr_mult=3.0)
    trainer.data = {}
    model = _TinyModel()

    optimizer = trainer_module.BaseTrainer.build_optimizer(
        trainer, model, name="SGD", lr=0.01, momentum=0.9, decay=1e-4
    )

    lora_params = {id(model.block.lora_A), id(model.block.lora_B)}
    lora_groups = [
        pg for pg in optimizer.param_groups
        if {id(param) for param in pg["params"]} == lora_params
    ]

    assert len(lora_groups) == 1
    assert lora_groups[0]["lr"] == pytest.approx(0.03)
    assert lora_groups[0]["initial_lr"] == pytest.approx(0.03)
    assert lora_groups[0]["weight_decay"] == 0.0


def test_detect():
    """Test YOLO object detection training, validation, and prediction functionality."""
    overrides = {"data": "coco8.yaml", "model": "yolo11n.yaml", "imgsz": 32, "epochs": 1, "save": False}
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = "coco8.yaml"
    cfg.imgsz = 32

    # Trainer
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "callback test failed"
    trainer.train()

    # Validator
    val = detect.DetectionValidator(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "callback test failed"
    val(model=trainer.best)  # validate best.pt

    # Predictor
    pred = detect.DetectionPredictor(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "callback test failed"
    # Confirm there is no issue with sys.argv being empty
    with mock.patch.object(sys, "argv", []):
        result = pred(source=ASSETS, model=MODEL)
        assert len(result), "predictor test failed"

    # Test resume functionality
    overrides["resume"] = trainer.last
    trainer = detect.DetectionTrainer(overrides=overrides)
    try:
        trainer.train()
    except Exception as e:
        print(f"Expected exception caught: {e}")
        return

    raise Exception("Resume test failed!")


def test_segment():
    """Test image segmentation training, validation, and prediction pipelines using YOLO models."""
    overrides = {
        "data": "coco8-seg.yaml",
        "model": "yolo11n-seg.yaml",
        "imgsz": 32,
        "epochs": 1,
        "save": False,
        "mask_ratio": 1,
        "overlap_mask": False,
    }
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = "coco8-seg.yaml"
    cfg.imgsz = 32

    # Trainer
    trainer = segment.SegmentationTrainer(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "callback test failed"
    trainer.train()

    # Validator
    val = segment.SegmentationValidator(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "callback test failed"
    val(model=trainer.best)  # validate best.pt

    # Predictor
    pred = segment.SegmentationPredictor(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "callback test failed"
    result = pred(source=ASSETS, model=WEIGHTS_DIR / "yolo11n-seg.pt")
    assert len(result), "predictor test failed"

    # Test resume functionality
    overrides["resume"] = trainer.last
    trainer = segment.SegmentationTrainer(overrides=overrides)
    try:
        trainer.train()
    except Exception as e:
        print(f"Expected exception caught: {e}")
        return

    raise Exception("Resume test failed!")


def test_classify():
    """Test image classification including training, validation, and prediction phases."""
    overrides = {"data": "imagenet10", "model": "yolo11n-cls.yaml", "imgsz": 32, "epochs": 1, "save": False}
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = "imagenet10"
    cfg.imgsz = 32

    # Trainer
    trainer = classify.ClassificationTrainer(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "callback test failed"
    trainer.train()

    # Validator
    val = classify.ClassificationValidator(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "callback test failed"
    val(model=trainer.best)

    # Predictor
    pred = classify.ClassificationPredictor(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "callback test failed"
    result = pred(source=ASSETS, model=trainer.best)
    assert len(result), "predictor test failed"


def test_nan_recovery():
    """Test NaN loss detection and recovery during training."""
    nan_injected = [False]

    def inject_nan(trainer):
        """Inject NaN into loss during batch processing to test recovery mechanism."""
        if trainer.epoch == 1 and trainer.tloss is not None and not nan_injected[0]:
            trainer.tloss *= torch.tensor(float("nan"))
            nan_injected[0] = True

    overrides = {"data": "coco8.yaml", "model": "yolo11n.yaml", "imgsz": 32, "epochs": 3}
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_batch_end", inject_nan)
    trainer.train()
    assert nan_injected[0], "NaN injection failed"
