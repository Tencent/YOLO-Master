# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .foundation_detection_model import D1FoundationDetectionModel
from .tasks import (
    BaseModel,
    ClassificationModel,
    DetectionModel,
    MultiTaskModel,
    SegmentationModel,
    guess_model_scale,
    guess_model_task,
    load_checkpoint,
    parse_model,
    torch_safe_load,
    yaml_model_load,
)

__all__ = (
    "BaseModel",
    "ClassificationModel",
    "D1FoundationDetectionModel",
    "DetectionModel",
    "MultiTaskModel",
    "SegmentationModel",
    "guess_model_scale",
    "guess_model_task",
    "load_checkpoint",
    "parse_model",
    "torch_safe_load",
    "yaml_model_load",
)
