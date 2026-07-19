"""Build a small deterministic baseline for native Ultralytics v8.4.101."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports/migration/v8.4.101-native-baseline.json"
SEED = 260719
sys.path[:] = [item for item in sys.path if item != str(ROOT)]
sys.path.insert(0, str(ROOT))

import torch

from ultralytics.nn.tasks import (
    ClassificationModel,
    DetectionModel,
    OBBModel,
    PoseModel,
    SegmentationModel,
    SemanticSegmentationModel,
    YOLOEModel,
    YOLOESegModel,
)

MODEL_CASES = (
    ("detect", "ultralytics/cfg/models/26/yolo26.yaml", DetectionModel, 80),
    ("segment", "ultralytics/cfg/models/26/yolo26-seg.yaml", SegmentationModel, 80),
    ("semantic", "ultralytics/cfg/models/26/yolo26-sem.yaml", SemanticSegmentationModel, 19),
    ("pose", "ultralytics/cfg/models/26/yolo26-pose.yaml", PoseModel, 1),
    ("obb", "ultralytics/cfg/models/26/yolo26-obb.yaml", OBBModel, 15),
    ("classify", "ultralytics/cfg/models/26/yolo26-cls.yaml", ClassificationModel, 1000),
    ("yoloe", "ultralytics/cfg/models/26/yoloe-26.yaml", YOLOEModel, 80),
    ("yoloe-seg", "ultralytics/cfg/models/26/yoloe-26-seg.yaml", YOLOESegModel, 80),
)


def _seed() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)


def _tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in _tensors(item)]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in _tensors(item)]
    return []


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().float().cpu().contiguous()
    return {
        "shape": list(tensor.shape),
        "mean": float(tensor.mean().item()) if tensor.numel() else 0.0,
        "std": float(tensor.std(unbiased=False).item()) if tensor.numel() else 0.0,
        "checksum": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
    }


def _model_record(name: str, config: str, model_cls: type, nc: int) -> dict[str, Any]:
    _seed()
    model = model_cls(config, ch=3, nc=nc, verbose=False)
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 64, 64))
    tensors = _tensors(output)
    head = model.model[-1]
    return {
        "name": name,
        "config": config,
        "model_class": model_cls.__name__,
        "head_class": type(head).__name__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "stride": _tensor_summary(model.stride),
        "reg_max": getattr(head, "reg_max", None),
        "end2end": getattr(model, "end2end", None),
        "one2many": hasattr(head, "one2many"),
        "one2one": hasattr(head, "one2one"),
        "output_count": len(tensors),
        "output_shapes": [list(tensor.shape) for tensor in tensors],
        "outputs_finite": all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors),
    }


def build_baseline() -> dict[str, Any]:
    """Build the native model structure and runtime baseline."""
    return {
        "schema_version": 1,
        "source": {
            "ref": "v8.4.101",
            "commit": "579b389c87c04b7f6a9a247730dac04922be8007",
            "seed": SEED,
        },
        "models": [_model_record(*case) for case in MODEL_CASES],
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cpu": True,
            "mps": bool(torch.backends.mps.is_available()),
            "cuda": bool(torch.cuda.is_available()),
        },
        "export": {
            "formats": {
                "pytorch": {"native": True, "status": "available"},
                "torchscript": {"native": True, "status": "available"},
                "onnx": {"native": True, "status": "available"},
            },
            "note": "Artifact roundtrips run in the export phase when optional dependencies are installed.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    report = build_baseline()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
