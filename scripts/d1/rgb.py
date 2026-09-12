"""Fixed-geometry RGB control and official prediction export for D1 comparisons."""

from __future__ import annotations

import math
from copy import copy, deepcopy
from pathlib import Path

import cv2
import torch

from ultralytics.data.augment import Compose, Format, LetterBox
from ultralytics.data.converter import coco80_to_coco91_class
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.modules.block import Attention
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.torch_utils import unwrap_model

MODEL_CFG = Path(__file__).resolve().parents[2] / "ultralytics/cfg/models/26/yolo26-d1-scratch-total-l.yaml"
PARAMETER_COUNTS = {10: 23_032_340, 80: 23_133_560}


def scratch_config(cfg=None, *, nc):
    """Accept only the registered total-matched topology, with dataset-specific classes."""
    if type(nc) is not int or nc not in PARAMETER_COUNTS:
        raise ValueError("Scratch requires nc=10 or nc=80")
    expected = YAML.load(MODEL_CFG)
    supplied = deepcopy(cfg) if isinstance(cfg, dict) else YAML.load(cfg or MODEL_CFG)
    metadata = {"nc", "channels", "yaml_file"}
    if {k: v for k, v in supplied.items() if k not in metadata} != {
        k: v for k, v in expected.items() if k not in metadata
    }:
        raise ValueError("Scratch requires the exact registered total-matched YOLO26 configuration")
    if supplied.get("channels", 3) != 3:
        raise ValueError("Scratch requires three RGB channels")
    expected["nc"] = nc
    return expected


def build_model(cfg=None, *, nc, verbose=False):
    """Apply the explicit Scratch precision policy without changing its native model class."""
    config = scratch_config(cfg, nc=nc)
    model = DetectionModel(config, nc=nc, ch=3, verbose=verbose)
    for module in model.modules():
        if isinstance(module, Attention):
            module.fp32_attention = config["fp32_attention"]
    audit_model(model)
    return model


def audit_model(model):
    """Check actual parameter sizes and the standard end-to-end detection head."""
    if type(model) is not DetectionModel:
        raise TypeError("Scratch requires a standard DetectionModel")
    head = model.model[-1]
    scratch_config(model.yaml, nc=head.nc)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total != PARAMETER_COUNTS[head.nc] or trainable != total:
        raise ValueError(f"Scratch parameter sizes differ: {total} total / {trainable} trainable")
    if (head.reg_max, head.end2end) != (1, True) or tuple(model.stride.tolist()) != (8, 16, 32):
        raise ValueError("Scratch requires the registered YOLO26 detection head and strides")
    attention = [module for module in model.modules() if isinstance(module, Attention)]
    if len(attention) != 3 or any(getattr(module, "fp32_attention", False) is not True for module in attention):
        raise ValueError("Scratch requires FP32 math in all three attention modules")
    return {"nc": head.nc, "total_parameters": total, "trainable_parameters": trainable, "fp32_attention": True}


class ScratchDataset(YOLODataset):
    """Decode original images and apply exactly one centered 640-square LetterBox."""

    prefetch_factor = 1

    def build_transforms(self, hyp=None):
        """Use the cached-feature geometry for both train and validation labels."""
        if self.imgsz != 640 or self.rect or self.augment or self.cache:
            raise ValueError("Scratch requires imgsz=640, rect=False, augment=False and cache=False")
        if self.channels != 3 or self.use_segments or self.use_keypoints or self.use_obb:
            raise ValueError("Scratch requires three-channel RGB detection data")
        return Compose(
            [
                LetterBox(
                    (640, 640),
                    auto=False,
                    scale_fill=False,
                    scaleup=True,
                    center=True,
                    stride=32,
                    padding_value=114,
                    interpolation=cv2.INTER_LINEAR,
                ),
                Format(bbox_format="xywh", normalize=True, batch_idx=True, bgr=0.0),
            ]
        )

    def load_image(self, i, rect_mode=True, resize_short=False):
        """Bypass the base dataset's preliminary resize and image buffer."""
        image = cv2.imread(self.im_files[i], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.im_files[i])
        return image, image.shape[:2], image.shape[:2]

    def __getitem__(self, index):
        """Retain the exact gain and integer padding needed for original-image boxes."""
        result = super().__getitem__(index)
        height, width = result["ori_shape"]
        gain = min(640 / height, 640 / width)
        left = round((640 - round(width * gain)) / 2 - 0.1)
        top = round((640 - round(height * gain)) / 2 - 0.1)
        result["ratio_pad"] = ((gain, gain), (left, top))
        return result


def build_dataset(args, data, img_path, batch):
    """Build a fixed square RGB dataset with the same label selection as cached D1."""
    if args.imgsz != 640 or getattr(args, "multi_scale", 0.0) != 0.0:
        raise ValueError("Scratch requires imgsz=640 and multi_scale=0")
    return ScratchDataset(
        img_path=img_path,
        data=data,
        imgsz=640,
        batch_size=batch,
        augment=False,
        hyp=args,
        rect=False,
        cache=False,
        stride=32,
        pad=0.0,
        task="detect",
        single_cls=getattr(args, "single_cls", False),
        classes=getattr(args, "classes", None),
        fraction=1.0,
    )


class ScratchValidator(DetectionValidator):
    """Use fixed square RGB geometry during training and independent validation."""

    def build_dataset(self, img_path, mode="val", batch=None):
        return build_dataset(self.args, self.data, img_path, batch or self.args.batch)


class ScratchTrainer(DetectionTrainer):
    """Train the random total-matched control using the shared lifecycle and native EMA."""

    def resolve_ddp_policy(self):
        """Use the stable dense graph without unused-parameter traversal."""
        return False, True

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Create the exact model; only an explicit resume may restore all state keys."""
        if weights is not None and not self.resume:
            raise ValueError("Pretrained weights are forbidden for the scratch control")
        if self.data.get("channels", 3) != 3:
            raise ValueError("Scratch requires three RGB channels")
        model = build_model(cfg, nc=self.data["nc"], verbose=verbose)
        if weights is not None:
            if type(weights) is not DetectionModel:
                raise TypeError("Scratch resume weights must contain a standard DetectionModel")
            scratch_config(weights.yaml, nc=self.data["nc"])
            if weights.model[-1].nc != self.data["nc"]:
                raise ValueError("Scratch resume classes do not match the dataset")
            state = weights.state_dict()
            if any(not bool(torch.isfinite(value).all()) for value in state.values()):
                raise FloatingPointError("Scratch resume contains nonfinite state")
            model.load_state_dict(state, strict=True)
        return self.set_model_names_for_load(model)

    def build_dataset(self, img_path, mode="train", batch=None):
        return build_dataset(self.args, self.data, img_path, batch or self.args.batch)

    def get_validator(self):
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
        return ScratchValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def check_amp_compatibility(self):
        """Check the actual local model without fetching an unrelated AMP test model."""
        if self.device.type != "cuda":
            return False
        model = unwrap_model(self.model)
        was_training = model.training
        try:
            model.eval()
            sample = torch.linspace(0, 1, 3 * 640 * 640, device=self.device).reshape(1, 3, 640, 640)
            with torch.inference_mode():
                fp32 = model(sample)[0]
                with torch.autocast("cuda", dtype=torch.float16):
                    fp16 = model(sample)[0]
            if fp32.shape != fp16.shape or not torch.isfinite(fp32).all() or not torch.isfinite(fp16).all():
                raise FloatingPointError("Scratch AMP forward is not finite or changes shape")
        except (RuntimeError, FloatingPointError) as error:
            # Return to the shared AMP broadcast so peer ranks receive the same decision.
            LOGGER.warning(f"AMP: Scratch local forward check failed: {error}")
            return False
        finally:
            model.train(was_training)
        return True


class ExportMixin:
    """Apply the D1 official export mapping without coupling RGB to the run entry point."""

    def __init__(self, *args, dataset_kind, **kwargs):
        if dataset_kind not in {"coco", "visdrone"}:
            raise ValueError("dataset_kind must be coco or visdrone")
        self.dataset_kind = dataset_kind
        self.degenerate_boxes_removed = 0
        super().__init__(*args, **kwargs)

    def init_metrics(self, model):
        super().init_metrics(model)
        self.is_coco = self.dataset_kind == "coco"
        self.is_lvis = False
        self.class_map = coco80_to_coco91_class() if self.is_coco else list(range(10))
        self.args.save_json = True
        self.degenerate_boxes_removed = 0

    def eval_json(self, stats):
        """Leave official scoring to an explicit caller, independent of dataset path names."""
        return stats

    def pred_to_json(self, predn, pbatch):
        start = len(self.jdict)
        super().pred_to_json(predn, pbatch)
        if self.dataset_kind == "visdrone":
            exported = []
            for row in self.jdict[start:]:
                row["image_id"] = Path(pbatch["im_file"]).stem
                if not all(math.isfinite(value) for value in (*row["bbox"], row["score"])):
                    raise FloatingPointError("Nonfinite prediction in VisDrone export")
                if not 0 <= row["score"] <= 1:
                    raise ValueError("Invalid confidence in VisDrone export")
                # Early regression outputs or clipping can yield nonpositive boxes.
                if row["bbox"][2] <= 0 or row["bbox"][3] <= 0:
                    self.degenerate_boxes_removed += 1
                else:
                    exported.append(row)
            self.jdict[start:] = exported


class ExportRGBValidator(ExportMixin, ScratchValidator):
    """Export RGB predictions with the same COCO and VisDrone semantics as cached D1."""
