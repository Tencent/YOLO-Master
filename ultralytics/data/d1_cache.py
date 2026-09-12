"""Dataset and batch utilities for D1 cached DINOv3 features."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.foundation.cache import FeatureCacheReader
from ultralytics.nn.foundation.npy_cache import open_feature_cache

D1_FEATURE_NAMES = ("block4", "block8", "block12")
D1_FEATURE_SHAPE = (384, 40, 40)
D1_IMAGE_SIZE = 640
D1_CACHE_DTYPE = torch.float16
FeatureProvider = Callable[[str], Mapping[str, torch.Tensor] | Any]


class D1FeatureBatch(dict[str, torch.Tensor]):
    """Feature mapping that exposes the virtual RGB shape expected by generic YOLO loops."""

    def __init__(self, features: Mapping[str, torch.Tensor], image_size: int = D1_IMAGE_SIZE) -> None:
        super().__init__(features)
        if tuple(self) != D1_FEATURE_NAMES:
            raise ValueError(f"D1 feature batch keys must be {D1_FEATURE_NAMES}, got {tuple(self)}.")
        if type(image_size) is not int or image_size <= 0:
            raise ValueError("image_size must be a positive integer.")
        if any(not isinstance(value, torch.Tensor) for value in self.values()):
            raise TypeError("D1 feature batch values must be tensors.")
        if any(value.ndim != 4 or tuple(value.shape[1:]) != D1_FEATURE_SHAPE for value in self.values()):
            raise ValueError(f"D1 batched features must have shape [B, {D1_FEATURE_SHAPE}].")
        batch_sizes = {value.shape[0] for value in self.values()}
        devices = {value.device for value in self.values()}
        dtypes = {value.dtype for value in self.values()}
        if len(batch_sizes) != 1 or len(devices) != 1 or len(dtypes) != 1:
            raise ValueError("D1 feature batch values must share batch size, device, and dtype.")
        if next(iter(batch_sizes)) <= 0:
            raise ValueError("D1 feature batch must not be empty.")
        self.image_size = image_size

    @property
    def shape(self) -> torch.Size:
        """Expose a virtual BCHW shape without allocating an RGB tensor."""
        batch_size = next(iter(self.values())).shape[0]
        return torch.Size((batch_size, 3, self.image_size, self.image_size))

    def to(
        self,
        device: str | torch.device,
        *,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> D1FeatureBatch:
        """Move every cached feature tensor while preserving the virtual image contract."""
        return type(self)(
            {name: value.to(device=device, dtype=dtype, non_blocking=non_blocking) for name, value in self.items()},
            image_size=self.image_size,
        )


class D1TrainingBatch(dict):
    """Store features once, with a virtual img alias for generic detector loops.

    DataLoader recursively pins mapping items. Storing both keys would allocate
    two pinned copies of the same large tensors before either reaches the GPU.
    """

    def __getitem__(self, key):
        return super().__getitem__("features" if key == "img" else key)

    def __setitem__(self, key, value):
        super().__setitem__("features" if key == "img" else key, value)

    def __contains__(self, key):
        return super().__contains__("features" if key == "img" else key)

    def get(self, key, default=None):
        return super().get("features" if key == "img" else key, default)


def _validate_cache_contract(reader: FeatureCacheReader) -> None:
    contract = reader.contract
    if tuple(contract["feature_names"]) != D1_FEATURE_NAMES:
        raise ValueError(f"D1 cache feature_names must be {D1_FEATURE_NAMES}, got {tuple(contract['feature_names'])}.")
    if tuple(contract["expected_shape"]) != D1_FEATURE_SHAPE:
        raise ValueError(
            f"D1 cache expected_shape must be {D1_FEATURE_SHAPE}, got {tuple(contract['expected_shape'])}."
        )
    if contract["dtype"] != "float16":
        raise ValueError(f"D1 cache dtype must be 'float16', got {contract['dtype']!r}.")
    if tuple(contract["output_layers"]) != (4, 8, 12):
        raise ValueError("D1 cache output_layers must be (4, 8, 12).")


def _sample_id(im_file: str) -> str:
    path = Path(im_file)
    split = path.parent.name
    if split not in {"train2017", "val2017", "visdrone-train", "visdrone-val"}:
        raise ValueError(f"D1 image must be below train2017/val2017 or visdrone-train/visdrone-val, got {im_file!r}.")
    return f"{split}/{path.stem}"


def _letterbox_geometry(shape: tuple[int, int], image_size: int) -> tuple[float, int, int]:
    """Return the exact WP0 centered LetterBox gain and top-left padding."""
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid original image shape {shape}.")
    gain = min(image_size / height, image_size / width)
    new_width, new_height = round(width * gain), round(height * gain)
    left = round((image_size - new_width) / 2 - 0.1)
    top = round((image_size - new_height) / 2 - 0.1)
    return gain, left, top


def _letterbox_boxes(
    bboxes: np.ndarray,
    shape: tuple[int, int],
    image_size: int,
) -> tuple[torch.Tensor, tuple[tuple[float, float], tuple[int, int]]]:
    """Map normalized original-image xywh labels to normalized WP0 letterbox coordinates."""
    gain, left, top = _letterbox_geometry(shape, image_size)
    boxes = torch.as_tensor(np.asarray(bboxes), dtype=torch.float32).reshape(-1, 4).clone()
    height, width = shape
    if boxes.numel():
        boxes[:, 0] = (boxes[:, 0] * width * gain + left) / image_size
        boxes[:, 1] = (boxes[:, 1] * height * gain + top) / image_size
        boxes[:, 2] = boxes[:, 2] * width * gain / image_size
        boxes[:, 3] = boxes[:, 3] * height * gain / image_size
    return boxes, ((gain, gain), (left, top))


def _validated_features(
    value: Mapping[str, torch.Tensor] | Any,
    *,
    sample_id: str,
    check_finite: bool = True,
) -> dict[str, torch.Tensor]:
    dense = value.dense if hasattr(value, "dense") else value
    if not isinstance(dense, Mapping):
        raise TypeError(f"feature provider for {sample_id!r} must return a mapping or FoundationFeatures.")
    if set(dense) != set(D1_FEATURE_NAMES):
        raise ValueError(f"features for {sample_id!r} must contain exactly {D1_FEATURE_NAMES}, got {tuple(dense)}.")
    result = {}
    for name in D1_FEATURE_NAMES:
        tensor = dense[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"feature {name!r} for {sample_id!r} must be a tensor.")
        if tuple(tensor.shape) != D1_FEATURE_SHAPE:
            raise ValueError(
                f"feature {name!r} for {sample_id!r} must have shape {D1_FEATURE_SHAPE}, got {tuple(tensor.shape)}."
            )
        if tensor.dtype != D1_CACHE_DTYPE:
            raise TypeError(f"feature {name!r} for {sample_id!r} must be float16, got {tensor.dtype}.")
        if tensor.device.type != "cpu":
            raise ValueError(f"feature {name!r} for {sample_id!r} must be on CPU before collation.")
        if tensor.requires_grad:
            raise ValueError(f"feature {name!r} for {sample_id!r} must not require gradients.")
        if check_finite and not torch.isfinite(tensor).all():
            raise ValueError(f"feature {name!r} for {sample_id!r} contains NaN or Inf.")
        result[name] = tensor
    return result


class D1FeatureCacheDataset(YOLODataset):
    """Pair registered COCO or VisDrone labels with immutable DINOv3 features."""

    def __init__(
        self,
        *,
        img_path: str | list[str],
        cache_dir: str | Path,
        data: dict[str, Any],
        feature_mode: str = "cache",
        online_feature_provider: FeatureProvider | None = None,
        trusted_cache: bool = False,
        max_open_shards: int = 0,
        prefetch_factor: int = 4,
        imgsz: int = D1_IMAGE_SIZE,
        batch_size: int = 16,
        hyp: Any,
        prefix: str = "",
        single_cls: bool = False,
        classes: list[int] | None = None,
        fraction: float = 1.0,
        stride: int = 32,
    ) -> None:
        if imgsz != D1_IMAGE_SIZE:
            raise ValueError(f"D1 cached features require imgsz={D1_IMAGE_SIZE}, got {imgsz}.")
        if feature_mode not in {"cache", "online"}:
            raise ValueError("feature_mode must be 'cache' or 'online'.")
        if feature_mode == "online" and not callable(online_feature_provider):
            raise ValueError("online mode requires an online_feature_provider callable.")
        if type(trusted_cache) is not bool:
            raise TypeError("trusted_cache must be a boolean.")
        if type(prefetch_factor) is not int or prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be a positive integer.")
        self.feature_reader = open_feature_cache(cache_dir, max_open_shards=max_open_shards)
        _validate_cache_contract(self.feature_reader)
        self.feature_mode = feature_mode
        self.online_feature_provider = online_feature_provider
        self.trusted_cache = trusted_cache
        self.prefetch_factor = prefetch_factor
        self._finite_check_pid = os.getpid()
        self._finite_checked_shards: set[str] = set()
        super().__init__(
            img_path=img_path,
            imgsz=imgsz,
            batch_size=batch_size,
            augment=False,
            hyp=hyp,
            rect=False,
            cache=False,
            single_cls=single_cls,
            stride=stride,
            pad=0.0,
            prefix=prefix,
            task="detect",
            classes=classes,
            data=data,
            fraction=fraction,
        )
        self.sample_ids = tuple(_sample_id(path) for path in self.im_files)
        if any(sid.startswith("visdrone-") for sid in self.sample_ids) and data.get("nc") != 10:
            raise ValueError("VisDrone cached training requires nc=10.")
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError("D1 dataset contains duplicate sample IDs.")
        self._validate_cache_coverage()

    def _validate_cache_coverage(self) -> None:
        missing = [sample_id for sample_id in self.sample_ids if sample_id not in self.feature_reader.records]
        if missing:
            preview = ", ".join(missing[:3])
            raise FileNotFoundError(f"feature cache is missing {len(missing)} dataset samples: {preview}")
        for im_file, sample_id in zip(self.im_files, self.sample_ids):
            record = self.feature_reader.records[sample_id]
            expected_tail = (Path(im_file).parent.name, Path(im_file).name)
            actual_tail = PurePosixPath(record["image_path"]).parts[-2:]
            if actual_tail != expected_tail:
                raise ValueError(
                    f"cache record {sample_id!r} points to {record['image_path']!r}, expected suffix {expected_tail}."
                )
            if record["split"] != expected_tail[0]:
                raise ValueError(f"cache record {sample_id!r} has inconsistent split {record['split']!r}.")

    def _requires_finite_check(self, sample_id: str) -> tuple[bool, str]:
        record = self.feature_reader.records[sample_id]
        shard = record.get("source_shard") or record["shard"]
        pid = os.getpid()
        if pid != self._finite_check_pid:
            self._finite_check_pid = pid
            self._finite_checked_shards = set()
        return not self.trusted_cache or shard not in self._finite_checked_shards, shard

    def build_transforms(self, hyp: Any = None):
        """RGB transforms are intentionally absent because WP2 already fixed preprocessing."""
        return

    def _load_features(self, index: int) -> dict[str, torch.Tensor]:
        sample_id = self.sample_ids[index]
        if self.feature_mode == "cache":
            value = self.feature_reader.get(sample_id)
            check_finite, shard = self._requires_finite_check(sample_id)
            result = _validated_features(value, sample_id=sample_id, check_finite=check_finite)
            if self.trusted_cache and check_finite:
                self._finite_checked_shards.add(shard)
            return result
        value = self.online_feature_provider(self.im_files[index])
        return _validated_features(value, sample_id=sample_id, check_finite=True)

    def compare_online_with_cache(
        self,
        index: int,
        *,
        rtol: float = 0.0,
        atol: float = 0.0,
    ) -> dict[str, float]:
        """Compare one online extraction against its immutable cache entry."""
        if not callable(self.online_feature_provider):
            # Preserve the established error contract when online mode is unavailable.
            raise RuntimeError("compare_online_with_cache requires an online_feature_provider.")  # noqa: TRY004
        sample_id = self.sample_ids[index]
        cached = _validated_features(self.feature_reader.get(sample_id), sample_id=sample_id)
        online = _validated_features(self.online_feature_provider(self.im_files[index]), sample_id=sample_id)
        max_abs = {}
        for name in D1_FEATURE_NAMES:
            delta = (cached[name].float() - online[name].float()).abs()
            max_abs[name] = float(delta.max())
            if not torch.allclose(cached[name], online[name], rtol=rtol, atol=atol):
                raise ValueError(f"online and cached {name} differ for {sample_id!r}; max_abs={max_abs[name]:.6g}.")
        return max_abs

    def __getitem__(self, index: int) -> dict[str, Any]:
        label = deepcopy(self.labels[index])
        shape = tuple(int(value) for value in label.pop("shape"))
        boxes, ratio_pad = _letterbox_boxes(label["bboxes"], shape, self.imgsz)
        cls = torch.as_tensor(np.asarray(label["cls"]), dtype=torch.float32).reshape(-1, 1)
        if len(cls) != len(boxes):
            raise ValueError(f"class/box count mismatch for {self.sample_ids[index]!r}.")
        return {
            "features": self._load_features(index),
            "sample_id": self.sample_ids[index],
            "im_file": self.im_files[index],
            "ori_shape": shape,
            "resized_shape": (self.imgsz, self.imgsz),
            "ratio_pad": ratio_pad,
            "cls": cls,
            "bboxes": boxes,
            "batch_idx": torch.zeros(len(cls), dtype=torch.float32),
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not batch:
            raise ValueError("cannot collate an empty D1 batch.")
        feature_values = [sample["features"] for sample in batch]
        payload = [{key: value for key, value in sample.items() if key != "features"} for sample in batch]
        result = D1TrainingBatch(YOLODataset.collate_fn(payload))
        features = D1FeatureBatch(
            {name: torch.stack([value[name] for value in feature_values], dim=0) for name in D1_FEATURE_NAMES},
            image_size=D1_IMAGE_SIZE,
        )
        result["features"] = features
        result["img"] = features
        return result


def move_d1_batch_to_device(
    batch: dict[str, Any],
    device: str | torch.device,
    *,
    feature_dtype: torch.dtype,
) -> dict[str, Any]:
    """Move labels and cached features without applying RGB normalization."""
    features = batch.get("features")
    if not isinstance(features, D1FeatureBatch):
        if not isinstance(features, Mapping):
            raise TypeError("D1 batch must contain a 'features' mapping.")
        features = D1FeatureBatch(features)
    non_blocking = torch.device(device).type == "cuda"
    features = features.to(device, dtype=feature_dtype, non_blocking=non_blocking)
    for key, value in tuple(batch.items()):
        if key not in {"features", "img"} and isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=non_blocking)
    batch["features"] = features
    batch["img"] = features
    return batch


__all__ = [
    "D1_CACHE_DTYPE",
    "D1_FEATURE_NAMES",
    "D1_FEATURE_SHAPE",
    "D1_IMAGE_SIZE",
    "D1FeatureBatch",
    "D1FeatureCacheDataset",
    "D1TrainingBatch",
    "move_d1_batch_to_device",
]
