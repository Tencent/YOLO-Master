"""Validation integration for the D1 cached-feature detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ultralytics.data import build_dataloader
from ultralytics.data.d1_cache import D1FeatureCacheDataset, FeatureProvider, move_d1_batch_to_device
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import colorstr


class D1FoundationDetectionValidator(DetectionValidator):
    """Evaluate D1 predictions with cached features and original COCO geometry."""

    def __init__(
        self,
        dataloader=None,
        save_dir=None,
        args=None,
        _callbacks: dict | None = None,
        *,
        feature_cache: str | Path,
        feature_mode: str = "cache",
        online_feature_provider: FeatureProvider | None = None,
        trusted_cache: bool = False,
        max_open_shards: int = 0,
        prefetch_factor: int = 4,
    ) -> None:
        self.feature_cache = Path(feature_cache).expanduser()
        if not self.feature_cache.is_dir():
            raise FileNotFoundError(self.feature_cache)
        if feature_mode not in {"cache", "online"}:
            raise ValueError("feature_mode must be 'cache' or 'online'.")
        if feature_mode == "online" and not callable(online_feature_provider):
            raise ValueError("online mode requires an online_feature_provider callable.")
        self.feature_mode = feature_mode
        self.online_feature_provider = online_feature_provider
        if type(trusted_cache) is not bool:
            raise TypeError("trusted_cache must be a boolean.")
        if type(max_open_shards) is not int or max_open_shards < 0:
            raise ValueError("max_open_shards must be a non-negative integer.")
        if type(prefetch_factor) is not int or prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be a positive integer.")
        self.trusted_cache = trusted_cache
        self.max_open_shards = max_open_shards
        self.prefetch_factor = prefetch_factor
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)

    def __call__(self, trainer=None, model=None):
        """Run cache validation through a Trainer, which bypasses AutoBackend RGB warmup."""
        if trainer is None:
            raise RuntimeError(
                "D1 WP5 validation must be driven by D1FoundationDetectionTrainer; "
                "standalone AutoBackend validation assumes RGB warmup inputs."
            )
        return super().__call__(trainer=trainer, model=model)

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        """Build the same cache-aware dataset used by D1 training validation."""
        if mode != "val":
            raise ValueError("D1FoundationDetectionValidator only builds the val split.")
        return D1FeatureCacheDataset(
            img_path=img_path,
            cache_dir=self.feature_cache,
            data=self.data,
            feature_mode=self.feature_mode,
            online_feature_provider=self.online_feature_provider,
            trusted_cache=self.trusted_cache and self.feature_mode == "cache",
            max_open_shards=self.max_open_shards,
            prefetch_factor=self.prefetch_factor,
            imgsz=self.args.imgsz,
            batch_size=batch or self.args.batch,
            hyp=self.args,
            prefix=colorstr("val: "),
            single_cls=self.args.single_cls or False,
            classes=self.args.classes,
            fraction=1.0,
            stride=32,
        )

    def get_dataloader(self, dataset_path: str, batch_size: int) -> torch.utils.data.DataLoader:
        """Build a deterministic cache dataloader without rectangular image batching."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(
            dataset,
            batch_size,
            self.args.workers,
            shuffle=False,
            rank=-1,
            drop_last=False,
            pin_memory=self.training,
        )

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move cached features and labels without treating features as RGB pixels."""
        use_fp16 = self.device.type == "cuda" and self.args.quantize == 16
        return move_d1_batch_to_device(
            batch,
            self.device,
            feature_dtype=torch.float16 if use_fp16 else torch.float32,
        )

    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Cached features contain no displayable RGB image."""

    def plot_predictions(self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int) -> None:
        """Cached features contain no displayable RGB image."""


__all__ = ["D1FoundationDetectionValidator"]
