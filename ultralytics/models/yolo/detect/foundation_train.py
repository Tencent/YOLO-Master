"""Training integration for the D1 cached-feature detector."""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from pathlib import Path
from typing import Any

import torch

from ultralytics.data.d1_cache import D1FeatureCacheDataset, FeatureProvider, move_d1_batch_to_device
from ultralytics.models.yolo.detect.foundation_val import D1FoundationDetectionValidator
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.foundation_detection_model import (
    D1_AUX_REPORT_NAMES,
    DEFAULT_D1_MODEL_CFG,
    D1FoundationDetectionModel,
)
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import TORCH_2_4, strip_optimizer, torch_distributed_zero_first, unwrap_model

_DISABLED_AUGMENTATIONS = (
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
    "erasing",
)


def _cache_paths(value: Mapping[str, str | Path]) -> dict[str, Path]:
    if not isinstance(value, Mapping):
        raise TypeError("feature_caches must map 'train' and 'val' to cache directories.")
    if set(value) != {"train", "val"}:
        raise ValueError("feature_caches must contain exactly the 'train' and 'val' keys.")
    paths = {name: Path(path).expanduser() for name, path in value.items()}
    missing = [f"{name}={path}" for name, path in paths.items() if not path.is_dir()]
    if missing:
        raise FileNotFoundError("D1 feature cache directory does not exist: " + ", ".join(missing))
    return paths


class D1FoundationDetectionTrainer(DetectionTrainer):
    """Train the WP4 downstream model from WP2 feature caches."""

    def __init__(
        self,
        cfg: Any = DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
        *,
        feature_caches: Mapping[str, str | Path],
        validation_feature_mode: str = "cache",
        online_feature_provider: FeatureProvider | None = None,
        trusted_feature_cache: bool = False,
        max_open_feature_shards: int = 0,
        feature_prefetch_factor: int = 4,
        amp_init_scale: float | None = None,
        amp_growth_interval: int = 2_000,
    ) -> None:
        self.feature_caches = _cache_paths(feature_caches)
        if type(trusted_feature_cache) is not bool:
            raise TypeError("trusted_feature_cache must be a boolean.")
        if type(max_open_feature_shards) is not int or max_open_feature_shards < 0:
            raise ValueError("max_open_feature_shards must be a non-negative integer.")
        if type(feature_prefetch_factor) is not int or feature_prefetch_factor <= 0:
            raise ValueError("feature_prefetch_factor must be a positive integer.")
        if amp_init_scale is not None and (not isinstance(amp_init_scale, (int, float)) or amp_init_scale <= 0):
            raise ValueError("amp_init_scale must be a positive number or None.")
        if type(amp_growth_interval) is not int or amp_growth_interval <= 0:
            raise ValueError("amp_growth_interval must be a positive integer.")
        if validation_feature_mode not in {"cache", "online"}:
            raise ValueError("validation_feature_mode must be 'cache' or 'online'.")
        if validation_feature_mode == "online" and not callable(online_feature_provider):
            raise ValueError("online validation requires an online_feature_provider callable.")
        self.validation_feature_mode = validation_feature_mode
        self.online_feature_provider = online_feature_provider
        self.trusted_feature_cache = trusted_feature_cache
        self.max_open_feature_shards = max_open_feature_shards
        self.feature_prefetch_factor = feature_prefetch_factor
        self.amp_init_scale = float(amp_init_scale) if amp_init_scale is not None else None
        self.amp_growth_interval = amp_growth_interval
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self._validate_d1_args()

    def resolve_ddp_policy(self) -> tuple[bool, bool]:
        """Use the stable dense D1 graph without per-step unused-parameter traversal."""
        return False, True

    def _setup_train(self) -> None:
        super()._setup_train()
        # Base setup already restored the scaler for resumed runs.
        if self.amp and self.amp_init_scale is not None and not self.resume:
            self.scaler = (
                torch.amp.GradScaler(
                    "cuda",
                    enabled=True,
                    init_scale=self.amp_init_scale,
                    growth_interval=self.amp_growth_interval,
                )
                if TORCH_2_4
                else torch.cuda.amp.GradScaler(
                    enabled=True,
                    init_scale=self.amp_init_scale,
                    growth_interval=self.amp_growth_interval,
                )
            )

    def _validate_d1_args(self) -> None:
        if self.args.imgsz != 640:
            raise ValueError(f"D1 cached training requires imgsz=640, got {self.args.imgsz}.")
        if self.args.rect:
            raise ValueError("D1 cached training requires rect=False.")
        if self.args.multi_scale != 0.0:
            raise ValueError("D1 cached training requires multi_scale=0.0.")
        if self.args.compile:
            raise ValueError("D1 WP5 does not support compile mode.")
        if self.args.batch == -1 or isinstance(self.args.batch, float):
            raise ValueError("D1 cached training requires an explicit integer batch size.")
        if self.args.cache:
            raise ValueError("D1 uses feature_caches; the RGB cache argument must remain disabled.")
        enabled = [name for name in _DISABLED_AUGMENTATIONS if float(getattr(self.args, name, 0.0)) != 0.0]
        if enabled:
            raise ValueError(f"D1 cached training requires disabled RGB/geometric augmentations: {enabled}.")
        if self.validation_feature_mode == "online" and self.args.workers != 0:
            raise ValueError("online parity validation requires workers=0 to keep the Teacher in the main process.")

    def check_amp_compatibility(self) -> bool:
        """Check AMP with D1 feature inputs without downloading an unrelated RGB model."""
        if self.device.type != "cuda":
            return False
        model = unwrap_model(self.model)
        if not isinstance(model, D1FoundationDetectionModel):
            raise TypeError("D1 AMP check requires D1FoundationDetectionModel.")
        inputs = self.checkpoint_smoke_inputs(model)[1]
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                fp32 = model(inputs)[0].float()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    mixed = model(inputs)[0].float()
        finally:
            model.train(was_training)
        passed = (
            fp32.shape == mixed.shape
            and bool(torch.isfinite(fp32).all())
            and bool(torch.isfinite(mixed).all())
            and fp32.numel() > 0
        )
        if passed:
            LOGGER.info("AMP: D1 cached-feature checks passed")
        else:
            LOGGER.warning("AMP: D1 cached-feature checks failed; AMP will be disabled")
        return passed

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build a COCO-label dataset backed by the selected split cache."""
        if mode not in {"train", "val"}:
            raise ValueError(f"unsupported D1 dataset mode {mode!r}.")
        feature_mode = "cache" if mode == "train" else self.validation_feature_mode
        return D1FeatureCacheDataset(
            img_path=img_path,
            cache_dir=self.feature_caches[mode],
            data=self.data,
            feature_mode=feature_mode,
            online_feature_provider=self.online_feature_provider if mode == "val" else None,
            trusted_cache=self.trusted_feature_cache and feature_mode == "cache",
            max_open_shards=self.max_open_feature_shards,
            prefetch_factor=self.feature_prefetch_factor,
            imgsz=self.args.imgsz,
            batch_size=batch or self.args.batch,
            hyp=self.args,
            prefix=colorstr(f"{mode}: "),
            single_cls=self.args.single_cls or False,
            classes=self.args.classes,
            fraction=self.args.fraction if mode == "train" else 1.0,
            stride=32,
        )

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move cached features and labels to the device without RGB ``/255`` normalization."""
        use_fp16 = self.device.type == "cuda" and bool(getattr(self, "amp", False))
        return move_d1_batch_to_device(
            batch,
            self.device,
            feature_dtype=torch.float16 if use_fp16 else torch.float32,
        )

    def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True):
        """Construct the WP4 downstream-only model and optionally restore a standard trainer checkpoint model."""
        model = D1FoundationDetectionModel(cfg or DEFAULT_D1_MODEL_CFG, verbose=verbose and RANK == -1)
        if model.detect.nc != self.data["nc"]:
            raise ValueError(f"D1 model nc={model.detect.nc} does not match dataset nc={self.data['nc']}.")
        model = self.set_model_names_for_load(model)
        if weights is not None:
            if not isinstance(weights, D1FoundationDetectionModel):
                raise TypeError("D1 trainer weights must contain a D1FoundationDetectionModel.")
            model.load_state_dict(weights.state_dict(), strict=True)
        return model

    def get_validator(self):
        """Return the cache-aware D1 detection validator."""
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", *D1_AUX_REPORT_NAMES)
        return D1FoundationDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
            feature_cache=self.feature_caches["val"],
            feature_mode=self.validation_feature_mode,
            online_feature_provider=self.online_feature_provider,
            trusted_cache=self.trusted_feature_cache,
            max_open_shards=self.max_open_feature_shards,
            prefetch_factor=self.feature_prefetch_factor,
        )

    def auto_batch(self, max_num_obj: int = 0, dataset_size: int = 0):
        """Reject RGB-based automatic batch probing for cached-feature training."""
        raise RuntimeError("D1 cached training requires an explicit batch size; auto_batch uses RGB probes.")

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Feature batches intentionally have no RGB pixels to plot."""
        return

    def checkpoint_smoke_inputs(self, model: D1FoundationDetectionModel) -> tuple[dict[str, torch.Tensor], ...]:
        """Build deterministic cached-feature inputs for recovery checkpoint health checks."""
        first = next(model.parameters(), None)
        device = first.device if first is not None else torch.device("cpu")
        zeros = {name: torch.zeros(1, 384, 40, 40, dtype=torch.float32, device=device) for name in model.source_names}
        ramp = torch.linspace(-1.0, 1.0, 384 * 40 * 40, dtype=torch.float32, device=device).reshape(1, 384, 40, 40)
        return (
            zeros,
            {name: ramp.clone() for name in model.source_names},
        )

    def final_eval(self) -> None:
        """Load the selected D1 checkpoint directly and evaluate it through cached features."""
        with torch_distributed_zero_first(LOCAL_RANK):
            if RANK in {-1, 0}:
                checkpoint = strip_optimizer(self.last) if self.last.exists() else {}
                if self.best.exists():
                    strip_optimizer(self.best, updates={"train_results": checkpoint.get("train_results")})
        candidates, rejected = self._select_final_eval_checkpoints()
        if not candidates:
            raise RuntimeError(
                "No healthy checkpoint is available for final evaluation: " + ("; ".join(rejected) or "none exist")
            )
        self.validator.args.plots = self.args.plots
        self.validator.args.compile = False
        self.validator.args.half = False
        from ultralytics.utils.errors import MoERouterError

        router_failures = []
        original_model = self.model
        original_ema = self.ema.ema if self.ema else None
        original_final_checkpoint = getattr(self, "_d1_final_eval_checkpoint", None)
        try:
            for checkpoint_path in candidates:
                self._reset_non_checkpoint_moe_runtime_state()
                LOGGER.info(f"\nValidating {checkpoint_path} through D1 cached features...")
                try:
                    checkpoint_model, _checkpoint = load_checkpoint(checkpoint_path, device=self.device)
                    checkpoint_model.args = self.args
                    checkpoint_model.criterion = None
                    if not isinstance(checkpoint_model, D1FoundationDetectionModel):
                        raise TypeError("D1 final-eval checkpoint does not contain a D1FoundationDetectionModel.")
                    self.model = checkpoint_model
                    if self.ema:
                        self.ema.ema = None
                    self._d1_final_eval_checkpoint = Path(checkpoint_path)
                    metrics = self.validator(trainer=self)
                    # Every rank validates, but only rank zero receives reduced metrics.
                    if RANK in {-1, 0}:
                        if not isinstance(metrics, Mapping):
                            raise TypeError("D1 final evaluation did not return metrics on the main rank.")
                        self.metrics = dict(metrics)
                        self.metrics.pop("fitness", None)
                        self.run_callbacks("on_fit_epoch_end")
                    return
                except MoERouterError as exc:
                    router_failures.append(f"{checkpoint_path.name}: {exc}")
        finally:
            self.model = original_model
            self._d1_final_eval_checkpoint = original_final_checkpoint
            if self.ema:
                self.ema.ema = original_ema
        raise RuntimeError(
            "No healthy checkpoint is available for final evaluation: " + "; ".join((*rejected, *router_failures))
        )


__all__ = ["D1FoundationDetectionTrainer"]
