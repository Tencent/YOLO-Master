"""D1 detection model built from cached DINOv3 feature maps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ultralytics.cfg import get_cfg
from ultralytics.nn.mixture_loss import build_composite_criterion, initialize_mixture_loss_ema_buffer
from ultralytics.nn.modules import Detect, DINOFeaturePyramidAdapter, LatentMixture
from ultralytics.nn.tasks import BaseModel
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss

DEFAULT_D1_MODEL_CFG = Path(__file__).resolve().parents[1] / "cfg" / "models" / "26" / "yolo26-d1-dinov3-latent-n.yaml"
CHECKPOINT_SCHEMA = "d1-downstream-v1"
_PYRAMID_NAMES = ("p3", "p4", "p5")
_REQUIRED_STRIDES = (8, 16, 32)
D1_AUX_REPORT_NAMES = ("latent_balance_loss", "latent_z_loss", "latent_aux_loss")
_MIXTURE_KEYS = {
    "balance_loss_coeff",
    "expert_ratio",
    "inference_top_k",
    "noise_std",
    "num_experts",
    "require_inference_calibration",
    "residual_init",
    "router_hidden_dim",
    "router_init_std",
    "router_z_loss_coeff",
    "temperature",
    "value_fusion_mode",
    "value_fusion_weights",
}


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}.")
    return dict(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def _ordered_names(value: Any) -> tuple[str, str, str]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("input_features.source_names must be an ordered sequence.")
    names = tuple(value)
    if len(names) != 3 or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("input_features.source_names must contain exactly three non-empty strings.")
    if len(set(names)) != 3:
        raise ValueError("input_features.source_names must not contain duplicates.")
    return names


def _load_config(cfg: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, Mapping):
        config = deepcopy(dict(cfg))
    else:
        path = Path(cfg).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"D1 model config does not exist: {path}")
        config = YAML.load(path)
    if config.get("model_type") != "d1_foundation_detection":
        raise ValueError("D1 model config must set model_type='d1_foundation_detection'.")
    if config.get("task", "detect") != "detect":
        raise ValueError("D1 Foundation Detection Model only supports task='detect'.")
    return config


class _D1DownstreamGraph(nn.Module):
    """Named downstream graph while retaining ``model[-1]`` Detect compatibility."""

    def __init__(
        self,
        adapter: DINOFeaturePyramidAdapter,
        mixtures: nn.ModuleDict,
        detect: Detect,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.mixtures = mixtures
        self.detect = detect

    def _ordered_modules(self) -> tuple[nn.Module, ...]:
        return (self.adapter, *(self.mixtures[name] for name in _PYRAMID_NAMES), self.detect)

    def __getitem__(self, index: int) -> nn.Module:
        return self._ordered_modules()[index]

    def __iter__(self):
        return iter(self._ordered_modules())

    def __len__(self) -> int:
        return 5


class D1FoundationDetectionModel(BaseModel):
    """Detect objects from cached block4/block8/block12 DINOv3 features."""

    def __init__(
        self,
        cfg: str | Path | Mapping[str, Any] = DEFAULT_D1_MODEL_CFG,
        *,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        config = _load_config(cfg)
        input_cfg = _mapping(config.get("input_features"), "input_features")
        adapter_cfg = _mapping(config.get("adapter"), "adapter")
        mixture_cfg = _mapping(config.get("latent_mixture"), "latent_mixture")
        detect_cfg = _mapping(config.get("detect"), "detect")
        teacher_reference = _mapping(config.get("teacher_reference"), "teacher_reference")

        source_names = _ordered_names(input_cfg.get("source_names"))
        in_channels = _positive_int(input_cfg.get("channels"), "input_features.channels")
        unexpected_mixture = sorted(set(mixture_cfg) - _MIXTURE_KEYS)
        if unexpected_mixture:
            raise ValueError(f"unsupported latent_mixture keys: {unexpected_mixture}.")
        unexpected_adapter = sorted(
            set(adapter_cfg)
            - {"pyramid_channels", "norm_groups", "p5_mode", "p5_bottleneck_channels", "p3_upsample_mode"}
        )
        if unexpected_adapter:
            raise ValueError(f"unsupported adapter keys: {unexpected_adapter}.")

        adapter = DINOFeaturePyramidAdapter(
            in_channels=in_channels,
            source_names=source_names,
            pyramid_channels=adapter_cfg.get("pyramid_channels", (64, 128, 256)),
            norm_groups=adapter_cfg.get("norm_groups", 8),
            p5_mode=adapter_cfg.get("p5_mode", "conv"),
            p5_bottleneck_channels=adapter_cfg.get("p5_bottleneck_channels"),
            p3_upsample_mode=adapter_cfg.get("p3_upsample_mode", "bilinear"),
        )
        strides = tuple(detect_cfg.get("strides", ()))
        if strides != _REQUIRED_STRIDES or strides != adapter.strides:
            raise ValueError(f"detect.strides must be {_REQUIRED_STRIDES}, got {strides}.")
        nc = _positive_int(detect_cfg.get("nc"), "detect.nc")
        reg_max = _positive_int(detect_cfg.get("reg_max"), "detect.reg_max")
        end2end = detect_cfg.get("end2end")
        if not isinstance(end2end, bool):
            raise TypeError(f"detect.end2end must be bool, got {type(end2end).__name__}.")

        mixtures = nn.ModuleDict(
            {
                name: LatentMixture([channels] * 3, channels, **mixture_cfg)
                for name, channels in zip(_PYRAMID_NAMES, adapter.out_channels)
            }
        )
        detect = Detect(nc=nc, reg_max=reg_max, end2end=end2end, ch=adapter.out_channels)
        detect.stride = torch.tensor(strides, dtype=torch.float32)
        detect.bias_init()

        self.model = _D1DownstreamGraph(adapter, mixtures, detect)
        self.yaml = deepcopy(config)
        self.names = {index: str(index) for index in range(nc)}
        self.inplace = True
        self.save: list[int] = []
        self.task = "detect"
        self.args = get_cfg(overrides=_mapping(config.get("loss", {}), "loss"))
        self.teacher_reference = deepcopy(teacher_reference)
        self.criterion = None
        if verbose:
            self.info()

    @property
    def adapter(self) -> DINOFeaturePyramidAdapter:
        return self.model.adapter

    @property
    def mixtures(self) -> nn.ModuleDict:
        return self.model.mixtures

    @property
    def detect(self) -> Detect:
        return self.model.detect

    @property
    def source_names(self) -> tuple[str, str, str]:
        return self.adapter.source_names

    @property
    def stride(self) -> torch.Tensor:
        return self.detect.stride

    @property
    def end2end(self) -> bool:
        return self.detect.end2end

    def set_head_attr(self, **kwargs: Any) -> None:
        """Set supported Detect-head runtime attributes used by Trainer and Validator."""
        for name, value in kwargs.items():
            if not hasattr(self.detect, name):
                LOGGER.warning(f"D1 Detect head has no attribute {name!r}.")
                continue
            setattr(self.detect, name, value)

    @property
    def last_latent_aux_metrics(self) -> dict[str, float]:
        """Return the latest detached WP6 aggregate and per-level metrics."""
        return dict(getattr(self, "_last_latent_aux_metrics", {}))

    def mixture_aux_report_items(self) -> torch.Tensor:
        """Validate one D1 aux step and return detached aggregate report items."""
        applied = getattr(self, "_last_mixture_aux_loss", None)
        if not isinstance(applied, torch.Tensor) or applied.numel() != 1:
            raise RuntimeError("CompositeCriterion must set scalar _last_mixture_aux_loss before D1 reporting.")
        applied = applied.detach().reshape(())
        if not self.training:
            zeros = applied.new_zeros(len(D1_AUX_REPORT_NAMES))
            self._last_latent_aux_metrics = {
                **{name: 0.0 for name in D1_AUX_REPORT_NAMES},
                "mixture_aux_loss": 0.0,
            }
            return zeros

        diagnostics = getattr(self, "_mixture_aux_diagnostics", None)
        if not isinstance(diagnostics, dict):
            # This is a missing criterion lifecycle state, not a user argument type error.
            raise RuntimeError("D1 training requires CompositeCriterion aux diagnostics.")  # noqa: TRY004
        counts = diagnostics.get("counts_by_kind", {})
        if counts.get("latent") != len(_PYRAMID_NAMES):
            raise RuntimeError(
                f"D1 requires exactly {len(_PYRAMID_NAMES)} latent aux publications per step, got {counts.get('latent')}."
            )
        for key in ("stale_skipped", "eval_skipped", "duplicate_skipped"):
            if diagnostics.get(key, 0) != 0:
                raise RuntimeError(f"D1 aux collection reported {key}={diagnostics[key]}.")
        values = diagnostics.get("values_by_kind", {}).get("latent", ())
        if len(values) != len(_PYRAMID_NAMES):
            raise RuntimeError(f"D1 requires three recorded latent aux values, got {len(values)}.")

        metrics: dict[str, float] = {}
        balance_total = 0.0
        z_total = 0.0
        snapshot_aux_total = 0.0
        for name in _PYRAMID_NAMES:
            snapshot = self.mixtures[name].routing_snapshot()
            required = ("balance_loss", "z_loss", "aux_loss")
            if any(key not in snapshot for key in required):
                raise RuntimeError(f"D1 {name} routing snapshot is missing WP6 aux fields.")
            level_values = {key: float(snapshot[key]) for key in required}
            if not all(torch.isfinite(torch.tensor(value)) for value in level_values.values()):
                raise FloatingPointError(f"D1 {name} routing snapshot contains non-finite aux metrics.")
            metrics.update({f"{name}_{key}": value for key, value in level_values.items()})
            balance_total += level_values["balance_loss"]
            z_total += level_values["z_loss"]
            snapshot_aux_total += level_values["aux_loss"]

        collected_aux_total = sum(float(value) for value in values)
        tolerance = max(1e-7, abs(collected_aux_total) * 1e-5)
        if abs(snapshot_aux_total - collected_aux_total) > tolerance:
            raise RuntimeError(
                "D1 latent aux snapshots do not match the three graph-connected values collected by CompositeCriterion."
            )
        aggregate = {
            "latent_balance_loss": balance_total,
            "latent_z_loss": z_total,
            "latent_aux_loss": collected_aux_total,
            "mixture_aux_loss": float(applied),
        }
        if not all(torch.isfinite(torch.tensor(value)) for value in aggregate.values()):
            raise FloatingPointError("D1 aggregate aux metrics contain NaN or Inf.")
        metrics.update(aggregate)
        metrics["aux_step"] = float(diagnostics.get("step", -1))
        metrics["latent_publications"] = float(counts["latent"])
        self._last_latent_aux_metrics = metrics
        return applied.new_tensor([aggregate[name] for name in D1_AUX_REPORT_NAMES]).detach()

    def _reset_training_routing_state(self) -> None:
        if not self.training:
            return
        from ultralytics.nn.modules.moe._common import _MOE_LOSS_REGISTRY_LOCK, MOE_LOSS_REGISTRY
        from ultralytics.nn.modules.routing_protocol import reset_routing_runtime_state

        with _MOE_LOSS_REGISTRY_LOCK:
            MOE_LOSS_REGISTRY.clear()
        reset_routing_runtime_state(self)

    def predict(
        self,
        features: Mapping[str, torch.Tensor],
        profile: bool = False,
        visualize: bool = False,
        augment: bool = False,
        embed: Any = None,
    ):
        """Run Adapter, three latent mixtures, and Detect on cached feature maps."""
        if profile or visualize or augment or embed is not None:
            raise NotImplementedError("D1 cached-feature prediction does not support RGB profiling/augmentation hooks.")
        if not isinstance(features, Mapping):
            raise TypeError(f"D1 model input must be a feature mapping, got {type(features).__name__}.")
        self._reset_training_routing_state()
        candidates = self.adapter(features)
        pyramid = [self.mixtures[name](candidates[name]) for name in _PYRAMID_NAMES]
        return self.detect(pyramid)

    def forward(self, x: Mapping[str, Any], *args, **kwargs):
        """Predict from a feature mapping, or compute loss from a batch containing ``features``."""
        if isinstance(x, Mapping) and "features" in x:
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def loss(self, batch: Mapping[str, Any], preds: Any = None):
        """Apply the native Detect/E2E loss and routed CompositeCriterion."""
        if not isinstance(batch, Mapping):
            raise TypeError(f"batch must be a mapping, got {type(batch).__name__}.")
        if self.criterion is None:
            self.criterion = self.init_criterion()
        if preds is None:
            if "features" not in batch:
                raise KeyError("D1 loss batch must contain a 'features' mapping.")
            preds = self.predict(batch["features"])
        return self.criterion(preds, batch)

    def init_criterion(self):
        """Build the existing native Detect loss wrapped by CompositeCriterion."""
        native = E2ELoss(self) if self.end2end else v8DetectionLoss(self)
        return build_composite_criterion(self, native)

    def config_dict(self) -> dict[str, Any]:
        """Return a detached, host-independent model configuration."""
        return deepcopy(self.yaml)

    def checkpoint_payload(self) -> dict[str, Any]:
        """Serialize downstream state plus config and immutable Teacher reference."""
        return {
            "schema_version": CHECKPOINT_SCHEMA,
            "state_dict": self.state_dict(),
            "config": self.config_dict(),
            "teacher_reference": deepcopy(self.teacher_reference),
        }

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> D1FoundationDetectionModel:
        """Construct and strictly restore a D1 downstream checkpoint payload."""
        data = _mapping(payload, "checkpoint payload")
        expected = {"schema_version", "state_dict", "config", "teacher_reference"}
        if set(data) != expected:
            raise ValueError(f"checkpoint keys must be {sorted(expected)}, got {sorted(data)}.")
        if data["schema_version"] != CHECKPOINT_SCHEMA:
            raise ValueError(f"unsupported checkpoint schema {data['schema_version']!r}.")
        model = cls(_mapping(data["config"], "checkpoint config"))
        teacher_reference = _mapping(data["teacher_reference"], "checkpoint teacher_reference")
        if teacher_reference != model.teacher_reference:
            raise ValueError("checkpoint teacher_reference does not match its D1 model config.")
        state_dict = data["state_dict"]
        if not isinstance(state_dict, Mapping):
            raise TypeError("checkpoint state_dict must be a mapping.")
        if "_mixture_loss_ema_buf" in state_dict:
            initialize_mixture_loss_ema_buffer(model)
        model.load_state_dict(state_dict, strict=strict)
        return model


__all__ = [
    "CHECKPOINT_SCHEMA",
    "D1_AUX_REPORT_NAMES",
    "DEFAULT_D1_MODEL_CFG",
    "D1FoundationDetectionModel",
]
