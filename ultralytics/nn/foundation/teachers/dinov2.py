"""DINOv2 Foundation Teacher backend.

DINOv2 是 F11 任务书指定的 DINO 系列教师(T1R 特征缓存/T2 q_teacher 即基于 DINOv2)。
与 DINOv3 同构(纯 ViT + ImageNet 归一化),权重 ``facebook/dinov2-base`` 为 HF 公开
非 gated 资源,可通过 hf-mirror 获取。The optional Transformers dependency is imported
only when a backend instance needs to load a model. Tests and local integrations can
inject an already constructed backbone without importing Transformers at all.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn as nn

from ..preprocessing import DINOV2_IMAGE_MEAN, DINOV2_IMAGE_STD, prepare_image_tensor
from ..protocol import FoundationFeatures


DEFAULT_DINOV2_MODEL = "facebook/dinov2-base"
_AUTO = "auto"
_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _is_auto(value: Any) -> bool:
    """Return whether a device/dtype request is the explicit auto sentinel."""
    return value is None or value == _AUTO


def _resolve_device(request: Any, *, model: nn.Module | None = None) -> torch.device:
    """Resolve a device request without requiring CUDA or Transformers."""
    if not _is_auto(request):
        return torch.device(request)
    if model is not None:
        try:
            return next(model.parameters()).device
        except StopIteration:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _model_dtype(model: nn.Module) -> torch.dtype:
    """Return the first floating parameter dtype, defaulting to fp32 for parameter-free test doubles."""
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


def _get_output_value(output: Any, name: str, default: Any = None) -> Any:
    """Read a field from ModelOutput-like, mapping-like, or simple tuple outputs."""
    if hasattr(output, name):
        return getattr(output, name)
    if isinstance(output, Mapping):
        return output.get(name, default)
    return default


class DINOv2Teacher(nn.Module):
    """Frozen DINOv2 ViT backend with a normalized dense feature interface.

    Args:
        model_id (str): Hugging Face model id or local model directory.
        dtype (str | torch.dtype): ``auto``, ``fp32``, ``fp16``, or ``bf16``.
        device (str | int | torch.device): Device request, or ``auto``.
        weights_path (str | Path | None): Optional local Transformers directory or state-dict file.
        model (nn.Module | None): Injected backbone, primarily for tests and offline integrations.
        model_loader (Callable | None): Optional loader receiving ``model_id`` and returning a backbone.
        local_files_only (bool): Restrict Hugging Face resolution to the local cache.
    """

    name = "dinov2"

    def __init__(
        self,
        model_id: str = DEFAULT_DINOV2_MODEL,
        *,
        dtype: str | torch.dtype = _AUTO,
        device: str | int | torch.device = _AUTO,
        weights_path: str | Path | None = None,
        model: nn.Module | None = None,
        model_loader: Callable[..., nn.Module] | None = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.model_id = str(model_id)
        self.weights_path = str(weights_path) if weights_path is not None else None
        self.local_files_only = bool(local_files_only)
        self._dtype_request = dtype
        self._device_request = device
        self.model = model if model is not None else self._load_model(model_loader)
        if not isinstance(self.model, nn.Module):
            raise TypeError(f"DINOv2 model must be an nn.Module, got {type(self.model).__name__}.")
        config = getattr(self.model, "config", None)
        self.patch_size = int(getattr(config, "patch_size", 14))
        self.hidden_size = int(getattr(config, "hidden_size", 0) or 0)
        if self.patch_size <= 0:
            raise ValueError(f"DINOv2 config patch_size must be positive, got {self.patch_size}.")
        self._device = _resolve_device(device, model=self.model)
        self._dtype = self._resolve_dtype(dtype, model=self.model)
        self.to(device=self._device, dtype=None if _is_auto(dtype) else self._dtype)
        self.freeze()

    def _load_model(self, model_loader: Callable[..., nn.Module] | None) -> nn.Module:
        """Load a Transformers backbone lazily, with an injectable offline loader."""
        if model_loader is not None:
            try:
                signature = inspect.signature(model_loader)
            except (TypeError, ValueError):
                signature = None
            if signature is None:
                return model_loader(self.model_id, self.weights_path)
            try:
                signature.bind(self.model_id, self.weights_path)
            except TypeError:
                try:
                    signature.bind(self.model_id)
                except TypeError as exc:
                    raise TypeError("model_loader must accept model_id, optionally followed by weights_path") from exc
                return model_loader(self.model_id)
            return model_loader(self.model_id, self.weights_path)
        try:
            from transformers import Dinov2Backbone
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "Foundation DINOv2 backend requires optional dependency 'transformers>=4.40,<6'. "
                "Install with: pip install -e '.[foundation]'"
            ) from exc

        source = self.weights_path if self.weights_path and Path(self.weights_path).is_dir() else self.model_id
        kwargs = {"local_files_only": self.local_files_only}
        if not _is_auto(self._dtype_request):
            kwargs["torch_dtype"] = self._resolve_dtype(self._dtype_request)
        model = Dinov2Backbone.from_pretrained(source, **kwargs)
        if self.weights_path and Path(self.weights_path).is_file():
            self._load_state_dict(model, Path(self.weights_path))
        return model

    @staticmethod
    def _load_state_dict(model: nn.Module, path: Path) -> None:
        """Load a local state-dict file without silently accepting missing model parameters."""
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # torch versions before the weights_only argument
            checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, Mapping):
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        else:
            state_dict = checkpoint
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Foundation checkpoint '{path}' does not contain a state dictionary.")
        normalized = {}
        for key, value in state_dict.items():
            key = str(key)
            for prefix in ("module.", "model.", "backbone."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            normalized[key] = value
        missing, unexpected = model.load_state_dict(normalized, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Foundation checkpoint '{path}' does not match DINOv2 architecture: "
                f"missing={list(missing)[:5]}, unexpected={list(unexpected)[:5]}"
            )

    @staticmethod
    def _resolve_dtype(request: str | torch.dtype, *, model: nn.Module | None = None) -> torch.dtype:
        """Resolve a dtype request, keeping auto mode stable and dependency-free."""
        if isinstance(request, torch.dtype):
            return request
        if _is_auto(request):
            return _model_dtype(model) if model is not None else torch.float32
        if request not in _DTYPE_MAP:
            raise ValueError(f"Unsupported DINOv2 dtype {request!r}; use auto, fp32, fp16, or bf16.")
        return _DTYPE_MAP[request]

    @property
    def device(self) -> torch.device:
        """Return the current teacher device."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Return the current teacher parameter/input dtype."""
        return self._dtype

    def freeze(self) -> None:
        """Freeze teacher parameters and force deterministic evaluation behavior."""
        super().train(False)
        self.model.requires_grad_(False)

    def train(self, mode: bool = True):
        """Keep the Foundation Teacher in eval mode even when a parent wrapper enters train mode."""
        super().train(False)
        return self

    def to(self, device=None, dtype=None, *args, **kwargs):
        """Move the teacher and retain explicit device/dtype bookkeeping."""
        if device is not None:
            self._device = _resolve_device(device, model=self.model)
            device = self._device
        if dtype is not None:
            self._dtype = self._resolve_dtype(dtype, model=self.model)
        result = super().to(device=device, dtype=dtype, *args, **kwargs)
        self.freeze()
        return result

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Convert YOLO ``[0, 1]`` images to padded, ImageNet-normalized DINOv2 inputs."""
        return prepare_image_tensor(
            images.to(device=self.device),
            patch_size=self.patch_size,
            mean=DINOV2_IMAGE_MEAN,
            std=DINOV2_IMAGE_STD,
        ).to(dtype=self.dtype)

    def encode(self, images: torch.Tensor) -> FoundationFeatures:
        """Run frozen DINOv2 inference and return the final spatial feature map as ``p4``."""
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"images must be a torch.Tensor, got {type(images).__name__}.")
        input_size = tuple(images.shape[-2:]) if images.ndim >= 2 else None
        pixel_values = self.preprocess(images)
        with torch.inference_mode():
            output = self.model(pixel_values=pixel_values)
        features = self._parse_output(output, batch_size=pixel_values.shape[0], spatial_size=pixel_values.shape[-2:])
        for name, feature in features.dense.items():
            if not torch.isfinite(feature).all():
                raise ValueError(f"DINOv2 feature '{name}' contains NaN or Inf values.")
        if features.pooled is not None and not torch.isfinite(features.pooled).all():
            raise ValueError("DINOv2 pooled feature contains NaN or Inf values.")
        features.metadata.update(
            {
                "input_size": input_size,
                "padded_size": tuple(pixel_values.shape[-2:]),
                "model_id": self.model_id,
                "backend": "transformers",
            }
        )
        return features

    def _parse_output(self, output: Any, *, batch_size: int, spatial_size: tuple[int, int]) -> FoundationFeatures:
        """Normalize a DINOv2 backbone output without assuming a fixed special-token count."""
        feature_maps = _get_output_value(output, "feature_maps")
        feature = None
        if feature_maps is not None:
            if isinstance(feature_maps, torch.Tensor):
                feature = feature_maps
            elif isinstance(feature_maps, (tuple, list)) and feature_maps:
                feature = feature_maps[-1]
        if feature is not None and feature.ndim == 3:
            feature = self._tokens_to_feature_map(feature, spatial_size)
        if feature is None:
            hidden = _get_output_value(output, "last_hidden_state")
            if hidden is None and isinstance(output, (tuple, list)) and output:
                hidden = output[0]
            if not isinstance(hidden, torch.Tensor):
                raise ValueError("DINOv2 output does not contain a spatial feature map or token sequence.")
            feature = self._tokens_to_feature_map(hidden, spatial_size)
        if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
            raise ValueError(
                f"DINOv2 spatial feature must be 4D BCHW, got {type(feature).__name__} {getattr(feature, 'shape', None)}."
            )
        if feature.shape[0] != batch_size:
            raise ValueError(f"DINOv2 feature batch {feature.shape[0]} does not match input batch {batch_size}.")
        expected_grid = (spatial_size[0] // self.patch_size, spatial_size[1] // self.patch_size)
        if tuple(feature.shape[-2:]) != expected_grid:
            raise ValueError(
                f"DINOv2 feature grid {tuple(feature.shape[-2:])} does not match padded input geometry {expected_grid}."
            )

        pooled = _get_output_value(output, "pooler_output")
        hidden = _get_output_value(output, "last_hidden_state")
        if pooled is None and isinstance(hidden, torch.Tensor) and hidden.ndim == 3:
            pooled = hidden[:, 0, :]
        if pooled is None:
            pooled = feature.mean(dim=(2, 3))
        if not isinstance(pooled, torch.Tensor) or pooled.ndim != 2:
            raise ValueError(f"DINOv2 pooled feature must be 2D, got {getattr(pooled, 'shape', None)}.")
        if pooled.shape[0] != batch_size or pooled.shape[1] != feature.shape[1]:
            raise ValueError(
                f"DINOv2 pooled feature shape {tuple(pooled.shape)} does not match feature shape {tuple(feature.shape)}."
            )

        config = getattr(self.model, "config", SimpleNamespace())
        metadata = {
            "grid_size": tuple(feature.shape[-2:]),
            "patch_size": self.patch_size,
            "hidden_dim": int(feature.shape[1]),
            "num_register_tokens": int(getattr(config, "num_register_tokens", 0)),
            "prefix_tokens": 1 + int(getattr(config, "num_register_tokens", 0)),
            "feature_maps_available": len(feature_maps)
            if isinstance(feature_maps, (tuple, list))
            else int(feature_maps is not None),
        }
        return FoundationFeatures(dense={"p4": feature}, pooled=pooled, metadata=metadata)

    def _tokens_to_feature_map(self, tokens: torch.Tensor, spatial_size: tuple[int, int]) -> torch.Tensor:
        """Convert a token sequence using config-declared prefix tokens and patch geometry."""
        if tokens.ndim != 3:
            raise ValueError(f"DINOv2 token sequence must be 3D (B, N, C), got {tuple(tokens.shape)}.")
        height, width = spatial_size
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        config = getattr(self.model, "config", SimpleNamespace())
        prefix = 1 + int(getattr(config, "num_register_tokens", 0))
        expected_patches = grid_h * grid_w
        if tokens.shape[1] == expected_patches:
            prefix = 0
        elif tokens.shape[1] != prefix + expected_patches:
            raise ValueError(
                "DINOv2 token sequence does not match input geometry: "
                f"tokens={tokens.shape[1]}, expected={prefix + expected_patches} "
                f"(grid={grid_h}x{grid_w}, prefix={prefix})."
            )
        patches = tokens[:, prefix:, :]
        return patches.reshape(tokens.shape[0], grid_h, grid_w, tokens.shape[-1]).permute(0, 3, 1, 2).contiguous()


__all__ = ["DEFAULT_DINOV2_MODEL", "DINOv2Teacher"]
