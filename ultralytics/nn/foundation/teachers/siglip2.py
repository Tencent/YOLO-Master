"""SigLIP2 Foundation Teacher backend for F12.

The backend is intentionally optional: importing :mod:`ultralytics.nn.foundation`
must not import Transformers.  A loaded teacher is frozen and can expose image
features plus cached text prototypes without entering the student module tree.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from ..protocol import FoundationFeatures


DEFAULT_SIGLIP2_MODEL = "google/siglip2-base-patch16-512"
_AUTO = "auto"
_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _is_auto(value: Any) -> bool:
    return value is None or value == _AUTO


def _resolve_device(request: Any, *, model: nn.Module | None = None) -> torch.device:
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
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


def _get_output_value(output: Any, name: str, default: Any = None) -> Any:
    if hasattr(output, name):
        return getattr(output, name)
    if isinstance(output, Mapping):
        return output.get(name, default)
    return default


def _call_loader(loader: Callable[..., Any], model_id: str, weights_path: str | None) -> Any:
    """Call an injected loader while preserving its original TypeError."""

    try:
        signature = inspect.signature(loader)
    except (TypeError, ValueError):
        return loader(model_id, weights_path)
    try:
        signature.bind(model_id, weights_path)
    except TypeError:
        try:
            signature.bind(model_id)
        except TypeError as exc:
            raise TypeError("loader must accept model_id, optionally followed by weights_path") from exc
        return loader(model_id)
    return loader(model_id, weights_path)


class SigLIP2Teacher(nn.Module):
    """Frozen SigLIP2 image/text encoder with the FoundationFeatures protocol."""

    name = "siglip2"

    def __init__(
        self,
        model_id: str = DEFAULT_SIGLIP2_MODEL,
        *,
        dtype: str | torch.dtype = _AUTO,
        device: str | int | torch.device = _AUTO,
        weights_path: str | Path | None = None,
        model: nn.Module | None = None,
        processor: Any | None = None,
        model_loader: Callable[..., nn.Module] | None = None,
        processor_loader: Callable[..., Any] | None = None,
        local_files_only: bool = False,
        max_num_patches: int | None = None,
    ) -> None:
        super().__init__()
        self.model_id = str(model_id)
        self.weights_path = str(weights_path) if weights_path is not None else None
        self.local_files_only = bool(local_files_only)
        self._dtype_request = dtype
        self._device_request = device
        self.model = model if model is not None else self._load_model(model_loader)
        if not isinstance(self.model, nn.Module):
            raise TypeError(f"SigLIP2 model must be an nn.Module, got {type(self.model).__name__}.")
        self.processor = processor if processor is not None else self._load_processor(processor_loader)
        if self.processor is None or not callable(self.processor):
            raise TypeError("SigLIP2 processor must be callable or supplied by a processor_loader.")
        config = getattr(self.model, "config", None)
        vision_config = getattr(config, "vision_config", config)
        self.patch_size = int(getattr(vision_config, "patch_size", 16))
        self.hidden_size = int(
            getattr(vision_config, "hidden_size", 0) or getattr(vision_config, "projection_size", 0) or 0
        )
        if self.patch_size <= 0:
            raise ValueError(f"SigLIP2 config patch_size must be positive, got {self.patch_size}.")
        self.max_num_patches = None if max_num_patches is None else int(max_num_patches)
        if self.max_num_patches is not None and self.max_num_patches <= 0:
            raise ValueError("max_num_patches must be positive when provided.")
        self._device = _resolve_device(device, model=self.model)
        self._dtype = self._resolve_dtype(dtype, model=self.model)
        self._text_cache: dict[tuple[str, ...], torch.Tensor] = {}
        self.to(device=self._device, dtype=None if _is_auto(dtype) else self._dtype)
        self.freeze()

    def _load_model(self, model_loader: Callable[..., nn.Module] | None) -> nn.Module:
        if model_loader is not None:
            return _call_loader(model_loader, self.model_id, self.weights_path)
        try:
            from transformers import AutoModel
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "F12 SigLIP2 backend requires optional dependency 'transformers>=5,<6'. "
                "Install with: pip install -e '.[foundation]'"
            ) from exc
        source = self.weights_path if self.weights_path and Path(self.weights_path).is_dir() else self.model_id
        kwargs = {"local_files_only": self.local_files_only}
        if not _is_auto(self._dtype_request):
            kwargs["torch_dtype"] = self._resolve_dtype(self._dtype_request)
        # Some early/mirrored SigLIP2 repositories advertise the predecessor
        # ``siglip`` config. AutoModel selects the matching implementation and
        # keeps those weights loadable while true SigLIP2 uses Siglip2Model.
        model = AutoModel.from_pretrained(source, **kwargs)
        if self.weights_path and Path(self.weights_path).is_file():
            checkpoint = torch.load(self.weights_path, map_location="cpu", weights_only=True)
            if isinstance(checkpoint, Mapping):
                checkpoint = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
            if not isinstance(checkpoint, Mapping):
                raise TypeError(f"Foundation checkpoint '{self.weights_path}' does not contain a state dictionary.")
            missing, unexpected = model.load_state_dict(checkpoint, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"Foundation checkpoint '{self.weights_path}' does not match SigLIP2: "
                    f"missing={list(missing)[:5]}, unexpected={list(unexpected)[:5]}"
                )
        return model

    def _load_processor(self, processor_loader: Callable[..., Any] | None) -> Any:
        if processor_loader is not None:
            return _call_loader(processor_loader, self.model_id, self.weights_path)
        try:
            from transformers import AutoProcessor
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError("F12 SigLIP2 processor requires optional dependency 'transformers>=5,<6'.") from exc
        source = self.weights_path if self.weights_path and Path(self.weights_path).is_dir() else self.model_id
        kwargs = {"local_files_only": self.local_files_only}
        try:
            return AutoProcessor.from_pretrained(source, **kwargs)
        except (OSError, ValueError, RuntimeError):
            # Image-only smoke/inference can still run from a model-only cache.
            # Text prototypes require a tokenizer and will report that boundary
            # when ``encode_text`` is requested.
            config = None
            try:
                from transformers import AutoConfig

                config = AutoConfig.from_pretrained(source, local_files_only=self.local_files_only)
            except Exception:
                pass
            try:
                if getattr(config, "model_type", "") == "siglip":
                    from transformers import SiglipImageProcessor

                    return SiglipImageProcessor.from_pretrained(source, **kwargs)
                from transformers import Siglip2ImageProcessor

                return Siglip2ImageProcessor.from_pretrained(source, **kwargs)
            except (ImportError, OSError, ValueError) as exc:
                raise RuntimeError(
                    "SigLIP2 processor files are unavailable. Supply a processor/processor_loader or install the "
                    "complete Hugging Face repository."
                ) from exc

    @staticmethod
    def _resolve_dtype(request: str | torch.dtype, *, model: nn.Module | None = None) -> torch.dtype:
        if isinstance(request, torch.dtype):
            return request
        if _is_auto(request):
            return _model_dtype(model) if model is not None else torch.float32
        if request not in _DTYPE_MAP:
            raise ValueError(f"Unsupported SigLIP2 dtype {request!r}; use auto, fp32, fp16, or bf16.")
        return _DTYPE_MAP[request]

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def freeze(self) -> None:
        super().train(False)
        self.model.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def to(self, device=None, dtype=None, *args, **kwargs):
        if device is not None:
            self._device = _resolve_device(device, model=self.model)
            device = self._device
        if dtype is not None:
            dtype = self._resolve_dtype(dtype, model=self.model)
            self._dtype = dtype
        result = super().to(device=device, dtype=dtype, *args, **kwargs)
        self.freeze()
        return result

    def preprocess(self, images: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """Convert YOLO images to SigLIP2 patch-packed processor inputs."""

        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must have shape [B,3,H,W], got {getattr(images, 'shape', None)}.")
        if not images.is_floating_point():
            images = images.float()
        if (
            not bool(torch.isfinite(images).all().item())
            or float(images.detach().min()) < 0
            or float(images.detach().max()) > 1
        ):
            raise ValueError("images must be finite and in [0, 1].")
        kwargs: dict[str, Any] = {"images": images.detach().cpu(), "return_tensors": "pt"}
        if self.max_num_patches is not None:
            kwargs["max_num_patches"] = self.max_num_patches
        encoded = self.processor(**kwargs)
        if not isinstance(encoded, Mapping):
            encoded = dict(encoded)
        result = {}
        for key, value in encoded.items():
            if isinstance(value, torch.Tensor):
                if value.is_floating_point():
                    value = value.to(device=self.device, dtype=self.dtype)
                else:
                    value = value.to(device=self.device)
                result[key] = value
        if "pixel_values" not in result:
            raise ValueError("SigLIP2 processor output must contain pixel_values.")
        return result

    def encode(self, images: torch.Tensor) -> FoundationFeatures:
        inputs = self.preprocess(images)
        forward_params = set(inspect.signature(self.model.forward).parameters)
        model_inputs = {key: value for key, value in inputs.items() if key in forward_params}
        with torch.inference_mode():
            if "input_ids" not in model_inputs and hasattr(self.model, "vision_model"):
                vision_params = set(inspect.signature(self.model.vision_model.forward).parameters)
                vision_inputs = {key: value for key, value in inputs.items() if key in vision_params}
                output = self.model.vision_model(**vision_inputs)
            else:
                output = self.model(**model_inputs)
        vision_output = _get_output_value(output, "vision_model_output", output)
        hidden = _get_output_value(vision_output, "last_hidden_state")
        pooled = _get_output_value(output, "image_embeds")
        if pooled is None:
            pooled = _get_output_value(vision_output, "pooler_output")
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise ValueError("SigLIP2 output must contain vision last_hidden_state with shape [B,N,C].")
        if pooled is None:
            pooled = hidden.mean(dim=1)
        if not isinstance(pooled, torch.Tensor) or pooled.ndim != 2:
            raise ValueError("SigLIP2 output must contain pooled image features with shape [B,C].")
        dense, metadata = self._dense_from_tokens(hidden, inputs)
        if pooled.shape[0] != hidden.shape[0]:
            raise ValueError(f"SigLIP2 pooled batch {pooled.shape[0]} does not match hidden batch {hidden.shape[0]}.")
        pooled = pooled.float()
        semantic = torch.nn.functional.normalize(pooled, dim=-1)
        if not bool(torch.isfinite(semantic).all().item()):
            raise ValueError("SigLIP2 semantic features contain NaN or Inf values.")
        metadata.update({"model_id": self.model_id, "backend": "transformers", "patch_size": self.patch_size})
        return FoundationFeatures(dense=dense, pooled=pooled, semantic=semantic, metadata=metadata)

    def _dense_from_tokens(
        self, hidden: torch.Tensor, inputs: Mapping[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        shapes = inputs.get("spatial_shapes")
        mask = inputs.get("pixel_attention_mask")
        if not isinstance(shapes, torch.Tensor) or shapes.ndim != 2 or shapes.shape[1] != 2:
            pixel_values = inputs.get("pixel_values")
            if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 4:
                return {}, {"dense_available": False}
            height, width = (
                int(pixel_values.shape[-2] // self.patch_size),
                int(pixel_values.shape[-1] // self.patch_size),
            )
            shapes = [[height, width]] * int(hidden.shape[0])
        else:
            shapes = shapes.detach().cpu().tolist()
        if len(shapes) != hidden.shape[0]:
            raise ValueError("SigLIP2 spatial_shapes batch does not match hidden features.")
        grids = [(int(height), int(width)) for height, width in shapes]
        if any(height <= 0 or width <= 0 for height, width in grids):
            raise ValueError(f"SigLIP2 spatial_shapes must be positive, got {grids}.")
        max_height, max_width = max(height for height, _ in grids), max(width for _, width in grids)
        dense = hidden.new_zeros(hidden.shape[0], hidden.shape[2], max_height, max_width)
        for index, (height, width) in enumerate(grids):
            count = height * width
            if count > hidden.shape[1]:
                raise ValueError("SigLIP2 spatial_shapes require more tokens than the vision output provides.")
            dense[index, :, :height, :width] = (
                hidden[index, :count].transpose(0, 1).reshape(hidden.shape[2], height, width)
            )
        return {"p4": dense}, {
            "dense_available": True,
            "grid_size": grids[0] if len(set(grids)) == 1 else grids,
            "spatial_shapes": grids,
            "attention_mask_available": isinstance(mask, torch.Tensor),
        }

    def encode_text(self, prompts: Sequence[str]) -> torch.Tensor:
        """Encode and cache normalized text prototypes for closed-set use."""

        prompts = tuple(str(prompt) for prompt in prompts)
        if not prompts or any(not prompt.strip() for prompt in prompts):
            raise ValueError("prompts must be a non-empty sequence of non-empty strings.")
        cached = self._text_cache.get(prompts)
        if cached is not None:
            return cached.to(device=self.device, dtype=self.dtype)
        encoded = self.processor(text=list(prompts), padding="max_length", return_tensors="pt")
        if not isinstance(encoded, Mapping):
            encoded = dict(encoded)
        inputs = {
            key: value.to(device=self.device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor) and key in {"input_ids", "attention_mask", "position_ids"}
        }
        get_text = getattr(self.model, "get_text_features", None)
        if not callable(get_text):
            raise AttributeError("SigLIP2 model must expose get_text_features for encode_text().")
        with torch.inference_mode():
            output = get_text(**inputs)
        features = _get_output_value(output, "pooler_output", output)
        if isinstance(features, Mapping):
            features = features.get("pooler_output", features.get("last_hidden_state"))
        if not isinstance(features, torch.Tensor):
            raise ValueError("SigLIP2 text encoder did not return a tensor feature.")
        if features.ndim == 3:
            features = features[:, 0]
        if features.ndim != 2 or features.shape[0] != len(prompts):
            raise ValueError(f"SigLIP2 text features must be [N,C], got {tuple(features.shape)}.")
        features = torch.nn.functional.normalize(features.float(), dim=-1).detach().cpu()
        if not bool(torch.isfinite(features).all().item()):
            raise ValueError("SigLIP2 text features contain NaN or Inf values.")
        self._text_cache[prompts] = features
        return features.to(device=self.device, dtype=self.dtype)

    def clear_text_cache(self) -> None:
        self._text_cache.clear()


__all__ = ["DEFAULT_SIGLIP2_MODEL", "SigLIP2Teacher"]
