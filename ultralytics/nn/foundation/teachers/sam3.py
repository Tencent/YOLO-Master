"""SAM3 (DART) Foundation Teacher backend.

Wraps the DART SAM3 detector (https://github.com/mkturkcan/DART) behind the dependency-free Foundation boundary.
The optional ``sam3`` package is imported only when a backend instance needs to load a real model; tests and offline
integrations can inject an already constructed model without importing DART at all.

Unlike the representation-only DINOv3/SigLIP2 teachers, SAM3 additionally exposes a detection-level response channel
(``detect``) with per-prompt query boxes and logits for response knowledge distillation.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..protocol import FoundationFeatures


DEFAULT_SAM3_IMAGE_SIZE = 1008
SAM3_IMAGE_MEAN = (0.5, 0.5, 0.5)
SAM3_IMAGE_STD = (0.5, 0.5, 0.5)
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
    if isinstance(output, Mapping):
        return output.get(name, default)
    return getattr(output, name, default)


def _call_loader(loader: Callable[..., Any], checkpoint: str | None, device: torch.device) -> Any:
    """Call an injected loader with (checkpoint, device), degrading gracefully to fewer arguments."""
    try:
        signature = inspect.signature(loader)
    except (TypeError, ValueError):
        return loader(checkpoint, device)
    try:
        signature.bind(checkpoint, device)
    except TypeError:
        try:
            signature.bind(checkpoint)
        except TypeError as exc:
            raise TypeError("model_loader must accept checkpoint, optionally followed by device") from exc
        return loader(checkpoint)
    return loader(checkpoint, device)


def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=0.0, max=1.0)
    return torch.log(x.clamp(min=eps) / (1.0 - x).clamp(min=eps))


class SAM3Teacher(nn.Module):
    """Frozen SAM3/DART detector exposing dense FPN features, text prototypes, and detection responses.

    Args:
        checkpoint (str | Path | None): Local ``sam3.pt`` checkpoint path. ``None`` downloads from Hugging Face.
        dart_repo (str | Path | None): Local DART repository root added to ``sys.path`` when ``sam3`` is not installed.
        image_size (int): Square input resolution used by SAM3 preprocessing (must be divisible by ``patch_size``).
        patch_size (int): ViT patch size, used only for input-size validation.
        dtype (str | torch.dtype): ``auto``, ``fp32``, ``fp16``, or ``bf16``.
        device (str | int | torch.device): Device request, or ``auto``.
        model (nn.Module | None): Injected SAM3 model, primarily for tests and offline integrations.
        model_loader (Callable | None): Optional loader receiving ``checkpoint`` (and optionally ``device``).
    """

    name = "sam3"

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        dart_repo: str | Path | None = None,
        image_size: int = DEFAULT_SAM3_IMAGE_SIZE,
        patch_size: int = 14,
        dtype: str | torch.dtype = _AUTO,
        device: str | int | torch.device = _AUTO,
        model: nn.Module | None = None,
        model_loader: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.dart_repo = str(dart_repo) if dart_repo is not None else None
        self.patch_size = int(patch_size)
        self.image_size = int(image_size)
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}.")
        if self.image_size <= 0 or self.image_size % self.patch_size:
            raise ValueError(
                f"image_size must be a positive multiple of patch_size={self.patch_size}, got {self.image_size}."
            )
        self._dtype_request = dtype
        self._device_request = device
        self.model = model if model is not None else self._load_model(model_loader)
        if not isinstance(self.model, nn.Module):
            raise TypeError(f"SAM3 model must be an nn.Module, got {type(self.model).__name__}.")
        if not hasattr(self.model, "backbone"):
            raise TypeError("SAM3 model must expose a 'backbone' with forward_image/forward_text methods.")
        self._device = _resolve_device(device, model=self.model)
        self._dtype = self._resolve_dtype(dtype, model=self.model)
        self._text_cache: dict[tuple[str, ...], torch.Tensor] = {}
        self.to(device=self._device, dtype=None if _is_auto(dtype) else self._dtype)
        self.freeze()

    def _load_model(self, model_loader: Callable[..., nn.Module] | None) -> nn.Module:
        """Load the DART SAM3 model lazily, keeping the Foundation import boundary dependency-free."""
        device = _resolve_device(self._device_request)
        if model_loader is not None:
            return _call_loader(model_loader, self.checkpoint, device)
        if importlib.util.find_spec("sam3") is None and self.dart_repo:
            repo = Path(self.dart_repo)
            if not repo.is_dir():
                raise FileNotFoundError(f"dart_repo '{repo}' is not a directory.")
            sys.path.insert(0, str(repo))
        try:
            from sam3.model_builder import build_sam3_image_model
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "Foundation SAM3 backend requires DART's 'sam3' package (https://github.com/mkturkcan/DART). "
                "Pass dart_repo=<local DART clone> or inject model=/model_loader=."
            ) from exc
        return build_sam3_image_model(
            device=str(device),
            eval_mode=True,
            checkpoint_path=self.checkpoint,
            load_from_HF=self.checkpoint is None,
            enable_segmentation=False,
        )

    @staticmethod
    def _resolve_dtype(request: str | torch.dtype, *, model: nn.Module | None = None) -> torch.dtype:
        if isinstance(request, torch.dtype):
            return request
        if _is_auto(request):
            return _model_dtype(model) if model is not None else torch.float32
        if request not in _DTYPE_MAP:
            raise ValueError(f"Unsupported SAM3 dtype {request!r}; use auto, fp32, fp16, or bf16.")
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

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Convert YOLO ``[0, 1]`` images to square, SAM3-normalized inputs at ``image_size``."""
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must have shape [B,3,H,W], got {getattr(images, 'shape', None)}.")
        images = images.float() if images.is_floating_point() else images.float().div(255.0)
        if not torch.isfinite(images).all():
            raise ValueError("images contains NaN or Inf values.")
        if images.numel() and (images.min() < 0 or images.max() > 1):
            raise ValueError("floating-point images must be normalized to [0, 1] before SAM3 preprocessing.")
        images = images.to(device=self.device)
        if tuple(images.shape[-2:]) != (self.image_size, self.image_size):
            images = F.interpolate(
                images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False, antialias=True
            )
        mean = torch.as_tensor(SAM3_IMAGE_MEAN, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
        std = torch.as_tensor(SAM3_IMAGE_STD, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
        return ((images - mean) / std).to(dtype=self.dtype)

    def _forward_backbone(self, pixel_values: torch.Tensor) -> tuple[Any, list[torch.Tensor]]:
        """Run the frozen SAM3 vision backbone and return (raw output, validated FPN pyramid)."""
        with torch.inference_mode():
            backbone_out = self.model.backbone.forward_image(pixel_values)
        fpn = _get_output_value(backbone_out, "backbone_fpn")
        if isinstance(fpn, torch.Tensor):
            fpn = [fpn]
        if not isinstance(fpn, (list, tuple)) or not fpn:
            raise ValueError("SAM3 backbone output must contain a non-empty 'backbone_fpn' feature list.")
        batch = pixel_values.shape[0]
        for level, feature in enumerate(fpn):
            if not isinstance(feature, torch.Tensor) or feature.ndim != 4 or feature.shape[0] != batch:
                raise ValueError(
                    f"SAM3 FPN level {level} must be a BCHW tensor with batch {batch}, "
                    f"got {getattr(feature, 'shape', None)}."
                )
            if not torch.isfinite(feature).all():
                raise ValueError(f"SAM3 FPN level {level} contains NaN or Inf values.")
        return backbone_out, list(fpn)

    def _features_from_fpn(
        self, fpn: list[torch.Tensor], *, input_size: tuple[int, int] | None, processed_size: tuple[int, int]
    ) -> FoundationFeatures:
        """Assemble FoundationFeatures from a validated FPN pyramid (high→low resolution named up to ``p4``)."""
        start = 4 - (len(fpn) - 1)
        names = [f"p{start + index}" for index in range(len(fpn))]
        dense = {name: feature.clone() for name, feature in zip(names, fpn)}
        pooled = fpn[-1].mean(dim=(2, 3)).clone()
        strides = {name: self.image_size / feature.shape[-2] for name, feature in dense.items()}
        metadata = {
            "input_size": input_size,
            "processed_size": processed_size,
            "patch_size": self.patch_size,
            "hidden_dim": int(fpn[-1].shape[1]),
            "strides": strides,
            "num_levels": len(fpn),
            "checkpoint": self.checkpoint,
            "backend": "dart-sam3",
        }
        return FoundationFeatures(dense=dense, pooled=pooled, metadata=metadata)

    def encode(self, images: torch.Tensor) -> FoundationFeatures:
        """Encode images into multi-scale dense FPN features (high→low resolution named up to ``p4``)."""
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"images must be a torch.Tensor, got {type(images).__name__}.")
        input_size = tuple(images.shape[-2:]) if images.ndim >= 2 else None
        pixel_values = self.preprocess(images)
        _, fpn = self._forward_backbone(pixel_values)
        return self._features_from_fpn(fpn, input_size=input_size, processed_size=tuple(pixel_values.shape[-2:]))

    def _forward_text(self, prompts: tuple[str, ...]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the SAM3 text encoder and return seq-first token features plus the padding mask."""
        with torch.inference_mode():
            output = self.model.backbone.forward_text(list(prompts), device=self.device)
        features = _get_output_value(output, "language_features")
        mask = _get_output_value(output, "language_mask")
        if not isinstance(features, torch.Tensor) or features.ndim != 3 or features.shape[1] != len(prompts):
            raise ValueError(
                f"SAM3 text features must have shape [seq,N,C] with N={len(prompts)}, "
                f"got {getattr(features, 'shape', None)}."
            )
        if mask is not None and (not isinstance(mask, torch.Tensor) or tuple(mask.shape) != (len(prompts), features.shape[0])):
            raise ValueError(f"SAM3 text mask must have shape [N,seq], got {getattr(mask, 'shape', None)}.")
        return features, mask

    def encode_text(self, prompts: Sequence[str]) -> torch.Tensor:
        """Encode and cache normalized per-class text prototypes from the SAM3 text encoder."""
        prompts = tuple(str(prompt) for prompt in prompts)
        if not prompts or any(not prompt.strip() for prompt in prompts):
            raise ValueError("prompts must be a non-empty sequence of non-empty strings.")
        cached = self._text_cache.get(prompts)
        if cached is not None:
            return cached.to(device=self.device, dtype=self.dtype)
        features, mask = self._forward_text(prompts)
        tokens = features.permute(1, 0, 2).float()  # (N, seq, C)
        if mask is None:
            pooled = tokens.mean(dim=1)
        else:
            valid = (~mask.bool()).to(tokens.dtype).unsqueeze(-1)
            pooled = (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        pooled = torch.nn.functional.normalize(pooled, dim=-1).detach().cpu()
        if not torch.isfinite(pooled).all():
            raise ValueError("SAM3 text prototypes contain NaN or Inf values.")
        self._text_cache[prompts] = pooled
        return pooled.to(device=self.device, dtype=self.dtype)

    def clear_text_cache(self) -> None:
        self._text_cache.clear()

    def detect(self, images: torch.Tensor, prompts: Sequence[str]) -> dict[str, Any]:
        """Run the full SAM3 encoder-decoder per prompt and return the raw detection response channel.

        Returns:
            dict: ``boxes`` (P, B, Q, 4) normalized cxcywh in [0, 1]; ``logits`` (P, B, Q) raw query logits;
                ``scores`` (P, B, Q) presence-calibrated sigmoid probabilities; plus prompt/geometry metadata.
        """
        input_size = tuple(images.shape[-2:]) if isinstance(images, torch.Tensor) and images.ndim >= 2 else None
        pixel_values = self.preprocess(images)
        backbone_out, fpn = self._forward_backbone(pixel_values)
        return self._detect_response(pixel_values, backbone_out, fpn, prompts, input_size=input_size)

    def encode_with_response(
        self, images: torch.Tensor, prompts: Sequence[str]
    ) -> tuple[FoundationFeatures, dict[str, Any]]:
        """Return dense features and the detection response from a single shared backbone forward pass."""
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"images must be a torch.Tensor, got {type(images).__name__}.")
        input_size = tuple(images.shape[-2:]) if images.ndim >= 2 else None
        pixel_values = self.preprocess(images)
        backbone_out, fpn = self._forward_backbone(pixel_values)
        features = self._features_from_fpn(fpn, input_size=input_size, processed_size=tuple(pixel_values.shape[-2:]))
        response = self._detect_response(pixel_values, backbone_out, fpn, prompts, input_size=input_size)
        return features, response

    def _detect_response(
        self,
        pixel_values: torch.Tensor,
        backbone_out: Any,
        fpn: list[torch.Tensor],
        prompts: Sequence[str],
        *,
        input_size: tuple[int, int] | None,
    ) -> dict[str, Any]:
        prompts = tuple(str(prompt) for prompt in prompts)
        if not prompts or any(not prompt.strip() for prompt in prompts):
            raise ValueError("prompts must be a non-empty sequence of non-empty strings.")
        transformer = getattr(self.model, "transformer", None)
        scoring = getattr(self.model, "dot_prod_scoring", None)
        if transformer is None or not callable(scoring):
            raise AttributeError("SAM3 model must expose transformer.encoder/decoder and dot_prod_scoring for detect().")
        batch = pixel_values.shape[0]
        pos_enc = _get_output_value(backbone_out, "vision_pos_enc")
        if not isinstance(pos_enc, (list, tuple)) or len(pos_enc) != len(fpn):
            raise ValueError("SAM3 backbone output must contain 'vision_pos_enc' matching the FPN pyramid.")
        levels = int(getattr(self.model, "num_feature_levels", 1))
        vis_feats = fpn[-levels:]
        vis_pos = list(pos_enc)[-levels:]
        feat_sizes = [tuple(feature.shape[-2:]) for feature in vis_pos]
        img_feats = [feature.flatten(2).permute(2, 0, 1) for feature in vis_feats]
        img_pos = [feature.flatten(2).permute(2, 0, 1) for feature in vis_pos]
        text_features, text_mask = self._forward_text(prompts)

        all_boxes, all_logits, all_scores = [], [], []
        with torch.inference_mode():
            for index in range(len(prompts)):
                prompt = text_features[:, index : index + 1, :].expand(-1, batch, -1)
                prompt_mask = None if text_mask is None else text_mask[index : index + 1, :].expand(batch, -1)
                memory = transformer.encoder(
                    src=[feature.clone() for feature in img_feats],
                    src_key_padding_mask=None,
                    src_pos=[feature.clone() for feature in img_pos],
                    prompt=prompt,
                    prompt_pos=torch.zeros_like(prompt),
                    prompt_key_padding_mask=prompt_mask,
                    feat_sizes=feat_sizes,
                )
                tgt = transformer.decoder.query_embed.weight.unsqueeze(1).repeat(1, batch, 1)
                hs, reference_boxes, presence, _ = transformer.decoder(
                    tgt=tgt,
                    memory=memory["memory"],
                    memory_key_padding_mask=memory.get("padding_mask"),
                    pos=memory.get("pos_embed"),
                    reference_boxes=None,
                    level_start_index=memory.get("level_start_index"),
                    spatial_shapes=memory.get("spatial_shapes"),
                    valid_ratios=memory.get("valid_ratios"),
                    tgt_mask=None,
                    memory_text=prompt,
                    text_attention_mask=prompt_mask,
                    apply_dac=False,
                )
                hs = hs.transpose(1, 2)  # (layers, B, Q, C)
                reference_boxes = reference_boxes.transpose(1, 2)
                logits = scoring(hs, prompt, prompt_mask)[-1].squeeze(-1)  # (B, Q)
                offsets = transformer.decoder.bbox_embed(hs)
                boxes = (_inverse_sigmoid(reference_boxes) + offsets).sigmoid()[-1]  # (B, Q, 4)
                scores = logits.sigmoid()
                if presence is not None:
                    scores = scores * presence[-1].reshape(batch, -1)[:, :1].sigmoid()
                all_boxes.append(boxes)
                all_logits.append(logits)
                all_scores.append(scores)

        boxes = torch.stack(all_boxes).clone().float()
        logits = torch.stack(all_logits).clone().float()
        scores = torch.stack(all_scores).clone().float()
        if not torch.isfinite(boxes).all() or not torch.isfinite(logits).all():
            raise ValueError("SAM3 detection response contains NaN or Inf values.")
        return {
            "boxes": boxes,
            "logits": logits,
            "scores": scores,
            "prompts": prompts,
            "box_format": "cxcywh_norm",
            "processed_size": tuple(pixel_values.shape[-2:]),
            "input_size": input_size,
        }


__all__ = ["DEFAULT_SAM3_IMAGE_SIZE", "SAM3Teacher"]
