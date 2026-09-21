"""Dependency-free image preprocessing helpers for Foundation Teachers."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


DINOV3_IMAGE_MEAN = (0.485, 0.456, 0.406)
DINOV3_IMAGE_STD = (0.229, 0.224, 0.225)
# DINOv2 使用与 DINOv3 相同的 ImageNet 默认归一化(HF convert_dinov2_to_hf.py 采用 IMAGENET_DEFAULT_*)
DINOV2_IMAGE_MEAN = DINOV3_IMAGE_MEAN
DINOV2_IMAGE_STD = DINOV3_IMAGE_STD


def prepare_image_tensor(
    images: torch.Tensor,
    *,
    patch_size: int,
    mean: Sequence[float] = DINOV3_IMAGE_MEAN,
    std: Sequence[float] = DINOV3_IMAGE_STD,
) -> torch.Tensor:
    """Validate, pad, and normalize a YOLO image batch for a patch-based teacher.

    The helper preserves the input resolution and pads only the bottom/right edges when the resolution is not evenly
    divisible by ``patch_size``. It never crops pixels or silently resizes the image.

    Args:
        images (torch.Tensor): Image batch in ``(B, 3, H, W)`` layout. Floating-point inputs must be in ``[0, 1]``;
            integer inputs are interpreted as ``[0, 255]`` and converted to ``[0, 1]``.
        patch_size (int): Teacher patch size used for spatial padding.
        mean (Sequence[float]): Per-channel normalization mean.
        std (Sequence[float]): Per-channel normalization standard deviation.

    Returns:
        (torch.Tensor): Normalized, patch-aligned image batch.
    """
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"images must be a torch.Tensor, got {type(images).__name__}.")
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"images must have shape (B, 3, H, W), got {tuple(images.shape)}.")
    if not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError(f"patch_size must be a positive integer, got {patch_size!r}.")
    if len(mean) != 3 or len(std) != 3 or any(float(value) <= 0 for value in std):
        raise ValueError("mean and std must each contain three values, with std values > 0.")

    if not images.is_floating_point():
        images = images.float().div(255.0)
    else:
        images = images.float()
    if not torch.isfinite(images).all():
        raise ValueError("images contains NaN or Inf values.")
    if images.numel() and (images.min() < 0 or images.max() > 1):
        raise ValueError("floating-point images must be normalized to [0, 1] before Foundation preprocessing.")

    height, width = images.shape[-2:]
    pad_h = (-height) % patch_size
    pad_w = (-width) % patch_size
    if pad_h or pad_w:
        images = F.pad(images, (0, pad_w, 0, pad_h), value=0.0)

    mean_tensor = torch.as_tensor(mean, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    std_tensor = torch.as_tensor(std, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    return (images - mean_tensor) / std_tensor


__all__ = ["DINOV3_IMAGE_MEAN", "DINOV3_IMAGE_STD", "prepare_image_tensor"]
