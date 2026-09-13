"""Fail-closed primitives for paired-view Foundation response distillation."""

from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from ultralytics.nn.modules._numeric import disabled_autocast

RESPONSE_FIELD_PAYLOAD_VERSION = "dinov3-p2-response-field-v1"
GLOBAL_BATCH_INDEX_VERSION = "epoch-major-v1"
RESPONSE_FIELD_CONDITIONS = (
    ("brightness", 0.8, "brightness:0.8"),
    ("brightness", 0.6, "brightness:0.6"),
    ("contrast", 0.75, "contrast:0.75"),
    ("contrast", 0.5, "contrast:0.5"),
    ("gaussian_blur", 1.0, "gaussian_blur:1.0"),
    ("gaussian_blur", 2.0, "gaussian_blur:2.0"),
    ("gaussian_noise", 0.03, "gaussian_noise:0.03"),
    ("gaussian_noise", 0.06, "gaussian_noise:0.06"),
)
_BN_BUFFER_NAMES = ("running_mean", "running_var", "num_batches_tracked")


def _validate_nonnegative_integer(name: str, value: int) -> int:
    """Return a validated non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    return value


def logical_global_batch_index(epoch_index: int, batch_index_within_epoch: int, num_batches_per_epoch: int) -> int:
    """Return the resume-stable zero-based logical batch position."""
    epoch_index = _validate_nonnegative_integer("epoch_index", epoch_index)
    batch_index_within_epoch = _validate_nonnegative_integer("batch_index_within_epoch", batch_index_within_epoch)
    if (
        isinstance(num_batches_per_epoch, bool)
        or not isinstance(num_batches_per_epoch, int)
        or num_batches_per_epoch <= 0
    ):
        raise ValueError(f"num_batches_per_epoch must be a positive integer, got {num_batches_per_epoch!r}.")
    if batch_index_within_epoch >= num_batches_per_epoch:
        raise ValueError(
            f"batch_index_within_epoch must be below num_batches_per_epoch, got "
            f"{batch_index_within_epoch} >= {num_batches_per_epoch}."
        )
    return epoch_index * num_batches_per_epoch + batch_index_within_epoch


def _validated_normalized_path(path: str | Path) -> str:
    """Validate an already-normalized portable image path used by the digest contract."""
    value = str(path)
    parts = value.split("/")
    drive_qualified = bool(parts and len(parts[0]) >= 2 and parts[0][0].isalpha() and parts[0][1] == ":")

    if (
        not value
        or "\0" in value
        or "\\" in value
        or value.startswith("/")
        or drive_qualified
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(
            f"normalized_image_path must be a non-empty canonical relative POSIX-style path, got {value!r}."
        )
    return value


def _response_field_payload(
    *, seed: int, epoch_index: int, global_batch_index: int, normalized_image_path: str | Path
) -> bytes:
    """Build the exact frozen response-field sampling payload."""
    seed = _validate_nonnegative_integer("seed", seed)
    epoch_index = _validate_nonnegative_integer("epoch_index", epoch_index)
    global_batch_index = _validate_nonnegative_integer("global_batch_index", global_batch_index)
    normalized_path = _validated_normalized_path(normalized_image_path)
    return (
        f"{RESPONSE_FIELD_PAYLOAD_VERSION}\0{seed}\0{epoch_index}\0{global_batch_index}\0{normalized_path}"
    ).encode()


@dataclass(frozen=True)
class ResponseFieldCondition:
    """One deterministic perturbation assignment for a logical sample position."""

    family: str
    value: float
    condition_id: str
    condition_index: int
    noise_seed: int | None


def response_field_condition(
    *, seed: int, epoch_index: int, global_batch_index: int, normalized_image_path: str | Path
) -> ResponseFieldCondition:
    """Resolve the frozen condition and optional Gaussian-noise seed for one sample."""
    payload = _response_field_payload(
        seed=seed,
        epoch_index=epoch_index,
        global_batch_index=global_batch_index,
        normalized_image_path=normalized_image_path,
    )
    condition_index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(RESPONSE_FIELD_CONDITIONS)
    family, value, condition_id = RESPONSE_FIELD_CONDITIONS[condition_index]
    noise_seed = response_field_noise_seed(payload, condition_id) if family == "gaussian_noise" else None
    return ResponseFieldCondition(family, value, condition_id, condition_index, noise_seed)


def response_field_noise_seed(payload: bytes, condition_id: str) -> int:
    """Return the frozen Gaussian-noise seed for one payload and condition ID."""
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("payload must be non-empty bytes.")
    if not isinstance(condition_id, str) or not condition_id:
        raise ValueError("condition_id must be a non-empty string.")
    noise_payload = payload + b"\0" + condition_id.encode() + b"\0noise"
    return int.from_bytes(hashlib.sha256(noise_payload).digest()[:8], "big") % (2**63)


def _gaussian_kernel1d(sigma: float, *, dtype: torch.dtype) -> torch.Tensor:
    """Return the frozen CPU Gaussian kernel with radius ``ceil(3*sigma)``."""
    radius = math.ceil(3.0 * sigma)
    coordinates = torch.arange(-radius, radius + 1, dtype=dtype)
    kernel = torch.exp(-(coordinates.square()) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def _perturb_one(image: torch.Tensor, condition: ResponseFieldCondition) -> torch.Tensor:
    """Apply one frozen perturbation to a CPU FP32 CHW image."""
    family, value = condition.family, condition.value
    if family == "brightness":
        output = image * value
    elif family == "contrast":
        mean = image.mean(dim=(-2, -1), keepdim=True)
        output = mean + value * (image - mean)
    elif family == "gaussian_blur":
        kernel = _gaussian_kernel1d(value, dtype=image.dtype)
        radius = kernel.numel() // 2
        if image.shape[-2] <= radius or image.shape[-1] <= radius:
            raise ValueError(f"image spatial size {tuple(image.shape[-2:])} is too small for blur radius {radius}.")
        channels = image.shape[0]
        horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
        vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
        output = F.conv2d(F.pad(image[None], (radius, radius, 0, 0), mode="reflect"), horizontal, groups=channels)
        output = F.conv2d(F.pad(output, (0, 0, radius, radius), mode="reflect"), vertical, groups=channels)[0]
    elif family == "gaussian_noise":
        if condition.noise_seed is None:
            raise RuntimeError("Gaussian-noise condition is missing its deterministic seed.")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(condition.noise_seed)
        output = image + value * torch.randn(image.shape, dtype=image.dtype, generator=generator)
    else:
        raise RuntimeError(f"Unsupported frozen response-field perturbation: {family!r}.")
    output = output.clamp(0.0, 1.0)
    if not torch.isfinite(output).all():
        raise ValueError(f"Perturbation {condition.condition_id!r} produced NaN or Inf.")
    return output


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash one finite contiguous CPU tensor including shape and dtype metadata."""
    if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
        raise ValueError("tensor_sha256 requires a finite torch.Tensor.")
    value = tensor.detach().contiguous().cpu()
    header = f"{tuple(value.shape)}|{value.dtype}|".encode()
    raw_bytes = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + raw_bytes).hexdigest()


def build_response_field_paired_view(
    clean_images: torch.Tensor,
    normalized_image_paths: list[str],
    *,
    seed: int,
    epoch_index: int,
    batch_index_within_epoch: int,
    num_batches_per_epoch: int,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    """Build deterministic perturbed views and per-image rolling-manifest records."""
    if not isinstance(clean_images, torch.Tensor) or clean_images.ndim != 4:
        raise ValueError("clean_images must be a BCHW torch.Tensor.")
    if clean_images.device.type != "cpu" or clean_images.dtype != torch.float32:
        raise ValueError("response-field perturbations require a CPU FP32 clean tensor.")
    if not torch.isfinite(clean_images).all():
        raise ValueError("clean_images contains NaN or Inf.")
    if (clean_images < 0).any() or (clean_images > 1).any():
        raise ValueError("clean_images must be scaled to the frozen [0, 1] range.")
    if len(normalized_image_paths) != clean_images.shape[0]:
        raise ValueError("normalized_image_paths must contain exactly one path per image.")
    global_index = logical_global_batch_index(epoch_index, batch_index_within_epoch, num_batches_per_epoch)
    perturbed_images, records = [], []
    for image, path in zip(clean_images, normalized_image_paths):
        normalized_path = _validated_normalized_path(path)
        condition = response_field_condition(
            seed=seed,
            epoch_index=epoch_index,
            global_batch_index=global_index,
            normalized_image_path=normalized_path,
        )
        perturbed = _perturb_one(image, condition)
        perturbed_images.append(perturbed)
        records.append(
            {
                "image_id": normalized_path,
                "condition_id": condition.condition_id,
                "noise_seed": condition.noise_seed,
                "epoch_index": epoch_index,
                "batch_index_within_epoch": batch_index_within_epoch,
                "num_batches_per_epoch": num_batches_per_epoch,
                "global_batch_index": global_index,
                "global_batch_index_version": GLOBAL_BATCH_INDEX_VERSION,
                "clean_tensor_sha256": tensor_sha256(image),
                "perturbed_tensor_sha256": tensor_sha256(perturbed),
            }
        )
    return torch.stack(perturbed_images), records


def apply_response_field_condition_batch(
    clean_images: torch.Tensor,
    normalized_image_paths: list[str],
    *,
    family: str,
    value: float,
    condition_id: str,
    seed: int,
    epoch_index: int,
    batch_index_within_epoch: int,
    num_batches_per_epoch: int,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    """Apply one explicitly selected frozen condition to a complete calibration batch."""
    if not isinstance(clean_images, torch.Tensor) or clean_images.ndim != 4:
        raise ValueError("clean_images must be a BCHW torch.Tensor.")
    if clean_images.device.type != "cpu" or clean_images.dtype != torch.float32:
        raise ValueError("response-field perturbations require a CPU FP32 clean tensor.")
    if not torch.isfinite(clean_images).all() or (clean_images < 0).any() or (clean_images > 1).any():
        raise ValueError("clean_images must be finite and scaled to the frozen [0, 1] range.")
    if len(normalized_image_paths) != clean_images.shape[0]:
        raise ValueError("normalized_image_paths must contain exactly one path per image.")
    frozen = {(item[0], item[1], item[2]): index for index, item in enumerate(RESPONSE_FIELD_CONDITIONS)}
    key = (family, float(value), condition_id)
    if key not in frozen:
        raise ValueError(f"Condition {key!r} is not one of the eight frozen response-field conditions.")
    global_index = logical_global_batch_index(epoch_index, batch_index_within_epoch, num_batches_per_epoch)
    perturbed_images, records = [], []
    for image, path in zip(clean_images, normalized_image_paths):
        normalized_path = _validated_normalized_path(path)
        payload = _response_field_payload(
            seed=seed,
            epoch_index=epoch_index,
            global_batch_index=global_index,
            normalized_image_path=normalized_path,
        )
        noise_seed = response_field_noise_seed(payload, condition_id) if family == "gaussian_noise" else None
        condition = ResponseFieldCondition(family, float(value), condition_id, frozen[key], noise_seed)
        perturbed = _perturb_one(image, condition)
        perturbed_images.append(perturbed)
        records.append(
            {
                "image_id": normalized_path,
                "condition_id": condition_id,
                "noise_seed": noise_seed,
                "epoch_index": epoch_index,
                "batch_index_within_epoch": batch_index_within_epoch,
                "num_batches_per_epoch": num_batches_per_epoch,
                "global_batch_index": global_index,
                "global_batch_index_version": GLOBAL_BATCH_INDEX_VERSION,
                "clean_tensor_sha256": tensor_sha256(image),
                "perturbed_tensor_sha256": tensor_sha256(perturbed),
            }
        )
    return torch.stack(perturbed_images), records


def _batchnorm_modules(roots: Mapping[str, nn.Module]) -> dict[str, _BatchNorm]:
    """Collect uniquely named BatchNorm modules from non-overlapping roots."""
    if not roots:
        raise ValueError("At least one module root is required for BatchNorm preservation.")
    modules: dict[str, _BatchNorm] = {}
    sources: dict[str, tuple[str, str]] = {}
    seen: dict[int, str] = {}
    for root_name, root in roots.items():
        if not root_name or not isinstance(root, nn.Module):
            raise TypeError("BatchNorm roots must map non-empty names to nn.Module instances.")
        for child_name, child in root.named_modules():
            if not isinstance(child, _BatchNorm):
                continue
            full_name = f"{root_name}.{child_name}" if child_name else root_name
            if full_name in modules:
                first_root, first_child = sources[full_name]
                raise ValueError(
                    f"duplicate BatchNorm snapshot name {full_name!r}: "
                    f"first source=(root={first_root!r}, module={first_child!r}); "
                    f"second source=(root={root_name!r}, module={child_name or '<root>'!r})."
                )
            previous = seen.get(id(child))
            if previous is not None:
                raise ValueError(f"BatchNorm module {full_name!r} duplicates {previous!r} across snapshot roots.")
            seen[id(child)] = full_name
            modules[full_name] = child
            sources[full_name] = (root_name, child_name or "<root>")
    if not modules:
        raise ValueError("No BatchNorm modules were found in the requested snapshot roots.")
    return modules


class BatchNormBufferSnapshot:
    """Exact restorable snapshot of running buffers after the clean forward."""

    def __init__(self, roots: Mapping[str, nn.Module], *, require_training: bool = True) -> None:
        self._roots = dict(roots)
        self._modules = _batchnorm_modules(self._roots)
        self.training_flags = {name: module.training for name, module in self._modules.items()}
        if require_training and not all(self.training_flags.values()):
            raise RuntimeError(
                f"Response-field Student BatchNorm modules must all be in train mode: {self.training_flags}"
            )
        self._buffers: dict[str, dict[str, torch.Tensor]] = {}
        for module_name, module in self._modules.items():
            values: dict[str, torch.Tensor] = {}
            for buffer_name in _BN_BUFFER_NAMES:
                value = getattr(module, buffer_name, None)
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"BatchNorm buffer {module_name}.{buffer_name} is missing.")
                values[buffer_name] = value.detach().clone()
            self._buffers[module_name] = values

    @property
    def module_names(self) -> tuple[str, ...]:
        """Return stable names of all protected BatchNorm modules."""
        return tuple(self._modules)

    def restore(self) -> None:
        """Restore every buffer exactly, failing closed on structure or metadata drift."""
        current_modules = _batchnorm_modules(self._roots)
        if tuple(current_modules) != tuple(self._modules) or any(
            current_modules[name] is not self._modules[name] for name in self._modules
        ):
            raise RuntimeError("BatchNorm module structure changed after the clean-forward snapshot.")
        for module_name, module in current_modules.items():
            for buffer_name, saved in self._buffers[module_name].items():
                current = getattr(module, buffer_name, None)
                if not isinstance(current, torch.Tensor):
                    raise TypeError(f"BatchNorm buffer {module_name}.{buffer_name} is missing during restore.")
                if current.shape != saved.shape or current.dtype != saved.dtype or current.device != saved.device:
                    raise RuntimeError(
                        f"BatchNorm buffer metadata changed for {module_name}.{buffer_name}: "
                        f"current={(tuple(current.shape), current.dtype, current.device)}, "
                        f"saved={(tuple(saved.shape), saved.dtype, saved.device)}."
                    )
                # Rebind instead of copying in place: BatchNorm backward may retain a reference to the perturbed
                # forward's running-stat tensor and rejects an in-place version bump before backward.
                setattr(module, buffer_name, saved.detach().clone())
                restored = getattr(module, buffer_name)
                if not torch.equal(restored, saved):
                    raise RuntimeError(
                        f"BatchNorm buffer restore was not bitwise exact for {module_name}.{buffer_name}."
                    )

    def assert_training_flags_unchanged(self) -> None:
        """Reject mode changes; eval is allowed when admitted by require_training=False."""
        current_modules = _batchnorm_modules(self._roots)
        current_flags = {name: module.training for name, module in current_modules.items()}
        if current_flags != self.training_flags:
            raise RuntimeError(
                f"Response-field BatchNorm training flags changed between paired forwards: "
                f"clean={self.training_flags}, current={current_flags}."
            )

    def matches(self) -> bool:
        """Return whether all current buffers are bitwise equal to the snapshot."""
        try:
            current_modules = _batchnorm_modules(self._roots)
        except (TypeError, ValueError):
            return False
        if tuple(current_modules) != tuple(self._modules):
            return False
        for module_name, module in current_modules.items():
            if module is not self._modules[module_name]:
                return False
            for buffer_name, saved in self._buffers[module_name].items():
                current = getattr(module, buffer_name, None)
                if (
                    not isinstance(current, torch.Tensor)
                    or current.shape != saved.shape
                    or current.dtype != saved.dtype
                    or current.device != saved.device
                    or not torch.equal(current, saved)
                ):
                    return False
        return True


@contextmanager
def preserve_batchnorm_buffers(
    roots: Mapping[str, nn.Module], *, require_training: bool = True
) -> Iterator[BatchNormBufferSnapshot]:
    """Restore buffers while preserving modes; require_training=False also admits eval BN.

    Modes must remain unchanged inside the context. Buffers are restored even if the branch raises.
    """
    snapshot = BatchNormBufferSnapshot(roots, require_training=require_training)
    try:
        yield snapshot
    finally:
        try:
            snapshot.assert_training_flags_unchanged()
        finally:
            snapshot.restore()


def _strict_token_cosine_loss(student: torch.Tensor, teacher: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Compute strict FP32 token cosine distance and reject zero-norm inputs."""
    for name, feature in (("student", student), ("teacher", teacher)):
        if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
            raise ValueError(f"{name} feature must be a BCHW torch.Tensor.")
        if not torch.isfinite(feature).all():
            raise ValueError(f"{name} feature contains NaN or Inf.")
    if student.shape != teacher.shape or student.device != teacher.device:
        raise ValueError("student and teacher features must have identical shapes and devices.")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError(f"eps must be a finite positive number, got {eps!r}.")
    with disabled_autocast(student.device.type):
        student_fp32 = student.float()
        teacher_fp32 = teacher.detach().float()
        student_norm = torch.linalg.vector_norm(student_fp32, dim=1)
        teacher_norm = torch.linalg.vector_norm(teacher_fp32, dim=1)
        if not torch.isfinite(student_norm).all() or not torch.isfinite(teacher_norm).all():
            raise ValueError("FP32 cosine norm is NaN or Inf.")
        if (student_norm < eps).any() or (teacher_norm < eps).any():
            raise ValueError("Cosine feature norm is below the fail-closed epsilon threshold.")
        dot = (student_fp32 * teacher_fp32).sum(dim=1)
        if not torch.isfinite(dot).all():
            raise ValueError("FP32 cosine dot product is NaN or Inf.")
        denominator = student_norm * teacher_norm
        if not torch.isfinite(denominator).all() or (denominator <= 0).any():
            raise ValueError("FP32 cosine denominator is non-finite or non-positive.")
        similarity = dot / denominator
        if not torch.isfinite(similarity).all():
            raise ValueError("FP32 cosine similarity is NaN or Inf.")
        loss = 1.0 - similarity.mean()
    if not torch.isfinite(loss):
        raise ValueError("Strict cosine loss is NaN or Inf.")
    return loss


def strict_cosine_kd_loss(
    student_feature: torch.Tensor, teacher_feature: torch.Tensor, *, eps: float = 1e-6
) -> torch.Tensor:
    """Return the P2 strict static token-cosine loss with detached teacher features."""
    return _strict_token_cosine_loss(student_feature, teacher_feature, eps=eps)


def response_field_kd_loss(
    student_clean: torch.Tensor,
    student_perturbed: torch.Tensor,
    teacher_clean: torch.Tensor,
    teacher_perturbed: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Match finite-difference Student and detached Teacher responses with strict token cosine."""
    if not all(
        isinstance(feature, torch.Tensor)
        for feature in (student_clean, student_perturbed, teacher_clean, teacher_perturbed)
    ):
        raise TypeError("All response-field features must be torch.Tensor instances.")
    shapes = {tuple(feature.shape) for feature in (student_clean, student_perturbed, teacher_clean, teacher_perturbed)}
    devices = {feature.device for feature in (student_clean, student_perturbed, teacher_clean, teacher_perturbed)}
    if len(shapes) != 1 or len(devices) != 1:
        raise ValueError("All response-field features must have identical shapes and devices.")
    student_response = student_perturbed.float() - student_clean.float()
    teacher_response = teacher_perturbed.detach().float() - teacher_clean.detach().float()
    return _strict_token_cosine_loss(student_response, teacher_response, eps=eps)


__all__ = [
    "GLOBAL_BATCH_INDEX_VERSION",
    "RESPONSE_FIELD_CONDITIONS",
    "RESPONSE_FIELD_PAYLOAD_VERSION",
    "BatchNormBufferSnapshot",
    "ResponseFieldCondition",
    "apply_response_field_condition_batch",
    "build_response_field_paired_view",
    "logical_global_batch_index",
    "preserve_batchnorm_buffers",
    "response_field_condition",
    "response_field_kd_loss",
    "response_field_noise_seed",
    "strict_cosine_kd_loss",
    "tensor_sha256",
]
