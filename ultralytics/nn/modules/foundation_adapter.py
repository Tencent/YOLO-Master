"""Trainable feature-pyramid adapters for frozen Foundation features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from .utils import get_safe_groups


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def _source_names(value: Sequence[str]) -> tuple[str, str, str]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("source_names must be an ordered sequence of three strings.")
    names = tuple(value)
    if len(names) != 3:
        raise ValueError(f"source_names must contain exactly three names, got {len(names)}.")
    if any(not isinstance(name, str) or not name or "." in name for name in names):
        raise ValueError("source_names must contain non-empty module-safe strings without dots.")
    if len(set(names)) != len(names):
        raise ValueError("source_names must not contain duplicates.")
    return names


def _pyramid_channels(value: Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("pyramid_channels must be an ordered sequence of three positive integers.")
    channels = tuple(_positive_int("pyramid_channels", channel) for channel in value)
    if len(channels) != 3:
        raise ValueError(f"pyramid_channels must contain exactly three values, got {len(channels)}.")
    return channels


def _projection(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int,
    stride: int,
    norm_groups: int,
    groups: int = 1,
) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
        nn.GroupNorm(get_safe_groups(out_channels, norm_groups), out_channels),
        nn.SiLU(inplace=True),
    )


def _p5_projection(
    in_channels: int,
    out_channels: int,
    norm_groups: int,
    mode: str,
    bottleneck_channels: int | None,
) -> nn.Sequential:
    """Keep the legacy layout or build an independently parameterized lightweight branch."""
    if mode == "conv":
        return _projection(in_channels, out_channels, kernel_size=3, stride=2, norm_groups=norm_groups)
    if mode == "depthwise":
        return nn.Sequential(
            _projection(in_channels, in_channels, kernel_size=3, stride=2, norm_groups=norm_groups, groups=in_channels),
            _projection(in_channels, out_channels, kernel_size=1, stride=1, norm_groups=norm_groups),
        )
    return nn.Sequential(
        _projection(in_channels, bottleneck_channels, kernel_size=1, stride=1, norm_groups=norm_groups),
        _projection(bottleneck_channels, out_channels, kernel_size=3, stride=2, norm_groups=norm_groups),
    )


def _double_axis(x: torch.Tensor, axis: int) -> torch.Tensor:
    x = x.movedim(axis, -1)
    previous = torch.cat((x[..., :1], x[..., :-1]), dim=-1)
    following = torch.cat((x[..., 1:], x[..., -1:]), dim=-1)
    # Preserve PyTorch's lerp order and align_corners=False endpoint replication.
    even = previous + (x - previous) * 0.75
    odd = x + (following - x) * 0.25
    return torch.stack((even, odd), dim=-1).flatten(-2).movedim(-1, axis)


class SeparableBilinear2x(nn.Module):
    """Parameter-free deterministic BCHW bilinear 2x interpolation with align_corners=False."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Interpolate width then height without the deterministic index-put backward path."""
        if x.ndim != 4 or not x.is_floating_point():
            raise TypeError("Expected a floating-point BCHW tensor.")
        if min(x.shape) <= 0:
            raise ValueError("All input dimensions must be positive.")
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        result = _double_axis(_double_axis(work, -1), -2).to(x.dtype)
        channels_last = x.is_contiguous(memory_format=torch.channels_last) and x.shape[1] >= 16
        return result.contiguous(memory_format=torch.channels_last if channels_last else torch.contiguous_format)


class DINOFeaturePyramidAdapter(nn.Module):
    """Convert aligned DINO transformer blocks into P3/P4/P5 candidate groups.

    Every source block owns an independent branch at every pyramid level. The
    returned candidates preserve source_names order so each scale can be passed
    directly to a single-scale LatentMixture.
    """

    pyramid_names = ("p3", "p4", "p5")
    strides = (8, 16, 32)

    def __init__(
        self,
        in_channels: int = 384,
        source_names: Sequence[str] = ("block4", "block8", "block12"),
        pyramid_channels: Sequence[int] = (64, 128, 256),
        norm_groups: int = 8,
        p5_mode: str = "conv",
        p5_bottleneck_channels: int | None = None,
        p3_upsample_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int("in_channels", in_channels)
        self.source_names = _source_names(source_names)
        self.out_channels = _pyramid_channels(pyramid_channels)
        self.norm_groups = _positive_int("norm_groups", norm_groups)
        if not isinstance(p5_mode, str) or p5_mode not in {"conv", "depthwise", "bottleneck"}:
            raise ValueError("p5_mode must be 'conv', 'depthwise', or 'bottleneck'.")
        if p5_mode == "bottleneck":
            p5_bottleneck_channels = _positive_int("p5_bottleneck_channels", p5_bottleneck_channels)
            if p5_bottleneck_channels >= self.in_channels:
                raise ValueError("p5_bottleneck_channels must be smaller than in_channels.")
        elif p5_bottleneck_channels is not None:
            raise ValueError("p5_bottleneck_channels is only valid with p5_mode='bottleneck'.")
        self.p5_mode = p5_mode
        self.p5_bottleneck_channels = p5_bottleneck_channels
        if not isinstance(p3_upsample_mode, str) or p3_upsample_mode not in {"bilinear", "separable_bilinear2x"}:
            raise ValueError("p3_upsample_mode must be 'bilinear' or 'separable_bilinear2x'.")
        self.p3_upsample_mode = p3_upsample_mode

        p3_channels, p4_channels, p5_channels = self.out_channels
        self.branches = nn.ModuleDict(
            {
                "p3": nn.ModuleDict(
                    {
                        name: nn.Sequential(
                            _projection(
                                self.in_channels,
                                p3_channels,
                                kernel_size=1,
                                stride=1,
                                norm_groups=self.norm_groups,
                            ),
                            SeparableBilinear2x()
                            if p3_upsample_mode == "separable_bilinear2x"
                            else nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
                        )
                        for name in self.source_names
                    }
                ),
                "p4": nn.ModuleDict(
                    {
                        name: _projection(
                            self.in_channels,
                            p4_channels,
                            kernel_size=1,
                            stride=1,
                            norm_groups=self.norm_groups,
                        )
                        for name in self.source_names
                    }
                ),
                "p5": nn.ModuleDict(
                    {
                        name: _p5_projection(
                            self.in_channels,
                            p5_channels,
                            self.norm_groups,
                            self.p5_mode,
                            self.p5_bottleneck_channels,
                        )
                        for name in self.source_names
                    }
                ),
            }
        )

    def _validate_features(self, features: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if not isinstance(features, Mapping):
            raise TypeError(f"features must be a mapping, got {type(features).__name__}.")
        expected = set(self.source_names)
        missing = tuple(name for name in self.source_names if name not in features)
        unexpected = tuple(name for name in features if name not in expected)
        if missing or unexpected:
            raise ValueError(f"feature keys do not match source_names; missing={missing}, unexpected={unexpected}.")

        checked: list[torch.Tensor] = []
        reference: torch.Tensor | None = None
        for name in self.source_names:
            feature = features[name]
            if not isinstance(feature, torch.Tensor):
                raise TypeError(f"feature {name!r} must be a torch.Tensor, got {type(feature).__name__}.")
            if feature.ndim != 4:
                raise ValueError(f"feature {name!r} must be BCHW, got shape {tuple(feature.shape)}.")
            if not feature.is_floating_point():
                raise TypeError(f"feature {name!r} must be floating point, got {feature.dtype}.")
            if feature.shape[0] <= 0 or feature.shape[-2] <= 0 or feature.shape[-1] <= 0:
                raise ValueError(f"feature {name!r} has invalid shape {tuple(feature.shape)}.")
            if int(feature.shape[1]) != self.in_channels:
                raise ValueError(f"feature {name!r} has {feature.shape[1]} channels; expected {self.in_channels}.")
            if reference is None:
                reference = feature
                if feature.shape[-2] % 2 or feature.shape[-1] % 2:
                    raise ValueError(
                        f"feature grid must have even height and width for P5, got {tuple(feature.shape[-2:])}."
                    )
            else:
                if feature.shape[0] != reference.shape[0]:
                    raise ValueError(f"feature {name!r} batch size does not match {self.source_names[0]!r}.")
                if feature.device != reference.device:
                    raise ValueError(f"feature {name!r} device does not match {self.source_names[0]!r}.")
                if feature.dtype != reference.dtype:
                    raise ValueError(f"feature {name!r} dtype does not match {self.source_names[0]!r}.")
                if feature.shape[-2:] != reference.shape[-2:]:
                    raise ValueError(f"feature {name!r} spatial size does not match {self.source_names[0]!r}.")
            checked.append(feature)
        return tuple(checked)

    def forward(self, features: Mapping[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, ...]]:
        """Return P3/P4/P5 candidates ordered by source_names."""
        checked = self._validate_features(features)
        return {
            level: tuple(self.branches[level][name](feature) for name, feature in zip(self.source_names, checked))
            for level in self.pyramid_names
        }


__all__ = ["DINOFeaturePyramidAdapter", "SeparableBilinear2x"]
