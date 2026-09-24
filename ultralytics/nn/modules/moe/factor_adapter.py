"""Explicitly constructed function-preserving residual factors for A1 experiments."""

import copy

import torch
from torch import nn


class ResidualFactorAdapter(nn.Module):
    """Wrap a pretrained block as base(x) + gain * factor(base(x)).

    The channel-wise gain starts at zero. The caller supplies a shape-preserving
    dense or MoE factor; this utility changes no router, parser or optimizer defaults.
    It is not a loader for historical C3k2ResidualFactor checkpoints.
    """

    def __init__(self, base: nn.Module, factor: nn.Module, channels: int, freeze_base: bool = True):
        """Keep the supplied pretrained weights and optionally freeze their path."""
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.base = base
        self.factor = factor
        self.gain = nn.Parameter(torch.zeros(channels))
        self.freeze_base = freeze_base
        for attribute in ("i", "f", "type"):
            if hasattr(base, attribute):
                setattr(self, attribute, copy.deepcopy(getattr(base, attribute)))
        self.np = sum(parameter.numel() for parameter in self.parameters())
        if freeze_base:
            self.freeze_base_parameters()

    def freeze_base_parameters(self):
        """Freeze weights and BN statistics; reapply after trainer-wide unfreezing."""
        self.base.eval()
        count = 0
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
            count += parameter.numel()
        return count

    def train(self, mode=True):
        """Retain frozen-base evaluation mode when training the factor."""
        super().train(mode)
        if self.freeze_base:
            self.base.eval()
        return self

    def forward(self, x):
        """Preserve base outputs at initialization for finite factor outputs."""
        output = self.base(x)
        return output + self.gain.view(1, -1, 1, 1) * self.factor(output)
