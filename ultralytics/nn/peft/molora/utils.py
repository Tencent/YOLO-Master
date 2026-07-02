"""MoLoRA utilities: parameter stats, merge/unmerge, init, domain allocation."""
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER


# ---------------------------------------------------------------------------
# rsLoRA scaling
# ---------------------------------------------------------------------------

def _molora_scales(r: int, alpha: int, use_rslora: bool = True) -> float:
    """Return the LoRA scaling factor.

    Standard: alpha / r
    rsLoRA:   alpha / sqrt(r)  (Kalajdzievski 2023)
    """
    if use_rslora:
        return alpha / math.sqrt(max(r, 1))
    return alpha / max(r, 1)


# ---------------------------------------------------------------------------
# Expert initialization
# ---------------------------------------------------------------------------

def init_lora_expert_a(weight: nn.Parameter, init_type: str = "default") -> None:
    """Initialize LoRA A (down-projection) weight.

    - default:     Kaiming uniform (Hu et al. 2021)
    - orthogonal:  orthogonal init via QR decomposition
    - gaussian:    simple N(0, 0.02)
    """
    if init_type == "default":
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    elif init_type == "orthogonal":
        # Flatten to 2D, QR, then restore
        w = weight.data
        orig_shape = w.shape
        w_2d = w.view(w.shape[0], -1)
        q, _ = torch.linalg.qr(w_2d, mode="reduced")
        # Pad or truncate to match shape
        if q.shape == w_2d.shape:
            w.copy_(q.view(orig_shape))
        else:
            nn.init.orthogonal_(weight)
    elif init_type == "gaussian":
        nn.init.normal_(weight, std=0.02)
    else:
        raise ValueError(f"Unknown init_type: {init_type}")


def init_lora_expert_b(weight: nn.Parameter, init_type: str = "default") -> None:
    """Initialize LoRA B (up-projection) weight to zero (default) or small Gaussian."""
    if init_type == "default":
        nn.init.zeros_(weight)
    elif init_type == "gaussian":
        nn.init.normal_(weight, std=0.02)
    else:
        # For orthogonal, B is still zero so training starts from base weights
        nn.init.zeros_(weight)


# ---------------------------------------------------------------------------
# Module shape introspection
# ---------------------------------------------------------------------------

def get_conv_shape(module: nn.Conv2d) -> Tuple[int, int, int, int, Tuple[int, int], int, int]:
    """Return (in_channels, out_channels, kernel_size_h, kernel_size_w, padding, stride, groups)."""
    k = module.kernel_size
    if isinstance(k, int):
        k = (k, k)
    p = module.padding
    if isinstance(p, int):
        p = (p, p)
    return (
        module.in_channels,
        module.out_channels,
        k[0],
        k[1],
        p,
        module.stride[0] if isinstance(module.stride, tuple) else module.stride,
        module.groups,
    )


def is_conv(module: nn.Module) -> bool:
    return isinstance(module, nn.Conv2d)


def is_linear(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear)


# ---------------------------------------------------------------------------
# Domain allocation for continual learning
# ---------------------------------------------------------------------------

def allocate_domain_experts(
    num_experts: int, domains: List[str]
) -> Dict[str, List[int]]:
    """Allocate expert indices evenly across domains.

    Args:
        num_experts: total number of experts
        domains: list of domain names

    Returns:
        dict mapping domain -> list of expert indices
    """
    if not domains:
        return {}
    n = len(domains)
    base = num_experts // n
    remainder = num_experts % n
    alloc: Dict[str, List[int]] = {}
    idx = 0
    for i, domain in enumerate(domains):
        count = base + (1 if i < remainder else 0)
        alloc[domain] = list(range(idx, idx + count))
        idx += count
    return alloc


# ---------------------------------------------------------------------------
# Parameter freezing / trainability
# ---------------------------------------------------------------------------

def mark_only_molora_as_trainable(model: nn.Module) -> None:
    """Freeze all parameters except MoLoRA adapter parameters.

    MoLoRA parameter names contain:
      - lora_A, lora_B  (expert low-rank matrices)
      - router          (routing network)
      - molora          (general molora prefix)
    """
    for name, param in model.named_parameters():
        if any(k in name for k in ("lora_A", "lora_B", "router", "molora")):
            param.requires_grad = True
        else:
            param.requires_grad = False


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return parameter statistics for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    molora = sum(
        p.numel()
        for n, p in model.named_parameters()
        if any(k in n for k in ("lora_A", "lora_B", "router", "molora"))
    )
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "molora": molora,
        "trainable_pct": 100 * trainable / total if total else 0.0,
        "molora_pct": 100 * molora / total if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Merge / Unmerge helpers
# ---------------------------------------------------------------------------

def _merge_conv_delta(
    base_weight: nn.Parameter,
    lora_a: nn.Conv2d,
    lora_b: nn.Conv2d,
    scale: float,
) -> None:
    """Merge a single LoRA expert delta into a Conv2d base weight.

    Conv2d weight shape: [out_c, in_c, kH, kW]
    lora_A: [r, in_c, 1, 1]  (1x1 conv)
    lora_B: [out_c, r, kH, kW]  (KxK conv)

    Equivalent delta = lora_B @ lora_A via matmul + expand.
    """
    with torch.no_grad():
        # lora_A weight: [r, in_c, 1, 1] -> [r, in_c]
        a = lora_a.weight.squeeze(-1).squeeze(-1)  # [r, in_c]
        # lora_B weight: [out_c, r, kH, kW]
        b = lora_b.weight  # [out_c, r, kH, kW]
        # delta = b @ a  -> [out_c, in_c, kH, kW]
        # einsum: b[o,r,kH,kW] * a[r,i] -> [o,i,kH,kW]
        delta = torch.einsum("orkw,ri->oikw", b, a) * scale
        base_weight.data.add_(delta)


def _merge_linear_delta(
    base_weight: nn.Parameter,
    lora_a: nn.Linear,
    lora_b: nn.Linear,
    scale: float,
) -> None:
    """Merge a single LoRA expert delta into a Linear base weight."""
    with torch.no_grad():
        # W' = W + B @ A
        delta = (lora_b.weight @ lora_a.weight) * scale
        base_weight.data.add_(delta)


def _unmerge_conv_delta(
    base_weight: nn.Parameter,
    lora_a: nn.Conv2d,
    lora_b: nn.Conv2d,
    scale: float,
) -> None:
    """Unmerge a single LoRA expert delta from a Conv2d base weight."""
    with torch.no_grad():
        a = lora_a.weight.squeeze(-1).squeeze(-1)
        b = lora_b.weight
        delta = torch.einsum("orkw,ri->oikw", b, a) * scale
        base_weight.data.sub_(delta)


def _unmerge_linear_delta(
    base_weight: nn.Parameter,
    lora_a: nn.Linear,
    lora_b: nn.Linear,
    scale: float,
) -> None:
    """Unmerge a single LoRA expert delta from a Linear base weight."""
    with torch.no_grad():
        delta = (lora_b.weight @ lora_a.weight) * scale
        base_weight.data.sub_(delta)
