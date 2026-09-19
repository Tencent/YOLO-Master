"""Opt-in EMA batching for the D1 frozen-feature trainer, not a global ModelEMA replacement."""

from __future__ import annotations

from collections import defaultdict

import torch

from ultralytics.nn.foundation_detection_model import D1FoundationDetectionModel
from ultralytics.utils.torch_utils import ModelEMA, unwrap_model

EMA_IMPLEMENTATIONS = ("scalar-v1", "foreach-v1")


def validate_ema_implementation(value):
    """Fail closed on unregistered runtime implementations."""
    if not isinstance(value, str) or value not in EMA_IMPLEMENTATIONS:
        raise ValueError("Unknown registered D1 EMA implementation")
    return value


def _tensor_state(model):
    """Read registered tensors without serializing extra-state diagnostics on every update."""
    result = dict(model.named_parameters(remove_duplicate=False))
    for prefix, module in model.named_modules(remove_duplicate=False):
        if module._state_dict_hooks or module._state_dict_pre_hooks:
            raise ValueError("D1 foreach EMA does not support state_dict hooks")
        for name, value in module._buffers.items():
            if value is not None and name not in module._non_persistent_buffers_set:
                result[f"{prefix}.{name}" if prefix else name] = value
    return result


class D1ModelEMA(ModelEMA):
    """Batch independent EMA operations while retaining scalar multiply-then-add rounding."""

    implementation = "foreach-v1"

    @classmethod
    def from_existing(cls, existing, model):
        """Keep the already initialized/restored EMA, decay schedule and update counter intact."""
        if type(existing) is not ModelEMA:
            raise TypeError("Expected an unmodified ModelEMA instance")
        if type(unwrap_model(model)) is not D1FoundationDetectionModel:
            raise TypeError("D1 foreach EMA requires D1FoundationDetectionModel")
        if type(existing.ema) is not D1FoundationDetectionModel:
            raise TypeError("D1 foreach EMA requires a matching EMA model")
        _tensor_state(unwrap_model(model))
        result = cls.__new__(cls)
        result.__dict__.update(existing.__dict__)
        return result

    @torch.no_grad()
    def update(self, model):
        if not self.enabled:
            return
        source = unwrap_model(model)
        if type(source) is not D1FoundationDetectionModel:
            raise TypeError("D1 foreach EMA requires D1FoundationDetectionModel")
        source_state = _tensor_state(source)
        runtime_names = {
            f"{prefix}.{name}" if prefix else name
            for prefix, module in self.ema.named_modules()
            if "mot" in module.__class__.__module__.lower() or "moa" in module.__class__.__module__.lower()
            for name, _ in module.named_buffers(recurse=False)
            if name in {"temperature", "_sparse_train_step"}
        }
        averaged, copied = defaultdict(list), defaultdict(list)
        entries = [(name, value, True) for name, value in self.ema.named_parameters()]
        entries.extend((name, value, False) for name, value in self.ema.named_buffers())
        for name, target, parameter in entries:
            value = source_state.get(name)
            if value is None:
                continue
            if target.shape != value.shape or target.device != value.device:
                raise ValueError(f"D1 EMA tensor shape/device mismatch: {name}")
            if target.layout != torch.strided or value.layout != torch.strided or target.is_complex():
                raise ValueError(f"Unsupported D1 EMA tensor layout/dtype: {name}")
            groups = averaged if parameter or (target.is_floating_point() and name not in runtime_names) else copied
            groups[target.device, target.dtype, value.dtype].append((target, value.detach()))

        self.updates += 1
        decay = self.decay(self.updates)
        for pairs in averaged.values():
            if pairs[0][0].dtype != torch.float32 or pairs[0][1].dtype != torch.float32:
                # Low-precision foreach scalar promotion can differ from the reference on CPU.
                for target, value in pairs:
                    target.mul_(decay)
                    target.add_((1 - decay) * value)
                continue
            targets, values = map(list, zip(*pairs))
            torch._foreach_mul_(targets, decay)
            # Keep the two multiplications separate: fused alpha-add changes FP rounding.
            scaled = torch._foreach_mul(values, 1 - decay)
            torch._foreach_add_(targets, scaled)
        for pairs in copied.values():
            targets, values = map(list, zip(*pairs))
            torch._foreach_copy_(targets, values)


def configure_d1_ema(existing, model, implementation):
    """Install after trainer setup/resume; ranks without an EMA remain unchanged."""
    validate_ema_implementation(implementation)
    if implementation == "scalar-v1" or existing is None:
        return existing
    return D1ModelEMA.from_existing(existing, model)
