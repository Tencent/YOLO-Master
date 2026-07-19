"""Composition layer for native task losses and routed auxiliary losses."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from ultralytics.utils import LOGGER

def _collect_moe_aux_loss(model: nn.Module | None, device: torch.device) -> torch.Tensor:
    """Collect canonical MoE and MoLoRA losses with registry fallback semantics."""
    if model is None or not getattr(model, "training", True):
        return torch.tensor(0.0, device=device)
    from ultralytics.nn.modules.routing_protocol import collect_aux_loss, get_aux_record

    moe_loss = collect_aux_loss(
        model,
        device=device,
        include_kinds=("moe", "molora"),
    )
    # Compatibility bridge for callers that intentionally populate only the
    # legacy registry (older hooks/checkpoint tests). Canonical publications
    # always win, so this cannot double-count normal forwards.
    try:
        from ultralytics.nn.modules.moe.modules import MOE_LOSS_REGISTRY
        for module in model.modules():
            if get_aux_record(module) is not None:
                continue
            value = MOE_LOSS_REGISTRY.get(module)
            if not isinstance(value, torch.Tensor):
                continue
            value = value.to(device)
            if torch.isfinite(value).all():
                moe_loss = moe_loss + value
    except Exception:
        pass
    if not torch.isfinite(moe_loss):
        LOGGER.warning(f"[NaN guard] canonical MoE/MoLoRA aux_loss is non-finite ({moe_loss}), skipping")
        return moe_loss.new_zeros(())
    return moe_loss


def _collect_mot_aux_loss(model: nn.Module | None, device: torch.device) -> torch.Tensor:
    """Sum graph-connected MoT router z-loss terms from all C2fMoT blocks."""
    mot_loss = torch.tensor(0.0, device=device)
    if model is None or not getattr(model, "training", True):
        return mot_loss
    try:
        from ultralytics.nn.modules.mot import collect_mot_aux_loss
    except Exception:
        return mot_loss
    loss_t = collect_mot_aux_loss(model)
    if isinstance(loss_t, torch.Tensor):
        lt = loss_t.to(device)
        if not torch.isfinite(lt):
            LOGGER.warning(f"[NaN guard] MoT aux_loss is non-finite ({lt}), skipping")
        else:
            mot_loss = mot_loss + lt
    return mot_loss


def _collect_moa_aux_loss(model: nn.Module | None, device: torch.device) -> torch.Tensor:
    """Sum graph-connected MoA router aux losses from C2fMoA/NeckMoAFusion blocks."""
    moa_loss = torch.tensor(0.0, device=device)
    if model is None or not getattr(model, "training", True):
        return moa_loss
    try:
        from ultralytics.nn.modules.moa import collect_moa_aux_loss
    except Exception:
        return moa_loss
    loss_t = collect_moa_aux_loss(model)
    if isinstance(loss_t, torch.Tensor):
        lt = loss_t.to(device)
        if not torch.isfinite(lt):
            LOGGER.warning(f"[NaN guard] MoA aux_loss is non-finite ({lt}), skipping")
        else:
            moa_loss = moa_loss + lt
    return moa_loss


_MIXTURE_LOSS_EMA_DECAY = 0.99
_MIXTURE_LOSS_EMA_FLOOR   = 1e-4
_MIXTURE_LOSS_MAX_ENTRY   = 1e4    # clamp EMA entry to prevent runaway growth
_MIXTURE_LOSS_EMA_DEFAULTS = {"moe": 1.0, "mot": 0.1, "moa": 0.1}
# Keys in fixed order for buffer indexing.
_MIXTURE_LOSS_EMA_KEYS = ("moe", "mot", "moa")


def _get_mixture_loss_ema(model: nn.Module | None) -> dict[str, float] | None:
    """Return (and lazily init) EMA scales for MoE/MoT/MoA aux-loss magnitudes.

    The EMA state is stored as a **persistent buffer** ``_mixture_loss_ema_buf``
    (shape [3], float32) on the model so it survives ``state_dict()`` round-trips
    and is correctly restored on resume.  Previously a plain dict attribute was
    used, which silently reset to defaults after checkpoint resume.
    """
    if model is None:
        return None
    buf = getattr(model, "_mixture_loss_ema_buf", None)
    # Determine target device from model parameters so the buffer stays aligned
    # with the model even after ``.to(device)`` calls.
    parameter = next(model.parameters(), None)
    if parameter is not None:
        target_device = parameter.device
    elif torch.cuda.is_available():
        # No parameters available (e.g. frozen params, stripped model).
        # Default to CUDA so the buffer doesn't end up on CPU, which would
        # break NCCL validation broadcasts.
        target_device = torch.device("cuda")
    else:
        target_device = torch.device("cpu")
    if buf is None:
        defaults = [_MIXTURE_LOSS_EMA_DEFAULTS[k] for k in _MIXTURE_LOSS_EMA_KEYS]
        model.register_buffer(
            "_mixture_loss_ema_buf",
            torch.tensor(defaults, dtype=torch.float32, device=target_device),
            persistent=True,
        )
        buf = model._mixture_loss_ema_buf
    elif buf.device != target_device:
        # Re-align buffer device if model was moved after lazy-init
        # (e.g. CPU checkpoint resumed then moved to CUDA).
        buf.data = buf.to(target_device)
    result = {}
    for i in range(len(_MIXTURE_LOSS_EMA_KEYS)):
        v = float(buf[i])
        # Guard against NaN/Inf leakage from corrupted buffers
        if not (v == v and abs(v) < 1e6):  # NaN self-check + magnitude bound
            result[_MIXTURE_LOSS_EMA_KEYS[i]] = _MIXTURE_LOSS_EMA_DEFAULTS[_MIXTURE_LOSS_EMA_KEYS[i]]
        else:
            result[_MIXTURE_LOSS_EMA_KEYS[i]] = v
    return result


def _mixture_aux_isolation_enabled(model: nn.Module | None) -> bool:
    """Return whether explicitly opted-in auxiliary-loss isolation is enabled."""
    args = getattr(model, "args", None)
    return bool(getattr(args, "mixture_aux_isolate_nonfinite", False))


def _mixture_aux_isolation_flags(losses: tuple[torch.Tensor, ...]) -> list[bool]:
    """Synchronize finite flags so every DDP rank makes the identical isolation choice."""
    flags = torch.tensor(
        [int(not bool(torch.isfinite(loss).all().item())) for loss in losses],
        dtype=torch.int32,
        device=losses[0].device,
    )
    if dist.is_available() and dist.is_initialized():
        try:
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        except Exception:
            # If the DDP process group is in a bad state (e.g. one rank has
            # already crashed or GPU-hung), swallow the error and fall back to
            # local flags.  This prevents a 600-second NCCL timeout; training
            # will likely fail soon anyway, but we get a cleaner traceback.
            pass
    return [bool(flag) for flag in flags.cpu().tolist()]


def _update_mixture_loss_ema(model: nn.Module | None, key: str, loss_t: torch.Tensor) -> None:
    """Update one EMA entry from a detached scalar loss magnitude."""
    if model is None or not getattr(model, "training", False):
        return
    buf = getattr(model, "_mixture_loss_ema_buf", None)
    if buf is None:
        _get_mixture_loss_ema(model)          # lazy-init buffer
        buf = model._mixture_loss_ema_buf
    idx = _MIXTURE_LOSS_EMA_KEYS.index(key)
    with torch.no_grad():
        val = float(loss_t.detach().abs().reshape(-1)[0]) if loss_t.numel() else 0.0
        # Self-heal: if buf[idx] is already NaN/Inf, reset to default before updating.
        # Without this, once corrupted the buffer stays NaN forever (NaN*x=NaN).
        old_val = float(buf[idx].item()) if buf[idx].isfinite().any() else _MIXTURE_LOSS_EMA_DEFAULTS.get(key, 0.1)
        buf[idx] = torch.tensor(old_val, dtype=buf.dtype, device=buf.device)
        # Clamp incoming value to prevent runaway buffer growth
        val = max(_MIXTURE_LOSS_EMA_FLOOR, min(val, _MIXTURE_LOSS_MAX_ENTRY))
        if val > _MIXTURE_LOSS_EMA_FLOOR:
            buf[idx] = _MIXTURE_LOSS_EMA_DECAY * float(buf[idx]) + (1.0 - _MIXTURE_LOSS_EMA_DECAY) * val


def _collect_mixture_aux_loss(
    model: nn.Module | None,
    device: torch.device,
    moe_gain: float = 1.0,
    mot_gain: float = 1.0,
    moa_gain: float = 1.0,
    aux_budget: float = 3.0,
) -> torch.Tensor:
    """Collect all mixture-routing auxiliary losses with **independent** gains.

    Per-type EMA normalization prevents large-scale losses (e.g. MoE GShard ~1.0)
    from drowning out smaller-scale losses (e.g. MoA/MoT ~0.01-0.1) while keeping
    gradient ratios stable across batches (unlike per-step detached magnitudes).

    Each loss type is scaled by its own gain (``moe_gain``, ``mot_gain``,
    ``moa_gain``) before summation, so users can up-weight MoT routing
    regularisation without inflating MoE balance loss, and vice versa.
    """
    # Guard: aux losses are training-only. During validation the model is in
    # eval mode and gradient/regularisation terms are irrelevant. Skipping
    # them here also prevents DDP collective deadlocks when ranks finish
    # validation at different speeds or one rank GPU-hangs on a particular batch.
    if model is not None and not getattr(model, "training", True):
        return torch.tensor(0.0, device=device)
    from ultralytics.nn.modules.routing_protocol import collect_aux_loss

    # Keep a structured per-kind view for logging/debugging. The numerical
    # path below remains unchanged, including the existing three EMA scales.
    if model is not None:
        _, model._mixture_aux_diagnostics = collect_aux_loss(
            model, device=device, return_diagnostics=True
        )
    moe_l = _collect_moe_aux_loss(model, device)
    mot_l = _collect_mot_aux_loss(model, device)
    moa_l = _collect_moa_aux_loss(model, device)
    aux_losses = (moe_l, mot_l, moa_l)
    nonfinite = _mixture_aux_isolation_flags(aux_losses)
    aux_names = ("moe", "mot", "moa")
    if model is not None:
        model._mixture_aux_nonfinite = {name: bad for name, bad in zip(aux_names, nonfinite)}
        model._mixture_aux_isolated = False
    if any(nonfinite):
        # Always isolate non-finite aux components — never let raw NaN/Inf
        # propagate to the main loss.  This replaces the old `mixture_aux_isolate_nonfinite`
        # flag which was never enabled in any config (it is now always on).
        # The flag is synchronized above: every DDP rank substitutes the same
        aux_losses = tuple(loss.new_zeros(()) if bad else loss for loss, bad in zip(aux_losses, nonfinite))
        moe_l, mot_l, moa_l = aux_losses

    _update_mixture_loss_ema(model, "moe", moe_l)
    _update_mixture_loss_ema(model, "mot", mot_l)
    _update_mixture_loss_ema(model, "moa", moa_l)

    ema = _get_mixture_loss_ema(model)
    if ema is not None:
        moe_scale_val = max(ema["moe"], _MIXTURE_LOSS_EMA_FLOOR)
        mot_scale_val = max(ema["mot"], _MIXTURE_LOSS_EMA_FLOOR)
        moa_scale_val = max(ema["moa"], _MIXTURE_LOSS_EMA_FLOOR)
    else:
        moe_scale_val = float(moe_l.detach().clamp(min=_MIXTURE_LOSS_EMA_FLOOR).item())
        mot_scale_val = float(mot_l.detach().clamp(min=_MIXTURE_LOSS_EMA_FLOOR).item())
        moa_scale_val = float(moa_l.detach().clamp(min=_MIXTURE_LOSS_EMA_FLOOR).item())

    # Guard: clamp scales to finite range before division — prevents NaN from
    # corrupted EMA buffers propagating through the normalisation step.
    SAFE_SCALE_RANGE = (1e-6, 1e6)
    moe_scale_val = float(torch.tensor(moe_scale_val).clamp(min=SAFE_SCALE_RANGE[0], max=SAFE_SCALE_RANGE[1]).item())
    mot_scale_val = float(torch.tensor(mot_scale_val).clamp(min=SAFE_SCALE_RANGE[0], max=SAFE_SCALE_RANGE[1]).item())
    moa_scale_val = float(torch.tensor(moa_scale_val).clamp(min=SAFE_SCALE_RANGE[0], max=SAFE_SCALE_RANGE[1]).item())

    terms = (
        moe_l / moe_scale_val * float(moe_gain),
        mot_l / mot_scale_val * float(mot_gain),
        moa_l / moa_scale_val * float(moa_gain),
    )
    # Enforce one global normalized budget without detaching the individual
    # terms' gradients. The scale is detached so budget control cannot create a
    # second gradient path through the observed loss magnitudes.
    budget = float(aux_budget)
    if not torch.isfinite(torch.tensor(budget)) or budget < 0:
        raise ValueError(f"mixture_aux_budget must be finite and >= 0, got {aux_budget}")
    observed = torch.stack([term.detach().abs() for term in terms])
    budget_scale = torch.minimum(
        observed.new_tensor(1.0),
        observed.new_tensor(budget) / observed.sum().clamp_min(_MIXTURE_LOSS_EMA_FLOOR),
    ).detach()
    aux_result = sum(terms) * budget_scale
    # Final non-finite guard + magnitude clamp: prevent any runaway aux_loss
    # from poisoning the total even if individual components are "finite" but extreme.
    if not torch.isfinite(aux_result).all() or torch.abs(aux_result) > _MIXTURE_LOSS_MAX_ENTRY:
        return moe_l.new_zeros(())
    return aux_result


def has_routed_modules(model: nn.Module | None) -> bool:
    """Return whether a model contains any registered routed module."""
    if model is None:
        return False
    from ultralytics.utils.export_capabilities import classify_routed_module

    return any(classify_routed_module(module) is not None for module in model.modules())


def _model_arg(model: nn.Module, name: str, default: float) -> float:
    args = getattr(model, "args", None)
    if isinstance(args, dict):
        value = args.get(name, default)
    else:
        value = getattr(args, name, default) if args is not None else default
    return default if value is None else float(value)


class CompositeCriterion:
    """Add one model-level routed auxiliary term after the native criterion."""

    def __init__(self, model: nn.Module, native_criterion: Any):
        self.model = model
        self.native_criterion = native_criterion
        self.enabled = has_routed_modules(model)

    def __getattr__(self, name: str):
        if name in {"model", "native_criterion", "enabled"}:
            raise AttributeError(name)
        return getattr(self.native_criterion, name)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]):
        native_result = self.native_criterion(preds, batch)
        if not self.enabled:
            return native_result
        if not isinstance(native_result, tuple) or len(native_result) != 2:
            raise TypeError("native criterion must return (loss, loss_items)")
        native_loss, native_items = native_result
        if not isinstance(native_loss, torch.Tensor):
            raise TypeError("native criterion loss must be a Tensor")
        aux = _collect_mixture_aux_loss(
            self.model,
            native_loss.device,
            moe_gain=_model_arg(self.model, "moe_aux_gain", 1.0),
            mot_gain=_model_arg(self.model, "mot_aux_gain", 1.0),
            moa_gain=_model_arg(self.model, "moa_aux_gain", 1.0),
            aux_budget=_model_arg(self.model, "mixture_aux_budget", 3.0),
        )
        self.model._last_mixture_aux_loss = aux.detach()
        total = native_loss + aux
        if isinstance(native_items, torch.Tensor):
            items = torch.cat((native_items.reshape(-1), aux.detach().reshape(1)))
        elif isinstance(native_items, (list, tuple)):
            items = [*native_items, aux.detach()]
            items = type(native_items)(items) if isinstance(native_items, tuple) else items
        else:
            items = native_items
        return total, items

    def update(self) -> None:
        update = getattr(self.native_criterion, "update", None)
        if callable(update):
            update()


def build_composite_criterion(model: nn.Module, native_criterion: Any):
    """Return a no-overhead native path for dense models and a wrapper for routed models."""
    return CompositeCriterion(model, native_criterion) if has_routed_modules(model) else native_criterion


def compose_native_result(model: nn.Module, native_loss: torch.Tensor, native_items: torch.Tensor):
    """Compose one already-computed native result for custom task loss paths."""
    if not has_routed_modules(model):
        return native_loss, native_items
    aux = _collect_mixture_aux_loss(
        model,
        native_loss.device,
        moe_gain=_model_arg(model, "moe_aux_gain", 1.0),
        mot_gain=_model_arg(model, "mot_aux_gain", 1.0),
        moa_gain=_model_arg(model, "moa_aux_gain", 1.0),
        aux_budget=_model_arg(model, "mixture_aux_budget", 3.0),
    )
    model._last_mixture_aux_loss = aux.detach()
    return native_loss + aux, torch.cat((native_items.reshape(-1), aux.detach().reshape(1)))


__all__ = [
    "CompositeCriterion",
    "build_composite_criterion",
    "compose_native_result",
    "has_routed_modules",
    "_collect_moe_aux_loss",
    "_collect_mot_aux_loss",
    "_collect_moa_aux_loss",
    "_collect_mixture_aux_loss",
    "_get_mixture_loss_ema",
]
