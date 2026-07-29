"""Composable hooks for the AdaptiveGateMoE routing pipeline.

Hooks deliberately remain lightweight Python objects instead of registered
``nn.Module`` containers.  Learnable hook modules stay on their historical
attributes (``detail_gate``, ``context_mixer``, and ``feature_refiner``), so
state-dict keys and old checkpoints remain unchanged while the execution
policy becomes configurable.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Sequence, Union


class RouterHook:
    """Base contract for one composable AdaptiveGateMoE hook."""

    name = "hook"
    stages = frozenset()

    def apply(self, module: Any, stage: str, value: Any) -> Any:
        """Transform ``value`` at ``stage`` and return the next pipeline value."""
        raise NotImplementedError


class DetailGateHook(RouterHook):
    """Apply the registered detail gate before routing."""

    name = "detail"
    stages = frozenset({"pre_route"})

    def apply(self, module: Any, stage: str, value: Any) -> Any:
        if stage == "pre_route":
            return module.detail_gate(value)
        return value


class ContextMixerHook(RouterHook):
    """Apply the registered pyramid context mixer after expert fusion."""

    name = "context"
    stages = frozenset({"post_fusion"})

    def apply(self, module: Any, stage: str, value: Any) -> Any:
        if stage == "post_fusion":
            return module.context_mixer(value)
        return value


class FeatureRefinementHook(RouterHook):
    """Apply the registered residual feature refinement after context mixing."""

    name = "refine"
    stages = frozenset({"post_fusion"})

    def apply(self, module: Any, stage: str, value: Any) -> Any:
        if stage == "post_fusion":
            return value + module.refine_scale.tanh() * module.feature_refiner(value) * module.feature_gate(value)
        return value


_HOOK_BUILDERS: Dict[str, Callable[[], RouterHook]] = {
    "detail": DetailGateHook,
    "context": ContextMixerHook,
    "refine": FeatureRefinementHook,
    "feature_refinement": FeatureRefinementHook,
}


def register_router_hook(name: str, builder: Callable[[], RouterHook], *, replace: bool = False) -> None:
    """Register a named hook factory for model/config integrations."""
    key = str(name).strip().lower()
    if not key:
        raise ValueError("router hook name must not be empty")
    if key in _HOOK_BUILDERS and not replace:
        raise KeyError(f"router hook {name!r} is already registered")
    _HOOK_BUILDERS[key] = builder


def build_router_hook(spec: Union[str, RouterHook]) -> RouterHook:
    """Resolve a hook instance from a registry name or return an instance."""
    if isinstance(spec, RouterHook):
        return spec
    if not isinstance(spec, str):
        raise TypeError(f"router hook must be a name or RouterHook, got {type(spec).__name__}")
    key = spec.strip().lower()
    try:
        hook = _HOOK_BUILDERS[key]()
    except KeyError as exc:
        raise KeyError(f"unknown router hook {spec!r}; available={sorted(_HOOK_BUILDERS)}") from exc
    if not isinstance(hook, RouterHook):
        raise TypeError(f"router hook builder {spec!r} returned {type(hook).__name__}, expected RouterHook")
    return hook


def resolve_router_hooks(specs: Union[None, str, RouterHook, Sequence[Union[str, RouterHook]]]) -> List[RouterHook]:
    """Normalize a hook config and reject duplicate hook identities."""
    if specs is None:
        return []
    if isinstance(specs, (str, RouterHook)):
        specs = [specs]
    hooks = [build_router_hook(spec) for spec in specs]
    names = [hook.name for hook in hooks]
    if len(names) != len(set(names)):
        raise ValueError(f"router hook names must be unique, got {names}")
    return hooks


def apply_router_hooks(module: Any, hooks: Iterable[RouterHook], stage: str, value: Any) -> Any:
    """Apply hooks in declaration order for one pipeline stage."""
    for hook in hooks:
        if stage in hook.stages:
            value = hook.apply(module, stage, value)
    return value


def registered_router_hooks() -> Dict[str, Callable[[], RouterHook]]:
    """Return a copy of the hook registry for diagnostics and config tools."""
    return dict(_HOOK_BUILDERS)


__all__ = [
    "RouterHook",
    "DetailGateHook",
    "ContextMixerHook",
    "FeatureRefinementHook",
    "register_router_hook",
    "build_router_hook",
    "resolve_router_hooks",
    "apply_router_hooks",
    "registered_router_hooks",
]
