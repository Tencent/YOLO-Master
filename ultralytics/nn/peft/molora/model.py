"""MoLoRA model wrapper and PEFT-style entry point.

Provides:
  - get_peft_molora_model(model, config): wrap an Ultralytics model with MoLoRA
  - MoLoRAModel: convenience wrapper with aux_loss, merge/unmerge, checkpointing
"""

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

from ultralytics.utils import LOGGER
from .config import MoLoRAConfig, MoLoRAConfigBuilder
from .layer import MoLoRALayer
from .utils import mark_only_molora_as_trainable, count_parameters

if TYPE_CHECKING:
    from ultralytics.vpeft import PlacementPlan


def get_peft_molora_model(
    model: nn.Module,
    config: Union[MoLoRAConfig, Dict[str, Any]],
    *,
    placement_plan: Optional[Union["PlacementPlan", Mapping[str, Any]]] = None,
) -> nn.Module:
    """Wrap an Ultralytics DetectionModel (or any nn.Module) with MoLoRA layers.

    Args:
        model: The base model to adapt.
        config: Either a MoLoRAConfig dataclass or a dict with keys:
            r, alpha, num_experts, top_k, router_type,
            target_modules (list of module names), dropout, etc.
        placement_plan: Optional V-PEFT PlacementPlan or serialized mapping.
            Explicit plans are strict: only ACCEPT/ADAPT plans with LoRA-compatible
            targets are applied, and the entire plan is validated before mutation.

    Returns:
        The model with selected layers replaced by MoLoRALayer wrappers.
        Note: this modifies the model **in-place**.
    """
    # Prevent double-wrapping (P1-8 fix: comprehensive check for peft layers)
    if getattr(model, "molora_enabled", False):
        LOGGER.warning("[MoLoRA] Model already has MoLoRA enabled. Skipping re-application.")
        return model

    plan, planned_ranks, planned_variants = _resolve_molora_placement_plan(model, placement_plan)

    # Check if any target module is already a PeftAdapter (prevents parameter name conflicts)
    try:
        from peft.tuners.adapter_prefix_tuning import PrefixEncoder

        _peft_classes = tuple([PrefixEncoder])
        # Common peft adapter classes to detect
        for cls_name in ("LoraLayer", "AdaLoRALayer", "IA3Layer", "AdaloraLayer"):
            try:
                mod = __import__("peft.tuners", fromlist=[cls_name])
                for part in cls_name.split("."):
                    mod = getattr(mod, part)
                _peft_classes = _peft_classes + (mod,)
            except AttributeError:
                pass
    except ImportError:
        _peft_classes = ()

    modules_dict = dict(model.named_modules())
    for name, module in modules_dict.items():
        if _peft_classes and isinstance(module, _peft_classes):
            if plan is not None:
                raise ValueError(f"PlacementPlan cannot be applied because layer {name!r} is already a PEFT adapter")
            LOGGER.warning(
                f"[MoLoRA] Layer '{name}' is already a PEFT adapter. Skipping full model wrap "
                f"to prevent parameter name conflicts. Use only_molora=True to add MoLoRA alongside."
            )
            return model

    if isinstance(config, MoLoRAConfig):
        cfg = config
    else:
        cfg = MoLoRAConfig(**{k: v for k, v in config.items() if k in MoLoRAConfig.__dataclass_fields__})

    # Resolve target modules
    target_modules = list(planned_ranks) if plan is not None else getattr(cfg, "target_modules", None)
    if plan is not None:
        cfg.target_modules = list(target_modules)
    elif target_modules is None or not target_modules:
        LOGGER.warning("[MoLoRA] No target_modules specified; running auto-detection.")
        target_modules = MoLoRAConfigBuilder.auto_detect_targets(
            model,
            r=cfg.r,
            include_moe=getattr(cfg, "include_moe", True),
            include_attention=getattr(cfg, "include_attention", False),
            include_head=getattr(cfg, "include_head", False),
            only_backbone=getattr(cfg, "only_backbone", False),
            exclude_modules=getattr(cfg, "exclude_modules", None),
            allow_depthwise=getattr(cfg, "allow_depthwise", False),
            kernels=getattr(cfg, "kernels", None),
            skip_stem=getattr(cfg, "skip_stem", False),
            min_channels=getattr(cfg, "min_channels", 0),
            only_3x3=getattr(cfg, "only_3x3", False),
        )
        if not target_modules:
            LOGGER.warning("[MoLoRA] Auto-detection found no compatible layers. Returning model unchanged.")
            return model
        cfg.target_modules = list(target_modules)

    # Pre-build every replacement before changing module ownership. Validation
    # errors therefore cannot leave a partially wrapped model.
    modules_dict = dict(model.named_modules())
    replacements = []
    original_trainable = {}
    for name in target_modules:
        if name not in modules_dict:
            if plan is not None:
                raise RuntimeError(f"validated PlacementPlan target {name!r} disappeared before MoLoRA injection")
            continue
        base_layer = modules_dict[name]
        if not isinstance(base_layer, (nn.Conv2d, nn.Linear)):
            if plan is not None:
                raise RuntimeError(f"validated PlacementPlan target {name!r} changed type before MoLoRA injection")
            continue

        # Parent module and local attribute name for in-place replacement
        parent_name, child_name = _parent_child_name(name)
        parent = _get_submodule(model, parent_name) if parent_name else model
        if parent is None or not hasattr(parent, child_name):
            if plan is not None:
                raise RuntimeError(f"validated PlacementPlan target {name!r} has no writable parent")
            continue

        layer_rank = planned_ranks.get(name, cfg.r)
        original_trainable.update((parameter, parameter.requires_grad) for parameter in base_layer.parameters())
        try:
            molora_layer = MoLoRALayer(
                base_layer=base_layer,
                r=layer_rank,
                alpha=cfg.alpha,
                num_experts=cfg.num_experts,
                top_k=cfg.top_k,
                router_type=cfg.router_type,
                dropout=cfg.dropout,
                use_rslora=getattr(cfg, "use_rslora", True),
                balance_loss_coef=cfg.balance_loss_coef,
                z_loss_coef=cfg.z_loss_coef,
                diversity_loss_coef=cfg.diversity_loss_coef,
                expert_init=cfg.expert_init,
                share_moe_registry=cfg.share_moe_registry,
                router_hidden_dim=getattr(cfg, "router_hidden_dim", None),
                capacity_factor=cfg.capacity_factor,
                expert_dropout=cfg.expert_dropout,
                top_k_warmup=cfg.top_k_warmup,
                warmup_steps=cfg.warmup_steps,
                domain_experts=getattr(cfg, "domain_experts", None),
            )
        except Exception:
            for parameter, requires_grad in original_trainable.items():
                parameter.requires_grad = requires_grad
            raise
        replacements.append((name, parent, child_name, base_layer, molora_layer))

    applied = []
    try:
        for name, parent, child_name, base_layer, molora_layer in replacements:
            if plan is not None:
                molora_layer.molora_placement = {
                    "target": name,
                    "variant": planned_variants[name],
                    "rank": planned_ranks[name],
                    "plan_fingerprint": plan.fingerprint,
                }
            setattr(parent, child_name, molora_layer)
            applied.append((parent, child_name, base_layer))
    except Exception:
        for parent, child_name, base_layer in reversed(applied):
            setattr(parent, child_name, base_layer)
        for parameter, requires_grad in original_trainable.items():
            parameter.requires_grad = requires_grad
        raise
    wrapped = len(replacements)

    LOGGER.info(f"[MoLoRA] Wrapped {wrapped} layers with MoLoRA (E={cfg.num_experts}, K={cfg.top_k}).")

    # Attach metadata
    model.molora_config = cfg  # type: ignore[union-attr]
    model.molora_enabled = True  # type: ignore[union-attr]
    if plan is not None:
        model.molora_placement_plan = plan.to_dict()  # type: ignore[union-attr]
        model.molora_placement_fingerprint = plan.fingerprint  # type: ignore[union-attr]
        model.molora_rank_map = dict(planned_ranks)  # type: ignore[union-attr]

    # Freeze all non-MoLoRA parameters so only adapter weights are trainable
    mark_only_molora_as_trainable(model)
    frozen_experts = list(getattr(cfg, "freeze_experts", None) or [])
    if frozen_experts:
        for module in model.modules():
            if isinstance(module, MoLoRALayer):
                module.freeze_experts(frozen_experts)
    LOGGER.info("[MoLoRA] Frozen non-MoLoRA parameters. Only adapter weights are trainable.")

    return model


def _resolve_molora_placement_plan(
    model: nn.Module,
    placement_plan: Optional[Union["PlacementPlan", Mapping[str, Any]]],
) -> Tuple[Optional["PlacementPlan"], Dict[str, int], Dict[str, str]]:
    """Normalize and fully validate an explicit V-PEFT plan for MoLoRA."""
    if placement_plan is None:
        return None, {}, {}

    from ultralytics.vpeft import PlacementPlan

    if isinstance(placement_plan, PlacementPlan):
        plan = placement_plan
    elif isinstance(placement_plan, Mapping):
        plan = PlacementPlan.from_dict(placement_plan)
    else:
        raise TypeError("placement_plan must be a PlacementPlan or serialized mapping")
    if plan.status not in {"ACCEPT", "ADAPT"}:
        raise ValueError(f"PlacementPlan status {plan.status!r} cannot be applied to MoLoRA")

    plan.validate_model(model)
    planned_ranks = {}
    planned_variants = {}
    for target in plan.targets:
        if target.name in planned_ranks:
            raise ValueError(f"PlacementPlan contains duplicate target {target.name!r}")
        variant = str(target.variant).lower()
        if variant not in {"lora", "molora"}:
            raise ValueError(f"PlacementPlan variant {target.variant!r} is not compatible with MoLoRA")
        planned_ranks[target.name] = int(target.rank)
        planned_variants[target.name] = variant
    return plan, planned_ranks, planned_variants


def _parent_child_name(full_name: str) -> tuple:
    """Split 'model.5.m.0.cv1' -> ('model.5.m.0', 'cv1')."""
    parts = full_name.rsplit(".", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "", parts[0]


def _get_submodule(model: nn.Module, path: str) -> Optional[nn.Module]:
    """Navigate to a submodule by dot-separated path."""
    if not path:
        return model
    parts = path.split(".")
    mod = model
    for p in parts:
        if hasattr(mod, p):
            mod = getattr(mod, p)
        elif p.isdigit() and isinstance(mod, (nn.Sequential, nn.ModuleList)):
            mod = mod[int(p)]
        else:
            return None
    return mod


def _move_calibration_batch(batch: Any, device: torch.device) -> Any:
    """Move nested calibration tensors to the model device without changing container types."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return type(batch)((key, _move_calibration_batch(value, device)) for key, value in batch.items())
    if isinstance(batch, tuple):
        return tuple(_move_calibration_batch(value, device) for value in batch)
    if isinstance(batch, list):
        return [_move_calibration_batch(value, device) for value in batch]
    return batch


def _run_calibration_forward(
    model: nn.Module,
    batch: Any,
    forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
) -> Any:
    """Run one calibration batch using common PyTorch batch conventions."""
    if forward_fn is not None:
        return forward_fn(model, batch)
    if isinstance(batch, torch.Tensor):
        return model(batch)
    if isinstance(batch, Mapping):
        return model(**batch)
    if isinstance(batch, (tuple, list)):
        return model(*batch)
    raise TypeError(
        "Unsupported calibration batch type. Provide a Tensor, tuple/list, mapping, or forward_fn(model, batch)."
    )


def calibrate_molora_merge_weights(
    model: nn.Module,
    calibration_data: Iterable[Any],
    *,
    max_batches: Optional[int] = None,
    forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
) -> Dict[str, Any]:
    """Collect layer-specific sparse routing weights from calibration forwards."""
    if calibration_data is None:
        raise ValueError("calibration_data is required for calibrated MoLoRA merge")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided")

    layers = {name: module for name, module in model.named_modules() if isinstance(module, MoLoRALayer)}
    if not layers:
        raise ValueError("No MoLoRALayer modules found for calibration")
    merged = [name for name, layer in layers.items() if layer.merged]
    if merged:
        raise RuntimeError(f"Cannot calibrate already merged MoLoRA layers: {', '.join(merged)}")

    modules = list(model.modules())
    training_states = [module.training for module in modules]
    device = next(model.parameters()).device
    batches = 0
    for layer in layers.values():
        layer.start_merge_calibration()

    try:
        model.eval()
        with torch.no_grad():
            for batch in calibration_data:
                if max_batches is not None and batches >= max_batches:
                    break
                batch = _move_calibration_batch(batch, device)
                _run_calibration_forward(model, batch, forward_fn)
                batches += 1
        if batches == 0:
            raise ValueError("calibration_data produced no batches")

        weights = {}
        observed_batches = {}
        for name, layer in layers.items():
            layer_weights, layer_batches = layer.finish_merge_calibration()
            weights[name] = layer_weights
            observed_batches[name] = layer_batches
        return {"batches": batches, "weights": weights, "observed_batches": observed_batches}
    finally:
        for layer in layers.values():
            layer.cancel_merge_calibration()
        for module, training in zip(modules, training_states):
            module.training = training


def _explicit_calibration_weights(
    calibration: Optional[Union[List[float], Mapping[str, List[float]]]],
    layer_name: str,
) -> Optional[List[float]]:
    """Resolve shared or per-layer explicit calibration weights."""
    if calibration is None:
        return None
    if isinstance(calibration, Mapping):
        weights = calibration.get(layer_name)
        if weights is None:
            raise ValueError(f"Missing explicit calibration weights for MoLoRA layer {layer_name!r}")
        return list(weights)
    return list(calibration)


def _validate_calibration_weights(weights: List[float], num_experts: int, layer_name: str) -> List[float]:
    """Validate and normalize a calibration vector before any layer is mutated."""
    tensor = torch.as_tensor(weights, dtype=torch.float32)
    if tensor.ndim != 1 or tensor.numel() != num_experts:
        raise ValueError(f"Calibration for MoLoRA layer {layer_name!r} must provide {num_experts} weights")
    if not torch.isfinite(tensor).all() or (tensor < 0).any() or float(tensor.sum()) <= 0:
        raise ValueError(f"Calibration for MoLoRA layer {layer_name!r} must be finite, non-negative, and non-zero")
    return (tensor / tensor.sum()).tolist()


def _calibration_fingerprint(weights: Mapping[str, List[float]], batches: int) -> str:
    """Hash normalized per-layer calibration weights for artifact reproducibility."""
    payload = json.dumps({"batches": int(batches), "weights": weights}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _materialize_calibration_batches(calibration_data: Iterable[Any], max_batches: Optional[int]) -> List[Any]:
    """Materialize the exact calibration subset used for pre/post-merge verification."""
    batches = []
    for batch in calibration_data:
        if max_batches is not None and len(batches) >= max_batches:
            break
        batches.append(batch)
    if not batches:
        raise ValueError("calibration_data produced no batches")
    return batches


def _flatten_output_tensors(value: Any, path: str = "output") -> List[Tuple[str, torch.Tensor]]:
    """Flatten tensors from common nested model-output containers."""
    if isinstance(value, torch.Tensor):
        return [(path, value)]
    if isinstance(value, Mapping):
        tensors = []
        for key in sorted(value, key=str):
            tensors.extend(_flatten_output_tensors(value[key], f"{path}.{key}"))
        return tensors
    if isinstance(value, (tuple, list)):
        tensors = []
        for index, item in enumerate(value):
            tensors.extend(_flatten_output_tensors(item, f"{path}[{index}]"))
        return tensors
    return []


def _capture_calibration_outputs(
    model: nn.Module,
    calibration_data: Iterable[Any],
    *,
    forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
) -> List[List[Tuple[str, torch.Tensor]]]:
    """Capture detached float32 tensor leaves while preserving module training flags."""
    modules = list(model.modules())
    training_states = [module.training for module in modules]
    device = next(model.parameters()).device
    captured = []
    try:
        model.eval()
        with torch.no_grad():
            for batch in calibration_data:
                output = _run_calibration_forward(model, _move_calibration_batch(batch, device), forward_fn)
                tensors = _flatten_output_tensors(output)
                if not tensors:
                    raise TypeError("MoLoRA fidelity verification requires model outputs containing tensors")
                captured.append([(path, tensor.detach().float().cpu().clone()) for path, tensor in tensors])
    finally:
        for module, training in zip(modules, training_states):
            module.training = training
    return captured


def _measure_output_fidelity(
    expected: List[List[Tuple[str, torch.Tensor]]],
    observed: List[List[Tuple[str, torch.Tensor]]],
    tolerance: Optional[float],
) -> Dict[str, Any]:
    """Measure aggregate normalized L2 output error across calibration batches."""
    if len(expected) != len(observed):
        raise RuntimeError(f"MoLoRA fidelity output batch mismatch: {len(expected)} != {len(observed)}")
    error_sq = 0.0
    reference_sq = 0.0
    absolute_error = 0.0
    max_absolute_error = 0.0
    compared_tensors = 0
    compared_elements = 0
    for batch_index, (expected_batch, observed_batch) in enumerate(zip(expected, observed)):
        if len(expected_batch) != len(observed_batch):
            raise RuntimeError(
                f"MoLoRA fidelity tensor-count mismatch in batch {batch_index}: "
                f"{len(expected_batch)} != {len(observed_batch)}"
            )
        for (expected_path, expected_tensor), (observed_path, observed_tensor) in zip(expected_batch, observed_batch):
            if expected_path != observed_path or expected_tensor.shape != observed_tensor.shape:
                raise RuntimeError(
                    "MoLoRA fidelity output structure changed after merge: "
                    f"{expected_path} {tuple(expected_tensor.shape)} != "
                    f"{observed_path} {tuple(observed_tensor.shape)}"
                )
            difference = observed_tensor - expected_tensor
            error_sq += float(difference.double().pow(2).sum())
            reference_sq += float(expected_tensor.double().pow(2).sum())
            absolute_error += float(difference.double().abs().sum())
            max_absolute_error = max(max_absolute_error, float(difference.abs().max()) if difference.numel() else 0.0)
            compared_tensors += 1
            compared_elements += difference.numel()

    normalized_l2_error = math.sqrt(error_sq) / max(math.sqrt(reference_sq), 1e-12)
    passed = tolerance is None or normalized_l2_error <= tolerance
    return {
        "verified": True,
        "passed": passed,
        "tolerance": tolerance,
        "batches": len(expected),
        "compared_tensors": compared_tensors,
        "compared_elements": compared_elements,
        "normalized_l2_error": normalized_l2_error,
        "mean_absolute_error": absolute_error / max(compared_elements, 1),
        "max_absolute_error": max_absolute_error,
    }


def merge_molora_layers(
    model: nn.Module,
    mode: str = "ema",
    *,
    sync_ema: bool = False,
    merge_authority: Optional[str] = None,
    calibration_data: Optional[Iterable[Any]] = None,
    calibration: Optional[Union[List[float], Mapping[str, List[float]]]] = None,
    max_batches: Optional[int] = None,
    forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
    verify_fidelity: bool = False,
    fidelity_tolerance: Optional[float] = None,
) -> Dict[str, Any]:
    """Merge MoLoRA layers, optionally verifying output fidelity with atomic rollback."""
    if mode not in {"ema", "uniform", "calibrated"}:
        raise ValueError("MoLoRA merge mode must be 'ema', 'uniform', or 'calibrated'")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided")
    if fidelity_tolerance is not None and (not math.isfinite(fidelity_tolerance) or fidelity_tolerance < 0):
        raise ValueError("fidelity_tolerance must be finite and non-negative")
    verify_fidelity = bool(verify_fidelity or fidelity_tolerance is not None)
    if verify_fidelity and calibration_data is None:
        raise ValueError("MoLoRA fidelity verification requires calibration_data")

    layers = {name: module for name, module in model.named_modules() if isinstance(module, MoLoRALayer)}
    if not layers:
        raise ValueError("No MoLoRALayer modules found for merge")
    already_merged = [name for name, layer in layers.items() if layer.merged]
    if verify_fidelity and already_merged:
        raise RuntimeError(f"Cannot verify already merged MoLoRA layers: {', '.join(already_merged)}")

    fidelity_batches = None
    calibration_input = calibration_data
    calibration_max_batches = max_batches
    if verify_fidelity:
        fidelity_batches = _materialize_calibration_batches(calibration_data, max_batches)
        calibration_input = fidelity_batches
        calibration_max_batches = None

    result: Dict[str, Any] = {"batches": 0, "weights": {}}
    if mode == "calibrated":
        if calibration is None and calibration_input is None:
            raise ValueError("calibrated merge requires calibration_data or explicit calibration weights")
        if calibration_input is not None:
            result = calibrate_molora_merge_weights(
                model,
                calibration_input,
                max_batches=calibration_max_batches,
                forward_fn=forward_fn,
            )

    resolved_weights = {}
    if mode == "calibrated":
        for name, layer in layers.items():
            weights = result.get("weights", {}).get(name)
            if weights is None:
                weights = _explicit_calibration_weights(calibration, name)
            resolved_weights[name] = _validate_calibration_weights(weights, layer.num_experts, name)

    calibration_fp = None
    if mode == "calibrated":
        calibration_fp = _calibration_fingerprint(resolved_weights, int(result.get("batches", 0)))

    expected_outputs = None
    if verify_fidelity:
        expected_outputs = _capture_calibration_outputs(model, fidelity_batches, forward_fn=forward_fn)

    merged_layers = []
    try:
        for name, layer in layers.items():
            was_merged = layer.merged
            weights = resolved_weights.get(name)
            metadata = None
            if mode == "calibrated":
                observed = result.get("observed_batches", {}).get(name, 0)
                metadata = {
                    "calibration_batches": observed,
                    "calibration_source": "data" if calibration_data is not None else "explicit",
                    "calibration_fingerprint": calibration_fp,
                }
            layer.merge_weights(
                mode=mode,
                calibration=weights,
                calibration_metadata=metadata,
                sync_ema=sync_ema,
                merge_authority=merge_authority,
            )
            if not was_merged and layer.merged:
                merged_layers.append(layer)

        if verify_fidelity:
            observed_outputs = _capture_calibration_outputs(model, fidelity_batches, forward_fn=forward_fn)
            fidelity = _measure_output_fidelity(expected_outputs, observed_outputs, fidelity_tolerance)
            result["fidelity"] = fidelity
            for layer in merged_layers:
                layer._merge_metadata.update(
                    {
                        "fidelity_verified": True,
                        "fidelity_passed": fidelity["passed"],
                        "fidelity_tolerance": fidelity["tolerance"],
                        "fidelity_normalized_l2_error": fidelity["normalized_l2_error"],
                        "fidelity_mean_absolute_error": fidelity["mean_absolute_error"],
                        "fidelity_max_absolute_error": fidelity["max_absolute_error"],
                        "fidelity_batches": fidelity["batches"],
                        "fidelity_compared_tensors": fidelity["compared_tensors"],
                    }
                )
            if not fidelity["passed"]:
                raise RuntimeError(
                    "MoLoRA merge fidelity check failed: "
                    f"normalized_l2_error={fidelity['normalized_l2_error']:.6g} exceeds "
                    f"tolerance={fidelity_tolerance:.6g}"
                )
    except Exception:
        for layer in reversed(merged_layers):
            layer.unmerge_weights()
        raise

    LOGGER.info("[MoLoRA] All layers merged.")
    return result


# ---------------------------------------------------------------------------
# Wrapper class (optional convenience)
# ---------------------------------------------------------------------------


class MoLoRAModel(nn.Module):
    """Thin wrapper around a base model that adds MoLoRA bookkeeping.

    Not required for training; you can use `get_peft_molora_model` directly
    and then call `mark_only_molora_as_trainable`. This wrapper is useful
    when you want a single object that exposes:
      - compute_aux_loss()
      - merge() / unmerge()
      - save_checkpoint() / load_checkpoint()
    """

    def __init__(
        self,
        model: nn.Module,
        config: Union[MoLoRAConfig, Dict[str, Any]],
        *,
        placement_plan: Optional[Union["PlacementPlan", Mapping[str, Any]]] = None,
    ):
        super().__init__()
        self.model = get_peft_molora_model(model, config, placement_plan=placement_plan)
        self.config = self.model.molora_config if hasattr(self.model, "molora_config") else config
        self.placement_plan = getattr(self.model, "molora_placement_plan", None)
        # get_peft_molora_model already called mark_only_molora_as_trainable

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_aux_loss(self) -> torch.Tensor:
        """Collect current-step MoLoRA auxiliary losses.

        Call this after forward() in the training loop and add it to the
        total loss. Canonical records are reset before each training forward.
        """
        device = next(self.model.parameters()).device
        from ultralytics.nn.modules.routing_protocol import collect_aux_loss

        return collect_aux_loss(self.model, device=device, include_kinds=("molora",))

    def merge(
        self,
        mode: str = "ema",
        *,
        sync_ema: bool = False,
        merge_authority: Optional[str] = None,
        calibration_data: Optional[Iterable[Any]] = None,
        calibration: Optional[Union[List[float], Mapping[str, List[float]]]] = None,
        max_batches: Optional[int] = None,
        forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
        verify_fidelity: bool = False,
        fidelity_tolerance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Merge all MoLoRA layers and optionally gate the artifact on measured output fidelity."""
        return merge_molora_layers(
            self.model,
            mode=mode,
            sync_ema=sync_ema,
            merge_authority=merge_authority,
            calibration_data=calibration_data,
            calibration=calibration,
            max_batches=max_batches,
            forward_fn=forward_fn,
            verify_fidelity=verify_fidelity,
            fidelity_tolerance=fidelity_tolerance,
        )

    def unmerge(self) -> None:
        """Unmerge all MoLoRALayer weights."""
        for m in self.model.modules():
            if isinstance(m, MoLoRALayer):
                m.unmerge_weights()
        LOGGER.info("[MoLoRA] All layers unmerged.")

    # ------------------------------------------------------------------
    # Domain & Continual Learning
    # ------------------------------------------------------------------

    def set_domain(self, domain: str) -> None:
        """Restrict all MoLoRALayers to domain-specific experts."""
        for m in self.model.modules():
            if isinstance(m, MoLoRALayer):
                m.set_domain(domain)
        LOGGER.info(f"[MoLoRA] Domain set to '{domain}'.")

    def clear_domain(self) -> None:
        """Clear domain restrictions on all MoLoRALayers."""
        for m in self.model.modules():
            if isinstance(m, MoLoRALayer):
                m.clear_domain()
        LOGGER.info("[MoLoRA] Domain restrictions cleared.")

    def freeze_experts(self, expert_indices: List[int]) -> None:
        """Freeze specified experts across all MoLoRALayers."""
        for m in self.model.modules():
            if isinstance(m, MoLoRALayer):
                m.freeze_experts(expert_indices)

    def unfreeze_experts(self, expert_indices: Optional[List[int]] = None) -> None:
        """Unfreeze specified or all experts across all MoLoRALayers."""
        for m in self.model.modules():
            if isinstance(m, MoLoRALayer):
                m.unfreeze_experts(expert_indices)

    # ------------------------------------------------------------------
    # Expert Replay (continual learning)
    # ------------------------------------------------------------------

    def save_expert_replay_buffer(self, domain: str, path: Optional[str] = None) -> Dict[str, Any]:
        """Save current expert weights for a domain into a replay buffer.

        This allows restoring old-domain experts later to prevent catastrophic
        forgetting when training on new domains.
        """
        buffer: Dict[str, Any] = {"domain": domain, "experts": {}}
        for name, m in self.model.named_modules():
            if isinstance(m, MoLoRALayer):
                buffer["experts"][name] = {
                    idx: {
                        "lora_A": m.experts[idx].lora_A.state_dict(),
                        "lora_B": m.experts[idx].lora_B.state_dict(),
                    }
                    for idx in range(m.num_experts)
                }
        if path is not None:
            torch.save(buffer, path)
            LOGGER.info(f"[MoLoRA] Expert replay buffer saved to {path} for domain '{domain}'")
        return buffer

    def load_expert_replay_buffer(self, buffer: Union[str, Dict[str, Any]], domain: Optional[str] = None) -> None:
        """Load expert weights from a replay buffer.

        Args:
            buffer: Either a file path or a dict returned by save_expert_replay_buffer.
            domain: Optional domain name to verify (if buffer is a dict with 'domain' key).
        """
        if isinstance(buffer, str):
            buffer = torch.load(buffer, map_location="cpu")
        if domain is not None and buffer.get("domain") != domain:
            LOGGER.warning(f"[MoLoRA] Replay buffer domain mismatch: {buffer.get('domain')} vs {domain}")
        for name, m in self.model.named_modules():
            if isinstance(m, MoLoRALayer) and name in buffer["experts"]:
                for idx, states in buffer["experts"][name].items():
                    m.experts[idx].lora_A.load_state_dict(states["lora_A"])
                    m.experts[idx].lora_B.load_state_dict(states["lora_B"])
        LOGGER.info(f"[MoLoRA] Expert replay buffer loaded for domain '{buffer.get('domain')}'")

    def save_checkpoint(self, path: str) -> None:
        """Save a versioned MoLoRA-only checkpoint with explicit compatibility metadata.

        Includes registered buffers (e.g. ``_step_count``, ``_usage_ema``)
        alongside trainable parameters, since these carry training state
        needed for correct resume.
        """
        molora_keys = ("lora_A", "lora_B", "router", "molora", "_step_count", "_usage_ema", "_domain_active_mask")
        config = asdict(self.config) if is_dataclass(self.config) else dict(self.config)
        state = {
            "schema_version": 1,
            "format": "molora_adapter",
            "config": config,
            "structure": _molora_structure(self.model),
            "model_fingerprint": _molora_model_fingerprint(self.model),
            "placement_plan": getattr(self.model, "molora_placement_plan", None),
            "merge_metadata": {
                name: dict(getattr(layer, "_merge_metadata", {}))
                for name, layer in self.model.named_modules()
                if isinstance(layer, MoLoRALayer)
            },
            "state_dict": {k: v for k, v in self.model.state_dict().items() if any(p in k for p in molora_keys)},
        }
        torch.save(state, path)
        LOGGER.info(f"[MoLoRA] Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load a checkpoint, rejecting incompatible configuration or partial state."""
        state = torch.load(path, map_location="cpu")
        if not isinstance(state, dict) or "state_dict" not in state:
            raise ValueError(
                "Invalid MoLoRA checkpoint: expected a dict with 'state_dict'. "
                "Legacy/unversioned checkpoints must be re-exported with save_checkpoint()."
            )
        if state.get("schema_version") != 1 or state.get("format") != "molora_adapter":
            raise ValueError(
                f"Unsupported MoLoRA checkpoint schema: version={state.get('schema_version')!r}, "
                f"format={state.get('format')!r}; expected version=1, format='molora_adapter'."
            )
        saved_config = state.get("config")
        if not isinstance(saved_config, dict):
            raise ValueError("Invalid MoLoRA checkpoint: missing complete 'config' dictionary.")
        current_config = asdict(self.config) if is_dataclass(self.config) else dict(self.config)
        for key in ("r", "num_experts", "top_k", "router_type", "target_modules"):
            saved_value = saved_config.get(key)
            current_value = current_config.get(key)
            if key == "target_modules":
                saved_value = sorted(saved_value or [])
                current_value = sorted(current_value or [])
            if saved_value != current_value:
                raise ValueError(
                    f"MoLoRA checkpoint config mismatch for {key}: checkpoint={saved_value!r}, model={current_value!r}."
                )
        saved_structure = state.get("structure")
        current_structure = _molora_structure(self.model)
        if saved_structure != current_structure:
            raise ValueError("MoLoRA checkpoint structure mismatch (target names, layer types, or dimensions differ).")
        saved_fingerprint = state.get("model_fingerprint")
        if not isinstance(saved_fingerprint, str) or not saved_fingerprint:
            raise ValueError("Invalid MoLoRA checkpoint: missing model_fingerprint binding.")
        if saved_fingerprint != _molora_model_fingerprint(self.model):
            raise ValueError("MoLoRA checkpoint model fingerprint mismatch; rebuild the adapter for this model.")
        saved_plan = state.get("placement_plan")
        current_plan = getattr(self.model, "molora_placement_plan", None)
        if saved_plan != current_plan:
            raise ValueError("MoLoRA checkpoint PlacementPlan mismatch; rebuild the adapter with the same plan.")
        saved_merge_metadata = state.get("merge_metadata")
        if not isinstance(saved_merge_metadata, dict):
            raise ValueError("Invalid MoLoRA checkpoint: missing merge_metadata dictionary.")
        checkpoint_sd = state["state_dict"]
        if not isinstance(checkpoint_sd, dict):
            raise ValueError("Invalid MoLoRA checkpoint: 'state_dict' must be a dictionary.")
        expected_keys = {k for k in self.model.state_dict() if _is_molora_state_key(k)}
        missing_usage_ema = sorted(k for k in expected_keys - set(checkpoint_sd) if k.endswith("._usage_ema"))
        if missing_usage_ema:
            current_state = self.model.state_dict()
            checkpoint_sd = dict(checkpoint_sd)
            checkpoint_sd.update({key: current_state[key] for key in missing_usage_ema})
            LOGGER.warning(
                f"[MoLoRA] Checkpoint predates routing usage EMA; initialized {len(missing_usage_ema)} "
                "layer buffer(s) with uniform expert weights."
            )
        missing = sorted(expected_keys - set(checkpoint_sd))
        unexpected = sorted(set(checkpoint_sd) - expected_keys)
        if missing or unexpected:
            raise RuntimeError(
                "MoLoRA checkpoint state mismatch: "
                f"missing={missing[:5]} ({len(missing)} total), "
                f"unexpected={unexpected[:5]} ({len(unexpected)} total)."
            )
        try:
            self.model.load_state_dict(checkpoint_sd, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(f"MoLoRA checkpoint tensor shape mismatch: {exc}") from exc
        for name, layer in self.model.named_modules():
            if isinstance(layer, MoLoRALayer) and name in saved_merge_metadata:
                metadata = saved_merge_metadata[name]
                if not isinstance(metadata, dict):
                    raise ValueError(f"Invalid MoLoRA merge metadata for layer {name!r}.")
                layer._merge_metadata = dict(metadata)
        LOGGER.info(f"[MoLoRA] Checkpoint loaded from {path}")

    def param_stats(self) -> Dict[str, Any]:
        """Return parameter statistics for the wrapped model."""
        return count_parameters(self.model)


def _is_molora_state_key(key: str) -> bool:
    """Return whether a state key belongs to adapter parameters or state buffers."""
    return any(
        token in key
        for token in (
            "lora_A",
            "lora_B",
            "router",
            "loss_fn",
            "molora",
            "_step_count",
            "_usage_ema",
            "_domain_active_mask",
        )
    )


def _molora_structure(model: nn.Module) -> List[Dict[str, Any]]:
    """Describe wrapped layers sufficiently to reject incompatible adapters."""
    structure = []
    for name, layer in model.named_modules():
        if not isinstance(layer, MoLoRALayer):
            continue
        base = layer.base_layer
        item: Dict[str, Any] = {
            "name": name,
            "base_type": type(base).__name__,
            "r": layer.r,
            "num_experts": layer.num_experts,
            "top_k": layer.top_k,
            "router_type": layer.router_type,
            "in_features": getattr(base, "in_features", getattr(base, "in_channels", None)),
            "out_features": getattr(base, "out_features", getattr(base, "out_channels", None)),
        }
        if isinstance(base, nn.Conv2d):
            item.update({"kernel_size": tuple(base.kernel_size), "groups": base.groups})
        structure.append(item)
    return structure


def _molora_model_fingerprint(model: nn.Module) -> str:
    """Return a stable binding hash for an injected MoLoRA graph."""

    entries = []
    for name, module in model.named_modules():
        if name:
            entries.append((name, module.__class__.__qualname__))
    for name, parameter in model.named_parameters():
        entries.append((f"param:{name}", tuple(parameter.shape), str(parameter.dtype)))
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
