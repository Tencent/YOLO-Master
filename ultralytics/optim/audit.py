# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Read-only optimizer parameter-group auditing utilities."""

from __future__ import annotations

from numbers import Real
from typing import Any


class OptimizerGroupAuditError(ValueError):
    """Raised when strict optimizer parameter-group validation fails."""

    def __init__(self, message: str, audit: dict[str, Any]) -> None:
        """Initialize the error and retain the serializable audit result."""
        super().__init__(message)
        self.audit = audit


def _number_or_none(value: Any) -> float | None:
    """Convert scalar optimizer metadata to a JSON-serializable float."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Real):
        return float(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except (RuntimeError, TypeError, ValueError):
            return None
        if isinstance(scalar, Real) and not isinstance(scalar, bool):
            return float(scalar)
    return None


def _parameter_numel(parameter: Any) -> int:
    """Return a parameter's element count without assuming a concrete tensor type."""
    numel = getattr(parameter, "numel", None)
    if not callable(numel):
        return 0
    try:
        return int(numel())
    except (RuntimeError, TypeError, ValueError):
        return 0


def _group_name(group: dict[str, Any], index: int) -> str:
    """Return explicit group semantics without inferring roles from parameter names."""
    # ``param_group`` is the existing Trainer field and is treated as a legacy
    # explicit semantic label. No optimizer group is mutated by this helper.
    for key in ("group_name", "name", "role", "param_group"):
        value = group.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return f"group_{index}"


def _parameter_record(
    parameter_id: int,
    model_parameters: dict[int, dict[str, Any]],
    occurrences: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Build a serializable issue record for one parameter identity."""
    model_record = model_parameters.get(parameter_id)
    name = model_record["name"] if model_record is not None else f"<unknown:{parameter_id}>"
    return {
        "name": name,
        "parameter_id": parameter_id,
        "numel": model_record["numel"] if model_record is not None else 0,
        "requires_grad": model_record["requires_grad"] if model_record is not None else None,
        "groups": [
            {
                "index": occurrence["group_index"],
                "name": occurrence["group_name"],
                "position": occurrence["position"],
            }
            for occurrence in occurrences.get(parameter_id, [])
        ],
    }


def _format_strict_error(audit: dict[str, Any], sample_size: int = 5) -> str:
    """Format strict validation failures with short parameter and group samples."""
    parts = []
    for issue_name in ("missing_trainable", "duplicated", "unknown_optimizer_parameters"):
        records = audit[issue_name]
        if not records:
            continue
        samples = []
        for record in records[:sample_size]:
            groups = ",".join(group["name"] for group in record.get("groups", []))
            samples.append(f"{record['name']}[{groups or 'no optimizer group'}]")
        parts.append(f"{issue_name}={len(records)} ({'; '.join(samples)})")
    return "Optimizer parameter-group audit failed: " + "; ".join(parts)


def audit_optimizer_param_groups(model: Any, optimizer: Any, *, strict: bool = False) -> dict[str, Any]:
    """Audit model-to-optimizer parameter coverage without mutating either object.

    The returned dictionary is a point-in-time snapshot of the optimizer. Calling
    the function again after scheduler or resume updates reports the then-current
    ``param_groups`` values. Missing ``initial_lr`` metadata is represented as
    ``None``.

    Args:
        model: Model exposing ``named_parameters()``.
        optimizer: Optimizer exposing ``param_groups`` and optional ``defaults``.
        strict: Raise for missing, duplicated, or unknown parameters. Frozen
            parameters in the optimizer are always reported but remain non-fatal.

    Returns:
        JSON-serializable optimizer parameter-group audit dictionary.

    Raises:
        OptimizerGroupAuditError: If ``strict=True`` and a fatal coverage issue is
            detected.
    """
    model_parameters: dict[int, dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        parameter_id = id(parameter)
        if parameter_id not in model_parameters:
            model_parameters[parameter_id] = {
                "name": name,
                "parameter_id": parameter_id,
                "requires_grad": bool(getattr(parameter, "requires_grad", False)),
                "numel": _parameter_numel(parameter),
            }

    optimizer_defaults = getattr(optimizer, "defaults", {}) or {}
    default_lr = _number_or_none(optimizer_defaults.get("lr")) if isinstance(optimizer_defaults, dict) else None
    occurrences: dict[int, list[dict[str, Any]]] = {}
    optimizer_parameters: dict[int, Any] = {}
    group_summaries = []
    optimizer_parameter_occurrence_count = 0

    for group_index, group in enumerate(getattr(optimizer, "param_groups", ())):
        semantic_name = _group_name(group, group_index)
        parameters = list(group.get("params", ()))
        parameter_names = []
        total_element_count = 0
        trainable_element_count = 0
        frozen_element_count = 0

        for position, parameter in enumerate(parameters):
            parameter_id = id(parameter)
            optimizer_parameters.setdefault(parameter_id, parameter)
            optimizer_parameter_occurrence_count += 1
            occurrences.setdefault(parameter_id, []).append(
                {"group_index": group_index, "group_name": semantic_name, "position": position}
            )
            model_record = model_parameters.get(parameter_id)
            parameter_names.append(model_record["name"] if model_record is not None else f"<unknown:{parameter_id}>")
            numel = _parameter_numel(parameter)
            total_element_count += numel
            if bool(getattr(parameter, "requires_grad", False)):
                trainable_element_count += numel
            else:
                frozen_element_count += numel

        lr = _number_or_none(group.get("lr"))
        initial_lr = _number_or_none(group.get("initial_lr"))
        weight_decay = _number_or_none(group.get("weight_decay"))
        explicit_lr_scale = _number_or_none(group.get("lr_scale"))
        base_lr = _number_or_none(group.get("base_lr"))
        if base_lr is None:
            base_lr = default_lr
        if explicit_lr_scale is not None:
            lr_scale = explicit_lr_scale
            lr_scale_source = "group.lr_scale"
        elif lr is not None and base_lr not in {None, 0.0}:
            lr_scale = lr / base_lr
            lr_scale_source = "lr/base_lr"
        else:
            lr_scale = None
            lr_scale_source = None

        group_summaries.append(
            {
                "index": group_index,
                "name": semantic_name,
                "tensor_count": len(parameters),
                "total_element_count": total_element_count,
                "trainable_element_count": trainable_element_count,
                "frozen_element_count": frozen_element_count,
                "lr": lr,
                "initial_lr": initial_lr,
                "weight_decay": weight_decay,
                "base_lr": base_lr,
                "lr_scale": lr_scale,
                "lr_scale_source": lr_scale_source,
                "parameter_names": parameter_names,
            }
        )

    trainable_ids = {parameter_id for parameter_id, record in model_parameters.items() if record["requires_grad"]}
    optimizer_ids = set(optimizer_parameters)
    missing_ids = sorted(trainable_ids - optimizer_ids, key=lambda item: model_parameters[item]["name"])
    duplicated_ids = sorted(
        (parameter_id for parameter_id, locations in occurrences.items() if len(locations) > 1),
        key=lambda item: model_parameters.get(item, {}).get("name", f"<unknown:{item}>"),
    )
    frozen_ids = sorted(
        (
            parameter_id
            for parameter_id in optimizer_ids & set(model_parameters)
            if not model_parameters[parameter_id]["requires_grad"]
        ),
        key=lambda item: model_parameters[item]["name"],
    )
    unknown_ids = sorted(optimizer_ids - set(model_parameters))

    missing_trainable = [
        {
            **model_parameters[parameter_id],
            "groups": [],
        }
        for parameter_id in missing_ids
    ]
    duplicated = [_parameter_record(parameter_id, model_parameters, occurrences) for parameter_id in duplicated_ids]
    frozen_in_optimizer = [
        _parameter_record(parameter_id, model_parameters, occurrences) for parameter_id in frozen_ids
    ]
    unknown_optimizer_parameters = [
        {
            **_parameter_record(parameter_id, model_parameters, occurrences),
            "numel": _parameter_numel(optimizer_parameters[parameter_id]),
            "requires_grad": bool(getattr(optimizer_parameters[parameter_id], "requires_grad", False)),
        }
        for parameter_id in unknown_ids
    ]

    duplicated_trainable_ids = trainable_ids & set(duplicated_ids)
    audit = {
        "group_count": len(group_summaries),
        "groups": group_summaries,
        "model_parameters": list(model_parameters.values()),
        "missing_trainable": missing_trainable,
        "duplicated": duplicated,
        "frozen_in_optimizer": frozen_in_optimizer,
        "unknown_optimizer_parameters": unknown_optimizer_parameters,
        "missing_trainable_count": len(missing_trainable),
        "duplicated_count": len(duplicated),
        "frozen_in_optimizer_count": len(frozen_in_optimizer),
        "unknown_optimizer_parameter_count": len(unknown_optimizer_parameters),
        "trainable_parameter_count": len(trainable_ids),
        "trainable_element_count": sum(model_parameters[item]["numel"] for item in trainable_ids),
        "optimizer_parameter_occurrence_count": optimizer_parameter_occurrence_count,
        "optimizer_unique_parameter_count": len(optimizer_ids),
        "optimizer_unique_element_count": sum(
            _parameter_numel(parameter) for parameter in optimizer_parameters.values()
        ),
        "trainable_coverage_complete": not missing_trainable,
        "trainable_coverage_exactly_once": not missing_trainable and not duplicated_trainable_ids,
        "has_duplicates": bool(duplicated),
        "has_missing_trainable": bool(missing_trainable),
        "has_frozen_in_optimizer": bool(frozen_in_optimizer),
        "has_unknown_parameters": bool(unknown_optimizer_parameters),
    }

    if strict and (missing_trainable or duplicated or unknown_optimizer_parameters):
        raise OptimizerGroupAuditError(_format_strict_error(audit), audit)
    return audit


__all__ = ["OptimizerGroupAuditError", "audit_optimizer_param_groups"]
