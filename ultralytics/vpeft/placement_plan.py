"""Versioned interchange contract between V-PEFT solvers and LoRA injection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class PlacementTarget:
    """One adapter placement target emitted by a planner."""

    name: str
    variant: str = "lora"
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "variant": self.variant, "rank": int(self.rank)}


@dataclass(frozen=True)
class PlacementPlan:
    """Serializable, auditable planner result consumed by the adapter layer."""

    model_fingerprint: str
    planner_backend: str
    solver: str
    budget: dict[str, int]
    targets: tuple[PlacementTarget, ...] = ()
    constraints: dict[str, list[str]] = field(default_factory=lambda: {"hard": [], "soft": []})
    predicted_delta: float | None = None
    confidence: float | None = None
    status: str = "FALLBACK"
    refusal_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported PlacementPlan schema_version={self.schema_version}")
        if self.status not in {"ADAPT", "ACCEPT", "REFUSE", "FALLBACK"}:
            raise ValueError(f"invalid PlacementPlan status={self.status!r}")
        if int(self.budget.get("max_adapter_params", 0)) < 0:
            raise ValueError("max_adapter_params must be non-negative")

    @property
    def fingerprint(self) -> str:
        """Return a stable hash of the plan payload."""
        payload = json.dumps(self.to_dict(include_fingerprint=False), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "model_fingerprint": self.model_fingerprint,
            "planner_backend": self.planner_backend,
            "solver": self.solver,
            "budget": dict(self.budget),
            "targets": [target.to_dict() for target in self.targets],
            "constraints": {key: list(value) for key, value in self.constraints.items()},
            "predicted_delta": self.predicted_delta,
            "confidence": self.confidence,
            "status": self.status,
            "refusal_reason": self.refusal_reason,
            "metadata": dict(self.metadata),
        }
        if include_fingerprint:
            payload["plan_fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlacementPlan":
        targets = tuple(
            PlacementTarget(str(item["name"]), str(item.get("variant", "lora")), int(item.get("rank", 0)))
            for item in payload.get("targets", [])
        )
        plan = cls(
            schema_version=int(payload.get("schema_version", 1)),
            model_fingerprint=str(payload.get("model_fingerprint", "")),
            planner_backend=str(payload.get("planner_backend", "legacy")),
            solver=str(payload.get("solver", "none")),
            budget={key: int(value) for key, value in dict(payload.get("budget", {})).items()},
            targets=targets,
            constraints={key: list(value) for key, value in dict(payload.get("constraints", {})).items()},
            predicted_delta=payload.get("predicted_delta"),
            confidence=payload.get("confidence"),
            status=str(payload.get("status", "FALLBACK")),
            refusal_reason=payload.get("refusal_reason"),
            metadata=dict(payload.get("metadata", {})),
        )
        expected = payload.get("plan_fingerprint")
        if expected is not None and expected != plan.fingerprint:
            raise ValueError("PlacementPlan fingerprint mismatch")
        return plan


__all__ = ["PlacementPlan", "PlacementTarget"]
