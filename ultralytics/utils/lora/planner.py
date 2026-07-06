# 🐧Please note that this file has been modified by Tencent on 2026/02/13. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Architecture-conditioned PEFT Planner for YOLO-Master.

Implements the regression model from Eq. 1 of the YOLO-Master PEFT paper:
    ΔmAP ≈ β₀ + β₁φ_attn + β₂φ_text + β₃φ_dw + β₄ξ_p

The Planner makes architecture-conditioned placement decisions for PEFT adapters,
including ACCEPT, REFUSE, and ADAPT decisions, with graceful fallback to
full fine-tuning when a refusal occurs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn

from ultralytics.utils import LOGGER
import weakref

# Weak-key cache: automatically invalidated when the model object is garbage-collected.
# This prevents stale entries from memory-address reuse across test runs or
# model re-creation, and avoids the need for explicit cache invalidation.
_fingerprint_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


class RefusalError(Exception):
    """Raised when the PEFT Planner refuses a configuration.

    This is a valid planning decision, not a failure. The caller should
    catch this and fall back to full fine-tuning (Full-SFT).
    """
    pass


@dataclass
class ArchitectureFingerprint:
    """Compact 5-dimensional architecture fingerprint.

    Attributes:
        phi_attn: Attention module ratio (attention modules / total conv+linear).
        phi_text: Text-fusion module ratio (text-fusion modules / total conv+linear).
        phi_dw: Depthwise convolution ratio (depthwise conv / total conv).
        phi_group: Grouped convolution ratio (grouped conv / total conv).
        phi_linear: Linear layer ratio (linear modules / total conv+linear).
    """
    phi_attn: float = 0.0
    phi_text: float = 0.0
    phi_dw: float = 0.0
    phi_group: float = 0.0
    phi_linear: float = 0.0

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        """Unwrap DDP / DataParallel / torch.compile wrapped models.

        Recursively drills through ``.module`` (DDP/DP) and ``._orig_mod``
        (torch.compile) to reach the underlying nn.Module.
        """
        while hasattr(model, "module"):
            model = model.module
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model

    @classmethod
    def compute(cls, model: nn.Module) -> "ArchitectureFingerprint":
        """Compute the architecture fingerprint from a PyTorch model.

        Always performs a real module scan via :meth:`_compute_from_modules`;
        the paper-calibrated family profiles are *not* used to override the
        fingerprint values.  This guarantees that the 5-dimensional vector is a
        true reflection of the model's weight topology, which is required for
        the regression model (Eq. 1) to generalise to novel architectures.

        Architecture-family detection (:meth:`_detect_architecture_family`) is
        still available as a standalone utility for downstream policy rules, but
        it does not alter the fingerprint values.

        Args:
            model: The PyTorch model to analyze.

        Returns:
            ArchitectureFingerprint: The computed 5-dimensional fingerprint.
        """
        model = cls._unwrap_model(model)
        cached = _fingerprint_cache.get(model)
        if cached is not None:
            return cached

        # Always compute from real module scan, never override with hardcoded
        # family profiles.  This ensures the 5-dimensional fingerprint is a
        # true reflection of the model's architecture.
        fingerprint = cls._compute_from_modules(model)
        _fingerprint_cache[model] = fingerprint
        return fingerprint

    @classmethod
    def invalidate_cache(cls, model: nn.Module) -> None:
        """Invalidate the cached fingerprint for a model.

        Call this after the model architecture has been modified
        (e.g. after PEFT adapter injection changes the module hierarchy).
        """
        model = cls._unwrap_model(model)
        _fingerprint_cache.pop(model, None)

    @staticmethod
    def _detect_architecture_family(model: nn.Module) -> Optional[str]:
        """Detect known architecture family from iconic module types.

        Heuristic rules (evaluated in priority order):
          - RT-DETR: contains RTDETRDecoder or MultiheadAttention
          - YOLO-World: contains text/clip fusion modules
          - YOLO12: contains A2C2f or AAttn (not C2PSA's internal Attention)
          - YOLO-Master-MoE: contains MoE router/expert layers
          - YOLO-CNN: default fallback (no attention, no text-fusion, no MoE)

        The C2PSA block in YOLO11 contains a small Attention module, but
        it is self-contained and does not change the overall architecture
        family; the paper treats YOLO11 as "dense-conv only" (φ_attn=0).
        Therefore, we only flag *A2C2f/AAttn* (YOLO12's explicit attention
        blocks) as attention architecture, not C2PSA's internal Attention.
        """
        has_a2c2f = False
        has_rtdetr = False
        has_text_fusion = False
        has_moe = False

        for name, module in model.named_modules():
            cls_name = module.__class__.__name__
            lname = name.lower()

            # YOLO12 signature: A2C2f blocks with AAttn layers
            if "A2C2f" in cls_name or "AAttn" in cls_name:
                has_a2c2f = True

            # RT-DETR signature: decoder or vanilla MultiheadAttention
            if "RTDETR" in cls_name or "MultiheadAttention" in cls_name:
                has_rtdetr = True

            # YOLO-World signature: text/clip fusion
            if any(k in lname for k in ("text_encoder", "clip", "text_fusion", "world_embed", "text_proj")):
                has_text_fusion = True

            # MoE signature
            if any(k in lname for k in ("moe_router", "moe_expert", "moe_gate")):
                has_moe = True

        # Priority order matters: RT-DETR > World > YOLO12 > MoE > CNN
        if has_rtdetr:
            return "rtdetr"
        if has_text_fusion:
            return "yolo_world"
        if has_a2c2f:
            return "yolo12"
        if has_moe:
            return "yolo_master_moe"
        # No iconic attention / text-fusion / MoE detected → CNN family
        return "yolo_cnn"

    @classmethod
    def _from_architecture_family(cls, family: str) -> "ArchitectureFingerprint":
        """Return paper-calibrated φ_attn / φ_text for a known family.

        These values are taken directly from the paper's experimental
        description (Sec. 4 Setup and Fig. 4):
          - YOLO-CNN (YOLOv8/v9/v10/v11): φ_attn = 0, φ_text = 0
          - YOLO12 (A2C2f attention):       φ_attn = 0.45, φ_text = 0
          - YOLO-World (text-fusion):       φ_attn = 0.45, φ_text = 0.5
          - RT-DETR (pure Transformer):     φ_attn = 0.85, φ_text = 0
          - YOLO-Master-MoE:                φ_attn = 0,  φ_text = 0
        """
        profiles = {
            "yolo_cnn":       cls(phi_attn=0.0,  phi_text=0.0,  phi_dw=0.0, phi_group=0.0, phi_linear=0.0),
            "yolo12":         cls(phi_attn=0.45, phi_text=0.0,  phi_dw=0.0, phi_group=0.0, phi_linear=0.0),
            "yolo_world":     cls(phi_attn=0.45, phi_text=0.5,  phi_dw=0.0, phi_group=0.0, phi_linear=0.0),
            "rtdetr":         cls(phi_attn=0.85, phi_text=0.0,  phi_dw=0.0, phi_group=0.0, phi_linear=0.0),
            "yolo_master_moe": cls(phi_attn=0.0, phi_text=0.0,  phi_dw=0.0, phi_group=0.0, phi_linear=0.0),
        }
        return profiles.get(family, cls())

    @classmethod
    def _compute_from_modules(cls, model: nn.Module) -> "ArchitectureFingerprint":
        """Improved module-scan counting that avoids deep-nesting inflation.

        Key differences from the original naive scan:
          - Attention counting uses **iconic module types** (A2C2f, AAttn,
            MultiheadAttention, RTDETRDecoder, MSDEFORMAttention) rather than
            string-matching on every submodule name.  This prevents a single
            AAttn container from being counted 10+ times via its qkv/proj/pe
            children.
          - Depthwise and grouped conv are counted on the actual Conv2d layers
            as before.
        """
        total_conv = 0
        total_linear = 0
        attn_count = 0
        text_count = 0
        dw_count = 0
        group_count = 0
        linear_count = 0

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                total_conv += 1
                if (
                    module.in_channels == module.out_channels
                    == module.groups
                ):
                    dw_count += 1
                elif module.groups > 1:
                    group_count += 1
            elif isinstance(module, nn.Linear):
                total_linear += 1
                linear_count += 1

            # Iconic attention modules only (not every child submodule)
            cls_name = module.__class__.__name__
            if cls_name in (
                "AAttn", "MultiheadAttention", "MSDEFORMAttention",
                "RTDETRDecoder", "DeformableAttention",
            ):
                attn_count += 1

            # Text-fusion detection (string-based, low false-positive rate)
            lname = name.lower()
            if any(k in lname for k in ("text_encoder", "clip", "text_fusion", "world_embed", "fusion")):
                text_count += 1

        total_modules = total_conv + total_linear
        if total_modules == 0:
            LOGGER.warning(
                "[Planner] Model has no Conv2d or Linear modules. "
                "Returning zero fingerprint."
            )
            return cls()
        if total_conv == 0:
            total_conv = 1

        return cls(
            phi_attn=attn_count / total_modules,
            phi_text=text_count / total_modules,
            phi_dw=dw_count / total_conv,
            phi_group=group_count / total_conv,
            phi_linear=linear_count / total_modules,
        )


@dataclass
class PEFTVariantProfile:
    """Variant-level profile used in the regression model.

    Attributes:
        xi: Variant-level coefficient (from fitted regression, Eq. 1).
        supports_conv: Whether this variant supports convolutional layers.
        supports_linear: Whether this variant supports linear layers.
        supports_attention: Whether this variant supports attention layers.
        supports_text_fusion: Whether this variant supports text-fusion layers.
    """
    xi: float = 0.0
    supports_conv: bool = True
    supports_linear: bool = True
    supports_attention: bool = False
    supports_text_fusion: bool = False

    @classmethod
    def from_variant(cls, variant: str) -> "PEFTVariantProfile":
        """Get the profile for a named PEFT variant.

        Args:
            variant: The PEFT variant name (e.g., 'lora', 'dora', 'loha').

        Returns:
            PEFTVariantProfile: The corresponding profile with default coefficients.
        """
        profiles = {
            # Calibrated against Table 1 (tab:core_wandb) of the YOLO-Master
            # PEFT paper.  xi values are fitted via least squares on the 12
            # canonical non-catastrophic data points (including ablations).
            "lora": cls(
                xi=0.0,
                supports_conv=True,
                supports_linear=True,
                supports_attention=True,
                supports_text_fusion=False,
            ),
            "dora": cls(
                xi=0.0050,
                supports_conv=True,
                supports_linear=True,
                supports_attention=True,
                supports_text_fusion=False,
            ),
            "loha": cls(
                xi=-0.0208,
                supports_conv=True,
                supports_linear=True,
                supports_attention=True,
                supports_text_fusion=True,
            ),
            "lokr": cls(
                xi=-0.0055,
                supports_conv=True,
                supports_linear=True,
                supports_attention=True,
                supports_text_fusion=False,
            ),
            "adalora": cls(
                xi=0.0,
                supports_conv=False,
                supports_linear=True,
                supports_attention=True,
                supports_text_fusion=False,
            ),
            "ia3": cls(
                xi=-0.0117,
                supports_conv=True,
                supports_linear=True,
                supports_attention=True,
                supports_text_fusion=True,
            ),
            # Uncalibrated placeholder — no experimental data in the paper.
            "oft": cls(
                xi=-0.1,
                supports_conv=True,
                supports_linear=True,
                supports_attention=True,
                supports_text_fusion=False,
            ),
            # Uncalibrated placeholder — no experimental data in the paper.
            "boft": cls(
                xi=-0.08,
                supports_conv=True,
                supports_linear=True,
                supports_attention=True,
                supports_text_fusion=False,
            ),
            "hra": cls(
                xi=0.0152,
                supports_conv=True,
                supports_linear=True,
                supports_attention=True,
                supports_text_fusion=False,
            ),
        }
        return profiles.get(
            variant.lower(),
            cls(
                xi=0.0,
                supports_conv=True,
                supports_linear=True,
                supports_attention=True,
                supports_text_fusion=False,
            ),
        )


@dataclass
class PlacementDecision:
    """Decision made by the PEFT Planner.

    Attributes:
        status: One of "ACCEPT", "REFUSE", or "ADAPT".
        recommended_variant: Recommended PEFT variant if ADAPT.
        recommended_rank: Recommended LoRA rank if ADAPT.
        predicted_delta: Predicted ΔmAP from the regression model.
        target_modules_hint: Hint list of target modules for downstream target
            detection.
        refusal_reason: Human-readable refusal reason if REFUSE.
        safety_overrides: Dict of config overrides to apply if ADAPT.
    """
    status: str = "ACCEPT"
    recommended_variant: Optional[str] = None
    recommended_rank: Optional[int] = None
    predicted_delta: Optional[float] = None
    target_modules_hint: Optional[List[str]] = None
    refusal_reason: Optional[str] = None
    safety_overrides: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.safety_overrides is None:
            self.safety_overrides = {}
        if self.status not in ("ACCEPT", "REFUSE", "ADAPT"):
            raise ValueError(f"Invalid status: {self.status}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize decision to a plain dictionary (for JSON / metadata)."""
        return {
            "status": self.status,
            "recommended_variant": self.recommended_variant,
            "recommended_rank": self.recommended_rank,
            "predicted_delta": self.predicted_delta,
            "refusal_reason": self.refusal_reason,
            "safety_overrides": dict(self.safety_overrides),
            "target_modules_hint_count": len(self.target_modules_hint or []),
        }


@dataclass
class DecisionAudit:
    """Structured audit record for a single Planner decision.

    Persisted to disk as JSON for post-hoc analysis, paper reproduction,
    and debugging.  One audit file per ``plan()`` call.
    """
    timestamp: str
    model_name: str
    fingerprint: Dict[str, float]
    variant: str
    requested_rank: int
    decision_status: str
    recommended_variant: Optional[str] = None
    recommended_rank: Optional[int] = None
    predicted_delta: Optional[float] = None
    refusal_reason: Optional[str] = None
    safety_overrides: Dict[str, Any] = field(default_factory=dict)
    target_modules_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "fingerprint": self.fingerprint,
            "variant": self.variant,
            "requested_rank": self.requested_rank,
            "decision_status": self.decision_status,
            "recommended_variant": self.recommended_variant,
            "recommended_rank": self.recommended_rank,
            "predicted_delta": self.predicted_delta,
            "refusal_reason": self.refusal_reason,
            "safety_overrides": self.safety_overrides,
            "target_modules_count": self.target_modules_count,
        }

    def save(self, audit_dir: Optional[Path] = None) -> Path:
        """Save the audit record to a JSON file.

        Args:
            audit_dir: Directory to store audit files. Defaults to
                ``runs/planner_audit/``.

        Returns:
            Path: The path of the written JSON file.
        """
        if audit_dir is None:
            audit_dir = Path("runs/planner_audit")
        audit_dir = Path(audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)

        # Use a timestamped filename to avoid collisions.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"planner_audit_{ts}.json"
        path = audit_dir / fname

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        LOGGER.info("[Planner] Audit saved to %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "DecisionAudit":
        """Load an audit record from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class LOVODataPoint:
    """Single (fingerprint, variant, ΔmAP) data point for LOVO cross-validation.

    Attributes:
        fingerprint: The 5-D architecture fingerprint.
        variant: PEFT variant name (e.g., 'lora', 'dora').
        delta_mAP: Measured ΔmAP from training.
        model_name: Optional model name for metadata.
        dataset: Optional dataset name.
        epochs: Training epochs.
        timestamp: ISO-8601 timestamp.
        notes: Free-form notes.
    """

    fingerprint: ArchitectureFingerprint
    variant: str
    delta_mAP: float
    model_name: str = ""
    dataset: str = ""
    epochs: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""

    def to_tuple(self) -> Tuple[ArchitectureFingerprint, str, float]:
        return (self.fingerprint, self.variant, self.delta_mAP)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fingerprint": {
                "phi_attn": self.fingerprint.phi_attn,
                "phi_text": self.fingerprint.phi_text,
                "phi_dw": self.fingerprint.phi_dw,
                "phi_group": self.fingerprint.phi_group,
                "phi_linear": self.fingerprint.phi_linear,
            },
            "variant": self.variant,
            "delta_mAP": self.delta_mAP,
            "model_name": self.model_name,
            "dataset": self.dataset,
            "epochs": self.epochs,
            "timestamp": self.timestamp,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LOVODataPoint":
        fp = data.get("fingerprint", {})
        return cls(
            fingerprint=ArchitectureFingerprint(
                phi_attn=fp.get("phi_attn", 0.0),
                phi_text=fp.get("phi_text", 0.0),
                phi_dw=fp.get("phi_dw", 0.0),
                phi_group=fp.get("phi_group", 0.0),
                phi_linear=fp.get("phi_linear", 0.0),
            ),
            variant=data["variant"],
            delta_mAP=data["delta_mAP"],
            model_name=data.get("model_name", ""),
            dataset=data.get("dataset", ""),
            epochs=data.get("epochs", 0),
            timestamp=data.get("timestamp", ""),
            notes=data.get("notes", ""),
        )


class LOVODataCollector:
    """Collects and persists (fingerprint, variant, ΔmAP) data points.

    This is the **data collection engine** for the LOVO cross-validation
    pipeline.  It stores points, serializes to JSON, and converts to the
    ``history`` format expected by :meth:`PEFTPlanner.fit`.
    """

    def __init__(self, data_points: Optional[List[LOVODataPoint]] = None):
        self.data_points: List[LOVODataPoint] = list(data_points) if data_points else []

    def add(
        self,
        point: Union[LOVODataPoint, Tuple[ArchitectureFingerprint, str, float]],
        **metadata,
    ) -> None:
        """Add a data point.

        Args:
            point: Either a LOVODataPoint or a (fingerprint, variant, delta_mAP) tuple.
            **metadata: Extra fields when passing a tuple.
        """
        if isinstance(point, tuple):
            fp, variant, delta_mAP = point
            point = LOVODataPoint(
                fingerprint=fp, variant=variant, delta_mAP=delta_mAP, **metadata
            )
        self.data_points.append(point)

    def extend(self, points: List[Union[LOVODataPoint, Tuple]]) -> None:
        """Add multiple data points."""
        for p in points:
            self.add(p)

    def to_history(self) -> List[Tuple[ArchitectureFingerprint, str, float]]:
        """Convert to the ``history`` format used by :meth:`PEFTPlanner.fit`."""
        return [p.to_tuple() for p in self.data_points]

    def save(self, path: Union[str, Path]) -> None:
        """Serialize to JSON.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [p.to_dict() for p in self.data_points], f, indent=2, ensure_ascii=False
            )
        LOGGER.info("[LOVO] Saved %d data points to %s", len(self.data_points), path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LOVODataCollector":
        """Deserialize from JSON."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls([LOVODataPoint.from_dict(d) for d in data])

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict."""
        if not self.data_points:
            return {"n_total": 0, "n_variants": 0}
        variants: Dict[str, int] = {}
        for p in self.data_points:
            variants[p.variant] = variants.get(p.variant, 0) + 1
        deltas = [p.delta_mAP for p in self.data_points]
        return {
            "n_total": len(self.data_points),
            "n_variants": len(variants),
            "variant_counts": variants,
            "delta_mAP_min": min(deltas),
            "delta_mAP_max": max(deltas),
            "delta_mAP_mean": sum(deltas) / len(deltas),
        }

    def filter_by_variant(self, variant: str) -> "LOVODataCollector":
        return LOVODataCollector(
            [p for p in self.data_points if p.variant.lower() == variant.lower()]
        )

    def filter_by_model(self, model_name: str) -> "LOVODataCollector":
        return LOVODataCollector(
            [p for p in self.data_points if p.model_name == model_name]
        )

    def __len__(self) -> int:
        return len(self.data_points)

    def __iter__(self):
        return iter(self.data_points)


@dataclass
class LOVOValidationResult:
    """Leave-One-Variant-Out cross-validation results.

    Attributes:
        lovo_predictions: List of (actual, predicted, variant) tuples.
        lovo_mse: Mean squared error.
        lovo_mae: Mean absolute error.
        lovo_r2: Coefficient of determination.
        coefficients: Final regression coefficients (fit on all data).
        n_samples: Number of unique data points used.
        n_variants: Number of unique variants.
        decision_threshold: Catastrophe threshold.
        metadata: Additional metadata.
    """

    lovo_predictions: List[Tuple[float, float, str]]
    lovo_mse: float
    lovo_mae: float
    lovo_r2: float
    coefficients: List[float]
    n_samples: int
    n_variants: int
    decision_threshold: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def lovo_rmse(self) -> float:
        return self.lovo_mse ** 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lovo_mse": self.lovo_mse,
            "lovo_mae": self.lovo_mae,
            "lovo_r2": self.lovo_r2,
            "lovo_rmse": self.lovo_rmse,
            "coefficients": self.coefficients,
            "n_samples": self.n_samples,
            "n_variants": self.n_variants,
            "decision_threshold": self.decision_threshold,
            "metadata": self.metadata,
        }

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        LOGGER.info("[LOVO] Validation result saved to %s", path)


class LOVOValidator:
    """Leave-One-Variant-Out cross-validation engine.

    Validates the PEFT regression model (Eq. 1) by iteratively leaving out
    each unique (fingerprint, variant) data point, fitting on the rest,
    and predicting the held-out value.  Produces R², MSE, MAE, and
    catastrophe-detection metrics.
    """

    def __init__(self, threshold: float = -0.05):
        self.threshold = threshold

    def cross_validate(self, data_points: List[LOVODataPoint]) -> LOVOValidationResult:
        """Run LOVO cross-validation.

        Args:
            data_points: List of data points.

        Returns:
            LOVOValidationResult: Validation metrics.

        Raises:
            ValueError: If fewer than 5 unique data points.
        """
        if len(data_points) < 5:
            raise ValueError(
                f"LOVO requires at least 5 data points, got {len(data_points)}"
            )

        # Deduplicate by (fingerprint, variant) key
        unique_points: List[LOVODataPoint] = []
        seen: set = set()
        for p in data_points:
            key = (
                round(p.fingerprint.phi_attn, 6),
                round(p.fingerprint.phi_text, 6),
                round(p.fingerprint.phi_dw, 6),
                round(p.fingerprint.phi_group, 6),
                round(p.fingerprint.phi_linear, 6),
                p.variant.lower(),
            )
            if key not in seen:
                seen.add(key)
                unique_points.append(p)

        if len(unique_points) < 5:
            raise ValueError(
                f"LOVO requires at least 5 unique data points, got {len(unique_points)}"
            )

        predictions: List[Tuple[float, float, str]] = []
        for left_out in unique_points:
            train_data = [p for p in unique_points if p is not left_out]
            train_history = [p.to_tuple() for p in train_data]

            planner = PEFTPlanner()
            planner.fit(train_history)
            predicted = planner.predict(left_out.fingerprint, left_out.variant)
            predictions.append((left_out.delta_mAP, predicted, left_out.variant))

        # Compute metrics
        try:
            import numpy as np
        except ImportError:
            LOGGER.warning("[LOVO] NumPy not available. Returning zero metrics.")
            return LOVOValidationResult(
                lovo_predictions=predictions,
                lovo_mse=0.0,
                lovo_mae=0.0,
                lovo_r2=0.0,
                coefficients=list(PEFTPlanner.DEFAULT_COEFFS),
                n_samples=len(unique_points),
                n_variants=len(set(p.variant.lower() for p in unique_points)),
                decision_threshold=self.threshold,
            )

        actual_arr = np.array([p[0] for p in predictions])
        pred_arr = np.array([p[1] for p in predictions])

        mse = float(np.mean((actual_arr - pred_arr) ** 2))
        mae = float(np.mean(np.abs(actual_arr - pred_arr)))
        ss_res = float(np.sum((actual_arr - pred_arr) ** 2))
        ss_tot = float(np.sum((actual_arr - np.mean(actual_arr)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        # Final fit on all unique points
        full_planner = PEFTPlanner()
        full_history = [p.to_tuple() for p in unique_points]
        full_planner.fit(full_history)

        return LOVOValidationResult(
            lovo_predictions=predictions,
            lovo_mse=mse,
            lovo_mae=mae,
            lovo_r2=r2,
            coefficients=full_planner._coeffs,
            n_samples=len(unique_points),
            n_variants=len(set(p.variant.lower() for p in unique_points)),
            decision_threshold=self.threshold,
        )

    def validate(self, collector: LOVODataCollector) -> LOVOValidationResult:
        """Convenience wrapper that validates a collector."""
        return self.cross_validate(collector.data_points)

    def evaluate_catastrophe_detection(
        self, collector: LOVODataCollector
    ) -> Dict[str, Any]:
        """Evaluate catastrophe detection metrics.

        Uses the LOVO-predicted values and the threshold to compute
        confusion matrix, precision, recall, F1, and accuracy.
        """
        result = self.cross_validate(collector.data_points)

        tp = fp = tn = fn = 0
        for actual, predicted, _ in result.lovo_predictions:
            actual_cat = actual < self.threshold
            pred_cat = predicted < self.threshold
            if actual_cat and pred_cat:
                tp += 1
            elif not actual_cat and pred_cat:
                fp += 1
            elif not actual_cat and not pred_cat:
                tn += 1
            else:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

        return {
            "threshold": self.threshold,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }

    def evaluate_decision_boundary(
        self, collector: LOVODataCollector
    ) -> Dict[str, Any]:
        """Evaluate ACCEPT/REFUSE decision boundary accuracy."""
        result = self.cross_validate(collector.data_points)

        correct_accept = 0
        correct_refuse = 0
        total = len(result.lovo_predictions)

        for actual, predicted, _ in result.lovo_predictions:
            actual_safe = actual >= self.threshold
            pred_safe = predicted >= self.threshold
            if actual_safe and pred_safe:
                correct_accept += 1
            elif not actual_safe and not pred_safe:
                correct_refuse += 1

        return {
            "total": total,
            "correct_accept": correct_accept,
            "correct_refuse": correct_refuse,
            "accuracy": (correct_accept + correct_refuse) / total if total > 0 else 0.0,
            "accept_accuracy": correct_accept / total if total > 0 else 0.0,
            "refuse_accuracy": correct_refuse / total if total > 0 else 0.0,
        }

    def full_report(self, collector: LOVODataCollector) -> Dict[str, Any]:
        """Generate a comprehensive validation report."""
        result = self.cross_validate(collector.data_points)
        cat_metrics = self.evaluate_catastrophe_detection(collector)
        decision_metrics = self.evaluate_decision_boundary(collector)

        return {
            "lovo": result.to_dict(),
            "catastrophe_detection": cat_metrics,
            "decision_boundary": decision_metrics,
            "summary": {
                "n_samples": result.n_samples,
                "n_variants": result.n_variants,
                "lovo_r2": result.lovo_r2,
                "lovo_rmse": result.lovo_rmse,
                "lovo_mae": result.lovo_mae,
                "catastrophe_recall": cat_metrics["recall"],
                "catastrophe_precision": cat_metrics["precision"],
                "catastrophe_f1": cat_metrics["f1"],
                "decision_accuracy": decision_metrics["accuracy"],
            },
        }


class PEFTPlanner:
    """Architecture-conditioned PEFT placement planner.

    Implements the regression model from Eq. 1:
        ΔmAP ≈ β₀ + β₁φ_attn + β₂φ_text + β₃φ_dw + β₄ξ_p

    where ξ_p is the variant-level coefficient (xi) from
    :class:`PEFTVariantProfile`. The planner uses this model together
    with hard policy rules to produce :class:`PlacementDecision` objects.

    Attributes:
        DEFAULT_COEFFS: Default regression coefficients (β₀, β₁, β₂, β₃, β₄).
            Calibrated against Table 1 of the YOLO-Master PEFT paper
            (R² ≈ 0.870 on 10 canonical data points).
        REFUSE_THRESHOLD: Threshold below which predicted ΔmAP triggers a REFUSE.
            Calibrated to match LOVO catastrophe recall 0.944 (paper Table 2).
    """

    # Calibrated against Table 1 (tab:core_wandb) canonical data points.
    # beta0 = 0.0656, beta1 = 0.0026, beta2 = 0.0, beta3 = 0.0054, beta4 = 1.0
    # Fitted via least squares on 12 non-catastrophic points (including ablations).
    DEFAULT_COEFFS: Tuple[float, float, float, float, float] = (
        0.0656, 0.0026, 0.0, 0.0054, 1.0
    )
    # Refuse threshold calibrated as a safety net for regression-predicted
    # catastrophic degradation.  The paper's catastrophic cases (RT-DETR
    # φ_attn≈0.85 and YOLO12s LoRA+DoRA no-rs Δ=-0.0550) are primarily
    # intercepted by hard policy rules above; the threshold catches edge
    # cases where the regression itself predicts strongly negative ΔmAP.
    # Paper Table 2 LOVO metrics: accuracy 86.7%, recall 0.944, F1=0.850.
    REFUSE_THRESHOLD: float = -0.05

    def __init__(
        self,
        calibration_data: Optional[Path] = None,
        audit_dir: Optional[Path] = None,
        lovo_collector: Optional["LOVODataCollector"] = None,
        lovo_validator: Optional["LOVOValidator"] = None,
    ):
        """Initialize the PEFT Planner.

        Args:
            calibration_data: Optional path to calibration data for fitting the
                regression model. Currently reserved for future use.
            audit_dir: Optional directory for persisting decision audit JSONs.
                Defaults to ``runs/planner_audit/``.
            lovo_collector: Optional LOVO data collector. When provided and
                containing at least 5 data points, the planner auto-fits
                coefficients before the first ``plan()`` call.
            lovo_validator: Optional LOVO validator for computing cross-validation
                metrics after auto-fitting.
        """
        self.calibration_data = calibration_data
        self.audit_dir = audit_dir
        self.lovo_collector = lovo_collector
        self.lovo_validator = lovo_validator
        self._coeffs = list(self.DEFAULT_COEFFS)
        self._history: List[
            Tuple[ArchitectureFingerprint, str, float]
        ] = []
        self._lovo_result: Optional[LOVOValidationResult] = None

    def _maybe_fit_from_lovo(self) -> None:
        """Auto-fit coefficients from LOVO collector if available and not yet fitted."""
        if self.lovo_collector is None or len(self.lovo_collector) < 5:
            return
        if self._history:
            return  # Already fitted from explicit history
        self.fit(self.lovo_collector.to_history())
        if self.lovo_validator is not None:
            try:
                self._lovo_result = self.lovo_validator.validate(self.lovo_collector)
                LOGGER.info(
                    "[Planner] LOVO R²=%.3f, RMSE=%.3f, n=%d",
                    self._lovo_result.lovo_r2,
                    self._lovo_result.lovo_rmse,
                    self._lovo_result.n_samples,
                )
            except Exception as exc:
                LOGGER.debug("[Planner] LOVO validation failed: %s", exc)

    def fit(
        self,
        history: List[Tuple[ArchitectureFingerprint, str, float]],
    ) -> None:
        """Fit regression coefficients from calibration history.

        Solves the least-squares problem for Eq. 1 using the normal equation.
        Falls back to default coefficients if the system is under-determined
        or singular.

        Args:
            history: List of (fingerprint, variant, delta_mAP) tuples.
        """
        self._history = history
        if len(history) < 5:
            LOGGER.warning(
                "[Planner] Insufficient calibration data (%d samples). "
                "Using default coefficients.",
                len(history),
            )
            return

        try:
            import numpy as np
        except ImportError:
            LOGGER.warning(
                "[Planner] NumPy not available. Using default coefficients."
            )
            return

        X = []
        y = []
        for fingerprint, variant, delta_map in history:
            profile = PEFTVariantProfile.from_variant(variant)
            xi = profile.xi
            X.append(
                [
                    1.0,
                    fingerprint.phi_attn,
                    fingerprint.phi_text,
                    fingerprint.phi_dw,
                    xi,
                ]
            )
            y.append(delta_map)

        X_arr = np.array(X, dtype=np.float64)
        y_arr = np.array(y, dtype=np.float64)

        # Use lstsq instead of solve to gracefully handle rank-deficient
        # matrices (e.g. when phi_text is constant across all data points).
        beta, residuals, rank, s = np.linalg.lstsq(X_arr, y_arr, rcond=None)
        self._coeffs = beta.tolist()
        LOGGER.info(
            "[Planner] Fitted regression coefficients: %s", self._coeffs
        )
        LOGGER.info(
            "[Planner] Fit rank: %d / %d (features).", rank, X_arr.shape[1]
        )

    def predict(
        self,
        fingerprint: ArchitectureFingerprint,
        variant: str,
    ) -> float:
        """Predict ΔmAP for a given architecture and variant.

        Args:
            fingerprint: The architecture fingerprint.
            variant: The PEFT variant name.

        Returns:
            float: Predicted ΔmAP.
        """
        profile = PEFTVariantProfile.from_variant(variant)
        xi = profile.xi
        b0, b1, b2, b3, b4 = self._coeffs
        delta = (
            b0
            + b1 * fingerprint.phi_attn
            + b2 * fingerprint.phi_text
            + b3 * fingerprint.phi_dw
            + b4 * xi
        )
        return float(delta)

    def plan(self, model: nn.Module, config: Any) -> PlacementDecision:
        """Generate a placement decision for the given model and config.

        Architecture-conditioned decision flow (regression-dominant):
            1. Compute architecture fingerprint and regression predictions for
               all compatible variants.
            2. Use regression prediction as the primary signal for ACCEPT /
               REFUSE / ADAPT.
            3. Apply hard safety guardrails only when the regression has not
               been trained on the relevant catastrophic data (i.e. when using
               DEFAULT_COEFFS) or when the requested variant is incompatible.
            4. LOVO data, if provided via ``lovo_collector``, is auto-fitted
               before prediction so the regression captures catastrophic patterns.

        Args:
            model: The model to analyze. If the model is an Ultralytics
                DetectionModel wrapper, the inner ``model.model`` is used.
            config: The LoRA configuration (LoRAConfig instance).

        Returns:
            PlacementDecision: The planner's decision.
        """
        from .config import LoRAConfig
        from .api import _effective_peft_variant

        if not isinstance(config, LoRAConfig):
            LOGGER.warning(
                "[Planner] Config is not LoRAConfig, skipping planner."
            )
            return PlacementDecision(status="ACCEPT", target_modules_hint=[])

        inner_model = getattr(model, "model", model)
        fingerprint = ArchitectureFingerprint.compute(inner_model)
        variant = _effective_peft_variant(config)
        rank = getattr(config, "r", 0)
        # Compute architecture-conditioned targets once for all decision paths.
        targets_hint = self.detect_targets(model, config)

        LOGGER.info(
            "[Planner] Architecture fingerprint: φ_attn=%.3f, "
            "φ_text=%.3f, φ_dw=%.3f, φ_group=%.3f, φ_linear=%.3f",
            fingerprint.phi_attn,
            fingerprint.phi_text,
            fingerprint.phi_dw,
            fingerprint.phi_group,
            fingerprint.phi_linear,
        )

        # Auto-fit from LOVO collector if available (regression-dominant
        # requires the model to be calibrated on catastrophic data when possible).
        self._maybe_fit_from_lovo()

        # === Phase 1: Regression-dominant evaluation of ALL variants ===
        ALL_VARIANTS = [
            "lora", "dora", "loha", "lokr", "ia3", "hra", "adalora", "oft", "boft"
        ]
        variant_scores: Dict[str, float] = {}
        for v in ALL_VARIANTS:
            profile = PEFTVariantProfile.from_variant(v)
            # Architecture compatibility: skip variants that don't support the
            # model's module types (e.g. LoRA on text-fusion architectures).
            if fingerprint.phi_attn > 0.05 and not profile.supports_attention:
                continue
            if fingerprint.phi_text > 0.05 and not profile.supports_text_fusion:
                continue
            variant_scores[v] = self.predict(fingerprint, v)

        if not variant_scores:
            decision = PlacementDecision(
                status="REFUSE",
                refusal_reason="No compatible PEFT variant found for this architecture.",
                predicted_delta=None,
                target_modules_hint=[],
                safety_overrides={"planner_refused": True},
            )
            self._save_audit(fingerprint, variant, rank, decision, targets_hint)
            return decision

        best_variant = max(variant_scores, key=variant_scores.get)
        best_delta = variant_scores[best_variant]

        requested_profile = PEFTVariantProfile.from_variant(variant)
        requested_compatible = (
            (fingerprint.phi_attn <= 0.05 or requested_profile.supports_attention)
            and (fingerprint.phi_text <= 0.05 or requested_profile.supports_text_fusion)
        )
        requested_delta = variant_scores.get(variant, self.predict(fingerprint, variant))

        LOGGER.info(
            "[Planner] Regression: requested %s Δ=%.4f, best %s Δ=%.4f",
            variant, requested_delta, best_variant, best_delta,
        )

        safety_overrides: Dict[str, Any] = {}
        recommended_variant: Optional[str] = None
        recommended_rank: Optional[int] = None

        # === Phase 2: Hard safety guardrails (only when regression hasn't
        # seen the catastrophic data, i.e. using DEFAULT_COEFFS) ===
        using_defaults = (self._coeffs == list(self.DEFAULT_COEFFS))

        # Guardrail A: DoRA on attention-rich architectures → downgrade to LoRA.
        # Paper Fig. 4: YOLO12n DoRA has 6/7 catastrophe rate. When LOVO-fitted,
        # regression itself catches this; the guardrail is a safety net for defaults.
        if variant.lower() == "dora" and fingerprint.phi_attn > 0.3:
            recommended_variant = "lora"
            safety_overrides["use_dora"] = False
            safety_overrides["variant_adapted"] = True
            LOGGER.info(
                "[Planner] Safety guardrail: DoRA on attention-rich (φ_attn=%.3f) "
                "→ downgrade to LoRA", fingerprint.phi_attn
            )

        # Guardrail B: RT-DETR-like (φ_attn > 0.7) + LoRA-family with default coeffs.
        # Paper Fig. 4: RT-DETR-l has 7/7 catastrophe rate for LoRA-family.
        if (
            fingerprint.phi_attn > 0.7
            and variant.lower() in ("lora", "dora", "loha", "lokr")
            and using_defaults
        ):
            decision = PlacementDecision(
                status="REFUSE",
                refusal_reason=(
                    f"RT-DETR-like architecture (φ_attn={fingerprint.phi_attn:.2f}): "
                    "LoRA-family variants destabilize attention-heavy backbones. "
                    "Use Full-SFT instead."
                ),
                predicted_delta=requested_delta,
                target_modules_hint=[],
                safety_overrides={"planner_refused": True},
            )
            self._save_audit(fingerprint, variant, rank, decision, targets_hint)
            return decision

        # === Phase 3: Regression-dominant decision ===

        # If the requested variant is architecturally incompatible.
        if not requested_compatible:
            if best_delta >= self.REFUSE_THRESHOLD:
                decision = PlacementDecision(
                    status="ADAPT",
                    recommended_variant=best_variant,
                    predicted_delta=best_delta,
                    target_modules_hint=targets_hint,
                    safety_overrides={"variant_adapted": True},
                )
                self._save_audit(fingerprint, variant, rank, decision, targets_hint)
                return decision
            decision = PlacementDecision(
                status="REFUSE",
                refusal_reason=(
                    f"Requested variant {variant} is incompatible with this architecture "
                    f"and no safe alternative exists (best {best_variant} Δ={best_delta:.4f})."
                ),
                predicted_delta=requested_delta,
                target_modules_hint=[],
                safety_overrides={"planner_refused": True},
            )
            self._save_audit(fingerprint, variant, rank, decision, targets_hint)
            return decision

        # If a safety guardrail already triggered a variant change.
        if recommended_variant is not None:
            new_delta = variant_scores.get(
                recommended_variant, self.predict(fingerprint, recommended_variant)
            )
            decision = PlacementDecision(
                status="ADAPT",
                recommended_variant=recommended_variant,
                predicted_delta=new_delta,
                target_modules_hint=targets_hint,
                safety_overrides=safety_overrides,
            )
            self._save_audit(fingerprint, variant, rank, decision, targets_hint)
            return decision

        # If the requested variant predicts catastrophic degradation.
        if requested_delta < self.REFUSE_THRESHOLD:
            if best_variant != variant and best_delta >= self.REFUSE_THRESHOLD:
                decision = PlacementDecision(
                    status="ADAPT",
                    recommended_variant=best_variant,
                    predicted_delta=best_delta,
                    target_modules_hint=targets_hint,
                    safety_overrides={"variant_adapted": True},
                )
                self._save_audit(fingerprint, variant, rank, decision, targets_hint)
                return decision
            decision = PlacementDecision(
                status="REFUSE",
                refusal_reason=(
                    f"Predicted ΔmAP ({requested_delta:.4f}) below threshold "
                    f"({self.REFUSE_THRESHOLD}) for {variant}. No safe alternative."
                ),
                predicted_delta=requested_delta,
                target_modules_hint=[],
                safety_overrides={"planner_refused": True},
            )
            self._save_audit(fingerprint, variant, rank, decision, targets_hint)
            return decision

        # Attention-rich architectures: cap rank and enable safe attention.
        # Paper Table 1: YOLO12s (φ_attn≈0.45) has 6/7 catastrophe rate;
        # rank capping mitigates risk (LoRA r=8: +0.0626, r=16: +0.0645,
        # r=32: +0.0701). Safe attention inclusion prevents destabilisation.
        if fingerprint.phi_attn > 0.3:
            if rank > 0 and rank > 8:
                recommended_rank = 8
                safety_overrides["r"] = 8
                LOGGER.info(
                    "[Planner] Capping rank to 8 for attention-rich architecture"
                )
            if not getattr(config, "include_attention", False):
                safety_overrides["include_attention"] = True
                LOGGER.info(
                    "[Planner] Enabling safe attention for attention-rich architecture"
                )

        # YOLO11s-like (no attention): disable attention targets.
        if fingerprint.phi_attn < 0.05:
            if getattr(config, "include_attention", False):
                safety_overrides["include_attention"] = False
                LOGGER.info(
                    "[Planner] No attention detected (φ_attn=%.3f), "
                    "disabling attention targets",
                    fingerprint.phi_attn,
                )
            else:
                LOGGER.info(
                    "[Planner] No attention detected (φ_attn=%.3f), "
                    "attention already disabled",
                    fingerprint.phi_attn,
                )

        # Only emit ADAPT if there is a material change (variant, rank, or
        # config override that differs from the current value).
        material_adapt = bool(recommended_variant or recommended_rank)
        if not material_adapt and safety_overrides:
            for k, v in safety_overrides.items():
                if getattr(config, k, None) != v:
                    material_adapt = True
                    break

        if material_adapt:
            decision = PlacementDecision(
                status="ADAPT",
                recommended_variant=recommended_variant,
                recommended_rank=recommended_rank,
                predicted_delta=requested_delta,
                target_modules_hint=targets_hint,
                safety_overrides=safety_overrides,
            )
        else:
            decision = PlacementDecision(
                status="ACCEPT",
                predicted_delta=requested_delta,
                target_modules_hint=targets_hint,
            )

        self._save_audit(fingerprint, variant, rank, decision, targets_hint)
        return decision

    def _save_audit(
        self,
        fingerprint: ArchitectureFingerprint,
        variant: str,
        requested_rank: int,
        decision: PlacementDecision,
        targets_hint: List[str],
    ) -> None:
        """Persist a decision audit record (best-effort, never raises)."""
        try:
            audit = DecisionAudit(
                timestamp=datetime.now().isoformat(),
                model_name="unknown",
                fingerprint={
                    "phi_attn": fingerprint.phi_attn,
                    "phi_text": fingerprint.phi_text,
                    "phi_dw": fingerprint.phi_dw,
                    "phi_group": fingerprint.phi_group,
                    "phi_linear": fingerprint.phi_linear,
                },
                variant=variant,
                requested_rank=requested_rank,
                decision_status=decision.status,
                recommended_variant=decision.recommended_variant,
                recommended_rank=decision.recommended_rank,
                predicted_delta=decision.predicted_delta,
                refusal_reason=decision.refusal_reason,
                safety_overrides=dict(decision.safety_overrides),
                target_modules_count=len(targets_hint),
            )
            audit.save(self.audit_dir)
        except Exception as exc:
            LOGGER.debug("[Planner] Audit save failed (non-critical): %s", exc)

    def plan_variant(
        self,
        model: nn.Module,
        variant: str,
        rank: int,
    ) -> PlacementDecision:
        """Generate a decision for a specific variant and rank.

        Convenience wrapper around :meth:`plan` that constructs a minimal
        LoRAConfig from the variant and rank.

        Args:
            model: The model to analyze.
            variant: The PEFT variant name.
            rank: The proposed LoRA rank.

        Returns:
            PlacementDecision: The planner's decision.
        """
        from .config import LoRAConfig

        config = LoRAConfig(peft_type=variant, r=rank)
        return self.plan(model, config)

    def detect_targets(
        self,
        model: nn.Module,
        config: Optional[Any] = None,
    ) -> List[str]:
        """Architecture-conditioned target module detection.

        Detects which modules should be targeted based on the architecture
        fingerprint. This is intended to replace or augment the generic
        :meth:`LoRAConfigBuilder.auto_detect_targets` with
        architecture-aware selection.

        Rules:
          - YOLO11s-like (φ_attn < 0.05): conv only, no attention.
          - YOLO12s-like (0.05 ≤ φ_attn < 0.7): conv + safe attention,
            excluding area-attention risky layers (qkv, proj, pe) and
            ABlock-internal MLP convs on the residual stream.
          - RT-DETR-like (φ_attn ≥ 0.7): no targets (refuse).
          - YOLO-World / text-fusion: text-fusion modules are always included
            when φ_text > 0.05.

        Args:
            model: The model to analyze. If the model is an Ultralytics
                DetectionModel wrapper, the inner ``model.model`` is used.
            config: Optional LoRA configuration for additional constraints
                (``only_backbone``, ``include_head``, ``exclude_modules``).

        Returns:
            List[str]: Sorted list of target module names.
        """
        inner_model = ArchitectureFingerprint._unwrap_model(
            getattr(model, "model", model)
        )
        fingerprint = ArchitectureFingerprint.compute(inner_model)
        targets: List[str] = []
        include_text = fingerprint.phi_text > 0.05

        for name, module in inner_model.named_modules():
            if not name:
                continue

            is_conv = isinstance(module, nn.Conv2d)
            is_linear = isinstance(module, nn.Linear)
            if not (is_conv or is_linear):
                continue

            lname = name.lower()
            is_text_fusion = any(
                k in lname for k in ("text", "clip", "lang", "fusion")
            )

            # Text-fusion modules are always included when detected.
            if is_text_fusion and include_text:
                targets.append(name)
                continue

            # YOLO11s-like (no attention): conv only, no attention.
            if fingerprint.phi_attn < 0.05:
                if is_conv and "attn" not in lname:
                    targets.append(name)
                continue

            # YOLO12s-like (moderate attention): conv + safe attention.
            if 0.05 <= fingerprint.phi_attn < 0.7:
                # Exclude area-attention risky conv layers (qkv, proj, pe).
                if is_conv and any(
                    p in lname for p in (".attn.qkv", ".attn.proj", ".attn.pe")
                ):
                    continue
                # Exclude ABlock-internal MLP convs on the residual stream.
                if is_conv and ".mlp." in lname and any(
                    b in lname for b in ("ablock", "a2c2f", "aattn")
                ):
                    continue
                # Exclude MSDeformAttn geometry-sensitive linear layers.
                if is_linear and any(
                    p in lname
                    for p in ("sampling_offsets", "attention_weights")
                ):
                    continue
                targets.append(name)
                continue

            # RT-DETR-like (high attention): refuse targets.
            if fingerprint.phi_attn >= 0.7:
                continue

        # Apply additional config-level filters if provided.
        if config is not None:
            only_backbone = getattr(config, "only_backbone", False)
            include_head = getattr(config, "include_head", False)
            exclude_modules = getattr(config, "exclude_modules", None) or []

            filtered = []
            for name in targets:
                lname = name.lower()
                if only_backbone and any(
                    p in lname
                    for p in (
                        "head",
                        "detect",
                        "box",
                        "cls",
                        "pred",
                        "fpn",
                        "pan",
                        "seg",
                        "pose",
                    )
                ):
                    continue
                if not include_head and any(
                    p in lname for p in ("head", "detect", "score_head", "bbox_head")
                ):
                    continue
                if any(ex in name for ex in exclude_modules):
                    continue
                filtered.append(name)
            targets = filtered

        return sorted(targets)


def is_planner_enabled(config: Any) -> bool:
    """Check whether the PEFT Planner is enabled on a configuration object.

    Args:
        config: A configuration object (e.g., LoRAConfig or trainer args).

    Returns:
        bool: True if the planner is enabled, False otherwise.
    """
    return bool(
        getattr(config, "lora_planner_enabled", False)
        or getattr(config, "planner_enabled", False)
    )


__all__ = [
    "ArchitectureFingerprint",
    "PEFTVariantProfile",
    "PlacementDecision",
    "DecisionAudit",
    "LOVODataPoint",
    "LOVODataCollector",
    "LOVOValidationResult",
    "LOVOValidator",
    "PEFTPlanner",
    "RefusalError",
    "is_planner_enabled",
]
