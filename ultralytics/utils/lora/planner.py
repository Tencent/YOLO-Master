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
from ultralytics.utils.errors import PEFTRefusalError
import weakref

# Weak-key cache: automatically invalidated when the model object is garbage-collected.
# This prevents stale entries from memory-address reuse across test runs or
# model re-creation, and avoids the need for explicit cache invalidation.
_fingerprint_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


class RefusalError(PEFTRefusalError):
    """Raised when the PEFT Planner refuses a configuration.

    This is a valid planning decision, not a failure. The caller should
    catch this and fall back to full fine-tuning (Full-SFT).

    Now inherits from PEFTRefusalError (ultralytics.utils.errors) to unify
    the exception hierarchy. Existing code catching RefusalError continues
    to work; new code can catch PEFTRefusalError for the same effect.
    """
    pass


@dataclass
class ArchitectureFingerprint:
    """Compact 10-dimensional architecture fingerprint.

    The original 5 dimensions (phi_attn, phi_text, phi_dw, phi_group,
    phi_linear) capture module-type ratios but cannot distinguish models
    of the same family at different scales (e.g. YOLOv8n vs YOLOv8x).

    The 5 extended dimensions add scale and structural information:
        phi_depth:  Normalised model depth (top-level block count / 30).
        phi_width:  Log-scale average channel width.
        phi_head:   Detection-head complexity (head params / total params).
        phi_residual: Residual connection density (residual modules / total).
        phi_norm:   Normalisation layer distribution (LN ratio among all norm layers).

    Attributes:
        phi_attn: Attention module ratio (attention modules / total conv+linear).
        phi_text: Text-fusion module ratio (text-fusion modules / total conv+linear).
        phi_dw: Depthwise convolution ratio (depthwise conv / total conv).
        phi_group: Grouped convolution ratio (grouped conv / total conv).
        phi_linear: Linear layer ratio (linear modules / total conv+linear).
        phi_depth: Normalised model depth (top-level blocks / 30, clamped to [0, 1]).
        phi_width: Log2-scale average channel width (log2(avg_channels) / 10).
        phi_head: Detection-head parameter ratio (head params / total params).
        phi_residual: Residual connection density (modules with add_hook / total).
        phi_norm: LayerNorm ratio among all norm layers (LN / (BN+LN+GN)).
    """
    phi_attn: float = 0.0
    phi_text: float = 0.0
    phi_dw: float = 0.0
    phi_group: float = 0.0
    phi_linear: float = 0.0
    # Extended dimensions (v2) — enable scale-aware regression predictions.
    phi_depth: float = 0.0
    phi_width: float = 0.0
    phi_head: float = 0.0
    phi_residual: float = 0.0
    phi_norm: float = 0.0

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
        fingerprint values.  This guarantees that the 10-dimensional vector is a
        true reflection of the model's weight topology, which is required for
        the regression model (Eq. 1) to generalise to novel architectures.

        Architecture-family detection (:meth:`_detect_architecture_family`) is
        still available as a standalone utility for downstream policy rules, but
        it does not alter the fingerprint values.

        Args:
            model: The PyTorch model to analyze.

        Returns:
            ArchitectureFingerprint: The computed 10-dimensional fingerprint.
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
          - Extended dimensions (v2): phi_depth, phi_width, phi_head,
            phi_residual, phi_norm provide scale-aware features that allow
            the regression model to distinguish e.g. YOLOv8n from YOLOv8x.
        """
        import math

        total_conv = 0
        total_linear = 0
        attn_count = 0
        text_count = 0
        dw_count = 0
        group_count = 0
        linear_count = 0

        # Extended dimension accumulators
        total_params = 0
        head_params = 0
        conv_channel_sums = []  # for phi_width
        ln_count = 0
        bn_count = 0
        gn_count = 0
        residual_count = 0
        total_modules_all = 0  # all nn.Module descendants

        # Detect head-related module class names
        _HEAD_CLASS_NAMES = frozenset({
            "Detect", "RTDETRDecoder", "Segment", "Segment26", "Pose", "Pose26", "OBB", "OBB26",
            "SemanticSegment", "WorldDetect", "YOLOEDetect", "YOLOESegment", "YOLOESegment26", "v10Detect",
        })

        # Residual detection: modules that have forward with add/residual
        # Heuristic: count modules whose class name contains common residual patterns
        _RESIDUAL_KEYWORDS = frozenset({
            "C2f", "C3k2", "C3", "C2PSA", "A2C2f", "Bottleneck",
            "ResidualBlock", "SPPF", "SPP", "C2fCIB",
            "RepC3", "C3k",
        })

        for name, module in model.named_modules():
            total_modules_all += 1
            cls_name = module.__class__.__name__

            if isinstance(module, nn.Conv2d):
                total_conv += 1
                conv_channel_sums.append(module.in_channels + module.out_channels)
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

            # Iconic attention modules only (not every child submodule).
            # Covers both YOLO12 (AAttn) and RT-DETR (MultiheadAttention,
            # MSDeformAttn, AIFI, RTDETRDecoder, DeformableTransformerDecoderLayer).
            if cls_name in (
                "AAttn", "MultiheadAttention", "MSDEFORMAttention",
                "MSDeformAttn", "RTDETRDecoder", "DeformableAttention",
                "AIFI", "DeformableTransformerDecoderLayer",
            ):
                attn_count += 1

            # Text-fusion detection — refined: use module type check first,
            # fall back to keyword matching only for well-known patterns.
            lname = name.lower()
            if cls_name in ("TextFusion", "WorldEmbed", "TextProj", "TextEncoder"):
                text_count += 1
            elif any(k in lname for k in ("text_encoder", "clip", "text_fusion", "world_embed", "text_proj")):
                text_count += 1
            # NOTE: bare "fusion" keyword removed to avoid false positives
            # (e.g. fusion_layer, feature_fusion in non-text architectures).

            # Normalisation layers
            if isinstance(module, nn.LayerNorm):
                ln_count += 1
            elif isinstance(module, nn.BatchNorm2d):
                bn_count += 1
            elif isinstance(module, nn.GroupNorm):
                gn_count += 1

            # Detection head params
            if cls_name in _HEAD_CLASS_NAMES:
                head_params += sum(p.numel() for p in module.parameters(recurse=True))

            # Residual connection density
            if any(kw in cls_name for kw in _RESIDUAL_KEYWORDS):
                residual_count += 1

        total_params = sum(p.numel() for p in model.parameters())
        total_modules = total_conv + total_linear
        if total_modules == 0:
            LOGGER.warning(
                "[Planner] Model has no Conv2d or Linear modules. "
                "Returning zero fingerprint."
            )
            return cls()
        if total_conv == 0:
            total_conv = 1

        # --- Extended dimension computations ---
        # phi_depth: normalise top-level block count to [0, 1].
        # YOLO models typically have 10-25 top-level blocks; normalise by 30.
        try:
            top_level_count = len(model) if hasattr(model, '__len__') else 0
        except TypeError:
            top_level_count = 0
        phi_depth = min(top_level_count / 30.0, 1.0) if top_level_count > 0 else 0.0

        # phi_width: log2-scale average channel width, normalised by 10.
        # YOLOv8n avg ~64ch (log2=6), YOLOv8x avg ~320ch (log2~8.3), RT-DETR ~256 (log2=8).
        if conv_channel_sums:
            avg_ch = sum(conv_channel_sums) / len(conv_channel_sums)
            phi_width = min(math.log2(max(avg_ch, 1.0)) / 10.0, 1.0)
        else:
            phi_width = 0.0

        # phi_head: head parameter ratio.
        phi_head = head_params / total_params if total_params > 0 else 0.0

        # phi_residual: residual module density.
        phi_residual = residual_count / total_modules_all if total_modules_all > 0 else 0.0

        # phi_norm: LayerNorm ratio among all norm layers (LN preferred for attention archs).
        total_norm = ln_count + bn_count + gn_count
        phi_norm = ln_count / total_norm if total_norm > 0 else 0.0

        return cls(
            phi_attn=attn_count / total_modules,
            phi_text=text_count / total_modules,
            phi_dw=dw_count / total_conv,
            phi_group=group_count / total_conv,
            phi_linear=linear_count / total_modules,
            phi_depth=phi_depth,
            phi_width=phi_width,
            phi_head=phi_head,
            phi_residual=phi_residual,
            phi_norm=phi_norm,
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
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.safety_overrides is None:
            self.safety_overrides = {}
        if self.metadata is None:
            self.metadata = {}
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
            "metadata": dict(self.metadata),
            "target_modules_hint": list(self.target_modules_hint or []),
            "target_modules_hint_count": len(self.target_modules_hint or []),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlacementDecision":
        """Restore a planner decision from a DDP-safe plain dictionary."""
        return cls(
            status=payload.get("status", "ACCEPT"),
            recommended_variant=payload.get("recommended_variant"),
            recommended_rank=payload.get("recommended_rank"),
            predicted_delta=payload.get("predicted_delta"),
            target_modules_hint=list(payload.get("target_modules_hint") or []),
            refusal_reason=payload.get("refusal_reason"),
            safety_overrides=dict(payload.get("safety_overrides") or {}),
            metadata=dict(payload.get("metadata") or {}),
        )


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
    metadata: Dict[str, Any] = field(default_factory=dict)
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
            "metadata": self.metadata,
            "target_modules_count": self.target_modules_count,
        }

    # Maximum number of audit files to keep in the directory.
    # Older files are automatically cleaned up when this limit is exceeded.
    MAX_AUDIT_FILES: int = 100

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

        # ── Audit file rotation: keep at most MAX_AUDIT_FILES ──
        try:
            existing = sorted(
                audit_dir.glob("planner_audit_*.json"),
                key=lambda p: p.name,
            )
            excess = len(existing) - self.MAX_AUDIT_FILES
            if excess > 0:
                for old_file in existing[:excess]:
                    try:
                        old_file.unlink()
                    except OSError:
                        pass
                LOGGER.debug(
                    "[Planner] Audit rotation: removed %d old file(s)", excess
                )
        except OSError:
            pass  # Non-fatal: rotation failure should not crash planning

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
                # Extended dimensions (v2)
                "phi_depth": self.fingerprint.phi_depth,
                "phi_width": self.fingerprint.phi_width,
                "phi_head": self.fingerprint.phi_head,
                "phi_residual": self.fingerprint.phi_residual,
                "phi_norm": self.fingerprint.phi_norm,
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
                # Extended dimensions — default to 0.0 for backward compat
                phi_depth=fp.get("phi_depth", 0.0),
                phi_width=fp.get("phi_width", 0.0),
                phi_head=fp.get("phi_head", 0.0),
                phi_residual=fp.get("phi_residual", 0.0),
                phi_norm=fp.get("phi_norm", 0.0),
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

        # Deduplicate by (fingerprint, variant, delta_mAP) key.
        # Including delta_mAP is essential: the same (architecture, variant) can
        # produce different outcomes under different training configs (rank, lr,
        # warmup).  These are distinct data points that must be preserved.
        unique_points: List[LOVODataPoint] = []
        seen: set = set()
        for p in data_points:
            key = (
                round(p.fingerprint.phi_attn, 6),
                round(p.fingerprint.phi_text, 6),
                round(p.fingerprint.phi_dw, 6),
                round(p.fingerprint.phi_group, 6),
                round(p.fingerprint.phi_linear, 6),
                # Extended dimensions
                round(p.fingerprint.phi_depth, 6),
                round(p.fingerprint.phi_width, 6),
                round(p.fingerprint.phi_head, 6),
                round(p.fingerprint.phi_residual, 6),
                round(p.fingerprint.phi_norm, 6),
                p.variant.lower(),
                round(p.delta_mAP, 6),
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
            # Use default rank=8 for LOVO prediction (rank is not part of LOVO data)
            predicted = planner.predict(left_out.fingerprint, left_out.variant, 8)
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
    # Original 5-coeff: beta0=0.0656, beta1=0.0026, beta2=0.0, beta3=0.0054, beta4=1.0
    # Extended 12-coeff (v3): adds phi_depth, phi_width, phi_head, phi_residual,
    #   phi_norm, log(r), and phi_attn² as regression features.
    # The log(r) coefficient (beta10) captures the rank's marginal effect on ΔmAP.
    # Paper Table 1: YOLO12s r=8→+0.0626, r=16→+0.0645, r=32→+0.0701 → slope≈0.0036/log2(r)
    # The phi_attn² coefficient (beta11) captures the non-linear catastrophic cliff:
    # RT-DETR (phi_attn≈0.85) has Δ=-0.600 while YOLO12s (phi_attn≈0.45) has Δ=+0.0645.
    # A quadratic term allows the regression to model this sharp transition.
    # New extended dims default to 0.0 coefficient (no impact until LOVO-fitted with
    # multi-scale data), preserving backward compatibility with the original model.
    DEFAULT_COEFFS: Tuple[float, ...] = (
        0.0656,   # beta0  – intercept
        0.0026,   # beta1  – phi_attn
        0.0,      # beta2  – phi_text
        0.0054,   # beta3  – phi_dw
        1.0,      # beta4  – xi (variant)
        0.0,      # beta5  – phi_depth (activated by LOVO fit)
        0.0,      # beta6  – phi_width
        0.0,      # beta7  – phi_head
        0.0,      # beta8  – phi_residual
        0.0,      # beta9  – phi_norm
        0.0,      # beta10 – log(r) rank effect
        0.0,      # beta11 – phi_attn² (non-linear catastrophe term)
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
        lovo_persist_path: Optional[Path] = None,
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
            lovo_persist_path: Optional path for auto-persisting LOVO data when
                :meth:`record_training_result` is called. Enables the online
                learning closed loop — each training result is appended to the
                collector and persisted to disk so the regression improves over
                successive runs.
        """
        self.calibration_data = calibration_data
        self.audit_dir = audit_dir
        self.lovo_collector = lovo_collector
        self.lovo_validator = lovo_validator
        self.lovo_persist_path = lovo_persist_path
        self._coeffs = list(self.DEFAULT_COEFFS)
        self._history: List[
            Tuple[ArchitectureFingerprint, str, float]
        ] = []
        self._lovo_result: Optional[LOVOValidationResult] = None
        self._fit_n_samples = 0
        self._fit_effective_rank = 0
        self._fit_n_features = len(self.DEFAULT_COEFFS)
        self._needs_refit = False  # Flag: re-fit on next plan() after new data

    def _maybe_fit_from_lovo(self) -> None:
        """Auto-fit coefficients from LOVO collector if available and not yet fitted."""
        if self.lovo_collector is None or len(self.lovo_collector) < 5:
            return
        if self._history and not self._needs_refit:
            return  # Already fitted and no new data since last fit
        self.fit(self.lovo_collector.to_history())
        self._needs_refit = False
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

    def record_training_result(
        self,
        model: nn.Module,
        variant: str,
        rank: int,
        delta_mAP: float,
        model_name: str = "",
        dataset: str = "",
        epochs: int = 0,
        notes: str = "",
    ) -> None:
        """Record a training result into the LOVO collector (online learning).

        After each training run, call this with the measured ΔmAP to feed
        the regression model. On the next ``plan()`` call, the planner
        automatically re-fits coefficients, incorporating the new data point.

        If ``lovo_persist_path`` was set in the constructor, the collector
        is also saved to disk so the data persists across runs.

        Args:
            model: The model that was trained (used to compute fingerprint).
            variant: PEFT variant used (e.g. 'lora', 'dora').
            rank: LoRA rank used.
            delta_mAP: Measured ΔmAP (new_mAP - base_mAP).
            model_name: Optional model name for metadata.
            dataset: Optional dataset name.
            epochs: Training epochs.
            notes: Free-form notes.
        """
        if self.lovo_collector is None:
            self.lovo_collector = LOVODataCollector()
        inner_model = getattr(model, "model", model)
        fingerprint = ArchitectureFingerprint.compute(inner_model)
        self.lovo_collector.add(LOVODataPoint(
            fingerprint=fingerprint,
            variant=variant,
            delta_mAP=delta_mAP,
            model_name=model_name,
            dataset=dataset,
            epochs=epochs,
            notes=notes or f"rank={rank}",
        ))
        self._needs_refit = True
        LOGGER.info(
            "[Planner] Recorded training result: variant=%s, rank=%d, "
            "ΔmAP=%.4f (collector now has %d points)",
            variant, rank, delta_mAP, len(self.lovo_collector),
        )
        if self.lovo_persist_path is not None:
            try:
                self.lovo_collector.save(self.lovo_persist_path)
            except Exception as exc:
                LOGGER.debug("[Planner] LOVO persist failed: %s", exc)

    def fit(
        self,
        history: List[Tuple[ArchitectureFingerprint, str, float]],
        ranks: Optional[List[int]] = None,
    ) -> None:
        """Fit regression coefficients from calibration history.

        Solves the least-squares problem for Eq. 1 using ridge regression
        (L2 regularisation) to handle rank-deficient design matrices gracefully.
        Falls back to default coefficients if the system is under-determined
        or singular.

        The regression model uses 12 features:
            [1, phi_attn, phi_text, phi_dw, xi,
             phi_depth, phi_width, phi_head, phi_residual, phi_norm, log(r),
             phi_attn²]

        The phi_attn² term captures the non-linear catastrophic cliff observed
        in RT-DETR (phi_attn≈0.85, Δ=-0.600) vs YOLO12s (phi_attn≈0.45, Δ=+0.065).
        A quadratic allows the regression to model this sharp transition without
        requiring hard guardrails for the regression path.

        Ridge regression (L2 penalty λ=0.01) addresses the rank-deficiency
        problem (12 features, ~10-12 samples, matrix rank 6/12).  The small
        λ stabilises the solution without significantly biasing identifiable
        coefficients.

        When ``ranks`` is not provided (backward-compatible path), log(r)
        defaults to log(8) ≈ 3.0, which is a neutral mid-range value.

        Args:
            history: List of (fingerprint, variant, delta_mAP) tuples.
            ranks: Optional list of LoRA ranks corresponding to each history
                entry. If None, a default rank of 8 is assumed for all entries.
        """
        import math

        self._history = history
        self._fit_n_samples = len(history)
        self._fit_effective_rank = 0
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
        for i, (fingerprint, variant, delta_map) in enumerate(history):
            profile = PEFTVariantProfile.from_variant(variant)
            xi = profile.xi
            # Determine rank for this data point
            if ranks is not None and i < len(ranks):
                r = max(ranks[i], 1)
            else:
                r = 8  # default rank assumption
            log_r = math.log2(r)
            phi_attn_sq = fingerprint.phi_attn ** 2

            X.append(
                [
                    1.0,
                    fingerprint.phi_attn,
                    fingerprint.phi_text,
                    fingerprint.phi_dw,
                    xi,
                    fingerprint.phi_depth,
                    fingerprint.phi_width,
                    fingerprint.phi_head,
                    fingerprint.phi_residual,
                    fingerprint.phi_norm,
                    log_r,
                    phi_attn_sq,
                ]
            )
            y.append(delta_map)

        X_arr = np.array(X, dtype=np.float64)
        y_arr = np.array(y, dtype=np.float64)

        # Ridge regression: (X^T X + λI)⁻¹ X^T y
        # λ=1e-6 provides minimal regularisation that resolves the singularity
        # without significantly biasing identifiable coefficients.
        # The previous λ=0.01 was too aggressive — it shrunk the xi coefficient
        # (which ranges from -0.02 to 0.0152) by 87%, breaking prediction accuracy.
        # λ=1e-6 is sufficient because the singularity comes from zero-variance
        # columns (extended dims are all-zero when test data only has 5-dim
        # fingerprints), not from multicollinearity among identifiable features.
        n_features = X_arr.shape[1]
        ridge_lambda = 1e-6
        XtX = X_arr.T @ X_arr
        XtX_reg = XtX + ridge_lambda * np.eye(n_features)
        Xty = X_arr.T @ y_arr
        try:
            beta = np.linalg.solve(XtX_reg, Xty)
        except np.linalg.LinAlgError:
            # Fallback to lstsq if ridge matrix is still singular
            beta, residuals, rank, s = np.linalg.lstsq(X_arr, y_arr, rcond=None)

        self._coeffs = beta.tolist()

        # Pad coefficients list if old 11-dim data was loaded
        if len(self._coeffs) < 12:
            self._coeffs = list(self._coeffs) + [0.0] * (12 - len(self._coeffs))

        # Compute matrix rank for diagnostic logging
        mat_rank = np.linalg.matrix_rank(X_arr)
        self._fit_effective_rank = int(mat_rank)
        self._fit_n_features = int(n_features)
        LOGGER.info(
            "[Planner] Fitted regression coefficients (12-dim, ridge λ=%.3f): %s",
            ridge_lambda, self._coeffs,
        )
        LOGGER.info(
            "[Planner] Fit matrix rank: %d / %d (features), %d samples.",
            mat_rank, n_features, len(history),
        )

    def predict(
        self,
        fingerprint: ArchitectureFingerprint,
        variant: str,
        rank: int = 8,
    ) -> float:
        """Predict ΔmAP for a given architecture and variant.

        Uses the 12-feature regression model:
            ΔmAP = β₀ + β₁φ_attn + β₂φ_text + β₃φ_dw + β₄ξ_p
                   + β₅φ_depth + β₆φ_width + β₇φ_head + β₈φ_residual
                   + β₉φ_norm + β₁₀log(r) + β₁₁φ_attn²

        The phi_attn² term captures the non-linear catastrophic cliff:
        when beta11 is negative, high-phi_attn architectures (RT-DETR)
        receive a sharply lower prediction, modelling the observed
        7/7 catastrophe rate without requiring hard guardrails.

        Args:
            fingerprint: The architecture fingerprint.
            variant: The PEFT variant name.
            rank: LoRA rank (default 8). Used in the log(r) feature.

        Returns:
            float: Predicted ΔmAP.
        """
        import math

        profile = PEFTVariantProfile.from_variant(variant)
        xi = profile.xi
        coeffs = self._coeffs
        log_r = math.log2(max(rank, 1))
        phi_attn_sq = fingerprint.phi_attn ** 2

        # Graceful fallback: if coefficients are still <12-dim (old data),
        # pad with zeros for the new features.
        if len(coeffs) < 12:
            coeffs = list(coeffs) + [0.0] * (12 - len(coeffs))

        b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11 = coeffs
        delta = (
            b0
            + b1 * fingerprint.phi_attn
            + b2 * fingerprint.phi_text
            + b3 * fingerprint.phi_dw
            + b4 * xi
            + b5 * fingerprint.phi_depth
            + b6 * fingerprint.phi_width
            + b7 * fingerprint.phi_head
            + b8 * fingerprint.phi_residual
            + b9 * fingerprint.phi_norm
            + b10 * log_r
            + b11 * phi_attn_sq
        )
        return float(delta)

    def predict_with_uncertainty(
        self,
        fingerprint: "ArchitectureFingerprint",
        variant: str,
        rank: int = 8,
    ) -> Tuple[float, float]:
        """Predict ΔmAP with bootstrap uncertainty estimate.

        Uses the stored calibration history to compute a bootstrap confidence
        interval.  When no history is available, returns a heuristic uncertainty
        based on the fingerprint's distance from the calibration centroid.

        Args:
            fingerprint: The architecture fingerprint.
            variant: The PEFT variant name.
            rank: LoRA rank (default 8).

        Returns:
            Tuple of (predicted_delta, std_error).
        """
        point_pred = self.predict(fingerprint, variant, rank)

        if not self._history or len(self._history) < 5:
            # Heuristic: higher uncertainty for extreme phi_attn values
            # (far from the calibration centroid ~0.3).
            distance = abs(fingerprint.phi_attn - 0.3)
            return point_pred, 0.02 + 0.1 * distance

        try:
            import numpy as np
        except ImportError:
            return point_pred, 0.02

        # Bootstrap: resample history with replacement, refit, predict.
        n_boot = min(50, len(self._history))
        n_history = len(self._history)
        boot_preds = []
        rng = np.random.default_rng(0)
        for _ in range(n_boot):
            idx = rng.integers(0, n_history, size=n_history)
            boot_history = [self._history[i] for i in idx]
            boot_planner = PEFTPlanner()
            boot_planner.fit(boot_history)
            boot_preds.append(boot_planner.predict(fingerprint, variant, rank))

        std_error = float(np.std(boot_preds)) if len(boot_preds) > 1 else 0.02
        return point_pred, std_error

    def _calibration_metadata(self) -> Dict[str, Any]:
        """Return decision evidence describing the active regression calibration."""
        fitted = len(self._history) >= 5
        n_samples = self._fit_n_samples if fitted else 0
        return {
            "calibration_fitted": fitted,
            "calibration_n_samples": n_samples,
            "calibration_effective_rank": self._fit_effective_rank if fitted else 0,
            "calibration_n_features": self._fit_n_features,
            "low_confidence": bool(fitted and n_samples < 30),
        }

    def plan(self, model: nn.Module, config: Any) -> PlacementDecision:
        """Compute the planner decision on rank 0 and broadcast it to every DDP rank."""
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return self._plan_local(model, config)
        rank = torch.distributed.get_rank()
        envelope = None
        if rank == 0:
            try:
                envelope = {"decision": self._plan_local(model, config).to_dict(), "error": None}
            except Exception as exc:
                envelope = {"decision": None, "error": f"{type(exc).__name__}: {exc}"}
        container = [envelope]
        torch.distributed.broadcast_object_list(container, src=0)
        envelope = container[0]
        if not isinstance(envelope, dict) or envelope.get("error") or not envelope.get("decision"):
            error = envelope.get("error") if isinstance(envelope, dict) else "invalid DDP planner envelope"
            reason = f"Rank-0 PEFT Planner failed: {error}. Falling back to Full-SFT."
            LOGGER.warning(f"[Planner] {reason}")
            return PlacementDecision(
                status="REFUSE",
                refusal_reason=reason,
                safety_overrides={"planner_refused": True, "planner_ddp_fallback": True},
                metadata={"ddp_rank0_error": error},
            )
        return PlacementDecision.from_dict(envelope["decision"])

    def _plan_local(self, model: nn.Module, config: Any) -> PlacementDecision:
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
            "φ_text=%.3f, φ_dw=%.3f, φ_group=%.3f, φ_linear=%.3f, "
            "φ_depth=%.3f, φ_width=%.3f, φ_head=%.3f, "
            "φ_residual=%.3f, φ_norm=%.3f",
            fingerprint.phi_attn,
            fingerprint.phi_text,
            fingerprint.phi_dw,
            fingerprint.phi_group,
            fingerprint.phi_linear,
            fingerprint.phi_depth,
            fingerprint.phi_width,
            fingerprint.phi_head,
            fingerprint.phi_residual,
            fingerprint.phi_norm,
        )

        # Auto-fit from LOVO collector if available (regression-dominant
        # requires the model to be calibrated on catastrophic data when possible).
        self._maybe_fit_from_lovo()
        calibration_metadata = self._calibration_metadata()

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
            variant_scores[v] = self.predict(fingerprint, v, rank)

        if not variant_scores:
            decision = PlacementDecision(
                status="REFUSE",
                refusal_reason="No compatible PEFT variant found for this architecture.",
                predicted_delta=None,
                target_modules_hint=[],
                safety_overrides={"planner_refused": True},
                metadata=calibration_metadata,
            )
            self._save_audit(fingerprint, variant, rank, decision, targets_hint)
            return decision

        # Select best variant with deterministic tie-breaking: when two variants
        # have identical predicted ΔmAP, prefer 'lora' (most stable), then
        # alphabetical order. This prevents non-deterministic ADAPT decisions
        # across runs due to floating-point noise in regression predictions.
        best_variant = max(
            variant_scores,
            key=lambda v: (variant_scores[v], v == "lora", -ord(v[0])),
        )
        best_delta = variant_scores[best_variant]

        requested_profile = PEFTVariantProfile.from_variant(variant)
        requested_compatible = (
            (fingerprint.phi_attn <= 0.05 or requested_profile.supports_attention)
            and (fingerprint.phi_text <= 0.05 or requested_profile.supports_text_fusion)
        )
        if variant in variant_scores:
            requested_delta = variant_scores[variant]
        elif requested_compatible:
            # Variant was not evaluated in the loop but is compatible — predict now.
            requested_delta = self.predict(fingerprint, variant, rank)
        else:
            # Variant is incompatible — use best_delta as reference, mark as incompatible.
            LOGGER.info(
                "[Planner] Requested variant '%s' is incompatible with this "
                "architecture (φ_attn=%.3f, φ_text=%.3f); will ADAPT to %s.",
                variant, fingerprint.phi_attn, fingerprint.phi_text, best_variant,
            )
            requested_delta = best_delta

        LOGGER.info(
            "[Planner] Regression: requested %s Δ=%.4f, best %s Δ=%.4f",
            variant, requested_delta, best_variant, best_delta,
        )

        safety_overrides: Dict[str, Any] = {}
        recommended_variant: Optional[str] = None
        recommended_rank: Optional[int] = None

        # === Phase 2: Hard safety guardrails ===
        # Guardrail B is UNCONDITIONAL: it applies regardless of whether
        # using_defaults is True or False. This prevents a LOVO-fitted
        # regression from incorrectly ACCEPTing a known-catastrophic
        # configuration (RT-DETR + LoRA-family has 7/7 catastrophe rate
        # per paper Fig. 4). The regression may not have seen enough
        # catastrophic data to generalise this pattern.
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

        # Guardrail B: RT-DETR-like architecture + LoRA-family.
        # UNCONDITIONAL — fires even when using_defaults=False.
        # Paper Fig. 4: RT-DETR-l has 7/7 catastrophe rate for LoRA-family.
        # Trigger conditions:
        #   (a) phi_attn > 0.7 (paper-calibrated threshold), OR
        #   (b) architecture family is "rtdetr" (detected via RTDETRDecoder /
        #       MultiheadAttention / MSDeformAttn class presence).
        # Condition (b) is necessary because actual module-scan phi_attn for
        # RT-DETR-l is ~0.10 (backbone has many conv layers that dilute the ratio),
        # while the paper-calibrated profile assumes phi_attn≈0.85.
        arch_family = ArchitectureFingerprint._detect_architecture_family(inner_model)
        is_rtdetr_like = (
            fingerprint.phi_attn > 0.7
            or arch_family == "rtdetr"
        )
        if is_rtdetr_like and variant.lower() in ("lora", "dora", "loha", "lokr"):
            LOGGER.warning(
                "[Planner] Guardrail B (unconditional): RT-DETR-like "
                "architecture (φ_attn=%.2f, family=%s) + %s → REFUSE "
                "(using_defaults=%s, regression Δ=%.4f ignored)",
                fingerprint.phi_attn, arch_family, variant, using_defaults, requested_delta,
            )
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
                metadata=calibration_metadata,
            )
            self._save_audit(fingerprint, variant, rank, decision, targets_hint)
            return decision

        # === Phase 3: Regression-dominant decision ===

        # Apply statistical gates only to a fitted calibration model. Default
        # coefficients have no empirical sampling distribution and retain the
        # established hard-guardrail behavior above.
        if calibration_metadata["calibration_fitted"]:
            requested_delta, requested_std_error = self.predict_with_uncertainty(fingerprint, variant, rank)
            requested_lower_bound = requested_delta - 1.96 * requested_std_error
            best_delta, best_std_error = self.predict_with_uncertainty(fingerprint, best_variant, rank)
            best_lower_bound = best_delta - 1.96 * best_std_error
            calibration_metadata.update(
                {
                    "prediction_std_error": requested_std_error,
                    "prediction_lower_95": requested_lower_bound,
                    "best_variant": best_variant,
                    "best_prediction_std_error": best_std_error,
                    "best_prediction_lower_95": best_lower_bound,
                }
            )
            if requested_lower_bound < self.REFUSE_THRESHOLD:
                if best_variant != variant and best_lower_bound >= self.REFUSE_THRESHOLD:
                    decision = PlacementDecision(
                        status="ADAPT",
                        recommended_variant=best_variant,
                        predicted_delta=best_delta,
                        target_modules_hint=targets_hint,
                        safety_overrides={"variant_adapted": True, "uncertainty_guard": True},
                        metadata=calibration_metadata,
                    )
                    self._save_audit(fingerprint, variant, rank, decision, targets_hint)
                    return decision
                decision = PlacementDecision(
                    status="REFUSE",
                    refusal_reason=(
                        f"Prediction 95% lower bound ({requested_lower_bound:.4f}) below threshold "
                        f"({self.REFUSE_THRESHOLD}) for {variant}."
                    ),
                    predicted_delta=requested_delta,
                    target_modules_hint=[],
                    safety_overrides={"planner_refused": True, "uncertainty_guard": True},
                    metadata=calibration_metadata,
                )
                self._save_audit(fingerprint, variant, rank, decision, targets_hint)
                return decision

        # If the requested variant is architecturally incompatible.
        if not requested_compatible:
            if best_delta >= self.REFUSE_THRESHOLD:
                decision = PlacementDecision(
                    status="ADAPT",
                    recommended_variant=best_variant,
                    predicted_delta=best_delta,
                    target_modules_hint=targets_hint,
                    safety_overrides={"variant_adapted": True},
                    metadata=calibration_metadata,
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
                metadata=calibration_metadata,
            )
            self._save_audit(fingerprint, variant, rank, decision, targets_hint)
            return decision

        # If a safety guardrail already triggered a variant change.
        if recommended_variant is not None:
            new_delta = variant_scores.get(
                recommended_variant, self.predict(fingerprint, recommended_variant, rank)
            )
            decision = PlacementDecision(
                status="ADAPT",
                recommended_variant=recommended_variant,
                predicted_delta=new_delta,
                target_modules_hint=targets_hint,
                safety_overrides=safety_overrides,
                metadata=calibration_metadata,
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
                    metadata=calibration_metadata,
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
                metadata=calibration_metadata,
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

        # CNN architecture rank cap: protect against memory blowup on large models.
        # For attention-poor architectures (φ_attn < 0.05), apply a parameter-count-
        # based rank ceiling. YOLOv8x (~68M params) with r=128 and ~200 target conv
        # layers generates ~13M adapter params + ~39M optimizer state (Adam 2x),
        # which can cause OOM on consumer GPUs.
        if fingerprint.phi_attn < 0.05:
            model_params = sum(p.numel() for p in inner_model.parameters())
            # Tiered rank ceiling based on model size:
            #   < 10M params  (n-series):  no cap (r up to 128)
            #   10-50M params (s/m-series): cap at 64
            #   > 50M params  (l/x-series): cap at 32
            if model_params > 50_000_000:
                _cnn_rank_cap = 32
            elif model_params > 10_000_000:
                _cnn_rank_cap = 64
            else:
                _cnn_rank_cap = None

            if _cnn_rank_cap is not None and rank > _cnn_rank_cap:
                recommended_rank = _cnn_rank_cap
                safety_overrides["r"] = _cnn_rank_cap
                LOGGER.info(
                    "[Planner] CNN rank cap: model has %.1fM params → "
                    "capping rank from %d to %d to prevent memory blowup",
                    model_params / 1e6, rank, _cnn_rank_cap,
                )

            # Grouped convolution rank alignment: when the model contains grouped
            # convolutions, LoRA rank should be aligned to the greatest common
            # divisor of group counts to avoid parameter dimension mismatches.
            # This is a low-priority safety measure; the safety_overrides["r"]
            # value is adjusted downward to the nearest multiple of all groups
            # if it isn't already compatible.
            _gcd = self._compute_group_gcd(inner_model)
            if _gcd > 1 and recommended_rank is not None and recommended_rank % _gcd != 0:
                aligned_rank = (recommended_rank // _gcd) * _gcd
                if aligned_rank > 0:
                    LOGGER.info(
                        "[Planner] Grouped conv rank alignment: "
                        "rank %d → %d (gcd=%d)",
                        recommended_rank, aligned_rank, _gcd,
                    )
                    recommended_rank = aligned_rank
                    safety_overrides["r"] = aligned_rank

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
                metadata=calibration_metadata,
            )
        elif calibration_metadata["low_confidence"]:
            decision = PlacementDecision(
                status="ADAPT",
                recommended_variant=variant,
                recommended_rank=rank if rank > 0 else None,
                predicted_delta=requested_delta,
                target_modules_hint=targets_hint,
                safety_overrides={"planner_low_confidence": True},
                metadata=calibration_metadata,
            )
        else:
            decision = PlacementDecision(
                status="ACCEPT",
                predicted_delta=requested_delta,
                target_modules_hint=targets_hint,
                metadata=calibration_metadata,
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
                    # Extended dimensions (v2)
                    "phi_depth": fingerprint.phi_depth,
                    "phi_width": fingerprint.phi_width,
                    "phi_head": fingerprint.phi_head,
                    "phi_residual": fingerprint.phi_residual,
                    "phi_norm": fingerprint.phi_norm,
                },
                variant=variant,
                requested_rank=requested_rank,
                decision_status=decision.status,
                recommended_variant=decision.recommended_variant,
                recommended_rank=decision.recommended_rank,
                predicted_delta=decision.predicted_delta,
                refusal_reason=decision.refusal_reason,
                safety_overrides=dict(decision.safety_overrides),
                metadata=dict(decision.metadata),
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
        # Detect architecture family for family-level guardrail logic.
        # This is necessary because RT-DETR-l's actual phi_attn is ~0.10
        # (many conv layers dilute the ratio), while the paper-calibrated
        # profile assumes phi_attn≈0.85.
        arch_family = ArchitectureFingerprint._detect_architecture_family(inner_model)
        is_rtdetr = (fingerprint.phi_attn >= 0.7 or arch_family == "rtdetr")
        targets: List[str] = []
        include_text = fingerprint.phi_text > 0.05

        from ultralytics.vpeft.graph import ComputationGraphBuilder

        graph = ComputationGraphBuilder().build(inner_model)
        annotations = {node.name: node.annotations or {} for node in graph.nodes}
        for name, module in inner_model.named_modules():
            if not name:
                continue

            is_conv = isinstance(module, nn.Conv2d)
            is_linear = isinstance(module, nn.Linear)
            if not (is_conv or is_linear):
                continue

            lname = name.lower()
            structural = annotations.get(name, {})
            if structural.get("in_head") and not bool(getattr(config, "include_head", False)):
                continue
            if structural.get("dynamic_routing") and not bool(getattr(config, "include_moe", True)):
                continue

            # YOLOv5 Focus layer: the Focus module uses a pixel-shuffle-like
            # operation followed by a Conv2d. The Focus conv has a 4× larger
            # input channel count (4× spatial concatenation), making LoRA
            # adapters on it both memory-heavy and unstable. Exclude by default.
            if "focus" in lname:
                continue
            # Text-fusion detection: use precise patterns (v2 improvement).
            # Bare "fusion" keyword removed to avoid false positives on
            # modules like feature_fusion or fusion_layer.
            is_text_fusion = any(
                k in lname for k in (
                    "text_fusion", "text_proj", "clip", "lang", "world_embed",
                )
            )

            # Text-fusion modules are always included when detected.
            if is_text_fusion and include_text:
                targets.append(name)
                continue

            # YOLO11s-like (no attention, not RT-DETR): conv only, no attention.
            if fingerprint.phi_attn < 0.05 and not is_rtdetr:
                if is_conv and "attn" not in lname:
                    targets.append(name)
                continue

            # YOLO12s-like (moderate attention): conv + safe attention.
            if 0.05 <= fingerprint.phi_attn < 0.7 and not is_rtdetr:
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

            # RT-DETR-like (high attention or rtdetr family): refuse targets.
            if is_rtdetr:
                continue

        # Apply additional config-level filters if provided.
        if config is not None:
            only_backbone = getattr(config, "only_backbone", False)
            include_head = getattr(config, "include_head", False)
            exclude_modules = getattr(config, "exclude_modules", None) or []

            filtered = []
            for name in targets:
                lname = name.lower()
                structural = annotations.get(name, {})
                if only_backbone and (structural.get("in_head") or any(
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
                )):
                    continue
                if not include_head and (
                    structural.get("in_head")
                    or any(p in lname for p in ("head", "detect", "score_head", "bbox_head"))
                ):
                    continue
                if any(ex in name for ex in exclude_modules):
                    continue
                filtered.append(name)
            targets = filtered

        return sorted(targets)

    @staticmethod
    def _compute_group_gcd(model: nn.Module) -> int:
        """Compute the GCD of all grouped convolution ``groups`` values.

        Used for rank alignment: LoRA rank should be a multiple of the
        group GCD to avoid dimension mismatches in grouped conv layers.

        Args:
            model: The model to scan.

        Returns:
            GCD of all ``Conv2d.groups`` values > 1. Returns 1 if no
            grouped convolutions are found.
        """
        import math as _math
        result = 0
        for module in model.modules():
            if isinstance(module, nn.Conv2d) and module.groups > 1:
                result = _math.gcd(result, module.groups)
        return result if result > 0 else 1


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
