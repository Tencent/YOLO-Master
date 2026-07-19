"""
MoE Weight Verification Utility (P0-3)

Verifies that MoE expert weights are correctly loaded from checkpoints,
detecting mismatched keys, shape errors, and uninitialized expert parameters.

Usage:
    from ultralytics.nn.modules.moe.weight_verify import verify_moe_weights
    report = verify_moe_weights(model, checkpoint_path)
    if report.has_issues():
        print(report.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ultralytics.utils.patches import torch_load

from .utils import is_core_moe_block

logger = logging.getLogger(__name__)


@dataclass
class WeightVerifyReport:
    """Report from MoE weight verification."""

    total_keys: int = 0
    matched_keys: int = 0
    mismatched_shape_keys: list[str] = field(default_factory=list)
    missing_in_checkpoint: list[str] = field(default_factory=list)
    missing_in_model: list[str] = field(default_factory=list)
    moe_keys_checked: int = 0
    moe_keys_matched: int = 0
    uninitialized_moe_params: list[str] = field(default_factory=list)
    expert_balance_issues: list[str] = field(default_factory=list)

    def has_issues(self) -> bool:
        """Return True if any problems were found."""
        return bool(
            self.mismatched_shape_keys
            or self.missing_in_checkpoint
            or self.missing_in_model
            or self.uninitialized_moe_params
            or self.expert_balance_issues
        )

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "MoE Weight Verification Report",
            f"  Total keys: {self.total_keys}",
            f"  Matched: {self.matched_keys}/{self.total_keys}",
            f"  MoE keys checked: {self.moe_keys_checked}",
            f"  MoE keys matched: {self.moe_keys_matched}/{self.moe_keys_checked}",
        ]
        if self.mismatched_shape_keys:
            lines.append(f"  Shape mismatches ({len(self.mismatched_shape_keys)}):")
            for k in self.mismatched_shape_keys[:10]:
                lines.append(f"    - {k}")
        if self.missing_in_checkpoint:
            lines.append(f"  Missing in checkpoint ({len(self.missing_in_checkpoint)}):")
            for k in self.missing_in_checkpoint[:10]:
                lines.append(f"    - {k}")
        if self.uninitialized_moe_params:
            lines.append(f"  Uninitialized MoE params ({len(self.uninitialized_moe_params)}):")
            for k in self.uninitialized_moe_params[:10]:
                lines.append(f"    - {k}")
        if self.expert_balance_issues:
            lines.append(f"  Expert balance issues ({len(self.expert_balance_issues)}):")
            for k in self.expert_balance_issues[:10]:
                lines.append(f"    - {k}")
        return "\n".join(lines)


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint file and extract the model state_dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch_load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model = ckpt["model"]
        if hasattr(model, "state_dict"):
            return model.state_dict()
        return model
    return ckpt


def _is_moe_key(key: str) -> bool:
    """Check if a state_dict key belongs to a MoE module."""
    moe_indicators = (
        ".experts.",
        ".routing.",
        ".fused_experts.",
        ".gate",
        ".router",
        ".balance_loss_coeff",
        ".moe_loss_fn",
        ".expert_usage_counts",
        ".load_balancing_loss",
        ".dynamic_channels",
    )
    return any(ind in key for ind in moe_indicators)


def _check_expert_balance(model: torch.nn.Module) -> list[str]:
    """Check if expert weights within each MoE module are diverse (not identical)."""
    issues = []
    for name, module in model.named_modules():
        if not is_core_moe_block(module):
            continue
        experts = getattr(module, "experts", None)
        if experts is None or not hasattr(experts, "__len__"):
            experts = getattr(module, "fused_experts", None)
        if experts is None or not hasattr(experts, "__len__"):
            continue

        # Compare first conv weight of each expert
        ref_weight = None
        for idx, expert in enumerate(experts if isinstance(experts, (list, torch.nn.ModuleList)) else [experts]):
            # Find first weight parameter
            for param_name, param in expert.named_parameters():
                if "weight" in param_name and param.dim() > 1:
                    if ref_weight is None:
                        ref_weight = param.data.clone()
                    elif ref_weight.shape == param.shape:
                        # Check if identical (potential init bug)
                        if torch.equal(ref_weight, param.data):
                            issues.append(
                                f"{name}.expert[{idx}].{param_name}: identical to expert[0] "
                                f"(possible uninitialized expert)"
                            )
                    break  # only check first weight per expert
    return issues


def verify_moe_weights(
    model: torch.nn.Module,
    checkpoint: str | Path | dict[str, Any],
    verbose: bool = True,
) -> WeightVerifyReport:
    """Verify MoE weight loading from a checkpoint into a model.

    Args:
        model: The target PyTorch model (with MoE layers).
        checkpoint: Path to checkpoint file, or a raw state_dict.
        verbose: If True, log findings.

    Returns:
        WeightVerifyReport with detailed findings.
    """
    if isinstance(checkpoint, (str, Path)):
        ckpt_sd = _load_checkpoint(checkpoint)
    else:
        ckpt_sd = checkpoint

    model_sd = model.state_dict()
    report = WeightVerifyReport(total_keys=len(model_sd))

    # Categorize keys
    ckpt_keys = set(ckpt_sd.keys())
    model_keys = set(model_sd.keys())

    for key in model_keys:
        ckpt_val = ckpt_sd.get(key)
        if ckpt_val is None:
            report.missing_in_checkpoint.append(key)
            continue
        model_val = model_sd[key]
        if ckpt_val.shape != model_val.shape:
            report.mismatched_shape_keys.append(
                f"{key}: ckpt={tuple(ckpt_val.shape)} vs model={tuple(model_val.shape)}"
            )
        else:
            report.matched_keys += 1
            if _is_moe_key(key):
                report.moe_keys_checked += 1
                report.moe_keys_matched += 1

    for key in ckpt_keys:
        if key not in model_keys:
            report.missing_in_model.append(key)

    # Count MoE keys that were not matched
    for key in report.missing_in_checkpoint:
        if _is_moe_key(key):
            report.moe_keys_checked += 1
            report.uninitialized_moe_params.append(key)

    # Check expert weight diversity
    report.expert_balance_issues = _check_expert_balance(model)

    if verbose and report.has_issues():
        logger.warning(report.summary())
    elif verbose:
        logger.info(
            f"MoE weight verification: PASS "
            f"({report.matched_keys}/{report.total_keys} keys matched, "
            f"{report.moe_keys_matched}/{report.moe_keys_checked} MoE keys)"
        )

    return report


def safe_load_with_verify(
    model: torch.nn.Module,
    checkpoint: str | Path,
    strict: bool = False,
    verbose: bool = True,
) -> WeightVerifyReport:
    """Load checkpoint into model with MoE weight verification.

    This is a drop-in replacement for model.load() that adds MoE-specific
    verification before and after loading.

    Args:
        model: Target model (must have .load() method or be a BaseModel).
        checkpoint: Path to checkpoint file.
        strict: If True, fail on any mismatch.
        verbose: If True, log findings.

    Returns:
        WeightVerifyReport.
    """
    ckpt = torch_load(str(checkpoint), map_location="cpu", weights_only=False)
    ckpt_model = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    ckpt_sd = ckpt_model.state_dict() if hasattr(ckpt_model, "state_dict") else ckpt_model

    report = verify_moe_weights(model, ckpt_sd, verbose=verbose)

    if strict and report.has_issues():
        raise RuntimeError(
            f"Strict MoE weight verification failed:\n{report.summary()}"
        )

    # Perform actual load (using the model's own load method if available)
    if hasattr(model, "load"):
        model.load(ckpt, verbose=verbose)
    else:
        from ultralytics.utils.torch_utils import intersect_dicts
        csd = intersect_dicts(ckpt_sd, model.state_dict())
        model.load_state_dict(csd, strict=False)

    # Post-load verification: re-check after loading
    post_report = verify_moe_weights(model, ckpt_sd, verbose=False)
    if verbose and post_report.uninitialized_moe_params:
        logger.warning(
            f"Post-load check: {len(post_report.uninitialized_moe_params)} "
            f"MoE params still uninitialized after loading"
        )

    return report
