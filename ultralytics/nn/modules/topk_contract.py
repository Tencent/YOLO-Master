"""Versioned deterministic Top-K policy names shared by eager and host runtimes."""

from __future__ import annotations


LEGACY_PRIORITY_BIAS_TOPK = "probability_minus_expert_id_times_tolerance"
DEADBAND_LOWEST_ID_TOPK = "max_deadband_then_lowest_expert_id"
DEFAULT_DETERMINISTIC_TOPK = DEADBAND_LOWEST_ID_TOPK
SUPPORTED_DETERMINISTIC_TOPK = frozenset((LEGACY_PRIORITY_BIAS_TOPK, DEADBAND_LOWEST_ID_TOPK))


__all__ = (
    "DEADBAND_LOWEST_ID_TOPK",
    "DEFAULT_DETERMINISTIC_TOPK",
    "LEGACY_PRIORITY_BIAS_TOPK",
    "SUPPORTED_DETERMINISTIC_TOPK",
)
