"""Route-authority policy for split dynamic-expert deployment and validation."""

from __future__ import annotations

from collections.abc import Sequence


EXPORTED_ROUTER_HOST_TOPK_AUTHORITY = "exported_router_host_topk"
EAGER_CHECKPOINT_ROUTE_DIAGNOSTIC = "eager_checkpoint_route_diagnostic_only"

ROUTE_GATE_REPORT_ONLY = "report_only"
ROUTE_GATE_EXACT_REFERENCE = "exact_reference"
ROUTE_GATE_EXPORTED_AUTHORITATIVE = "exported_authoritative"
SUPPORTED_ROUTE_GATE_MODES = frozenset(
    (ROUTE_GATE_REPORT_ONLY, ROUTE_GATE_EXACT_REFERENCE, ROUTE_GATE_EXPORTED_AUTHORITATIVE)
)


def evaluate_route_gate(
    block_summaries: Sequence[dict],
    *,
    mode: str,
    reference_audit_enabled: bool,
) -> dict:
    """Evaluate deployment authority separately from eager-reference route drift.

    The exported router's host Top-K is the only route used for deployment
    dispatch. Eager checkpoint routes remain an important diagnostic, but two
    floating-point backends cannot be required to make the same discrete choice
    at every decision boundary unless they consume the same routing result.
    """
    if mode not in SUPPORTED_ROUTE_GATE_MODES:
        raise ValueError(f"unsupported route gate mode: {mode!r}")

    mismatches = sum(int(item.get("route_location_mismatch_count", 0)) for item in block_summaries)
    locations = sum(int(item.get("route_location_total", 0)) for item in block_summaries)
    exact_match = (mismatches == 0) if reference_audit_enabled else None
    authority_declared = bool(block_summaries) and all(
        item.get("route_authority") == EXPORTED_ROUTER_HOST_TOPK_AUTHORITY for item in block_summaries
    )
    dispatch_verified = bool(block_summaries) and all(
        bool(item.get("authoritative_route_dispatch_verified")) for item in block_summaries
    )
    authority_passed = authority_declared and dispatch_verified

    if mode == ROUTE_GATE_EXACT_REFERENCE:
        passed = bool(reference_audit_enabled and exact_match)
    elif mode == ROUTE_GATE_EXPORTED_AUTHORITATIVE:
        passed = authority_passed
    else:
        passed = True

    return {
        "mode": mode,
        "passed": passed,
        "authoritative_route": {
            "name": EXPORTED_ROUTER_HOST_TOPK_AUTHORITY,
            "declared_by_all_blocks": authority_declared,
            "dispatch_verified_by_all_blocks": dispatch_verified,
            "passed": authority_passed,
        },
        "eager_reference": {
            "role": EAGER_CHECKPOINT_ROUTE_DIAGNOSTIC,
            "audit_enabled": reference_audit_enabled,
            "exact_match": exact_match,
            "mismatched_locations": mismatches,
            "total_locations": locations,
            "mismatch_ratio": mismatches / locations if locations else 0.0,
        },
        # Compatibility fields retained for existing evidence consumers.
        "audit_enabled": reference_audit_enabled,
        "exact_route_required": mode == ROUTE_GATE_EXACT_REFERENCE,
        "mismatched_locations": mismatches,
        "total_locations": locations,
        "mismatch_ratio": mismatches / locations if locations else 0.0,
    }


__all__ = (
    "EAGER_CHECKPOINT_ROUTE_DIAGNOSTIC",
    "EXPORTED_ROUTER_HOST_TOPK_AUTHORITY",
    "ROUTE_GATE_EXACT_REFERENCE",
    "ROUTE_GATE_EXPORTED_AUTHORITATIVE",
    "ROUTE_GATE_REPORT_ONLY",
    "SUPPORTED_ROUTE_GATE_MODES",
    "evaluate_route_gate",
)
