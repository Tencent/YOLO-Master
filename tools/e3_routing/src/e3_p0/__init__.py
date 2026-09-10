"""E3 P0 unified routing diagnostics."""

from .adapters import SCHEMA_VERSION, adapt_snapshot, routing_metrics
from .collector import RoutingCollector, discover_routed_modules

__all__ = (
    "SCHEMA_VERSION",
    "RoutingCollector",
    "adapt_snapshot",
    "discover_routed_modules",
    "routing_metrics",
)
