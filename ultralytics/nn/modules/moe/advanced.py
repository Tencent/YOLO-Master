# 🐧Please note that this file has been modified by Tencent on 2026/02/13. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Backward-compatibility re-export shim for the former monolithic ``advanced.py``.

The 1 077-line file has been split into three focused modules:

- ``routers_advanced.py``  → ``DualStreamGateRouter``, ``DualStreamGateRouterV2``,
  ``ZeroCostRouter``
- ``experts_advanced.py``  → ``FusedExpertGroup``, ``LowRankFusedExpertGroup``
- ``blocks_advanced.py``   → ``AdaptiveGateMoE``, ``HyperSplitMoE``,
  ``HyperFusedMoE``

All symbols are re-exported here so that existing
``from .advanced import ...`` statements continue to work without changes.
"""

from .routers_advanced import (
    DualStreamGateRouter,
    DualStreamGateRouterV2,
    ZeroCostRouter,
)
from .experts_advanced import (
    FusedExpertGroup,
    LowRankFusedExpertGroup,
)
from .blocks_advanced import (
    AdaptiveGateMoE,
    HyperSplitMoE,
    HyperFusedMoE,
)

__all__ = (
    "DualStreamGateRouter",
    "DualStreamGateRouterV2",
    "ZeroCostRouter",
    "FusedExpertGroup",
    "LowRankFusedExpertGroup",
    "AdaptiveGateMoE",
    "HyperSplitMoE",
    "HyperFusedMoE",
)
