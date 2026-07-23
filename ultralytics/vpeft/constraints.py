"""Constraint registry for V-PEFT Solver.

Provides 7 concrete constraint classes + ConstraintRegistry for hard/soft
constraint projection used by the placement policy and solver modules.

Hard constraints  → binary feasibility mask (AND logic).
Soft constraints  → scalar penalties (≥ 0) weighted by Lagrange multipliers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .graph import ComputationGraph, GraphNode, ModuleNode, _estimate_adapter_params

__all__ = [
    "NodeInfo",
    "Constraint",
    "OperatorCompatibilityConstraint",
    "SemanticProtectionConstraint",
    "BudgetConstraint",
    "DeploymentCompatibilityConstraint",
    "VariantModuleCompatibilityConstraint",
    "MoEConsistencyConstraint",
    "DivisibilityConstraint",
    "CandidateTargetConstraint",
    "ConstraintRegistry",
]


# ---------------------------------------------------------------------------
# NodeInfo — unified wrapper for module / node metadata
# ---------------------------------------------------------------------------

class NodeInfo:
    """Lightweight wrapper that normalises GraphNode, ModuleNode, nn.Module, or dict
    into a uniform interface consumed by constraints.
    """

    def __init__(self, obj: Union[GraphNode, ModuleNode, nn.Module, Dict[str, Any], None] = None, **kwargs):
        self._src = obj
        self._kwargs = kwargs

        # Extract fields
        self.name: str = self._extract("name", default="")
        self.operator_type: str = self._extract("operator_type", "op_type", default="Other")
        self.in_channels: int = self._extract("in_channels", "c_in", default=0)
        self.out_channels: int = self._extract("out_channels", "c_out", default=0)
        self.groups: int = self._extract("groups", default=1)
        self.kernel_size: Union[int, Tuple[int, int]] = self._extract("kernel_size", default=(1, 1))
        self.semantic_role: str = self._extract("semantic_role", "sigma_i", default="other")
        self.moe_group: str = self._extract("moe_group", default="")
        self.module: Optional[nn.Module] = self._extract_module()

        # Normalise operator_type from GraphNode tau_i
        if isinstance(obj, GraphNode):
            module = getattr(obj, "module", None)
            self.groups = int(getattr(module, "groups", self.groups))
            module_kernel = getattr(module, "kernel_size", None)
            if module_kernel is not None:
                self.kernel_size = module_kernel
            tau_map = {
                0: "Conv2d",
                1: "Linear",
                2: "MultiheadAttention",
                3: "BatchNorm2d",
                4: "LayerNorm",
                5: "DepthwiseConv2d",
                6: "GroupConv2d",
                7: "AttentionLinear",
                8: "Other",
            }
            tau = getattr(obj.attributes, "tau_i", 8)
            if self.operator_type in ("", "Other"):
                self.operator_type = tau_map.get(tau, "Other")
            # Semantic role from sigma_i
            sigma_map = {
                0: "stem",
                1: "backbone",
                2: "neck",
                3: "head",
                4: "DFL",
                5: "text_fusion",
                6: "MoE_router",
                7: "MoE_expert",
                8: "MSDeformAttn",
                9: "AAttn",
                10: "other",
            }
            sigma = getattr(obj.attributes, "sigma_i", 10)
            if self.semantic_role in ("", "other"):
                self.semantic_role = sigma_map.get(sigma, "other")
            self.moe_group = str((getattr(obj, "annotations", {}) or {}).get("moe_group", self.moe_group) or "")
            # kernel_size from k_i
            k_i = getattr(obj.attributes, "k_i", 1)
            if k_i > 0 and self.kernel_size == (1, 1):
                self.kernel_size = (k_i, k_i)
        elif isinstance(obj, ModuleNode):
            self.semantic_role = obj.semantic_role

        # Normalise kernel_size to tuple
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)

        # Depthwise detection
        self.is_depthwise: bool = self._compute_is_depthwise()

    def _extract(self, *keys: str, default: Any = None) -> Any:
        obj = self._src
        if obj is None:
            return self._kwargs.get(keys[0], default)
        if isinstance(obj, dict):
            for k in keys:
                if k in obj:
                    return obj[k]
            return self._kwargs.get(keys[0], default)
        for k in keys:
            if hasattr(obj, k):
                return getattr(obj, k)
        # Try attributes dataclass for GraphNode
        if isinstance(obj, GraphNode) and hasattr(obj, "attributes"):
            attr = obj.attributes
            for k in keys:
                if hasattr(attr, k):
                    return getattr(attr, k)
        return self._kwargs.get(keys[0], default)

    def _extract_module(self) -> Optional[nn.Module]:
        obj = self._src
        if isinstance(obj, nn.Module) and not isinstance(obj, (GraphNode, ModuleNode)):
            return obj
        if isinstance(obj, GraphNode):
            return getattr(obj, "module", None)
        if isinstance(obj, dict):
            return obj.get("module")
        return None

    def _compute_is_depthwise(self) -> bool:
        if self.operator_type == "DepthwiseConv2d":
            return True
        if self.groups > 1 and self.in_channels == self.out_channels == self.groups:
            return True
        return False

    def __repr__(self) -> str:
        return (
            f"NodeInfo({self.name}, {self.operator_type}, "
            f"in={self.in_channels}, out={self.out_channels}, "
            f"groups={self.groups}, role={self.semantic_role})"
        )


# ---------------------------------------------------------------------------
# Constraint(ABC) — abstract base
# ---------------------------------------------------------------------------

class Constraint(ABC):
    """Abstract base class for a V-PEFT constraint.

    Attributes:
        name:  Unique constraint identifier.
        weight: Lagrange multiplier weight for soft penalty (default 1.0).
    """

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def is_feasible(self, node_info: NodeInfo, variant: str, rank: int) -> bool:
        """Return True iff the node passes the *hard* constraint."""
        ...

    def penalty(self, node_info: NodeInfo, variant: str, rank: int) -> float:
        """Return a non-negative scalar penalty for *soft* violation.
        Default: 0.0 (hard-only constraints)."""
        return 0.0


# ---------------------------------------------------------------------------
# 1. OperatorCompatibilityConstraint (C_op)
# ---------------------------------------------------------------------------

class OperatorCompatibilityConstraint(Constraint):
    """Hard constraint: variant must support the operator type.

    Mapping (from planner detect_targets rules):
        - LoRA     : Linear, Conv2d (groups=1)
        - DoRA     : Linear only
        - LoHa     : Linear, Conv2d (groups=1)
        - LoKr     : Linear, Conv2d (groups=1)
        - IA3      : Linear only
        - AdaLoRA  : Linear only
        - HRA      : Conv2d (groups=1)
        - OFT      : Linear (block_size must divide in_features)
        - BOFT     : Linear (block_size must divide in_features)
    """

    _VARIANT_OP_MAP: Dict[str, List[str]] = {
        "lora": ["Linear", "Conv2d", "GroupConv2d", "DepthwiseConv2d"],
        "dora": ["Linear"],
        "loha": ["Linear", "Conv2d"],
        "lokr": ["Linear", "Conv2d"],
        "ia3": ["Linear"],
        "adalora": ["Linear"],
        "hra": ["Conv2d"],
        "oft": ["Linear"],
        "boft": ["Linear"],
    }

    def __init__(self, allow_depthwise: bool = False, weight: float = 1.0):
        super().__init__("C_op", weight)
        self.allow_depthwise = allow_depthwise

    def is_feasible(self, node_info: NodeInfo, variant: str, rank: int) -> bool:
        v = variant.lower()
        op = node_info.operator_type

        # Depthwise conv is skipped unless explicitly allowed
        if node_info.is_depthwise and not self.allow_depthwise:
            return False

        supported = self._VARIANT_OP_MAP.get(v, [])
        if op not in supported:
            return False

        # A node labelled as plain Conv2d must not hide grouped semantics.
        # Proper GroupConv2d nodes are supported by LoRA and validated by C_div.
        if op == "Conv2d" and node_info.groups != 1:
            return False

        return True


# ---------------------------------------------------------------------------
# 2. SemanticProtectionConstraint (C_sem)
# ---------------------------------------------------------------------------

class SemanticProtectionConstraint(Constraint):
    """Hard constraint: certain semantic roles are never adapted.

    Protected roles (from planner detect_targets):
        - DFL, MSDeformAttn, text_fusion, stem, focus
        - Head / detect / bbox / score / cls / pred (unless include_head=True)
        - When only_backbone=True: also neck, fpn, pan, seg, pose
    """

    _ALWAYS_PROTECTED = {
        "dfl",
        "msdeformattn",
        "stem",
        "focus",
    }

    def __init__(
        self,
        include_head: bool = False,
        only_backbone: bool = False,
        exclude_modules: Optional[List[str]] = None,
        weight: float = 1.0,
    ):
        super().__init__("C_sem", weight)
        self.include_head = include_head
        self.only_backbone = only_backbone
        self.exclude_modules = {name.lower() for name in exclude_modules or []}

    def is_feasible(self, node_info: NodeInfo, variant: str, rank: int) -> bool:
        role = node_info.semantic_role.lower()
        name = node_info.name.lower()

        # Text-fusion is a valid semantic target only for variants with the
        # paper's text-side parameterization; plain LoRA is adapted to LoHa by
        # the planner before the budget solver runs.
        if role == "text_fusion" and variant.lower() not in {"loha", "ia3"}:
            return False

        # Always-protected roles
        if role in self._ALWAYS_PROTECTED:
            return False

        # Head protection
        if role == "head" and not self.include_head:
            return False

        # Only-backbone: exclude neck, head, and detection-related modules
        if self.only_backbone:
            if role in ("neck", "head"):
                return False
            if any(k in name for k in ("fpn", "pan", "seg", "pose", "box", "cls", "pred")):
                return False

        # Explicit exclude list by name substring
        if any(ex in name for ex in self.exclude_modules):
            return False

        return True


class CandidateTargetConstraint(Constraint):
    """Hard constraint limiting placement to the planner's candidate set."""

    def __init__(self, candidates: Optional[Iterable[str]] = None, weight: float = 1.0):
        super().__init__("C_candidates", weight)
        self.candidates = {str(name) for name in (candidates or [])}

    def is_feasible(self, node_info: NodeInfo, variant: str, rank: int) -> bool:
        return not self.candidates or node_info.name in self.candidates


# ---------------------------------------------------------------------------
# 3. BudgetConstraint (C_budget)
# ---------------------------------------------------------------------------

class BudgetConstraint(Constraint):
    """Hard + soft constraint: total adapter parameter budget.

    Supports both *global* budget checking (legacy solver interface) and
    *incremental* per-node budget tracking (new ConstraintRegistry interface).
    """

    def __init__(self, max_params: int = 2_100_000, weight: float = 1.0):
        super().__init__("C_budget", weight)
        self.max_params = max_params
        self._used: int = 0

    # -- legacy graph-level interface (required by policy.py / solver.py) --

    def evaluate(self, graph: ComputationGraph, placement, ranks, variant) -> float:
        """Return budget violation (positive if over budget)."""
        used = sum(
            graph.estimate_params(i, int(ranks[i].item()), variant)
            for i in range(graph.n_nodes)
            if placement[i] > 0.5
        )
        return max(0.0, used - self.max_params)

    # -- incremental per-node interface --

    def get_usage(self, node_info: NodeInfo, variant: str, rank: int) -> int:
        """Adapter parameters for a single node."""
        module = node_info.module
        if module is not None and hasattr(module, "params_for_rank"):
            return int(module.params_for_rank(rank, variant))
        return int(_estimate_adapter_params(
            rank,
            variant,
            node_info.operator_type,
            node_info.in_channels,
            node_info.out_channels,
            node_info.kernel_size,
            node_info.groups,
        ))

    def update_usage(self, node_info: NodeInfo, variant: str, rank: int) -> None:
        """Incrementally add a node's usage to the running total."""
        self._used += self.get_usage(node_info, variant, rank)

    def remaining(self) -> int:
        return max(0, self.max_params - self._used)

    def is_feasible(self, node_info: NodeInfo, variant: str, rank: int) -> bool:
        cost = self.get_usage(node_info, variant, rank)
        return (self._used + cost) <= self.max_params

    def penalty(self, node_info: NodeInfo, variant: str, rank: int) -> float:
        cost = self.get_usage(node_info, variant, rank)
        over = max(0, self._used + cost - self.max_params)
        # Normalise by budget to keep penalty scale invariant
        return float(over) / max(1.0, float(self.max_params))

    def reset(self) -> None:
        self._used = 0


# ---------------------------------------------------------------------------
# 4. DeploymentCompatibilityConstraint (C_deploy)
# ---------------------------------------------------------------------------

class DeploymentCompatibilityConstraint(Constraint):
    """Hard/soft constraint: platform export compatibility.

    Platform matrix (from detect_targets / ONNX/TensorRT rules):
        - ONNX:     mergeable LoRA only; no DoRA/LoHa/LoKr
        - TensorRT: same restriction as ONNX for now
        - PyTorch:  all variants allowed (no penalty)
    """

    _PLATFORM_VARIANTS = {
        "onnx": {"lora"},
        "tensorrt": {"lora"},
        "pytorch": {
            "lora", "dora", "loha", "lokr", "ia3",
            "adalora", "hra", "oft", "boft",
        },
    }

    def __init__(self, platform: str = "pytorch", weight: float = 1.0):
        super().__init__("C_deploy", weight)
        self.platform = platform.lower()

    def is_feasible(self, node_info: NodeInfo, variant: str, rank: int) -> bool:
        allowed = self._PLATFORM_VARIANTS.get(self.platform, set())
        return variant.lower() in allowed

    def penalty(self, node_info: NodeInfo, variant: str, rank: int) -> float:
        if not self.is_feasible(node_info, variant, rank):
            return 1.0  # unit penalty for incompatible variant
        return 0.0


# ---------------------------------------------------------------------------
# 5. VariantModuleCompatibilityConstraint (C_compat)
# ---------------------------------------------------------------------------

class VariantModuleCompatibilityConstraint(Constraint):
    """Fine-grained variant × module compatibility (hard constraint).

    Covers divisibility / block-size requirements that go beyond
    operator-level support:
        - HRA     : groups must be 1 (redundant with C_op but explicit)
        - AdaLoRA : only Linear (redundant with C_op but explicit)
        - OFT/BOFT: block_size must divide in_features (in_channels for Linear)
        - Conv2d  : rank must be divisible by groups when groups > 1
    """

    def __init__(self, block_size: Optional[int] = None, weight: float = 1.0):
        super().__init__("C_compat", weight)
        self.block_size = block_size

    def is_feasible(self, node_info: NodeInfo, variant: str, rank: int) -> bool:
        v = variant.lower()
        op = node_info.operator_type

        # HRA: groups == 1
        if v == "hra" and node_info.groups != 1:
            return False

        # AdaLoRA: Linear only
        if v == "adalora" and op != "Linear":
            return False

        # OFT / BOFT: block_size must divide in_features
        if v in ("oft", "boft"):
            if op == "Linear":
                in_feat = node_info.in_channels
                bs = self.block_size
                if bs is not None and in_feat % bs != 0:
                    return False
            else:
                # OFT/BOFT only supports Linear in current mapping
                return False

        # Conv2d divisibility: rank % groups == 0 when groups > 1
        if op in ("Conv2d", "GroupConv2d", "DepthwiseConv2d") and node_info.groups > 1:
            if rank % node_info.groups != 0:
                return False

        return True

    def penalty(self, node_info: NodeInfo, variant: str, rank: int) -> float:
        if not self.is_feasible(node_info, variant, rank):
            return 1.0
        return 0.0


# ---------------------------------------------------------------------------
# 6. MoEConsistencyConstraint (C_moe)
# ---------------------------------------------------------------------------

class MoEConsistencyConstraint(Constraint):
    """Hard constraint: MoE expert homogeneity.

    Rules:
        - All registered experts must use the same variant (xi).
        - Rank difference between any two experts must not exceed epsilon.
    """

    def __init__(self, epsilon: int = 4, weight: float = 1.0):
        super().__init__("C_moe", weight)
        self.epsilon = epsilon
        self.registered_experts: List[Tuple[str, int, str]] = []

    def register_expert(self, name: str, rank: int, variant: str) -> None:
        """Register an expert configuration for consistency checking."""
        self.registered_experts.append((name, rank, variant))

    def check_consistency(self) -> Tuple[bool, Optional[str]]:
        """Return (is_consistent, reason_or_None)."""
        if len(self.registered_experts) < 2:
            return True, None

        # Variant consistency
        variants = {v for _, _, v in self.registered_experts}
        if len(variants) > 1:
            return False, f"MoE variant mismatch: {variants}"

        # Rank consistency
        ranks = [r for _, r, _ in self.registered_experts]
        if max(ranks) - min(ranks) > self.epsilon:
            return False, f"MoE rank spread {max(ranks) - min(ranks)} > ε={self.epsilon}"

        return True, None

    def is_feasible(self, node_info: NodeInfo, variant: str, rank: int) -> bool:
        # MoE constraints apply only to MoE_expert role nodes
        if node_info.semantic_role != "MoE_expert":
            return True
        ok, _ = self.check_consistency()
        if not ok:
            return False
        # Also check the proposed expert against existing registry
        if self.registered_experts:
            existing_variant = self.registered_experts[0][2]
            if variant.lower() != existing_variant.lower():
                return False
            existing_ranks = [r for _, r, _ in self.registered_experts]
            if existing_ranks and max(max(existing_ranks), rank) - min(min(existing_ranks), rank) > self.epsilon:
                return False
        return True

    def penalty(self, node_info: NodeInfo, variant: str, rank: int) -> float:
        if not self.is_feasible(node_info, variant, rank):
            return 1.0
        return 0.0

    def reset(self) -> None:
        self.registered_experts.clear()

    def check_group_ranks(self, infos: List[NodeInfo], variant: str, ranks: List[int], epsilon: int | None = None) -> bool:
        """Validate a graph-discovered expert group without mutable registration."""
        if len(infos) < 2:
            return True
        eps = self.epsilon if epsilon is None else int(epsilon)
        if any(info.semantic_role != "MoE_expert" for info in infos):
            return True
        return len(ranks) == len(infos) and max(ranks) - min(ranks) <= eps


# ---------------------------------------------------------------------------
# 7. DivisibilityConstraint (C_div)
# ---------------------------------------------------------------------------

class DivisibilityConstraint(Constraint):
    """Hard/soft constraint: rank must be divisible by groups for Conv2d.

    Applies when groups > 1 (GroupConv2d / DepthwiseConv2d).
    Penalty returns the normalised remainder.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__("C_div", weight)

    def is_feasible(self, node_info: NodeInfo, variant: str, rank: int) -> bool:
        op = node_info.operator_type
        if op in ("Conv2d", "GroupConv2d", "DepthwiseConv2d") and node_info.groups > 1:
            return rank % node_info.groups == 0
        return True

    def penalty(self, node_info: NodeInfo, variant: str, rank: int) -> float:
        op = node_info.operator_type
        if op in ("Conv2d", "GroupConv2d", "DepthwiseConv2d") and node_info.groups > 1:
            g = node_info.groups
            rem = rank % g
            if rem == 0:
                return 0.0
            # Normalised penalty: min(rem, g-rem) / g  (distance to nearest multiple)
            return float(min(rem, g - rem)) / float(g)
        return 0.0


# ---------------------------------------------------------------------------
# ConstraintRegistry — orchestrates all constraints
# ---------------------------------------------------------------------------

class ConstraintRegistry:
    """Registry of hard and soft constraints.

    Hard constraints are projected into a binary feasibility mask.
    Soft constraints return scalar violations (≥ 0) for Lagrangian penalties.

    Backward compatibility:
      - Legacy ``__init__(hard_constraints, soft_constraints)`` is preserved
        as a thin wrapper around the new constraint list.
      - The three legacy methods ``get_hard_mask``, ``evaluate_soft``,
        ``get_budget_usage`` retain their original signatures.
    """

    def __init__(
        self,
        constraints: Optional[List[Constraint]] = None,
        config: Optional[Dict[str, Any]] = None,
        hard_constraints: Optional[List[str]] = None,
        soft_constraints: Optional[List[str]] = None,
    ):
        """Args:
            constraints: List of instantiated Constraint objects (new path).
            config: Optional dict for auto-building default constraints.
            hard_constraints: Legacy list of constraint names (kept for compat).
            soft_constraints: Legacy list of constraint names (kept for compat).
        """
        self._constraints: List[Constraint] = []
        self._hard_constraints: List[Constraint] = []
        self._soft_constraints: List[Constraint] = []
        self._budget_constraint: Optional[BudgetConstraint] = None
        self._moe_constraint: Optional[MoEConsistencyConstraint] = None
        self._legacy_hard_names = set(hard_constraints or [])
        self._legacy_soft_names = set(soft_constraints or [])

        if constraints is not None:
            self._register_list(constraints)
        elif config is not None:
            default_reg = ConstraintRegistry.default(config)
            self._constraints = default_reg._constraints
            self._hard_constraints = default_reg._hard_constraints
            self._soft_constraints = default_reg._soft_constraints
            self._budget_constraint = default_reg._budget_constraint
            self._moe_constraint = default_reg._moe_constraint

        # Legacy string-list support: build default constraints from names
        if hard_constraints or soft_constraints:
            self._build_from_legacy_names(hard_constraints or [], soft_constraints or [])

    @staticmethod
    def normalize_name(name: str) -> str:
        """Return the canonical constraint id used by solver logs and duals."""
        value = str(name).strip()
        if not value:
            return value
        return value if value.startswith("C_") else f"C_{value}"

    def hard_constraint_names(self) -> List[str]:
        """Return canonical names of constraints participating in projection."""
        return [constraint.name for constraint in self._hard_constraints]

    def soft_constraint_names(self) -> List[str]:
        """Return canonical names of constraints participating in dual penalties."""
        return [constraint.name for constraint in self._soft_constraints]

    def _build_from_legacy_names(self, hard_names: List[str], soft_names: List[str]) -> None:
        """Build default constraint instances from legacy name lists."""
        name_map = self._default_name_map()
        for name in hard_names:
            c = name_map.get(self.normalize_name(name))
            if c is not None and c not in self._constraints:
                self._register(c, as_hard=True)
            elif c is not None:
                self._add_classification(c, as_hard=True)
        for name in soft_names:
            c = name_map.get(self.normalize_name(name))
            if c is not None and c not in self._constraints:
                self._register(c, as_hard=False)
            elif c is not None:
                self._add_classification(c, as_hard=False)

    @staticmethod
    def _default_name_map() -> Dict[str, Constraint]:
        return {
            "C_op": OperatorCompatibilityConstraint(),
            "C_sem": SemanticProtectionConstraint(),
            "C_budget": BudgetConstraint(),
            "C_deploy": DeploymentCompatibilityConstraint(),
            "C_compat": VariantModuleCompatibilityConstraint(),
            "C_moe": MoEConsistencyConstraint(),
            "C_div": DivisibilityConstraint(),
        }

    def _register(self, c: Constraint, as_hard: bool = True) -> None:
        if c not in self._constraints:
            self._constraints.append(c)
        self._add_classification(c, as_hard=as_hard)
        if isinstance(c, BudgetConstraint):
            self._budget_constraint = c
        if isinstance(c, MoEConsistencyConstraint):
            self._moe_constraint = c

    def _add_classification(self, c: Constraint, *, as_hard: bool) -> None:
        """Classify an existing constraint without duplicating its instance."""
        target = self._hard_constraints if as_hard else self._soft_constraints
        if c not in target:
            target.append(c)

    def _register_list(self, constraints: List[Constraint]) -> None:
        for c in constraints:
            # A flat legacy list still enforces every supplied constraint.
            self._register(c, as_hard=True)
            if isinstance(c, BudgetConstraint):
                self._add_classification(c, as_hard=False)

    @property
    def constraints(self) -> List[Constraint]:
        """Return all registered constraints (read-only copy)."""
        return list(self._constraints)

    # -- public builder --

    @classmethod
    def default(cls, config: Optional[Dict[str, Any]] = None) -> "ConstraintRegistry":
        """Build a registry with explicit hard and soft constraint semantics."""
        cfg = config or {}
        registry = cls()
        all_constraints = [
            OperatorCompatibilityConstraint(allow_depthwise=cfg.get("allow_depthwise", False)),
            SemanticProtectionConstraint(
                include_head=cfg.get("include_head", False),
                only_backbone=cfg.get("only_backbone", False),
                exclude_modules=cfg.get("exclude_modules", None),
            ),
            BudgetConstraint(max_params=cfg.get("max_params", 2_100_000)),
            DeploymentCompatibilityConstraint(platform=cfg.get("platform", "pytorch")),
            VariantModuleCompatibilityConstraint(block_size=cfg.get("block_size", None)),
            MoEConsistencyConstraint(epsilon=cfg.get("moe_epsilon", 4)),
            DivisibilityConstraint(),
        ]
        by_name = {constraint.name: constraint for constraint in all_constraints}
        default_hard = ["C_op", "C_sem", "C_budget", "C_deploy", "C_compat", "C_moe", "C_div"]
        default_soft = ["C_budget", "C_deploy"]
        hard_names = [cls.normalize_name(name) for name in cfg.get("hard_constraints", default_hard)]
        soft_names = [cls.normalize_name(name) for name in cfg.get("soft_constraints", default_soft)]
        for constraint in all_constraints:
            if constraint.name in hard_names:
                registry._register(constraint, as_hard=True)
            elif constraint.name in soft_names:
                registry._register(constraint, as_hard=False)
        for name in soft_names:
            if name in by_name:
                constraint = by_name[name]
                if constraint not in registry._constraints:
                    registry._register(constraint, as_hard=False)
                else:
                    registry._add_classification(constraint, as_hard=False)
        candidates = cfg.get("candidate_targets")
        if candidates:
            registry._register(CandidateTargetConstraint(candidates), as_hard=True)
        return registry

    # -- legacy interface (required by policy.py & solver.py) --

    def get_hard_mask(
        self,
        graph: ComputationGraph,
        variant: str,
        candidate_ranks: Union[int, Iterable[int]] = 1,
    ) -> torch.Tensor:
        """Return a binary mask [N] where ``1`` means the module can be adapted
        with the given variant under hard constraints, ``0`` means forbidden.

        A node is feasible when at least one candidate rank satisfies all hard
        constraints. This matters for grouped convolutions, where rank 4 may be
        invalid but rank 8 is valid.
        """
        ranks = (candidate_ranks,) if isinstance(candidate_ranks, int) else tuple(candidate_ranks)
        if not ranks or any(rank <= 0 for rank in ranks):
            raise ValueError("candidate_ranks must contain positive integers")
        mask = torch.ones(graph.n_nodes, dtype=torch.bool)
        for i in range(graph.n_nodes):
            # Build NodeInfo from the graph's module/node at index i
            node_info = self._node_info_from_graph(graph, i)
            if not any(self.check_hard(node_info, variant, rank=rank) for rank in ranks):
                mask[i] = False

        # Enforce expert homogeneity from graph annotations.  This removes the
        # old requirement for callers to manually call register_expert().
        if self._moe_constraint is not None:
            groups: Dict[str, List[int]] = {}
            for i in range(graph.n_nodes):
                info = self._node_info_from_graph(graph, i)
                if info.semantic_role == "MoE_expert" and info.moe_group:
                    groups.setdefault(info.moe_group, []).append(i)
            for indices in groups.values():
                if len(indices) < 2:
                    continue
                feasible_sets = []
                for i in indices:
                    info = self._node_info_from_graph(graph, i)
                    feasible_sets.append({
                        rank for rank in ranks
                        if all(
                            c.is_feasible(info, variant, rank)
                            for c in self._hard_constraints
                            if c is not self._moe_constraint
                        )
                    })
                common = set.intersection(*feasible_sets) if feasible_sets else set()
                if not common:
                    for i in indices:
                        mask[i] = False
        return mask

    def enforce_moe_consistency(
        self,
        graph: ComputationGraph,
        placement: torch.Tensor,
        ranks: torch.Tensor,
        variant: Union[str, Sequence[str]],
        candidate_ranks: Iterable[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Project placed graph-discovered expert groups onto one feasible variant and rank."""
        variants = [variant] * graph.n_nodes if isinstance(variant, str) else list(variant)
        if len(variants) != graph.n_nodes:
            raise ValueError("variant list must have one entry per graph node")
        if self._moe_constraint is None:
            return placement, ranks, variants
        rank_candidates = sorted({int(rank) for rank in candidate_ranks if int(rank) > 0})
        groups: Dict[str, List[int]] = {}
        for index in range(graph.n_nodes):
            info = self._node_info_from_graph(graph, index)
            if placement[index] > 0.5 and info.semantic_role == "MoE_expert" and info.moe_group:
                groups.setdefault(info.moe_group, []).append(index)
        for indices in groups.values():
            if len(indices) < 2:
                continue
            candidate_variants = list(dict.fromkeys(variants[index] for index in indices))
            compatible_variants = [
                candidate
                for candidate in candidate_variants
                if all(
                    any(
                        all(
                            constraint.is_feasible(self._node_info_from_graph(graph, index), candidate, rank)
                            for constraint in self._hard_constraints
                            if constraint is not self._moe_constraint
                        )
                        for rank in rank_candidates
                    )
                    for index in indices
                )
            ]
            if not compatible_variants:
                for index in indices:
                    placement[index] = 0.0
                    ranks[index] = 0
                continue
            chosen_variant = compatible_variants[0]
            for index in indices:
                variants[index] = chosen_variant
            common_ranks = [
                rank
                for rank in rank_candidates
                if all(
                    all(
                        constraint.is_feasible(self._node_info_from_graph(graph, index), chosen_variant, rank)
                        for constraint in self._hard_constraints
                        if constraint is not self._moe_constraint
                    )
                    for index in indices
                )
            ]
            if not common_ranks:
                for index in indices:
                    placement[index] = 0.0
                    ranks[index] = 0
                continue
            chosen_rank = min(
                common_ranks,
                key=lambda rank: (sum(abs(int(ranks[index].item()) - rank) for index in indices), rank),
            )
            for index in indices:
                ranks[index] = chosen_rank
        return placement, ranks, variants

    def evaluate_soft(
        self,
        graph: ComputationGraph,
        placement: torch.Tensor,
        ranks: torch.Tensor,
        variant: Union[str, Sequence[str]],
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """Evaluate soft constraints and return a dict of violation scalars.
        Positive values indicate violation; zero means satisfied.
        """
        variants = [variant] * graph.n_nodes if isinstance(variant, str) else list(variant)
        if len(variants) != graph.n_nodes:
            raise ValueError("variant list must have one entry per graph node")
        differentiable = bool(placement.requires_grad or ranks.requires_grad)
        total_penalties: Dict[str, Union[float, torch.Tensor]] = {}
        for c in self._soft_constraints:
            if differentiable and isinstance(c, BudgetConstraint):
                costs = []
                for index in range(graph.n_nodes):
                    cost = graph.estimate_params(index, ranks[index], variants[index])
                    if not isinstance(cost, torch.Tensor):
                        cost = torch.as_tensor(cost, dtype=placement.dtype, device=placement.device)
                    costs.append(cost)
                used = torch.sum(placement * torch.stack(costs))
                total_penalties[c.name] = torch.relu(used - c.max_params) / max(float(c.max_params), 1.0)
                continue
            total = 0.0
            for index in range(graph.n_nodes):
                node_info = self._node_info_from_graph(graph, index)
                rank = int(ranks[index].item()) if isinstance(ranks[index], torch.Tensor) else int(ranks[index])
                penalty = c.penalty(node_info, variants[index], rank)
                if differentiable:
                    total = total + placement[index] * penalty
                elif placement[index] > 0.5:
                    total += penalty
            total_penalties[c.name] = total * c.weight
        return total_penalties

    def get_budget_usage(
        self,
        graph: ComputationGraph,
        placement: torch.Tensor,
        ranks: torch.Tensor,
        variant: Union[str, Sequence[str]],
    ) -> int:
        """Sum of adapter parameters for all placed modules."""
        variants = [variant] * graph.n_nodes if isinstance(variant, str) else list(variant)
        if len(variants) != graph.n_nodes:
            raise ValueError("variant list must have one entry per graph node")
        total = 0
        for index in range(graph.n_nodes):
            if placement[index] > 0.5:
                total += int(graph.estimate_params(index, int(ranks[index].item()), variants[index]))
        return total

    # -- new per-node interface --

    def check_hard(self, node_info: NodeInfo, variant: str, rank: int) -> bool:
        """Return True iff ALL hard constraints are satisfied."""
        for c in self._hard_constraints:
            if not c.is_feasible(node_info, variant, rank):
                return False
        return True

    def is_rank_feasible(self, graph: ComputationGraph, idx: int, variant: str, rank: int) -> bool:
        """Return whether a concrete graph node, variant, and rank satisfy all hard constraints."""
        return self.check_hard(self._node_info_from_graph(graph, idx), variant, rank)

    def check_hard_with_reason(
        self, node_info: NodeInfo, variant: str, rank: int
    ) -> Tuple[bool, List[str]]:
        """Return (feasible, list_of_violated_constraint_names)."""
        violated: List[str] = []
        for c in self._hard_constraints:
            if not c.is_feasible(node_info, variant, rank):
                violated.append(c.name)
        return len(violated) == 0, violated

    def compute_penalty(self, node_info: NodeInfo, variant: str, rank: int) -> float:
        """Weighted sum of all soft-constraint penalties."""
        total = 0.0
        for c in self._soft_constraints:
            total += c.weight * c.penalty(node_info, variant, rank)
        return total

    def compute_penalty_breakdown(
        self, node_info: NodeInfo, variant: str, rank: int
    ) -> Dict[str, float]:
        """Per-constraint soft penalty decomposition."""
        return {
            c.name: c.weight * c.penalty(node_info, variant, rank)
            for c in self._soft_constraints
        }

    def get_budget_usage_per_node(self, node_info: NodeInfo, variant: str, rank: int) -> int:
        """Single adapter parameter count."""
        if self._budget_constraint is not None:
            return self._budget_constraint.get_usage(node_info, variant, rank)
        # Fallback: estimate from node_info fields
        return self._fallback_budget_usage(node_info, variant, rank)

    def update_budget(self, node_info: NodeInfo, variant: str, rank: int) -> None:
        """Incrementally update the running budget tally."""
        if self._budget_constraint is not None:
            self._budget_constraint.update_usage(node_info, variant, rank)

    def reset(self) -> None:
        """Reset all stateful constraints (budget and MoE)."""
        if self._budget_constraint is not None:
            self._budget_constraint.reset()
        if self._moe_constraint is not None:
            self._moe_constraint.reset()

    # -- helpers used by policy.py (via hasattr fallback) --

    def get_budget_per_node(self, graph: ComputationGraph, variant: str) -> torch.Tensor:
        """Per-node adapter cost at unit rank (used by policy.py)."""
        costs = torch.zeros(graph.n_nodes, dtype=torch.float32)
        for i in range(graph.n_nodes):
            node_info = self._node_info_from_graph(graph, i)
            costs[i] = float(self.get_budget_usage_per_node(node_info, variant, rank=1))
        return costs

    def get_deployment_weights(self, graph: ComputationGraph, profile: str = "onnx") -> torch.Tensor:
        """Per-node deployment penalty weight (used by policy.py).
        Returns 1.0 for nodes that would violate deployment constraints.
        """
        weights = torch.zeros(graph.n_nodes, dtype=torch.float32)
        # Find the deployment constraint if any
        deploy_c = None
        for c in self._constraints:
            if isinstance(c, DeploymentCompatibilityConstraint):
                deploy_c = c
                break
        if deploy_c is None:
            return weights
        # Temporarily swap platform to the requested profile
        original_platform = deploy_c.platform
        deploy_c.platform = profile.lower()
        for i in range(graph.n_nodes):
            node_info = self._node_info_from_graph(graph, i)
            # We don't know the variant yet; assume a generic check.
            # For ONNX/TensorRT, only lora is allowed.
            if not deploy_c.is_feasible(node_info, variant="lora", rank=1):
                weights[i] = 1.0
        deploy_c.platform = original_platform
        return weights

    def get_conflict_edges(self, graph: ComputationGraph) -> List[Tuple[int, int, float]]:
        """Return pairwise compatibility conflict edges (used by policy.py).
        Currently empty placeholder; conflicts are handled by hard masks.
        """
        return []

    # -- internal helpers --

    @staticmethod
    def _node_info_from_graph(graph: ComputationGraph, idx: int) -> NodeInfo:
        """Build a NodeInfo from graph index."""
        if graph.nodes and idx < len(graph.nodes):
            return NodeInfo(graph.nodes[idx])
        # Legacy path: try to build from modules list
        if hasattr(graph, "modules") and graph.modules and idx < len(graph.modules):
            return NodeInfo(graph.modules[idx])
        # Fallback: build minimal NodeInfo from graph metadata
        name = graph.get_module_names()[idx] if idx < len(graph.get_module_names()) else ""
        op_type = graph.get_module_types()[idx] if idx < len(graph.get_module_types()) else "Other"
        return NodeInfo(name=name, operator_type=op_type)

    @staticmethod
    def _fallback_budget_usage(node_info: NodeInfo, variant: str, rank: int) -> int:
        """Fallback parameter estimation when no BudgetConstraint is registered."""
        return int(_estimate_adapter_params(
            rank,
            variant,
            node_info.operator_type,
            node_info.in_channels,
            node_info.out_channels,
            node_info.kernel_size,
            node_info.groups,
        ))
