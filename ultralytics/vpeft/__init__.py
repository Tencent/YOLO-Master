"""V-PEFT Compiler — Constraint-aware Optimization Solver Framework."""

from .solver import (
    PlacementDecision,
    ConstraintSolver,
    AlternatingOptimizationSolver,
    DifferentiableOptimizationSolver,
    MIPRelaxationSolver,
)
from .graph import (
    ComputationGraph,
    ModuleNode,
    NodeAttributes,
    GraphNode,
    GraphEdge,
    ComputationGraphBuilder,
    GATv2ArchitectureEncoder,
)
from .constraints import (
    ConstraintRegistry,
    BudgetConstraint,
    Constraint,
    NodeInfo,
    OperatorCompatibilityConstraint,
    SemanticProtectionConstraint,
    DeploymentCompatibilityConstraint,
    VariantModuleCompatibilityConstraint,
    MoEConsistencyConstraint,
    DivisibilityConstraint,
)
from .policy import (
    PlacementPolicy,
    RankAllocator,
    SoftRankAllocator,
    GreedyRankAllocator,
    RLRankAllocator,
    HybridTrainingProtocol,
    SEMANTIC_UTILITY,
    RANK_SET,
)

# MoE-aware Dynamic Adapter (Module 5)
from .moe_adapter import (
    DynamicAdapterExpert,
    DynamicAdapterLayer,
    DynamicAdapterModel,
    DynamicMoEConstraint,
    get_peft_dynamic_molora_model,
)

__all__ = [
    # Solver
    "PlacementDecision",
    "ConstraintSolver",
    "AlternatingOptimizationSolver",
    "DifferentiableOptimizationSolver",
    "MIPRelaxationSolver",
    # Graph representation (Module 1)
    "NodeAttributes",
    "GraphNode",
    "GraphEdge",
    "ComputationGraph",
    "ModuleNode",
    "ComputationGraphBuilder",
    "GATv2ArchitectureEncoder",
    # Constraints
    "ConstraintRegistry",
    "BudgetConstraint",
    "Constraint",
    "NodeInfo",
    "OperatorCompatibilityConstraint",
    "SemanticProtectionConstraint",
    "DeploymentCompatibilityConstraint",
    "VariantModuleCompatibilityConstraint",
    "MoEConsistencyConstraint",
    "DivisibilityConstraint",
    # Policy
    "RankAllocator",
    "SoftRankAllocator",
    "GreedyRankAllocator",
    "RLRankAllocator",
    "HybridTrainingProtocol",
    "SEMANTIC_UTILITY",
    "RANK_SET",
    # Dynamic Adapter (MoE)
    "DynamicAdapterExpert",
    "DynamicAdapterLayer",
    "DynamicAdapterModel",
    "DynamicMoEConstraint",
    "get_peft_dynamic_molora_model",
]
