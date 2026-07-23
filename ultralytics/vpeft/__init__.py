"""V-PEFT Compiler - constraint-aware optimization solver framework.

The graph, policy, constraint, and solver APIs are usable today. The dynamic
MoE adapter proposed for the research track is intentionally not exported
until its implementation and integration contracts are complete.
"""

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
    CandidateTargetConstraint,
)
from .policy import (
    PlacementPolicy,
    RankAllocator,
    SoftRankAllocator,
    GreedyRankAllocator,
    RLRankAllocator,
    PPORankTrajectory,
    HybridTrainingProtocol,
    SEMANTIC_UTILITY,
    RANK_SET,
)
from .placement_plan import PlacementPlan, PlacementTarget

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
    "CandidateTargetConstraint",
    # Policy
    "PlacementPolicy",
    "RankAllocator",
    "SoftRankAllocator",
    "GreedyRankAllocator",
    "RLRankAllocator",
    "PPORankTrajectory",
    "HybridTrainingProtocol",
    "SEMANTIC_UTILITY",
    "RANK_SET",
    "PlacementPlan",
    "PlacementTarget",
]
