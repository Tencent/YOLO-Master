"""
V-PEFT Solver Module — Constraint-aware Optimization Framework for AAAI 2026.

Implements three solvers for the combinatorial PEFT placement problem:
1. AlternatingOptimizationSolver (AO) — block-coordinate ascent with greedy sub-routines.
2. DifferentiableOptimizationSolver (DCO) — end-to-end continuous relaxation + dual ascent.
3. MIPRelaxationSolver (MIPR) — exact/offline MIP via OR-Tools with iterative rounding fallback.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import ComputationGraph
from .constraints import ConstraintRegistry
from .policy import PlacementPolicy, GreedyRankAllocator

__all__ = [
    "PlacementDecision",
    "ConstraintSolver",
    "AlternatingOptimizationSolver",
    "DifferentiableOptimizationSolver",
    "MIPRelaxationSolver",
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _utility_per_rank(rank: float, rank_max: int = 64) -> float:
    """Marginal utility f(r) = log2(r) / log2(r_max)."""
    if rank <= 0:
        return 0.0
    return math.log2(rank) / math.log2(rank_max)


def _softplus_penalty(x: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
    """Smooth constraint penalty: (1/beta) * log(1 + exp(beta * x))."""
    return (1.0 / beta) * torch.log1p(torch.exp(beta * x))


# ---------------------------------------------------------------------------
# PlacementDecision
# ---------------------------------------------------------------------------

@dataclass
class PlacementDecision:
    """Unified output container for all ConstraintSolver implementations."""

    status: str
    """One of {'ACCEPT', 'ADAPT', 'REFUSE'}."""

    placement: torch.Tensor
    """Binary tensor of shape [N] with values in {0, 1}."""

    ranks: torch.Tensor
    """Integer rank tensor of shape [N] with values in R ∪ {0}."""

    variant: str
    """Selected PEFT variant (global default or per-module if optimized)."""

    budget_used: int
    """Total adapter parameters consumed."""

    budget_remaining: int
    """Budget - budget_used."""

    target_modules: List[str]
    """Names of modules where placement[i] == 1."""

    reason: str
    """Human-readable refusal / adaptation reason."""

    utility: float
    """Total objective value U(π, r, ξ)."""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ConstraintSolver(ABC):
    """Abstract base for all V-PEFT constraint-aware optimizers."""

    @abstractmethod
    def solve(
        self,
        graph: ComputationGraph,
        budget: int,
        variant: str,
        constraints: ConstraintRegistry,
    ) -> PlacementDecision:
        """
        Solve the constrained PEFT placement problem.

        Args:
            graph: Model computation graph with node attributes and embeddings.
            budget: Maximum trainable adapter parameters (e.g. 2_100_000).
            variant: Requested PEFT variant (e.g. 'lora', 'dora').
            constraints: Registry of hard / soft constraints.

        Returns:
            PlacementDecision with discrete π, r, and metadata.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Alternating Optimization (AO)
# ---------------------------------------------------------------------------

class AlternatingOptimizationSolver(ConstraintSolver):
    """
    Block-coordinate ascent solver.

    Iterates:
      1. Fix (r, ξ) → optimize π by greedy density sorting + hard mask.
      2. Fix (π, ξ) → optimize r by GreedyRankAllocator.
      3. Fix (π, r) → optimize ξ by enumerating a small candidate set per module.
      4. Dual ascent on soft-constraint multipliers.
    """

    def __init__(
        self,
        max_iter: int = 15,
        tol: float = 1e-4,
        dual_lr: float = 0.01,
        rank_min: int = 4,
        rank_max: int = 64,
        rank_step: int = 4,
    ):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.dual_lr = dual_lr
        self.rank_min = rank_min
        self.rank_max = rank_max
        self.rank_step = rank_step
        self.rank_set = list(range(rank_min, rank_max + 1, rank_step))
        self._rank_allocator = GreedyRankAllocator(
            rank_set=self.rank_set, r_max=self.rank_max
        )

    # ------------------------------------------------------------------
    # Sub-routines
    # ------------------------------------------------------------------

    def _compute_objective(
        self,
        graph: ComputationGraph,
        pi: torch.Tensor,
        r: torch.Tensor,
        utilities: torch.Tensor,
    ) -> float:
        total = 0.0
        for i in range(graph.n_nodes):
            if pi[i] > 0.5 and r[i] > 0:
                total += utilities[i].item() * _utility_per_rank(
                    r[i].item(), self.rank_max
                )
        return total

    def _optimize_pi(
        self,
        graph: ComputationGraph,
        r: torch.Tensor,
        xi: List[str],
        budget: int,
        utilities: torch.Tensor,
        hard_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Greedy placement by utility density under fixed ranks."""
        n = graph.n_nodes
        scores = torch.full((n,), float("-inf"))
        for i in range(n):
            if not hard_mask[i] or r[i] <= 0:
                continue
            cost = graph.estimate_params(i, r[i].item(), xi[i])
            if cost <= 0:
                continue
            util = utilities[i].item() * _utility_per_rank(r[i].item(), self.rank_max)
            scores[i] = util / cost

        sorted_idx = torch.argsort(scores, descending=True)
        pi_new = torch.zeros(n)
        used = 0
        for idx in sorted_idx:
            if scores[idx] == float("-inf"):
                break
            cost = graph.estimate_params(idx.item(), r[idx].item(), xi[idx])
            if used + cost <= budget:
                pi_new[idx] = 1.0
                used += cost
        return pi_new

    def _optimize_xi(
        self,
        graph: ComputationGraph,
        pi: torch.Tensor,
        r: torch.Tensor,
        utilities: torch.Tensor,
        constraints: ConstraintRegistry,
        current_xi: List[str],
    ) -> List[str]:
        """Local enumeration of a small candidate variant set per module."""
        n = graph.n_nodes
        all_variants = ["lora", "dora", "loha", "lokr", "ia3", "oft", "boft", "hra"]
        # Small candidate set: current + two common variants
        candidates = list(set([current_xi[0]] + ["lora", "ia3"]))

        xi_new = list(current_xi)
        for i in range(n):
            if pi[i] < 0.5:
                continue
            best_v = xi_new[i]
            best_score = -1e9
            for v in candidates:
                if not constraints.get_hard_mask(graph, v)[i]:
                    continue
                cost = graph.estimate_params(i, r[i].item(), v)
                if cost <= 0:
                    continue
                score = utilities[i].item() / cost
                if score > best_score:
                    best_score = score
                    best_v = v
            xi_new[i] = best_v
        return xi_new

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(
        self,
        graph: ComputationGraph,
        budget: int,
        variant: str,
        constraints: ConstraintRegistry,
    ) -> PlacementDecision:
        n = graph.n_nodes
        utilities = graph.get_node_importances()
        hard_mask = constraints.get_hard_mask(graph, variant).bool()

        # --- initialisation ------------------------------------------------
        pi = torch.zeros(n)
        r = torch.zeros(n, dtype=torch.long)
        xi = [variant] * n

        feasible_idx = torch.where(hard_mask)[0]
        for i in feasible_idx:
            pi[i] = 1.0
            r[i] = self.rank_min

        # Project to budget (greedy drop if over)
        used = constraints.get_budget_usage(graph, pi, r, variant)
        if used > budget:
            sorted_idx = feasible_idx[torch.argsort(utilities[feasible_idx])]
            for i in sorted_idx:
                if used <= budget:
                    break
                used -= graph.estimate_params(i.item(), r[i].item(), variant)
                pi[i] = 0.0
                r[i] = 0

        # Dual variables for budget / latency / memory / deploy
        soft_keys = ["budget", "latency", "memory", "deploy"]
        lambda_dual: Dict[str, float] = {k: 0.0 for k in soft_keys}

        # --- alternating optimisation loop ---------------------------------
        for iteration in range(self.max_iter):
            pi_prev = pi.clone()
            r_prev = r.clone()
            xi_prev = list(xi)

            # Step 1: fix (r, ξ) → optimise π
            pi = self._optimize_pi(graph, r, xi, budget, utilities, hard_mask)

            # Step 2: fix (π, ξ) → optimise r
            r = self._rank_allocator.allocate(graph, pi, budget, variant)

            # Step 3: fix (π, r) → optimise ξ (local enumeration)
            xi = self._optimize_xi(graph, pi, r, utilities, constraints, xi)

            # Step 4: dual ascent on soft constraints
            soft_violations = constraints.evaluate_soft(graph, pi, r, variant)
            for key in soft_keys:
                violation = soft_violations.get(key, 0.0)
                if key == "budget":
                    used_now = constraints.get_budget_usage(graph, pi, r, variant)
                    violation = max(0.0, used_now - budget)
                lambda_dual[key] = max(0.0, lambda_dual[key] + self.dual_lr * violation)

            # Convergence check: L1(π) + L2(r) + Hamming(ξ)
            delta_pi = torch.sum(torch.abs(pi - pi_prev)).item()
            delta_r = torch.sum(torch.abs(r.float() - r_prev.float())).item()
            delta_xi = sum(1 for a, b in zip(xi, xi_prev) if a != b)
            if delta_pi + delta_r + delta_xi < self.tol:
                break

        # --- post-processing -----------------------------------------------
        budget_used = int(constraints.get_budget_usage(graph, pi, r, variant))
        budget_remaining = max(0, budget - budget_used)
        target_modules = [
            graph.get_module_names()[i] for i in range(n) if pi[i] > 0.5
        ]
        utility = self._compute_objective(graph, pi, r, utilities)

        if len(target_modules) == 0:
            status = "REFUSE"
            reason = "No feasible modules found under hard constraints / budget."
        elif budget_used > budget:
            status = "ADAPT"
            reason = (
                f"Budget exceeded ({budget_used} > {budget}); "
                "solution was adapted by dropping low-utility modules."
            )
        else:
            status = "ACCEPT"
            reason = ""

        return PlacementDecision(
            status=status,
            placement=pi,
            ranks=r,
            variant=variant,
            budget_used=budget_used,
            budget_remaining=budget_remaining,
            target_modules=target_modules,
            reason=reason,
            utility=utility,
        )


# ---------------------------------------------------------------------------
# 2. Differentiable Constraint Optimization (DCO)
# ---------------------------------------------------------------------------

class DifferentiableOptimizationSolver(ConstraintSolver):
    """
    End-to-end differentiable solver.

    Continuous relaxation:
      π̂ = σ(MLP_place(features))                (placement)
      r̂ = r_min + (r_max - r_min) · σ(MLP_rank) (rank)
      ξ = GumbelSoftmax(MLP_variant)            (variant, optional)

    Maximises the Lagrangian:
      L = U - Σ_j λ_j · softplus(C_j - ε_j)
    via gradient ascent (Adam) on the MLP parameters, with dual ascent on λ.
    """

    def __init__(
        self,
        max_iter: int = 200,
        lr: float = 0.05,
        dual_lr: float = 0.01,
        beta_softplus: float = 10.0,
        tau_gumbel: float = 0.5,
        rank_min: int = 4,
        rank_max: int = 64,
        rank_step: int = 4,
        variant_candidates: Optional[List[str]] = None,
        optimize_variant: bool = False,
    ):
        super().__init__()
        self.max_iter = max_iter
        self.lr = lr
        self.dual_lr = dual_lr
        self.beta_softplus = beta_softplus
        self.tau_gumbel = tau_gumbel
        self.rank_min = rank_min
        self.rank_max = rank_max
        self.rank_step = rank_step
        self.rank_set = list(range(rank_min, rank_max + 1, rank_step))
        self.optimize_variant = optimize_variant
        self.variant_candidates = variant_candidates or [
            "lora",
            "dora",
            "loha",
            "lokr",
            "ia3",
            "oft",
            "boft",
            "hra",
        ]
        self.n_variants = len(self.variant_candidates)

        # Small MLPs that map 1-D utility to placement / rank logits.
        # In a full training loop these would be replaced by pre-trained networks.
        self._placement_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self._rank_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        if self.optimize_variant:
            self._variant_mlp = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, self.n_variants),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_objective(
        self,
        graph: ComputationGraph,
        pi: torch.Tensor,
        r: torch.Tensor,
        utilities: torch.Tensor,
    ) -> float:
        total = 0.0
        for i in range(graph.n_nodes):
            if pi[i] > 0.5 and r[i] > 0:
                total += utilities[i].item() * _utility_per_rank(
                    r[i].item(), self.rank_max
                )
        return total

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(
        self,
        graph: ComputationGraph,
        budget: int,
        variant: str,
        constraints: ConstraintRegistry,
    ) -> PlacementDecision:
        n = graph.n_nodes
        utilities = graph.get_node_importances()
        hard_mask = constraints.get_hard_mask(graph, variant).bool()

        # Feature vector: normalised utility per node (kept 1-D for the stub MLPs)
        feat = utilities.view(-1, 1)  # [N, 1]

        soft_keys = ["budget", "latency", "memory", "deploy"]
        lambda_dual = {k: 0.0 for k in soft_keys}
        epsilon = {k: 0.0 for k in soft_keys}  # slack thresholds
        epsilon["budget"] = budget

        if self.optimize_variant:
            params = (
                list(self._placement_mlp.parameters())
                + list(self._rank_mlp.parameters())
                + list(self._variant_mlp.parameters())
            )
        else:
            params = (
                list(self._placement_mlp.parameters())
                + list(self._rank_mlp.parameters())
            )

        optimizer = torch.optim.Adam(params, lr=self.lr)

        # --- gradient optimisation loop ------------------------------------
        for iteration in range(self.max_iter):
            optimizer.zero_grad()

            # Recompute forward pass (MLP parameters change each step)
            pi_logits = self._placement_mlp(feat).squeeze(-1)  # [N]
            r_raw = self._rank_mlp(feat).squeeze(-1)           # [N]

            if self.optimize_variant:
                xi_logits = self._variant_mlp(feat)  # [N, n_variants]

            # Continuous relaxation of placement
            pi_hat = torch.sigmoid(pi_logits) * hard_mask.float()

            # Continuous rank in [rank_min, rank_max]
            r_cont = self.rank_min + (self.rank_max - self.rank_min) * torch.sigmoid(
                r_raw
            )

            # Variant relaxation (Gumbel-Softmax if multiple candidates)
            if self.optimize_variant and xi_logits is not None:
                xi_soft = F.gumbel_softmax(
                    xi_logits, tau=self.tau_gumbel, hard=False
                )  # [N, n_variants]
            else:
                xi_soft = None

            # Objective U = Σ_i u_i · π̂_i · f(r_i)
            f_r = torch.log2(r_cont + 1e-6) / math.log2(self.rank_max)
            U = torch.sum(utilities * pi_hat * f_r)

            # Budget penalty (differentiable)
            costs = []
            for i in range(n):
                if self.optimize_variant and xi_soft is not None:
                    cost_i = sum(
                        xi_soft[i, v_idx] * graph.estimate_params(i, r_cont[i], v)
                        for v_idx, v in enumerate(self.variant_candidates)
                    )
                else:
                    cost_i = graph.estimate_params(i, r_cont[i], variant)
                costs.append(cost_i)
            costs = torch.stack(costs)  # [N]
            budget_used = torch.sum(pi_hat * costs)

            penalty = _softplus_penalty(
                budget_used - epsilon["budget"], self.beta_softplus
            )

            # Other soft constraints (via registry)
            soft_violations = constraints.evaluate_soft(graph, pi_hat, r_cont, variant)
            total_penalty = penalty
            for key in soft_keys:
                if key == "budget":
                    continue
                val = soft_violations.get(key, 0.0)
                if isinstance(val, (int, float)):
                    total_penalty = total_penalty + _softplus_penalty(
                        torch.tensor(float(val)) - epsilon[key], self.beta_softplus
                    )

            # Hard-constraint projection loss (penalise placement on forbidden nodes)
            L_hard = torch.sum(pi_hat * (1.0 - hard_mask.float()))

            # Lagrangian = -U + λᵀ · penalty + μ · L_hard
            # We minimise the negative Lagrangian (Adam is a minimiser)
            lagrangian = -U + sum(lambda_dual.values()) * total_penalty + 10.0 * L_hard

            lagrangian.backward()
            optimizer.step()

            # Dual ascent every 10 steps
            if iteration % 10 == 0 and iteration > 0:
                for key in soft_keys:
                    if key == "budget":
                        violation = max(0.0, budget_used.item() - budget)
                    else:
                        violation = max(0.0, soft_violations.get(key, 0.0))
                    lambda_dual[key] = max(
                        0.0, lambda_dual[key] + self.dual_lr * violation
                    )

        # --- discretisation ------------------------------------------------
        # Recompute one last time to get final continuous values
        with torch.no_grad():
            pi_logits = self._placement_mlp(feat).squeeze(-1)
            pi_hat = torch.sigmoid(pi_logits) * hard_mask.float()
            r_raw = self._rank_mlp(feat).squeeze(-1)
            r_cont = self.rank_min + (self.rank_max - self.rank_min) * torch.sigmoid(r_raw)
            if self.optimize_variant:
                xi_logits = self._variant_mlp(feat)
                xi_soft = F.gumbel_softmax(xi_logits, tau=self.tau_gumbel, hard=False)
            else:
                xi_soft = None

        pi_discrete = (pi_hat > 0.5).float()
        pi_discrete = pi_discrete * hard_mask.float()

        r_discrete = torch.round(r_cont / self.rank_step) * self.rank_step
        r_discrete = torch.clamp(
            r_discrete, min=self.rank_min, max=self.rank_max
        ).long()

        if self.optimize_variant and xi_soft is not None:
            xi_indices = torch.argmax(xi_soft, dim=-1)
            xi_selected = [self.variant_candidates[i] for i in xi_indices.tolist()]
            # Use the most common variant for the global field
            from collections import Counter
            variant_global = Counter(xi_selected).most_common(1)[0][0]
        else:
            xi_selected = [variant] * n
            variant_global = variant

        # Post-process: ensure budget is respected by dropping lowest-utility placed modules
        used = int(constraints.get_budget_usage(graph, pi_discrete, r_discrete, variant_global))
        if used > budget:
            placed = (pi_discrete > 0.5).nonzero(as_tuple=True)[0]
            sorted_by_util = placed[torch.argsort(utilities[placed])]
            for i in sorted_by_util:
                if used <= budget:
                    break
                used -= graph.estimate_params(i.item(), r_discrete[i].item(), variant_global)
                pi_discrete[i] = 0.0
                r_discrete[i] = 0

        budget_used = int(constraints.get_budget_usage(graph, pi_discrete, r_discrete, variant_global))
        budget_remaining = max(0, budget - budget_used)
        target_modules = [
            graph.get_module_names()[i] for i in range(n) if pi_discrete[i] > 0.5
        ]
        utility = self._compute_objective(graph, pi_discrete, r_discrete, utilities)

        if len(target_modules) == 0:
            status = "REFUSE"
            reason = "DCO produced zero placements; constraints may be too restrictive."
        elif budget_used > budget:
            status = "ADAPT"
            reason = f"DCO discretisation exceeded budget ({budget_used} > {budget})."
        else:
            status = "ACCEPT"
            reason = ""

        return PlacementDecision(
            status=status,
            placement=pi_discrete,
            ranks=r_discrete,
            variant=variant_global,
            budget_used=budget_used,
            budget_remaining=budget_remaining,
            target_modules=target_modules,
            reason=reason,
            utility=utility,
        )


# ---------------------------------------------------------------------------
# 3. MIP Relaxation (MIPR)
# ---------------------------------------------------------------------------

class MIPRelaxationSolver(ConstraintSolver):
    """
    Offline exact / near-exact solver via Mixed-Integer Programming.

    Primary engine: OR-Tools SCIP.
    Fallback (if OR-Tools is installed but infeasible / time-limit):
        greedy + iterative rounding (Algorithm 3 from design doc).

    If OR-Tools is **not** installed, ``solve()`` raises ``ImportError``
    and recommends ``AlternatingOptimizationSolver``.
    """

    def __init__(
        self,
        rank_set: Optional[List[int]] = None,
        rank_max: int = 64,
        time_limit_ms: int = 10_000,
    ):
        super().__init__()
        if rank_set is None:
            rank_set = [4, 8, 12, 16, 32, 64]
        self.rank_set = sorted(rank_set)
        self.rank_max = rank_max
        self.time_limit_ms = time_limit_ms

    # ------------------------------------------------------------------
    # Fallback: iterative rounding
    # ------------------------------------------------------------------

    def _iterative_rounding_fallback(
        self,
        graph: ComputationGraph,
        budget: int,
        variant: str,
        constraints: ConstraintRegistry,
    ) -> PlacementDecision:
        """Greedy + iterative rounding when MIP solver fails or is unavailable."""
        n = graph.n_nodes
        utilities = graph.get_node_importances()
        hard_mask = constraints.get_hard_mask(graph, variant).bool()

        pi = torch.zeros(n)
        r = torch.zeros(n, dtype=torch.long)

        # Greedy initial solution (continuous relaxation → top-k by utility / cost)
        scores = torch.full((n,), float("-inf"))
        for i in range(n):
            if not hard_mask[i]:
                continue
            cost = graph.estimate_params(i, self.rank_set[0], variant)
            scores[i] = utilities[i].item() / (cost + 1e-8)

        sorted_idx = torch.argsort(scores, descending=True)
        used = 0
        for idx in sorted_idx:
            if scores[idx] == float("-inf"):
                break
            cost = graph.estimate_params(idx.item(), self.rank_set[0], variant)
            if used + cost <= budget:
                pi[idx] = 1.0
                r[idx] = self.rank_set[0]
                used += cost

        # Iterative rounding: fix most-certain variables and re-optimise
        S_fixed: set = set()
        for _ in range(min(n, 50)):
            unfixed = [i for i in range(n) if i not in S_fixed]
            if not unfixed:
                break
            best_i = max(unfixed, key=lambda i: abs(pi[i].item() - 0.5))
            pi[best_i] = 1.0 if pi[best_i] > 0.5 else 0.0
            S_fixed.add(best_i)

            # Re-optimise ranks for all placed modules with remaining budget
            remaining = budget - sum(
                graph.estimate_params(i, r[i].item(), variant)
                for i in range(n)
                if pi[i] > 0.5
            )
            for i in range(n):
                if pi[i] > 0.5 and i not in S_fixed:
                    for rank_val in reversed(self.rank_set):
                        extra = (
                            graph.estimate_params(i, rank_val, variant)
                            - graph.estimate_params(i, r[i].item(), variant)
                        )
                        if extra <= remaining:
                            remaining -= extra
                            r[i] = rank_val
                            break

        # Final greedy upgrade pass (same as AO rank allocator)
        allocator = GreedyRankAllocator(
            rank_set=self.rank_set, r_max=self.rank_max
        )
        r = allocator.allocate(graph, pi, budget, variant)

        budget_used = int(constraints.get_budget_usage(graph, pi, r, variant))
        budget_remaining = max(0, budget - budget_used)
        target_modules = [
            graph.get_module_names()[i] for i in range(n) if pi[i] > 0.5
        ]
        utility = sum(
            utilities[i].item() * _utility_per_rank(r[i].item(), self.rank_max)
            for i in range(n)
            if pi[i] > 0.5 and r[i] > 0
        )

        if len(target_modules) == 0:
            status = "REFUSE"
            reason = "MIP fallback produced zero placements."
        elif budget_used > budget:
            status = "ADAPT"
            reason = f"MIP fallback exceeded budget ({budget_used} > {budget})."
        else:
            status = "ACCEPT"
            reason = "MIP solver failed; fallback solution accepted."

        return PlacementDecision(
            status=status,
            placement=pi,
            ranks=r,
            variant=variant,
            budget_used=budget_used,
            budget_remaining=budget_remaining,
            target_modules=target_modules,
            reason=reason,
            utility=utility,
        )

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(
        self,
        graph: ComputationGraph,
        budget: int,
        variant: str,
        constraints: ConstraintRegistry,
    ) -> PlacementDecision:
        try:
            from ortools.linear_solver import pywraplp
        except ImportError as exc:
            raise ImportError(
                "OR-Tools is not installed. MIPRelaxationSolver requires "
                "OR-Tools (`pip install ortools`). Consider using "
                "AlternatingOptimizationSolver as a lightweight fallback."
            ) from exc

        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            return self._iterative_rounding_fallback(
                graph, budget, variant, constraints
            )

        n = graph.n_nodes
        utilities = graph.get_node_importances().tolist()
        hard_mask = constraints.get_hard_mask(graph, variant).bool().tolist()
        m = len(self.rank_set)

        # Variables
        pi_vars = [solver.IntVar(0, 1, f"pi_{i}") for i in range(n)]
        y_vars: Dict[int, List] = {}
        w_vars: Dict[tuple, any] = {}
        for i in range(n):
            y_vars[i] = []
            for k, r_val in enumerate(self.rank_set):
                y_ik = solver.IntVar(0, 1, f"y_{i}_{k}")
                y_vars[i].append(y_ik)
                w_ik = solver.IntVar(0, 1, f"w_{i}_{k}")
                w_vars[(i, k)] = w_ik
                solver.Add(w_ik <= pi_vars[i])
                solver.Add(w_ik <= y_ik)
                solver.Add(w_ik >= pi_vars[i] + y_ik - 1)
            solver.Add(sum(y_vars[i]) <= 1)
            if not hard_mask[i]:
                solver.Add(pi_vars[i] == 0)

        # Budget constraint
        budget_expr = sum(
            graph.estimate_params(i, r_val, variant) * w_vars[(i, k)]
            for i in range(n)
            for k, r_val in enumerate(self.rank_set)
        )
        solver.Add(budget_expr <= budget)

        # Objective
        obj_expr = sum(
            utilities[i]
            * (math.log2(r_val) / math.log2(self.rank_max))
            * w_vars[(i, k)]
            for i in range(n)
            for k, r_val in enumerate(self.rank_set)
        )
        solver.Maximize(obj_expr)

        solver.set_time_limit(self.time_limit_ms)
        status = solver.Solve()

        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return self._iterative_rounding_fallback(
                graph, budget, variant, constraints
            )

        pi_vals = torch.tensor([pi_vars[i].solution_value() for i in range(n)])
        ranks = torch.zeros(n, dtype=torch.long)
        for i in range(n):
            if pi_vals[i] > 0.5:
                for k, r_val in enumerate(self.rank_set):
                    if y_vars[i][k].solution_value() > 0.5:
                        ranks[i] = r_val
                        break

        budget_used = int(constraints.get_budget_usage(graph, pi_vals, ranks, variant))
        budget_remaining = max(0, budget - budget_used)
        target_modules = [
            graph.get_module_names()[i] for i in range(n) if pi_vals[i] > 0.5
        ]
        utility = sum(
            utilities[i] * _utility_per_rank(ranks[i].item(), self.rank_max)
            for i in range(n)
            if pi_vals[i] > 0.5 and ranks[i] > 0
        )

        if len(target_modules) == 0:
            status_str = "REFUSE"
            reason = "MIP solver returned an empty placement."
        elif budget_used > budget:
            status_str = "ADAPT"
            reason = f"MIP solution slightly exceeds budget ({budget_used} > {budget})."
        else:
            status_str = "ACCEPT"
            reason = ""

        return PlacementDecision(
            status=status_str,
            placement=pi_vals,
            ranks=ranks,
            variant=variant,
            budget_used=budget_used,
            budget_remaining=budget_remaining,
            target_modules=target_modules,
            reason=reason,
            utility=utility,
        )
