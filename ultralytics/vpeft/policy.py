"""V-PEFT Compiler — Module 3: Placement Policy + Rank Allocation.

Contains:
  - PlacementPolicy     : neural per-module placement with hard/soft constraints
  - RankAllocator (ABC) : abstract rank-allocation interface
  - SoftRankAllocator   : continuous-relaxation + Gaussian soft-projection
  - GreedyRankAllocator : utility-greedy discrete allocation
  - RLRankAllocator     : PPO-based sequential allocator (placeholder)
  - HybridTrainingProtocol : SL warm-start + RL fine-tuning entry points

All implementations follow `method_placement.md` and `method_rank_allocation.md`.
Target venue: AAAI 2026.

Backward-compatibility notes:
  - GreedyRankAllocator.allocate() accepts an optional ``utilities`` kwarg so
    that `solver.py` (which passes it explicitly) continues to work unchanged.
  - Graph API uses the existing `ComputationGraph.estimate_params(idx, rank, variant)`
    and `GraphNode.name` heuristics instead of a custom `params_for_rank` method.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER

from .graph import ComputationGraph, GraphNode, GATv2ArchitectureEncoder
from .constraints import ConstraintRegistry, BudgetConstraint

__all__ = [
    "PlacementPolicy",
    "RankAllocator",
    "SoftRankAllocator",
    "GreedyRankAllocator",
    "RLRankAllocator",
    "HybridTrainingProtocol",
    "SEMANTIC_UTILITY",
    "RANK_SET",
    "r_utility_fn",
]


# ═══════════════════════════════════════════════════════════════════════════
# Constants & helpers
# ═══════════════════════════════════════════════════════════════════════════

# Phase-1 fixed unit-rank utilities: semantic role → u_i
SEMANTIC_UTILITY: Dict[str, float] = {
    "backbone": 0.5,
    "neck": 0.8,
    "head": 1.0,
    "attention": 1.2,
}

# Allowed discrete rank set  R = {4, 8, 12, 16, 32, 64}
RANK_SET: List[int] = [4, 8, 12, 16, 32, 64]
_RANK_TENSOR = torch.tensor(RANK_SET, dtype=torch.float32)


def _infer_semantic_role_from_name(name: str) -> str:
    """Infer semantic role from module name (fallback when GraphNode has no role)."""
    lname = name.lower()
    if any(k in lname for k in ("head", "detect", "segment", "pose", "obb", "v10detect", "yoloedetect", "cls", "box", "pred")):
        return "head"
    if any(k in lname for k in ("neck", "fpn", "pan", "upsample", "concat")):
        return "neck"
    if any(k in lname for k in ("attention", "attn", "aattn", "mhsa")):
        return "attention"
    return "backbone"


def r_utility_fn(r: Union[int, float, torch.Tensor], r_max: int = 64) -> Union[float, torch.Tensor]:
    """Marginal utility of rank: f(r) = log2(r) / log2(r_max).

    Args:
        r: Rank value (int, float, or tensor).
        r_max: Maximum rank in the allowed set (default 64).

    Returns:
        Scalar or tensor utility in (0, 1].
    """
    if isinstance(r, torch.Tensor):
        return torch.log2(r) / math.log2(r_max)
    return math.log2(r) / math.log2(r_max)


# ═══════════════════════════════════════════════════════════════════════════
# 1. PlacementPolicy
# ═══════════════════════════════════════════════════════════════════════════

class PlacementPolicy(nn.Module):
    """Neural per-module placement policy with constraint projection.

    Architecture:
        Input  : [h_i || h_G || xi_p]   (node || global || variant)
        MLP    : 2d+k → 256 → 128 → 64 → 1
        Mask   : Hard feasibility mask M_i (product of binary indicators)
        Sigmoid: pi_hat_i = σ( (z_i·M_i + (1-M_i)·(-1e9)) / τ )

    Training objective (composite):
        L = L_placement
            + λ_budget · L_budget
            + λ_deploy · L_deploy
            + λ_compat · L_compat
            - λ_ent    · H(pi_hat)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_variants: int,
        constraint_registry: ConstraintRegistry,
        variant_feature_dim: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_variants = num_variants
        self.constraint_registry = constraint_registry
        self.variant_feature_dim = variant_feature_dim

        # ── MLP: raw placement logits z_i ──
        mlp_in = hidden_dim * 2 + variant_feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Annealable temperature (initialised to 1.0, sharpened during training)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        node_embeddings: torch.Tensor,  # [N, D]
        global_embedding: torch.Tensor,  # [D]
        variant_embedding: torch.Tensor,  # [V]  (V = variant_feature_dim)
    ) -> torch.Tensor:
        """Compute raw placement logits z_i for each node.

        Returns:
            z: [N] raw logits before masking / sigmoid.
        """
        N = node_embeddings.shape[0]
        h_g = global_embedding.view(-1).unsqueeze(0).expand(N, -1)   # [N, D]
        xi_p = variant_embedding.view(-1).unsqueeze(0).expand(N, -1)  # [N, V]
        x = torch.cat([node_embeddings, h_g, xi_p], dim=-1)  # [N, 2D+V]
        z = self.mlp(x).squeeze(-1)  # [N]
        return z

    def compute_constrained_probs(
        self,
        z: torch.Tensor,
        hard_mask: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """Apply hard-constraint mask and temperature-sharpened sigmoid.

        Math:
            tilde_z_i = z_i · M_i + (1 - M_i) · (-1e9)
            pi_hat_i  = σ( tilde_z_i / τ )

        Args:
            z: Raw logits [N].
            hard_mask: Binary feasibility mask [N] (1 = feasible, 0 = infeasible).
            temperature: Optional override for τ. Defaults to self.temperature.

        Returns:
            pi_hat: Constrained placement probabilities [N] in (0, 1).
        """
        tau = temperature if temperature is not None else self.temperature
        neg_inf = -1e9
        # Ensure M_i is float and on the same device as z
        M = hard_mask.to(z.dtype)
        tilde_z = z * M + (1.0 - M) * neg_inf
        pi_hat = torch.sigmoid(tilde_z / tau)
        return pi_hat

    def sample_placement(
        self, pi_hat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a discrete placement vector from Bernoulli(pi_hat).

        Returns:
            pi:      Binary placement decisions [N].
            log_prob: Scalar log-probability of the sampled vector.
        """
        dist = torch.distributions.Bernoulli(probs=pi_hat)
        pi = dist.sample()
        log_prob = dist.log_prob(pi).sum()
        return pi, log_prob

    def compute_loss(
        self,
        pi_hat: torch.Tensor,
        oracle_labels: Optional[torch.Tensor] = None,
        budget_per_node: Optional[torch.Tensor] = None,
        budget_max: Optional[float] = None,
        deployment_weights: Optional[torch.Tensor] = None,
        conflict_edges: Optional[List[Tuple[int, int, float]]] = None,
        lambda_budget: float = 1.0,
        lambda_deploy: float = 0.5,
        lambda_compat: float = 0.3,
        lambda_ent: float = 0.1,
        mode: str = "sl",
        reward: Optional[float] = None,
        log_prob: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the composite placement loss.

        Args:
            pi_hat: Constrained probabilities [N].
            oracle_labels: Ground-truth binary placements [N] (required for SL).
            budget_per_node: Per-node adapter parameter counts [N].
            budget_max: Global parameter budget B_max.
            deployment_weights: Per-node deployment penalty weights [N].
            conflict_edges: List of (i, j, kappa) conflict edges.
            lambda_budget, lambda_deploy, lambda_compat, lambda_ent:
                Constraint-loss coefficients.
            mode: "sl" or "rl".
            reward: Scalar episode reward (required for RL).
            log_prob: Scalar log-prob of sampled placement (required for RL).

        Returns:
            Scalar total loss.
        """
        device = pi_hat.device

        # ── Primary placement loss ──
        if mode == "sl":
            if oracle_labels is None:
                raise ValueError("SL mode requires `oracle_labels`.")
            L_placement = F.binary_cross_entropy(
                pi_hat, oracle_labels.to(pi_hat.dtype)
            )
        elif mode == "rl":
            if reward is None or log_prob is None:
                raise ValueError("RL mode requires `reward` and `log_prob`.")
            # REINFORCE with baseline (baseline = 0 for simplicity; caller may subtract)
            L_placement = -reward * log_prob
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'sl' or 'rl'.")

        # ── Budget penalty (squared hinge) ──
        if budget_per_node is not None and budget_max is not None:
            B_total = (pi_hat * budget_per_node.to(pi_hat.dtype)).sum()
            margin = torch.relu(B_total - budget_max)
            L_budget = (margin ** 2) / max(budget_max, 1.0)
        else:
            L_budget = torch.tensor(0.0, device=device)

        # ── Deployment compatibility penalty ──
        if deployment_weights is not None:
            w = deployment_weights.to(pi_hat.dtype)
            L_deploy = (w * pi_hat).sum() / max(pi_hat.numel(), 1)
        else:
            L_deploy = torch.tensor(0.0, device=device)

        # ── Pairwise compatibility penalty ──
        if conflict_edges is not None and len(conflict_edges) > 0:
            L_compat = 0.0
            for i, j, kappa in conflict_edges:
                L_compat += pi_hat[i] * pi_hat[j] * kappa
            L_compat = L_compat / max(len(conflict_edges), 1)
        else:
            L_compat = torch.tensor(0.0, device=device)

        # ── Entropy regularisation (encourages exploration) ──
        eps = 1e-8
        H = -(pi_hat * torch.log(pi_hat + eps) + (1 - pi_hat) * torch.log(1 - pi_hat + eps)).mean()

        total_loss = (
            L_placement
            + lambda_budget * L_budget
            + lambda_deploy * L_deploy
            + lambda_compat * L_compat
            - lambda_ent * H
        )
        return total_loss


# ═══════════════════════════════════════════════════════════════════════════
# 2. RankAllocator (ABC)
# ═══════════════════════════════════════════════════════════════════════════

class RankAllocator(ABC):
    """Abstract base class for rank-allocation strategies.

    Each concrete allocator receives a computation graph, a placement vector,
    a parameter budget, and a variant string, and returns a per-node rank.
    """

    @abstractmethod
    def allocate(
        self,
        graph: ComputationGraph,
        placement: torch.Tensor,
        budget: int,
        variant: str,
    ) -> torch.Tensor:
        """Allocate ranks to placed modules under a parameter budget.

        Args:
            graph: The target model's computation graph.
            placement: Binary placement decisions [N] (1 = adapt this node).
            budget: Total parameter budget B (trainable adapter params).
            variant: PEFT variant name (e.g., "lora", "dora", "ia3").

        Returns:
            r_alloc: [N] per-node rank allocation (continuous for Soft,
                     discrete long/ float for Greedy/RL).
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# 3. SoftRankAllocator  — Continuous Relaxation + Gaussian Projection
# ═══════════════════════════════════════════════════════════════════════════

class SoftRankAllocator(RankAllocator, nn.Module):
    """Differentiable rank allocator via continuous relaxation.

    For each node, a lightweight MLP predicts a continuous  hat_r_i  from the
    node embedding.  A Gaussian kernel soft-projects it onto the discrete set
    R = {4, 8, 12, 16, 32, 64}:

        w_r = exp( -(hat_r - r)^2 / (2σ^2) ) / Σ_r' exp( ... )
        r_i = Σ_r  w_r · r        (expectation, differentiable)

    At inference, the caller may argmax(w_r) for hard discretisation.
    """

    def __init__(
        self,
        hidden_dim: int,
        rank_set: Optional[List[int]] = None,
        sigma: float = 4.0,
    ):
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.rank_set = rank_set if rank_set is not None else RANK_SET[:]
        self.sigma = sigma
        self.sigma_sq = sigma ** 2

        # Rank predictor: small MLP → continuous hat_r_i
        self.rank_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        # Ensure positivity (rank must be > 0)
        self.rank_act = nn.Softplus()

    def allocate(
        self,
        graph: ComputationGraph,
        placement: torch.Tensor,
        budget: int,
        variant: str,
    ) -> torch.Tensor:
        """Return soft expected ranks [N].

        Budget enforcement is left to the caller (e.g. via the placement
        policy's L_budget penalty or a downstream hard projection step).
        """
        # SoftRankAllocator expects node embeddings to be passed directly or
        # pre-computed on the graph.  If the graph was built with GATv2, the
        # node embeddings are available via the encoder.  For a simple stub
        # we use the graph's node_importances as a proxy or raise if nothing
        # is available.
        if hasattr(graph, "node_embeddings") and graph.node_embeddings is not None:
            node_embeddings = graph.node_embeddings
        elif hasattr(graph, "_node_features") and graph._node_features is not None:
            node_embeddings = graph._node_features
        else:
            # Fallback: create dummy embeddings (for testing / cold-start)
            node_embeddings = torch.randn(graph.n_nodes, self.hidden_dim, dtype=torch.float32)

        N = node_embeddings.shape[0]

        # Continuous prediction hat_r_i
        hat_r = self.rank_predictor(node_embeddings).squeeze(-1)  # [N]
        hat_r = self.rank_act(hat_r)

        # Zero out for non-placed nodes
        hat_r = hat_r * placement.to(hat_r.dtype)

        # Soft projection onto discrete rank set
        rank_t = torch.tensor(
            self.rank_set, dtype=hat_r.dtype, device=hat_r.device
        )  # [R]
        diff = hat_r.unsqueeze(1) - rank_t.unsqueeze(0)  # [N, R]
        weights = F.softmax(-diff.pow(2) / (2.0 * self.sigma_sq), dim=1)  # [N, R]
        r_alloc = (weights * rank_t.unsqueeze(0)).sum(dim=1)  # [N]

        return r_alloc

    def discretize(self, r_alloc: torch.Tensor) -> torch.Tensor:
        """Hard projection: argmax over Gaussian soft weights.

        Args:
            r_alloc: Continuous ranks [N] (the hat_r values).

        Returns:
            Discrete ranks [N] (long tensor, values in self.rank_set).
        """
        rank_t = torch.tensor(
            self.rank_set, dtype=r_alloc.dtype, device=r_alloc.device
        )
        diff = r_alloc.unsqueeze(1) - rank_t.unsqueeze(0)  # [N, R]
        weights = F.softmax(-diff.pow(2) / (2.0 * self.sigma_sq), dim=1)
        idx = weights.argmax(dim=1)  # [N]
        return rank_t[idx]


# ═══════════════════════════════════════════════════════════════════════════
# 4. GreedyRankAllocator — Utility-Greedy Discrete Allocation
# ═══════════════════════════════════════════════════════════════════════════

class GreedyRankAllocator(RankAllocator):
    """Greedy allocator that maximises marginal utility per parameter.

    Score for each candidate (v_i, r):
        score = ΔU / ΔC = (u_i · f(r)) / params(v_i, r)
        f(r) = log2(r) / log2(r_max)

    Two-pass algorithm:
        1. Sort all (node, rank) candidates by score descending.
        2. Greedily pick the best affordable candidate per node.
        3. Upgrade pass: if budget remains, evaluate upgrading already-
           assigned nodes to higher ranks based on marginal gain / cost.

    Backward-compatibility: ``utilities`` can be passed explicitly (as done by
    `solver.py`) or omitted (in which case SEMANTIC_UTILITY is used).
    """

    def __init__(self, rank_set: Optional[List[int]] = None, r_max: int = 64):
        self.rank_set = rank_set if rank_set is not None else RANK_SET[:]
        self.r_max = r_max

    def _get_utility(self, graph: ComputationGraph, idx: int, utilities: Optional[torch.Tensor] = None) -> float:
        """Return unit-rank utility u_i for module ``idx``."""
        if utilities is not None:
            return float(utilities[idx].item())
        # Fallback: infer from module name heuristics
        if hasattr(graph, "nodes") and idx < len(graph.nodes):
            name = graph.nodes[idx].name
        else:
            names = graph.get_module_names()
            name = names[idx] if idx < len(names) else ""
        role = _infer_semantic_role_from_name(name)
        return SEMANTIC_UTILITY.get(role, 0.5)

    def allocate(
        self,
        graph: ComputationGraph,
        placement: torch.Tensor,
        budget: int,
        variant: str,
        utilities: Optional[torch.Tensor] = None,  # backward-compat with solver.py
    ) -> torch.Tensor:
        N = graph.n_nodes
        device = placement.device
        r_alloc = torch.zeros(N, dtype=torch.float32, device=device)

        placed_indices = torch.where(placement > 0.5)[0].tolist()
        if not placed_indices:
            return r_alloc

        # ── Build candidate pool ──
        candidates: List[Tuple[float, int, int, int]] = []
        for i in placed_indices:
            u_i = self._get_utility(graph, i, utilities)
            for r in self.rank_set:
                cost = int(graph.estimate_params(i, r, variant))
                if cost <= 0:
                    continue
                f_r = r_utility_fn(r, self.r_max)
                score = (u_i * f_r) / cost
                candidates.append((score, i, r, cost))

        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        # ── First pass: greedy assignment ──
        B_rem = budget
        assigned: set = set()
        for score, i, r, cost in candidates:
            if i in assigned:
                continue
            if B_rem >= cost:
                r_alloc[i] = float(r)
                B_rem -= cost
                assigned.add(i)

        # ── Second pass: upgrade already-assigned nodes ──
        if B_rem > 0 and assigned:
            # Sort by current utility descending for upgrade priority
            assigned_sorted = sorted(
                assigned,
                key=lambda idx: self._get_utility(graph, idx, utilities),
                reverse=True,
            )
            for i in assigned_sorted:
                current_r = int(r_alloc[i].item())
                # Try higher ranks in ascending order (cheapest first)
                for r in sorted(self.rank_set):
                    if r <= current_r:
                        continue
                    cost_diff = int(graph.estimate_params(i, r, variant)) - int(graph.estimate_params(i, current_r, variant))
                    if cost_diff <= 0 or B_rem < cost_diff:
                        continue
                    u_i = self._get_utility(graph, i, utilities)
                    gain = u_i * (r_utility_fn(r, self.r_max) - r_utility_fn(current_r, self.r_max))
                    if gain > 0 and gain / cost_diff >= 0.0:
                        r_alloc[i] = float(r)
                        B_rem -= cost_diff
                        break

        return r_alloc


# ═══════════════════════════════════════════════════════════════════════════
# 5. RLRankAllocator — PPO-based Sequential Allocation (Placeholder)
# ═══════════════════════════════════════════════════════════════════════════

class RLRankAllocator(RankAllocator, nn.Module):
    """PPO-based sequential rank allocator.

    .. warning::
        **FUTURE WORK — Phase 3 placeholder.** The MDP formulation, state
        encoder, policy head, and value head are implemented, but the PPO
        training loop (GAE, clipped surrogate, value update) is not yet
        wired.  ``allocate()`` falls back to a greedy-like sequential fill
        so the full V-PEFT pipeline can be end-to-end tested immediately.
        Tracked as Issue #11 in the analysis report.

    MDP formulation (per method_rank_allocation.md §3.3):
        State  : s_t = (h_{v_t}, h_G, B_rem, {r_j}_{j<t})
        Action : a_t ∈ {0} ∪ {r ∈ R | params(v_t, r) ≤ B_rem}
        Reward : R_t = u_{v_t} · f(r_t)  (or 0 if skip)

    The full PPO training loop (GAE, clipped surrogate, value update) will
    be implemented in a follow-up PR.  This class provides the complete
    interface and a naive fallback allocation so that the pipeline can be
    end-to-end tested immediately.
    """

    def __init__(
        self,
        hidden_dim: int,
        rank_set: Optional[List[int]] = None,
        r_max: int = 64,
    ):
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.rank_set = rank_set if rank_set is not None else RANK_SET[:]
        self.r_max = r_max
        self.action_size = len(self.rank_set) + 1  # +1 for "skip" action (0)

        # Shared state encoder (mirrors SoftRankAllocator front-end)
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, 128),  # [h_i, h_G, b_rem_norm]
            nn.LayerNorm(128),
            nn.GELU(),
        )
        # Policy head: distribution over actions
        self.policy_head = nn.Linear(128, self.action_size)
        # Value head: state-value V_ψ(s_t)
        self.value_head = nn.Linear(128, 1)

    def allocate(
        self,
        graph: ComputationGraph,
        placement: torch.Tensor,
        budget: int,
        variant: str,
    ) -> torch.Tensor:
        """Placeholder allocation — falls back to greedy-like sequential fill.

        Returns:
            r_alloc: [N] float tensor with assigned ranks.
        """
        N = graph.n_nodes
        device = placement.device
        r_alloc = torch.zeros(N, dtype=torch.float32, device=device)

        placed_indices = torch.where(placement > 0.5)[0].tolist()
        B_rem = budget

        for i in placed_indices:
            # Try highest affordable rank (simple heuristic until PPO is ready)
            assigned = False
            for r in sorted(self.rank_set, reverse=True):
                cost = int(graph.estimate_params(i, r, variant))
                if cost > 0 and B_rem >= cost:
                    r_alloc[i] = float(r)
                    B_rem -= cost
                    assigned = True
                    break
            if not assigned:
                # Try smallest rank
                for r in sorted(self.rank_set):
                    cost = int(graph.estimate_params(i, r, variant))
                    if cost > 0 and B_rem >= cost:
                        r_alloc[i] = float(r)
                        B_rem -= cost
                        break

        LOGGER.warning(
            "[RLRankAllocator] allocate() is a placeholder. "
            "Full PPO-based sequential allocation will be implemented in Phase 3."
        )
        return r_alloc


# ═══════════════════════════════════════════════════════════════════════════
# 6. HybridTrainingProtocol
# ═══════════════════════════════════════════════════════════════════════════

class HybridTrainingProtocol:
    """Two-stage training: SL warm-start → RL fine-tuning.

    Stage 1 (SL):
        Pre-train the placement policy with approximate oracle labels.
        Learns a strong prior for "reasonable" placements.

    Stage 2 (RL):
        Fine-tune with PPO on held-out architectures.
        Adapts to task-specific utility surfaces.
    """

    @staticmethod
    def train_sl(
        policy: PlacementPolicy,
        graph_data: List[Any],
        oracle_labels: List[torch.Tensor],
        epochs: int = 50,
        lr: float = 1e-3,
        lambda_budget: float = 1.0,
        lambda_deploy: float = 0.5,
        lambda_compat: float = 0.3,
        lambda_ent: float = 0.1,
        device: Union[str, torch.device] = "cpu",
        log_interval: int = 10,
    ) -> PlacementPolicy:
        """Supervised-learning warm-start.

        Args:
            policy: PlacementPolicy instance (untrained or partially trained).
            graph_data: List of graph objects.  Each element must be either:
                - a ``ComputationGraph`` with ``get_node_embeddings()``,
                  ``get_global_embedding()`` and ``get_variant_embedding()``
                  methods, OR
                - a tuple ``(graph, node_emb, global_emb, var_emb)`` where
                  ``graph`` is a ``ComputationGraph`` and the remaining items
                  are tensors.
            oracle_labels: List of [N] binary tensors, one per graph.
            epochs: Number of training epochs.
            lr: Adam learning rate.
            lambda_budget, lambda_deploy, lambda_compat, lambda_ent:
                Constraint-loss coefficients.
            device: "cpu" or "cuda".
            log_interval: Log every N epochs.

        Returns:
            The trained policy (same instance, mutated in-place).
        """
        policy = policy.to(device)
        policy.train()
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

        num_samples = len(graph_data)
        for epoch in range(epochs):
            epoch_loss = 0.0

            for sample, labels in zip(graph_data, oracle_labels):
                # Unpack sample
                if isinstance(sample, (tuple, list)) and len(sample) == 4:
                    graph, node_emb, global_emb, var_emb = sample
                else:
                    graph = sample
                    node_emb = graph.get_node_embeddings().to(device)
                    global_emb = graph.get_global_embedding().to(device)
                    var_emb = graph.get_variant_embedding().to(device)

                node_emb = node_emb.to(device)
                global_emb = global_emb.to(device)
                var_emb = var_emb.to(device)
                labels = labels.to(device)

                # Forward
                z = policy(node_emb, global_emb, var_emb)
                hard_mask = policy.constraint_registry.get_hard_mask(
                    graph, variant="lora"
                ).to(device)
                pi_hat = policy.compute_constrained_probs(z, hard_mask)

                # Budget & penalty inputs
                if hasattr(policy.constraint_registry, "get_budget_per_node"):
                    b_per_node = policy.constraint_registry.get_budget_per_node(
                        graph, variant="lora"
                    ).to(device)
                else:
                    # Fallback: build budget_per_node from graph.estimate_params at unit rank
                    b_per_node = torch.tensor(
                        [graph.estimate_params(i, 1, "lora") for i in range(graph.n_nodes)],
                        dtype=torch.float32,
                        device=device,
                    )
                budget_max = getattr(
                    graph, "budget_max", b_per_node.sum().item() * 0.5
                )
                if hasattr(policy.constraint_registry, "get_deployment_weights"):
                    deploy_weights = policy.constraint_registry.get_deployment_weights(
                        graph, profile="onnx"
                    ).to(device)
                else:
                    deploy_weights = torch.zeros(graph.n_nodes, device=device)
                if hasattr(policy.constraint_registry, "get_conflict_edges"):
                    conflict_edges = policy.constraint_registry.get_conflict_edges(graph)
                else:
                    conflict_edges = []

                loss = policy.compute_loss(
                    pi_hat,
                    oracle_labels=labels,
                    budget_per_node=b_per_node,
                    budget_max=budget_max,
                    deployment_weights=deploy_weights,
                    conflict_edges=conflict_edges,
                    lambda_budget=lambda_budget,
                    lambda_deploy=lambda_deploy,
                    lambda_compat=lambda_compat,
                    lambda_ent=lambda_ent,
                    mode="sl",
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(num_samples, 1)
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                LOGGER.info(
                    f"[HybridTrainingProtocol] SL Epoch {epoch + 1}/{epochs} — "
                    f"avg loss: {avg_loss:.4f}"
                )

        return policy

    @staticmethod
    def train_rl(
        policy: PlacementPolicy,
        env: Any,
        total_timesteps: float = 1e6,
        lr: float = 3e-4,
        device: Union[str, torch.device] = "cpu",
    ) -> PlacementPolicy:
        """Reinforcement-learning fine-tuning (PPO placeholder).

        Args:
            policy: Pre-trained PlacementPolicy (from SL warm-start).
            env: RL environment that yields (graph, reward) tuples.
            total_timesteps: Total environment steps for PPO training.
            lr: Optimiser learning rate for PPO.
            device: "cpu" or "cuda".

        Returns:
            The policy (currently returned unmodified; PPO logic to be added).
        """
        policy = policy.to(device)
        LOGGER.warning(
            "[HybridTrainingProtocol] RL fine-tuning (PPO) is a placeholder. "
            "Full PPO implementation with GAE, clipped surrogate objective, "
            "and value-head updates will be added in Phase 3. "
            "Returning policy without RL updates."
        )
        return policy
