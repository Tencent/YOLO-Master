"""V-PEFT Compiler — Module 3: Placement Policy + Rank Allocation.

Contains:
  - PlacementPolicy     : neural per-module placement with hard/soft constraints
  - RankAllocator (ABC) : abstract rank-allocation interface
  - SoftRankAllocator   : continuous-relaxation + Gaussian soft-projection
  - GreedyRankAllocator : utility-greedy discrete allocation
  - RLRankAllocator     : PPO-based sequential allocator
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
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER

from .constraints import ConstraintRegistry
from .graph import ComputationGraph

__all__ = [
    "PlacementPolicy",
    "RankAllocator",
    "SoftRankAllocator",
    "GreedyRankAllocator",
    "RLRankAllocator",
    "PPORankTrajectory",
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
        constraints: Optional[ConstraintRegistry] = None,
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
                if constraints is not None and not constraints.is_rank_feasible(graph, i, variant, r):
                    continue
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
                    if constraints is not None and not constraints.is_rank_feasible(graph, i, variant, r):
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
# 5. RLRankAllocator — PPO-based Sequential Allocation
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PPORankTrajectory:
    """One sequential rank-allocation rollout consumed by PPO."""

    states: torch.Tensor
    actions: torch.Tensor
    action_masks: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    allocation: torch.Tensor
    budget_used: int


class RLRankAllocator(RankAllocator, nn.Module):
    """PPO-based sequential rank allocator with hard action masking.

    MDP formulation (per method_rank_allocation.md §3.3):
        State  : s_t = (h_{v_t}, h_G, B_rem, {r_j}_{j<t})
        Action : a_t ∈ {0} ∪ {r ∈ R | params(v_t, r) ≤ B_rem}
        Reward : R_t = u_{v_t} · f(r_t)  (or 0 if skip)

    The encoded state uses the current node embedding, mean/global graph
    embedding, and normalized remaining budget. Prior actions affect the
    state through the remaining-budget term. Graphs without embeddings use
    the deterministic greedy allocator rather than random policy inputs.
    """

    def __init__(
        self,
        hidden_dim: int,
        rank_set: Optional[List[int]] = None,
        r_max: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_loss_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        nn.Module.__init__(self)
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.hidden_dim = hidden_dim
        ranks = rank_set if rank_set is not None else RANK_SET[:]
        self.rank_set = sorted({int(rank) for rank in ranks})
        if not self.rank_set or any(rank <= 0 for rank in self.rank_set):
            raise ValueError("rank_set must contain positive integer ranks")
        self.r_max = r_max
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.value_loss_coeff = float(value_loss_coeff)
        self.entropy_coeff = float(entropy_coeff)
        self.max_grad_norm = float(max_grad_norm)
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        if self.clip_ratio < 0 or self.value_loss_coeff < 0 or self.entropy_coeff < 0 or self.max_grad_norm <= 0:
            raise ValueError("PPO loss coefficients must be non-negative and max_grad_norm must be positive")
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
        self.register_buffer("ppo_updates", torch.zeros((), dtype=torch.long), persistent=True)
        self.last_training_metrics: Dict[str, float] = {}

    @property
    def is_trained(self) -> bool:
        """Return whether at least one PPO optimizer update has completed."""

        return bool(self.ppo_updates.item() > 0)

    def _fit_embedding_dim(self, embedding: torch.Tensor) -> torch.Tensor:
        """Pad or truncate embeddings to the policy's configured hidden size."""

        embedding = embedding.float()
        if embedding.shape[-1] == self.hidden_dim:
            return embedding
        if embedding.shape[-1] > self.hidden_dim:
            return embedding[..., : self.hidden_dim]
        return F.pad(embedding, (0, self.hidden_dim - embedding.shape[-1]))

    def _graph_embeddings(self, graph: ComputationGraph) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return deterministic node/global embeddings or ``None`` for cold-start graphs."""

        node_embeddings = getattr(graph, "node_embeddings", None)
        if node_embeddings is None:
            node_embeddings = getattr(graph, "_node_features", None)
        if not isinstance(node_embeddings, torch.Tensor) or node_embeddings.dim() != 2:
            return None
        if node_embeddings.shape[0] != graph.n_nodes:
            raise ValueError(
                f"graph embeddings contain {node_embeddings.shape[0]} nodes, expected {graph.n_nodes}"
            )
        node_embeddings = self._fit_embedding_dim(node_embeddings)
        global_embedding = getattr(graph, "global_embedding", None)
        if not isinstance(global_embedding, torch.Tensor):
            global_embedding = node_embeddings.mean(dim=0)
        global_embedding = self._fit_embedding_dim(global_embedding.reshape(1, -1)).reshape(-1)
        return node_embeddings, global_embedding

    def encode_state(
        self,
        node_embedding: torch.Tensor,
        global_embedding: torch.Tensor,
        remaining_budget: int,
        initial_budget: int,
    ) -> torch.Tensor:
        """Encode one MDP state as ``[node, global, normalized budget]``."""

        device = next(self.parameters()).device
        node = self._fit_embedding_dim(node_embedding.to(device)).reshape(-1)
        global_emb = self._fit_embedding_dim(global_embedding.to(device)).reshape(-1)
        budget_fraction = max(float(remaining_budget), 0.0) / max(float(initial_budget), 1.0)
        budget_feature = node.new_tensor([min(budget_fraction, 1.0)])
        return torch.cat((node, global_emb, budget_feature), dim=0)

    def feasible_action_mask(
        self,
        graph: ComputationGraph,
        node_index: int,
        remaining_budget: int,
        variant: str,
        constraints: Optional[ConstraintRegistry] = None,
        *,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Return a boolean mask over skip plus every feasible rank action."""

        mask = torch.zeros(self.action_size, dtype=torch.bool, device=device)
        mask[0] = True
        for action, rank in enumerate(self.rank_set, start=1):
            feasible = constraints is None or constraints.is_rank_feasible(graph, node_index, variant, rank)
            cost = int(graph.estimate_params(node_index, rank, variant))
            mask[action] = feasible and cost > 0 and cost <= remaining_budget
        return mask

    def _distribution(self, states: torch.Tensor, action_masks: torch.Tensor):
        """Build the masked categorical policy and value estimates."""

        encoded = self.state_encoder(states)
        logits = self.policy_head(encoded)
        masks = action_masks.to(device=logits.device, dtype=torch.bool)
        if masks.shape != logits.shape:
            raise ValueError(f"action mask shape {tuple(masks.shape)} does not match logits {tuple(logits.shape)}")
        if not masks.any(dim=-1).all():
            raise ValueError("every PPO state must allow at least the skip action")
        masked_logits = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
        distribution = torch.distributions.Categorical(logits=masked_logits)
        values = self.value_head(encoded).squeeze(-1)
        return distribution, values

    def select_action(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select one feasible action and return action, log-probability, and value."""

        distribution, values = self._distribution(state.unsqueeze(0), action_mask.unsqueeze(0))
        action = distribution.logits.argmax(dim=-1) if deterministic else distribution.sample()
        return action[0], distribution.log_prob(action)[0], values[0]

    @staticmethod
    def compute_gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        *,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        next_value: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute generalized advantages and bootstrapped returns."""

        if not (rewards.shape == values.shape == dones.shape):
            raise ValueError("rewards, values, and dones must have identical shapes")
        advantages = torch.zeros_like(rewards)
        gae = rewards.new_zeros(())
        bootstrap = rewards.new_zeros(()) if next_value is None else next_value.to(rewards)
        for index in range(rewards.numel() - 1, -1, -1):
            nonterminal = 1.0 - dones[index].to(rewards.dtype)
            delta = rewards[index] + gamma * bootstrap * nonterminal - values[index]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[index] = gae
            bootstrap = values[index]
        return advantages, advantages + values

    def collect_trajectory(
        self,
        graph: ComputationGraph,
        placement: torch.Tensor,
        budget: int,
        variant: str,
        *,
        constraints: Optional[ConstraintRegistry] = None,
        deterministic: bool = False,
        reward_fn: Optional[Callable[..., float]] = None,
    ) -> PPORankTrajectory:
        """Roll out one budget-constrained rank allocation trajectory."""

        embeddings = self._graph_embeddings(graph)
        if embeddings is None:
            raise ValueError("PPO rollout requires graph.node_embeddings or graph._node_features")
        if budget < 0:
            raise ValueError(f"budget must be non-negative, got {budget}")
        node_embeddings, global_embedding = embeddings
        device = next(self.parameters()).device
        placement = placement.to(device)
        if placement.numel() != graph.n_nodes:
            raise ValueError(f"placement has {placement.numel()} entries, expected {graph.n_nodes}")

        allocation = torch.zeros(graph.n_nodes, dtype=torch.float32, device=device)
        placed_indices = torch.where(placement > 0.5)[0].tolist()
        states: List[torch.Tensor] = []
        actions: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        log_probs: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        rewards: List[float] = []
        remaining_budget = int(budget)

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                for node_index in placed_indices:
                    state = self.encode_state(
                        node_embeddings[node_index], global_embedding, remaining_budget, int(budget)
                    )
                    mask = self.feasible_action_mask(
                        graph, node_index, remaining_budget, variant, constraints, device=device
                    )
                    action, log_prob, value = self.select_action(state, mask, deterministic=deterministic)
                    action_index = int(action.item())
                    rank = 0 if action_index == 0 else self.rank_set[action_index - 1]
                    cost = 0 if rank == 0 else int(graph.estimate_params(node_index, rank, variant))
                    default_reward = 0.0
                    if rank > 0:
                        role = _infer_semantic_role_from_name(graph.get_module_names()[node_index])
                        default_reward = SEMANTIC_UTILITY.get(role, 0.5) * float(r_utility_fn(rank, self.r_max))
                        allocation[node_index] = float(rank)
                        remaining_budget -= cost
                    reward = (
                        reward_fn(
                            graph=graph,
                            node_index=node_index,
                            rank=rank,
                            cost=cost,
                            remaining_budget=remaining_budget,
                            default_reward=default_reward,
                        )
                        if reward_fn is not None
                        else default_reward
                    )
                    states.append(state)
                    masks.append(mask)
                    actions.append(action)
                    log_probs.append(log_prob)
                    values.append(value)
                    rewards.append(float(reward))
        finally:
            self.train(was_training)

        if not states:
            empty_states = torch.empty((0, self.hidden_dim * 2 + 1), device=device)
            return PPORankTrajectory(
                states=empty_states,
                actions=torch.empty(0, dtype=torch.long, device=device),
                action_masks=torch.empty((0, self.action_size), dtype=torch.bool, device=device),
                old_log_probs=torch.empty(0, device=device),
                old_values=torch.empty(0, device=device),
                rewards=torch.empty(0, device=device),
                dones=torch.empty(0, device=device),
                allocation=allocation,
                budget_used=0,
            )
        dones = torch.zeros(len(states), dtype=torch.float32, device=device)
        dones[-1] = 1.0
        return PPORankTrajectory(
            states=torch.stack(states),
            actions=torch.stack(actions).long(),
            action_masks=torch.stack(masks),
            old_log_probs=torch.stack(log_probs),
            old_values=torch.stack(values),
            rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
            dones=dones,
            allocation=allocation,
            budget_used=int(budget) - remaining_budget,
        )

    def ppo_update(
        self,
        trajectory: PPORankTrajectory,
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 3e-4,
        epochs: int = 4,
        minibatch_size: int = 64,
    ) -> Dict[str, float]:
        """Update policy and value heads with clipped PPO and GAE targets."""

        if trajectory.actions.numel() == 0:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        if epochs <= 0 or minibatch_size <= 0:
            raise ValueError("epochs and minibatch_size must be positive")
        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=lr)
        advantages, returns = self.compute_gae(
            trajectory.rewards,
            trajectory.old_values,
            trajectory.dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        states = trajectory.states.detach()
        actions = trajectory.actions.detach()
        action_masks = trajectory.action_masks.detach()
        old_log_probs = trajectory.old_log_probs.detach()
        returns = returns.detach()
        metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        updates = 0
        self.train()
        for _ in range(epochs):
            for indices in torch.randperm(actions.numel(), device=actions.device).split(minibatch_size):
                distribution, values = self._distribution(states[indices], action_masks[indices])
                log_probs = distribution.log_prob(actions[indices])
                ratio = (log_probs - old_log_probs[indices]).exp()
                advantage = advantages[indices]
                surrogate = torch.minimum(
                    ratio * advantage,
                    ratio.clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantage,
                )
                policy_loss = -surrogate.mean()
                value_loss = F.mse_loss(values, returns[indices])
                entropy = distribution.entropy().mean()
                loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                optimizer.step()
                metrics["loss"] += float(loss.detach())
                metrics["policy_loss"] += float(policy_loss.detach())
                metrics["value_loss"] += float(value_loss.detach())
                metrics["entropy"] += float(entropy.detach())
                updates += 1
        for key in metrics:
            metrics[key] /= max(updates, 1)
        metrics["updates"] = float(updates)
        self.ppo_updates.add_(updates)
        self.last_training_metrics = metrics
        return metrics

    def allocate(
        self,
        graph: ComputationGraph,
        placement: torch.Tensor,
        budget: int,
        variant: str,
        constraints: Optional[ConstraintRegistry] = None,
    ) -> torch.Tensor:
        """Allocate ranks with the trained policy or a deterministic cold-start fallback."""

        if not self.is_trained or self._graph_embeddings(graph) is None:
            return GreedyRankAllocator(self.rank_set, self.r_max).allocate(
                graph, placement, budget, variant, constraints=constraints
            )
        return self.collect_trajectory(
            graph,
            placement,
            budget,
            variant,
            constraints=constraints,
            deterministic=True,
        ).allocation.to(placement.device)


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
                    graph, variant="lora", candidate_ranks=RANK_SET
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
        policy: RLRankAllocator,
        env: Any,
        total_timesteps: float = 1e6,
        lr: float = 3e-4,
        device: Union[str, torch.device] = "cpu",
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
        log_interval: int = 100,
    ) -> RLRankAllocator:
        """Fine-tune an :class:`RLRankAllocator` from sampled graph episodes.

        Args:
            policy: Rank allocator to optimize.
            env: Episode source. Each sample is a mapping with ``graph``,
                ``placement``, ``budget`` and ``variant`` keys, or the same
                values as a tuple. The source may be an iterable, callable, or
                expose ``sample()``. Optional keys are ``constraints``,
                ``reward_fn`` and ``episode_reward``.
            total_timesteps: Total environment steps for PPO training.
            lr: Optimiser learning rate for PPO.
            device: "cpu" or "cuda".

        Returns:
            The trained allocator (same instance, mutated in-place).
        """
        if not isinstance(policy, RLRankAllocator):
            raise TypeError("train_rl expects an RLRankAllocator with policy and value heads")
        target_steps = int(total_timesteps)
        if target_steps <= 0:
            raise ValueError("total_timesteps must be positive")
        policy = policy.to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        iterator = iter(env) if isinstance(env, Iterable) and not isinstance(env, (Mapping, str, bytes)) else None

        def next_episode():
            if isinstance(env, Mapping):
                return env
            if hasattr(env, "sample") and callable(env.sample):
                return env.sample()
            if callable(env):
                return env()
            if iterator is not None:
                try:
                    return next(iterator)
                except StopIteration as exc:
                    raise RuntimeError("RL episode source was exhausted before total_timesteps") from exc
            raise TypeError("env must be an episode mapping, iterable, callable, or expose sample()")

        steps = episodes = 0
        while steps < target_steps:
            sample = next_episode()
            if isinstance(sample, Mapping):
                graph = sample["graph"]
                placement = sample["placement"]
                budget = sample["budget"]
                variant = sample["variant"]
                constraints = sample.get("constraints")
                reward_fn = sample.get("reward_fn")
                episode_reward = sample.get("episode_reward")
            elif isinstance(sample, (tuple, list)) and 4 <= len(sample) <= 7:
                graph, placement, budget, variant = sample[:4]
                constraints = sample[4] if len(sample) > 4 else None
                reward_fn = sample[5] if len(sample) > 5 else None
                episode_reward = sample[6] if len(sample) > 6 else None
            else:
                raise TypeError("RL episode must be a mapping or a 4-7 item tuple")
            trajectory = policy.collect_trajectory(
                graph,
                placement,
                int(budget),
                str(variant),
                constraints=constraints,
                reward_fn=reward_fn,
            )
            if trajectory.actions.numel() == 0:
                raise ValueError("RL episode placement contains no active nodes")
            if episode_reward is not None:
                trajectory.rewards[-1] += float(episode_reward)
            metrics = policy.ppo_update(
                trajectory,
                optimizer=optimizer,
                epochs=ppo_epochs,
                minibatch_size=minibatch_size,
            )
            steps += int(trajectory.actions.numel())
            episodes += 1
            if episodes == 1 or episodes % max(log_interval, 1) == 0:
                LOGGER.info(
                    f"[HybridTrainingProtocol] PPO episode {episodes}, steps {steps}/{target_steps}, "
                    f"loss={metrics['loss']:.4f}, entropy={metrics['entropy']:.4f}"
                )
        return policy
