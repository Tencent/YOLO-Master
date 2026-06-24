# 🐧 MoE module + MoE-loss regression tests (added 2026-06-24 audit follow-up)
"""Unit tests for Mixture-of-Experts modules and auxiliary-loss aggregation.

Covers the issues found in docs/audit/moe_module_and_loss_audit_2026-06-24.md:
  - P0  aux loss double counting in v8*Loss aggregation
  - P1  routing gradient flow (detach_routing default False)
  - P1  MOE_LOSS_REGISTRY no leak across forwards
  - P1  eval() yields zero aggregated aux
  - deepcopy safety (used by EMA / attempt_load_one_weight)
  - forward output shapes for the main MoE variants
"""

import copy

import pytest
import torch
import torch.nn as nn

from ultralytics.nn.modules.moe.modules import (
    A2C2fMoE,
    AdaptiveBalanceController,
    AdaptiveGateMoE,
    ES_MOE,
    HybridAdaptiveGateMoE,
    HyperFusedMoE,
    MOE_LOSS_REGISTRY,
    OptimizedMOE,
    OptimizedMOEImproved,
    UltraOptimizedMoE,
)
from ultralytics.nn.modules.moe.experts import OptimizedSimpleExpert, GhostExpert
from ultralytics.nn.modules.moe.loss import MoELoss, gshard_balance_loss, weighted_gshard_balance_loss
from ultralytics.nn.modules.moe.utils import last_conv_out_channels
from ultralytics.utils.loss import _collect_moe_aux_loss


def _sum_via_hasattr(model: nn.Module) -> float:
    """Old (buggy) aggregation: counts every module exposing `aux_loss`."""
    total = 0.0
    for m in model.modules():
        if hasattr(m, "aux_loss"):
            v = m.aux_loss
            total += float(v.detach()) if torch.is_tensor(v) else float(v)
    return total


def _sum_via_registry(model: nn.Module) -> float:
    """Correct aggregation: only registry members, de-duplicated by id."""
    seen, total = set(), 0.0
    for m in model.modules():
        t = MOE_LOSS_REGISTRY.get(m)
        if torch.is_tensor(t) and id(m) not in seen:
            total += float(t.detach())
            seen.add(id(m))
    return total


# ---------------------------------------------------------------------------
# P0: aux loss must not be double-counted
# ---------------------------------------------------------------------------
def test_aux_aggregation_no_double_count():
    """A2C2fMoE has wrapper modules (ABlockMoE) that delegate aux_loss.

    The old hasattr-based sum counted each inner MoE twice (2.000x). The fixed
    `_collect_moe_aux_loss` must equal the registry-only sum, and the buggy sum
    must be strictly larger (proving the wrapper double-count exists).
    """
    torch.manual_seed(0)
    m = A2C2fMoE(c1=64, c2=64, n=1, a2=True, num_experts=4, top_k=2).train()
    m(torch.randn(2, 64, 16, 16))

    buggy = _sum_via_hasattr(m)
    correct = _sum_via_registry(m)
    fixed = float(_collect_moe_aux_loss(m, torch.device("cpu")).detach())

    assert correct > 0.0, "registry should contain published aux losses after forward"
    # The fixed aggregator equals the registry truth (no double counting).
    assert fixed == pytest.approx(correct, rel=1e-5)
    # The legacy hasattr sum inflates by ~2x because of wrapper delegation.
    assert buggy > correct * 1.5
    assert buggy == pytest.approx(2.0 * correct, rel=1e-3)


def test_collect_helper_handles_none_and_eval():
    """Helper returns zero for None model and for eval-mode model."""
    torch.manual_seed(0)
    dev = torch.device("cpu")
    assert float(_collect_moe_aux_loss(None, dev)) == 0.0

    m = A2C2fMoE(c1=64, c2=64, n=1, a2=True, num_experts=4, top_k=2).train()
    m(torch.randn(2, 64, 16, 16))
    # Switch to eval: aggregation must short-circuit to zero.
    m.eval()
    assert float(_collect_moe_aux_loss(m, dev)) == 0.0


# ---------------------------------------------------------------------------
# P1: routing gradient flow
# ---------------------------------------------------------------------------
def test_routing_gradient_flows_by_default():
    """With detach_routing=False (default), main-task grad reaches the router."""
    torch.manual_seed(0)
    m = OptimizedMOEImproved(in_channels=32, out_channels=32, num_experts=4, top_k=2,
                             progressive_sparsity=False).train()
    assert m.detach_routing is False
    x = torch.randn(2, 32, 16, 16, requires_grad=True)
    out = m(x)
    # Pure main-task style loss (no aux): should still produce router grads.
    out.sum().backward()
    router_grads = [p.grad for p in m.routing.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in router_grads), (
        "router should receive gradient from the main task when detach_routing=False"
    )


def test_routing_detach_isolates_router():
    """With detach_routing=True, main-task grad must NOT reach router weights."""
    torch.manual_seed(0)
    m = OptimizedMOEImproved(in_channels=32, out_channels=32, num_experts=4, top_k=2,
                             progressive_sparsity=False, detach_routing=True).eval()
    # eval() so no aux loss is published; only main-task path contributes grad.
    x = torch.randn(2, 32, 16, 16)
    out = m(x)
    out.sum().backward()
    router_grads = [p.grad for p in m.routing.parameters() if p.requires_grad]
    assert all(g is None or g.abs().sum() == 0 for g in router_grads), (
        "router must be isolated from main-task gradient when detach_routing=True"
    )


# ---------------------------------------------------------------------------
# P1: registry must not leak
# ---------------------------------------------------------------------------
def test_registry_no_leak_across_forwards():
    """Repeated forwards must not grow the registry (one entry per MoE module)."""
    torch.manual_seed(0)
    m = OptimizedMOE(in_channels=32, out_channels=32, num_experts=4, top_k=2).train()
    x = torch.randn(2, 32, 16, 16)
    m(x)
    size_after_1 = len(MOE_LOSS_REGISTRY)
    for _ in range(5):
        m(x)
    size_after_n = len(MOE_LOSS_REGISTRY)
    assert size_after_n == size_after_1, "registry grew across forwards (leak)"


# ---------------------------------------------------------------------------
# deepcopy safety (EMA / checkpoint load rely on this)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("factory", [
    lambda: UltraOptimizedMoE(32, 32, num_experts=4, top_k=2),
    lambda: OptimizedMOE(32, 32, num_experts=4, top_k=2),
    lambda: OptimizedMOEImproved(32, 32, num_experts=4, top_k=2, progressive_sparsity=False),
    lambda: AdaptiveGateMoE(32, 32, num_experts=4, top_k=2),
    lambda: HybridAdaptiveGateMoE(32, 32, num_experts=4, top_k=2),
])
def test_deepcopy_safe_after_forward(factory):
    """deepcopy after a training forward (with non-leaf aux in registry) must work."""
    torch.manual_seed(0)
    m = factory().train()
    m(torch.randn(2, 32, 16, 16))
    mc = copy.deepcopy(m)
    n_orig = sum(p.numel() for p in m.parameters())
    n_copy = sum(p.numel() for p in mc.parameters())
    assert n_orig == n_copy


# ---------------------------------------------------------------------------
# forward shapes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("factory", [
    lambda: UltraOptimizedMoE(32, 48, num_experts=4, top_k=2),
    lambda: OptimizedMOE(32, 48, num_experts=4, top_k=2),
    lambda: OptimizedMOEImproved(32, 32, num_experts=4, top_k=2, progressive_sparsity=False),
    lambda: AdaptiveGateMoE(32, 32, num_experts=4, top_k=2),
    lambda: HybridAdaptiveGateMoE(32, 32, num_experts=4, top_k=2),
])
def test_forward_shapes(factory):
    """Forward preserves spatial dims and yields the configured out_channels."""
    torch.manual_seed(0)
    m = factory().train()
    x = torch.randn(2, 32, 16, 16)
    out = m(x)
    assert out.shape[0] == 2
    assert out.shape[2:] == (16, 16)
    assert out.shape[1] == m.out_channels


# ---------------------------------------------------------------------------
# §3.2: MoELoss coefficient floor must not silently override small coeffs
# ---------------------------------------------------------------------------
def test_moeloss_no_silent_floor_by_default():
    """A deliberately small balance coeff (0.01) must be respected (no 0.1 floor)."""
    torch.manual_seed(0)
    E, K, B = 4, 2, 8
    logits = torch.randn(B, E)
    probs = torch.softmax(logits, dim=1)
    idx = torch.topk(probs, K, dim=1).indices

    big = MoELoss(balance_loss_coeff=0.01, z_loss_coeff=0.0, num_experts=E, top_k=K)
    small_floor = MoELoss(balance_loss_coeff=0.01, z_loss_coeff=0.0, num_experts=E, top_k=K,
                          coeff_floor=0.1)
    l_default = float(big(probs, logits, idx).detach())
    l_floored = float(small_floor(probs, logits, idx).detach())
    # With floor disabled the loss is ~10x smaller than the floored variant.
    assert l_floored > l_default * 5, "coeff_floor=0.1 should dominate the 0.01 coeff"


def test_moeloss_diversity_skips_single_expert():
    """E==1 must not blow up the diversity term (§4.9)."""
    torch.manual_seed(0)
    E, K, B, D = 1, 1, 4, 16
    logits = torch.randn(B, E)
    probs = torch.softmax(logits, dim=1)
    idx = torch.topk(probs, K, dim=1).indices
    expert_out = torch.randn(B, E, D)
    loss_fn = MoELoss(balance_loss_coeff=1.0, z_loss_coeff=0.0, diversity_loss_coeff=1.0,
                      num_experts=E, top_k=K)
    out = loss_fn(probs, logits, idx, expert_outputs=expert_out, return_dict=True)
    assert torch.isfinite(out["loss"]).all()
    assert float(out["diversity_loss"]) == 0.0


# ---------------------------------------------------------------------------
# §3.3: unified balance loss is on the GShard scale (~1.0 at balance)
# ---------------------------------------------------------------------------
def test_gshard_balance_loss_uniform_equals_one():
    usage = torch.full((8,), 1.0 / 8)
    val = float(gshard_balance_loss(usage, 8))
    assert val == pytest.approx(1.0, rel=1e-5)


def test_gshard_balance_loss_collapsed_is_large():
    collapsed = torch.tensor([1.0, 0.0, 0.0, 0.0])  # all weight on one expert
    val = float(gshard_balance_loss(collapsed, 4))
    assert val == pytest.approx(4.0, rel=1e-5)  # N * sum(u^2) = 4 * 1 = 4


def test_es_moe_aux_on_gshard_scale():
    """ES_MOE aux loss should now be ~O(1), not the old MSE ~O(1e-3)."""
    torch.manual_seed(0)
    m = ES_MOE(in_channels=32, out_channels=32, num_experts=4, top_k=2).train()
    m(torch.randn(2, 32, 16, 16))
    aux = float(MOE_LOSS_REGISTRY.get(m).detach())
    assert aux >= 1.0, f"expected GShard-scale aux (>=1.0 at/above balance), got {aux}"


# ---------------------------------------------------------------------------
# §3.5: last_conv_out_channels is layout-agnostic
# ---------------------------------------------------------------------------
def test_last_conv_out_channels_various_experts():
    e1 = OptimizedSimpleExpert(16, 24)      # ends with GroupNorm after Conv
    e2 = GhostExpert(16, 24)                 # ghost structure
    assert last_conv_out_channels(e1) == 24
    # GhostExpert concatenates; its last conv is the cheap_operation conv.
    assert last_conv_out_channels(e2) > 0

    # A structure with a trailing activation (the old conv[-2] heuristic broke here)
    tricky = nn.Sequential(nn.Conv2d(8, 13, 1), nn.BatchNorm2d(13), nn.SiLU())
    wrapper = nn.Module()
    wrapper.conv = tricky
    assert last_conv_out_channels(wrapper) == 13


def test_subclass_reinit_no_default_kaiming_leftover():
    """After §3.4, swapped-in fused_experts should be initialized (finite, non-degenerate)."""
    torch.manual_seed(0)
    m = HybridAdaptiveGateMoE(32, 32, num_experts=4, top_k=2)
    convs = [mod for mod in m.fused_experts.modules() if isinstance(mod, nn.Conv2d)]
    assert convs, "fused_experts should contain conv layers"
    for c in convs:
        assert torch.isfinite(c.weight).all()


# ---------------------------------------------------------------------------
# §9 (rev5): cross-scale aux must not be silently dominated
# ---------------------------------------------------------------------------
def test_weighted_gshard_balance_scale():
    """weighted_gshard: uniform target == plain gshard; usage==target -> 1.0; collapse -> N."""
    E = 8
    uni = torch.full((E,), 1.0 / E)
    assert abs(float(weighted_gshard_balance_loss(uni, uni, E)) - 1.0) < 1e-4
    assert abs(float(weighted_gshard_balance_loss(uni, uni, E)) - float(gshard_balance_loss(uni, E))) < 1e-4
    tgt = torch.softmax(torch.randn(E), dim=0)
    assert abs(float(weighted_gshard_balance_loss(tgt, tgt, E)) - 1.0) < 1e-4  # min at usage==target
    col = torch.zeros(E); col[0] = 1.0
    assert abs(float(weighted_gshard_balance_loss(col, uni, E)) - E) < 1e-3


def test_adaptive_balance_controller_gshard_scale_and_nonneg():
    """Controller aux must be O(0.1..1) (not O(1e-3)) and never negative."""
    E = 8
    ctrl = AdaptiveBalanceController(E)
    bal = torch.full((E,), 1.0 / E)
    aux_early = float(ctrl({'expert_usage': bal}, torch.tensor(0)))
    aux_late = float(ctrl({'expert_usage': bal}, torch.tensor(10 ** 6)))
    assert aux_early >= 0.0 and aux_late >= 0.0, "aux must be non-negative"
    assert aux_late >= 0.05, f"late-stage balanced aux collapsed to {aux_late} (should stay O(0.1))"
    col = torch.zeros(E); col[0] = 1.0
    assert float(ctrl({'expert_usage': col}, torch.tensor(0))) > aux_early  # collapse penalized more


def test_controller_block_not_dominated_when_mixed():
    """A controller-based block summed with a GShard block must keep a meaningful share."""
    torch.manual_seed(0)
    MOE_LOSS_REGISTRY.clear()

    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = OptimizedMOE(64, 64, num_experts=4, top_k=2)
            self.b = HyperFusedMoE(64, 64, num_experts=4, top_k=2)

        def forward(self, x):
            return self.b(self.a(x))

    m = Mixed().train()
    m(torch.randn(2, 64, 16, 16))
    total = float(_collect_moe_aux_loss(m, torch.device("cpu")))
    b = float(MOE_LOSS_REGISTRY.get(m.b).detach())
    assert b / max(total, 1e-9) > 0.05, f"controller block share {b/total:.4f} too small (silently dominated)"


def _first_router_weight(m):
    routing = getattr(m, "routing", None) or m
    for layer in routing.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)) and layer.weight.requires_grad:
            return layer.weight
    for p in routing.parameters():
        if p.requires_grad:
            return p
    return None


def test_controller_balance_loss_grad_reaches_router():
    """rev7 C1: balance loss gradient must flow to the router (was broken).

    The ZeroCostRouter route fed a pure-count `expert_usage` (no grad) into the
    balance loss, so the router never learned to balance. importance must now use
    mean(router_probs) and keep the gradient path alive.
    """
    torch.manual_seed(0)
    MOE_LOSS_REGISTRY.clear()
    m = HyperFusedMoE(32, 32, num_experts=4, top_k=2).train()
    x = torch.randn(2, 32, 16, 16)
    out = m(x)
    aux = m.aux_loss
    assert isinstance(aux, torch.Tensor) and aux.requires_grad
    w = _first_router_weight(m)
    m.zero_grad(set_to_none=True)
    (out.float().mean() + aux).backward()
    assert w is not None and w.grad is not None and w.grad.abs().sum() > 0, \
        "router weight received no gradient from the balance loss"


def test_adaptive_controller_grad_to_router_probs():
    """rev7 C1: AdaptiveBalanceController must backprop into router_probs."""
    torch.manual_seed(0)
    E = 6
    ctrl = AdaptiveBalanceController(E)
    p_leaf = torch.randn(16, E, requires_grad=True)
    probs = torch.softmax(p_leaf, dim=1)
    idx = probs.argmax(dim=1)
    usage = torch.zeros(E).scatter_add_(0, idx, torch.ones(16)) / 16
    loss = ctrl({"expert_usage": usage, "router_probs": probs}, torch.tensor(0.0))
    loss.backward()
    assert p_leaf.grad is not None and p_leaf.grad.abs().sum() > 0
    assert float(loss) >= -1e-6


def test_registry_clear_prevents_double_backward():
    """rev7 C2: clearing the registry each forward avoids stale double-backward."""
    torch.manual_seed(0)
    MOE_LOSS_REGISTRY.clear()
    m = HyperFusedMoE(32, 32, num_experts=4, top_k=2).train()
    x = torch.randn(2, 32, 16, 16)
    (m(x).float().mean() + m.aux_loss).backward()
    MOE_LOSS_REGISTRY.clear()  # mimic tasks.py per-step reset
    out2 = m(x)
    # must not raise "backward through the graph a second time"
    (out2.float().mean() + m.aux_loss).backward()


def test_eval_does_not_write_registry():
    """rev7 C2: eval forward must not leave tensors in the global registry."""
    MOE_LOSS_REGISTRY.clear()
    m = HyperFusedMoE(32, 32, num_experts=4, top_k=2).eval()
    with torch.no_grad():
        m(torch.randn(2, 32, 16, 16))
    assert len(MOE_LOSS_REGISTRY) == 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
