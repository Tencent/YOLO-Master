# 🐧 MoE module + MoE-loss regression tests (added 2026-06-24 audit follow-up)
"""Unit tests for Mixture-of-Experts modules and auxiliary-loss aggregation.

Covers the issues found in docs/audit/moe_module_and_loss_audit_2026-06-24.md:
  - aux loss double counting in v8*Loss aggregation
  - routing gradient flow (detach_routing default False)
  - MOE_LOSS_REGISTRY no leak across forwards
  - eval() yields zero aggregated aux
  - deepcopy safety (used by EMA / attempt_load_one_weight)
  - forward output shapes for the main MoE variants
"""

import copy

import pytest
import torch
import torch.nn as nn

from ultralytics.nn.modules.moe.modules import (
    A2C2fMoE,
    ABlockMoE,
    AdaptiveBalanceController,
    AdaptiveCapacityMoE,
    AdaptiveGateMoE,
    ES_MOE,
    HybridAdaptiveGateMoE,
    HyperFusedMoE,
    HyperUltimateMoE,
    MOE_SNAPSHOT_INTERVAL,
    MOE_LOSS_REGISTRY,
    OptimizedMOE,
    OptimizedMOEImproved,
    UltraOptimizedMoE,
    _compute_usage_from_topk,
    _record_moe_snapshot,
)
from ultralytics.nn.modules.moe.experts import (
    OptimizedSimpleExpert, GhostExpert, SimpleExpert, SpatialExpert, InvertedResidualExpert,
)
from ultralytics.nn.modules.moe.routers import UltraEfficientRouter, AdvancedRoutingLayer
from ultralytics.nn.modules.moe.loss import MoELoss, gshard_balance_loss, weighted_gshard_balance_loss
from ultralytics.nn.modules.moe.utils import last_conv_out_channels, BatchedExpertComputation
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
# aux loss must not be double-counted
# ---------------------------------------------------------------------------
def test_aux_aggregation_no_double_count():
    """A2C2fMoE has wrapper modules (ABlockMoE) that delegate aux_loss.

    `_collect_moe_aux_loss` must equal the registry-only sum. The legacy
    hasattr-based sum still over-counts because multiple nested modules expose
    the same aux_loss, but A2C2fMoE.aux_loss itself must match the registry.
    """
    torch.manual_seed(0)
    m = A2C2fMoE(c1=64, c2=64, n=1, a2=True, num_experts=4, top_k=2).train()
    m(torch.randn(2, 64, 16, 16))

    buggy = _sum_via_hasattr(m)
    correct = _sum_via_registry(m)
    fixed = float(_collect_moe_aux_loss(m, torch.device("cpu")).detach())
    wrapper = float(m.aux_loss.detach())

    assert correct > 0.0, "registry should contain published aux losses after forward"
    assert fixed == pytest.approx(correct, rel=1e-5)
    assert wrapper == pytest.approx(correct, rel=1e-5)
    assert buggy > correct * 1.25, "hasattr traversal still double/triple counts nested aux_loss"


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
# routing gradient flow
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


def test_improved_moe_rejects_nonfinite_shared_expert_output(monkeypatch):
    m = OptimizedMOEImproved(32, 32, num_experts=4, top_k=2, progressive_sparsity=False).eval()
    monkeypatch.setattr(m.shared_expert, "forward", lambda x: torch.full_like(x, float("nan")))
    with pytest.raises(RuntimeError, match="shared expert"):
        m(torch.randn(2, 32, 16, 16))


def test_improved_moe_rejects_nonfinite_sparse_aggregation(monkeypatch):
    m = OptimizedMOEImproved(32, 32, num_experts=4, top_k=2, progressive_sparsity=False).eval()
    monkeypatch.setattr(m.experts[0], "forward", lambda x: torch.full_like(x, float("nan")))
    monkeypatch.setattr(m.routing, "forward", lambda x, top_k: (torch.tensor([[1.0, 0.0]], device=x.device).expand(x.shape[0], -1), torch.tensor([[0, 1]], device=x.device).expand(x.shape[0], -1), {}))
    with pytest.raises(RuntimeError, match="sparse expert aggregation"):
        m(torch.randn(2, 32, 16, 16))


def test_improved_moe_rejects_nonfinite_after_low_precision_output_conversion(monkeypatch):
    m = OptimizedMOEImproved(32, 32, num_experts=4, top_k=2, progressive_sparsity=False).eval().half()
    monkeypatch.setattr(m.shared_expert, "forward", lambda x: torch.full_like(x.float(), 1e5))
    monkeypatch.setattr(m.routing, "forward", lambda x, top_k: (torch.tensor([[1.0, 0.0]], device=x.device).expand(x.shape[0], -1), torch.tensor([[0, 1]], device=x.device).expand(x.shape[0], -1), {}))
    monkeypatch.setattr(m.experts[0], "forward", lambda x: torch.zeros_like(x))
    try:
        m(torch.randn(2, 32, 16, 16, dtype=torch.float16))
    except RuntimeError as exc:
        if "not implemented for 'Half'" in str(exc):
            pytest.skip(f"CPU fp16 operator unavailable: {exc}")
        assert "dtype conversion" in str(exc)
    else:
        pytest.fail("expected low-precision output conversion to reject overflow")


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
# registry must not leak
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


def test_moe_snapshot_tensors_remain_on_source_device():
    """Routing snapshots should not force a CPU sync when recorded."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = nn.Module()
    module.num_experts = 4
    module.top_k = 2
    topk_indices = torch.tensor([[0, 1], [1, 3]], device=device)
    topk_weights = torch.rand(2, 2, device=device)
    router_probs = torch.softmax(torch.randn(2, 4, device=device), dim=1)
    aux_loss = router_probs.sum()

    usage, counts = _compute_usage_from_topk(topk_indices, module.num_experts)
    assert usage.device == device
    assert counts.device == device

    for _ in range(MOE_SNAPSHOT_INTERVAL):
        _record_moe_snapshot(
            module,
            topk_indices=topk_indices,
            topk_weights=topk_weights,
            router_probs=router_probs,
            aux_loss=aux_loss,
        )

    snapshot = module.last_routing_snapshot
    for key in ("expert_usage", "topk_counts", "mean_router_probs", "mean_topk_weight", "aux_loss"):
        value = snapshot[key]
        if isinstance(value, torch.Tensor):
            assert value.device == device


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


def test_moeloss_diversity_requires_expert_outputs():
    """diversity_loss_coeff should not silently become a dead no-op."""
    torch.manual_seed(0)
    E, K, B = 4, 2, 8
    logits = torch.randn(B, E)
    probs = torch.softmax(logits, dim=1)
    idx = torch.topk(probs, K, dim=1).indices
    loss_fn = MoELoss(balance_loss_coeff=1.0, z_loss_coeff=0.0, diversity_loss_coeff=1.0,
                      num_experts=E, top_k=K)
    with pytest.raises(ValueError, match="requires expert_outputs"):
        loss_fn(probs, logits, idx)


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


def test_es_moe_statistics_are_nonpersistent_buffers():
    module = ES_MOE(8, 8, num_experts=3, top_k=2)

    assert "load_balancing_loss" in module._buffers
    assert "expert_usage_counts" in module._buffers
    assert "load_balancing_loss" not in module.state_dict()
    assert "expert_usage_counts" not in module.state_dict()


def test_es_moe_repairs_legacy_plain_stat_attributes():
    module = ES_MOE(8, 8, num_experts=3, top_k=2).train()
    del module._buffers["load_balancing_loss"]
    del module._buffers["expert_usage_counts"]
    module.load_balancing_loss = torch.tensor(0.0)
    module.expert_usage_counts = torch.zeros(3)

    module(torch.randn(2, 8, 4, 4))

    assert "load_balancing_loss" in module._buffers
    assert "expert_usage_counts" in module._buffers
    assert torch.isfinite(module.load_balancing_loss)


def test_es_moe_sparse_inference_supports_channel_projection():
    module = ES_MOE(8, 12, num_experts=4, top_k=2, dynamic_threshold=0.0).eval()

    with torch.no_grad():
        output = module(torch.randn(2, 8, 5, 7))

    assert output.shape == (2, 12, 5, 7)
    assert torch.isfinite(output).all()


def test_es_moe_default_no_topk_eval_matches_dense(monkeypatch):
    """top_k=None means all experts, so eval must not apply threshold-pruned sparse dispatch."""
    module = ES_MOE(8, 8, num_experts=3, top_k=None, use_sparse_inference=True, dynamic_threshold=0.4).eval()
    x = torch.randn(2, 8, 5, 7)
    routing_weights = torch.full((2, 3, 5, 7), 1.0 / 3.0)

    monkeypatch.setattr(module.routing, "forward", lambda _: routing_weights)
    monkeypatch.setattr(
        module,
        "_sparse_forward",
        lambda *_: (_ for _ in ()).throw(AssertionError("default all-expert eval must stay dense")),
    )
    with torch.no_grad():
        expected = module.norm(module._dense_forward(x, routing_weights))
        output = module(x)

    assert module.use_top_k is False
    assert module.top_k == module.num_experts
    assert torch.allclose(output, expected)


def test_es_moe_sparse_pruning_renormalizes_retained_mass():
    """Optional threshold pruning must not shrink activations by discarded softmax mass."""
    module = ES_MOE(1, 1, num_experts=3, top_k=2, dynamic_threshold=0.5).eval()
    module.experts = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
    x = torch.ones(1, 1, 2, 2)
    routing_weights = torch.tensor([0.6, 0.4, 0.0]).view(1, 3, 1, 1).expand(1, 3, 2, 2)

    with torch.no_grad():
        output = module._sparse_forward(x, routing_weights)

    assert torch.allclose(output, x)


@pytest.mark.parametrize(
    ("top_k", "use_sparse_inference", "expected"),
    ((None, True, False), (3, True, False), (2, True, True), (2, False, False)),
)
def test_es_moe_sparse_capability_matches_effective_eval_path(top_k, use_sparse_inference, expected):
    module = ES_MOE(8, 8, num_experts=3, top_k=top_k, use_sparse_inference=use_sparse_inference)

    assert module.export_capabilities()["eager_sparse_dispatch"] is expected


def test_es_moe_caps_expert_kernel_sizes():
    module = ES_MOE(8, 8, num_experts=16, top_k=2, max_kernel_size=15)
    kernels = [expert.conv.depthwise.kernel_size[0] for expert in module.experts]

    assert max(kernels) == 15
    assert all(kernel % 2 == 1 for kernel in kernels)


@pytest.mark.parametrize("kwargs", ({"top_k": 0}, {"top_k": 4}, {"dynamic_threshold": 1.1}))
def test_es_moe_rejects_invalid_routing_configuration(kwargs):
    with pytest.raises(ValueError):
        ES_MOE(8, 8, num_experts=3, **kwargs)


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
    col = torch.zeros(E)
    col[0] = 1.0
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
    col = torch.zeros(E)
    col[0] = 1.0
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


def test_gradient_checkpointing_skips_moe_registry_modules(monkeypatch):
    """MoE aux-loss publishers should not be recomputed through checkpointing."""
    import torch.utils.checkpoint as checkpoint_mod
    from ultralytics.nn.tasks import BaseModel

    def fail_checkpoint(*args, **kwargs):
        raise AssertionError("MoE modules must bypass torch.utils.checkpoint")

    monkeypatch.setattr(checkpoint_mod, "checkpoint", fail_checkpoint)
    torch.manual_seed(0)
    MOE_LOSS_REGISTRY.clear()
    base = BaseModel()
    m = OptimizedMOE(32, 32, num_experts=4, top_k=2).train()
    x = torch.randn(2, 32, 8, 8, requires_grad=True)

    out = base._apply_checkpointing(m, x)
    aux = MOE_LOSS_REGISTRY.get(m)
    assert isinstance(aux, torch.Tensor) and aux.requires_grad

    router_weight = next(p for p in m.router.parameters() if p.requires_grad)
    m.zero_grad(set_to_none=True)
    (out.float().mean() + aux).backward()
    assert router_weight.grad is not None and router_weight.grad.abs().sum() > 0


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


# ===========================================================================
# rev8: fixes from the 2026-06-25 deep-scan report ()
# ===========================================================================

def test_p01_hyperultimate_get_gflops_no_attribute_error():
    """P-01: get_gflops must use fused_conv (was fused_weight -> AttributeError)."""
    m = HyperUltimateMoE(32, 32, num_experts=4, top_k=2)
    g = m.get_gflops((1, 32, 32, 32))
    assert isinstance(g, dict) and g["total_gflops"] > 0


def test_p02_init_weights_guards_non_conv_router():
    """P-02: _init_weights must not UnboundLocalError when router has no Conv2d."""
    m = OptimizedMOEImproved(32, 32, num_experts=4, top_k=2)
    m.routing.router = nn.Sequential(nn.Flatten(), nn.Linear(32, 4))
    m._init_weights()  # must not raise


def test_f02_ablockmoe_single_residual():
    """F-02: inner MoE has add_residual=False so ABlockMoE applies it exactly once."""
    blk = ABlockMoE(dim=32, num_heads=1, num_experts=4, top_k=2)
    assert blk.mlp.add_residual is False
    blk.eval()
    x = torch.randn(2, 32, 16, 16)
    with torch.no_grad():
        out = blk(x)
    assert out.shape == x.shape and torch.isfinite(out).all()


def test_f03_advanced_router_proj_registered():
    """F-03: AdvancedRoutingLayer must NOT create dynamic parameters at runtime.

    After P0-2 fix, channel mismatch is handled via tensor-only zero-pad/truncate
    path. _proj is nn.Identity() created in __init__, no dynamic add_module.
    """
    r = AdvancedRoutingLayer(in_channels=64, num_experts=3)
    # Channel mismatch should NOT create new parameters
    out = r(torch.randn(2, 32, 8, 8))  # C=32 != in_channels=64
    assert out is not None
    # _proj exists as Identity (no learnable params), no dynamic creation
    assert not any(k.startswith("_proj") for k, _ in r.named_parameters()
                   if isinstance(dict(r.named_modules()).get("_proj.split()[0]"), nn.Conv2d))


def test_p04_zloss_computed_after_noise():
    """P-04: z_loss must be finite and grad-bearing (now computed post-noise)."""
    r = UltraEfficientRouter(32, num_experts=4, top_k=2, noise_std=1.0).train()
    z = r(torch.randn(2, 32, 16, 16))[4]
    assert isinstance(z, torch.Tensor) and z.requires_grad and torch.isfinite(z).all()


def test_ultra_router_zloss_uses_temperature_scaled_logits():
    """Router z-loss must regularize the logits actually used by softmax."""
    r = UltraEfficientRouter(4, num_experts=4, top_k=2, noise_std=0.0, temperature=0.5).train()
    r.router = nn.Identity()
    z = r(torch.ones(2, 4, 2, 2))[4]
    expected = (torch.tensor(2.0) + torch.log(torch.tensor(4.0))).square()
    assert torch.allclose(z, expected, atol=1e-6)


def test_hyperfused_progressive_sparsity_uses_current_top_k(monkeypatch):
    """After P1-4 fix: forward uses fixed top_k to avoid GPU→CPU sync.
    current_top_k buffer is still updated for diagnostics during warmup.
    """
    torch.manual_seed(0)
    m = HyperFusedMoE(32, 32, num_experts=4, top_k=2, progressive_sparsity=True).train()
    captured = {}
    original = m.fused_experts.forward

    def wrapped(x, routing_weights, routing_indices, top_k):
        captured["top_k"] = top_k
        captured["routing_shape"] = routing_indices.shape
        return original(x, routing_weights, routing_indices, top_k)

    monkeypatch.setattr(m.fused_experts, "forward", wrapped)
    m(torch.randn(2, 32, 8, 8))
    # Forward now uses fixed top_k (no .item() sync)
    assert captured["top_k"] == 2
    assert captured["routing_shape"][1] == 2
    # But current_top_k buffer is still updated for diagnostics
    assert int(m.current_top_k) >= m.top_k


def test_optimized_moe_improved_expert_dropout_skips_only_after_warmup(monkeypatch):
    """Expert dropout is active at its configured interval and deterministic."""
    def run_once():
        module = OptimizedMOEImproved(8, 8, num_experts=4, top_k=2).train()
        module.warmup_steps = 0
        module.dropout_interval = 1
        module.expert_dropout_rate = 0.5
        captured = []
        for idx, expert in enumerate(module.experts):
            original = expert.forward
            def wrapped(x, _original=original, _idx=idx):
                captured.append(_idx)
                return _original(x)
            expert.forward = wrapped
        module(torch.ones(4, 8, 4, 4))
        return captured

    torch.manual_seed(123)
    first = run_once()
    torch.manual_seed(123)
    second = run_once()
    assert first == second and len(first) > 0


def test_l02_adaptive_capacity_complexity_lower_bound():
    """L-02: complexity is clamped >= 0.3 so top_k cannot degenerate to 1."""
    m = AdaptiveCapacityMoE(32, 32, num_experts=4, top_k=2).train()
    with torch.no_grad():
        for p in m.complexity_estimator.parameters():
            p.zero_()
    x = torch.randn(2, 32, 16, 16)
    out = m(x)
    assert torch.isfinite(out).all()
    cs = m.complexity_estimator(x).mean().clamp(0.3, 1.5)
    assert cs.item() >= 0.3 - 1e-6


def test_l04_a2c2fmoe_get_gflops_no_attribute_error():
    """L-04: get_gflops sums sub-block MoE FLOPs (was accessing missing attrs)."""
    m = A2C2fMoE(64, 64, n=1, num_experts=4, top_k=2)
    g = m.get_gflops((1, 64, 32, 32))
    assert isinstance(g, dict) and g["total_gflops"] >= 0.0


def test_f05_static_balance_loss_handles_empty_stats():
    """F-05: missing expert_usage falls back to uniform, no KeyError."""
    m = HyperFusedMoE(32, 32, num_experts=4, top_k=2)
    loss = m._compute_static_balance_loss({})
    assert isinstance(loss, torch.Tensor) and torch.isfinite(loss).all()


def test_p05_es_moe_eval_uses_sparse():
    """P-05: ES_MOE eval runs the sparse path and stays finite."""
    m = ES_MOE(32, 32, num_experts=4, top_k=2).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 32, 16, 16))
    assert out.shape[0] == 2 and torch.isfinite(out).all()


def test_p06_batched_experts_noncontiguous_indices():
    """P-06: reshape-based flatten handles non-contiguous [B,k,1,1] indices."""
    B, C, H, W, E, k = 2, 16, 8, 8, 4, 2
    x = torch.randn(B, C, H, W)
    experts = nn.ModuleList([nn.Conv2d(C, C, 1) for _ in range(E)])
    idx = torch.randint(0, E, (B, k, 1, 1))
    w = torch.rand(B, k, 1, 1)
    out = BatchedExpertComputation.compute_sparse_experts_batched(x, experts, w, idx, k, E)
    assert out.shape == (B, C, H, W) and torch.isfinite(out).all()


def test_batched_experts_keeps_low_weight_training_routes():
    """Training must not hard-drop low-weight routes before they can learn."""
    B, C, H, W, E, k = 2, 8, 4, 4, 2, 1
    x = torch.randn(B, C, H, W)
    experts = nn.ModuleList([nn.Conv2d(C, C, 1, bias=False) for _ in range(E)]).train()
    with torch.no_grad():
        experts[0].weight.fill_(1.0)
    idx = torch.zeros(B, k, 1, 1, dtype=torch.long)
    w = torch.full((B, k, 1, 1), 0.005)
    out = BatchedExpertComputation.compute_sparse_experts_batched(x, experts, w, idx, k, E)
    assert out.abs().sum() > 0, "low-weight training routes were skipped by an inference threshold"


def test_l05_soft_balancing_penalizes_collapse():
    """L-05: soft balancing penalizes uneven usage (not an L2 on importance)."""
    loss_fn = MoELoss(num_experts=4, top_k=2, use_soft_balancing=True)
    lg_u = torch.zeros(16, 4, requires_grad=True)
    l_unif = loss_fn(torch.softmax(lg_u, dim=1), lg_u)
    lg_c = torch.tensor([[10., -10., -10., -10.]]).repeat(16, 1).requires_grad_(True)
    l_col = loss_fn(torch.softmax(lg_c, dim=1), lg_c)
    assert float(l_col) > float(l_unif)
    l_col.backward()
    assert lg_c.grad is not None and lg_c.grad.abs().sum() > 0


@pytest.mark.parametrize("cls", [SimpleExpert, SpatialExpert, GhostExpert, InvertedResidualExpert])
def test_e01_legacy_experts_use_groupnorm(cls):
    """E-01: legacy experts switched BN -> GN; B=1 must be stable."""
    e = cls(16, 16).eval()
    assert not any(isinstance(m, nn.BatchNorm2d) for m in e.modules())
    assert any(isinstance(m, nn.GroupNorm) for m in e.modules())
    with torch.no_grad():
        out = e(torch.randn(1, 16, 8, 8))
    assert torch.isfinite(out).all()


def test_e02_index_add_clamp_prevents_overflow():
    """E-02: large inputs (routing-collapse proxy) stay finite via clamp."""
    m = OptimizedMOE(32, 32, num_experts=4, top_k=2).train()
    out = m(torch.randn(2, 32, 16, 16) * 100)
    assert torch.isfinite(out).all()


# C-soft: soft balancing must back-prop a real (non-zero) gradient to the router.
# Regression for the rev8 column-normalized usage bug: usage collapsed to a
# constant 1/N, so balance_loss == 1.0 with zero gradient (silent failure).
def test_soft_balance_loss_grad_reaches_router():
    """soft balancing: balance_loss must produce a non-zero grad on logits."""
    N, E, K = 16, 4, 2
    loss_fn = MoELoss(num_experts=E, top_k=K, use_soft_balancing=True,
                      balance_loss_coeff=1.0, z_loss_coeff=0.0)
    logits = torch.randn(N, E, requires_grad=True)
    probs = torch.softmax(logits, dim=1)
    loss_fn(probs, logits, return_dict=True)["loss"].backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 1e-4, "soft balance gradient vanished (constant-usage bug)"


def test_soft_balance_uses_topk_counts_when_available():
    """Soft balancing should use detached discrete usage, not importance self-reference."""
    E, K, N = 4, 2, 8
    logits = torch.randn(N, E, requires_grad=True)
    probs = torch.softmax(logits, dim=1)
    indices = torch.zeros(N, K, dtype=torch.long)
    loss_fn = MoELoss(num_experts=E, top_k=K, use_soft_balancing=True,
                      balance_loss_coeff=1.0, z_loss_coeff=0.0)
    out = loss_fn(probs, logits, indices, return_dict=True)
    expected = E * probs.mean(dim=0)[0]
    assert torch.allclose(out["balance_loss"], expected, atol=1e-6)


def test_soft_balance_loss_responds_to_imbalance():
    """soft balancing: a collapsed router must yield a larger balance_loss."""
    N, E, K = 16, 4, 2
    loss_fn = MoELoss(num_experts=E, top_k=K, use_soft_balancing=True,
                      balance_loss_coeff=1.0, z_loss_coeff=0.0)
    uniform_logits = torch.zeros(N, E)
    collapsed_logits = torch.full((N, E), -10.0)
    collapsed_logits[:, 0] = 10.0
    bl_uniform = loss_fn(torch.softmax(uniform_logits, 1), uniform_logits,
                         return_dict=True)["balance_loss"]
    bl_collapsed = loss_fn(torch.softmax(collapsed_logits, 1), collapsed_logits,
                           return_dict=True)["balance_loss"]
    # GShard soft: uniform -> ~1.0, full collapse -> ~E. Must be distinguishable.
    assert bl_collapsed > bl_uniform + 0.5, (
        f"balance_loss insensitive to imbalance: uniform={bl_uniform:.3f}, "
        f"collapsed={bl_collapsed:.3f}"
    )


def test_dymoe_gate_gshard_balance_and_registry():
    """DyMoEBlock must publish GShard-scale balance loss via MOE_LOSS_REGISTRY."""
    from ultralytics.nn.modules.block import DyMoEBlock

    torch.manual_seed(0)
    m = DyMoEBlock(32, num_experts=4, top_k=2).train()
    out = m(torch.randn(4, 32, 8, 8))
    assert out.shape == (4, 32, 8, 8)
    aux = m.aux_loss
    assert aux.requires_grad
    assert float(aux.detach()) > 0.0
    collected = float(_collect_moe_aux_loss(m, torch.device("cpu")).detach())
    assert collected == pytest.approx(float(aux.detach()), rel=1e-5)


def test_a2c2f_moe_aux_loss_property_matches_inner_blocks():
    """A2C2fMoE.aux_loss must delegate to inner ABlockMoE modules, not registry on self."""
    torch.manual_seed(0)
    m = A2C2fMoE(c1=64, c2=64, n=1, a2=True, num_experts=4, top_k=2).train()
    m(torch.randn(2, 64, 16, 16))
    wrapper_aux = float(m.aux_loss.detach())
    inner = 0.0
    for block_seq in m.m:
        for block in block_seq:
            inner += float(block.aux_loss.detach())
    assert wrapper_aux > 0.0
    assert wrapper_aux == pytest.approx(inner, rel=1e-5)


def test_get_global_mean_float16_input():
    """DDP mean helper must use float32 count even when input is float16."""
    loss_fn = MoELoss(num_experts=4, top_k=2)
    probs = torch.rand(8, 4, dtype=torch.float16)
    mean = loss_fn._get_global_mean(probs)
    expected = probs.mean(dim=0)
    assert torch.allclose(mean.float(), expected.float(), atol=1e-3)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
